from __future__ import annotations

import math
from typing import Tuple, Optional, Dict

import torch

try:
    import faiss  # optional
except Exception:
    faiss = None

from .impostor_bank import _parse_bin_label  # reuse helpers


class AgeMatchedImpostorMiner:
    """FAISS-based semi-hard negative miner with age-aware filtering.

    Expects a .pt file with structure compatible with `AgeMatchedImpostorBank`:
      bank['bins'][label] -> Tensor [N, 512], L2-normalized
      bank['ages'][label] -> Tensor [N]
    """

    def __init__(self, path: str, use_faiss: bool = True):
        # Load bins/ages (CPU)
        self.bank = torch.load(path, map_location="cpu")
        bins: Dict[str, torch.Tensor] = self.bank["bins"]   # {label: [Ni,512]}
        ages: Dict[str, torch.Tensor] = self.bank["ages"]   # {label: [Ni]}

        self.labels = list(bins.keys())
        self.bin_ranges = {k: _parse_bin_label(k) for k in self.labels}
        # Concat
        X_list, A_list, L_list = [], [], []
        for lbl in self.labels:
            X = bins[lbl].float()              # [Ni,512], L2-normalized
            a = ages[lbl].float()              # [Ni]
            X_list.append(X)
            A_list.append(a)
            L_list.append(torch.full((X.size(0),), self.labels.index(lbl), dtype=torch.int64))
        self.X_all = torch.cat(X_list, 0)      # [N,512]
        self.age_all = torch.cat(A_list, 0)    # [N]
        self.bin_id_all = torch.cat(L_list, 0) # [N]

        # Global FAISS index (IP = cosine when normalized)
        self.faiss_ok = bool(use_faiss and (faiss is not None))
        if self.faiss_ok:
            index = faiss.IndexFlatIP(self.X_all.size(1))
            index.add(self.X_all.numpy().astype("float32"))
            self.index = index
        else:
            self.index = None  # fallback path uses torch.mm

        # Cache for bin centers & quick routing
        self.bin_centers = {k: (r[0] + r[1]) / 2.0 for k, r in self.bin_ranges.items()}

    def _age_mask_for(self, target_age: int, radius: int) -> torch.Tensor:
        # bins covering same age ± neighbor radius by ordered centers
        # 1) find covering labels (or nearest)
        covers = [k for k, (lo, hi) in self.bin_ranges.items() if lo <= target_age <= hi]
        if not covers:
            # nearest by center
            nearest = min(self.labels, key=lambda k: abs(self.bin_centers[k] - target_age))
            covers = [nearest]
        # 2) expand by radius using bin center order
        if radius > 0:
            ordered = sorted(self.labels, key=lambda k: self.bin_centers[k])
            idxs = sorted([ordered.index(c) for c in covers])
            lo = max(0, min(idxs) - radius)
            hi = min(len(ordered), max(idxs) + radius + 1)
            covers = ordered[lo:hi]
        # mask: keep samples whose bin_id is in selected bins
        cover_ids = torch.tensor([self.labels.index(k) for k in covers], dtype=torch.int64)
        return torch.isin(self.bin_id_all, cover_ids)

    @torch.no_grad()
    def query(
        self,
        q: torch.Tensor,                # [B,512], L2-normalized
        target_ages: torch.Tensor,      # [B], ints
        k: int,
        min_sim: float,
        max_sim: float,
        top_m: int,
        radius: int,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = q.size(0)
        device = device or q.device
        sel_vecs = torch.empty((B, k, q.size(1)), dtype=torch.float32)
        sel_sims = torch.empty((B, k), dtype=torch.float32)

        # Global search
        if self.faiss_ok:
            D, I = self.index.search(q.detach().cpu().numpy().astype("float32"), top_m)  # D: [B,top_m]
            sims = torch.from_numpy(D)  # cosine (since normalized)
            idxs = torch.from_numpy(I).long()  # [B,top_m]
        else:
            # fallback: torch.mm against full bank
            sims_full = q.detach().cpu() @ self.X_all.t()  # [B,N]
            vals, inds = torch.topk(sims_full, k=int(top_m), dim=1)
            sims, idxs = vals, inds

        for i in range(B):
            age = int(target_ages[i].item())
            mask_age = self._age_mask_for(age, radius)      # [N]
            # candidates for this query
            cand_idx = idxs[i]                               # [top_m]
            cand_sim = sims[i]                               # [top_m]
            # age filter
            cand_mask = mask_age[cand_idx]
            cand_idx = cand_idx[cand_mask]
            cand_sim = cand_sim[cand_mask]
            # similarity band
            sim_mask = (cand_sim >= float(min_sim)) & (cand_sim <= float(max_sim))
            cand_idx = cand_idx[sim_mask]
            cand_sim = cand_sim[sim_mask]

            # relax if needed
            r = int(radius)
            ms = float(min_sim)
            while cand_idx.numel() < k:
                if r < 3:
                    r += 1
                    mask_age = self._age_mask_for(age, r)
                elif ms > 0.05:
                    ms = max(0.0, ms - 0.05)
                else:
                    # final fallback: random from age bins (no sim band)
                    mask = self._age_mask_for(age, radius)
                    pool = torch.nonzero(mask, as_tuple=False).squeeze(1)
                    if pool.numel() == 0:
                        pool = torch.arange(self.X_all.size(0))
                    extra = pool[torch.randint(0, pool.numel(), (k - cand_idx.numel(),))]
                    extra_s = torch.zeros_like(extra, dtype=torch.float32)
                    cand_idx = torch.cat([cand_idx, extra], 0)
                    cand_sim = torch.cat([cand_sim, extra_s], 0)
                    break
                # recompute filtered set with expanded radius / relaxed min_sim
                # (re-filter on original top_m list)
                cand_idx = idxs[i]
                cand_sim = sims[i]
                cand_mask = mask_age[cand_idx]
                cand_idx = cand_idx[cand_mask]
                cand_sim = cand_sim[cand_mask]
                sim_mask = (cand_sim >= ms) & (cand_sim <= float(max_sim))
                cand_idx = cand_idx[sim_mask]
                cand_sim = cand_sim[sim_mask]

            # sample K without replacement if possible
            if cand_idx.numel() >= k:
                perm = torch.randperm(cand_idx.numel())[:k]
                take = cand_idx[perm]
                take_sim = cand_sim[perm]
            else:
                rep = k - cand_idx.numel()
                if cand_idx.numel() > 0:
                    pad = cand_idx[torch.randint(0, cand_idx.numel(), (rep,))]
                    pad_sim = cand_sim[torch.randint(0, cand_sim.numel(), (rep,))]
                else:
                    pad = torch.randint(0, self.X_all.size(0), (rep,))
                    pad_sim = torch.zeros((rep,), dtype=torch.float32)
                take = torch.cat([cand_idx, pad], 0)[:k]
                take_sim = torch.cat([cand_sim, pad_sim], 0)[:k]

            sel_vecs[i] = self.X_all.index_select(0, take).float()
            sel_sims[i] = take_sim.float()

        return sel_vecs.to(device), sel_sims.to(device)


