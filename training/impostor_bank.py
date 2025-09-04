from __future__ import annotations

import torch
from typing import Dict, List, Tuple


def _parse_bin_label(lbl: str) -> Tuple[int, int]:
	"""Parse labels like '33-37' -> (33, 37)."""
	a, b = lbl.split("-")
	return int(a), int(b)


class AgeMatchedImpostorBank:
	"""Lightweight age-matched impostor bank for negative sampling.

	Expects a .pt file with structure:
	  bank['bins'][label] -> Tensor [N, 512], L2-normalized
	  bank['ages'][label] -> Tensor [N]

	The bank is kept on CPU; sampled negatives are moved by the caller.
	"""

	def __init__(self, path: str, device: torch.device | str = "cpu"):
		self.device = torch.device(device)
		self.bank = torch.load(path, map_location="cpu")
		self.bins: Dict[str, torch.Tensor] = self.bank["bins"]
		self.ages: Dict[str, torch.Tensor] = self.bank["ages"]
		self._bin_ranges: Dict[str, Tuple[int, int]] = {k: _parse_bin_label(k) for k in self.bins.keys()}
		self._bin_centers: Dict[str, float] = {k: (r[0] + r[1]) / 2.0 for k, r in self._bin_ranges.items()}
		self._labels: List[str] = list(self.bins.keys())

		for k, X in self.bins.items():
			assert X.dim() == 2 and X.size(1) == 512, f"Bin {k} has invalid shape {tuple(X.shape)}"

	def _labels_covering_age(self, age: int) -> List[str]:
		return [k for k, (lo, hi) in self._bin_ranges.items() if lo <= age <= hi]

	def _nearest_bin(self, age: int) -> str:
		return min(self._labels, key=lambda k: abs(self._bin_centers[k] - age))

	def _neighbor_bins(self, label: str, radius: int) -> List[str]:
		if radius <= 0:
			return [label]
		sorted_lbls = sorted(self._labels, key=lambda k: self._bin_centers[k])
		idx = sorted_lbls.index(label)
		lo = max(0, idx - radius)
		hi = min(len(sorted_lbls), idx + radius + 1)
		return sorted_lbls[lo:hi]

	def sample(self, target_ages: torch.Tensor, k: int, radius: int) -> torch.Tensor:
		"""Sample negatives for each target age.

		Args:
		  target_ages: Int/Long tensor [B]
		  k: number of negatives per sample
		  radius: include this many neighboring 5y bins on each side

		Returns:
		  Tensor [B, k, 512] on CPU
		"""
		B = int(target_ages.numel())
		out = []
		for i in range(B):
			age = int(target_ages[i].item())
			labels = self._labels_covering_age(age)
			if not labels:
				labels = [self._nearest_bin(age)]
			if radius > 0:
				exp: List[str] = []
				for lbl in labels:
					exp.extend(self._neighbor_bins(lbl, radius))
				labels = list(dict.fromkeys(exp))

			pools: List[torch.Tensor] = [self.bins[lbl] for lbl in labels if lbl in self.bins]
			if not pools:
				pools = [self.bins[lbl] for lbl in self._labels]
			pool = torch.cat(pools, dim=0)

			if pool.size(0) >= k:
				idx = torch.randint(low=0, high=pool.size(0), size=(k,))
				sel = pool.index_select(0, idx)
			else:
				idx = torch.randint(low=0, high=pool.size(0), size=(k,))
				sel = pool.index_select(0, idx)

			out.append(sel.unsqueeze(0))

		return torch.cat(out, dim=0)


