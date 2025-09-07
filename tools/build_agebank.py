import argparse
import time
import os
from pathlib import Path
from typing import List, Tuple
import sys

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Ensure repository root is on sys.path when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.transforms_config import AgingTransforms
from criteria.id_loss import IDLoss
from criteria.aging_loss import AgingLoss
from training.id_backbone import IDBackbone


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--images-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--min-age', type=int, default=18)
    p.add_argument('--max-age', type=int, default=80)
    p.add_argument('--bin-size', type=int, default=5)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--quality-filter', type=str, default='')  # e.g. 'laplacian:100.0'
    p.add_argument('--actor-center', type=Path, default=None)
    p.add_argument('--actor-max-cos', type=float, default=0.65)
    # Identity backbone selection to match training backbone
    p.add_argument('--id_backbone', type=str, default='ir50', choices=['ir50', 'ir100', 'adaface'])
    p.add_argument('--id_backbone_path', type=str, default='')
    return p.parse_args()


def laplacian_var(img_rgb_np: np.ndarray) -> float:
    import cv2
    g = cv2.cvtColor(img_rgb_np, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def age_to_bin(age: float, mn: int, mx: int, sz: int) -> str:
    if age < mn or age > mx:
        return ''
    lo = int((age - mn) // sz) * sz + mn
    hi = lo + sz - 1
    return f'{lo}-{hi}'


class FFHQAlignedDataset(Dataset):
    def __init__(self, root: Path, transform, compute_lap: bool = False):
        self.files: List[Path] = sorted([p for p in root.rglob('*.png')])
        self.transform = transform
        self.compute_lap = compute_lap

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float, str]:
        p = self.files[idx]
        try:
            im = Image.open(p)
            im.load()
            im = im.convert('RGB')
        except Exception:
            # Return a dummy sample that will be filtered by quality or skipped upstream
            # Create a tiny valid image tensor to avoid crashing the DataLoader
            dummy = Image.new('RGB', (256, 256), color=(0, 0, 0))
            im = dummy
        x = self.transform(im)
        if self.compute_lap:
            lv = laplacian_var(np.array(im))
        else:
            lv = float('inf')
        return x, lv, str(p)


def main():
    a = parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    # Transforms identical to training ([-1,1] range, 256 resize)
    transforms_dict = AgingTransforms(opts=None).get_transforms()
    transform_img = transforms_dict['transform_test']

    # Models identical to training
    device = torch.device(a.device if a.device != 'cuda' or torch.cuda.is_available() else 'cpu')
    # Use requested backbone to ensure the bank matches the training identity space
    if a.id_backbone and a.id_backbone.lower() != 'ir50':
        id_extractor = IDBackbone(name=a.id_backbone.lower().strip(),
                                  weights_path=a.id_backbone_path if len(a.id_backbone_path) > 0 else None,
                                  embed_dim=512, normalize='l2', input_size=56).to(device).eval()
        # Wrap with a simple shim to reuse downstream extract_feats calls
        class _Wrap:
            def __init__(self, enc):
                self.enc = enc
                self._pool = torch.nn.AdaptiveAvgPool2d((256, 256))
                self._crop = True
            @torch.no_grad()
            def extract_feats(self, x: torch.Tensor) -> torch.Tensor:
                if x.shape[2] != 256 or x.shape[3] != 256:
                    x = self._pool(x)
                if self._crop:
                    x = x[:, :, 35:223, 32:220]
                feats = self.enc(x)
                return feats
        id_extractor = _Wrap(id_extractor)
    else:
        # Default legacy IR-SE50 extractor
        id_extractor = IDLoss().to(device).eval()
    age_estimator = AgingLoss(opts=None).to(device).eval()

    # Optional paranoia filter
    actor_mu = None
    if a.actor_center and a.actor_center.exists():
        actor_mu = torch.load(a.actor_center, map_location='cpu').float()
        actor_mu = torch.nn.functional.normalize(actor_mu, dim=0)

    # Dataset and loader
    qf = None
    if a.quality_filter.startswith('laplacian:'):
        try:
            qf = float(a.quality_filter.split(':', 1)[1])
        except Exception:
            qf = None

    dataset = FFHQAlignedDataset(a.images_root, transform_img, compute_lap=(qf is not None))
    print(f'Found {len(dataset)} images.')
    loader = DataLoader(
        dataset,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        pin_memory=(device.type == 'cuda'),
        shuffle=False,
        drop_last=False,
    )

    bins, ages = {}, {}

    with torch.no_grad():
        for batch in tqdm(loader, ncols=100):
            x, lapvars, paths = batch

            # Quality filter mask
            if qf is not None:
                mask = (lapvars >= qf)
                if mask.ndim > 0:
                    # Ensure boolean tensor
                    mask = mask.to(torch.bool)
                else:
                    mask = torch.tensor([bool(mask)], dtype=torch.bool)
                if not mask.any():
                    continue
                x = x[mask]
                paths = [p for p, m in zip(paths, mask.tolist()) if m]

            if x.numel() == 0:
                continue

            x = x.to(device).float()

            # Batched forward through training models
            f_batch = id_extractor.extract_feats(x)  # [B,512], already L2-normalized
            age_batch = age_estimator.extract_ages(x)  # [B], float ages

            f_batch = f_batch.detach().cpu().float()
            age_batch = age_batch.detach().cpu().float()

            for f, age in zip(f_batch, age_batch):
                age_val = float(age)
                bkey = age_to_bin(age_val, a.min_age, a.max_age, a.bin_size)
                if not bkey:
                    continue
                if actor_mu is not None:
                    cos = float(torch.dot(f, actor_mu))
                    if cos >= a.actor_max_cos:
                        continue
                bins.setdefault(bkey, []).append(f)
                ages.setdefault(bkey, []).append(age_val)

    # Stack and save
    out_bins, out_ages = {}, {}
    for k in sorted(bins.keys(), key=lambda s: int(s.split('-')[0])):
        X = torch.stack(bins[k]) if len(bins[k]) else torch.empty(0, 512)
        Y = torch.tensor(ages[k], dtype=torch.float32) if len(ages[k]) else torch.empty(0)
        out_bins[k] = X
        out_ages[k] = Y
        print(f'Bin {k}: {X.shape[0]} samples')

    payload = {
        'bins': out_bins,
        'ages': out_ages,
        'meta': {
            'norm': 'l2',
            'embed': (a.id_backbone.lower().strip() if a.id_backbone else 'ir_se50'),
            'age_model': 'dex_vgg',
            'bin_size': a.bin_size,
            'min_age': a.min_age,
            'max_age': a.max_age,
            'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'source': 'FFHQ-1024',
        },
    }
    tmp = a.out.with_suffix('.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, a.out)
    total = sum(v.size(0) for v in out_bins.values())
    print(f'Wrote {a.out} with {total} embeddings.')


if __name__ == '__main__':
    main()


