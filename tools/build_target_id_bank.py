import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


# Ensure repository root is on sys.path when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from criteria.id_loss import IDLoss  # IR-SE50 backbone wrapper used across training
from configs.transforms_config import AgingTransforms


class RefImageDataset(Dataset):
    """
    Simple image dataset for recursively loading actor reference images and parsing age from filename.

    Expects filenames like "40_XXXX.jpg" where the age is the integer prefix before the first underscore.
    """

    def __init__(self, root: Path, transform: T.Compose):
        self.files: List[Path] = sorted([p for p in Path(root).rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _parse_age(p: Path) -> int:
        name = p.name
        # Expect pattern: "<age>_*.ext"; fallback to -1 if parse fails
        try:
            age_str = name.split('_', 1)[0]
            return int(age_str)
        except Exception:
            return -1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        p = self.files[idx]
        age = self._parse_age(p)
        img = Image.open(p).convert('RGB')
        x = self.transform(img)
        return x, age, str(p)


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build target-ID bank (IR-SE50) from actor references')
    p.add_argument('--refs_root', type=Path, required=True, help='Root directory with reference images (recursively scanned)')
    p.add_argument('--id_backbone', type=str, default='ir50', choices=['ir50'], help='Only ir50 is supported (IR-SE50)')
    p.add_argument('--out', type=Path, default=Path('banks/actor40_ir.pt'), help='Output .pt path')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def main() -> None:
    a = parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    # Transforms consistent with training ID usage ([-1,1], 256 resize)
    transforms_dict = AgingTransforms(opts=None).get_transforms()
    transform_img = transforms_dict['transform_test']

    dataset = RefImageDataset(a.refs_root, transform_img)
    if len(dataset) == 0:
        raise RuntimeError(f'No images found under {a.refs_root}')

    device = torch.device(a.device if a.device != 'cuda' or torch.cuda.is_available() else 'cpu')
    id_extractor = IDLoss().to(device).eval()  # Uses IR-SE50 under the hood; outputs L2-normalized 512-D

    loader = DataLoader(
        dataset,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        pin_memory=(device.type == 'cuda'),
        shuffle=False,
        drop_last=False,
    )

    # Aggregate per-age global embeddings
    age_to_embs: Dict[int, List[torch.Tensor]] = {}

    with torch.no_grad():
        for batch in loader:
            imgs, ages, paths = batch
            # Filter invalid ages
            mask = torch.tensor([int(age) >= 0 for age in ages], dtype=torch.bool)
            if not mask.any():
                continue
            imgs = imgs[mask].to(device).float()
            ages = [int(ages[i]) for i, m in enumerate(mask.tolist()) if m]

            feats = id_extractor.extract_feats(imgs)  # [B,512], already L2-normalized
            feats = feats.detach().cpu().float()

            for f, age in zip(feats, ages):
                age_to_embs.setdefault(age, []).append(f)

    # Compute per-age prototypes: mean then L2 normalize
    global_protos: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    for age in sorted(age_to_embs.keys()):
        X = torch.stack(age_to_embs[age], dim=0) if len(age_to_embs[age]) else torch.empty(0, 512)
        if X.numel() == 0:
            continue
        mu = X.mean(dim=0)
        proto = l2_normalize(mu, dim=0)
        global_protos[int(age)] = proto
        counts[int(age)] = int(X.shape[0])

    if len(global_protos) == 0:
        raise RuntimeError('No valid age prototypes computed. Check filename patterns and input data.')

    payload = {
        'global_protos': global_protos,
        'meta': {
            'embed': 'ir_se50',
            'norm': 'l2',
            'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'refs_root': str(a.refs_root),
            'counts': counts,
        },
    }

    tmp = a.out.with_suffix('.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, a.out)
    total = sum(counts.values())
    print(f"Wrote {a.out} with {len(global_protos)} ages and {total} total embeddings.")


if __name__ == '__main__':
    main()


