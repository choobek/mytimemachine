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


def _age_from_name(p: Path) -> int:
    name = p.name
    try:
        return int(name.split('_', 1)[0])
    except Exception:
        return -1


class RefImageDataset(Dataset):
    def __init__(self, root: Path, transform: T.Compose):
        self.files: List[Path] = sorted([p for p in Path(root).rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        p = self.files[idx]
        age = _age_from_name(p)
        img = Image.open(p).convert('RGB')
        x = self.transform(img)
        return x, age, str(p)


class _EncoderWrapper(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, img, return_wplus: bool = True):
        latents = self.enc(img)
        return latents  # [B, L, 512]


def load_psp(encoder_ckpt_path: str, output_size: int = 1024, input_nc: int = 3) -> _EncoderWrapper:
    from argparse import Namespace
    # lazy import of encoder class
    from models.encoders.psp_encoders import GradualStyleEncoder
    n_styles = int(torch.log2(torch.tensor(float(output_size))).item()) * 2 - 2
    opts = Namespace(input_nc=int(input_nc))
    encoder = GradualStyleEncoder(50, 'ir_se', n_styles, opts)
    ckpt = torch.load(encoder_ckpt_path, map_location='cpu')
    state = ckpt['encoder'] if 'encoder' in ckpt else ckpt.get('state_dict', {})
    if 'state_dict' in ckpt and 'encoder' not in ckpt:
        state = {k[len('encoder.'):]: v for k, v in ckpt['state_dict'].items() if k.startswith('encoder.')}
    encoder.load_state_dict(state, strict=False)
    return _EncoderWrapper(encoder)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Build 1-year actor anchors (W or W+) using pSp encoder')
    ap.add_argument('--refs_root', type=Path, required=True)
    ap.add_argument('--encoder_checkpoint', type=Path, required=True)
    ap.add_argument('--space', type=str, choices=['w', 'wplus'], default='w')
    ap.add_argument('--bin_from', type=int, default=35)
    ap.add_argument('--bin_to', type=int, default=45)
    ap.add_argument('--bin_size', type=int, default=1)
    ap.add_argument('--out', type=Path, default=Path('anchors/actor_w_age1.pt'))
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--output_size', type=int, default=1024, help='StyleGAN output size to set number of styles')
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    # transforms consistent with training/inference
    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = RefImageDataset(a.refs_root, tfm)
    if len(dataset) == 0:
        raise RuntimeError(f'No images found under {a.refs_root}')

    device = torch.device(a.device if a.device != 'cuda' or torch.cuda.is_available() else 'cpu')
    encoder = load_psp(str(a.encoder_checkpoint), output_size=int(a.output_size)).to(device).eval()

    loader = DataLoader(
        dataset,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        pin_memory=(device.type == 'cuda'),
        shuffle=False,
        drop_last=False,
    )

    # Accumulate per-age latents
    latents_per_age: Dict[int, List[torch.Tensor]] = {}

    with torch.no_grad():
        for imgs, ages, paths in loader:
            # select ages within bins
            ages = [int(x) for x in ages]
            keep_mask = [((ag >= a.bin_from) and (ag <= a.bin_to) and ((ag - a.bin_from) % a.bin_size == 0)) for ag in ages]
            if not any(keep_mask):
                continue
            idxs = [i for i, k in enumerate(keep_mask) if k]
            imgs = imgs[idxs].to(device).float()
            ages_kept = [ages[i] for i in idxs]

            wplus = encoder(imgs, return_wplus=True)  # [B, L, 512]
            if a.space == 'w':
                lat_batch = wplus.mean(dim=1)  # [B, 512]
            else:
                lat_batch = wplus  # [B, L, 512]

            lat_batch = lat_batch.detach().cpu().float()
            for lat, age in zip(lat_batch, ages_kept):
                latents_per_age.setdefault(int(age), []).append(lat)

    # compute mean per age
    anchors: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    for age in sorted(latents_per_age.keys()):
        X = latents_per_age[age]
        if len(X) == 0:
            continue
        if a.space == 'w':
            stack = torch.stack(X, dim=0)  # [N, 512]
            mean_lat = stack.mean(dim=0)
        else:
            stack = torch.stack(X, dim=0)  # [N, L, 512]
            mean_lat = stack.mean(dim=0)  # [L, 512]
        anchors[int(age)] = mean_lat
        counts[int(age)] = int(stack.shape[0])

    if len(anchors) == 0:
        raise RuntimeError('No anchors computed. Check filename ages and bin range.')

    payload = {
        'space': str(a.space),
        'anchors': anchors,  # dict {age:int -> tensor}
        'range': {'from': int(a.bin_from), 'to': int(a.bin_to), 'bin_size': int(a.bin_size)},
        'config': {
            'encoder_ckpt': str(a.encoder_checkpoint),
            'output_size': int(a.output_size),
            'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'refs_root': str(a.refs_root),
            'counts': counts,
        }
    }

    tmp = a.out.with_suffix('.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, a.out)
    total = sum(counts.values())
    print(f"Wrote {a.out} with {len(anchors)} ages and {total} total samples.")


if __name__ == '__main__':
    main()


