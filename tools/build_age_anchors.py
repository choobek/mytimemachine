import os
import sys
import re
import json
from glob import glob
from datetime import datetime

import torch
from torch import nn
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T


def parse_age_from_name(name, rx):
    m = re.search(rx, os.path.basename(name))
    return int(m.group(1)) if m else None


def bin_mid(age, bin_size):
    b = (age // bin_size) * bin_size
    return int(b + bin_size // 2)


def robust_mean(stack: torch.Tensor, iters: int = 2, z: float = 2.5) -> torch.Tensor:
    x = stack
    for _ in range(int(iters)):
        mu = x.mean(0, keepdim=True)
        sd = x.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
        zmask = ((x - mu).abs() / sd) < z
        x = x[zmask.all(dim=1)]
        if x.shape[0] < 3:
            break
    return x.mean(0)


class _EncoderWrapper(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, img, return_wplus=True):
        latents = self.enc(img)
        # latents: [B, L, 512]
        return latents


def load_psp(encoder_ckpt_path: str, output_size: int = 1024, input_nc: int = 3):
    """
    Load only the pSp encoder to produce W+ codes. Avoids StyleGAN decode.
    """
    from argparse import Namespace
    # Ensure repo root is on sys.path so we can import models.*
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if root not in sys.path:
        sys.path.insert(0, root)
    from models.encoders.psp_encoders import GradualStyleEncoder
    n_styles = int(torch.log2(torch.tensor(float(output_size))).item()) * 2 - 2
    # Minimal opts with required fields for encoder construction
    opts = Namespace(input_nc=int(input_nc))
    encoder = GradualStyleEncoder(50, 'ir_se', n_styles, opts)
    ckpt = torch.load(encoder_ckpt_path, map_location='cpu')
    state = ckpt['encoder'] if 'encoder' in ckpt else ckpt.get('state_dict', {})
    if 'state_dict' in ckpt and 'encoder' not in ckpt:
        # filter keys that start with 'encoder.'
        state = {k[len('encoder.'):] : v for k, v in ckpt['state_dict'].items() if k.startswith('encoder.')}
    encoder.load_state_dict(state, strict=False)
    return _EncoderWrapper(encoder)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = load_psp(args.encoder_ckpt).to(device).eval()
    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    bins = {}
    files = sorted(glob(os.path.join(args.dataset_dir, '*')))
    for fp in tqdm(files):
        age = parse_age_from_name(fp, args.age_regex)
        if age is None:
            continue
        try:
            img = Image.open(fp).convert('RGB')
        except Exception:
            continue
        img_t = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            wplus = encoder(img_t, return_wplus=True)  # [1, L, 512]
        w = wplus.mean(dim=1).squeeze(0).float().cpu()  # [512]
        mid = bin_mid(int(age), int(args.bin_size))
        bins.setdefault(mid, []).append(w)

    mids = sorted(bins.keys())
    anchors, counts = [], []
    for m in mids:
        stack = torch.stack(bins[m], dim=0)
        if args.robust_mean and stack.shape[0] >= 6:
            a = robust_mean(stack)
        else:
            a = stack.mean(0)
        anchors.append(a)
        counts.append(int(stack.shape[0]))

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    out = {
        'space': 'w',
        'bin_size': int(args.bin_size),
        'anchors': torch.stack(anchors, 0) if len(anchors) > 0 else torch.zeros(0, 512),
        'bin_mids': mids,
        'counts': counts,
        'config': {
            'encoder_ckpt': args.encoder_ckpt,
            'dataset_dir': args.dataset_dir,
            'regex': args.age_regex,
            'date': datetime.now().isoformat(timespec='seconds'),
            'robust': bool(args.robust_mean),
        }
    }
    torch.save(out, args.out_path)
    print(f"Saved {len(mids)} bins to {args.out_path}. Counts: {counts}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--encoder_ckpt', required=True)
    ap.add_argument('--age_regex', default=r'^(\d+)_')
    ap.add_argument('--bin_size', type=int, default=5)
    ap.add_argument('--robust_mean', action='store_true')
    ap.add_argument('--out_path', default='anchors/actor_w_age5.pt')
    main(ap.parse_args())


