#!/usr/bin/env python3
# scripts/prepro_gfpgan.py
import os, sys, glob, argparse, traceback
import cv2
import torch
from gfpgan import GFPGANer

def parse_args():
    p = argparse.ArgumentParser(description="Batch enhance faces with GFPGAN")
    p.add_argument("-i", "--input",  default="dataset_prep/040_topaz", help="Input folder with PNGs")
    p.add_argument("-o", "--output", default="dataset_prep/050_enhanced", help="Output folder")
    p.add_argument("-m", "--model",  default="experiments/pretrained_models/GFPGANv1.4.pth",
                   help="Path to GFPGAN model .pth (e.g., GFPGANv1.3/1.4)")
    p.add_argument("-v", "--version", default="1.4", choices=["1.2","1.3","1.4"], help="GFPGAN model version")
    p.add_argument("-s", "--upscale", type=int, default=1, help="Upscale factor (1 = keep size)")
    p.add_argument("--bg", choices=["none","realesrgan-x4plus"], default="none",
                   help="Background upsampler (needs RealESRGAN weights if used)")
    p.add_argument("--only-center-face", action="store_true", help="Restore only the most centered face")
    return p.parse_args()

def get_bg_upsampler(name, device):
    if name == "none":
        return None
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except Exception:
        print("! Real-ESRGAN not available; continuing without background upsampling.")
        return None
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    weights = "experiments/pretrained_models/RealESRGAN_x4plus.pth"
    if not os.path.isfile(weights):
        print(f"! Missing background upsampler weights: {weights}. Continuing without it.")
        return None
    return RealESRGANer(scale=4, model_path=weights, model=model, tile=0, tile_pad=10,
                        pre_pad=0, half=(device != "cpu"))

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isdir(args.input):
        print(f"! Input folder not found: {args.input}")
        sys.exit(1)
    os.makedirs(args.output, exist_ok=True)
    if not os.path.isfile(args.model):
        print(f"! Model file not found: {args.model}")
        print("  Download the GFPGAN model (e.g., GFPGANv1.4.pth) to that path.")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(args.input, "*.png"))) + \
            sorted(glob.glob(os.path.join(args.input, "*.PNG")))
    if not files:
        print(f"! No PNGs found in {args.input}")
        sys.exit(1)

    arch = "clean"
    channel_multiplier = 2
    bg_upsampler = get_bg_upsampler(args.bg, device)

    print(f"GFPGAN init… model={args.model} version={args.version} upscale={args.upscale} device={device}")
    restorer = GFPGANer(model_path=args.model, upscale=args.upscale,
                        arch=arch, channel_multiplier=channel_multiplier,
                        bg_upsampler=bg_upsampler, device=device)
    print(f"Found {len(files)} files. Writing to: {args.output}")

    failed = 0
    for idx, fp in enumerate(files, 1):
        fn = os.path.basename(fp)
        try:
            img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError("cv2.imread returned None")
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            with torch.no_grad():
                _, _, restored = restorer.enhance(
                    img, has_aligned=False,
                    only_center_face=args.only_center_face,
                    paste_back=True
                )

            outp = os.path.join(args.output, fn)
            if not cv2.imwrite(outp, restored):
                raise RuntimeError("cv2.imwrite failed")
            if idx == 1 or idx % 10 == 0 or idx == len(files):
                print(f"[{idx}/{len(files)}] {fn} -> {outp}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] {fn}: {e}")
            traceback.print_exc(limit=1)

    print(f"Done. Failed: {failed}. Output: {args.output}")

if __name__ == "__main__":
    main()
