#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/prepro_filtering.py

Step 2: Quality filtering + redundancy removal for MyTimeMachine.
- Filters small, grayscale, and blurry images.
- Removes duplicates with perceptual hashing.
- Optionally removes near-duplicates using face embeddings (InsightFace if available).
- Writes results to dataset_prep/060_filtered/{valid,trash,duplicates}.

Usage (defaults match your repo):
  python scripts/prepro_filtering.py \
    --input_dir dataset_prep/040_topaz \
    --output_dir dataset_prep/060_filtered \
    --min_size 256 --blur_thresh 100.0 \
    --hash_hamm_thresh 4 --cosine_thresh 0.97
  # Add --move to move files instead of copying.
  # If you have 'insightface' installed: pip install insightface onnxruntime-gpu (or onnxruntime)
"""
import argparse
import os
import sys
import shutil
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set

import cv2
import numpy as np

# Optional acceleration niceties
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # fallback no-op

# ----------------------------- I/O helpers -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def unique_path(dst_dir: Path, name: str) -> Path:
    base, ext = os.path.splitext(name)
    out = dst_dir / name
    i = 1
    while out.exists():
        out = dst_dir / f"{base}_{i}{ext}"
        i += 1
    return out

def transfer(src: Path, dst_dir: Path, do_move: bool) -> Path:
    ensure_dir(dst_dir)
    dst = unique_path(dst_dir, src.name)
    if do_move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))
    return dst

# ----------------------------- image utils -----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def load_bgr(path: Path) -> Optional[np.ndarray]:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    return img

def variance_of_laplacian(gray: np.ndarray) -> float:
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_grayscale_bgr(bgr: np.ndarray, tol: int = 0) -> bool:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        return True
    b, g, r = bgr[:,:,0].astype(np.int16), bgr[:,:,1].astype(np.int16), bgr[:,:,2].astype(np.int16)
    return (np.max(np.abs(b - g)) <= tol) and (np.max(np.abs(b - r)) <= tol)

def brightness_stats(gray: np.ndarray) -> Tuple[float, float]:
    return float(gray.mean()), float(gray.std())

def quality_tuple(bgr: np.ndarray, lap_var: float) -> Tuple[float, int, float]:
    h, w = bgr.shape[:2]
    min_dim = min(h, w)
    mean_b, std_b = brightness_stats(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    # Lexicographic comparison prefers higher lap_var, then larger min_dim, then higher contrast
    return (lap_var, min_dim, std_b)

# ----------------------------- perceptual hash (dHash) -----------------------------
def dhash(gray: np.ndarray, hash_size: int = 8) -> int:
    # Resize to (hash_size+1, hash_size), compare adjacent columns
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    # Pack to 64-bit int
    bits = 0
    idx = 0
    for row in diff:
        for v in row:
            if v:
                bits |= (1 << idx)
            idx += 1
    return bits

def hamming_distance64(a: int, b: int) -> int:
    return (a ^ b).bit_count()

# ----------------------------- optional InsightFace embeddings -----------------------------
def try_init_insightface(det_size=(320, 320)):
    """
    Returns (app, use_gpu) or (None, False) if insightface not available.
    """
    try:
        from insightface.app import FaceAnalysis
        import onnxruntime as ort  # type: ignore
        providers = ort.get_available_providers()
        # Prefer CUDA if present
        cuda_ok = any("CUDAExecutionProvider" in p for p in providers)
        app = FaceAnalysis(name="buffalo_l")  # includes detection + recognition
        # ctx_id: 0 for GPU if available; -1 for CPU
        app.prepare(ctx_id=0 if cuda_ok else -1, det_size=det_size)
        return app, cuda_ok
    except Exception as e:
        print(f"[info] insightface not available or failed to init ({e}). Skipping embedding dedup.", flush=True)
        return None, False

def embed_face_insightface(app, bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Run detection + recognition, return L2-normalized embedding for the largest face.
    """
    try:
        faces = app.get(bgr)
        if not faces:
            return None
        # choose largest bbox
        areas = [(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in faces]
        f = faces[int(np.argmax(areas))]
        # Some models expose normed_embedding; else normalize
        emb = getattr(f, "normed_embedding", None)
        if emb is None:
            emb = getattr(f, "embedding", None)
            if emb is None:
                return None
            emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb.astype(np.float32)
    except Exception:
        return None

# ----------------------------- main pipeline -----------------------------
def collect_images(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and is_image(p):
            files.append(p)
    return files

def main():
    ap = argparse.ArgumentParser(description="Quality filter + dedup images")
    ap.add_argument("--input_dir", type=Path, default=Path("dataset_prep/040_topaz"))
    ap.add_argument("--output_dir", type=Path, default=Path("dataset_prep/060_filtered"))
    ap.add_argument("--min_size", type=int, default=256, help="Minimum width/height")
    ap.add_argument("--blur_thresh", type=float, default=100.0, help="Variance of Laplacian threshold")
    ap.add_argument("--hash_hamm_thresh", type=int, default=4, help="Max Hamming distance to consider hashes duplicate")
    ap.add_argument("--cosine_thresh", type=float, default=0.97, help="Cosine similarity for embedding near-duplicates")
    ap.add_argument("--move", action="store_true", help="Move files instead of copying")
    ap.add_argument("--skip_embeddings", action="store_true", help="Skip embedding-based dedup")
    args = ap.parse_args()

    in_dir: Path = args.input_dir
    out_dir: Path = args.output_dir
    valid_dir = out_dir / "valid"
    trash_dir = out_dir / "trash"
    dup_dir = out_dir / "duplicates"

    ensure_dir(valid_dir); ensure_dir(trash_dir); ensure_dir(dup_dir)

    src_images = collect_images(in_dir)
    if not src_images:
        print(f"[err] No images found in {in_dir}")
        sys.exit(1)

    print(f"[info] Found {len(src_images)} images in {in_dir}")

    kept: List[Tuple[Path, Path, Tuple[float,int,float], int]] = []  # (src, dst_valid, quality_tuple, dhash)
    trashed: List[Path] = []
    dup_hash_removed: List[Path] = []

    # ---------- Pass 1: quality filtering ----------
    seen_hashes: Dict[int, Tuple[Path, Tuple[float,int,float]]] = {}
    for src in tqdm(src_images, desc="Quality check + phash"):
        try:
            bgr = load_bgr(src)
            if bgr is None:
                trashed.append(transfer(src, trash_dir, args.move))
                continue
            h, w = bgr.shape[:2]
            if min(h, w) < args.min_size:
                trashed.append(transfer(src, trash_dir, args.move))
                continue
            if is_grayscale_bgr(bgr, tol=0):
                trashed.append(transfer(src, trash_dir, args.move))
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            lap = variance_of_laplacian(gray)
            if lap < args.blur_thresh:
                trashed.append(transfer(src, trash_dir, args.move))
                continue

            q = quality_tuple(bgr, lap)
            # Perceptual hash duplicate screen
            hsh = dhash(gray, hash_size=8)
            # If exact hash seen, choose better quality to keep
            if hsh in seen_hashes:
                prev_src, prev_q = seen_hashes[hsh]
                # Compare qualities: keep better, other -> duplicates
                if q > prev_q:
                    # replace kept with current; move previous to dup
                    transfer(prev_src, dup_dir, True)  # move out of valid if we already placed it
                    seen_hashes[hsh] = (src, q)
                    dst = transfer(src, valid_dir, args.move)
                    kept.append((src, dst, q, hsh))
                else:
                    dup_hash_removed.append(transfer(src, dup_dir, args.move))
                continue

            # For near-equal hashes (small Hamming distance), also dedup here
            # (We compare only against a subset: this is O(N) per image; manageable for ~hundreds)
            is_near_dup = False
            for hprev, (psrc, pq) in list(seen_hashes.items()):
                if hamming_distance64(hsh, hprev) <= args.hash_hamm_thresh:
                    # near identical -> keep better quality
                    if q > pq:
                        transfer(psrc, dup_dir, True)
                        del seen_hashes[hprev]
                        seen_hashes[hsh] = (src, q)
                        dst = transfer(src, valid_dir, args.move)
                        kept.append((src, dst, q, hsh))
                    else:
                        dup_hash_removed.append(transfer(src, dup_dir, args.move))
                    is_near_dup = True
                    break
            if is_near_dup:
                continue

            # Keep candidate
            seen_hashes[hsh] = (src, q)
            dst = transfer(src, valid_dir, args.move)
            kept.append((src, dst, q, hsh))

        except Exception as e:
            print(f"[warn] Failed {src}: {e}")
            trashed.append(transfer(src, trash_dir, args.move))

    print(f"[info] Kept after quality+hash: {len(kept)} | trashed: {len(trashed)} | hash-duplicates: {len(dup_hash_removed)}")

    # ---------- Pass 2: embedding-based near-duplicate removal (optional) ----------
    if not args.skip_embeddings and len(kept) > 1:
        app, _gpu = try_init_insightface()
        if app is not None:
            # Build embeddings for all kept images in 'valid'
            valid_items = []
            for (src, dst, q, hsh) in tqdm(kept, desc="Embedding"):
                bgr = load_bgr(dst)  # use the copied/moved file
                if bgr is None:
                    continue
                emb = embed_face_insightface(app, bgr)
                if emb is None:
                    # No face found; you can choose to ignore or consider as unique
                    continue
                valid_items.append((src, dst, q, emb))

            # Pairwise compare and remove high-similarity duplicates (keep best quality)
            removed_by_emb: Set[Path] = set()
            n = len(valid_items)
            for i in tqdm(range(n), desc="Embedding dedup"):
                si, di, qi, ei = valid_items[i]
                if di in removed_by_emb:
                    continue
                for j in range(i + 1, n):
                    sj, dj, qj, ej = valid_items[j]
                    if dj in removed_by_emb:
                        continue
                    sim = float(np.dot(ei, ej))
                    if sim >= args.cosine_thresh:
                        # choose better by quality tuple
                        if qi >= qj:
                            # remove j
                            transfer(dj, dup_dir, True)  # move from valid to duplicates
                            removed_by_emb.add(dj)
                        else:
                            transfer(di, dup_dir, True)
                            removed_by_emb.add(di)
                            break  # current i was removed; stop comparing it

            # Recompute final kept list (exclude removed_by_emb)
            kept = [(s, d, q, h) for (s, d, q, h) in kept if d not in removed_by_emb]
            print(f"[info] Embedding dedup removed: {len(removed_by_emb)}")
        else:
            print("[info] Skipped embedding dedup (insightface unavailable).")

    # ---------- Summary ----------
    # Count outputs
    kept_count = len(kept)
    trash_count = len(list(trash_dir.glob("*")))
    dup_count = len(list(dup_dir.glob("*")))

    print("\n==== Summary ====")
    print(f"Input images:         {len(src_images)}")
    print(f"Kept (valid):         {kept_count}")
    print(f"Moved to trash:       {trash_count}")
    print(f"Moved to duplicates:  {dup_count}")
    print(f"Output dir:           {out_dir}")
    print("Done.")

if __name__ == "__main__":
    main()
