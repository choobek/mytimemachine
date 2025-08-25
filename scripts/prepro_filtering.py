#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/prepro_filtering.py

Quality filtering + redundancy removal for MyTimeMachine.
Now with:
- Face-aware blur scoring (--blur_scope full|face|face_inner)
- Resize before measuring sharpness (--resize_max_side)
- Tenengrad metric + Laplacian (--blur_metric lap|tenengrad|both)
- Clear reason logs + optional CSV

Usage:
  python scripts/prepro_filtering.py \
    --input_dir dataset_prep/040_topaz \
    --output_dir dataset_prep/060_filtered \
    --resize_max_side 1024 \
    --blur_scope face_inner \
    --blur_metric both \
    --blur_thresh 45 --tenengrad_thresh 500 \
    --hash_hamm_thresh 4 --cosine_thresh 0.97 \
    --log_csv
"""
import argparse
import os
import sys
import shutil
import csv
import collections
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set

import cv2
import numpy as np

# Progress niceties
try:
    from tqdm import tqdm
    tqdmp = tqdm
    def pwrite(msg: str) -> None:
        tqdm.write(msg)
except Exception:
    def tqdmp(x, **k): return x
    def pwrite(msg: str) -> None:
        print(msg, flush=True)

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
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def resize_keep_aspect(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_wh = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(img, new_wh, interpolation=cv2.INTER_AREA)

def variance_of_laplacian(gray: np.ndarray) -> float:
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    g2 = gx * gx + gy * gy
    return float(np.mean(g2))

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
    _, std_b = brightness_stats(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    return (lap_var, min_dim, std_b)

# ----------------------------- perceptual hash (dHash) -----------------------------
def dhash(gray: np.ndarray, hash_size: int = 8) -> int:
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
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

# ----------------------------- optional InsightFace (for blur scope + embeddings) -----------------------------
def try_init_insightface(det_size=(320, 320)):
    """
    Returns (app, use_gpu) or (None, False) if insightface not available.
    """
    try:
        from insightface.app import FaceAnalysis
        import onnxruntime as ort  # type: ignore
        providers = ort.get_available_providers()
        cuda_ok = any("CUDAExecutionProvider" in p for p in providers) or any("TensorrtExecutionProvider" in p for p in providers)
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0 if cuda_ok else -1, det_size=det_size)
        pwrite(f"[info] ORT providers: {providers} | using {'GPU' if cuda_ok else 'CPU'} for insightface")
        return app, cuda_ok
    except Exception as e:
        pwrite(f"[info] insightface not available or failed to init ({e}). Skipping face-aware blur & embedding dedup.")
        return None, False

def get_face_roi(bgr: np.ndarray, app, focus: str = "face") -> Optional[np.ndarray]:
    """
    focus: 'face' -> largest bbox
           'face_inner' -> central crop within face (eyes/nose/mouth emphasis)
    """
    try:
        faces = app.get(bgr)
        if not faces:
            return None
        # largest face
        areas = [(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in faces]
        f = faces[int(np.argmax(areas))]
        x1, y1, x2, y2 = map(int, f.bbox)
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(bgr.shape[1], x2); y2 = min(bgr.shape[0], y2)
        roi = bgr[y1:y2, x1:x2].copy()
        if roi.size == 0:
            return None
        if focus == "face_inner":
            # shrink bbox to central region (emphasize eyes/nose/mouth)
            h, w = roi.shape[:2]
            fx, fy = 0.65, 0.65  # keep central 65%
            nx, ny = int(round(w * fx)), int(round(h * fy))
            sx = (w - nx) // 2; sy = (h - ny) // 2
            roi = roi[sy:sy+ny, sx:sx+nx].copy()
        return roi
    except Exception:
        return None

def embed_face_insightface(app, bgr: np.ndarray) -> Optional[np.ndarray]:
    try:
        faces = app.get(bgr)
        if not faces:
            return None
        areas = [(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in faces]
        f = faces[int(np.argmax(areas))]
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
    # Blur/Sharpness params
    ap.add_argument("--resize_max_side", type=int, default=1024, help="Resize longer side before measuring sharpness (0=off)")
    ap.add_argument("--blur_scope", type=str, choices=["full", "face", "face_inner"], default="full",
                    help="Region to measure sharpness on")
    ap.add_argument("--blur_metric", type=str, choices=["lap", "tenengrad", "both"], default="lap",
                    help="Which sharpness metric to decide with")
    ap.add_argument("--blur_thresh", type=float, default=100.0, help="Variance of Laplacian threshold")
    ap.add_argument("--tenengrad_thresh", type=float, default=500.0, help="Tenengrad mean(G^2) threshold")
    ap.add_argument("--hash_hamm_thresh", type=int, default=4, help="Max Hamming distance to consider hashes duplicate")
    ap.add_argument("--cosine_thresh", type=float, default=0.97, help="Cosine similarity for embedding near-duplicates")
    ap.add_argument("--move", action="store_true", help="Move files instead of copying")
    ap.add_argument("--skip_embeddings", action="store_true", help="Skip embedding-based dedup")
    ap.add_argument("--log_csv", action="store_true", help="Also write CSV logs for trash/kept/dups")
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
    trashed_paths: List[Path] = []
    dup_hash_removed: List[Path] = []

    # reason logging
    trash_reasons = collections.Counter()
    trash_rows = []
    kept_rows = []
    dup_rows = []

    # Init insightface early if we need face crops for blur or for embeddings
    need_face_for_blur = args.blur_scope in ("face", "face_inner")
    app = None
    if need_face_for_blur or (not args.skip_embeddings):
        app, _gpu = try_init_insightface()

    def record_trash(src: Path, reason: str, dst: Optional[Path] = None,
                     w: Optional[int] = None, h: Optional[int] = None,
                     lap: Optional[float] = None, ten: Optional[float] = None,
                     mean: Optional[float] = None, std: Optional[float] = None):
        nonlocal trashed_paths
        if dst is None:
            dst = transfer(src, trash_dir, args.move)
        trashed_paths.append(dst)
        trash_reasons[reason] += 1
        msg = f"[trash] {src} -> {dst.name}: {reason}"
        details = []
        if w is not None and h is not None:
            details.append(f"{w}x{h}")
        if lap is not None:
            details.append(f"lap={lap:.1f}")
        if ten is not None:
            details.append(f"ten={ten:.1f}")
        if mean is not None and std is not None:
            details.append(f"μ={mean:.1f},σ={std:.1f}")
        if details:
            msg += " [" + ", ".join(details) + "]"
        pwrite(msg)
        trash_rows.append([str(src), str(dst), reason,
                           w if w is not None else "",
                           h if h is not None else "",
                           f"{lap:.6f}" if lap is not None else "",
                           f"{ten:.6f}" if ten is not None else "",
                           f"{mean:.6f}" if mean is not None else "",
                           f"{std:.6f}" if std is not None else ""])

    # ---------- Pass 1: quality filtering ----------
    seen_hashes: Dict[int, Tuple[Path, Tuple[float,int,float]]] = {}

    for src in tqdmp(src_images, desc="Quality check + phash"):
        try:
            bgr_full = load_bgr(src)
            if bgr_full is None:
                record_trash(src, "decode_failed")
                continue

            h, w = bgr_full.shape[:2]
            if min(h, w) < args.min_size:
                record_trash(src, f"too_small(<{args.min_size})", w=w, h=h)
                continue

            if is_grayscale_bgr(bgr_full, tol=0):
                gray_tmp = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2GRAY)
                mean_b, std_b = brightness_stats(gray_tmp)
                record_trash(src, "grayscale", w=w, h=h, mean=mean_b, std=std_b)
                continue

            # Choose region for blur metric
            region = bgr_full
            if need_face_for_blur and app is not None:
                roi = get_face_roi(bgr_full, app, focus=("face_inner" if args.blur_scope == "face_inner" else "face"))
                if roi is not None and roi.size > 0:
                    region = roi

            # Resize before metrics for stability
            region = resize_keep_aspect(region, args.resize_max_side)
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

            # Sharpness metrics
            lap = variance_of_laplacian(gray)
            ten = tenengrad(gray)
            mean_b, std_b = brightness_stats(gray)

            # Decide blur
            blurry = False
            if args.blur_metric == "lap":
                blurry = lap < args.blur_thresh
            elif args.blur_metric == "tenengrad":
                blurry = ten < args.tenengrad_thresh
            else:  # both -> only trash if both are low (conservative)
                blurry = (lap < args.blur_thresh) and (ten < args.tenengrad_thresh)

            if blurry:
                record_trash(src, f"blurry({args.blur_metric})", w=w, h=h, lap=lap, ten=ten, mean=mean_b, std=std_b)
                continue

            # Quality tuple for tie-breaks later
            q = quality_tuple(region, lap)

            # Perceptual hash duplicate screen (use region gray so it's consistent)
            hsh = dhash(gray, hash_size=8)

            if hsh in seen_hashes:
                prev_src, prev_q = seen_hashes[hsh]
                if q > prev_q:
                    transfer(prev_src, dup_dir, True)
                    seen_hashes[hsh] = (src, q)
                    dst = transfer(src, valid_dir, args.move)
                    kept.append((src, dst, q, hsh))
                    dup_rows.append([str(prev_src), "exact_hash", str(dst)])
                else:
                    dst_dup = transfer(src, dup_dir, args.move)
                    dup_hash_removed.append(dst_dup)
                    dup_rows.append([str(src), "exact_hash", str(dst_dup)])
                continue

            is_near_dup = False
            for hprev, (psrc, pq) in list(seen_hashes.items()):
                if hamming_distance64(hsh, hprev) <= args.hash_hamm_thresh:
                    if q > pq:
                        transfer(psrc, dup_dir, True)
                        del seen_hashes[hprev]
                        seen_hashes[hsh] = (src, q)
                        dst = transfer(src, valid_dir, args.move)
                        kept.append((src, dst, q, hsh))
                        dup_rows.append([str(psrc), f"hash_hamm<={args.hash_hamm_thresh}", str(dst)])
                    else:
                        dst_dup = transfer(src, dup_dir, args.move)
                        dup_hash_removed.append(dst_dup)
                        dup_rows.append([str(src), f"hash_hamm<={args.hash_hamm_thresh}", str(dst_dup)])
                    is_near_dup = True
                    break
            if is_near_dup:
                continue

            seen_hashes[hsh] = (src, q)
            dst = transfer(src, valid_dir, args.move)
            kept.append((src, dst, q, hsh))
            kept_rows.append([str(src), str(dst)])

        except Exception as e:
            record_trash(src, f"exception:{type(e).__name__}:{e}")

    print(f"[info] Kept after quality+hash: {len(kept)} | trashed: {len(trashed_paths)} | hash-duplicates: {len(dup_hash_removed)}")

    # ---------- Pass 2: embedding-based near-duplicate removal (optional) ----------
    if not args.skip_embeddings and len(kept) > 1 and app is not None:
        valid_items = []
        for (src, dst, q, hsh) in tqdmp(kept, desc="Embedding"):
            bgr = load_bgr(dst)
            if bgr is None:
                continue
            emb = embed_face_insightface(app, bgr)
            if emb is None:
                continue
            valid_items.append((src, dst, q, emb))

        removed_by_emb: Set[Path] = set()
        n = len(valid_items)
        for i in tqdmp(range(n), desc="Embedding dedup"):
            si, di, qi, ei = valid_items[i]
            if di in removed_by_emb:
                continue
            for j in range(i + 1, n):
                sj, dj, qj, ej = valid_items[j]
                if dj in removed_by_emb:
                    continue
                sim = float(np.dot(ei, ej))
                if sim >= args.cosine_thresh:
                    if qi >= qj:
                        transfer(dj, dup_dir, True)
                        removed_by_emb.add(dj)
                        dup_rows.append([str(dj), f"emb_sim>={args.cosine_thresh}", "moved_from_valid"])
                    else:
                        transfer(di, dup_dir, True)
                        removed_by_emb.add(di)
                        dup_rows.append([str(di), f"emb_sim>={args.cosine_thresh}", "moved_from_valid"])
                        break
        kept = [(s, d, q, h) for (s, d, q, h) in kept if d not in removed_by_emb]
        print(f"[info] Embedding dedup removed: {len(removed_by_emb)}")
    elif not args.skip_embeddings:
        print("[info] Skipped embedding dedup (insightface unavailable).")

    # ---------- Summary ----------
    kept_count = len(kept)
    trash_count = len(list(trash_dir.glob("*")))
    dup_count = len(list(dup_dir.glob("*")))

    print("\n==== Summary ====")
    print(f"Input images:         {len(src_images)}")
    print(f"Kept (valid):         {kept_count}")
    print(f"Moved to trash:       {trash_count}")
    print(f"Moved to duplicates:  {dup_count}")
    if trash_reasons:
        print("\nTop trash reasons:")
        for reason, cnt in trash_reasons.most_common():
            print(f"  {reason:20s} : {cnt}")
    print(f"\nOutput dir:           {out_dir}")
    print("Done.")

    # ---------- Optional CSV logs ----------
    if args.log_csv:
        try:
            with open(out_dir / "trash_log.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["src", "dst", "reason", "width", "height", "lap_var", "tenengrad", "brightness_mean", "brightness_std"])
                w.writerows(trash_rows)
            with open(out_dir / "kept_log.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["src", "dst"]); w.writerows(kept_rows)
            with open(out_dir / "duplicates_log.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["src", "reason", "dst"]); w.writerows(dup_rows)
            print(f"[info] CSV logs written under {out_dir}")
        except Exception as e:
            print(f"[warn] Failed to write CSV logs: {e}")

if __name__ == "__main__":
    main()
