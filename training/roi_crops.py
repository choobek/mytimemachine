from __future__ import annotations

import random
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F

try:
    import dlib  # type: ignore
    _HAS_DLIB = True
except Exception:
    _HAS_DLIB = False


def _to_uint8_img(t: torch.Tensor) -> torch.Tensor:
    """expects CHW float in [-1,1] or [0,1]; returns HWC uint8 on CPU"""
    x = t.detach().cpu()
    if x.min() < 0:
        x = (x + 1.0) * 0.5
    x = (x.clamp(0, 1) * 255.0).round().to(torch.uint8)
    x = x.permute(1, 2, 0).contiguous()  # HWC
    return x


def _bbox_from_landmarks(pts: torch.Tensor) -> Tuple[int, int, int, int]:
    """pts: [N,2] xy -> integer bbox (x0,y0,x1,y1) inclusive-right/top clamped later"""
    x0 = int(pts[:, 0].min().item())
    y0 = int(pts[:, 1].min().item())
    x1 = int(pts[:, 0].max().item())
    y1 = int(pts[:, 1].max().item())
    return x0, y0, x1, y1


def _expand_pad_jitter(x0: int, y0: int, x1: int, y1: int, H: int, W: int,
                       pad: float, jitter: float, train: bool) -> Tuple[int, int, int, int]:
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = (x1 - x0)
    h = (y1 - y0)
    s = max(w, h)
    s = s * (1.0 + float(pad))
    if train and jitter > 0:
        jx = (random.uniform(-jitter, jitter)) * s
        jy = (random.uniform(-jitter, jitter)) * s
        js = (1.0 + random.uniform(-jitter, jitter))
        cx += jx
        cy += jy
        s *= js
    half = s / 2.0
    nx0 = int(round(max(0.0, cx - half)))
    ny0 = int(round(max(0.0, cy - half)))
    nx1 = int(round(min(float(W - 1), cx + half)))
    ny1 = int(round(min(float(H - 1), cy + half)))
    # ensure at least 2px
    if nx1 <= nx0:
        nx1 = min(W - 1, nx0 + 2)
    if ny1 <= ny0:
        ny1 = min(H - 1, ny0 + 2)
    return nx0, ny0, nx1, ny1


class LandmarkCropper:
    def __init__(self, predictor_path: str = ""):
        self.detector = None
        self.predictor = None
        self.ready = False
        # Try to get default predictor path from configs if not provided
        default_predictor = None
        if len(predictor_path) == 0:
            try:
                from configs.paths_config import model_paths  # lazy import
                default_predictor = model_paths.get('shape_predictor', None)
            except Exception:
                default_predictor = None
        try:
            if _HAS_DLIB:
                self.detector = dlib.get_frontal_face_detector()
                use_path = predictor_path or default_predictor
                if use_path and len(str(use_path)) > 0:
                    try:
                        self.predictor = dlib.shape_predictor(use_path)
                    except Exception:
                        self.predictor = None
                self.ready = self.detector is not None
        except Exception:
            self.ready = False

    def _detect_landmarks(self, img_hwc_uint8: torch.Tensor) -> Optional[torch.Tensor]:
        if (not _HAS_DLIB) or (self.detector is None) or (self.predictor is None):
            return None
        gray = img_hwc_uint8.numpy()
        dets = self.detector(gray, 0)
        if len(dets) == 0:
            return None
        rect = dets[0]
        shape = self.predictor(gray, rect)
        pts = torch.tensor([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=torch.float32)
        return pts  # [68,2]

    def landmarks(self, img_chw: torch.Tensor) -> Optional[torch.Tensor]:
        """Return 68x2 landmarks for a single CHW tensor or None if detection fails."""
        img_u8 = _to_uint8_img(img_chw)
        return self._detect_landmarks(img_u8)

    def landmarks_batch(self, imgs_bchw: torch.Tensor) -> Optional[torch.Tensor]:
        """Return Bx68x2 landmarks or None if any sample fails (caller may choose to skip batch)."""
        if imgs_bchw is None:
            return None
        B = int(imgs_bchw.shape[0])
        out = []
        for b in range(B):
            pts = self.landmarks(imgs_bchw[b])
            if pts is None:
                return None
            out.append(pts.unsqueeze(0))
        return torch.cat(out, dim=0)  # [B,68,2]

    def rois(self, img_chw: torch.Tensor, pad: float, jitter: float, roi_size: int, train: bool,
             use_eyes: bool = True, use_mouth: bool = True, return_info: bool = False) -> Dict[str, torch.Tensor]:
        """
        Returns dict of crops {'eyes': [3,H,W], 'mouth': [3,H,W]} resized to roi_size.
        Works with aligned faces; falls back to heuristic boxes if landmarks unavailable.
        If return_info is True, also returns (crops, {'landmarks_used': bool}).
        """
        H, W = int(img_chw.shape[1]), int(img_chw.shape[2])
        img_u8 = _to_uint8_img(img_chw)  # HWC uint8

        pts = self._detect_landmarks(img_u8)
        crops: Dict[str, torch.Tensor] = {}
        landmarks_used = pts is not None

        def _crop_resize(x0: int, y0: int, x1: int, y1: int) -> torch.Tensor:
            c = img_chw[:, y0:y1, x0:x1].unsqueeze(0)  # 1,C,h,w
            c = F.interpolate(c, size=(roi_size, roi_size), mode="bilinear", align_corners=False)
            return c.squeeze(0)

        if pts is not None:
            if use_eyes:
                eye_idx = list(range(36, 48))
                ex0, ey0, ex1, ey1 = _bbox_from_landmarks(pts[eye_idx])
                ex0, ey0, ex1, ey1 = _expand_pad_jitter(ex0, ey0, ex1, ey1, H, W, pad, jitter, train)
                crops["eyes"] = _crop_resize(ex0, ey0, ex1, ey1)
            if use_mouth:
                m_idx = list(range(48, 68))
                mx0, my0, mx1, my1 = _bbox_from_landmarks(pts[m_idx])
                mx0, my0, mx1, my1 = _expand_pad_jitter(mx0, my0, mx1, my1, H, W, pad, jitter, train)
                crops["mouth"] = _crop_resize(mx0, my0, mx1, my1)
        else:
            # Heuristic fallback: boxes anchored to canonical aligned face regions (scaled by H/W)
            def _box_rel(x0: float, y0: float, x1: float, y1: float) -> torch.Tensor:
                X0 = int(x0 * W)
                Y0 = int(y0 * H)
                X1 = int(x1 * W)
                Y1 = int(y1 * H)
                X0, Y0, X1, Y1 = _expand_pad_jitter(X0, Y0, X1, Y1, H, W, pad, jitter, train)
                return _crop_resize(X0, Y0, X1, Y1)
            if use_eyes:
                crops["eyes"] = _box_rel(0.26, 0.30, 0.74, 0.53)
            if use_mouth:
                crops["mouth"] = _box_rel(0.35, 0.60, 0.65, 0.86)

        if return_info:
            return crops, {"landmarks_used": landmarks_used}
        return crops

    def rois_from_landmarks(self, img_chw: torch.Tensor, pts: torch.Tensor, pad: float, jitter: float,
                            roi_size: int, train: bool, use_eyes: bool = True, use_mouth: bool = True) -> Dict[str, torch.Tensor]:
        """Crop ROIs using provided 68x2 landmarks; avoids running detector again."""
        H, W = int(img_chw.shape[1]), int(img_chw.shape[2])
        crops: Dict[str, torch.Tensor] = {}

        def _crop_resize(x0: int, y0: int, x1: int, y1: int) -> torch.Tensor:
            c = img_chw[:, y0:y1, x0:x1].unsqueeze(0)
            c = F.interpolate(c, size=(roi_size, roi_size), mode="bilinear", align_corners=False)
            return c.squeeze(0)

        if pts is not None and isinstance(pts, torch.Tensor) and pts.numel() == 68 * 2:
            if use_eyes:
                eye_idx = list(range(36, 48))
                ex0, ey0, ex1, ey1 = _bbox_from_landmarks(pts[eye_idx])
                ex0, ey0, ex1, ey1 = _expand_pad_jitter(ex0, ey0, ex1, ey1, H, W, pad, jitter, train)
                crops["eyes"] = _crop_resize(ex0, ey0, ex1, ey1)
            if use_mouth:
                m_idx = list(range(48, 68))
                mx0, my0, mx1, my1 = _bbox_from_landmarks(pts[m_idx])
                mx0, my0, mx1, my1 = _expand_pad_jitter(mx0, my0, mx1, my1, H, W, pad, jitter, train)
                crops["mouth"] = _crop_resize(mx0, my0, mx1, my1)
        return crops


