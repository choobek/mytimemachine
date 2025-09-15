from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_ce(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 0.0, reduction: str = 'mean') -> torch.Tensor:
    """
    Numerically stable focal cross-entropy on logits.
    - logits: [B, C]
    - targets: [B] int64 class indices (0..C-1)
    - gamma: focusing parameter (0 -> standard CE)

    Works in mixed precision by operating in logits space and using log_softmax.
    """
    # Convert gamma to float on same device for AMP safety
    gamma = float(gamma)
    # Standard CE path when gamma == 0
    if gamma == 0.0:
        return F.cross_entropy(logits, targets, reduction=reduction)

    # log p_t for the true class via log_softmax
    log_probs = F.log_softmax(logits, dim=1)
    # Gather true-class log prob
    log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
    # p_t in [0,1]; clamp to avoid diff issues at 0
    pt = log_pt.exp().clamp(min=1e-12, max=1.0)
    # Focal weight (1 - p_t)^gamma
    focal_weight = (1.0 - pt).pow(gamma)
    loss = -focal_weight * log_pt
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def logit_margin_hinge(z_actor: torch.Tensor, z_nonactor: torch.Tensor, margin: float = 0.0, reduction: str = 'mean') -> torch.Tensor:
    """
    Margin hinge on binary logits z_actor (positive class) and z_nonactor (negative class).
    Loss = relu( m - (z_actor - z_nonactor) ).

    Shapes:
    - z_actor: [B] or [B,1]
    - z_nonactor: [B] or [B,1]
    - margin m >= 0
    """
    m = float(margin)
    # ensure vector shapes
    za = z_actor.view(-1)
    zn = z_nonactor.view(-1)
    gap = za - zn
    loss_vec = torch.relu(m - gap)
    if reduction == 'mean':
        return loss_vec.mean()
    elif reduction == 'sum':
        return loss_vec.sum()
    else:
        return loss_vec


def conf_weight(p_actor: torch.Tensor, k: float, p_thr: float) -> torch.Tensor:
    """
    Confidence-adaptive weight in [0,1] based on actor-class probability.
    Intuition: downweight when model is already confident it's actor (p_actor high).
    A smooth decreasing curve around threshold p_thr with steepness k:
        w = 1 / (1 + exp(k * (p_actor - p_thr)))
    - If p_actor >> p_thr => w ~ 0 (less push needed)
    - If p_actor << p_thr => w ~ 1 (more push needed)
    """
    k = float(k)
    p_thr = float(p_thr)
    # Normalize slope by p_thr*(1-p_thr) so k has consistent sharpness near threshold
    denom = max(p_thr * (1.0 - p_thr), 1e-6)
    x = (p_actor - p_thr) * (k / denom)
    # Use clamp to keep exponent in stable range
    x = torch.clamp(x, min=-60.0, max=60.0)
    w = 1.0 / (1.0 + torch.exp(x))
    return w.clamp(0.0, 1.0)


def parse_conf_weight_spec(spec: str | None):
    """
    Parse spec like "k=6,p_thr=0.9" -> (k, p_thr). Returns None if empty/invalid.
    """
    if not spec:
        return None
    try:
        parts = [t.strip() for t in str(spec).split(',') if t.strip()]
        kv = {}
        for p in parts:
            if '=' in p:
                k, v = p.split('=', 1)
                kv[k.strip()] = float(v.strip())
        if 'k' in kv and 'p_thr' in kv:
            return float(kv['k']), float(kv['p_thr'])
    except Exception:
        pass
    return None


