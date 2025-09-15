import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.losses_idadv import conf_weight, parse_conf_weight_spec


def test_conf_weight_monotonic_and_limits():
    k = 6.0
    p_thr = 0.9
    p = torch.linspace(0.0, 1.0, 101)
    w = conf_weight(p, k=k, p_thr=p_thr)
    # Range
    assert torch.all(w >= 0.0) and torch.all(w <= 1.0)
    # Monotonic decreasing in p
    diffs = (w[1:] - w[:-1]).detach().cpu().numpy()
    assert (diffs <= 1e-8).all()
    # Ends: high p -> ~0; low p -> ~1
    assert w[0].item() > 0.9
    assert w[-1].item() < 0.1


def test_parse_conf_weight_spec():
    assert parse_conf_weight_spec(None) is None
    assert parse_conf_weight_spec("") is None
    val = parse_conf_weight_spec("k=6,p_thr=0.9")
    assert isinstance(val, tuple) and len(val) == 2
    k, p_thr = val
    assert abs(k - 6.0) < 1e-9 and abs(p_thr - 0.9) < 1e-9



