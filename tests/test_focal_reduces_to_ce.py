import torch
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_focal_gamma_zero_matches_ce():
    torch.manual_seed(0)
    logits = torch.randn(8, 2, dtype=torch.float32)
    targets = torch.randint(0, 2, (8,), dtype=torch.int64)
    from training.losses_idadv import focal_ce
    ce = torch.nn.functional.cross_entropy(logits, targets)
    foc = focal_ce(logits, targets, gamma=0.0)
    assert torch.allclose(ce, foc, atol=1e-7, rtol=1e-7)


