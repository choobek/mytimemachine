import torch
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_margin_hinge_zero_when_gap_exceeds_margin():
    from training.losses_idadv import logit_margin_hinge
    # z_actor higher than z_nonactor by > m => zero loss
    za = torch.tensor([2.0, 1.5, 0.0])
    zn = torch.tensor([0.0, 0.0, -1.0])
    m = 0.5
    loss = logit_margin_hinge(za, zn, margin=m, reduction='none')
    # gaps: [2, 1.5, 1.0] -> relu(m - gap) -> [0, 0, 0]
    assert torch.all(loss == 0)


def test_margin_hinge_positive_when_violated():
    from training.losses_idadv import logit_margin_hinge
    # gaps: [0.2, 0.6], m=0.5 -> [0.3, 0.0]
    za = torch.tensor([0.2, 0.6])
    zn = torch.tensor([0.0, 0.0])
    m = 0.5
    loss = logit_margin_hinge(za, zn, margin=m, reduction='none')
    assert torch.allclose(loss, torch.tensor([0.3, 0.0]))


