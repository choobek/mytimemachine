import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.binary_identity_model import build_identity_model, IdentityModelWrapper


def test_forward_multi_matches_single_clean():
    if (not torch.cuda.is_available()) or torch.cuda.device_count() == 0:
        import pytest
        pytest.skip("CUDA not available; classifier weights expect CUDA")
    model = build_identity_model(backend='arcface', weights_path='pretrained_models/model_ir_se50.pth', num_outputs=2, input_size=112)
    wrapper = IdentityModelWrapper(model).cuda()
    x = torch.randn(2, 3, 112, 112, device='cuda')
    out_single = wrapper.forward_single(x)
    out_multi = wrapper.forward_multi([x])
    # numerical equivalence per element
    assert torch.allclose(out_single['logits'], out_multi['logits'][0], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out_single['probs'], out_multi['probs'][0], atol=1e-6, rtol=1e-6)



