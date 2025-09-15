import os
import torch
import types


def test_faiss_miner_api_unchanged():
    # Import without side effects
    mod = __import__('training.impostor_faiss', fromlist=['AgeMatchedImpostorMiner'])
    assert hasattr(mod, 'AgeMatchedImpostorMiner')
    cls = getattr(mod, 'AgeMatchedImpostorMiner')
    assert isinstance(cls, type)

    # Build a tiny fake bank on disk with 2 bins
    tmp = {
        'bins': {
            '33-37': torch.nn.functional.normalize(torch.randn(10, 512), dim=1),
            '38-42': torch.nn.functional.normalize(torch.randn(12, 512), dim=1),
        },
        'ages': {
            '33-37': torch.randint(33, 38, (10,)),
            '38-42': torch.randint(38, 43, (12,)),
        },
    }
    path = 'tests/tmp_fake_bank.pt'
    os.makedirs('tests', exist_ok=True)
    torch.save(tmp, path)

    miner = cls(path, use_faiss=False)
    assert hasattr(miner, 'query') and isinstance(miner.query, types.MethodType)

    # Query signature and outputs
    q = torch.nn.functional.normalize(torch.randn(4, 512), dim=1)
    ages = torch.tensor([34, 41, 36, 39], dtype=torch.long)
    vecs, sims, meta = miner.query(q, ages, k=5, min_sim=0.2, max_sim=0.7, top_m=16, radius=0)
    assert isinstance(vecs, torch.Tensor) and vecs.shape == (4, 5, 512)
    assert isinstance(sims, torch.Tensor) and sims.shape == (4, 5)
    assert isinstance(meta, dict)
    # Expected diagnostics keys present
    for key in ['candidate_count', 'sim_mean', 'sim_p75', 'sim_p90', 'k_effective', 'band_min', 'band_max']:
        assert key in meta



