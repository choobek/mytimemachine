import sys
import re
from pathlib import Path


def run_train(max_steps: int, exp_dir: Path):
    from scripts.train import main as train_main
    # Monkeypatch image logging to avoid large tmp writes
    import training.coach_aging_orig_nn as coach_mod
    coach_mod.Coach.parse_and_log_images = lambda self, *args, **kwargs: None
    argv_backup = list(sys.argv)
    sys.argv = [
        'train.py',
        '--dataset_type', 'ffhq_aging',
        '--coach', 'orig_nn',
        '--seed', '123',
        '--workers', '0',
        '--test_workers', '0',
        '--batch_size', '1',
        '--test_batch_size', '1',
        '--target_age_fixed', '40',
        '--exp_dir', str(exp_dir),
        '--max_steps', str(max_steps),
        '--board_interval', '15',
        '--image_interval', '1000000',
        '--disable_validation',
        '--val_interval', str(max_steps),
        '--val_deterministic',
        '--val_max_batches', '1',
        '--save_interval', str(max_steps),
        '--train_encoder',
        '--start_from_encoded_w_plus',
        '--learning_rate', '1e-4',
        '--lpips_lambda', '0.1',
        '--l2_lambda', '0.05',
        '--nan_guard',
    ]
    try:
        train_main()
    finally:
        sys.argv = argv_backup


def _extract_total_loss(ts_text: str):
    # Look for "'loss': <num>" in the saved loss-dict lines
    m = re.search(r"['\"]loss['\"]:\s*([-+eE0-9\.]+)", ts_text)
    return float(m.group(1)) if m else None


def test_flags_defaults_noop(tmp_path, monkeypatch):
    import torch
    if (not torch.cuda.is_available()) or torch.cuda.device_count() == 0:
        import pytest
        pytest.skip("CUDA not available; baseline coach requires CUDA")
    # Use the tiny fixtures from Module 0
    REPO_ROOT = Path(__file__).resolve().parents[1]
    fixtures = REPO_ROOT / 'tests' / 'fixtures' / 'small_train'
    # Ensure path config points to fixtures
    sys.path.insert(0, str(REPO_ROOT))
    from configs import paths_config
    paths_config.dataset_paths['ffhq'] = str(fixtures / 'train')
    paths_config.dataset_paths['celeba_test'] = str(fixtures / 'test')

    # Baseline run (no new flags)
    exp_a = tmp_path / 'exp_a'
    run_train(30, exp_a)
    run_a = sorted([p for p in exp_a.iterdir() if p.is_dir()])[0]
    ts_a = (run_a / 'checkpoints' / 'timestamp.txt').read_text()
    loss_a = _extract_total_loss(ts_a)

    # Run with new flags omitted (defaults)
    exp_b = tmp_path / 'exp_b'
    run_train(30, exp_b)
    run_b = sorted([p for p in exp_b.iterdir() if p.is_dir()])[0]
    ts_b = (run_b / 'checkpoints' / 'timestamp.txt').read_text()
    loss_b = _extract_total_loss(ts_b)

    # No new keys such as id_adv_* should appear in logs by default
    assert ('id_adv_' not in ts_a) and ('id_adv_' not in ts_b)
    # Losses should be identical (deterministic run)
    assert loss_a is not None and loss_b is not None
    assert abs(loss_a - loss_b) < 1e-4


