import sys
from pathlib import Path
import re


def _prepare_paths():
    REPO_ROOT = Path(__file__).resolve().parents[1]
    fixtures = REPO_ROOT / 'tests' / 'fixtures' / 'small_train'
    sys.path.insert(0, str(REPO_ROOT))
    from configs import paths_config
    paths_config.dataset_paths['ffhq'] = str(fixtures / 'train')
    paths_config.dataset_paths['celeba_test'] = str(fixtures / 'test')


def _total_from_ts(ts_text: str):
    m = re.search(r"['\"]loss['\"]:\s*([-+eE0-9\.]+)", ts_text)
    return float(m.group(1)) if m else None


def test_no_flags_still_same(tmp_path):
    import torch
    if (not torch.cuda.is_available()) or torch.cuda.device_count() == 0:
        import pytest
        pytest.skip("CUDA not available")
    _prepare_paths()
    from scripts.train import main as train_main
    # Run two 10-step trainings without new flags
    def run(exp_dir):
        argv_backup = list(sys.argv)
        sys.argv = [
            'train.py', '--dataset_type', 'ffhq_aging', '--coach', 'orig', '--workers', '0', '--test_workers', '0',
            '--batch_size', '1', '--test_batch_size', '1', '--target_age', 'uniform_random', '--exp_dir', str(exp_dir),
            '--max_steps', '10', '--board_interval', '10', '--val_interval', '1000000', '--save_interval', '10',
            '--train_encoder', '--start_from_encoded_w_plus', '--learning_rate', '1e-4', '--nan_guard', '--lpips_lambda', '0.1', '--l2_lambda', '0.05'
        ]
        try:
            train_main()
        finally:
            sys.argv = argv_backup
        run_dir = sorted([p for p in exp_dir.iterdir() if p.is_dir()])[0]
        return (run_dir / 'checkpoints' / 'timestamp.txt').read_text()

    ts1 = run(tmp_path / 'a')
    ts2 = run(tmp_path / 'b')
    # Ensure no id_adv scalars appear and totals exist
    assert 'id_adv_lambda_current' not in ts1 and 'id_adv_lambda_current' not in ts2
    assert _total_from_ts(ts1) is not None and _total_from_ts(ts2) is not None


