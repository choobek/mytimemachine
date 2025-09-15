import sys
from pathlib import Path


def _prepare_paths():
    REPO_ROOT = Path(__file__).resolve().parents[1]
    fixtures = REPO_ROOT / 'tests' / 'fixtures' / 'small_train'
    sys.path.insert(0, str(REPO_ROOT))
    from configs import paths_config
    paths_config.dataset_paths['ffhq'] = str(fixtures / 'train')
    paths_config.dataset_paths['celeba_test'] = str(fixtures / 'test')


def test_train_50steps_idadv_tta(tmp_path):
    import torch
    if (not torch.cuda.is_available()) or torch.cuda.device_count() == 0:
        import pytest
        pytest.skip("CUDA not available")
    # No special IO throttles; rely on default logging

    _prepare_paths()
    from scripts.train import main as train_main
    exp = tmp_path / 'exp'
    argv_backup = list(sys.argv)
    sys.argv = [
        'train.py', '--dataset_type', 'ffhq_aging', '--coach', 'orig', '--workers', '0', '--test_workers', '0',
        '--batch_size', '1', '--test_batch_size', '1', '--target_age', 'uniform_random', '--exp_dir', str(exp),
        '--max_steps', '50', '--board_interval', '25', '--val_interval', '1000000', '--save_interval', '50',
        '--train_encoder', '--start_from_encoded_w_plus', '--learning_rate', '1e-4', '--nan_guard', '--lpips_lambda', '0.1', '--l2_lambda', '0.05',
        '--id_adv_enabled', '--id_adv_lambda', '0.02', '--id_adv_model_path', 'pretrained_models/model_ir_se50.pth', '--id_adv_backend', 'arcface',
        '--id_adv_input_size', '112', '--id_adv_tta', 'clean,flip,jpeg75,blur0.6', '--id_adv_agg', 'mean(clean,flip)+0.5*min(jpeg75,blur0.6)', '--id_adv_focal_gamma', '0.0', '--id_adv_margin', '0.0',
        '--contrastive_id_lambda', '0.01', '--mb_index_path', 'banks/ffhq_ir50_age_5y.pt', '--mb_use_faiss', '--mb_k', '8'
    ]
    try:
        train_main()
    finally:
        sys.argv = argv_backup

    run = sorted([p for p in exp.iterdir() if p.is_dir()])[0]
    ts = (run / 'checkpoints' / 'timestamp.txt').read_text().splitlines()
    # Find last log line with id_adv scalars
    mean_val = None
    min_val = None
    for line in reversed(ts):
        if 'id_adv_p_actor_mean' in line and 'id_adv_p_actor_min_aug' in line:
            # crude parse
            try:
                parts = line.split(',')
                for p in parts:
                    if 'id_adv_p_actor_mean' in p:
                        mean_val = float(p.split(':')[-1])
                    if 'id_adv_p_actor_min_aug' in p:
                        min_val = float(p.split(':')[-1])
                break
            except Exception:
                continue
    assert mean_val is not None and min_val is not None
    assert min_val <= mean_val + 1e-6
