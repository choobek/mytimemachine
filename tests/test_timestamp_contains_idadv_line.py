import sys
from pathlib import Path


def _prepare_paths():
	REPO_ROOT = Path(__file__).resolve().parents[1]
	sys.path.insert(0, str(REPO_ROOT))
	fixtures = REPO_ROOT / 'tests' / 'fixtures' / 'small_train'
	from configs import paths_config
	paths_config.dataset_paths['ffhq'] = str(fixtures / 'train')
	paths_config.dataset_paths['celeba_test'] = str(fixtures / 'test')


def test_timestamp_contains_idadv_line(tmp_path):
	import torch
	if (not torch.cuda.is_available()) or torch.cuda.device_count() == 0:
		import pytest
		pytest.skip("CUDA not available")
	_prepare_paths()
	from scripts.train import main as train_main

	exp = tmp_path / 'exp'
	argv_backup = list(sys.argv)
	sys.argv = [
		'train.py', '--dataset_type', 'ffhq_aging', '--coach', 'orig_nn', '--workers', '0', '--test_workers', '0',
		'--batch_size', '1', '--test_batch_size', '1', '--target_age', 'uniform_random', '--exp_dir', str(exp),
		'--max_steps', '10', '--board_interval', '10', '--val_interval', '10', '--val_max_batches', '1', '--save_interval', '10',
		'--train_encoder', '--start_from_encoded_w_plus', '--learning_rate', '1e-4', '--nan_guard', '--lpips_lambda', '0.1', '--l2_lambda', '0.05',
		'--id_adv_enabled', '--id_adv_lambda', '0.02', '--id_adv_model_path', 'pretrained_models/model_ir_se50.pth', '--id_adv_backend', 'arcface', '--id_adv_input_size', '112',
	]
	try:
		train_main()
	finally:
		sys.argv = argv_backup

	run = sorted([p for p in exp.iterdir() if p.is_dir()])[0]
	ts = (run / 'checkpoints' / 'timestamp.txt').read_text().splitlines()
	last = ""
	for line in reversed(ts):
		if line.startswith('idadv:'):
			last = line
			break
	assert last.startswith('idadv:')
	# Must contain these tokens
	for tok in ['lam=', 'p=', 'p_min=', 'margin=', 'focal', 'm=', 'views=', 'agg=']:
		assert tok in last

