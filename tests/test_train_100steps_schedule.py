from pathlib import Path
import sys


def _prepare_paths():
	REPO_ROOT = Path(__file__).resolve().parents[1]
	fixtures = REPO_ROOT / 'tests' / 'fixtures' / 'small_train'
	sys.path.insert(0, str(REPO_ROOT))
	from configs import paths_config
	paths_config.dataset_paths['ffhq'] = str(fixtures / 'train')
	paths_config.dataset_paths['celeba_test'] = str(fixtures / 'test')


def test_train_100steps_schedule(tmp_path):
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
		'--max_steps', '30', '--board_interval', '10', '--val_interval', '1000000', '--save_interval', '10',
		'--train_encoder', '--start_from_encoded_w_plus', '--learning_rate', '1e-4', '--nan_guard', '--lpips_lambda', '0.1', '--l2_lambda', '0.05',
		'--id_adv_enabled', '--id_adv_model_path', 'pretrained_models/model_ir_se50.pth', '--id_adv_backend', 'arcface', '--id_adv_input_size', '112',
		'--id_adv_tta', 'clean,flip,jpeg75,blur0.6', '--id_adv_agg', 'mean(clean,flip)+0.5*min(jpeg75,blur0.6)',
		'--id_adv_schedule_s1', '0:0.01,10:0.02,20:0.03'
	]
	try:
		train_main()
	finally:
		sys.argv = argv_backup
	# Read timeline of logged schedule from timestamp
	run = sorted([p for p in exp.iterdir() if p.is_dir()])[0]
	ts_lines = (run / 'checkpoints' / 'timestamp.txt').read_text().splitlines()
	lam_vals = []
	for line in ts_lines:
		if 'id_adv_lambda_current' in line:
			try:
				# parse rough float value after ':'
				parts = line.split(',')
				for p in parts:
					if 'id_adv_lambda_current' in p:
						lam_vals.append(float(p.split(':')[-1]))
			except Exception:
				pass
	# Expect values to include the staged lambdas
	assert any(abs(v - 0.01) < 1e-6 for v in lam_vals)
	assert any(abs(v - 0.02) < 1e-6 for v in lam_vals)
	assert any(abs(v - 0.03) < 1e-6 for v in lam_vals)
