import os
import random
import collections
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from criteria import id_loss, w_norm
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from datasets.augmentations import AgeTransformer
from criteria.lpips.lpips import LPIPS
from criteria.aging_loss import AgingLoss
from models.psp import pSp
from training.ranger import Ranger
import math


class Coach:
	def __init__(self, opts):
		self.opts = opts

		self.global_step = 0

		self.device = 'cuda'
		self.opts.device = self.device

		# Initialize network
		self.net = pSp(self.opts).to(self.device)

		# Initialize loss
		self.mse_loss = nn.MSELoss().to(self.device).eval()
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		# Ensure ID loss is available if used directly, via NN regularizer, or via contrastive impostor loss
		use_nn_lambda = float(getattr(self.opts, 'nearest_neighbor_id_loss_lambda', 0.0) or 0.0) > 0
		use_contrastive_impostor = float(getattr(self.opts, 'contrastive_id_lambda', 0.0) or 0.0) > 0
		if self.opts.id_lambda > 0 or use_nn_lambda or use_contrastive_impostor:
			self.id_loss = id_loss.IDLoss().to(self.device).eval()
		if self.opts.w_norm_lambda > 0:
			self.w_norm_loss = w_norm.WNormLoss(opts=self.opts)
		if self.opts.aging_lambda > 0:
			self.aging_loss = AgingLoss(self.opts)
		# Ensure aging_loss is available when needed by training/NN-ID regardless of lambda
		if not hasattr(self, 'aging_loss'):
			self.aging_loss = AgingLoss(self.opts)

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)

		self.age_transformer = AgeTransformer(target_age=self.opts.target_age)

		# Personalization features (for NN identity regularizer)
		self._maybe_build_training_features()

		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps
		
		# Load checkpoint if resuming
		if hasattr(opts, 'resume_checkpoint') and opts.resume_checkpoint:
			self.load_checkpoint(opts.resume_checkpoint)

		# Contrastive impostor bank and options
		self.contrastive_id_lambda = float(getattr(self.opts, 'contrastive_id_lambda', 0.0) or 0.0)
		self.mb = None
		self.miner = None
		mb_index_path = getattr(self.opts, 'mb_index_path', '') or ''
		self.mb_use_faiss = bool(getattr(self.opts, 'mb_use_faiss', False))
		self.mb_top_m = int(getattr(self.opts, 'mb_top_m', 512) or 512)
		self.mb_min_sim = float(getattr(self.opts, 'mb_min_sim', 0.20) or 0.20)
		self.mb_max_sim = float(getattr(self.opts, 'mb_max_sim', 0.70) or 0.70)
		if self.contrastive_id_lambda > 0 and len(mb_index_path) > 0:
			if self.mb_use_faiss:
				try:
					from training.impostor_faiss import AgeMatchedImpostorMiner
					self.miner = AgeMatchedImpostorMiner(mb_index_path, use_faiss=True)
					self.logger.add_text("setup/mb_backend", "faiss", self.global_step)
				except Exception:
					from training.impostor_bank import AgeMatchedImpostorBank
					self.mb = AgeMatchedImpostorBank(mb_index_path)
					self.logger.add_text("setup/mb_backend", "random", self.global_step)
			else:
				from training.impostor_bank import AgeMatchedImpostorBank
				self.mb = AgeMatchedImpostorBank(mb_index_path)
				self.logger.add_text("setup/mb_backend", "random", self.global_step)
		self.mb_k = int(getattr(self.opts, 'mb_k', 64) or 64)
		self.mb_bin_neighbor_radius = int(getattr(self.opts, 'mb_bin_neighbor_radius', 0) or 0)
		# apply bounds are in years; None disables bound
		self.mb_apply_min_age = getattr(self.opts, 'mb_apply_min_age', None)
		self.mb_apply_max_age = getattr(self.opts, 'mb_apply_max_age', None)
		self.mb_temperature = float(getattr(self.opts, 'mb_temperature', 0.07) or 0.07)

		# Build optional scheduler for stability
		self.scheduler = self._build_scheduler(self.optimizer)

	def _maybe_build_training_features(self):
		"""
		Build feature dictionary keyed by ground-truth ages to support nearest-neighbor
		identity regularization during interpolation.
		"""
		nn_lambda = float(getattr(self.opts, 'nearest_neighbor_id_loss_lambda', 0.0) or 0.0)
		if nn_lambda <= 0:
			self.feats_dict = None
			self.train_min_age = None
			self.train_max_age = None
			return
		assert hasattr(self, 'id_loss'), 'ID loss must be initialized for NN regularizer'
		self.feats_dict = collections.defaultdict(list)
		print('Training face feature extraction:')
		processed = 0
		for x_img, _, x_path, _ in self.train_dataset:
			img = x_img.unsqueeze(0).to(self.device)
			feat = self.id_loss.extract_feats(img)  # shape [1, 512]
			x_age = self.aging_loss.extract_ages_gt(x_path) / 100
			key = round(float(x_age.item()), 2)
			# store CPU tensors to conserve GPU memory
			self.feats_dict[key].append((feat.detach().cpu(), x_path))
			processed += 1
		print(f'Processed all {processed} training images for personalization')
		self.train_min_age = int(min(self.feats_dict.keys()) * 100)
		self.train_max_age = int(max(self.feats_dict.keys()) * 100)
		print('Training data min_age (ground truth):', self.train_min_age)
		print('Training data max_age (ground truth):', self.train_max_age)
		assert processed == len(self.train_dataset)
		assert 0 <= self.train_min_age <= self.train_max_age <= 100

	def perform_forward_pass(self, x):
		y_hat, latent = self.net.forward(x, return_latents=True)
		return y_hat, latent

	def __set_target_to_source(self, x, input_ages):
		return [torch.cat((img, age * torch.ones((1, img.shape[1], img.shape[2])).to(self.device)))
				for img, age in zip(x, input_ages)]

	def _is_interpolation(self, target_ages):
		if self.train_min_age is None or self.train_max_age is None:
			return False
		min_age = self.train_min_age / 100.0
		max_age = self.train_max_age / 100.0
		return bool(torch.all(target_ages >= min_age) and torch.all(target_ages <= max_age))

	def _nearest_neighbor_id_loss(self, y_hat, target_ages):
		"""Compute nearest-neighbor identity loss for each sample and average."""
		# Extract features for current outputs
		reconstructed_feats = self.id_loss.extract_feats(y_hat)  # [b, 512]
		max_sims = []
		for i in range(y_hat.shape[0]):
			closest_age = min(self.feats_dict.keys(), key=lambda a: abs(a - float(target_ages[i].item())))
			# gather candidates within ±0.03 around closest age bucket
			candidates = [feat_tuple[0] for k in self.feats_dict.keys() if abs(k - closest_age) <= 0.03 for feat_tuple in self.feats_dict[k]]
			ref = torch.stack([f.squeeze(0) if f.ndim > 2 else f.squeeze(0) for f in candidates]).to(self.device)
			sim = F.cosine_similarity(reconstructed_feats[i].unsqueeze(0), ref, dim=1)  # [num_refs]
			max_sims.append(torch.max(sim))
		max_sims = torch.stack(max_sims)
		return torch.mean(1 - max_sims)

	def train(self):
		self.net.train()
		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				x, y, *_ = batch
				x, y = x.to(self.device).float(), y.to(self.device).float()
				self.optimizer.zero_grad()

				input_ages = self.aging_loss.extract_ages(x) / 100.

				# perform no aging in 33% of the time
				no_aging = random.random() <= (1. / 3)
				if no_aging:
					x_input = self.__set_target_to_source(x=x, input_ages=input_ages)
				else:
					x_input = [self.age_transformer(img.cpu()).to(self.device) for img in x]

				x_input = torch.stack(x_input)
				target_ages = x_input[:, -1, 0, 0]

				# perform forward/backward pass on real images
				y_hat, latent = self.perform_forward_pass(x_input)
				loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent,
														  target_ages=target_ages,
														  input_ages=input_ages,
														  no_aging=no_aging,
														  data_type="real")
				loss.backward()

				# perform cycle on generate images by setting the target ages to the original input ages
				y_hat_clone = y_hat.clone().detach().requires_grad_(True)
				input_ages_clone = input_ages.clone().detach().requires_grad_(True)
				y_hat_inverse = self.__set_target_to_source(x=y_hat_clone, input_ages=input_ages_clone)
				y_hat_inverse = torch.stack(y_hat_inverse)
				reverse_target_ages = y_hat_inverse[:, -1, 0, 0]
				y_recovered, latent_cycle = self.perform_forward_pass(y_hat_inverse)
				loss, cycle_loss_dict, cycle_id_logs = self.calc_loss(x, y, y_recovered, latent_cycle,
														  target_ages=reverse_target_ages,
														  input_ages=input_ages,
														  no_aging=no_aging,
														  data_type="cycle")
				loss.backward()
				# gradient clipping + NaN guard + scheduler
				# clip only the parameters that are actually being optimized
				trainable_params = []
				for group in self.optimizer.param_groups:
					for p in group.get('params', []):
						trainable_params.append(p)
				self._clip_and_step(self.optimizer, trainable_params)
				if self.scheduler is not None:
					self.scheduler.step()

				# combine the logs of both forwards
				for idx, cycle_log in enumerate(cycle_id_logs):
					id_logs[idx].update(cycle_log)
				loss_dict.update(cycle_loss_dict)
				loss_dict["loss"] = loss_dict["loss_real"] + loss_dict["loss_cycle"]

				# Logging related
				if self.global_step % self.opts.image_interval == 0 or \
						(self.global_step < 1000 and self.global_step % 25 == 0):
					self.parse_and_log_images(id_logs, x, y, y_hat, y_recovered,
											  title='images/train/faces')

				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')

				# Validation related
				val_loss_dict = None
				val_start_step = int(getattr(self.opts, 'val_start_step', 0) or 0)
				if (not getattr(self.opts, 'disable_validation', False)) and (self.global_step >= val_start_step) and (self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps):
					val_loss_dict = self.validate()
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1

	def validate(self):
		self.net.eval()
		# Snapshot RNG states so validation randomness does not influence training
		_py_state = random.getstate()
		_torch_state = torch.get_rng_state()
		_cuda_states = None
		try:
			_cuda_states = torch.cuda.get_rng_state_all()
		except Exception:
			pass
		# Optional deterministic validation
		deterministic = bool(getattr(self.opts, 'val_deterministic', False))
		if deterministic:
			seed = 12345
			random.seed(seed)
			torch.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)
		agg_loss_dict = []
		max_batches = int(getattr(self.opts, 'val_max_batches', 0) or 0)
		for batch_idx, batch in enumerate(self.test_dataloader):
			x, y, *_ = batch
			with torch.no_grad():
				x, y = x.to(self.device).float(), y.to(self.device).float()

				# Validation mode: optional reconstruction-only (no aging)
				if bool(getattr(self.opts, 'val_disable_aging', False)):
					input_ages = self.aging_loss.extract_ages(x) / 100.
					x_input = self.__set_target_to_source(x=x, input_ages=input_ages)
					no_aging = True
				else:
					input_ages = self.aging_loss.extract_ages(x) / 100.
					# no randomness if deterministic
					if deterministic:
						no_aging = True
						x_input = self.__set_target_to_source(x=x, input_ages=input_ages)
					else:
						no_aging = random.random() <= (1. / 3)
						if no_aging:
							x_input = self.__set_target_to_source(x=x, input_ages=input_ages)
						else:
							x_input = [self.age_transformer(img.cpu()).to(self.device) for img in x]

				x_input = torch.stack(x_input)
				target_ages = x_input[:, -1, 0, 0]

				# perform forward/backward pass on real images
				y_hat, latent = self.perform_forward_pass(x_input)
				_, cur_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent,
														   target_ages=target_ages,
														   input_ages=input_ages,
														   no_aging=no_aging,
														   data_type="real")

				# perform cycle on generate images by setting the target ages to the original input ages
				y_hat_inverse = self.__set_target_to_source(x=y_hat, input_ages=input_ages)
				y_hat_inverse = torch.stack(y_hat_inverse)
				reverse_target_ages = y_hat_inverse[:, -1, 0, 0]
				y_recovered, latent_cycle = self.perform_forward_pass(y_hat_inverse)
				loss, cycle_loss_dict, cycle_id_logs = self.calc_loss(x, y, y_recovered, latent_cycle,
														   target_ages=reverse_target_ages,
														   input_ages=input_ages,
														   no_aging=no_aging,
														   data_type="cycle")

				# combine the logs of both forwards
				for idx, cycle_log in enumerate(cycle_id_logs):
					id_logs[idx].update(cycle_log)
				cur_loss_dict.update(cycle_loss_dict)
				cur_loss_dict["loss"] = cur_loss_dict["loss_real"] + cur_loss_dict["loss_cycle"]

			agg_loss_dict.append(cur_loss_dict)

			# Logging related
			self.parse_and_log_images(id_logs, x, y, y_hat, y_recovered, title='images/test/faces',
									  subscript='{:04d}'.format(batch_idx))

			# Early-exit caps for validation cost
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch
			if max_batches > 0 and (batch_idx + 1) >= max_batches:
				break

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		self.net.train()
		# Restore RNG states
		random.setstate(_py_state)
		torch.set_rng_state(_torch_state)
		try:
			if _cuda_states is not None:
				torch.cuda.set_rng_state_all(_cuda_states)
		except Exception:
			pass
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write('**Best**: Step - {}, '
						'Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
			else:
				f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

	def configure_optimizers(self):
		# Respect train flags and set requires_grad accordingly
		train_encoder = bool(getattr(self.opts, 'train_encoder', False))
		train_decoder = bool(getattr(self.opts, 'train_decoder', False))

		encoder_params = list(self.net.encoder.parameters())
		decoder_params = list(self.net.decoder.parameters()) if hasattr(self.net, 'decoder') else []

		for p in encoder_params:
			p.requires_grad = train_encoder
		for p in decoder_params:
			p.requires_grad = train_decoder

		params = []
		if train_encoder:
			params += encoder_params
		if train_decoder:
			params += decoder_params

		# Fallback: if neither flag is set, default to training encoder to avoid empty optimizer
		if len(params) == 0:
			for p in encoder_params:
				p.requires_grad = True
			params = encoder_params

		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def _build_scheduler(self, optimizer):
		if optimizer is None:
			return None
		scheduler_type = getattr(self.opts, 'scheduler_type', 'none')
		if scheduler_type is None or scheduler_type.lower() == 'none':
			return None
		if scheduler_type.lower() != 'cosine':
			return None
		warmup_steps = max(0, int(getattr(self.opts, 'warmup_steps', 0)))
		min_lr = float(getattr(self.opts, 'min_lr', 0.0))
		max_steps = int(getattr(self.opts, 'max_steps', 0))
		for group in optimizer.param_groups:
			group.setdefault('initial_lr', group.get('lr', 1e-8))
		def lr_lambda(step):
			if max_steps <= 0:
				return 1.0
			if warmup_steps > 0 and step < warmup_steps:
				return float(step + 1) / float(warmup_steps)
			progress = min(1.0, float(step - warmup_steps) / max(1.0, float(max_steps - warmup_steps)))
			cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
			base_lr = optimizer.param_groups[0].get('initial_lr', optimizer.param_groups[0].get('lr', 1e-8))
			min_ratio = min_lr / max(base_lr, 1e-12)
			min_ratio = max(0.0, min(1.0, min_ratio))
			return min_ratio + (1.0 - min_ratio) * cosine
		return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=max(-1, int(self.global_step) - 1))

	def _clip_and_step(self, optimizer, params_iter):
		max_norm = float(getattr(self.opts, 'grad_clip_norm', 0.0) or 0.0)
		params = [p for p in params_iter if p.requires_grad and p.grad is not None]
		grad_norm = None
		if len(params) == 0:
			optimizer.step()
			optimizer.zero_grad(set_to_none=True)
			return True
		if max_norm > 0.0:
			grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
		else:
			total = 0.0
			for p in params:
				param_norm = p.grad.data.norm(2)
				total += param_norm.item() ** 2
			grad_norm = math.sqrt(total)
		nan_guard = bool(getattr(self.opts, 'nan_guard', False))
		if nan_guard and (not torch.isfinite(torch.as_tensor(grad_norm))):
			print(f"Warning: grad norm is not finite ({grad_norm}); skipping step at {self.global_step}")
			optimizer.zero_grad(set_to_none=True)
			return False
		optimizer.step()
		optimizer.zero_grad(set_to_none=True)
		return True

	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
		print(f'Loading dataset for {self.opts.dataset_type}')
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
									  target_root=dataset_args['train_target_root'],
									  source_transform=transforms_dict['transform_source'],
									  target_transform=transforms_dict['transform_gt_train'],
									  opts=self.opts)
		test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
									 target_root=dataset_args['test_target_root'],
									 source_transform=transforms_dict['transform_source'],
									 target_transform=transforms_dict['transform_test'],
									 opts=self.opts)
		print(f"Number of training samples: {len(train_dataset)}")
		print(f"Number of test samples: {len(test_dataset)}")
		return train_dataset, test_dataset

	def calc_loss(self, x, y, y_hat, latent, target_ages, input_ages, no_aging, data_type="real"):
		loss_dict = {}
		id_logs = []
		loss = 0.0
		if self.opts.id_lambda > 0:
			weights = None
			if self.opts.use_weighted_id_loss:  # compute weighted id loss only on forward pass
				age_diffs = torch.abs(target_ages - input_ages)
				weights = train_utils.compute_cosine_weights(x=age_diffs)
			loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x, label=data_type, weights=weights)
			loss_dict[f'loss_id_{data_type}'] = float(loss_id)
			loss_dict[f'id_improve_{data_type}'] = float(sim_improvement)
			loss = loss_id * self.opts.id_lambda
		if self.opts.l2_lambda > 0:
			loss_l2 = F.mse_loss(y_hat, y)
			loss_dict[f'loss_l2_{data_type}'] = float(loss_l2)
			if data_type == "real" and not no_aging:
				l2_lambda = self.opts.l2_lambda_aging
			else:
				l2_lambda = self.opts.l2_lambda
			loss += loss_l2 * l2_lambda
		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(y_hat, y)
			loss_dict[f'loss_lpips_{data_type}'] = float(loss_lpips)
			if data_type == "real" and not no_aging:
				lpips_lambda = self.opts.lpips_lambda_aging
			else:
				lpips_lambda = self.opts.lpips_lambda
			loss += loss_lpips * lpips_lambda
		if self.opts.lpips_lambda_crop > 0:
			loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
			loss += loss_lpips_crop * self.opts.lpips_lambda_crop
		if self.opts.l2_lambda_crop > 0:
			loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_l2_crop'] = float(loss_l2_crop)
			loss += loss_l2_crop * self.opts.l2_lambda_crop
		# decoder-phase scaled regularizers
		decoder_phase = bool(getattr(self.opts, 'train_decoder', False)) and not bool(getattr(self.opts, 'train_encoder', False))
		effective_w_norm_lambda = float(getattr(self.opts, 'w_norm_lambda', 0.0) or 0.0)
		effective_aging_lambda = float(getattr(self.opts, 'aging_lambda', 0.0) or 0.0)
		if decoder_phase:
			effective_w_norm_lambda *= float(getattr(self.opts, 'w_norm_lambda_decoder_scale', 1.0))
			effective_aging_lambda *= float(getattr(self.opts, 'aging_lambda_decoder_scale', 1.0))
		if effective_w_norm_lambda > 0:
			loss_w_norm = self.w_norm_loss(latent, latent_avg=self.net.latent_avg)
			loss_dict[f'loss_w_norm_{data_type}'] = float(loss_w_norm)
			loss += loss_w_norm * effective_w_norm_lambda
		if effective_aging_lambda > 0:
			aging_loss, id_logs = self.aging_loss(y_hat, y, target_ages, id_logs, label=data_type)
			loss_dict[f'loss_aging_{data_type}'] = float(aging_loss)
			loss += aging_loss * effective_aging_lambda
		# Impostor-only age-aware contrastive ID loss (real pass only)
		if (data_type == "real"):
			apply_mask = None
			try:
				if (self.mb is not None or self.miner is not None) and self.contrastive_id_lambda > 0:
					# Convert target ages from [0,1] to integer years [0,100]
					target_age_years = torch.clamp((target_ages * 100.0).round().to(torch.int64), 0, 200)
					apply_mask = torch.ones(y_hat.size(0), dtype=torch.bool, device=y_hat.device)
					if self.mb_apply_min_age is not None:
						apply_mask &= (target_age_years.to(y_hat.device) >= int(self.mb_apply_min_age))
					if self.mb_apply_max_age is not None:
						apply_mask &= (target_age_years.to(y_hat.device) <= int(self.mb_apply_max_age))
					loss_contrast = torch.tensor(0.0, device=y_hat.device)
					if apply_mask.any():
						# Sample negatives for active subset on CPU (avoid cross-device mask indexing)
						ages_cpu = target_age_years[apply_mask].detach().to("cpu")
						# Compute query embeddings with gradients flowing to generator
						q = self.id_loss.extract_feats(y_hat[apply_mask])  # [b',512]
						q = F.normalize(q, dim=1)
						if (self.miner is not None) and self.mb_use_faiss:
							negs, sims_sel = self.miner.query(
								q.detach(), ages_cpu, k=self.mb_k,
								min_sim=self.mb_min_sim, max_sim=self.mb_max_sim,
								top_m=self.mb_top_m, radius=self.mb_bin_neighbor_radius,
								device=y_hat.device
							)
							self.logger.add_scalar("train/mb_neg_sim_mean", float(sims_sel.mean().item()), self.global_step)
							# kthvalue expects k>=1; approximate p90 via sorted index
							flat = sims_sel.flatten()
							kth = max(1, int(0.9 * flat.numel()))
							values, _ = torch.topk(flat, kth)
							p90 = float(values.min().item()) if values.numel() > 0 else 0.0
							self.logger.add_scalar("train/mb_neg_sim_p90", p90, self.global_step)
						else:
							negs = self.mb.sample(ages_cpu, k=self.mb_k, radius=self.mb_bin_neighbor_radius)
						negs = negs.to(y_hat.device, non_blocking=True)
						negs = F.normalize(negs, dim=2)
						# Cosine similarities and log-sum-exp repel
						sims = torch.einsum("bd,bkd->bk", q, negs) / float(self.mb_temperature)
						loss_contrast_active = torch.logsumexp(sims, dim=1).mean()
						loss_contrast = loss_contrast + loss_contrast_active
						loss = loss + self.contrastive_id_lambda * loss_contrast
						# TensorBoard scalars + loss_dict entries for console prints
						self.logger.add_scalar("train/loss_contrastive_id", float(loss_contrast_active.item()), self.global_step)
						ratio = float(apply_mask.float().mean().item())
						self.logger.add_scalar("train/mb_applied_ratio", ratio, self.global_step)
						loss_dict['loss_contrastive_id'] = float(loss_contrast_active.item())
						loss_dict['mb_applied_ratio'] = ratio
					else:
						# No active samples; still log zeros for clarity
						self.logger.add_scalar("train/loss_contrastive_id", 0.0, self.global_step)
						self.logger.add_scalar("train/mb_applied_ratio", 0.0, self.global_step)
						loss_dict['loss_contrastive_id'] = 0.0
						loss_dict['mb_applied_ratio'] = 0.0
				else:
					# Bank disabled; log zeros
					self.logger.add_scalar("train/loss_contrastive_id", 0.0, self.global_step)
					# If we can compute mask without bank, base on range only
					try:
						target_age_years = torch.clamp((target_ages * 100.0).round().to(torch.int64), 0, 200)
						apply_mask = torch.ones(y_hat.size(0), dtype=torch.bool, device=y_hat.device)
						if self.mb_apply_min_age is not None:
							apply_mask &= (target_age_years.to(y_hat.device) >= int(self.mb_apply_min_age))
						if self.mb_apply_max_age is not None:
							apply_mask &= (target_age_years.to(y_hat.device) <= int(self.mb_apply_max_age))
						ratio = float(apply_mask.float().mean().item())
						self.logger.add_scalar("train/mb_applied_ratio", ratio, self.global_step)
						loss_dict['loss_contrastive_id'] = 0.0
						loss_dict['mb_applied_ratio'] = ratio
					except Exception:
						self.logger.add_scalar("train/mb_applied_ratio", 0.0, self.global_step)
						loss_dict['loss_contrastive_id'] = 0.0
						loss_dict['mb_applied_ratio'] = 0.0
			except Exception as e:
				# Robust to any runtime issues; do not break training
				try:
					print(f"Warning: contrastive impostor loss error at step {self.global_step}: {e}")
				except Exception:
					pass
				self.logger.add_scalar("train/loss_contrastive_id", 0.0, self.global_step)
				self.logger.add_scalar("train/mb_applied_ratio", 0.0, self.global_step)
				loss_dict['loss_contrastive_id'] = 0.0
				loss_dict['mb_applied_ratio'] = 0.0
		# Nearest-neighbor identity regularizer during interpolation only
		nn_lambda = float(getattr(self.opts, 'nearest_neighbor_id_loss_lambda', 0.0) or 0.0)
		if (nn_lambda > 0) and (not no_aging) and self._is_interpolation(target_ages) and (self.feats_dict is not None):
			nearest_neighbor_id_loss = self._nearest_neighbor_id_loss(y_hat, target_ages)
			loss_dict['nearest_neighbor_id_loss'] = float(nearest_neighbor_id_loss)
			loss += nearest_neighbor_id_loss * nn_lambda
		loss_dict[f'loss_{data_type}'] = float(loss)
		if data_type == "cycle":
			loss = loss * self.opts.cycle_lambda
		# Ensure tensor scalar return for backward safety even if no losses active
		if not torch.is_tensor(loss):
			loss = torch.zeros((), device=y_hat.device, requires_grad=True)
		return loss, loss_dict, id_logs

	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print(f'Metrics for {prefix}, step {self.global_step}')
		for key, value in metrics_dict.items():
			print(f'\t{key} = ', value)

	def parse_and_log_images(self, id_logs, x, y, y_hat, y_recovered, title, subscript=None, display_count=2):
		im_data = []
		# Cap display count to the smallest available across tensors and logs
		max_i = min(display_count, int(x.size(0)), int(y.size(0)), int(y_hat.size(0)), int(y_recovered.size(0)))
		for i in range(max_i):
			cur_im_data = {
				'input_face': common.tensor2im(x[i]),
				'target_face': common.tensor2im(y[i]),
				'output_face': common.tensor2im(y_hat[i]),
				'recovered_face': common.tensor2im(y_recovered[i])
			}
			if id_logs is not None and i < len(id_logs):
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
		if len(im_data) > 0:
			self.log_images(title, im_data=im_data, subscript=subscript)

	def log_images(self, name, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
		else:
			path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def load_checkpoint(self, checkpoint_path):
		"""Load checkpoint and resume training"""
		print(f"Loading checkpoint from: {checkpoint_path}")
		checkpoint = torch.load(checkpoint_path, map_location=self.device)
		
		# Load model state
		self.net.load_state_dict(checkpoint['state_dict'])
		
		# Load latent average if available
		if 'latent_avg' in checkpoint:
			self.net.latent_avg = checkpoint['latent_avg']
		
		# Load optimizer state if available
		if 'optimizer' in checkpoint:
			try:
				self.optimizer.load_state_dict(checkpoint['optimizer'])
				print("Optimizer state loaded successfully")
			except Exception as e:
				print(f"Warning: Failed to load optimizer state ({e}). Reinitializing optimizer.")
		else:
			print("Warning: No optimizer state found in checkpoint. Starting with fresh optimizer.")
		
		# Load global step from checkpoint if available, otherwise extract from filename
		if 'global_step' in checkpoint:
			self.global_step = checkpoint['global_step']
			print(f"Resuming from step: {self.global_step}")
		else:
			# Fallback: Extract global step from checkpoint filename if available
			checkpoint_name = os.path.basename(checkpoint_path)
			if 'iteration_' in checkpoint_name:
				step_str = checkpoint_name.replace('iteration_', '').replace('.pt', '')
				try:
					self.global_step = int(step_str)
					print(f"Resuming from step: {self.global_step}")
				except ValueError:
					print("Could not extract step from checkpoint name, starting from 0")
					self.global_step = 0
			else:
				print("Could not determine step from checkpoint name, starting from 0")
				self.global_step = 0
		
		print(f"Checkpoint loaded successfully. Resuming from step {self.global_step}")

	def load_checkpoint_legacy(self, checkpoint_path):
		"""Load legacy checkpoint that doesn't have optimizer state"""
		print(f"Loading legacy checkpoint from: {checkpoint_path}")
		checkpoint = torch.load(checkpoint_path, map_location=self.device)
		
		# Load model state
		self.net.load_state_dict(checkpoint['state_dict'])
		
		# Load latent average if available
		if 'latent_avg' in checkpoint:
			self.net.latent_avg = checkpoint['latent_avg']
		
		# Extract global step from checkpoint filename
		checkpoint_name = os.path.basename(checkpoint_path)
		if 'iteration_' in checkpoint_name:
			step_str = checkpoint_name.replace('iteration_', '').replace('.pt', '')
			try:
				self.global_step = int(step_str)
				print(f"Resuming from step: {self.global_step}")
			except ValueError:
				print("Could not extract step from checkpoint name, starting from 0")
				self.global_step = 0
		else:
			print("Could not determine step from checkpoint name, starting from 0")
			self.global_step = 0
		
		# For legacy checkpoints, we need to warm up the optimizer
		# Run a few dummy forward passes to initialize optimizer state properly
		print("Warming up optimizer with dummy forward passes...")
		self.net.train()
		dummy_input = torch.randn(1, 4, 256, 256).to(self.device)
		
		for i in range(10):  # Warm up with 10 dummy passes
			self.optimizer.zero_grad()
			with torch.no_grad():
				_, _ = self.net.forward(dummy_input, return_latents=True)
			# Don't actually step the optimizer, just initialize its state
		
		print(f"Legacy checkpoint loaded successfully. Resuming from step {self.global_step}")

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'global_step': self.global_step,
			'opts': vars(self.opts)
		}
		# save the latent avg in state_dict for inference if truncation of w was used during training
		if self.net.latent_avg is not None:
			save_dict['latent_avg'] = self.net.latent_avg
		return save_dict


