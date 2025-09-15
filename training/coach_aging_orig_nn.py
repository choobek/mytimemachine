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
from training.losses.anchor_loss import AgeAnchorLoss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from datasets.augmentations import AgeTransformer
from criteria.lpips.lpips import LPIPS
from criteria.aging_loss import AgingLoss
from models.psp import pSp
from training.ranger import Ranger
import math
import copy
from models.binary_identity_model import build_identity_model


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

		# ID/Aging lambda scheduling (Stage-aware)
		self.id_lambda_base = float(getattr(self.opts, 'id_lambda', 0.0) or 0.0)
		self.aging_lambda_base = float(getattr(self.opts, 'aging_lambda', 0.0) or 0.0)
		self.cur_id_lambda = float(self.id_lambda_base)
		self.cur_aging_lambda = float(self.aging_lambda_base)
		self.id_lambda_s2 = getattr(self.opts, 'id_lambda_s2', None)
		self.aging_lambda_s2 = getattr(self.opts, 'aging_lambda_s2', None)
		try:
			from training.utils.schedules import parse_step_schedule
			self.id_s1_schedule = parse_step_schedule(getattr(self.opts, 'id_lambda_schedule_s1', None))
			self.aging_s1_schedule = parse_step_schedule(getattr(self.opts, 'aging_lambda_schedule_s1', None))
		except Exception:
			self.id_s1_schedule = []
			self.aging_s1_schedule = []

		# Target-age ID guidance (Task 3)
		self.target_id_bank = None
		self.target_id_apply_min_age = int(getattr(self.opts, 'target_id_apply_min_age', 38) or 38)
		self.target_id_apply_max_age = int(getattr(self.opts, 'target_id_apply_max_age', 42) or 42)
		self.target_id_lambda_s1 = float(getattr(self.opts, 'target_id_lambda_s1', 0.10) or 0.10)
		self.target_id_lambda_s2 = float(getattr(self.opts, 'target_id_lambda_s2', 0.05) or 0.05)
		bank_path = str(getattr(self.opts, 'target_id_bank_path', '') or '')
		if len(bank_path) > 0 and os.path.exists(bank_path):
			try:
				bank_obj = torch.load(bank_path, map_location='cpu')
				protos = bank_obj.get('global_protos', None)
				if isinstance(protos, dict) and len(protos) > 0:
					self.target_id_bank = {int(k): v.to(self.device).float() for k, v in protos.items()}
					print(f"[target-id] Loaded {len(self.target_id_bank)} age prototypes from {bank_path}")
				else:
					print(f"[target-id] No global_protos found in {bank_path}; feature OFF")
			except Exception as e:
				print(f"[target-id] Failed to load target-id bank: {e}; feature OFF")

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

		# Age sampling: support fixed age with optional jitter if requested
		if getattr(self.opts, 'target_age_fixed', None) is not None:
			clip_min = getattr(self, 'train_min_age', getattr(self, 'aging_loss', None).min_age if hasattr(self, 'aging_loss') else 0)
			clip_max = getattr(self, 'train_max_age', getattr(self, 'aging_loss', None).max_age if hasattr(self, 'aging_loss') else 100)
			self.age_transformer = AgeTransformer(
				target_age=int(self.opts.target_age_fixed),
				jitter=int(getattr(self.opts, 'target_age_jitter', 0) or 0),
				clip_bounds=(clip_min, clip_max)
			)
		else:
			self.age_transformer = AgeTransformer(target_age=self.opts.target_age)

		# Age-anchor configuration (Task 4)
		self.anchor_lambda = float(getattr(self.opts, 'age_anchor_lambda', 0.0) or 0.0)
		self.anchor_stage = str(getattr(self.opts, 'age_anchor_stage', 's1') or 's1')
		self.anchor_space = str(getattr(self.opts, 'age_anchor_space', 'w') or 'w')
		self.anchor_bin_size = int(getattr(self.opts, 'age_anchor_bin_size', 5) or 5)
		self.anchor_loss = None
		self.anchor_bins_count = 0
		self.anchor_bin_mids = []
		self.anchor_enabled = False
		anchor_path = str(getattr(self.opts, 'age_anchor_path', '') or '')
		if (self.anchor_lambda > 0.0) and (len(anchor_path) > 0):
			try:
				anchors_obj = torch.load(anchor_path, map_location='cpu')
				space = anchors_obj.get('space', 'w')
				if space != self.anchor_space:
					print(f"[anchor] Space mismatch: file space={space} vs flag={self.anchor_space}. Disabling.")
				else:
					# Support both legacy tensor format and new dict{age:int -> Tensor}
					file_bin = anchors_obj.get('bin_size', None)
					if file_bin is None:
						# Try nested range.bin_size
						try:
							file_bin = int(anchors_obj.get('range', {}).get('bin_size', self.anchor_bin_size))
						except Exception:
							file_bin = self.anchor_bin_size
					else:
						file_bin = int(file_bin)
					if file_bin != self.anchor_bin_size:
						print(f"[anchor] Bin size mismatch: file {file_bin} vs flag {self.anchor_bin_size}. Using file value.")
						self.anchor_bin_size = file_bin
					anchors_w = anchors_obj.get('anchors', None)
					bin_mids = anchors_obj.get('bin_mids', [])
					# If saved as dict {age:int -> tensor}, convert to stacked tensor + sorted mids
					if isinstance(anchors_w, dict) and len(anchors_w) > 0:
						ages_sorted = sorted(int(a) for a in anchors_w.keys())
						stack_list = []
						for a in ages_sorted:
							v = anchors_w[int(a)]
							if isinstance(v, torch.Tensor):
								# Support W+ tensors by averaging over styles
								if v.ndim == 2 and v.size(1) == 512:
									w = v.mean(dim=0)
								elif v.ndim == 1 and v.numel() == 512:
									w = v
								else:
									continue
								stack_list.append(w.float().unsqueeze(0))
						anchors_w = torch.cat(stack_list, dim=0) if len(stack_list) > 0 else None
						bin_mids = ages_sorted
					if isinstance(anchors_w, torch.Tensor) and anchors_w.ndim == 2 and anchors_w.size(1) == 512 and len(bin_mids) == anchors_w.size(0):
						self.anchor_loss = AgeAnchorLoss(anchors_w.to(self.device), bin_mids=bin_mids, bin_size=self.anchor_bin_size)
						self.anchor_bins_count = int(anchors_w.size(0))
						self.anchor_bin_mids = list(bin_mids)
						self.anchor_enabled = True
						self.logger.add_text("setup/anchor", f"loaded bins={self.anchor_bins_count} size={self.anchor_bin_size} λ={self.anchor_lambda}", self.global_step)
						try:
							with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
								f.write(f"anchor: bins={self.anchor_bins_count} bin_size={self.anchor_bin_size} space={self.anchor_space} λ={self.anchor_lambda}\n")
						except Exception:
							pass
					else:
						print("[anchor] Invalid anchors content; disabling.")
			except Exception as e:
				print(f"[anchor] Could not load anchors from {anchor_path}: {e}. Feature OFF.")
		else:
			if self.anchor_lambda > 0.0:
				print("[anchor] age_anchor_lambda > 0 but no valid --age_anchor_path provided. Feature OFF.")

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
		
		# Identity-adversarial configuration (schedule + flags)
		try:
			from training.utils.schedules import parse_step_schedule
			self.id_adv_s1_schedule = parse_step_schedule(getattr(self.opts, 'id_adv_schedule_s1', None))
		except Exception:
			self.id_adv_s1_schedule = []
		self.id_adv_enabled = bool(getattr(self.opts, 'id_adv_enabled', False))
		self.id_adv_gamma = float(getattr(self.opts, 'id_adv_focal_gamma', 0.0) or 0.0)
		self.id_adv_margin = float(getattr(self.opts, 'id_adv_margin', 0.0) or 0.0)
		self.id_adv_tta = [t.strip() for t in str(getattr(self.opts, 'id_adv_tta', 'clean') or 'clean').split(',') if len(t.strip()) > 0]
		self.id_adv_agg = str(getattr(self.opts, 'id_adv_agg', 'mean(clean)') or 'mean(clean)')
		from training.losses_idadv import parse_conf_weight_spec as _parse_cw
		self.id_adv_conf = _parse_cw(getattr(self.opts, 'id_adv_conf_weight', ''))

		# Initialize identity-adversarial discriminator (frozen)
		self.id_adv = None
		self.id_adv_preproc = None
		self._init_id_adv()

		# (moved) Load checkpoint happens after EMA helpers init so EMA can be restored
		# Symlink convenience for latest phase3 directory
		try:
			latest = os.path.join('experiments', '_latest_phase3')
			run_abs = os.path.realpath(self.opts.exp_dir)
			os.makedirs(os.path.dirname(latest), exist_ok=True)
			# Force-update symlink
			if os.path.islink(latest) or os.path.exists(latest):
				try:
					os.remove(latest)
				except Exception:
					pass
			os.symlink(run_abs, latest)
		except Exception:
			pass

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
		self.mb_profile = str(getattr(self.opts, 'mb_profile', 'custom') or 'custom')
		self.last_miner_meta = None
		# Setup breadcrumbs for miner params
		try:
			self.logger.add_text("setup/mb_params",
				f"profile={self.mb_profile}, k={self.mb_k}, top_m={self.mb_top_m}, min_sim={self.mb_min_sim}, max_sim={self.mb_max_sim}, temp={self.mb_temperature}, age=[{self.mb_apply_min_age},{self.mb_apply_max_age}], use_faiss={self.mb_use_faiss}",
				self.global_step)
		except Exception:
			pass

		# Startup breadcrumb: phase-3 fixed-age + miner window + lr + resume
		try:
			fixed = getattr(self.opts, 'target_age_fixed', None)
			jitter = int(getattr(self.opts, 'target_age_jitter', 0) or 0)
			lo = getattr(self.opts, 'mb_apply_min_age', None)
			hi = getattr(self.opts, 'mb_apply_max_age', None)
			lr = float(getattr(self.opts, 'learning_rate', 0.0) or 0.0)
			resume = str(getattr(self.opts, 'resume_checkpoint', '') or '')
			with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
				f.write(f"phase3: fixed_age={fixed} jitter={jitter} miner_window=[{lo},{hi}] lr={lr} resume={resume}\n")
		except Exception:
			pass

		# ROI-ID micro-loss configuration
		self.roi_id_lambda = float(getattr(self.opts, 'roi_id_lambda', 0.0) or 0.0)
		self.roi_size = int(getattr(self.opts, 'roi_size', 112) or 112)
		self.roi_pad = float(getattr(self.opts, 'roi_pad', 0.35) or 0.35)
		self.roi_jitter = float(getattr(self.opts, 'roi_jitter', 0.08) or 0.08)
		self.roi_use_eyes = bool(getattr(self.opts, 'roi_use_eyes', False))
		self.roi_use_mouth = bool(getattr(self.opts, 'roi_use_mouth', False))
		self.roi_use_nose = bool(getattr(self.opts, 'roi_use_nose', False))
		self.roi_use_broweyes = bool(getattr(self.opts, 'roi_use_broweyes', False))
		self.roi_cropper = None
		if self.roi_id_lambda > 0 and (self.roi_use_eyes or self.roi_use_mouth or self.roi_use_nose or self.roi_use_broweyes):
			try:
				from training.roi_crops import LandmarkCropper
				self.roi_cropper = LandmarkCropper(getattr(self.opts, 'roi_landmarks_model', ''))
			except Exception:
				self.roi_cropper = None
			# Ensure ID encoder is available for ROI-ID even if other ID losses are off
			if not hasattr(self, 'id_loss'):
				self.id_loss = id_loss.IDLoss().to(self.device).eval()

		# ROI schedule configuration
		self.cur_roi_lambda = float(self.roi_id_lambda)
		try:
			from training.utils.schedules import parse_step_schedule
			self.roi_s1_schedule = parse_step_schedule(getattr(self.opts, 'roi_id_schedule_s1', None))
		except Exception:
			self.roi_s1_schedule = []

		# --- Phase-3 acceptance tracking ---
		# Target age lock bounds for assertions/logging
		try:
			fixed_age = getattr(self.opts, 'target_age_fixed', None)
			if fixed_age is not None:
				jit = int(getattr(self.opts, 'target_age_jitter', 0) or 0)
				tight = max(2, jit)
				self._age_lock_lo = max(int(fixed_age) - tight, 0)
				self._age_lock_hi = int(fixed_age) + tight
			else:
				self._age_lock_lo = None
				self._age_lock_hi = None
		except Exception:
			self._age_lock_lo = None
			self._age_lock_hi = None
		# Identity trend trackers
		self._id_loss_window = collections.deque()
		self._best_train_id_loss = float('inf')
		self._best_id_improve_real = float('-inf')
		self._last_best_train_id_step = 0
		self._last_best_val_step = 0
		self.roi_lambda_base = float(getattr(self.opts, 'roi_id_lambda', 0.0) or 0.0)
		self.roi_lambda_s2 = getattr(self.opts, 'roi_id_lambda_s2', None)

		# Geometry loss configuration (scale-invariant shape ratios)
		self.geom_lambda = float(getattr(self.opts, 'geom_lambda', 0.0) or 0.0)
		self.geom_stage = str(getattr(self.opts, 'geom_stage', 's1') or 's1')
		# parts and weights parsing
		self.geom_parts = tuple([s.strip() for s in str(getattr(self.opts, 'geom_parts', 'eyes,nose,mouth')).split(',') if len(s.strip()) > 0])
		try:
			w_str = str(getattr(self.opts, 'geom_weights', '1.0,0.6,0.4'))
			w_vals = [float(x) for x in w_str.split(',')]
			while len(w_vals) < 3:
				w_vals.append(0.0)
			self.geom_weights = (w_vals[0], w_vals[1], w_vals[2])
		except Exception:
			self.geom_weights = (1.0, 0.6, 0.4)
		self.geom_huber_delta = float(getattr(self.opts, 'geom_huber_delta', 0.03) or 0.03)
		self.geom_norm = str(getattr(self.opts, 'geom_norm', 'interocular') or 'interocular')
		self.geom_landmarks_model = str(getattr(self.opts, 'geom_landmarks_model', '') or '')
		self.geom_enabled = self.geom_lambda > 0.0
		self.geom_loss = None
		self.geom_cropper = None
		if self.geom_enabled:
			try:
				from training.losses.geometry_loss import GeometryLoss
				self.geom_loss = GeometryLoss(parts=self.geom_parts, weights=self.geom_weights, delta=self.geom_huber_delta)
			except Exception:
				self.geom_loss = None
			try:
				from training.roi_crops import LandmarkCropper
				# Prefer reusing ROI landmark model path if geometry-specific not provided
				lm_path = self.geom_landmarks_model or getattr(self.opts, 'roi_landmarks_model', '')
				self.geom_cropper = LandmarkCropper(lm_path)
			except Exception:
				self.geom_cropper = None

		# EMA configuration
		self.ema_enabled = bool(getattr(self.opts, 'ema', False))
		self.eval_with_ema = bool(getattr(self.opts, 'eval_with_ema', True)) and self.ema_enabled
		self.ema_decay = float(getattr(self.opts, 'ema_decay', 0.999) or 0.999)
		self.ema_scope = str(getattr(self.opts, 'ema_scope', 'decoder') or 'decoder')
		self.ema_helpers = {}
		if self.ema_enabled:
			from training.utils.ema import EMAHelper
			track_modules = {}
			if self.ema_scope == 'decoder':
				if hasattr(self.net, 'decoder'):
					track_modules['decoder'] = self.net.decoder
			elif self.ema_scope == 'decoder+adapter':
				if hasattr(self.net, 'decoder'):
					track_modules['decoder'] = self.net.decoder
				if hasattr(self.net, 'blender'):
					track_modules['blender'] = self.net.blender
			elif self.ema_scope == 'all':
				candidates = {}
				if hasattr(self.net, 'encoder'):
					candidates['encoder'] = self.net.encoder
				if hasattr(self.net, 'decoder'):
					candidates['decoder'] = self.net.decoder
				if hasattr(self.net, 'blender'):
					candidates['blender'] = self.net.blender
				for name, mod in candidates.items():
					any_trainable = any(p.requires_grad for p in mod.parameters(recurse=True))
					if any_trainable:
						track_modules[name] = mod
			for name, mod in track_modules.items():
				self.ema_helpers[name] = EMAHelper(mod, self.ema_decay)

		# Load checkpoint if resuming (after EMA helpers are ready)
		if hasattr(opts, 'resume_checkpoint') and opts.resume_checkpoint:
			self.load_checkpoint(opts.resume_checkpoint)

		# Build optional scheduler for stability
		self.scheduler = self._build_scheduler(self.optimizer)

	def _update_loss_lambdas_for_step(self):
		stage1 = bool(getattr(self.opts, 'train_encoder', False))
		stage2 = bool(getattr(self.opts, 'train_decoder', False)) and not stage1
		# ID lambda
		if stage1:
			try:
				from training.utils.schedules import value_for_step
				val = value_for_step(self.id_s1_schedule, int(self.global_step))
				self.cur_id_lambda = float(val) if val is not None else float(self.id_lambda_base)
			except Exception:
				self.cur_id_lambda = float(self.id_lambda_base)
		elif stage2:
			if self.id_lambda_s2 is not None:
				self.cur_id_lambda = float(self.id_lambda_s2)
			else:
				self.cur_id_lambda = float(self.id_lambda_base)
		else:
			self.cur_id_lambda = float(self.id_lambda_base)
		# Aging lambda
		if stage1:
			try:
				from training.utils.schedules import value_for_step
				val = value_for_step(self.aging_s1_schedule, int(self.global_step))
				self.cur_aging_lambda = float(val) if val is not None else float(self.aging_lambda_base)
			except Exception:
				self.cur_aging_lambda = float(self.aging_lambda_base)
		elif stage2:
			if self.aging_lambda_s2 is not None:
				self.cur_aging_lambda = float(self.aging_lambda_s2)
			else:
				self.cur_aging_lambda = float(self.aging_lambda_base)
		else:
			self.cur_aging_lambda = float(self.aging_lambda_base)

	def _update_roi_lambda_for_step(self):
		stage1 = bool(getattr(self.opts, 'train_encoder', False))
		stage2 = bool(getattr(self.opts, 'train_decoder', False)) and not stage1
		if stage1:
			try:
				from training.utils.schedules import value_for_step
				val = value_for_step(self.roi_s1_schedule, int(self.global_step))
				if val is not None:
					self.cur_roi_lambda = float(val)
				else:
					self.cur_roi_lambda = float(self.roi_lambda_base)
			except Exception:
				self.cur_roi_lambda = float(self.roi_lambda_base)
		elif stage2:
			if self.roi_lambda_s2 is not None:
				self.cur_roi_lambda = float(self.roi_lambda_s2)
			else:
				self.cur_roi_lambda = float(self.roi_lambda_base)
		else:
			self.cur_roi_lambda = float(self.roi_lambda_base)

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

				# Update ROI-ID lambda schedule each step
				self._update_roi_lambda_for_step()
				# Update ID/Aging lambdas schedule each step
				self._update_loss_lambdas_for_step()

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
				# Track train ID trends window to check non-increasing over 1500 steps
				try:
					if 'loss_id_real' in loss_dict:
						cur_id = float(loss_dict['loss_id_real'])
						self._best_train_id_loss = min(self._best_train_id_loss, cur_id)
						if cur_id <= self._best_train_id_loss + 1e-12:
							self._last_best_train_id_step = self.global_step
						self._id_loss_window.append((self.global_step, cur_id))
						while len(self._id_loss_window) > 0 and (self.global_step - self._id_loss_window[0][0]) > 1500:
							self._id_loss_window.popleft()
				except Exception:
					pass

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
				step_ok, grad_norm = self._clip_and_step(self.optimizer, trainable_params)
				if self.scheduler is not None:
					self.scheduler.step()
				# EMA update after optimizer step
				if self.ema_enabled and step_ok and len(self.ema_helpers) > 0:
					for name, helper in self.ema_helpers.items():
						mod = None
						if name == 'decoder' and hasattr(self.net, 'decoder'):
							mod = self.net.decoder
						elif name == 'blender' and hasattr(self.net, 'blender'):
							mod = self.net.blender
						elif name == 'encoder' and hasattr(self.net, 'encoder'):
							mod = self.net.encoder
						if mod is not None:
							helper.update(mod)
				# Log LR and grad diagnostics
				try:
					current_lr = float(self.optimizer.param_groups[0].get('lr', 0.0))
					self.logger.add_scalar("train/lr", current_lr, self.global_step)
				except Exception:
					current_lr = 0.0
				self.logger.add_scalar("train/grad_norm", float(grad_norm) if grad_norm is not None else 0.0, self.global_step)
				self.logger.add_scalar("train/step_skipped", 0 if step_ok else 1, self.global_step)

				# combine the logs of both forwards
				for idx, cycle_log in enumerate(cycle_id_logs):
					id_logs[idx].update(cycle_log)
				loss_dict.update(cycle_loss_dict)
				loss_dict["loss"] = loss_dict["loss_real"] + loss_dict["loss_cycle"]
				# Add Priority B metrics to loss_dict for timestamp visibility
				loss_dict['lr'] = float(current_lr)
				loss_dict['grad_norm'] = float(grad_norm) if grad_norm is not None else 0.0
				loss_dict['step_skipped'] = 0 if step_ok else 1
				flag = 1.0 if no_aging else 0.0
				self.logger.add_scalar("train/no_aging_flag", flag, self.global_step)
				loss_dict['no_aging_flag'] = flag
				loss_dict['no_aging_ratio'] = flag

				# Logging related
				if self.global_step % self.opts.image_interval == 0 or \
						(self.global_step < 1000 and self.global_step % 25 == 0):
					self.parse_and_log_images(id_logs, x, y, y_hat, y_recovered,
											  title='images/train/faces')

				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')
					# Diagnostics
					try:
						self.logger.add_scalar("train/roi_id_lambda_current", float(self.cur_roi_lambda), self.global_step)
						self.logger.add_scalar("train/id_lambda_current", float(self.cur_id_lambda), self.global_step)
						self.logger.add_scalar("train/aging_lambda_current", float(self.cur_aging_lambda), self.global_step)
						self.logger.add_scalar("train/ema_enabled", 1.0 if self.ema_enabled else 0.0, self.global_step)
						self.logger.add_scalar("train/ema_decay", float(self.ema_decay) if self.ema_enabled else 0.0, self.global_step)
					except Exception:
						pass

				# Validation related
				val_loss_dict = None
				val_start_step = int(getattr(self.opts, 'val_start_step', 0) or 0)
				if (not getattr(self.opts, 'disable_validation', False)) and (self.global_step >= val_start_step) and (self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps):
					used_ema = 0
					if self.ema_enabled and self.eval_with_ema and len(self.ema_helpers) > 0:
						orig_states = {}
						for name, helper in self.ema_helpers.items():
							mod = None
							if name == 'decoder' and hasattr(self.net, 'decoder'):
								mod = self.net.decoder
							elif name == 'blender' and hasattr(self.net, 'blender'):
								mod = self.net.blender
							elif name == 'encoder' and hasattr(self.net, 'encoder'):
								mod = self.net.encoder
							if mod is not None:
								orig_states[name] = copy.deepcopy(mod.state_dict())
								helper.copy_to(mod)
						used_ema = 1
					val_loss_dict = self.validate()
					try:
						if isinstance(val_loss_dict, dict):
							val_loss_dict['used_ema'] = float(used_ema)
					except Exception:
						pass
					if used_ema == 1:
						for name, state in orig_states.items():
							if name == 'decoder' and hasattr(self.net, 'decoder'):
								self.net.decoder.load_state_dict(state, strict=True)
							elif name == 'blender' and hasattr(self.net, 'blender'):
								self.net.blender.load_state_dict(state, strict=True)
							elif name == 'encoder' and hasattr(self.net, 'encoder'):
								self.net.encoder.load_state_dict(state, strict=True)
					self.logger.add_scalar("eval/used_ema", float(used_ema), self.global_step)
					# Append compact miner summary to timestamp.txt at each validation interval
					try:
						miner_line = None
						profile = getattr(self, 'mb_profile', 'custom')
						if hasattr(self, 'last_miner_meta') and (self.last_miner_meta is not None):
							m = self.last_miner_meta
							miner_line = f"mb: prof={profile} k={self.mb_k} band=[{self.mb_min_sim:.2f},{self.mb_max_sim:.2f}] cand≈{m.get('candidate_count', 0.0):.1f} simμ≈{m.get('sim_mean', 0.0):.3f} p75≈{m.get('sim_p75', 0.0):.3f} p90≈{m.get('sim_p90', 0.0):.3f}"
						else:
							miner_line = f"mb: prof={profile} k={self.mb_k} band=[{self.mb_min_sim:.2f},{self.mb_max_sim:.2f}]"
						with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
							# used_ema marker and optional losses for quick inspection
							line = f"used_ema={int(used_ema)}"
							try:
								if isinstance(val_loss_dict, dict):
									if 'loss_id_real' in val_loss_dict:
										line += f" id={float(val_loss_dict['loss_id_real']):.3f}"
									if 'loss_lpips_real' in val_loss_dict:
										line += f" lpips={float(val_loss_dict['loss_lpips_real']):.3f}"
									if 'loss_l2_real' in val_loss_dict:
										line += f" l2={float(val_loss_dict['loss_l2_real']):.3f}"
							except Exception:
								pass
							f.write(line + "\n")
							f.write(miner_line + "\n")
							# ID-ADV compact summary line (if metrics present)
							try:
								if isinstance(val_loss_dict, dict) and ('id_adv_lambda_current' in val_loss_dict):
									lam = float(val_loss_dict.get('id_adv_lambda_current', 0.0))
									p_mean = float(val_loss_dict.get('id_adv_p_actor_mean', 0.0))
									p_min = float(val_loss_dict.get('id_adv_p_actor_min_aug', 0.0))
									margin = float(val_loss_dict.get('id_adv_logit_margin', 0.0))
									gamma = float(getattr(self, 'id_adv_gamma', getattr(self.opts, 'id_adv_focal_gamma', 0.0)) or 0.0)
									m_h = float(getattr(self, 'id_adv_margin', getattr(self.opts, 'id_adv_margin', 0.0)) or 0.0)
									views = ','.join(list(getattr(self, 'id_adv_tta', getattr(self.opts, 'id_adv_tta', 'clean')).split(',')))
									expr = str(getattr(self, 'id_adv_agg', getattr(self.opts, 'id_adv_agg', 'mean(clean)')) or 'mean(clean)')
									import re as _re
									simple = _re.sub(r"mean\([^\)]*\)", "mean", expr)
									simple = _re.sub(r"min\([^\)]*\)", "min", simple)
									simple = simple.replace(" ", "")
									f.write(f"idadv: lam={lam:.3f} p={p_mean:.3f} p_min={p_min:.3f} margin={margin:.3f} focalγ={gamma} m={m_h} views={views} agg={simple}\n")
							except Exception:
								pass
						# Plateau hint: if total val loss hasn't improved in 1200 steps and loss_id_real hasn't improved in 800 steps
						try:
							no_val_improve = (self.global_step - getattr(self, '_last_best_val_step', 0)) >= 1200
							no_id_improve = (self.global_step - getattr(self, '_last_best_train_id_step', 0)) >= 800
							if no_val_improve and no_id_improve:
								with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
									f.write("PHASE3: plateau\n")
						except Exception:
							pass
					except Exception:
						pass
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)
						# Reset val improvement timer
						self._last_best_val_step = self.global_step
						# Also snapshot best-EMA weights for convenience if EMA used during eval
						try:
							if getattr(self, 'ema_enabled', False) and getattr(self, 'eval_with_ema', False):
								best_ema_path = os.path.join(self.checkpoint_dir, 'phase3_best_ema.pt')
								torch.save(self.__get_save_dict(), best_ema_path)
						except Exception:
							pass

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)
					# Save explicit last-weights snapshot for hygiene
					try:
						last_path = os.path.join(self.checkpoint_dir, 'phase3_last.pt')
						torch.save(self.__get_save_dict(), last_path)
					except Exception:
						pass

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				# Append breadcrumbs to loss_dict so timestamp.txt shows schedule & EMA info
				try:
					loss_dict['roi_lambda_current'] = float(self.cur_roi_lambda)
					loss_dict['id_lambda_current'] = float(self.cur_id_lambda)
					loss_dict['aging_lambda_current'] = float(self.cur_aging_lambda)
					loss_dict['ema'] = 'on' if self.ema_enabled else 'off'
					loss_dict['ema_scope'] = str(self.ema_scope)
					loss_dict['ema_decay'] = float(self.ema_decay) if self.ema_enabled else 0.0
					# Geometry config breadcrumbs
					loss_dict['geom_lambda'] = float(getattr(self, 'geom_lambda', 0.0))
					loss_dict['geom_stage'] = str(getattr(self, 'geom_stage', 's1'))
					loss_dict['geom_parts'] = ','.join(list(getattr(self, 'geom_parts', ('eyes','nose','mouth'))))
					loss_dict['geom_weights'] = ','.join([str(x) for x in list(getattr(self, 'geom_weights', (1.0,0.6,0.4)))])
					loss_dict['geom_huber_delta'] = float(getattr(self, 'geom_huber_delta', 0.03))
				except Exception:
					pass
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
		# Enrich loss_dict with EMA/ROI breadcrumbs for timestamp visibility
		try:
			loss_dict = dict(loss_dict) if isinstance(loss_dict, dict) else {'loss': float(loss_dict)}
			loss_dict['roi_lambda_current'] = float(getattr(self, 'cur_roi_lambda', getattr(self, 'roi_id_lambda', 0.0)))
			loss_dict['ema'] = 'on' if getattr(self, 'ema_enabled', False) else 'off'
			loss_dict['ema_scope'] = str(getattr(self, 'ema_scope', ''))
			loss_dict['ema_decay'] = float(getattr(self, 'ema_decay', 0.0)) if getattr(self, 'ema_enabled', False) else 0.0
		except Exception:
			pass
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
			return True, 0.0
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
			return False, float(grad_norm) if grad_norm is not None else float('nan')
		optimizer.step()
		optimizer.zero_grad(set_to_none=True)
		return True, float(grad_norm) if grad_norm is not None else 0.0

	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
		print(f'Loading dataset for {self.opts.dataset_type}')
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		train_root = dataset_args['train_source_root']
		if hasattr(self.opts, 'train_dataset') and self.opts.train_dataset and os.path.isdir(self.opts.train_dataset):
			train_root = self.opts.train_dataset
			dataset_args['train_target_root'] = train_root
			print(f"Overwritting training dataset to {train_root}")
		train_dataset = ImagesDataset(source_root=train_root,
									  target_root=dataset_args['train_target_root'],
									  source_transform=transforms_dict['transform_source'],
									  target_transform=transforms_dict['transform_gt_train'],
									  opts=self.opts)
		test_root = dataset_args['test_source_root']
		if hasattr(self.opts, 'test_dataset') and self.opts.test_dataset and os.path.isdir(self.opts.test_dataset):
			test_root = self.opts.test_dataset
			dataset_args['test_target_root'] = test_root
			print(f"Overwritting test dataset to {test_root}")
		test_dataset = ImagesDataset(source_root=test_root,
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
		if self.opts.id_lambda > 0 or getattr(self, 'cur_id_lambda', 0.0) > 0.0:
			weights = None
			if self.opts.use_weighted_id_loss:  # compute weighted id loss only on forward pass
				age_diffs = torch.abs(target_ages - input_ages)
				weights = train_utils.compute_cosine_weights(x=age_diffs)
			loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x, label=data_type, weights=weights)
			loss_dict[f'loss_id_{data_type}'] = float(loss_id)
			loss_dict[f'id_improve_{data_type}'] = float(sim_improvement)
			# Use scheduled ID lambda if available
			lambda_id = float(getattr(self, 'cur_id_lambda', getattr(self.opts, 'id_lambda', 0.0)))
			loss = loss_id * lambda_id
			# Optional margin-based identity hinge: encourage cos(y_hat,y) >= m
			if getattr(self.opts, 'id_margin_enabled', False) and float(getattr(self.opts, 'id_margin_lambda', 0.0) or 0.0) > 0:
				# Compute cosine similarities with gradients to y_hat
				y_feats = self.id_loss.extract_feats(y)
				y_feats = F.normalize(y_feats, dim=1)
				yhat_feats = self.id_loss.extract_feats(y_hat)
				yhat_feats = F.normalize(yhat_feats, dim=1)
				cos_sim = torch.sum(y_feats * yhat_feats, dim=1)
				m = float(getattr(self.opts, 'id_margin_target', 0.90) or 0.90)
				margin_loss = torch.clamp(m - cos_sim, min=0.0).mean()
				loss = loss + float(self.opts.id_margin_lambda) * margin_loss
				loss_dict[f'loss_id_margin_{data_type}'] = float(margin_loss)
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
		# Use scheduled aging lambda (with decoder-phase scaling handled below)
		effective_aging_lambda = float(getattr(self, 'cur_aging_lambda', getattr(self.opts, 'aging_lambda', 0.0)) or 0.0)
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
		# Target-age ID guidance (Task 3): apply only on real pass and only for target ages in [min,max]
		try:
			if (data_type == 'real') and (self.target_id_bank is not None):
				# Determine stage and pick lambda
				stage1 = bool(getattr(self.opts, 'train_encoder', False))
				stage2 = bool(getattr(self.opts, 'train_decoder', False)) and not stage1
				lam = 0.0
				if stage1:
					lam = float(self.target_id_lambda_s1)
				elif stage2:
					lam = float(self.target_id_lambda_s2)
				# Active mask based on integer-rounded years
				years = torch.clamp((target_ages * 100.0).round().to(torch.int64), 0, 200)
				mask = (years >= int(self.target_id_apply_min_age)) & (years <= int(self.target_id_apply_max_age))
				applied_ratio = float(mask.float().mean().item()) if mask.numel() > 0 else 0.0
				self.logger.add_scalar(f"train/target_id_applied_ratio_{data_type}", applied_ratio, self.global_step)
				loss_dict[f'target_id_applied_ratio_{data_type}'] = applied_ratio
				L_tid = torch.tensor(0.0, device=y_hat.device)
				if lam > 0.0 and mask.any():
					# Ensure ID encoder present
					if not hasattr(self, 'id_loss'):
						self.id_loss = id_loss.IDLoss().to(self.device).eval()
					feat = self.id_loss.extract_feats(y_hat[mask])  # [B',512]
					feat = F.normalize(feat, dim=1)
					ages_act = years[mask].tolist()
					protos = []
					for a in ages_act:
						if len(self.target_id_bank) == 0:
							proto = None
						else:
							closest = min(self.target_id_bank.keys(), key=lambda k: abs(int(k) - int(a)))
							proto = self.target_id_bank.get(int(closest), None)
						if proto is None:
							proto = torch.zeros(512, device=y_hat.device)
						protos.append(proto.unsqueeze(0))
					if len(protos) > 0:
						P = torch.cat(protos, dim=0)
						P = F.normalize(P, dim=1)
						cos = torch.sum(feat * P, dim=1)
						L_tid = (1.0 - cos).mean()
						loss = loss + lam * L_tid
						self.logger.add_scalar(f"train/loss_target_id_{data_type}", float(L_tid.item()), self.global_step)
						loss_dict[f'loss_target_id_{data_type}'] = float(L_tid.item())
					else:
						self.logger.add_scalar(f"train/loss_target_id_{data_type}", 0.0, self.global_step)
						loss_dict[f'loss_target_id_{data_type}'] = 0.0
				else:
					self.logger.add_scalar(f"train/loss_target_id_{data_type}", 0.0, self.global_step)
					loss_dict[f'loss_target_id_{data_type}'] = 0.0
		except Exception:
			self.logger.add_scalar(f"train/loss_target_id_{data_type}", 0.0, self.global_step)
			loss_dict[f'loss_target_id_{data_type}'] = 0.0
			loss_dict[f'target_id_applied_ratio_{data_type}'] = 0.0
		# Age-anchor loss (Task 4) — apply only on real pass, gated by stage and flags
		try:
			if (data_type == "real") and self.anchor_enabled and (self.anchor_lambda > 0.0):
				apply_s1 = (self.anchor_stage == 's1' and bool(getattr(self.opts, 'train_encoder', False)))
				apply_s2 = (self.anchor_stage == 's2' and bool(getattr(self.opts, 'train_decoder', False)) and not bool(getattr(self.opts, 'train_encoder', False)))
				apply_both = (self.anchor_stage == 'both')
				if apply_s1 or apply_s2 or apply_both:
					# Convert W+ -> W mean per sample; keep gradients
					w_mean = latent.mean(dim=1)
					# target_ages provided in [0,1]; convert to years float
					target_ages_years = torch.clamp(target_ages * 100.0, 0.0, 200.0)
					L_anchor = self.anchor_loss(w_mean, target_ages_years)
					loss = loss + float(self.anchor_lambda) * L_anchor
					self.logger.add_scalar("train/loss_anchor", float(L_anchor.item()), self.global_step)
					# Diagnostics: chosen bin mid mean (approximate by recomputing indices)
					mids = torch.tensor(self.anchor_bin_mids, device=target_ages_years.device, dtype=target_ages_years.dtype)
					idx = torch.argmin((target_ages_years.view(-1,1) - mids.view(1,-1)).abs(), dim=1)
					bin_mid_mean = float(mids.index_select(0, idx).mean().item()) if mids.numel() > 0 else 0.0
					self.logger.add_scalar("train/anchor_age_bin_mean", bin_mid_mean, self.global_step)
					loss_dict['loss_anchor'] = float(L_anchor.item())
					loss_dict['anchor_age_bin_mean'] = float(bin_mid_mean)
					# Missing ratio is zero with nearest-bin strategy
					self.logger.add_scalar("train/anchor_bin_missing_ratio", 0.0, self.global_step)
					loss_dict['anchor_bin_missing_ratio'] = 0.0
			else:
				loss_dict.setdefault('loss_anchor', 0.0)
				loss_dict.setdefault('anchor_age_bin_mean', 0.0)
				loss_dict.setdefault('anchor_bin_missing_ratio', 0.0)
		except Exception:
			# On any error, keep training stable and log zeros
			self.logger.add_scalar("train/loss_anchor", 0.0, self.global_step)
			self.logger.add_scalar("train/anchor_bin_missing_ratio", 0.0, self.global_step)
			self.logger.add_scalar("train/anchor_age_bin_mean", 0.0, self.global_step)
			loss_dict.setdefault('loss_anchor', 0.0)
			loss_dict.setdefault('anchor_bin_missing_ratio', 0.0)
			loss_dict.setdefault('anchor_age_bin_mean', 0.0)
		# Expose effective regularizer weights for diagnostics
		loss_dict['effective_w_norm_lambda'] = float(effective_w_norm_lambda)
		loss_dict['effective_aging_lambda'] = float(effective_aging_lambda)
		# Impostor-only age-aware contrastive ID loss (real pass only)
		if (data_type == "real"):
			apply_mask = None
			# Log interpolation coverage within training age range (for NN-ID visibility)
			try:
				interp_ratio = 0.0
				if hasattr(self, 'train_min_age') and hasattr(self, 'train_max_age') and \
					self.train_min_age is not None and self.train_max_age is not None:
					min_age = float(self.train_min_age) / 100.0
					max_age = float(self.train_max_age) / 100.0
					mask_interp = (target_ages >= min_age) & (target_ages <= max_age)
					interp_ratio = float(mask_interp.float().mean().item())
				self.logger.add_scalar("train/nn_interpolation_ratio", interp_ratio, self.global_step)
				loss_dict['nn_interpolation_ratio'] = interp_ratio
			except Exception:
				self.logger.add_scalar("train/nn_interpolation_ratio", 0.0, self.global_step)
				loss_dict['nn_interpolation_ratio'] = 0.0
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
							negs, sims_sel, miner_meta = self.miner.query(
								q.detach(), ages_cpu, k=self.mb_k,
								min_sim=self.mb_min_sim, max_sim=self.mb_max_sim,
								top_m=self.mb_top_m, radius=self.mb_bin_neighbor_radius,
								device=y_hat.device
							)
							# Log miner diagnostics to TensorBoard (at board interval)
							try:
								if (self.global_step % int(getattr(self.opts, 'board_interval', 50) or 50)) == 0:
									self.logger.add_scalar("train/mb_candidate_count", float(miner_meta.get('candidate_count', 0.0)), self.global_step)
									self.logger.add_scalar("train/mb_sim_mean", float(miner_meta.get('sim_mean', 0.0)), self.global_step)
									self.logger.add_scalar("train/mb_sim_std", float(miner_meta.get('sim_std', 0.0)), self.global_step)
									self.logger.add_scalar("train/mb_sim_p50", float(miner_meta.get('sim_p50', 0.0)), self.global_step)
									self.logger.add_scalar("train/mb_sim_p75", float(miner_meta.get('sim_p75', 0.0)), self.global_step)
									self.logger.add_scalar("train/mb_sim_p90", float(miner_meta.get('sim_p90', 0.0)), self.global_step)
									self.logger.add_scalar("train/mb_k_effective", float(miner_meta.get('k_effective', 0.0)), self.global_step)
									self.logger.add_scalar("train/mb_band_min", float(miner_meta.get('band_min', self.mb_min_sim)), self.global_step)
									self.logger.add_scalar("train/mb_band_max", float(miner_meta.get('band_max', self.mb_max_sim)), self.global_step)
							except Exception:
								pass
							# Also store summary stats to loss_dict for timestamp visibility
							loss_dict['mb_candidate_count'] = float(miner_meta.get('candidate_count', 0.0))
							loss_dict['mb_sim_mean'] = float(miner_meta.get('sim_mean', 0.0))
							loss_dict['mb_sim_std'] = float(miner_meta.get('sim_std', 0.0))
							loss_dict['mb_sim_p50'] = float(miner_meta.get('sim_p50', 0.0))
							loss_dict['mb_sim_p75'] = float(miner_meta.get('sim_p75', 0.0))
							loss_dict['mb_sim_p90'] = float(miner_meta.get('sim_p90', 0.0))
							loss_dict['mb_k_effective'] = float(miner_meta.get('k_effective', 0.0))
							self.last_miner_meta = miner_meta
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
				try:
					print(f"Warning: contrastive impostor loss error at step {self.global_step}: {e}")
				except Exception:
					pass
				self.logger.add_scalar("train/loss_contrastive_id", 0.0, self.global_step)
				self.logger.add_scalar("train/mb_applied_ratio", 0.0, self.global_step)
				loss_dict['loss_contrastive_id'] = 0.0
				loss_dict['mb_applied_ratio'] = 0.0
			# ROI-ID micro-loss on real pass only — include in loss_dict for timestamp visibility
			if (self.cur_roi_lambda > 0 or self.roi_id_lambda > 0) and (self.roi_cropper is not None) and (self.roi_use_eyes or self.roi_use_mouth or self.roi_use_nose or self.roi_use_broweyes):
				try:
					roi_losses = []
					roi_count = 0
					roi_eyes = 0
					roi_mouth = 0
					roi_nose = 0
					roi_broweyes = 0
					landmark_failures = 0
					x_src = x
					x_gen = y_hat
					# Precompute landmarks batch if geometry is enabled to reuse for ROI crops and geometry ratios
					pre_pts_src = None
					pre_pts_gen = None
					if getattr(self, 'geom_enabled', False) and (getattr(self, 'geom_cropper', None) is not None):
						try:
							pre_pts_src = self.geom_cropper.landmarks_batch(x_src)
							pre_pts_gen = self.geom_cropper.landmarks_batch(x_gen)
						except Exception:
							pre_pts_src = None
							pre_pts_gen = None
					for b in range(int(x_gen.size(0))):
						src = x_src[b]
						gen = x_gen[b]
						if pre_pts_src is not None and pre_pts_gen is not None:
							crops_src = self.roi_cropper.rois_from_landmarks(src, pre_pts_src[b], self.roi_pad, self.roi_jitter, self.roi_size, train=True,
																use_eyes=self.roi_use_eyes, use_mouth=self.roi_use_mouth, use_nose=self.roi_use_nose, use_broweyes=self.roi_use_broweyes)
							crops_gen = self.roi_cropper.rois_from_landmarks(gen, pre_pts_gen[b], self.roi_pad, self.roi_jitter, self.roi_size, train=True,
																use_eyes=self.roi_use_eyes, use_mouth=self.roi_use_mouth, use_nose=self.roi_use_nose, use_broweyes=self.roi_use_broweyes)
							info_src = {'landmarks_used': True}
							info_gen = {'landmarks_used': True}
						else:
							crops_src, info_src = self.roi_cropper.rois(src, self.roi_pad, self.roi_jitter, self.roi_size, train=True,
																use_eyes=self.roi_use_eyes, use_mouth=self.roi_use_mouth, use_nose=self.roi_use_nose, use_broweyes=self.roi_use_broweyes, return_info=True)
							crops_gen, info_gen = self.roi_cropper.rois(gen, self.roi_pad, self.roi_jitter, self.roi_size, train=True,
																use_eyes=self.roi_use_eyes, use_mouth=self.roi_use_mouth, use_nose=self.roi_use_nose, use_broweyes=self.roi_use_broweyes, return_info=True)
						if not (info_src.get('landmarks_used', False) and info_gen.get('landmarks_used', False)):
							landmark_failures += 1
						for key in list(crops_src.keys()):
							if key in crops_gen:
								# Use IR-SE50 features; keep grad path to y_hat
								e_src = self.id_loss.extract_feats(crops_src[key].unsqueeze(0).to(self.device))
								e_gen = self.id_loss.extract_feats(crops_gen[key].unsqueeze(0).to(self.device))
								e_src = F.normalize(e_src, dim=1)
								e_gen = F.normalize(e_gen, dim=1)
								cos = torch.sum(e_src * e_gen, dim=1)  # [1]
								loss_roi = (1.0 - cos).mean()
								roi_losses.append(loss_roi)
								roi_count += 1
								if key == 'eyes':
									roi_eyes += 1
								elif key == 'mouth':
									roi_mouth += 1
								elif key == 'nose':
									roi_nose += 1
								elif key == 'broweyes':
									roi_broweyes += 1
						if len(roi_losses) > 0:
							L_roi = torch.stack(roi_losses).mean()
							# Use current scheduled lambda
							loss = loss + float(self.cur_roi_lambda) * L_roi
							self.logger.add_scalar("train/loss_roi_id", float(L_roi.item()), self.global_step)
							self.logger.add_scalar("train/roi_pairs", int(roi_count), self.global_step)
							self.logger.add_scalar("train/roi_pairs_eyes", int(roi_eyes), self.global_step)
							self.logger.add_scalar("train/roi_pairs_mouth", int(roi_mouth), self.global_step)
							self.logger.add_scalar("train/roi_pairs_nose", int(roi_nose), self.global_step)
							self.logger.add_scalar("train/roi_pairs_broweyes", int(roi_broweyes), self.global_step)
							self.logger.add_scalar("train/roi_landmark_failures", int(landmark_failures), self.global_step)
							loss_dict['loss_roi_id'] = float(L_roi.item())
							loss_dict['roi_pairs'] = int(roi_count)
							loss_dict['roi_pairs_eyes'] = int(roi_eyes)
							loss_dict['roi_pairs_mouth'] = int(roi_mouth)
							loss_dict['roi_pairs_nose'] = int(roi_nose)
							loss_dict['roi_pairs_broweyes'] = int(roi_broweyes)
							loss_dict['roi_landmark_failures'] = int(landmark_failures)
						else:
							self.logger.add_scalar("train/loss_roi_id", 0.0, self.global_step)
							self.logger.add_scalar("train/roi_pairs", 0, self.global_step)
							self.logger.add_scalar("train/roi_pairs_eyes", 0, self.global_step)
							self.logger.add_scalar("train/roi_pairs_mouth", 0, self.global_step)
							self.logger.add_scalar("train/roi_pairs_nose", 0, self.global_step)
							self.logger.add_scalar("train/roi_pairs_broweyes", 0, self.global_step)
							self.logger.add_scalar("train/roi_landmark_failures", 0, self.global_step)
							loss_dict['loss_roi_id'] = 0.0
							loss_dict['roi_pairs'] = 0
							loss_dict['roi_pairs_eyes'] = 0
							loss_dict['roi_pairs_mouth'] = 0
							loss_dict['roi_pairs_nose'] = 0
							loss_dict['roi_pairs_broweyes'] = 0
							loss_dict['roi_landmark_failures'] = 0
				except Exception:
						self.logger.add_scalar("train/loss_roi_id", 0.0, self.global_step)
						self.logger.add_scalar("train/roi_pairs", 0, self.global_step)
						self.logger.add_scalar("train/roi_pairs_eyes", 0, self.global_step)
						self.logger.add_scalar("train/roi_pairs_mouth", 0, self.global_step)
						self.logger.add_scalar("train/roi_pairs_nose", 0, self.global_step)
						self.logger.add_scalar("train/roi_pairs_broweyes", 0, self.global_step)
						self.logger.add_scalar("train/roi_landmark_failures", 0, self.global_step)
						loss_dict['loss_roi_id'] = 0.0
						loss_dict['roi_pairs'] = 0
						loss_dict['roi_pairs_eyes'] = 0
						loss_dict['roi_pairs_mouth'] = 0
						loss_dict['roi_pairs_nose'] = 0
						loss_dict['roi_pairs_broweyes'] = 0
						loss_dict['roi_landmark_failures'] = 0
				# Geometry loss (shape ratios) with stage gating and NaN guard
				try:
					if getattr(self, 'geom_enabled', False) and (getattr(self, 'geom_loss', None) is not None) and (getattr(self, 'geom_cropper', None) is not None):
						apply_s1 = (self.geom_stage == 's1' and bool(getattr(self.opts, 'train_encoder', False)))
						apply_s2 = (self.geom_stage == 's2' and bool(getattr(self.opts, 'train_decoder', False)) and not bool(getattr(self.opts, 'train_encoder', False)))
						apply_both = (self.geom_stage == 'both')
						if apply_s1 or apply_s2 or apply_both:
							land_src = self.geom_cropper.landmarks_batch(x)
							land_out = self.geom_cropper.landmarks_batch(y_hat)
							if (land_src is not None) and (land_out is not None):
								from training.losses.geometry_loss import ratios as geom_ratios
								g_src = geom_ratios(land_src)
								g_out = geom_ratios(land_out)
								L_geom = self.geom_loss(g_out, g_src)
								if torch.isfinite(L_geom):
									loss = loss + float(self.geom_lambda) * L_geom
									self.logger.add_scalar("train/loss_geom", float(L_geom.item()), self.global_step)
									loss_dict['loss_geom'] = float(L_geom.item())
								else:
									self.logger.add_scalar("train/loss_geom", 0.0, self.global_step)
									loss_dict['loss_geom'] = 0.0
							else:
								self.logger.add_scalar("train/loss_geom", 0.0, self.global_step)
								loss_dict['loss_geom'] = 0.0
				except Exception:
					self.logger.add_scalar("train/loss_geom", 0.0, self.global_step)
					loss_dict['loss_geom'] = 0.0
		# Nearest-neighbor identity regularizer during interpolation only
		nn_lambda = float(getattr(self.opts, 'nearest_neighbor_id_loss_lambda', 0.0) or 0.0)
		if (nn_lambda > 0) and (not no_aging) and self._is_interpolation(target_ages) and (self.feats_dict is not None):
			nearest_neighbor_id_loss = self._nearest_neighbor_id_loss(y_hat, target_ages)
			loss_dict['nearest_neighbor_id_loss'] = float(nearest_neighbor_id_loss)
			loss += nearest_neighbor_id_loss * nn_lambda
		else:
			# Ensure metric visibility when inactive
			loss_dict.setdefault('nearest_neighbor_id_loss', 0.0)
		loss_dict[f'loss_{data_type}'] = float(loss)
		if data_type == "cycle":
			loss = loss * self.opts.cycle_lambda
		# Ensure tensor scalar return for backward safety even if no losses active
		if not torch.is_tensor(loss):
			loss = torch.zeros((), device=y_hat.device, requires_grad=True)
		# Identity-adversarial loss on generated image (encourage actor class)
		lambda_adv_base = float(getattr(self.opts, 'id_adv_lambda', 0.0) or 0.0)
		if bool(getattr(self, 'id_adv_enabled', getattr(self.opts, 'id_adv_enabled', False))) and (getattr(self, 'id_adv', None) is not None) and (data_type == 'real') and bool(getattr(self.opts, 'train_encoder', False)):
			# Ensure helper attributes exist (built in __init__ in this coach)
			try:
				from training.utils.schedules import value_for_step
				lam_cur = value_for_step(getattr(self, 'id_adv_s1_schedule', []), int(self.global_step))
			except Exception:
				lam_cur = None
			lam_cur = float(lam_cur) if lam_cur is not None else float(lambda_adv_base)
			if lam_cur > 0.0:
				# Preprocess and build TTA views
				y_in = self.id_adv_preproc(y_hat)
				views = []
				name_list = [t.strip() for t in str(getattr(self.opts, 'id_adv_tta', 'clean') or 'clean').split(',') if len(t.strip())>0]
				for name in name_list:
					v = y_in
					if name == 'clean':
						pass
					elif name == 'flip':
						from models.binary_identity_model import tta_flip_horizontal
						v = tta_flip_horizontal(v)
					elif name.startswith('jpeg'):
						from models.binary_identity_model import tta_jpeg75
						v = tta_jpeg75(v)
					elif name.startswith('blur'):
						from models.binary_identity_model import tta_gaussian_blur
						try:
							sigma = float(name.split('blur',1)[1])
						except Exception:
							sigma = 0.6
						v = tta_gaussian_blur(v, sigma=sigma)
					else:
						continue
					views.append(v)
				# Run classifier via wrapper
				from models.binary_identity_model import IdentityModelWrapper
				wrap = getattr(self, 'id_adv_wrap', None)
				if wrap is None:
					wrap = IdentityModelWrapper(self.id_adv)
				out = wrap.forward_multi(views)
				# Extract clean probs/logits and aggregates
				p_clean = None
				logits_clean = None
				name_to_tensor = {}
				idx = 0
				for name in name_list:
					if name not in ('clean','flip') and not name.startswith('jpeg') and not name.startswith('blur'):
						continue
					name_to_tensor[name] = out['probs'][idx][:, 1]
					if name == 'clean':
						p_clean = out['probs'][idx][:, 1]
						logits_clean = out['logits'][idx]
					idx += 1
				if p_clean is None:
					p_clean = out['probs'][0][:, 1]
					logits_clean = out['logits'][0]
				# Aggregate expression
				import re
				def eval_func(token: str):
					token = token.strip()
					if token.startswith('mean(') and token.endswith(')'):
						inside = token[5:-1]
						names = [t.strip() for t in inside.split(',') if t.strip()]
						vals = [name_to_tensor[n] for n in names if n in name_to_tensor]
						return torch.stack(vals, dim=0).mean(dim=0) if len(vals)>0 else p_clean
					if token.startswith('min(') and token.endswith(')'):
						inside = token[4:-1]
						names = [t.strip() for t in inside.split(',') if t.strip()]
						vals = [name_to_tensor[n] for n in names if n in name_to_tensor]
						return (torch.stack(vals, dim=0).min(dim=0).values) if len(vals)>0 else p_clean
					return name_to_tensor.get(token, p_clean)
				expr = str(getattr(self.opts, 'id_adv_agg', 'mean(clean)') or 'mean(clean)')
				sum_vec = torch.zeros_like(p_clean)
				for part in expr.split('+'):
					part = part.strip()
					m = re.match(r"^([0-9\.]+)\s*\*\s*(.+)$", part)
					if m:
						scale = float(m.group(1))
						func = m.group(2).strip()
						val = eval_func(func)
						sum_vec = sum_vec + scale * val
					else:
						val = eval_func(part)
						sum_vec = sum_vec + val
				p_actor_mean = sum_vec
				aug_names = [n for n in name_list if n != 'clean']
				if len(aug_names) > 0:
					vals = [name_to_tensor[n] for n in aug_names if n in name_to_tensor]
					p_actor_min_aug = (torch.stack(vals, dim=0).min(dim=0).values) if len(vals)>0 else p_clean
				else:
					p_actor_min_aug = p_clean
				# Compute losses
				from training.losses_idadv import focal_ce, logit_margin_hinge, conf_weight
				gamma = float(getattr(self.opts, 'id_adv_focal_gamma', 0.0) or 0.0)
				margin = float(getattr(self.opts, 'id_adv_margin', 0.0) or 0.0)
				labels_actor = torch.ones((logits_clean.size(0),), dtype=torch.long, device=logits_clean.device)
				ce_vec = focal_ce(logits_clean, labels_actor, gamma=gamma, reduction='none')
				margin_vec = logit_margin_hinge(logits_clean[:,1], logits_clean[:,0], margin=margin, reduction='none')
				combined = 0.8 * ce_vec + 0.2 * margin_vec
				from training.losses_idadv import parse_conf_weight_spec
				cw = parse_conf_weight_spec(getattr(self.opts, 'id_adv_conf_weight', ''))
				if cw is not None:
					k, p_thr = cw
					w = conf_weight(p_clean, k=k, p_thr=p_thr)
					combined = combined * w
				L_adv = combined.mean()
				loss = loss + lam_cur * L_adv
				# Log into loss_dict for timestamp
				loss_dict['id_adv_lambda_current'] = float(lam_cur)
				loss_dict['id_adv_p_actor_mean'] = float(p_actor_mean.mean().item())
				loss_dict['id_adv_p_actor_min_aug'] = float(p_actor_min_aug.mean().item())
				loss_dict['id_adv_logit_margin'] = float((logits_clean[:,1] - logits_clean[:,0]).mean().item())
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

		# If resuming, extend max_steps by additional_steps (if provided)
		try:
			extra = int(getattr(self.opts, 'additional_steps', 0) or 0)
			if extra > 0:
				new_max = int(self.global_step) + extra
				if new_max > int(self.opts.max_steps):
					self.opts.max_steps = new_max
					print(f"Continuing training: +{extra} steps (total max_steps={self.opts.max_steps})")
		except Exception:
			pass
		
		# Restore EMA buffers if present and EMA is enabled
		if self.ema_enabled and 'ema' in checkpoint:
			try:
				ema_states = checkpoint.get('ema', {})
				for name, helper in self.ema_helpers.items():
					if name in ema_states:
						helper.load_state_dict(ema_states[name])
				print("EMA state restored from checkpoint")
			except Exception as e:
				print(f"Warning: failed to restore EMA state: {e}")

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
		# Save EMA helper states if enabled
		if self.ema_enabled and len(self.ema_helpers) > 0:
			ema_states = {}
			for name, helper in self.ema_helpers.items():
				ema_states[name] = helper.state_dict()
			save_dict['ema'] = ema_states
			save_dict['ema_scope'] = str(self.ema_scope)
			save_dict['ema_decay'] = float(self.ema_decay)
		return save_dict

	def _init_id_adv(self):
		model_path = str(getattr(self.opts, 'id_adv_model_path', '') or '')
		backend = str(getattr(self.opts, 'id_adv_backend', 'arcface') or 'arcface')
		inp = int(getattr(self.opts, 'id_adv_input_size', 112) or 112)
		# Load model if feature enabled and path exists, regardless of base lambda (schedule may drive lambda)
		if (not bool(getattr(self, 'id_adv_enabled', getattr(self.opts, 'id_adv_enabled', False)))) or len(model_path) == 0 or (not os.path.exists(model_path)):
			self.id_adv = None
			self.id_adv_preproc = None
			return
		print(f"[ID-ADV] Loading discriminator from {model_path} (backend={backend})")
		model = build_identity_model(backend=backend, weights_path=None, num_outputs=2, input_size=inp)
		ckpt = torch.load(model_path, map_location='cpu')
		state = ckpt.get('state_dict', ckpt)
		model.load_state_dict(state, strict=False)
		model.eval()
		for p in model.parameters():
			p.requires_grad = False
		self.id_adv = model.to(self.device)
		self.id_adv_preproc = torch.nn.Sequential(
			torch.nn.Upsample(size=(inp, inp), mode='bilinear', align_corners=False)
		)


