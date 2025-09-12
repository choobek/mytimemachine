import os
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import torch.distributed as dist
import tempfile
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import common, train_utils
from criteria import id_loss, w_norm
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from datasets.augmentations import AgeTransformer
from criteria.lpips.lpips import LPIPS
from criteria.aging_loss import AgingLoss
from models.psp import pSp
from training.ranger import Ranger

import collections
import gpytorch
import copy
class Coach:
	def __init__(self, opts):
		self.opts = opts

		self.global_step = 0

		self.device = 'cuda'
		self.opts.device = self.device

		# Initialize loss
		self.mse_loss = nn.MSELoss().to(self.device).eval()
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.id_lambda > 0:
			self.id_loss = id_loss.IDLoss().to(self.device).eval()
		if self.opts.w_norm_lambda > 0:
			self.w_norm_loss = w_norm.WNormLoss(opts=self.opts)
		if self.opts.aging_lambda > 0:
			self.aging_loss = AgingLoss(self.opts)

		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=False)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)
		
		# personalization
		print('Training face feature extraction:')
		self.feats_dict = collections.defaultdict(list)
		# store all the training features in a dictionary
		for x, y, x_path, y_path in self.train_dataset:
			# x and y are the same in this case
			x = x.to(self.device).float()
			img = x.unsqueeze(0)
			feat = self.id_loss.extract_feats(img)
			# self.feats_dict[x_path] = feat
			# x_age = self.aging_loss.extract_ages(img) / 100.
			x_age = self.aging_loss.extract_ages_gt(x_path) / 100
			# round to 2 decimal places
			key = round(x_age.item(), 2)
			self.feats_dict[key].append((feat, x_path))
		print(f'Processed all {len(self.feats_dict)} training images for personalization')
		# save into opts for future use
		self.opts.train_min_age = int(min(self.feats_dict.keys()) * 100)
		self.opts.train_max_age = int(max(self.feats_dict.keys()) * 100)
		print('Training data min_age (ground truth):', self.opts.train_min_age)
		print('Training data max_age (ground truth):', self.opts.train_max_age)
		assert sum([len(v) for v in self.feats_dict.values()]) == len(self.train_dataset)
		self.opts.feats_dict = self.feats_dict

		# Initialize network
		self.net = pSp(self.opts).to(self.device)
		# make a copy of the original net
		self.net_global = copy.deepcopy(self.net)
		self.net_global.eval()

		self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.net.likelihood, self.net.blender)


		# Initialize optimizer
		self.optimizer = self.configure_optimizers()
		self.optimizer_blender = self.configure_optimizers_blender()



		self.age_transformer = AgeTransformer(target_age="uniform_random")


		# print('Training face age extraction:')
		# self.age_dict = {}
		# # store all the training ages in a dictionary
		# for x, y, x_path, y_path in self.train_dataset:
		# 	# x and y are the same in this case
		# 	x = x.to(self.device).float()
		# 	img = x.unsqueeze(0)
		# 	age = self.aging_loss.extract_ages(img)
		# 	self.age_dict[x_path] = age.detach().cpu().numpy()
		# print(f'Processed all {len(self.age_dict)} training images for personalization')
		# assert len(self.age_dict) == len(self.train_dataset)
		# min_age = min(self.age_dict.values())
		# max_age = max(self.age_dict.values())
		# print('Training data min_age (estimated):', min_age)
		# print('Training data max_age (estimated):', max_age)
		
		# interpolation mode is False for original SAM loss
		self.interpolation = False
		print('Change AgeTransformer to personalization mode (restricted to the range of the training data)')
		self.age_transformer_interpolation = AgeTransformer(target_age="interpolation", range=(self.opts.train_min_age, self.opts.train_max_age))
		# self.age_transformer = self.age_transformer_interpolation 
		self.age_transformer_extrapolation = AgeTransformer(target_age="extrapolation", range=(self.opts.train_min_age, self.opts.train_max_age))
		
		# no restriction on the range of age transformation
		# print('sanity check for feats_dict: ', self.feats_dict.keys())


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

	def perform_forward_pass(self, x):
		y_hat, latent = self.net.forward(x, return_latents=True)
		return y_hat, latent


	# def perform_forward_pass_blender(self, x):
	# 	# x: [b, 4, 256, 256]
	# 	_, latent_local = self.net.forward(x, return_styles=True)
	# 	_, latent_global = self.net_global.forward(x, return_styles=True)
	# 	target_ages = x[:, -1, 0, 0]
	# 	latent_blended = self.net.blender(latent_local, latent_global, target_ages)
	# 	y_hat, _ = self.net.decoder(
	# 		[latent_blended], 
	# 		input_is_latent=True, 
	# 		randomize_noise=False
	# 		)
	# 	# resize
	# 	y_hat = self.net.face_pool(y_hat)
	# 	return y_hat, latent_blended

	
	def __set_target_to_source(self, x, input_ages):
		return [torch.cat((img, age * torch.ones((1, img.shape[1], img.shape[2])).to(self.device)))
				for img, age in zip(x, input_ages)]

	def train(self):
		self.net.train()
		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				x, y, x_path, y_path = batch
				x, y = x.to(self.device).float(), y.to(self.device).float()

				# input_ages = self.aging_loss.extract_ages(x) / 100.
				input_ages = self.aging_loss.extract_ages_gt(x_path) / 100
				input_ages = input_ages.to(self.device)

				# perform no aging in 33% of the time
				no_aging = random.random() <= (1. / 3)
				if no_aging:
					x_input = self.__set_target_to_source(x=x, input_ages=input_ages)
				else:
					# perform in-domain aging in 50% of the time
					self.interpolation = random.random() <= (1. / 2)
					if self.interpolation:
						x_input = [self.age_transformer_interpolation(img.cpu()).to(self.device) for img in x]
					else:
						x_input = [self.age_transformer_extrapolation(img.cpu()).to(self.device) for img in x]

				x_input = torch.stack(x_input)
				target_ages = x_input[:, -1, 0, 0]

				# print('sanity check:')
				# print('x_input shape:', x_input.shape)
				# x_input shape: torch.Size([b, 4, 256, 256])
				# print('target_ages shape:', target_ages.shape)
				# target_ages shape: torch.Size([b])

				if self.opts.train_encoder:
					self.optimizer.zero_grad()
					# self.optimizer_blender.zero_grad()

					# todo
					personalization_step = 1e4
					# personalization_step = 100
					if self.global_step <= personalization_step:
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

						# combine the logs of both forwards
						for idx, cycle_log in enumerate(cycle_id_logs):
							id_logs[idx].update(cycle_log)
						loss_dict.update(cycle_loss_dict)
						loss_dict["loss"] = loss_dict["loss_real"] + loss_dict["loss_cycle"]

						self.optimizer.step()

					else:
						self.optimizer_blender.zero_grad()

						train_x = self.feats_dict.keys()
						train_x = torch.tensor(list(train_x)).float()
						train_y = torch.ones_like(train_x)

						train_x = train_x.to(self.device).float()
						train_y = train_y.to(self.device).float()

						output = self.net.blender(train_x)
						loss = -self.mll(output, train_y)
						loss.backward()

						# print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
						# 	self.global_step + 1, self.opts.max_steps, loss.item(),
						# 	self.net.blender.covar_module.base_kernel.lengthscale.item(),
						# 	self.net.blender.likelihood.noise.item()
						# ))
						
						loss_dict = {}
						loss_dict['gaussian_process_loss'] = loss.item()

						self.optimizer_blender.step()

					# if self.global_step <= personalization_step:
					# 	self.optimizer.step()
					# else:
					# 	self.optimizer_blender.step()

				if self.opts.train_decoder:
					# input_ages_clone = input_ages.clone().detach().requires_grad_(True)
					# x_input_person = self.__set_target_to_source(x=x, input_ages=input_ages_clone)
					# x_input_person = torch.stack(x_input_person)
					# y_hat_person, latent_person = self.perform_forward_pass(x_input_person)
					# personalization_loss, personalization_loss_dict, personalization_id_logs = self.calc_loss(
					# 	# x, y, y_hat, latent,
					# 	x=x_input_person,
					# 	y=y,
					# 	y_hat=y_hat_person,
					# 	latent=latent_person,
					# 	target_ages=input_ages_clone,
					# 	input_ages=input_ages_clone,
					# 	no_aging=True,
					# 	data_type="real",
					# 	personalization=True)

					# perform no aging in 33% of the time
					# no_aging = random.random() <= (1. / 3)
					# if no_aging:
					# 	x_input_person = self.__set_target_to_source(x=x, input_ages=input_ages)
					# else:
					# 	x_input_person = [self.age_transformer(img.cpu()).to(self.device) for img in x]

					# for decoder, solely reconstruction loss purpose
					no_aging = True
					x_input_person = self.__set_target_to_source(x=x, input_ages=input_ages)

					x_input_person = torch.stack(x_input_person)
					input_ages_clone_person = input_ages.clone().detach().requires_grad_(True)
					# perform forward/backward pass on real images
					y_hat_person, latent_person = self.perform_forward_pass(x_input_person)
					
					# x_input_person = x_input.clone().detach().requires_grad_(True)
					# input_ages_clone_person = input_ages.clone().detach().requires_grad_(True)
					# y_hat_person, latent_person = self.perform_forward_pass(x_input_person)

					decoder_loss, decoder_loss_dict, decoder_id_logs = self.calc_loss(
						x=x,
						y=y,
						y_hat=y_hat_person,
						latent=latent_person,
						target_ages=x_input_person[:, -1, 0, 0],
						input_ages=input_ages_clone_person,
						no_aging=no_aging,
						data_type="real"
						)
					loss = decoder_loss
					loss.backward()

				# self.optimizer.step()

				# Logging related
				# if self.global_step % self.opts.image_interval == 0 or \
				# 		(self.global_step < 1000 and self.global_step % 25 == 0):
				# 	self.parse_and_log_images(id_logs, x, y, y_hat, y_recovered,
				# 							  title='images/train/faces')

				if self.global_step % self.opts.board_interval == 0:
					if self.opts.train_encoder:
						# prefix = 'encoder_decoder' if self.opts.train_decoder else 'encoder'
						prefix = 'encoder' if self.global_step <= personalization_step else 'blender'
						self.print_metrics(loss_dict, prefix=prefix)
						self.log_metrics(loss_dict, prefix=prefix)
					elif self.opts.train_decoder:
						self.print_metrics(decoder_loss_dict, prefix='decoder')
						self.log_metrics(decoder_loss_dict, prefix='decoder')

				# Validation related
				# val_loss_dict = None
				# if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
				# 	# todo: remove validation
				# 	# val_loss_dict = self.validate()
				# 	if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
				# 		self.best_val_loss = val_loss_dict['loss']
				# 		self.checkpoint_me(val_loss_dict, is_best=True)

				# if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
				# 	if val_loss_dict is not None:
				# 		self.checkpoint_me(val_loss_dict, is_best=False)
				# 	else:
				# 		self.checkpoint_me(loss_dict, is_best=False)


				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1

	# def validate(self):
	# 	self.net.eval()
	# 	agg_loss_dict = []
	# 	for batch_idx, batch in enumerate(self.test_dataloader):
	# 		x, y = batch
	# 		with torch.no_grad():
	# 			x, y = x.to(self.device).float(), y.to(self.device).float()

	# 			input_ages = self.aging_loss.extract_ages(x) / 100.
	# 			print('sanity check: input_ages:', input_ages)
	# 			print('sanity check: input_ages shape:', input_ages.shape)

	# 			# perform no aging in 33% of the time
	# 			no_aging = random.random() <= (1. / 3)
	# 			if no_aging:
	# 				x_input = self.__set_target_to_source(x=x, input_ages=input_ages)
	# 			else:
	# 				x_input = [self.age_transformer(img.cpu()).to(self.device) for img in x]

	# 			x_input = torch.stack(x_input)
	# 			target_ages = x_input[:, -1, 0, 0]

	# 			# perform forward/backward pass on real images
	# 			y_hat, latent = self.perform_forward_pass(x_input)
	# 			_, cur_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent,
	# 													   target_ages=target_ages,
	# 													   input_ages=input_ages,
	# 													   no_aging=no_aging,
	# 													   data_type="real")

	# 			# perform cycle on generate images by setting the target ages to the original input ages
	# 			y_hat_inverse = self.__set_target_to_source(x=y_hat, input_ages=input_ages)
	# 			y_hat_inverse = torch.stack(y_hat_inverse)
	# 			reverse_target_ages = y_hat_inverse[:, -1, 0, 0]
	# 			y_recovered, latent_cycle = self.perform_forward_pass(y_hat_inverse)
	# 			loss, cycle_loss_dict, cycle_id_logs = self.calc_loss(x, y, y_recovered, latent_cycle,
	# 																  target_ages=reverse_target_ages,
	# 																  input_ages=input_ages,
	# 																  no_aging=no_aging,
	# 																  data_type="cycle")

	# 			# combine the logs of both forwards
	# 			for idx, cycle_log in enumerate(cycle_id_logs):
	# 				id_logs[idx].update(cycle_log)
	# 			cur_loss_dict.update(cycle_loss_dict)
	# 			cur_loss_dict["loss"] = cur_loss_dict["loss_real"] + cur_loss_dict["loss_cycle"]

	# 		agg_loss_dict.append(cur_loss_dict)

	# 		# Logging related
	# 		self.parse_and_log_images(id_logs, x, y, y_hat, y_recovered, title='images/test/faces',
	# 								  subscript='{:04d}'.format(batch_idx))

	# 		# For first step just do sanity test on small amount of data
	# 		if self.global_step == 0 and batch_idx >= 4:
	# 			self.net.train()
	# 			return None  # Do not log, inaccurate in first batch

	# 	loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
	# 	# self.log_metrics(loss_dict, prefix='test')
	# 	# self.print_metrics(loss_dict, prefix='test')

	# 	self.net.train()
	# 	return loss_dict

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
		params = []
		if self.opts.train_encoder:
			params = list(self.net.encoder.parameters())
			for param in self.net.encoder.parameters():
				param.requires_grad = True
		if self.opts.train_decoder:
			params += list(self.net.decoder.parameters())
			for param in self.net.decoder.parameters():
				param.requires_grad = True
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def configure_optimizers_blender(self):
		params = list(self.net.blender.parameters())
		for param in self.net.blender.parameters():
			param.requires_grad = True
		# if self.opts.optim_name == 'adam':
		# 	optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		# else:
		# 	optimizer = Ranger(params, lr=self.opts.learning_rate)
		# todo
		optimizer = Ranger(params, lr=0.001)
		return optimizer
	
	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
		print(f'Loading dataset for {self.opts.dataset_type}')
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		if hasattr(self.opts, 'train_dataset') and os.path.isdir(self.opts.train_dataset):
			dataset_args['train_source_root'] = self.opts.train_dataset
			dataset_args['train_target_root'] = self.opts.train_dataset
			print(f'Overwritting train dataset to {self.opts.train_dataset}')
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
		# loss = 0.0
		loss_dict = {}
		id_logs = []
		desc = self.opts.exp_dir.split('/')[-2] # do not include 00000x in the path
		debug_dir_exp = 'debug/'
		debug_dir_exp = os.path.join(debug_dir_exp, desc)
		os.makedirs(debug_dir_exp, exist_ok=True)
		if self.global_step == 0:
			print('saving debug results to:', debug_dir_exp)

		if self.opts.train_encoder:
			# original sam loss
			if self.opts.id_lambda > 0:
				weights = None
				if self.opts.use_weighted_id_loss:  # compute weighted id loss only on forward pass
					age_diffs = torch.abs(target_ages - input_ages)
					weights = train_utils.compute_cosine_weights(x=age_diffs)
				loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x, label=data_type, weights=weights)
				loss_dict[f'loss_id_{data_type}'] = float(loss_id)
				loss_dict[f'id_improve_{data_type}'] = float(sim_improvement)
				loss = loss_id * self.opts.id_lambda
				# Optional margin-based identity hinge: encourage cos(y_hat,y) >= m
				if getattr(self.opts, 'id_margin_enabled', False) and float(getattr(self.opts, 'id_margin_lambda', 0.0) or 0.0) > 0:
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
				if data_type == "real":
					l2_lambda = self.opts.l2_lambda_aging
				else:
					l2_lambda = self.opts.l2_lambda
				loss += loss_l2 * l2_lambda
			if self.opts.lpips_lambda > 0:
				loss_lpips = self.lpips_loss(y_hat, y)
				loss_dict[f'loss_lpips_{data_type}'] = float(loss_lpips)
				if data_type == "real":
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
			if self.opts.w_norm_lambda > 0:
				loss_w_norm = self.w_norm_loss(latent, latent_avg=self.net.latent_avg)
				loss_dict[f'loss_w_norm_{data_type}'] = float(loss_w_norm)
				loss += loss_w_norm * self.opts.w_norm_lambda
			if self.opts.aging_lambda > 0:
				# todo
				# ? omit aging loss during no_aging stage? we already have the ground truth age?
				aging_loss, id_logs = self.aging_loss(y_hat, y, target_ages, id_logs, label=data_type)
				loss_dict[f'loss_aging_{data_type}'] = float(aging_loss)
				loss += aging_loss * self.opts.aging_lambda

			# add personal aging loss
			if (not no_aging) and self.interpolation:
				# id loss - output vs. self.feats_dict to find the most similar face from the training set
				reconstructed_face_feats = self.id_loss.extract_feats(y_hat) # [b, 512]
				max_sims = []
				# # todo: reconsider age_diffs between target_ages(y_hat) and training set(self.feats_dict)
				# age_diffs = torch.abs(target_ages - )
				# weights = train_utils.compute_cosine_weights(x=age_diffs)
				# weights = torch.from_numpy(weights).to(self.device)

				# iterate over the batch
				for i, reconstructed_face_feat in enumerate(reconstructed_face_feats):
					# # find the most similar face in the training set using cosine similarity
					# sims = []
					# for feat in self.feats_dict.values():
					# 	sim = F.cosine_similarity(reconstructed_face_feat, feat)
					# 	sims.append(sim)
					# sims = torch.stack(sims)

					# find the most similar face in the training set using cosine similarity
					# but only for the same age
					sims = []
					# input_age = input_ages[i].item()
					# find closest age in the training set
					closest_age = min(self.feats_dict.keys(), key=lambda x: abs(x - target_ages[i].item()))

					# # iterate over the faces with the closest age
					# reference_feats = torch.stack([feat[0] for feat in self.feats_dict[closest_age]])
					# iterate over closest age +- 0.03
					reference_feats = torch.stack([feat[0] for k in self.feats_dict.keys() if abs(k - closest_age) <= 0.03 for feat in self.feats_dict[k]])
					reference_paths = [feat[1] for k in self.feats_dict.keys() if abs(k - closest_age) <= 0.03 for feat in self.feats_dict[k]]
					for feat in reference_feats:
						sim = F.cosine_similarity(reconstructed_face_feat, feat)
						sims.append(sim)
					sims = torch.stack(sims)

					# sanity check: save all images
					if self.global_step % 100 == 0 and i == 0:
						max_sim_idx = torch.argmax(sims)
						max_sim = sims[max_sim_idx]
						debug_dir = os.path.join(debug_dir_exp, 'encoder_real_aging') if data_type == "real" else os.path.join(debug_dir_exp, 'encoder_cycle_aging')
						os.makedirs(debug_dir, exist_ok=True)

						# save input face
						input_face = common.tensor2im(x[i])
						input_face.save(os.path.join(debug_dir, f'{self.global_step}_{i}_input_face_input_age_{input_ages[i]}.png'))
						# save reconstructed face
						reconstructed = common.tensor2im(y_hat[i])
						reconstructed.save(os.path.join(debug_dir, f'{self.global_step}_{i}_reconstructed_face_target_age_{target_ages[i]}.png'))
						# save the most similar face
						# most_similar_face_path = list(self.feats_dict.keys())[max_sim_idx]
						# most_similar_face_path = self.feats_dict[closest_age][max_sim_idx][1]
						most_similar_face_path = reference_paths[max_sim_idx]
						from PIL import Image
						most_similar_face = Image.open(most_similar_face_path)
						reference_age = os.path.basename(most_similar_face_path).split('.')[0].split('_')[0].split(' ')[0]
						most_similar_face.save(os.path.join(debug_dir, f'{self.global_step}_{i}_most_similar_face_ref_age_{reference_age}_sim_{max_sim.item():.4f}.png'))

					max_sim = torch.max(sims)
					max_sims.append(max_sim)
				max_sims = torch.stack(max_sims)


				# todo: add lambda weights
				# loss += torch.mean(1 - max_sims)
				# loss_dict[f'nearest_neighbor_id_loss'] = float(loss)

				nearest_neighbor_id_loss = torch.mean(1 - max_sims)
				# todo: add lambda weights
				loss += nearest_neighbor_id_loss
				loss_dict[f'nearest_neighbor_id_loss'] = float(nearest_neighbor_id_loss)

			# ------------------- extrapolation stage - regularization ------------------- #
			# if (not no_aging) and (not self.interpolation):
			# 	x_input = [self.age_transformer_extrapolation(img.cpu()).to(self.device) for img in x]
			# 	x_input = torch.stack(x_input)
			# 	y_global, latent_global = self.net_global.forward(x_input, return_styles=True)
			# 	y_local, latent_local = self.net.forward(x_input, return_styles=True)
			# 	# lpips and l2 loss
			# 	# todo: add dynamic weights
			#	# age_diffs = torch.abs(target_ages - input_ages)
			#	# weights = train_utils.compute_cosine_weights(x=age_diffs)
			# 	if self.opts.l2_lambda > 0:
			# 		loss_l2 = F.mse_loss(y_local, y_global)
			# 		l2_lambda = self.opts.l2_lambda
			# 		extrapolation_loss = loss_l2 * l2_lambda
			# 	if self.opts.lpips_lambda > 0:
			# 		loss_lpips = self.lpips_loss(y_local, y_global)
			# 		lpips_lambda = self.opts.lpips_lambda
			# 		extrapolation_loss += loss_lpips * lpips_lambda
			# 	if self.opts.lpips_lambda_crop > 0:
			# 		loss_lpips_crop = self.lpips_loss(y_local[:, :, 35:223, 32:220], y_global[:, :, 35:223, 32:220])
			# 		extrapolation_loss += loss_lpips_crop * self.opts.lpips_lambda_crop
			# 	if self.opts.l2_lambda_crop > 0:
			# 		loss_l2_crop = F.mse_loss(y_local[:, :, 35:223, 32:220], y_global[:, :, 35:223, 32:220])
			# 		extrapolation_loss += loss_l2_crop * self.opts.l2_lambda_crop
			# 	loss += extrapolation_loss * lambda_recon
			# 	loss_dict["extrapolation_loss"] = float(loss)

			# 	# sanity check: save all images
			# 	if self.global_step % 100 == 0:
			# 		debug_dir = os.path.join(debug_dir_exp, 'extrapolation')
			# 		os.makedirs(debug_dir, exist_ok=True)
			# 		target_ages = x_input[:, -1, 0, 0]
			# 		target_age = target_ages[0].item()
			# 		# # save input face
			# 		# input_face = common.tensor2im(x[0])
			# 		# input_face.save(os.path.join(debug_dir, f'{self.global_step}_input_face.png'))
			# 		# save reconstructed face
			# 		reconstructed = common.tensor2im(y_local[0])
			# 		reconstructed.save(os.path.join(debug_dir, f'{self.global_step}_{target_age:.4f}_reconstructed_face.png'))
			# 		# save the most similar face
			# 		reconstructed_global = common.tensor2im(y_global[0])
			# 		reconstructed_global.save(os.path.join(debug_dir, f'{self.global_step}_{target_age:.4f}_reconstructed_face_global.png'))

			loss_dict[f'loss_{data_type}'] = float(loss)
			if data_type == "cycle":
				loss = loss * self.opts.cycle_lambda

			if self.opts.train_decoder:
				# do not return here, continue to calculate the loss for the decoder
				pass
			else:
				return loss, loss_dict, id_logs

		if self.opts.train_decoder:
			# if not no_aging:
			# 	# id loss - output vs. self.feats_dict to find the most similar face from the training set
			# 	reconstructed_face_feats = self.id_loss.extract_feats(y_hat) # [b, 512]
			# 	max_sims = []
			# 	# # todo: reconsider age_diffs between target_ages(y_hat) and training set(self.feats_dict)
			# 	# age_diffs = torch.abs(target_ages - )
			# 	# weights = train_utils.compute_cosine_weights(x=age_diffs)
			# 	# weights = torch.from_numpy(weights).to(self.device)

			# 	# iterate over the batch
			# 	for i, reconstructed_face_feat in enumerate(reconstructed_face_feats):
			# 		# # find the most similar face in the training set using cosine similarity
			# 		# sims = []
			# 		# for feat in self.feats_dict.values():
			# 		# 	sim = F.cosine_similarity(reconstructed_face_feat, feat)
			# 		# 	sims.append(sim)
			# 		# sims = torch.stack(sims)

			# 		# find the most similar face in the training set using cosine similarity
			# 		# but only for the same age
			# 		sims = []
			# 		# input_age = input_ages[i].item()
			# 		# find closest age in the training set
			# 		closest_age = min(self.feats_dict.keys(), key=lambda x: abs(x - target_ages[i].item()))

			# 		# # iterate over the faces with the closest age
			# 		# reference_feats = torch.stack([feat[0] for feat in self.feats_dict[closest_age]])
			# 		# iterate over closest age +- 0.03
			# 		reference_feats = torch.stack([feat[0] for k in self.feats_dict.keys() if abs(k - closest_age) <= 0.03 for feat in self.feats_dict[k]])
			# 		reference_paths = [feat[1] for k in self.feats_dict.keys() if abs(k - closest_age) <= 0.03 for feat in self.feats_dict[k]]
			# 		for feat in reference_feats:
			# 			sim = F.cosine_similarity(reconstructed_face_feat, feat)
			# 			sims.append(sim)
			# 		sims = torch.stack(sims)

			# 		# sanity check: save all images
			# 		if self.global_step % 100 == 0 and i == 0:
			# 			max_sim_idx = torch.argmax(sims)
			# 			max_sim = sims[max_sim_idx]

			# 			debug_dir = os.path.join(debug_dir_exp, 'aging')
			# 			os.makedirs(debug_dir, exist_ok=True)

			# 			# save input face
			# 			input_face = common.tensor2im(x[i])
			# 			input_face.save(os.path.join(debug_dir, f'{self.global_step}_{i}_input_face_input_age_{input_ages[i]}.png'))
			# 			# save reconstructed face
			# 			reconstructed = common.tensor2im(y_hat[i])
			# 			reconstructed.save(os.path.join(debug_dir, f'{self.global_step}_{i}_reconstructed_face_target_age_{target_ages[i]}.png'))
			# 			# save the most similar face
			# 			# most_similar_face_path = list(self.feats_dict.keys())[max_sim_idx]
			# 			# most_similar_face_path = self.feats_dict[closest_age][max_sim_idx][1]
			# 			most_similar_face_path = reference_paths[max_sim_idx]
			# 			from PIL import Image
			# 			most_similar_face = Image.open(most_similar_face_path)
			# 			reference_age = os.path.basename(most_similar_face_path).split('.')[0].split('_')[0].split(' ')[0]
			# 			most_similar_face.save(os.path.join(debug_dir, f'{self.global_step}_{i}_most_similar_face_ref_age_{reference_age}_sim_{max_sim.item():.4f}.png'))

			# 		max_sim = torch.max(sims)
			# 		max_sims.append(max_sim)
			# 	max_sims = torch.stack(max_sims)
			# 	# loss = torch.mean((1 - max_sims) * weights)
			# 	loss = torch.mean(1 - max_sims)
			# 	loss_dict[f'nearest_neighbor_id_loss'] = float(loss)

			# if no_aging:
			# l2 loss
			loss_l2 = F.mse_loss(y_hat, y)
			loss_dict[f'l2'] = loss_l2.item()

			# lpips loss
			loss_lpips = self.lpips_loss(y_hat, y)
			loss_dict[f'lpips'] = loss_lpips.item()

			if self.opts.train_encoder:
				loss += loss_l2 + loss_lpips
			else:
				loss = loss_l2 + loss_lpips

			# sanity check: save all images
			if self.global_step % 100 == 0:

				debug_dir = os.path.join(debug_dir_exp, 'decoder_reconstruction')
				os.makedirs(debug_dir, exist_ok=True)
				
				input_face = common.tensor2im(y[0])
				input_face.save(os.path.join(debug_dir, f'{self.global_step}_input_face_input_age_{input_ages[0]}.png'))
				reconstructed = common.tensor2im(y_hat[0])
				reconstructed.save(os.path.join(debug_dir, f'{self.global_step}_reconstructed_face.png'))

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
		for i in range(display_count):
			cur_im_data = {
				'input_face': common.tensor2im(x[i]),
				'target_face': common.tensor2im(y[i]),
				'output_face': common.tensor2im(y_hat[i]),
				'recovered_face': common.tensor2im(y_recovered[i])
			}
			if id_logs is not None:
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
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
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			print("Optimizer state loaded successfully")
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
