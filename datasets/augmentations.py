import numpy as np
import torch


class AgeTransformer(object):

	def __init__(self, target_age, range=(0., 101.), jitter=0, clip_bounds=None):
		self.target_age = target_age
		self.range = range
		# Optional: jitter around numeric target age (int years)
		self.jitter = int(jitter or 0)
		# Optional: (min_age, max_age) hard clip for sampled ages
		self.clip_bounds = clip_bounds

	def __call__(self, img):
		img = self.add_aging_channel(img)
		return img

	def add_aging_channel(self, img):
		target_age = self.__get_target_age()
		target_age = int(target_age) / 100  # normalize aging amount to be in range [-1,1]
		img = torch.cat((img, target_age * torch.ones((1, img.shape[1], img.shape[2]))))
		return img

	def __get_target_age(self):
		if self.target_age == "uniform_random":
			return np.random.randint(low=0., high=101, size=1)[0]
		elif self.target_age == "interpolation":
			# Handle case where range has same min and max values
			if self.range[0] >= self.range[1]:
				return self.range[0]
			return np.random.randint(low=self.range[0], high=self.range[1], size=1)[0]
		elif self.target_age == "extrapolation":
			return np.random.choice([np.random.randint(low=0., high=self.range[0], size=1)[0],
			                         np.random.randint(low=self.range[1], high=101, size=1)[0]])
		else:
			# numeric target age with optional jitter and clipping
			try:
				base = int(self.target_age)
			except Exception:
				# Fallback: if not numeric, return as-is
				return self.target_age
			age = base
			if self.jitter > 0:
				lo, hi = base - self.jitter, base + self.jitter
				age = np.random.randint(lo, hi + 1)
			# hard clip to provided bounds or default to range
			clip_lo, clip_hi = (self.clip_bounds if self.clip_bounds is not None else self.range)
			age = int(np.clip(age, clip_lo, clip_hi))
			return age
