from typing import List, Tuple, Optional

from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from utils import data_utils


class BinaryIdentityDataset(Dataset):

	def __init__(self,
				 actor_root: str,
				 non_actor_root: str,
				 transform: Optional[transforms.Compose] = None):
		"""
		A simple binary classification dataset for identity discrimination.

		- All images under `actor_root` are labeled 1 (positive class).
		- All images under `non_actor_root` (including all subfolders) are labeled 0 (negative class).
		"""
		self.transform = transform
		actor_paths: List[str] = data_utils.make_dataset(actor_root)
		non_actor_paths: List[str] = data_utils.make_dataset(non_actor_root)
		# Order does not matter; keep stable ordering for reproducibility
		self.samples: List[Tuple[str, int]] = [(p, 1) for p in actor_paths] + [(p, 0) for p in non_actor_paths]

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int):
		path, label = self.samples[index]
		img = Image.open(path).convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		return img, label


def build_default_transform(input_size: int = 112,
							 mean = [0.5, 0.5, 0.5],
							 std = [0.5, 0.5, 0.5]) -> transforms.Compose:
	"""
	Default preprocessing:
	- Resize to input_size x input_size (ArcFace IR-50 uses 112x112)
	- Convert to tensor
	- Normalize to mean/std provided (default 0.5)
	"""
	return transforms.Compose([
		transforms.Resize((input_size, input_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean, std),
	])


def get_binary_identity_dataloader(actor_root: str = 'data/all_130925',
								 	 non_actor_root: str = 'data/images1024x1024',
								 	 batch_size: int = 32,
								 	 num_workers: int = 4,
								 	 shuffle: bool = True,
								 	 input_size: int = 112,
								 	 mean = [0.5, 0.5, 0.5],
								 	 std = [0.5, 0.5, 0.5]):
	"""
	Create a DataLoader for the binary identity classifier.
	Returns (dataloader, dataset).
	"""
	transform = build_default_transform(input_size=input_size, mean=mean, std=std)
	dataset = BinaryIdentityDataset(
		actor_root=actor_root,
		non_actor_root=non_actor_root,
		transform=transform,
	)
	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=True,
		drop_last=False,
	)
	return dataloader, dataset


