from typing import Literal, Optional
from typing import List, Dict

import torch
import torch.nn as nn

try:
	from facenet_pytorch import InceptionResnetV1  # type: ignore
	facenet_available = True
except Exception:
	facenet_available = False

from models.encoders.model_irse import Backbone


class IdentityClassifier(nn.Module):

	def __init__(self, feature_extractor: nn.Module, num_outputs: int = 2):
		super().__init__()
		self.feature_extractor = feature_extractor
		self.classifier = nn.Linear(512, num_outputs)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		features = self.feature_extractor(x)
		logits = self.classifier(features)
		return logits


def load_arcface_ir_se50_backbone(weights_path: str,
								 input_size: int = 112,
								 mode: Literal['ir', 'ir_se'] = 'ir_se') -> nn.Module:
	"""
	Load ArcFace IR/IR-SE backbone (feature extractor, 512-dim output) and load weights.
	Removes any old classification head implicitly (Backbone outputs 512 normalized features).
	"""
	backbone = Backbone(input_size=input_size, num_layers=50, mode=mode)
	state = torch.load(weights_path, map_location='cpu')
	# Some checkpoints store under 'state_dict'
	if isinstance(state, dict) and 'state_dict' in state:
		state = state['state_dict']
	# Filter keys with shape mismatch to allow loading weights trained with different input_size
	model_state = backbone.state_dict()
	filtered = {}
	for k, v in state.items():
		if k in model_state and model_state[k].shape == v.shape:
			filtered[k] = v
	# Load partial state dict
	backbone.load_state_dict(filtered, strict=False)
	return backbone


def build_identity_model(
		backend: Literal['arcface', 'resnet50', 'facenet'] = 'arcface',
		weights_path: Optional[str] = 'pretrained_models/model_ir_se50.pth',
		num_outputs: int = 2,
		input_size: int = 112) -> nn.Module:
	"""
	Build an identity classifier with a new binary/2-class head.
	- 'arcface': load IR-SE50 backbone from weights, add Linear(512, num_outputs)
	- 'resnet50': torchvision resnet50 pretrained=True, replace fc with Linear(2048, num_outputs)
	- 'facenet': InceptionResnetV1(pretrained='vggface2'), replace last linear with Linear(512, num_outputs)
	"""
	if backend == 'arcface':
		if weights_path is not None:
			backbone = load_arcface_ir_se50_backbone(weights_path=weights_path, input_size=input_size, mode='ir_se')
		else:
			# Construct backbone with random init; intended to be loaded from a full classifier checkpoint later
			backbone = Backbone(input_size=input_size, num_layers=50, mode='ir_se')
		model = IdentityClassifier(feature_extractor=backbone, num_outputs=num_outputs)
		return model
	elif backend == 'resnet50':
		from torchvision import models
		m = models.resnet50(pretrained=True)
		in_features = m.fc.in_features
		m.fc = nn.Linear(in_features, num_outputs)
		return m
	elif backend == 'facenet':
		if not facenet_available:
			raise RuntimeError('facenet_pytorch not installed; choose arcface or resnet50')
		m = InceptionResnetV1(pretrained='vggface2')
		# InceptionResnetV1 ends with last_linear (512 -> 512) + last_bn + logit layer sometimes omitted;
		# We'll attach our own classifier on top of 512-d features by replacing the logit layer if present
		# Ensure consistent output by using the built-in 'classify' property if available
		if hasattr(m, 'logits') and isinstance(m.logits, nn.Linear):
			in_features = m.logits.in_features
			m.logits = nn.Linear(in_features, num_outputs)
		else:
			# Fallback: append head after forward features
			head = nn.Linear(512, num_outputs)
			class FacenetWrapper(nn.Module):
				def __init__(self, base, head):
					super().__init__()
					self.base = base
					self.head = head
				def forward(self, x):
					feat = self.base(x)
					return self.head(feat)
			m = FacenetWrapper(m, head)
		return m
	else:
		raise ValueError(f'Unknown backend: {backend}')



class IdentityModelWrapper(nn.Module):
	"""
	Thin, non-intrusive wrapper around a frozen binary identity classifier.
	Provides single- and multi-view inference utilities and returns logits/probs.
	"""

	def __init__(self, classifier: nn.Module):
		super().__init__()
		self.classifier = classifier.eval()
		for p in self.classifier.parameters():
			p.requires_grad = False

	def forward_single(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
		"""
		Run the wrapped classifier on a single tensor batch.
		Returns dict with 'logits' and 'probs' (softmax over dim=1).
		"""
		logits = self.classifier(x)
		probs = torch.softmax(logits, dim=1)
		return {"logits": logits, "probs": probs}

	def forward_multi(self, views: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
		"""
		Run the wrapped classifier on multiple view tensors (e.g., TTA variants).
		Each view must be a tensor of shape [B, C, H, W].
		Returns dict {'logits': [tensors], 'probs': [tensors]} aligned per view.
		"""
		logits_list: List[torch.Tensor] = []
		probs_list: List[torch.Tensor] = []
		for v in views:
			logits = self.classifier(v)
			probs = torch.softmax(logits, dim=1)
			logits_list.append(logits)
			probs_list.append(probs)
		return {"logits": logits_list, "probs": probs_list}


def tta_flip_horizontal(x: torch.Tensor) -> torch.Tensor:
	"""
	Differentiable horizontal flip. Expects [-1,1] normalized float tensor [B,C,H,W].
	"""
	return torch.flip(x, dims=[-1])


def _avgpool_reencode_like(x: torch.Tensor, scale: int = 2) -> torch.Tensor:
	"""
	Differentiable approximation of JPEG compression via downsample+upsample.
	Scale=2 roughly imitates JPEG(≈75). Keeps dtype/device and gradients.
	"""
	if scale <= 1:
		return x
	B, C, H, W = x.shape
	fh = max(1, H // scale)
	fw = max(1, W // scale)
	# Bilinear down-up preserves gradients
	low = torch.nn.functional.interpolate(x, size=(fh, fw), mode='bilinear', align_corners=False)
	up = torch.nn.functional.interpolate(low, size=(H, W), mode='bilinear', align_corners=False)
	return up


def tta_jpeg75(x: torch.Tensor) -> torch.Tensor:
	"""
	Differentiable JPEG(≈75) approximation using down-up sampling.
	"""
	return _avgpool_reencode_like(x, scale=2)


def _gaussian_kernel1d(sigma: float, radius: int, device, dtype):
	x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
	weight = torch.exp(-(x ** 2) / (2 * (sigma ** 2) + 1e-12))
	weight = weight / weight.sum()
	return weight


def tta_gaussian_blur(x: torch.Tensor, sigma: float = 0.6) -> torch.Tensor:
	"""
	Differentiable Gaussian blur via separable conv. sigma≈0.6 ~ mild blur.
	"""
	# Choose radius ~ 3*sigma
	radius = max(1, int(3.0 * float(sigma)))
	weight = _gaussian_kernel1d(float(sigma), radius, x.device, x.dtype)
	# Create depthwise conv filters
	C = x.shape[1]
	ker_h = weight.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
	ker_w = weight.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
	pad_h = (radius, radius)
	pad_w = (radius, radius)
	y = torch.nn.functional.conv2d(x, ker_h, padding=(pad_h[0], 0), groups=C)
	y = torch.nn.functional.conv2d(y, ker_w, padding=(0, pad_w[0]), groups=C)
	return y

