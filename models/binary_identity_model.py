from typing import Literal, Optional

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
		assert weights_path is not None, 'ArcFace backend requires weights_path'
		backbone = load_arcface_ir_se50_backbone(weights_path=weights_path, input_size=input_size, mode='ir_se')
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


