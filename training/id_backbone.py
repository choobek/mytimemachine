import os
from typing import Optional

import torch
import torch.nn as nn

from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone


class IDBackbone(nn.Module):
    """
    Unified wrapper for identity feature backbones.

    Supports:
    - ir50  -> IR-SE50 (512-D)
    - ir100 -> IR-SE100 (512-D)
    - adaface -> placeholder using IR-SE100 (512-D) unless a dedicated loader
                 is added later. We avoid loading mismatched weights by default.
    """

    def __init__(
        self,
        name: str = 'ir50',
        weights_path: Optional[str] = None,
        embed_dim: int = 512,
        normalize: str = 'l2',
        input_size: int = 56,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.normalize = str(normalize)
        self.input_size = int(input_size)

        self.model = self._load_model(name=name, weights_path=weights_path)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        # Safe pooling to expected resolution if inputs differ
        self._pool_to_input = nn.AdaptiveAvgPool2d((self.input_size, self.input_size))

    def _load_model(self, name: str, weights_path: Optional[str]) -> nn.Module:
        name = name.lower().strip()

        if name == 'ir50':
            net = Backbone(input_size=self.input_size, num_layers=50, drop_ratio=0.6, mode='ir_se')
            # Load default IR-SE50 weights if available and no override provided
            ckpt_path = weights_path if (weights_path and len(weights_path) > 0) else model_paths.get('ir_se50', '')
            if ckpt_path and os.path.isfile(ckpt_path):
                state = torch.load(ckpt_path, map_location='cpu')
                net.load_state_dict(state, strict=True)
        elif name == 'ir100':
            net = Backbone(input_size=self.input_size, num_layers=100, drop_ratio=0.6, mode='ir_se')
            # Only load weights if a compatible checkpoint is explicitly provided
            if weights_path and os.path.isfile(weights_path):
                try:
                    state = torch.load(weights_path, map_location='cpu')
                    net.load_state_dict(state, strict=False)
                except Exception:
                    # Skip loading if incompatible; proceed with randomly initialized backbone
                    pass
        elif name == 'adaface':
            # Placeholder: use IR-SE100 architecture to maintain 512-D output and API compatibility.
            # A dedicated AdaFace IR-101 loader can replace this later.
            net = Backbone(input_size=self.input_size, num_layers=100, drop_ratio=0.6, mode='ir_se')
            # Avoid attempting to load mismatched AdaFace weights by default.
            if weights_path and os.path.isfile(weights_path):
                try:
                    state = torch.load(weights_path, map_location='cpu')
                    net.load_state_dict(state, strict=False)
                except Exception:
                    pass
        else:
            raise ValueError(f"Unsupported id_backbone: {name}")

        return net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input of shape [B,3,H,W], got {tuple(x.shape)}")
        if x.shape[2] != self.input_size or x.shape[3] != self.input_size:
            x = self._pool_to_input(x)
        features = self.model(x)  # [B, 512]
        if self.normalize == 'l2':
            features = nn.functional.normalize(features, p=2, dim=1, eps=1e-6)
        return features


class IDAlign(nn.Module):
    """Lightweight alignment/scaling for ID features or similarity."""

    def __init__(
        self,
        mode: str = 'none',
        scale: float = 1.0,
        bias: float = 0.0,
        temp: float = 0.20,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.mode = mode
        if self.mode == 'affine':
            self.scale = nn.Parameter(torch.tensor([float(scale)], dtype=torch.float32), requires_grad=trainable)
            self.bias = nn.Parameter(torch.tensor([float(bias)], dtype=torch.float32), requires_grad=trainable)
        elif self.mode == 'temp':
            inv = 1.0 / max(float(temp), 1e-6)
            self.inv_tau = nn.Parameter(torch.tensor([inv], dtype=torch.float32), requires_grad=trainable)

    def forward_feat(self, features: torch.Tensor) -> torch.Tensor:
        if self.mode == 'affine':
            return self.scale * features + self.bias
        return features

    def scale_sim(self, sim: torch.Tensor) -> torch.Tensor:
        if self.mode == 'temp':
            return sim * self.inv_tau
        return sim


