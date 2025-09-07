import torch
import torch.nn as nn

from training.id_backbone import IDBackbone, IDAlign


class GlobalIDLoss(nn.Module):
    def __init__(self, id_encoder: IDBackbone, align: IDAlign, loss_type: str = 'cosine', use_crop: bool = True):
        super().__init__()
        self.id_encoder = id_encoder
        self.id_align = align
        self.loss_type = str(loss_type)
        self.use_crop = bool(use_crop)
        # Preprocessors to match legacy pipeline
        self._pool256 = nn.AdaptiveAvgPool2d((256, 256))

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        # Expect [B,3,H,W] in [-1,1]
        x = images
        if x.shape[2] != 256 or x.shape[3] != 256:
            x = self._pool256(x)
        if self.use_crop:
            # Legacy crop used in criteria/IDLoss: x[:, :, 35:223, 32:220]
            x = x[:, :, 35:223, 32:220]
        return x

    def forward(self, real_images: torch.Tensor, generated_images: torch.Tensor):
        y = self._preprocess(real_images)
        y_hat = self._preprocess(generated_images)
        # Extract identity features
        f_real = self.id_encoder(y)        # [B, 512]
        f_gen = self.id_encoder(y_hat)     # [B, 512]
        # Affine feature alignment if any
        f_gen_aligned = self.id_align.forward_feat(f_gen)
        # Cosine similarity via dot since features are L2-normalized by default
        cosine_sim = torch.sum(f_real * f_gen_aligned, dim=1)  # [B]
        # Temperature scaling on similarity if any
        cosine_sim = self.id_align.scale_sim(cosine_sim)
        # Loss
        if self.loss_type == 'cosine':
            id_losses = 1.0 - cosine_sim
        else:
            id_losses = 1.0 - cosine_sim
        return id_losses.mean(), cosine_sim.mean().detach()


