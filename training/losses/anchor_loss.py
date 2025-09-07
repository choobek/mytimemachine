import torch
import torch.nn as nn


class AgeAnchorLoss(nn.Module):
    """
    Pulls the model's current pre-generator latent (W mean) toward the nearest
    age-bin anchor. Operates in W space.

    anchors_w: Tensor [Bbins, 512]
    bin_mids: list[int] mids like [17,22,27,...]
    bin_size: int, for info/sanity only
    """

    def __init__(self, anchors_w: torch.Tensor, bin_mids, bin_size: int = 5):
        super().__init__()
        if not torch.is_tensor(anchors_w):
            raise ValueError("anchors_w must be a torch.Tensor")
        if anchors_w.dim() != 2 or anchors_w.size(1) != 512:
            raise ValueError("anchors_w must have shape [Bbins, 512]")
        self.register_buffer("anchors", anchors_w.float())
        # store mids as Python list; tensor constructed on the fly on correct device
        self.bin_mids = list(int(x) for x in bin_mids)
        self.bin_size = int(bin_size)

    def _nearest_bin_indices(self, ages_years: torch.Tensor) -> torch.Tensor:
        """
        ages_years: [B] ages in years (float or int tensor).
        Returns indices [B] into anchors for the nearest mid.
        """
        mids = torch.tensor(self.bin_mids, device=ages_years.device, dtype=ages_years.dtype)
        # shape: [B, M]
        diffs = (ages_years.view(-1, 1) - mids.view(1, -1)).abs()
        idx = torch.argmin(diffs, dim=1)
        return idx

    def forward(self, w_mean: torch.Tensor, target_ages_years: torch.Tensor) -> torch.Tensor:
        """
        w_mean: [B, 512]
        target_ages_years: [B] in years (0..100)
        """
        if w_mean.dim() != 2 or w_mean.size(1) != 512:
            raise ValueError("w_mean must have shape [B, 512]")
        if target_ages_years.dim() != 1 or target_ages_years.size(0) != w_mean.size(0):
            raise ValueError("target_ages_years must be [B]")
        idx = self._nearest_bin_indices(target_ages_years)
        anchors = self.anchors.index_select(0, idx)
        diff = (w_mean - anchors)
        # Mean squared error per-sample, then mean over batch
        return diff.pow(2).mean(dim=1).mean()


