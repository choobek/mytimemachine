import torch
import torch.nn as nn


IDX = dict(
    L_eye=(36, 37, 38, 39, 40, 41),
    R_eye=(42, 43, 44, 45, 46, 47),
    mouth=(48, 54, 62, 66),
    nose=(27, 31, 33, 35),
)


def eye_center(pts6: torch.Tensor) -> torch.Tensor:
    return pts6.mean(dim=1)


def dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(a - b, dim=-1)


def interocular(land: torch.Tensor) -> torch.Tensor:
    L = land[:, list(IDX['L_eye'])]
    R = land[:, list(IDX['R_eye'])]
    return dist(eye_center(L), eye_center(R)).clamp_min(1e-6)


def eye_width(land: torch.Tensor, left: bool = True) -> torch.Tensor:
    i = IDX['L_eye'] if left else IDX['R_eye']
    a = land[:, i[0]]
    b = land[:, i[3]]
    return dist(a, b)


def ear(land: torch.Tensor, left: bool = True) -> torch.Tensor:
    i = IDX['L_eye'] if left else IDX['R_eye']
    a = dist(land[:, i[1]], land[:, i[5]])
    b = dist(land[:, i[2]], land[:, i[4]])
    w = eye_width(land, left)
    return ((a + b) * 0.5) / w.clamp_min(1e-6)


def nose_width(land: torch.Tensor) -> torch.Tensor:
    return dist(land[:, IDX['nose'][3]], land[:, IDX['nose'][1]])


def nose_len(land: torch.Tensor) -> torch.Tensor:
    return dist(land[:, IDX['nose'][0]], land[:, IDX['nose'][2]])


def mouth_width(land: torch.Tensor) -> torch.Tensor:
    return dist(land[:, IDX['mouth'][1]], land[:, IDX['mouth'][0]])


def mar(land: torch.Tensor) -> torch.Tensor:
    return dist(land[:, IDX['mouth'][2]], land[:, IDX['mouth'][3]]) / mouth_width(land).clamp_min(1e-6)


def ratios(land: torch.Tensor) -> torch.Tensor:
    ipd = interocular(land).unsqueeze(-1)
    g = torch.stack([
        eye_width(land, True) / ipd.squeeze(-1),
        eye_width(land, False) / ipd.squeeze(-1),
        ear(land, True),
        ear(land, False),
        nose_width(land) / ipd.squeeze(-1),
        nose_len(land) / ipd.squeeze(-1),
        mouth_width(land) / ipd.squeeze(-1),
        mar(land),
    ], dim=-1)
    return g


class HuberLoss(nn.Module):
    def __init__(self, delta: float = 0.03):
        super().__init__()
        self.delta = float(delta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        absx = x.abs()
        quad = 0.5 * (x ** 2)
        lin = self.delta * (absx - 0.5 * self.delta)
        return torch.where(absx <= self.delta, quad, lin)


class GeometryLoss(nn.Module):
    def __init__(self, parts=('eyes', 'nose', 'mouth'), weights=(1.0, 0.6, 0.4), delta: float = 0.03):
        super().__init__()
        self.huber = HuberLoss(delta)
        idx_map = dict(eyes=[0, 1, 2, 3], nose=[4, 5], mouth=[6, 7])
        active = []
        wvec = []
        for p, w in zip(['eyes', 'nose', 'mouth'], weights):
            if p in parts:
                active += idx_map[p]
                wvec += [float(w)] * len(idx_map[p])
        if len(active) == 0:
            # Fallback to eyes to avoid empty selection
            active = idx_map['eyes']
            wvec = [float(weights[0])] * len(active)
        self.register_buffer('active_idx', torch.tensor(active, dtype=torch.long), persistent=False)
        self.register_buffer('wvec', torch.tensor(wvec, dtype=torch.float32), persistent=False)

    def forward(self, g_out: torch.Tensor, g_src: torch.Tensor) -> torch.Tensor:
        sel_out = g_out.index_select(1, self.active_idx)
        sel_src = g_src.index_select(1, self.active_idx)
        diff = sel_out - sel_src
        loss = self.huber(diff)
        weights = self.wvec.view(1, -1).to(loss.device)
        return (loss * weights).mean()


