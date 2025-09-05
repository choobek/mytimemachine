from copy import deepcopy
import torch


class EMAHelper:
    def __init__(self, module, decay: float):
        self.decay = float(decay)
        # torch.nn.Module.state_dict includes buffers; shadow tracks all
        self.shadow = {k: v.detach().clone() for k, v in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module):
        sd = module.state_dict()
        for k, v in sd.items():
            if k not in self.shadow:
                # Create new entry if module gained a new param/buffer
                self.shadow[k] = v.detach().clone()
            else:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, module):
        module.load_state_dict(self.shadow, strict=True)

    def state_dict(self):
        return {"decay": float(self.decay), "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = float(state["decay"]) if "decay" in state else float(self.decay)
        self.shadow = state["shadow"]


