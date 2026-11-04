import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Callable


# ----------------------
# SUGAR
# ----------------------

class SUGAR_NeLU(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.where(x > 0, x, -self.alpha * torch.reciprocal(1 + x.square()))
        y = gx - gx.detach() + F.relu(x).detach()
        return y

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"alpha={self.alpha})")


class SUGAR_BSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = (x + 1.67) * torch.sigmoid(x) - 0.835
        gx[x >= 6] = 0.
        return gx - gx.detach() + F.relu6(x).detach()


class SUGAR_LeakyReLU(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.where((x > 0), 1, self.alpha)
        gx[x >= 6] = 0.
        m = x * gx.detach()
        y = m - m.detach() + F.relu6(x).detach()
        return y

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"alpha={self.alpha})")


# ----------------------
# helper functions
# ----------------------
def replace_modules_(m: nn.Module,
                     old_cls: Type[nn.Module],
                     new_ctor: Callable[[], nn.Module]) -> int:
    """
    Recursively replace all modules of type `old_cls` in `m` with `new_ctor()`.
    Returns the number of replacements made.
    """
    count = 0
    for name, child in list(m.named_children()):
        # First recurse
        count += replace_modules_(child, old_cls, new_ctor)
        # Then possibly replace this child
        if isinstance(child, old_cls):
            setattr(m, name, new_ctor())
            count += 1
    return count


def list_activations(model: nn.Module) -> set:
    types = set()
    for mod in model.modules():
        if isinstance(mod, (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.Hardsigmoid,
                            SUGAR_NeLU, SUGAR_LeakyReLU, SUGAR_BSiLU)):
            types.add(type(mod).__name__)
    return types