import torch
from torch import nn


def relu(x, alpha=0.0):
    return torch.where(x > 0, x, 0)


def bsilu(x, alpha=1.67):
    return (x + alpha) * torch.sigmoid(x) - (alpha / 2.0)


def leaky(x, alpha=0.01):
    return torch.where(x > 0, x, alpha * x)


def nelu(x, alpha=0.0):
    return torch.where(x > 0, x, -alpha * torch.reciprocal(1 + x.square()))


class ReLUSurrogate(nn.Module):
    def __init__(self, surrogate, alpha):
        super().__init__()
        self.surrogate = {"relu": relu, "bsilu": bsilu, "leaky": leaky, "nelu": nelu}[
            surrogate
        ]
        self.alpha = alpha

    def forward(self, x):
        surrogate = self.surrogate(x, self.alpha)
        return surrogate - surrogate.detach() + torch.relu(x).detach()


def replace_activations(model, surrogate, alpha):
    for module in model.children():
        if hasattr(module, "act") and isinstance(module.act, nn.Module):
            if not isinstance(module.act, ReLUSurrogate):
                module.act = ReLUSurrogate(surrogate, alpha)
        replace_activations(module, surrogate, alpha)
    return model
