from functools import partial

import torch
from torch._dynamo.eval_frame import OptimizedModule
from torch.nn import ReLU


class ReLUSurrogate(torch.nn.Module):
    def __init__(self, surrogate, *args):
        super().__init__()
        self.surrogate = surrogate
        self.args = args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = self.surrogate(x, *self.args)
        fx = torch.nn.functional.relu(x)
        return gx - gx.detach() + fx.detach()


def replace_layers(model, old, new_fn):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers(module, old, new_fn)
        if isinstance(module, old):
            setattr(model, n, new_fn())


def make_surrogate_factory(surrogate, *args, compile_layer=True):
    holder = {}

    def create_layer():
        layer = ReLUSurrogate(surrogate, *args)
        if compile_layer:
            layer = torch.compile(layer)
            holder["cls"] = OptimizedModule
        else:
            holder["cls"] = ReLUSurrogate
        return layer

    create_layer.layer_type = lambda: holder["cls"]
    return create_layer


def bounded_silu(x):
    lower_bound = 1.67
    sig_x = torch.sigmoid(x)
    f_x = x * sig_x - lower_bound * (1 - sig_x)
    c = -lower_bound * (1 - torch.sigmoid(torch.tensor(0.0)))
    return f_x - c


def nelu(x: torch.Tensor, alpha: float = 0.05):
    return torch.where(x > 0, x, -alpha * torch.reciprocal(1 + x.square()))


def activation_function(activation, alpha=0.1):
    functions = {
        "relu": torch.nn.functional.relu,
        "none": torch.nn.functional.relu,
        "NONE": torch.nn.functional.relu,
        "leaky": torch.nn.functional.leaky_relu,
        "leakyrelu": torch.nn.functional.leaky_relu,
        "elu": torch.nn.functional.elu,
        "celu": torch.nn.functional.celu,
        "gelu": torch.nn.functional.gelu,
        "selu": torch.nn.functional.selu,
        "mish": torch.nn.functional.mish,
        "swish": torch.nn.functional.silu,
        "param_swish": bounded_silu,
        "nelu": partial(nelu, alpha=alpha),
    }
    if activation not in functions:
        raise ValueError(f"Unknown activation: {activation}")
    return functions[activation]


def apply_sugar(model, activation="relu", alpha=0.1, compile_layer=True):
    if activation in ("NONE", "relu", "none"):
        return (model, torch.nn.ReLU)
    surrogate = activation_function(activation, alpha)
    factory = make_surrogate_factory(surrogate, compile_layer=compile_layer)
    replace_layers(model, ReLU, factory)
    layer_type = factory.layer_type()
    return (model, layer_type)
