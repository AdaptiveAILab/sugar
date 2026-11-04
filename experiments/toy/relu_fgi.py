"""
relu_fgi.py -- SUGAR (Surrogate Gradient Learning for ReLU), standalone module.

Forward-gradient injection (FGI): every `nn.ReLU` in a model is swapped for a
`ReluSurrogate` layer whose forward output is still exactly ReLU, while the
backward pass uses the gradient of a smooth surrogate `g`. Only depends on torch.

    from relu_fgi import perform_relu_fgi
    model, _ = perform_relu_fgi(model, "Custom")   # "Custom" = B-SiLU surrogate

`fgi_type="NONE"` leaves the model unchanged (plain ReLU).
"""
import torch
import torch.nn as nn

class ReluSurrogate(torch.nn.Module):
    """
    Custom surrogate ReLU layer that replaces standard ReLU activation with a differentiable function `g`.
    This allows for custom gradient behavior while preserving the original ReLU outputs.
    """

    def __init__(self, g, *args):
        super(ReluSurrogate, self).__init__()
        self.g = g  # Custom gradient function
        self.args = args  # Additional arguments for `g`

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the surrogate function for backpropagation while keeping ReLU's output for inference.
        """
        gx = self.g(x, *self.args)  # Compute the surrogate gradient function
        fx = torch.nn.functional.relu(x)  # Standard ReLU activation
        return gx - gx.detach() + fx.detach()  # Ensures correct forward output with custom gradients


def replace_layers(model, old, new_fn):
    """
    Recursively replaces all instances of `old` layer in `model` with layers created by `new_fn`.

    Args:
        model (torch.nn.Module): The model to modify.
        old (torch.nn.Module): The layer type to replace (e.g., `torch.nn.ReLU`).
        new_fn (function): A function that returns a new instance of the replacement layer.
    """
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            # If the module has children, recursively replace layers inside it
            replace_layers(module, old, new_fn)

        if isinstance(module, old):
            # Replace the module if it matches `old`
            setattr(model, n, new_fn())


def get_surrogate_layer(g, *args):
    """
    Returns a function that creates a `ReluSurrogate` layer with a given gradient function `g`.
    """

    def get_surrogate_layer_internal():
        return ReluSurrogate(g, *args)

    return get_surrogate_layer_internal


def fgi_relu(model, g, *args):
    """
    Replaces all `ReLU` activations in `model` with `ReluSurrogate` using function `g`.
    """
    replace_layers(model, torch.nn.ReLU, get_surrogate_layer(g, *args))


def perform_relu_fgi(model, fgi_type):
    """
    Applies Functionally Generated Injection (FGI) to replace ReLU layers with custom surrogates.

    Args:
        model (torch.nn.Module): The model whose ReLU layers will be modified.
        fgi_type (str): The type of surrogate function to use (default: "LeakyReLU").

    Returns:
        model (torch.nn.Module): Modified model with ReLU layers replaced by surrogates.
        layer_type (type): The new layer type used for activation.
    """
    if fgi_type == "NONE":
        layer_type = torch.nn.ReLU  # No change
    elif fgi_type == "LeakyReLU":
        fgi_relu(model, torch.nn.functional.leaky_relu)
        layer_type = ReluSurrogate
    elif fgi_type == "ELU":
        fgi_relu(model, torch.nn.functional.elu)
        layer_type = ReluSurrogate
    elif fgi_type == "CeLU":
        fgi_relu(model, torch.nn.functional.celu)
        layer_type = ReluSurrogate
    elif fgi_type == "GeLU":
        fgi_relu(model, torch.nn.functional.gelu)
        layer_type = ReluSurrogate
    elif fgi_type == "SeLU":
        fgi_relu(model, torch.nn.functional.selu)
        layer_type = ReluSurrogate
    elif fgi_type == "Mish":
        fgi_relu(model, torch.nn.functional.mish)
        layer_type = ReluSurrogate
    elif fgi_type == "Custom": # Our B-SiLU formulation
        fgi_relu(model, custom_func)
        layer_type = ReluSurrogate
    else:
        raise ValueError(f"Invalid FGI type {fgi_type}")

    return model, layer_type

def custom_func(x: torch.Tensor):
    fx = (x+1.67) * torch.sigmoid(x) - 0.835
    return fx
