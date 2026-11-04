"""Randomized Asymmetric Initialization (RAI), Lu, Shin, Su & Karniadakis (2020),
"Dying ReLU and Initialization", Commun. Comput. Phys. 28(5), 1671-1706.

PyTorch port of the paper's numpy reference initializer, with equivalent
sampling (same distributions) but using torch's RNG so it (a) rides a single
torch `generator` for reproducibility -- no separate numpy seed -- and (b) can
sample directly on the model's device (CPU or GPU).

Distribution equivalence to the paper's `np.random.*` version:
  * Gaussian bulk:  N(0, sigma^2), sigma = 0.6007 / sqrt(fan_in)   (identical)
  * chosen index :  uniform over {0,...,fan_in}  (fan_in+1 choices) (identical)
  * P = Beta(2,1):  sampled by inverse-CDF sqrt(U), U~Uniform(0,1). Beta(2,1)
                    has CDF F(x)=x^2, so sqrt(U) ~ Beta(2,1) exactly -- same
                    distribution as np.random.beta(2,1) (not bit-identical RNG).
"""
import torch
import torch.nn as nn


def RAI(fan_in, fan_out, generator: torch.Generator | None = None,
        device=None, dtype=torch.float32):
    """Randomized asymmetric initializer (PyTorch).

    fan_in / fan_out are the input/output units of the weight tensor. Builds
    V of shape (fan_out, fan_in+1); each row (output neuron) has all-but-one
    coordinate ~ N(0, sigma^2) and ONE randomly chosen coordinate ~ Beta(2,1).
    Returns W with shape (fan_in, fan_out) and b with shape (fan_out,), matching
    the paper's layout (W is meant for x @ W).
    """
    V = torch.randn(fan_out, fan_in + 1, generator=generator,
                    device=device, dtype=dtype) * (0.6007 / fan_in ** 0.5)
    k = torch.randint(0, fan_in + 1, (fan_out,), generator=generator, device=device)
    beta = torch.rand(fan_out, generator=generator, device=device, dtype=dtype).sqrt()  # Beta(2,1)
    V[torch.arange(fan_out, device=device), k] = beta
    W = V[:, :-1].t()      # (fan_in, fan_out)
    b = V[:, -1]           # (fan_out,)
    return W, b


class RAILinear(nn.Linear):
    """Marker subclass of nn.Linear: identical behaviour, but its repr shows
    `init=RAI` so `print(model)` reveals which layers were RAI-initialized."""
    def extra_repr(self):
        return super().extra_repr() + ", init=RAI"


@torch.no_grad()
def rai_linear_(linear: nn.Linear, generator: torch.Generator | None = None):
    """Fill `linear` in place from RAI() (on the layer's own device/dtype), then
    tag it as RAILinear. If `generator` is on CUDA it must match the layer's
    device."""
    fan_out, fan_in = linear.weight.shape
    W, b = RAI(fan_in, fan_out, generator=generator,
               device=linear.weight.device, dtype=linear.weight.dtype)
    linear.weight.copy_(W.t())                  # (fan_in,fan_out) -> (fan_out,fan_in)
    if linear.bias is not None:
        linear.bias.copy_(b)
    linear.__class__ = RAILinear                # so print(model) shows init=RAI


@torch.no_grad()
def apply_rai(model: nn.Module, generator: torch.Generator | None = None):
    """Apply RAI to every nn.Linear except the first, which gets He init (b=0).
    Everything rides `generator` (torch), so one seed reproduces the whole init.
    """
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    for i, lin in enumerate(linears):
        if i == 0:  # layer 1: He init (fan_in, ReLU) with zero bias, per the paper
            nn.init.kaiming_normal_(lin.weight, mode="fan_in", nonlinearity="relu", generator=generator)
            if lin.bias is not None:
                lin.bias.zero_()
        else:
            rai_linear_(lin, generator=generator)
    return model


if __name__ == "__main__":
    # ---- self-test: apply RAI to a SimpleReLUNetwork, verify the two weight ---
    # populations, and show that print(model) marks the RAI layers.
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class SimpleReLUNetwork(nn.Module):
        """Same architecture as the experiment (Linear+ReLU blocks, final Linear)."""
        def __init__(self, input_dim, hidden_widths, output_dim):
            super().__init__()
            dims = [input_dim] + list(hidden_widths)
            self.num_hidden = len(hidden_widths)
            for i in range(1, self.num_hidden + 1):
                setattr(self, f"layer{i}",
                        nn.Sequential(nn.Linear(dims[i - 1], dims[i]), nn.ReLU()))
            self.final_idx = self.num_hidden + 1
            setattr(self, f"layer{self.final_idx}",
                    nn.Sequential(nn.Linear(dims[-1], output_dim)))

        def forward(self, x):
            out = x
            for i in range(1, self.final_idx + 1):
                out = getattr(self, f"layer{i}")(out)
            return out

    g = torch.Generator().manual_seed(0)

    # (1) Show the model repr before/after so RAI tagging is visible.
    demo = SimpleReLUNetwork(1, [2] * 9, 1)
    print("=== model BEFORE apply_rai (all plain Linear) ===")
    print(demo)
    apply_rai(demo, generator=g)
    print("\n=== model AFTER apply_rai (layers >=2 shown as RAILinear, init=RAI) ===")
    print(demo)

    # (2) Pool weights over many tiny nets so the histograms are populated.
    N_NETS = 2500
    first_w, rai_w = [], []
    for _ in range(N_NETS):
        net = SimpleReLUNetwork(2, [4] * 19, 2)
        apply_rai(net, generator=g)
        linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
        first_w.append(linears[0].weight.detach().flatten())          # layer 1: He
        for lin in linears[1:]:                                        # layers >=2: RAI
            rai_w.append(lin.weight.detach().flatten())
            if lin.bias is not None:
                rai_w.append(lin.bias.detach().flatten())
    first_w = torch.cat(first_w).numpy()
    rai_w = torch.cat(rai_w).numpy()

    he_std = np.sqrt(2.0 / 1)           # first-layer fan_in = input_dim = 1
    print(f"\n[layer 1  / He ] n={first_w.size:>7}  mean={first_w.mean():+.3f}  "
          f"std={first_w.std():.3f}  (expected He std={he_std:.3f})")
    print(f"[layers>=2/ RAI] n={rai_w.size:>7}  mean={rai_w.mean():+.3f}  "
          f"frac_positive={(rai_w > 0).mean():.3f}  (He/symmetric would be ~0.50)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(first_w, bins=80, color="#4C72B0"); ax[0].axvline(0, color="gray", ls=":")
    ax[0].set_title(f"Layer 1 — He init\nmean={first_w.mean():+.3f}, std={first_w.std():.2f} (symmetric)")
    ax[0].set_xlabel("weight value"); ax[0].set_ylabel("count")
    ax[1].hist(rai_w, bins=80, color="#DD8452"); ax[1].axvline(0, color="gray", ls=":")
    ax[1].set_title(f"Layers ≥2 — RAI\nmean={rai_w.mean():+.3f}, {100*(rai_w>0).mean():.0f}% positive (asymmetric)")
    ax[1].set_xlabel("weight value"); ax[1].set_ylabel("count")
    fig.suptitle("RAI init verification (PyTorch) on SimpleReLUNetwork (widths=[4]×19)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("rai_init_verification_relu20_pytorch.png", dpi=150)
    print("saved rai_init_verification.png")