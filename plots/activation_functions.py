import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F


def bounded_silu(x, alpha=1.67):
    return (x + alpha) * torch.sigmoid(x) - 0.5 * alpha


def plot_activations(output):
    activations = (
        ("ReLU", F.relu, "#e377c2"),
        ("ELU", F.elu, "#1f77b4"),
        ("GELU", F.gelu, "#ff7f0e"),
        ("SiLU", F.silu, "#17becf"),
        ("SELU", F.selu, "#7f7f7f"),
        ("Leaky ReLU", lambda x: F.leaky_relu(x, negative_slope=0.2), "#2ca02c"),
        ("Bounded SiLU", bounded_silu, "#8c564b"),
    )
    style = {
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.5,
    }
    with plt.rc_context(style):
        x = np.linspace(-3, 3, 1000)
        inputs = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.2), facecolor="none")
        lines = []
        for name, function, color in activations:
            values = function(inputs)
            derivative = torch.autograd.grad(values, inputs, torch.ones_like(values))[0]
            width = 0.8 if name == "ReLU" else 0.6
            zorder = 10 if name in {"ReLU", "Bounded SiLU"} else 2
            (line,) = axes[0].plot(
                x,
                values.detach().numpy(),
                label=name,
                color=color,
                linewidth=width,
                zorder=zorder,
            )
            lines.append(line)
            axes[1].plot(
                x,
                derivative.detach().numpy(),
                color=color,
                linewidth=width,
                zorder=zorder,
            )
        for index, title in enumerate(("Activation functions", "Derivatives")):
            ax = axes[index]
            ax.text(
                -0.05,
                1.1,
                "ab"[index],
                transform=ax.transAxes,
                fontsize=8,
                fontweight="bold",
                va="top",
                ha="right",
            )
            ax.set_title(title, pad=10)
            ax.set_xlabel("Input ($x$)")
            ax.set_ylabel("$f(x)$" if index == 0 else "$f'(x)$")
            ax.grid(True, linestyle=":", linewidth=0.3, alpha=0.5)
            ax.set_facecolor("none")
            ax.tick_params(width=0.5, length=3)
        fig.legend(
            handles=lines,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=7,
            fontsize=6,
            frameon=False,
        )
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, transparent=True, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot activation functions and derivatives."
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_suffix(".pdf")
    )
    args = parser.parse_args()
    plot_activations(args.output)


if __name__ == "__main__":
    main()
