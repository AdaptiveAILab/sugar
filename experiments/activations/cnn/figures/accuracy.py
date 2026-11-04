from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from cnn.checkpoints import load_checkpoint, seeded_files

BASELINE_COLOR = "#9AA2A9"
BASELINE_EDGE = "#69727A"
SUGAR_COLOR = "#1478A6"
SUGAR_EDGE = "#0D5D82"
RELU_COLOR = "#C84C4C"
RELU_EDGE = "#8E3434"
GRID_COLOR = "#DDE2E6"
TEXT_COLOR = "#202428"
CONFIGS = [
    ("param_swish", None, "B-SiLU"),
    ("selu", None, "SELU"),
    ("elu", None, "ELU"),
    ("mish", None, "Mish"),
    ("leaky", None, "Leaky ReLU"),
    ("gelu", None, "GELU"),
    ("relu", None, "ReLU"),
    ("nelu", "alpha_0.01", "NeLU ($\\alpha=0.01$)"),
    ("nelu", "alpha_0.05", "NeLU ($\\alpha=0.05$)"),
    ("nelu", "alpha_0.1", "NeLU ($\\alpha=0.1$)"),
    ("swish", None, "Swish"),
]


def collect_values(root, activation, alpha=None):
    path = Path(root) / activation
    if alpha:
        path = path / alpha
    files = seeded_files(path)
    return np.asarray(
        [float(load_checkpoint(path)["test_accuracy"]) for path in files], dtype=float
    )


def mean_sd(values):
    return (float(values.mean()), float(values.std(ddof=0)))


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 9.5,
            "axes.titleweight": "normal",
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": "#AAB1B7",
            "axes.linewidth": 0.55,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.7,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 6.7,
            "text.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": True,
        }
    )


def add_value_label(ax, x, mean, sd, color, weight="normal"):
    ax.text(
        x,
        mean + sd + 0.85,
        f"{mean:.1f}",
        color=color,
        fontsize=6.2,
        fontweight=weight,
        ha="center",
        va="bottom",
    )


def plot_accuracy(results, output, panel_label="c", save_png=True):
    configure_style()
    root = Path(results) / "cifar100" / "vgg16"
    rows = []
    for activation, alpha, label in CONFIGS:
        if activation == "relu":
            baseline_values = collect_values(root / "fgi", activation, alpha)
            sugar_values = None
        else:
            baseline_values = collect_values(root / "non_fgi", activation, alpha)
            sugar_values = collect_values(root / "fgi", activation, alpha)
        baseline_mean, baseline_sd = mean_sd(baseline_values)
        sugar_stats = mean_sd(sugar_values) if sugar_values is not None else None
        rows.append((label, baseline_mean, baseline_sd, sugar_stats))
    figure_width = 7.03
    figure_height = 2.52
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    x = np.arange(len(rows), dtype=float)
    width = 0.34
    error_style = {"elinewidth": 0.65, "capthick": 0.65, "capsize": 2.1}
    for index, (_, baseline_mean, baseline_sd, sugar_stats) in enumerate(rows):
        if sugar_stats is None:
            ax.bar(
                index,
                baseline_mean,
                width=width,
                yerr=baseline_sd,
                color=RELU_COLOR,
                edgecolor=RELU_EDGE,
                linewidth=0.55,
                error_kw={**error_style, "ecolor": RELU_EDGE},
                zorder=3,
            )
            add_value_label(
                ax, index, baseline_mean, baseline_sd, RELU_COLOR, weight="medium"
            )
            continue
        sugar_mean, sugar_sd = sugar_stats
        baseline_x = index - width / 2
        sugar_x = index + width / 2
        ax.bar(
            baseline_x,
            baseline_mean,
            width=width,
            yerr=baseline_sd,
            color=BASELINE_COLOR,
            edgecolor=BASELINE_EDGE,
            linewidth=0.5,
            error_kw={**error_style, "ecolor": BASELINE_EDGE},
            zorder=3,
        )
        ax.bar(
            sugar_x,
            sugar_mean,
            width=width,
            yerr=sugar_sd,
            color=SUGAR_COLOR,
            edgecolor=SUGAR_EDGE,
            linewidth=0.5,
            error_kw={**error_style, "ecolor": SUGAR_EDGE},
            zorder=3,
        )
        add_value_label(ax, baseline_x, baseline_mean, baseline_sd, BASELINE_EDGE)
        add_value_label(ax, sugar_x, sugar_mean, sugar_sd, SUGAR_COLOR, weight="medium")
    ax.set_xlim(-0.72, len(rows) - 0.28)
    ax.set_ylim(0, 71.5)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xticks(x)
    ax.set_xticklabels(
        [row[0] for row in rows], rotation=28, ha="right", rotation_mode="anchor"
    )
    ax.set_ylabel("Test accuracy (%)", labelpad=5)
    ax.set_title("VGG-16 on CIFAR-100", pad=7)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AAB1B7")
    handles = [
        Patch(
            facecolor=BASELINE_COLOR,
            edgecolor=BASELINE_EDGE,
            linewidth=0.5,
            label="Baseline",
        ),
        Patch(
            facecolor=SUGAR_COLOR, edgecolor=SUGAR_EDGE, linewidth=0.5, label="SUGAR"
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.015),
        frameon=False,
        ncol=2,
        handlelength=1.35,
        handleheight=0.8,
        columnspacing=1.0,
        borderaxespad=0,
    )
    if panel_label:
        fig.text(
            0.018,
            0.94,
            panel_label,
            fontsize=10.5,
            fontweight="bold",
            ha="left",
            va="top",
            color=TEXT_COLOR,
        )
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.24, top=0.84)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    if save_png:
        fig.savefig(
            output.with_suffix(".png"), dpi=300, facecolor="white", transparent=False
        )
    plt.close(fig)
    print(f"Saved {output}")
