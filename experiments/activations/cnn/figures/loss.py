from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from cnn.checkpoints import load_checkpoint, seeded_files

TEXT_COLOR = "#202428"
AXIS_COLOR = "#AAB1B7"
GRID_COLOR = "#DDE2E6"
LABELS = {
    "param_swish": "B-SiLU",
    "elu": "ELU",
    "gelu": "GELU",
    "leaky": "Leaky ReLU",
    "mish": "Mish",
    "nelu": "NeLU",
    "relu": "ReLU",
    "selu": "SELU",
    "swish": "Swish",
}
ORDER = [
    "param_swish",
    "elu",
    "gelu",
    "leaky",
    "mish",
    "nelu",
    "relu",
    "selu",
    "swish",
]
COLORS = {
    "param_swish": "#1478A6",
    "elu": "#3A8D7D",
    "gelu": "#D98A2B",
    "leaky": "#8066A8",
    "mish": "#B45579",
    "nelu": "#9A9632",
    "relu": "#C84C4C",
    "selu": "#69727A",
    "swish": "#36A5B5",
}


def collect_runs(base, mode):
    root = Path(base) / "cifar100" / "resnet18" / mode
    runs = {}
    for activation in ORDER:
        if activation == "relu" and mode == "non_fgi":
            continue
        folder = root / activation
        if activation == "nelu":
            folder /= "alpha_0.01"
        for path in seeded_files(folder):
            checkpoint = load_checkpoint(path)
            if checkpoint["fgi_type"] != activation:
                raise ValueError(f"Activation metadata mismatch in {path}")
            if activation == "nelu" and checkpoint["alpha"] != 0.01:
                raise ValueError(f"Expected NeLU alpha=0.01 in {path}")
            losses = checkpoint["val_losses"]
            epochs = sorted(losses)
            if epochs != list(range(100)):
                raise ValueError(f"Expected all 100 validation epochs in {path}")
            values = np.asarray([losses[epoch] for epoch in epochs], dtype=float)
            accuracy = float(checkpoint["test_accuracy"])
            runs.setdefault(activation, []).append(
                (np.asarray(epochs), values, accuracy)
            )
    return runs


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
            "axes.edgecolor": AXIS_COLOR,
            "axes.linewidth": 0.55,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 6.5,
            "text.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": True,
        }
    )


def summarize_curves(records):
    length = min((len(values) for _, values, _ in records))
    epochs = records[0][0][:length]
    values = np.vstack([record[1][:length] for record in records])
    mean = values.mean(axis=0)
    if len(records) > 1:
        ci95 = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(len(records))
    else:
        ci95 = np.zeros_like(mean)
    accuracy = np.mean([record[2] for record in records])
    return (epochs, mean, ci95, accuracy)


def plot_loss_curves(results, output):
    configure_style()
    baseline_runs = collect_runs(results, "non_fgi")
    sugar_runs = collect_runs(results, "fgi")
    if "relu" in sugar_runs:
        baseline_runs["relu"] = sugar_runs["relu"]
    figure_width = 7.03
    figure_height = 2.27
    fig, axes = plt.subplots(
        1, 2, figsize=(figure_width, figure_height), sharex=True, sharey=True
    )
    legend_handles = {}
    legend_labels = {}
    panel_runs = [baseline_runs, sugar_runs]
    for panel_index, (ax, runs) in enumerate(zip(axes, panel_runs)):
        for activation in ORDER:
            if activation == "relu" and panel_index == 1:
                continue
            records = runs.get(activation)
            if not records:
                continue
            epochs, mean, ci95, accuracy = summarize_curves(records)
            emphasized = activation in {"param_swish", "relu"}
            (line,) = ax.plot(
                epochs,
                mean,
                color=COLORS[activation],
                linewidth=1.15 if emphasized else 0.78,
                zorder=5 if emphasized else 3,
            )
            ax.fill_between(
                epochs,
                mean - ci95,
                mean + ci95,
                color=COLORS[activation],
                alpha=0.12 if emphasized else 0.08,
                linewidth=0,
                zorder=2 if emphasized else 1,
            )
            if panel_index == 1 or activation == "relu":
                legend_handles[activation] = line
                legend_labels[activation] = f"{LABELS[activation]} ({accuracy:.1f}%)"
        ax.set_title("non-SUGAR" if panel_index == 0 else "SUGAR", pad=6)
        ax.set_xlabel("Epoch", labelpad=4)
        ax.set_yscale("log")
        ax.set_xlim(-1, 104)
        ax.set_ylim(1.65, 4.35)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_yticks([1.8, 2.0, 2.5, 3.0, 4.0])
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.grid(axis="both", color=GRID_COLOR, linewidth=0.45)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(AXIS_COLOR)
    axes[0].set_ylabel("Validation loss", labelpad=5)
    handles = [legend_handles[key] for key in ORDER if key in legend_handles]
    labels = [legend_labels[key] for key in ORDER if key in legend_labels]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.0,
        handletextpad=0.55,
    )
    fig.text(
        0.018,
        0.94,
        "a",
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="top",
        color=TEXT_COLOR,
    )
    fig.text(
        0.512,
        0.94,
        "b",
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="top",
        color=TEXT_COLOR,
    )
    fig.subplots_adjust(left=0.085, right=0.988, bottom=0.31, top=0.8, wspace=0.08)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(
        output.with_suffix(".png"), dpi=300, facecolor="white", transparent=False
    )
    plt.close(fig)
    print(f"Saved {output}")
