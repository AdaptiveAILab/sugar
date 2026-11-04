from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D

from cnn.figures import accuracy as accuracy_plot
from cnn.figures import activations as activation_plot
from cnn.figures import loss as loss_plot


def prepare_loss_runs(results):
    baseline_runs = loss_plot.collect_runs(results, "non_fgi")
    sugar_runs = loss_plot.collect_runs(results, "fgi")
    if "relu" in sugar_runs:
        baseline_runs["relu"] = sugar_runs["relu"]
    return (baseline_runs, sugar_runs)


def plot_loss_panel(runs, title, output, include_relu):
    loss_plot.configure_style()
    legend_labels = {}
    for activation, records in runs.items():
        _, _, _, accuracy = loss_plot.summarize_curves(records)
        legend_labels[activation] = f"{loss_plot.LABELS[activation]} ({accuracy:.1f}%)"
    fig, ax = plt.subplots(figsize=(3.54, 2.72))
    for activation in loss_plot.ORDER:
        if activation == "relu" and (not include_relu):
            continue
        records = runs.get(activation)
        if not records:
            continue
        epochs, mean, ci95, _ = loss_plot.summarize_curves(records)
        emphasized = activation in {"param_swish", "relu"}
        ax.plot(
            epochs,
            mean,
            color=loss_plot.COLORS[activation],
            linewidth=1.15 if emphasized else 0.78,
            zorder=5 if emphasized else 3,
        )
        ax.fill_between(
            epochs,
            mean - ci95,
            mean + ci95,
            color=loss_plot.COLORS[activation],
            alpha=0.12 if emphasized else 0.08,
            linewidth=0,
            zorder=2 if emphasized else 1,
        )
    ax.set_title(title, pad=6)
    ax.set_xlabel("Epoch", labelpad=4)
    ax.set_ylabel("Validation loss", labelpad=5)
    ax.set_yscale("log")
    ax.set_xlim(-1, 104)
    ax.set_ylim(1.65, 4.35)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([1.8, 2.0, 2.5, 3.0, 4.0])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.grid(axis="both", color=loss_plot.GRID_COLOR, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(loss_plot.AXIS_COLOR)
    handles = [
        Line2D(
            [0],
            [0],
            color=loss_plot.COLORS[activation],
            linewidth=1.15 if activation in {"param_swish", "relu"} else 0.78,
        )
        for activation in loss_plot.ORDER
        if activation in legend_labels
    ]
    labels = [
        legend_labels[activation]
        for activation in loss_plot.ORDER
        if activation in legend_labels
    ]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        frameon=False,
        handlelength=1.65,
        columnspacing=0.78,
        handletextpad=0.45,
        fontsize=5.75,
    )
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.35, top=0.84)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def activation_z_max(relu, sugar, bins=100):
    frequencies = []
    for activations in (relu, sugar):
        maximum = max(
            (np.max(np.asarray(activations[key])) for key in activation_plot.LAYER_KEYS)
        )
        edges = np.linspace(0, maximum + 1, bins + 1)
        for key in activation_plot.LAYER_KEYS:
            counts, _ = np.histogram(np.asarray(activations[key]).ravel(), bins=edges)
            frequencies.append(counts.astype(float) / max(counts.sum(), 1))
    raw_max = max((frequency.max() for frequency in frequencies))
    return np.ceil(raw_max / 0.02) * 0.02


def plot_3d_panel(activations, title, z_max, output):
    activation_plot.configure_style()
    fig = plt.figure(figsize=(3.7, 3.05))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    colors = np.asarray(
        [mpl.colors.to_rgba(color) for color in activation_plot.LAYER_COLORS]
    )
    activation_plot.draw_3d_histogram(
        ax,
        activations,
        colors,
        title,
        z_max,
        bins=100,
        show_z_label=True,
        z_axis_side="back-left",
        z_annotation_side="right",
        z_label_side="left",
        z_label_offset=0.105,
        center_layer_ticks=True,
        fc_label_offset=1.0,
        fc_label_shift_right=0.5,
        fc_label_shift_down=1.4,
    )
    fig.subplots_adjust(left=0.0, right=0.98, bottom=0.0, top=0.94)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def plot_comparison_panel(relu, sugar, output):
    activation_plot.configure_style()
    fig, ax = plt.subplots(figsize=(3.48, 3.05))
    edges = np.linspace(0, 45001, 101)
    activation_plot.draw_layer_comparison(ax, relu, sugar, edges)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.13, top=0.88)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def export_individuals(results, relu_path, sugar_path, output_dir, epoch):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_runs, sugar_runs = prepare_loss_runs(results)
    plot_loss_panel(
        baseline_runs,
        "non-SUGAR",
        output_dir / "fig3_a.pdf",
        include_relu=True,
    )
    plot_loss_panel(
        sugar_runs,
        "SUGAR",
        output_dir / "fig3_b.pdf",
        include_relu=False,
    )
    accuracy_plot.plot_accuracy(
        results, output_dir / "fig3_c.pdf", panel_label=None, save_png=False
    )
    relu, relu_epoch = activation_plot.load_activations(relu_path, epoch)
    sugar, sugar_epoch = activation_plot.load_activations(sugar_path, epoch)
    if relu_epoch != sugar_epoch:
        raise RuntimeError(f"Epoch mismatch: ReLU={relu_epoch}, SUGAR={sugar_epoch}")
    z_max = activation_z_max(relu, sugar)
    plot_3d_panel(relu, "ReLU", z_max, output_dir / "fig3_d.pdf")
    plot_3d_panel(sugar, "SUGAR (B-SiLU)", z_max, output_dir / "fig3_e.pdf")
    plot_comparison_panel(relu, sugar, output_dir / "fig3_f.pdf")
