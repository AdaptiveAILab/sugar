from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.textpath import TextPath
from matplotlib.ticker import FormatStrFormatter
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.mplot3d import proj3d

from cnn.checkpoints import load_checkpoint

LAYER_KEYS = [f"layer{index}.2" for index in range(1, 14)] + ["fc1.3"]
LAYER_LABELS = [str(index) for index in range(1, 14)] + ["FC"]
RELU_COLOR = "#C84C4C"
SUGAR_COLOR = "#1478A6"
GRID_COLOR = "#DDE2E6"
TEXT_COLOR = "#202428"
LAYER_COLORS = [
    "#4477AA",
    "#EE7733",
    "#228833",
    "#EE6677",
    "#AA3377",
    "#33BBEE",
    "#CCBB44",
    "#CC3311",
    "#009988",
    "#EE3377",
    "#999933",
    "#882255",
    "#BBBBBB",
    "#AA4499",
]


def load_activations(path, epoch):
    checkpoint = load_checkpoint(path)
    activations = checkpoint["nonzero_neuron_acts"]
    if epoch not in activations:
        raise ValueError(f"Stored epoch key {epoch} is missing from {path}")
    missing = set(LAYER_KEYS) - activations[epoch].keys()
    if missing:
        raise ValueError(f"Missing activation layers in {path}: {sorted(missing)}")
    return (activations[epoch], epoch)


def histogram(values, edges):
    counts, _ = np.histogram(np.asarray(values).ravel(), bins=edges)
    return counts.astype(float) / max(counts.sum(), 1)


def smooth_histogram(values):
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    radius = len(kernel) // 2
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 9.0,
            "axes.titleweight": "normal",
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": "#AAB1B7",
            "axes.linewidth": 0.55,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
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


def clean_3d_axis(ax):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("#CDD3D8")
        axis._axinfo["grid"].update(
            {"color": GRID_COLOR, "linewidth": 0.35, "linestyle": "-"}
        )
        axis._axinfo["axisline"].update({"color": "#9FA7AE", "linewidth": 0.5})


def set_z_axis_side(ax, side):
    if side == "left":
        ax.zaxis._axinfo["juggled"] = (1, 2, 0)
    elif side == "right":
        ax.zaxis._axinfo["juggled"] = (0, 2, 1)
    elif side == "back-left":
        ax.zaxis.set_ticks_position("none")
        ax.zaxis.set_label_position("none")
        for tick in ax.zaxis.get_major_ticks():
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
    else:
        raise ValueError(f"Unsupported z-axis side: {side}")


def project_to_axes(ax, x, y, z):
    projected_x, projected_y, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    display = ax.transData.transform((projected_x, projected_y))
    return ax.transAxes.inverted().transform(display)


def draw_back_left_z_axis(
    ax, ticks, show_label, annotation_side="left", label_side=None, label_offset=0.145
):
    if annotation_side not in {"left", "right"}:
        raise ValueError(f"Unsupported z-axis annotation side: {annotation_side}")
    label_side = annotation_side if label_side is None else label_side
    if label_side not in {"left", "right"}:
        raise ValueError(f"Unsupported z-axis label side: {label_side}")
    x_min, _ = ax.get_xlim3d()
    _, y_max = ax.get_ylim3d()
    axis_points = np.asarray(
        [project_to_axes(ax, x_min, y_max, tick) for tick in ticks]
    )
    axis_color = "#9FA7AE"
    tick_length = 0.012
    label_offset = 0.038
    direction = 1 if annotation_side == "right" else -1
    text_alignment = "left" if annotation_side == "right" else "right"
    label_direction = 1 if label_side == "right" else -1
    ax.add_line(
        Line2D(
            [axis_points[0, 0], axis_points[-1, 0]],
            [axis_points[0, 1], axis_points[-1, 1]],
            transform=ax.transAxes,
            color=axis_color,
            linewidth=0.5,
            clip_on=False,
            zorder=20,
        )
    )
    for tick, (x_position, y_position) in zip(ticks, axis_points):
        ax.add_line(
            Line2D(
                [x_position, x_position + direction * tick_length],
                [y_position, y_position],
                transform=ax.transAxes,
                color=TEXT_COLOR,
                linewidth=0.5,
                clip_on=False,
                zorder=21,
            )
        )
        ax.text2D(
            x_position + direction * (tick_length + label_offset),
            y_position,
            f"{tick:.2f}",
            transform=ax.transAxes,
            fontsize=6.5,
            color=TEXT_COLOR,
            ha=text_alignment,
            va="center",
            clip_on=False,
            zorder=21,
        )
    if show_label:
        midpoint = axis_points.mean(axis=0)
        ax.text2D(
            midpoint[0] + label_direction * label_offset,
            midpoint[1],
            "Norm. freq.",
            transform=ax.transAxes,
            rotation=90,
            fontsize=7.0,
            color=TEXT_COLOR,
            ha="center",
            va="center",
            clip_on=False,
            zorder=21,
        )


def draw_offset_fc_label(
    ax, fc_tick_position, offset_steps, shift_right_widths=0.0, shift_down_heights=0.0
):
    _, x_max = ax.get_xlim3d()
    z_min, _ = ax.get_zlim3d()
    x_position, y_position = project_to_axes(
        ax, x_max, fc_tick_position + offset_steps, z_min
    )
    font_size = 6.5
    font = FontProperties(family="sans-serif", size=font_size)
    text_bounds = TextPath((0, 0), "FC", prop=font, size=font_size).get_extents()
    display_offset = ScaledTranslation(
        shift_right_widths * text_bounds.width / 72.0,
        -shift_down_heights * text_bounds.height / 72.0,
        ax.figure.dpi_scale_trans,
    )
    ax.text2D(
        x_position + 0.012,
        y_position,
        "FC",
        transform=ax.transAxes + display_offset,
        fontproperties=font,
        color=TEXT_COLOR,
        ha="left",
        va="center",
        clip_on=False,
        zorder=21,
    )


def draw_3d_histogram(
    ax,
    activations,
    colors,
    title,
    z_max,
    bins=100,
    show_z_label=True,
    z_axis_side="right",
    z_annotation_side="left",
    z_label_side=None,
    z_label_offset=0.145,
    center_layer_ticks=False,
    separate_fc=False,
    fc_label_offset=0.0,
    fc_label_shift_right=0.0,
    fc_label_shift_down=0.0,
):
    max_activation = max((np.max(np.asarray(activations[key])) for key in LAYER_KEYS))
    edges = np.linspace(0, max_activation + 1, bins + 1)
    for layer_index in reversed(range(len(LAYER_KEYS))):
        key = LAYER_KEYS[layer_index]
        color = colors[layer_index]
        counts, _ = np.histogram(np.asarray(activations[key]).ravel(), bins=edges)
        frequency = counts.astype(float) / max(counts.sum(), 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges) * 0.88
        layer_position = layer_index
        if separate_fc and layer_index == len(LAYER_KEYS) - 1:
            layer_position += 1
        y = np.full_like(centers, layer_position, dtype=float)
        ax.bar3d(
            centers,
            y,
            np.zeros_like(frequency),
            widths,
            np.full_like(centers, 0.54),
            frequency,
            color=color,
            edgecolor=(1.0, 1.0, 1.0, 0.14),
            linewidth=0.055,
            shade=True,
            alpha=0.94,
            zsort="max",
        )
    ax.set_title(title, pad=1.5)
    ax.set_xlim(0, 45000)
    fc_gap = 1 if separate_fc else 0
    ax.set_ylim(-0.35, len(LAYER_KEYS) - 0.05 + fc_gap)
    ax.set_zlim(0, z_max)
    ax.set_xticks([0, 15000, 30000, 45000])
    ax.set_xticklabels(["0", "15k", "30k", "45k"])
    layer_ticks = np.asarray([0, 2, 4, 6, 8, 10, 12, 13], dtype=float)
    if separate_fc:
        layer_ticks[-1] += 1
    if center_layer_ticks:
        layer_ticks += 0.27
    ax.set_yticks(layer_ticks)
    layer_tick_labels = ["1", "3", "5", "7", "9", "11", "13", "FC"]
    if fc_label_offset:
        layer_tick_labels[-1] = ""
    ax.set_yticklabels(layer_tick_labels)
    ax.set_zticks(np.linspace(0, z_max, 4))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("Activation count", labelpad=2.5)
    ax.set_ylabel("Layer", labelpad=1.0)
    if z_axis_side in {"left", "back-left"}:
        ax.set_zlabel("")
        if show_z_label and z_axis_side == "left":
            ax.text2D(
                -0.07,
                0.52,
                "Norm. freq.",
                transform=ax.transAxes,
                rotation=90,
                fontsize=7.0,
                color=TEXT_COLOR,
                ha="center",
                va="center",
            )
    else:
        ax.set_zlabel("Norm. freq." if show_z_label else "", labelpad=1.5)
    ax.tick_params(axis="x", pad=-1)
    ax.tick_params(axis="y", pad=-1)
    ax.tick_params(axis="z", pad=0)
    ax.view_init(elev=25, azim=-60)
    layer_aspect = 1.11 if separate_fc else 1.04
    ax.set_box_aspect((1.45, layer_aspect, 0.82))
    clean_3d_axis(ax)
    set_z_axis_side(ax, z_axis_side)
    if z_axis_side == "back-left":
        draw_back_left_z_axis(
            ax,
            ax.get_zticks(),
            show_z_label,
            annotation_side=z_annotation_side,
            label_side=z_label_side,
            label_offset=z_label_offset,
        )
    if fc_label_offset:
        draw_offset_fc_label(
            ax,
            layer_ticks[-1],
            fc_label_offset,
            shift_right_widths=fc_label_shift_right,
            shift_down_heights=fc_label_shift_down,
        )


def draw_layer_comparison(ax, relu, sugar, edges):
    centers = 0.5 * (edges[:-1] + edges[1:])
    levels = np.arange(len(LAYER_KEYS))
    for level, key in zip(levels, LAYER_KEYS):
        relu_frequency = smooth_histogram(histogram(relu[key], edges))
        sugar_frequency = smooth_histogram(histogram(sugar[key], edges))
        joint_peak = max(relu_frequency.max(), sugar_frequency.max(), 1e-12)
        relu_profile = relu_frequency / joint_peak
        sugar_profile = sugar_frequency / joint_peak
        relu_y = level + 0.5 * relu_profile
        sugar_y = level + 0.5 * sugar_profile
        ax.hlines(
            level, edges[0], edges[-1], color=GRID_COLOR, linewidth=0.38, zorder=0
        )
        ax.fill_between(
            centers, level, relu_y, color=RELU_COLOR, alpha=0.075, linewidth=0, zorder=1
        )
        ax.fill_between(
            centers, level, sugar_y, color=SUGAR_COLOR, alpha=0.1, linewidth=0, zorder=2
        )
        ax.plot(
            centers,
            relu_y,
            color=RELU_COLOR,
            linewidth=0.58,
            solid_capstyle="round",
            zorder=3,
        )
        ax.plot(
            centers,
            sugar_y,
            color=SUGAR_COLOR,
            linewidth=0.72,
            solid_capstyle="round",
            zorder=4,
        )
    ax.set_title("Layer-wise comparison", pad=13)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(-0.28, len(LAYER_KEYS) - 0.02)
    ax.set_xticks([0, 15000, 30000, 45000])
    ax.set_xticklabels(["0", "15k", "30k", "45k"])
    ax.set_yticks(levels + 0.18)
    ax.set_yticklabels(LAYER_LABELS)
    ax.set_xlabel("Activation count", labelpad=3)
    ax.set_ylabel("Layer", labelpad=3)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.38)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#AAB1B7")
    handles = [
        Line2D([0], [0], color=RELU_COLOR, linewidth=0.9, label="ReLU"),
        Line2D([0], [0], color=SUGAR_COLOR, linewidth=1.0, label="SUGAR (B-SiLU)"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.055),
        frameon=False,
        ncol=2,
        handlelength=1.7,
        columnspacing=1.2,
        borderaxespad=0,
    )


def add_panel_label(ax, label, x=-0.05, y=1.04):
    ax.text2D(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_combined(
    relu,
    sugar,
    output,
    epoch,
    z_axis_side="right",
    z_annotation_side="left",
    z_label_side=None,
    z_label_offset=0.145,
    show_both_z_labels=False,
    center_layer_ticks=False,
    separate_fc=False,
    fc_label_offset=0.0,
    fc_label_shift_right=0.0,
    fc_label_shift_down=0.0,
):
    configure_style()
    bins_3d = 100
    edges_2d = np.linspace(0, 45001, bins_3d + 1)
    all_frequencies = []
    for activations in (relu, sugar):
        max_activation = max(
            (np.max(np.asarray(activations[key])) for key in LAYER_KEYS)
        )
        edges_3d = np.linspace(0, max_activation + 1, bins_3d + 1)
        for key in LAYER_KEYS:
            counts, _ = np.histogram(
                np.asarray(activations[key]).ravel(), bins=edges_3d
            )
            all_frequencies.append(counts.astype(float) / max(counts.sum(), 1))
    raw_max = max((frequency.max() for frequency in all_frequencies))
    z_max = np.ceil(raw_max / 0.02) * 0.02
    fig = plt.figure(figsize=(11.1, 3.68))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.08, 1.08, 0.88],
        left=0.018,
        right=0.992,
        bottom=0.13,
        top=0.93,
        wspace=0.1,
    )
    relu_ax = fig.add_subplot(grid[0, 0], projection="3d", computed_zorder=False)
    sugar_ax = fig.add_subplot(grid[0, 1], projection="3d", computed_zorder=False)
    comparison_ax = fig.add_subplot(grid[0, 2])
    layer_colors = np.asarray([mpl.colors.to_rgba(color) for color in LAYER_COLORS])
    draw_3d_histogram(
        relu_ax,
        relu,
        layer_colors,
        "ReLU",
        z_max,
        bins=bins_3d,
        z_axis_side=z_axis_side,
        z_annotation_side=z_annotation_side,
        z_label_side=z_label_side,
        z_label_offset=z_label_offset,
        center_layer_ticks=center_layer_ticks,
        separate_fc=separate_fc,
        fc_label_offset=fc_label_offset,
        fc_label_shift_right=fc_label_shift_right,
        fc_label_shift_down=fc_label_shift_down,
    )
    draw_3d_histogram(
        sugar_ax,
        sugar,
        layer_colors,
        "SUGAR (B-SiLU)",
        z_max,
        bins=bins_3d,
        show_z_label=show_both_z_labels,
        z_axis_side=z_axis_side,
        z_annotation_side=z_annotation_side,
        z_label_side=z_label_side,
        z_label_offset=z_label_offset,
        center_layer_ticks=center_layer_ticks,
        separate_fc=separate_fc,
        fc_label_offset=fc_label_offset,
        fc_label_shift_right=fc_label_shift_right,
        fc_label_shift_down=fc_label_shift_down,
    )
    draw_layer_comparison(comparison_ax, relu, sugar, edges_2d)
    add_panel_label(relu_ax, "d", x=-0.03, y=1.035)
    add_panel_label(sugar_ax, "e", x=-0.03, y=1.035)
    comparison_ax.text(
        -0.08,
        1.035,
        "f",
        transform=comparison_ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="top",
        ha="left",
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(
        output.with_suffix(".png"),
        dpi=260,
        bbox_inches="tight",
        pad_inches=0.025,
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)
    print(f"Saved {output} (epoch {epoch})")
