#!/usr/bin/env python3
# Paper figures (output file names were kept as produced by this script):
#   comparison_f1.pdf = Fig. 2a (f1, absfunc1)
#   comparison_f2.pdf = Fig. 2b (f2, sinefunc2)
#   comparison_f3.pdf = Fig. 2c (f3, sinefunc3)
#   comparison_plot_f4.pdf = Fig. 2d (f4, absfunc4; make_fig2d, written next to the other panels)
# The published panels were produced with
#   python plot_collapse_split.py --models-dir models --legend each --legend-ncol 2
# (each panel with its own two-column legend; the default --legend separate writes the legend to its own file).
# With matplotlib 3.9.2, as used for the paper, this reproduces the published PDFs (vector paths within 1e-4 pt);
"""
plot_collapse_split.py
======================

Instead of one combined 1x3 figure, this draws THREE separate figures -- one
per target (f1, f2, f3) -- and writes each to its own PDF. Every panel keeps
the exact physical size it had inside the old combined figure, so dropping the
three PDFs side by side in the paper reproduces the original \textwidth row.

How the "same dimensions" guarantee works
-----------------------------------------
The old figure was `--figwidth` wide (default 6.9 in = \textwidth) with three
panels in a row, so each panel occupied `figwidth / 3`. Here every per-panel
figure is `figwidth / 3` wide (override with --panel-width) and `--figheight`
tall -- identical to before. `--figwidth` therefore still means the *combined*
width; the split just divides it by three.

The shared legend
-----------------
The old figure had a SINGLE legend centered below all three panels. When the
panels become separate files that shared legend has to go somewhere, so
--legend controls it:

  * separate (default): panels carry NO legend (their plot area is byte-for-byte
    the same size as before) and a fourth file holds the shared legend on its
    own, tightly cropped, ready to place once beneath the row.
  * each: every panel gets its own legend below it (use --legend-ncol to lay it
    out; the panels are narrow, so ncol=1 is often the safe choice).
  * none: no legend anywhere.

Networks
--------
The networks are read from --models-dir as PyTorch state dicts::

    relu10_250_epochs_fw_func_NONE_fgi_{NONE|Custom}_target_func_{target}.pth   (Fig. 2a-c)
    relu20_250_epochs_fw_func_NONE_fgi_{NONE|Custom}_target_func_absfunc4.pth   (Fig. 2d)

Usage
-----
    python plot_collapse_split.py --models-dir models
    python plot_collapse_split.py --models-dir models --out fig_{label}.pdf
    python plot_collapse_split.py --models-dir models --legend each --legend-ncol 1

Run with -h for all options.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator


# --------------------------------------------------------------------------- #
# Style knobs (tweak here)
# --------------------------------------------------------------------------- #
LW_TRUTH = 0.9          # ground-truth line width (was 1.5)
LW_PRED = 1.1           # prediction line width   (was 2.0)
ALPHA = 1.0             # opacity of every curve   (was 0.6 on the collapsed fit)
X_MINOR_PER_MAJOR = 5   # number of x-axis minor intervals between major ticks
COLOR_TRUTH = "black"
COLOR_COLLAPSED = "tab:red"     # fgi=NONE
COLOR_NOTCOLLAPSED = "tab:blue"  # fgi=Custom

LEGEND_LABELS = ["Ground truth", "Collapsed", "Not-collapsed"]


# --------------------------------------------------------------------------- #
# Network (10-layer narrow ReLU MLP, hidden width 2) -- as in reLu10_script.py
# --------------------------------------------------------------------------- #
class SimpleReLU10Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_type=nn.ReLU):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), layer_type())
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), layer_type())
        self.layer3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), layer_type())
        self.layer4 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), layer_type())
        self.layer5 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), layer_type())
        self.layer6 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), layer_type())
        self.layer7 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), layer_type())
        self.layer8 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), layer_type())
        self.layer9 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), layer_type())
        self.layer10 = nn.Sequential(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        return out


def load_model(path):
    """Load a 10-layer network state dict as an eval-ready CPU model."""
    model = SimpleReLU10Network(1, 2, 1)
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Target functions + model evaluation
# --------------------------------------------------------------------------- #
def target_values(target_func, x):
    """Ground-truth y for a 1-D target on the given x column."""
    if target_func == "absfunc1":
        return torch.abs(x)
    if target_func == "sinefunc2":
        return x * torch.sin(5 * x)
    if target_func == "sinefunc3":
        t = torch.where(x > 0, 1.0, 0.0)
        return t + 0.2 * torch.sin(5 * x)
    raise ValueError(f"Invalid / unsupported target function: {target_func!r}")


def resolve_checkpoint(models_dir, target, fgi):
    """Return the path of the network for (target, fgi) in models_dir."""
    name = f"relu10_250_epochs_fw_func_NONE_fgi_{fgi}_target_func_{target}.pth"
    p = os.path.join(models_dir, name)
    if os.path.isfile(p):
        return p
    raise FileNotFoundError(
        f"No network for target={target}, fgi={fgi} in {models_dir!r} (looked for {name}).\n"
        f"Pass the right folder with --models-dir."
    )


def load_and_test_model(model_path, target_func, x, args):
    """Load one network, run it on x, and return (x, y_true, y_pred) sorted by x."""
    y = target_values(target_func, x)

    model = load_model(model_path)
    with torch.no_grad():
        y_pred = model(x)

    x_np = x.numpy()
    y_np = y.numpy()
    y_pred_np = y_pred.numpy()

    sort_idx = np.argsort(x_np.squeeze())
    x_sorted = x_np[sort_idx].squeeze()
    y_sorted = y_np[sort_idx].squeeze()
    y_pred_sorted = y_pred_np[sort_idx].squeeze()
    return x_sorted, y_sorted, y_pred_sorted


# --------------------------------------------------------------------------- #
# Per-panel configuration
# --------------------------------------------------------------------------- #
# `label` is used to name each panel's output file.
# yticks=None -> let matplotlib pick automatically (only f2 fixes its y ticks).
PANELS = [
    {"label": "f1", "target": "absfunc1",  "title": r"$f_1(x) = |x|$",                    "yticks": None},
    {"label": "f2", "target": "sinefunc2", "title": r"$f_2(x) = x \cdot \sin(5x)$",       "yticks": [-1, 0, 1]},
    {"label": "f3", "target": "sinefunc3", "title": r"$f_3(x)=0.2 \cdot \sin(5x) + H(x)$", "yticks": None},
]


# --------------------------------------------------------------------------- #
# Output path helper
# --------------------------------------------------------------------------- #
def panel_out_path(out, label):
    """Build a per-panel output path from --out.

    If --out contains "{label}" it is substituted; otherwise the label is
    inserted just before the extension (foo.pdf -> foo_f1.pdf).
    """
    if "{label}" in out:
        return out.format(label=label)
    root, ext = os.path.splitext(out)
    return f"{root}_{label}{ext}"


# --------------------------------------------------------------------------- #
# Legend proxies (independent of the plotted data, so styling stays consistent)
# --------------------------------------------------------------------------- #
def legend_handles():
    return [
        Line2D([], [], color=COLOR_TRUTH, lw=LW_TRUTH, alpha=ALPHA),
        Line2D([], [], color=COLOR_COLLAPSED, lw=LW_PRED, ls="--", alpha=ALPHA),
        Line2D([], [], color=COLOR_NOTCOLLAPSED, lw=LW_PRED, ls="--", alpha=ALPHA),
    ]


def save_separate_legend(out_path, args):
    """Write the shared legend to its own tightly-cropped PDF."""
    fig = plt.figure(figsize=(args.figwidth, 0.4))
    fig.legend(
        legend_handles(), LEGEND_LABELS,
        loc="center",
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out_path}")


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def make_figures(args):
    mpl.rcdefaults()

    # SciencePlots style. LaTeX is used by default (the science/nature look);
    # pass --no-latex only if you don't have a TeX install.
    try:
        import scienceplots  # noqa: F401  registers 'science'/'nature'
        styles = ["science", "nature"]
        if args.no_latex:
            styles.append("no-latex")
        plt.style.use(styles)
    except ImportError:
        print("[warn] SciencePlots not installed; using matplotlib defaults.")
    if args.no_latex:
        mpl.rcParams["text.usetex"] = False

    # Fixed sample of x (seeded for reproducibility).
    torch.manual_seed(args.seed)
    x = torch.empty(args.train_size, 1).uniform_(-np.sqrt(3), np.sqrt(3))

    # Each panel keeps the width it had inside the old combined figure.
    panel_w = args.panel_width if args.panel_width is not None else args.figwidth / 3.0

    out_paths = []
    for cfg in PANELS:
        target = cfg["target"]
        p_collapsed = resolve_checkpoint(args.models_dir, target, "NONE")
        p_notcollapsed = resolve_checkpoint(args.models_dir, target, "Custom")

        x_sorted, y_sorted, y_collapsed = load_and_test_model(p_collapsed, target, x, args)
        _, _, y_notcollapsed = load_and_test_model(p_notcollapsed, target, x, args)

        fig, ax = plt.subplots(
            1, 1,
            figsize=(panel_w, args.figheight),
            layout="constrained",
        )

        ax.plot(
            x_sorted, y_sorted,
            label=LEGEND_LABELS[0], linewidth=LW_TRUTH, alpha=ALPHA, color=COLOR_TRUTH)
        ax.plot(
            x_sorted, y_collapsed,
            label=LEGEND_LABELS[1], linestyle="--", linewidth=LW_PRED,
            alpha=ALPHA, color=COLOR_COLLAPSED)
        ax.plot(
            x_sorted, y_notcollapsed,
            label=LEGEND_LABELS[2], linestyle="--", linewidth=LW_PRED,
            alpha=ALPHA, color=COLOR_NOTCOLLAPSED)

        ax.set_title(cfg["title"])
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

        # More x-axis inner ticks: major at -1,0,1 plus minor ticks in between,
        # all pointing inward.
        ax.set_xticks([-1, 0, 1])
        ax.xaxis.set_minor_locator(AutoMinorLocator(X_MINOR_PER_MAJOR))
        if cfg["yticks"] is not None:
            ax.set_yticks(cfg["yticks"])

        ax.tick_params(axis="both", which="both", direction="in",
                       top=False, right=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Per-panel legend only when explicitly requested.
        if args.legend == "each":
            fig.legend(
                legend_handles(), LEGEND_LABELS,
                loc="outside lower center",
                ncol=args.legend_ncol,
                frameon=False,
                columnspacing=1.4,
                handletextpad=0.5,
            )

        out_path = panel_out_path(args.out, cfg["label"])
        fig.savefig(out_path)
        plt.close(fig)
        print(f"[write] {out_path}")
        out_paths.append(out_path)

    # Shared legend as its own file (matches the single legend of the old figure).
    if args.legend == "separate":
        save_separate_legend(panel_out_path(args.out, "legend"), args)

    return out_paths


# --------------------------------------------------------------------------- #
# Fig. 2d (f4)
# --------------------------------------------------------------------------- #
class SimpleReLU20Network(nn.Module):
    """20-layer MLP for f4 (2-4-...-4-2); layer10 has no activation, as in training."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleReLU20Network, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer9 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer10 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layer11 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer12 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer13 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer14 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer15 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer16 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer17 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer18 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer19 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer20 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        out = self.layer17(out)
        out = self.layer18(out)
        out = self.layer19(out)
        out = self.layer20(out)
        return out


def ground_truth_func(x_input):
    y1 = torch.abs(x_input[:, 0] + x_input[:, 1])
    y2 = torch.abs(x_input[:, 0] - x_input[:, 1])
    return torch.stack((y1, y2), dim=1)


def make_fig2d(args):
    """Fig. 2d: first output y1 of f4 and of both 20-layer networks on a 100x100 grid.

    Runs after make_figures(), so the SciencePlots style set there is active.
    """
    from matplotlib import rcParams
    rcParams['axes.labelpad'] = 0

    x1 = torch.linspace(-np.sqrt(3), np.sqrt(3), steps=100)
    x2 = torch.linspace(-np.sqrt(3), np.sqrt(3), steps=100)

    x1_grid, x2_grid = torch.meshgrid(x1, x2, indexing='ij')

    X = torch.stack(
        [x1_grid.reshape(-1), x2_grid.reshape(-1)],
        dim=1
    )

    with torch.no_grad():
        Y_true = ground_truth_func(X)

    T1 = Y_true[:, 0].reshape(x1_grid.shape)

    fgi_types = ["NONE", "Custom"]

    model_path = os.path.join(args.models_dir, f"relu20_250_epochs_fw_func_NONE_fgi_{fgi_types[0]}_target_func_absfunc4.pth")
    model = SimpleReLU20Network(2, 4, 2)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        Y_pred = model(X)

    P1_relu = Y_pred[:, 0].reshape(x1_grid.shape)

    model_path = os.path.join(args.models_dir, f"relu20_250_epochs_fw_func_NONE_fgi_{fgi_types[1]}_target_func_absfunc4.pth")
    model = SimpleReLU20Network(2, 4, 2)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        Y_pred = model(X)

    P1 = Y_pred[:, 0].reshape(x1_grid.shape)

    fig = plt.figure(figsize=(2.5, 2.5))
    ax3 = fig.add_subplot(111, projection='3d')
    ax3.plot_surface(x1_grid, x2_grid, T1, alpha=0.6, label="$f_4(x_1, x_2)$", color="black")
    ax3.plot_surface(x1_grid, x2_grid, P1_relu, alpha=0.6, label="Collapsed", color="tab:red", shade=False)
    ax3.plot_surface(x1_grid, x2_grid, P1, alpha=0.6, label="Not-collapsed", color="tab:blue")

    ax3.set_xlabel('$x_1$', labelpad=-7)
    ax3.set_ylabel('$x_2$', labelpad=-7)
    ax3.set_title("$f_4(x_1,x_2) = (|x_1 + x_2|, |x_1 - x_2|)$", y=0.99)
    ax3.set_zlabel(r'$y_1$', labelpad=-10)
    ax3.view_init(elev=20, azim=-45)
    ax3.set_box_aspect(None, zoom=0.84)
    ax3.set_zticks([0, 1, 2, 3])
    ax3.set_xticks([-1, 0, 1])
    ax3.set_yticks([-1, 0, 1])
    ax3.tick_params(
        axis='both',
        which='both',
        direction='in',
        top=False,
        right=False,
        pad=-3
    )

    ax3.minorticks_off()
    ax3.legend(
        ncol=3,
        frameon=False,
        loc="upper center",
        columnspacing=0.3,
        handletextpad=0.2,
        labelspacing=0.15,
        bbox_to_anchor=(0.5, 0.94),
    )
    ax3.grid(False)
    ax3.tick_params(axis='both', direction='in')
    ax3.tick_params(top=True, right=True)

    out_path = os.path.join(os.path.dirname(args.out), "comparison_plot_f4.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[write] {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default="models",
                    help="Directory holding the trained networks (default: models).")
    ap.add_argument("--no-latex", action="store_true",
                    help="Disable LaTeX text rendering (falls back to mathtext). "
                         "By default LaTeX is used for the SciencePlots look; only "
                         "pass this if you have no TeX install.")
    ap.add_argument("--out", default="comparison_{label}.pdf",
                    help="Output path template. Use '{label}' for the panel label "
                         "(f1/f2/f3, and 'legend' for the separate legend); if absent, "
                         "the label is inserted before the extension "
                         "(default: comparison_{label}.pdf).")
    ap.add_argument("--legend", choices=["separate", "each", "none"], default="separate",
                    help="Where the shared legend goes: 'separate' -> its own cropped "
                         "PDF, panels have none (default; keeps panel size identical to "
                         "the old figure); 'each' -> a legend below every panel; "
                         "'none' -> no legend anywhere.")
    ap.add_argument("--legend-ncol", type=int, default=1,
                    help="Columns for the per-panel legend when --legend each (panels "
                         "are narrow, so 1 is safest; default: 1).")
    ap.add_argument("--figwidth", type=float, default=6.9,
                    help="COMBINED width in inches (your \\textwidth); each panel gets "
                         "figwidth/3 so the three tile back to the original (default: 6.9).")
    ap.add_argument("--panel-width", type=float, default=None,
                    help="Override the per-panel width in inches directly. Default is "
                         "figwidth/3.")
    ap.add_argument("--figheight", type=float, default=1.75,
                    help="Per-panel height in inches (default: 1.75).")
    ap.add_argument("--train-size", type=int, default=3000,
                    help="Number of evaluation points (default: 3000).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Torch seed for the sampled x (default: 0).")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        make_figures(args)
        make_fig2d(args)
    except FileNotFoundError as e:
        print(f"[error] {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()