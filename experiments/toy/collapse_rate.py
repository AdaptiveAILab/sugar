#!/usr/bin/env python3
# Paper figures (output file names were kept as produced by this script):
#   --data-dir results/absfunc1         -> absfunc1_collapse_rate_vs_capacity.pdf        = Fig. 2e  (f1)
#   --data-dir results/sinefunc2        -> sinefunc2_collapse_rate_vs_capacity.pdf       = Fig. S2a (f2)
#   --data-dir results/sinefunc3        -> sinefunc3_collapse_rate_vs_capacity.pdf       = Fig. S2b (f3)
#   --data-dir results/absfunc4_relu20  -> absfunc4_relu20_collapse_rate_vs_capacity.pdf = Fig. S2c (f4)
# The baseline-width "B-SiLU (He-init)" rows of collapse_rates.csv are the SUGAR column of Table 1.
"""
collapse_rate.py
================

Estimate the *collapse rate* of deep-narrow ReLU networks as a function of
model capacity, from the per-run loss histories stored in the `.npy` result
files.

Background
----------
Each `.npy` file holds the results of N (=1000) independent training runs for
one network configuration. Its structure is::

    { run_id : { 'loss' : { epoch : loss_value } } }

A *collapsed* run is one that never escapes the trivial / degenerate solution:
its loss stays on a high plateau (approx the variance of the target) instead of
converging. In log-loss space the runs form two well-separated clusters
(converged vs. collapsed), so the **collapse rate** for a file is simply::

    collapse_rate = (# runs with final loss > threshold) / (# runs)

The threshold is read off the data (the *valley* between the two modes, or the
dead-network *plateau*) rather than being hand-picked.

Capacity
--------
The filename encodes the per-layer hidden widths, e.g. `widths_2-2-2-2-2-2-2-2-2`.
The baseline is the smallest width everywhere; capacity is increased by turning
those into larger widths one layer at a time. Capacity is summarised as the
total number of parameters; total hidden neurons and the number of widened
layers are also recorded.

Grouping & thresholds
---------------------
Files are grouped into *series* by everything in the filename except the widths
(activation/#layers, epochs, fw_func, fgi, target, init). Each series becomes
one line `collapse_rate (y) vs capacity (x)`, and different target functions are
drawn in separate panels.

The collapse threshold is, by default, estimated **per target**
(`--threshold-scope per_target`, used for the paper): the baseline-width runs of
all three models of a target are pooled into one threshold. Use `per_model` to
give each model baseline -- (fgi=Custom, init=standard), (fgi=NONE,
init=standard), (fgi=NONE, init=rai) -- its own threshold from its OWN
baseline-width runs, or `global` for one threshold overall.

Usage
-----
    python collapse_rate.py --data-dir /path/to/npy_files --out-dir ./out

Key options: --loss-metric {final,min,mean_lastN}, --threshold-scope
{per_model,per_target,global}, --threshold-method {plateau,valley,manual},
--manual-threshold LOSS, --x {capacity,num_neurons,n_widened}.
Run with -h for the full list.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import re
from collections import defaultdict
import scienceplots
import numpy as np

import matplotlib
matplotlib.use("Agg")

import shutil
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401  — registers 'science'/'nature'
    styles = ["science", "nature"]
    if shutil.which("latex") is None:
        styles.append("no-latex")   # render without a TeX install instead of crashing
    plt.style.use(styles)
except ImportError:
    print("[warn] SciencePlots not installed; using matplotlib defaults.")
    
try:
    from scipy.stats import gaussian_kde
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# Model baselines
# --------------------------------------------------------------------------- #
# A "model baseline" is identified by the (fgi, init) pair. These three are the
# configurations we estimate separate collapse thresholds for.
MODEL_LABELS = {
    ("Custom", "standard"): "B-SiLU (He-init)",
    ("NONE",   "standard"): "ReLU (He-init)",
    ("NONE",   "rai"):      "ReLU (RAI-init)",
}

MODEL_COLORS = {
    ("Custom", "standard"): "tab:blue",   # fgi=Custom, init=standard
    ("NONE",   "standard"): "tab:red",    # fgi=NONE,   init=standard
    ("NONE",   "rai"):      "black",  # fgi=NONE,   init=rai
}

MODEL_ORDER = {
    ("NONE",   "standard"): 0,   # ReLU (He-init)
    ("NONE",   "rai"):      1,   # ReLU (RAI-init)
    ("Custom", "standard"): 2,   # B-SiLU (He-init)
}

def model_label(meta):
    """Human-readable name of the (fgi, init) model baseline for a file."""
    return MODEL_LABELS.get(
        (meta["fgi"], meta["init"]),
        f"fgi={meta['fgi']}, init={meta['init']}",
    )
def model_color(meta, default=None):
    """Fixed plot color for the (fgi, init) model baseline, or `default`."""
    return MODEL_COLORS.get((meta["fgi"], meta["init"]), default)
    
def model_order(meta):
    """Sort index for the (fgi, init) model baseline; unknowns go last."""
    return MODEL_ORDER.get((meta["fgi"], meta["init"]), len(MODEL_ORDER))
    
def group_key(meta, scope):
    """String key identifying the threshold-estimation group for a file.

    'global'     -> one threshold for everything.
    'per_target' -> one threshold per target (pools all models; the default).
    'per_model'  -> one threshold per (target, model) baseline.
    """
    if scope == "global":
        return "__global__"
    if scope == "per_target":
        return meta["target"]
    return f"{meta['target']} | {model_label(meta)}"


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #
# Example:
#   relu10_250_epochs_fw_func_NONE_fgi_Custom_target_func_absfunc1_widths_2-2-2-2-2-2-2-2-2.npy
#   ..._target_func_sinefunc3_rai_init_widths_2-2-2-2-2-2-2-2-2.npy   (RAI init runs)
_FNAME_RE = re.compile(
    r"""^(?P<act>[a-zA-Z]+)(?P<layers>\d+)          # relu10
        _(?P<epochs>\d+)_epochs
        _fw_func_(?P<fw_func>.+?)
        _fgi_(?P<fgi>.+?)
        _target_func_(?P<target>.+?)
        (?P<rai>_rai_init)?                          # optional RAI-init marker
        _widths_(?P<widths>[\d-]+)
        \.npy$""",
    re.VERBOSE,
)


def num_parameters(widths, target_fun):
    """Total number of trainable parameters (weights + biases) of the MLP."""
    inp, out = (2, 2) if target_fun == "absfunc4" else (1, 1)
    total = inp * widths[0] + widths[-1] * out + out          # first/last weights + out bias
    for i in range(1, len(widths)):
        total += widths[i] * widths[i - 1]                    # hidden weight matrices
    total += int(sum(widths))                                 # hidden biases
    return total


def parse_filename(path):
    """Return a dict of metadata parsed from an .npy filename, or None."""
    name = os.path.basename(path)
    m = _FNAME_RE.match(name)
    if not m:
        return None
    g = m.groupdict()
    widths = [int(w) for w in g["widths"].split("-") if w != ""]
    baseline = min(widths) if widths else 2
    init = "rai" if g.get("rai") else "standard"     # RAI vs He/standard weight init
    meta = {
        "path": path,
        "name": name,
        "activation": g["act"],
        "n_layers": int(g["layers"]),
        "epochs": int(g["epochs"]),
        "fw_func": g["fw_func"],
        "fgi": g["fgi"],
        "target": g["target"],
        "init": init,
        "widths": widths,
        "num_neurons": int(sum(widths)),              # total hidden neurons
        "capacity": num_parameters(widths, g["target"]),
        "n_widened": int(sum(w > baseline for w in widths)),
        "widths_str": g["widths"],
    }
    # Series key = everything that identifies a line (i.e. NOT the widths).
    meta["series_key"] = (
        meta["target"], meta["activation"], meta["n_layers"],
        meta["fw_func"], meta["fgi"], meta["init"], meta["epochs"],
    )
    meta["series_label"] = (
        f"fw={meta['fw_func']}, fgi={meta['fgi']}, init={meta['init']} "
        f"({meta['activation']}{meta['n_layers']}, {meta['epochs']}ep)"
    )
    return meta


# --------------------------------------------------------------------------- #
# Loading & per-run loss reduction
# --------------------------------------------------------------------------- #
def _as_dict(obj):
    """Unwrap a possibly 0-d numpy-object-wrapped dict."""
    if isinstance(obj, np.ndarray):
        obj = obj.item()
    return obj


def _num(v):
    """JSON-safe float: NaN/inf -> None, numpy scalar -> Python float."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def per_run_losses(path, metric="final", last_n=10):
    """
    Load one .npy file and return a 1-D array of one scalar loss per run.

    metric:
        'final'      -> loss at the last epoch
        'min'        -> best (minimum) loss over training
        'mean_lastN' -> mean of the last `last_n` epochs (robust to end spikes)
    """
    data = _as_dict(np.load(path, allow_pickle=True))
    out = []
    for run in data.values():
        run = _as_dict(run)
        loss = _as_dict(run["loss"])
        # Order by epoch key so 'final' is truly the last epoch.
        epochs = sorted(loss.keys())
        vals = np.array([loss[e] for e in epochs], dtype=float)
        if metric == "final":
            out.append(vals[-1])
        elif metric == "min":
            out.append(np.nanmin(vals))
        elif metric == "mean_lastN":
            out.append(np.nanmean(vals[-last_n:]))
        else:
            raise ValueError(f"unknown metric {metric!r}")
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# Threshold (valley / antimode of the bimodal log-loss distribution)
# --------------------------------------------------------------------------- #
def find_valley_threshold(losses, grid=512):
    """
    Find a separating threshold between the 'converged' and 'collapsed' modes of
    a bimodal loss distribution.

    Works in log10 space (losses span many orders of magnitude). Returns the
    threshold in the ORIGINAL (linear) loss units, plus a small diagnostics dict.
    """
    x = np.asarray(losses, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return np.nan, {"method": "empty"}
    lx = np.log10(x)
    lo, hi = lx.min(), lx.max()
    if hi - lo < 1e-9:  # degenerate: all runs identical
        return float(10 ** lo), {"method": "degenerate"}

    xs = np.linspace(lo, hi, grid)

    # Density estimate: KDE if available, else a smoothed histogram.
    if _HAVE_SCIPY and x.size >= 5:
        dens = gaussian_kde(lx)(xs)
        method = "kde"
    else:
        counts, edges = np.histogram(lx, bins=min(60, max(10, x.size // 20)))
        centers = 0.5 * (edges[:-1] + edges[1:])
        dens = np.interp(xs, centers, counts.astype(float))
        method = "hist"

    # Peaks = local maxima of the density.
    peaks = [i for i in range(1, grid - 1)
             if dens[i] > dens[i - 1] and dens[i] >= dens[i + 1]]
    if len(peaks) >= 2:
        # Two most prominent peaks, then deepest valley strictly between them.
        peaks.sort(key=lambda i: dens[i], reverse=True)
        p1, p2 = sorted(peaks[:2])
        j = p1 + int(np.argmin(dens[p1:p2 + 1]))
        thr_log = xs[j]
        method += "+valley"
    else:
        # Fallback: Otsu's method on the log-loss (max between-class variance).
        thr_log = _otsu(lx, xs)
        method += "+otsu"

    return float(10 ** thr_log), {
        "method": method, "log_threshold": float(thr_log),
        "log_min": float(lo), "log_max": float(hi),
    }


def _otsu(values, xs):
    """1-D Otsu threshold; returns the log-loss cut maximising between-class var."""
    counts, edges = np.histogram(values, bins=256)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = counts.astype(float) / counts.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    k = int(np.nanargmax(sigma_b2))
    return float(np.interp(centers[k], centers, centers))  # == centers[k]


def find_plateau_threshold(losses, margin=0.03, grid=512):
    """
    Plateau-anchored collapse threshold (faithful to Lu et al. 2020, "Dying ReLU").

    A network that dies collapses to a ~constant function, so its loss piles up
    on a tight high-loss plateau at approx the target's L2 energy. We locate that
    plateau (the high-loss mode) and cut a small margin below it, so that only
    genuinely *dead* runs count as collapsed (matching the paper's 'A' category,
    not the broader 'not-converged' set).

    Steps
    -----
      1. Work in log10 loss.
      2. Find the density modes (KDE). The plateau is the high-loss basin.
      3. Collapse cluster = runs above the antimode (valley) left of that basin.
         plateau_value = median of that cluster (robust to the tight spike).
      4. threshold = plateau_value * (1 - margin).

    Returns (threshold_linear, info). `info['valley_loss']` is the converged/
    intermediate boundary (antimode), reused for the optional A/B/C breakdown.
    """
    x = np.asarray(losses, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return np.nan, {"method": "empty"}
    lx = np.log10(x)
    lo, hi = lx.min(), lx.max()
    if hi - lo < 1e-9:  # every run identical -> single plateau, no margin needed
        P = float(10 ** lo)
        return P * (1 - margin), {"method": "degenerate", "plateau_value": P,
                                  "valley_loss": np.nan}

    xs = np.linspace(lo, hi, grid)
    if _HAVE_SCIPY and x.size >= 5:
        dens = gaussian_kde(lx)(xs)
        method = "kde"
    else:
        counts, edges = np.histogram(lx, bins=min(60, max(10, x.size // 20)))
        centers = 0.5 * (edges[:-1] + edges[1:])
        dens = np.interp(xs, centers, counts.astype(float))
        method = "hist"

    # Local maxima of the density, keeping only PROMINENT ones (>=5% of the
    # tallest peak) so KDE noise bumps don't count as modes. Boundary points are
    # included as candidates because the dead-network plateau is a tight spike
    # that often sits at the HIGH-LOSS EDGE, where the interior local-max test
    # would miss it.
    peaks = [i for i in range(1, grid - 1)
             if dens[i] > dens[i - 1] and dens[i] >= dens[i + 1]]
    if dens[grid - 1] > dens[grid - 2]:
        peaks.append(grid - 1)                 # rising into the right edge = edge plateau
    if dens[0] > dens[1]:
        peaks.insert(0, 0)
    peaks = sorted(set(peaks))
    if peaks:
        dmax = max(dens[i] for i in peaks)
        prom = [i for i in peaks if dens[i] >= 0.05 * dmax]   # ascending in loss
    else:
        prom = []

    # PLATEAU = the rightmost (highest-loss) prominent mode = the dead-network
    # constant-function loss. Anchoring to the rightmost mode (not "the two most
    # prominent") keeps the collapse cut just below the true plateau even when a
    # large INTERMEDIATE / partial-fit mode sits between converged and plateau
    # (e.g. absfunc4). hi_valley is the antimode just below the plateau; the
    # collapse cluster is only that top basin, so its median is the plateau.
    if len(prom) >= 2:
        hi_valley_log = xs[prom[-2] + int(np.argmin(dens[prom[-2]:prom[-1] + 1]))]
        method += "+rightmode"
    else:
        hi_valley_log = max(lo, hi - 0.5)   # single mode: base of the top spike
        method += "+toptail"

    # CONVERGED = the leftmost prominent mode. lo_valley is the antimode just
    # above it, so the converged cut hugs the converged mass and does NOT get
    # dragged up to an intermediate mode.
    if len(prom) >= 2:
        lo_valley_log = xs[prom[0] + int(np.argmin(dens[prom[0]:prom[1] + 1]))]
    else:
        lo_valley_log = hi_valley_log

    cluster = x[np.log10(x) >= hi_valley_log]
    if cluster.size == 0:
        cluster = x[np.log10(x) >= hi - 1e-9]
    plateau_value = float(np.median(cluster))

    # --- collapse cut: sit in the EMPTY BAND just below the plateau spike ----
    # The dead-network plateau is a tight, tall spike; anchoring the cut a fixed
    # few-percent below it can land visually *on* the spike (e.g. sinefunc3).
    # Instead we place the cut at the geometric midpoint between the plateau's
    # lower edge and the highest "straggler" beneath it, so the line always
    # falls in empty space while the collapse COUNT is unchanged (the band holds
    # no runs by construction).
    hi_edge = plateau_value * (1.0 - margin)          # assumed lower edge of spike
    below = x[(x >= 10 ** hi_valley_log) & (x < hi_edge)]
    top_straggler = float(below.max()) if below.size else float(10 ** hi_valley_log)
    collapse_cut = float(np.sqrt(top_straggler * hi_edge))

    # --- converged cut ------------------------------------------------------
    # Normally the leftmost prominent mode IS the converged mode, and we cut just
    # above it. But when a large INTERMEDIATE/partial-fit mode is the leftmost
    # prominent one (e.g. absfunc4) and true convergence is rare & sparse, the
    # real converged mode hides BELOW it. Detect that by scanning to the left
    # foot (density minimum) of the leftmost prominent mode: if appreciable mass
    # sits below that foot, cut there; otherwise use the usual just-above logic.
    foot = prom[0] if prom else 0
    while foot > 0 and dens[foot - 1] <= dens[foot]:
        foot -= 1
    foot_loss = 10.0 ** xs[foot]
    mass_below_foot = int(np.sum(x < foot_loss))

    if prom and foot > 0 and mass_below_foot >= 0.01 * x.size:
        # A separate lower (converged) mode exists below an intermediate mode.
        converged_cut = float(foot_loss)
        low = x[x < foot_loss]
        converged_mode = float(np.median(low)) if low.size else np.nan
    else:
        # Leftmost prominent mode is the converged mode: cut just above its bulk.
        # Use a high percentile of the converged mass (robust whether that mode
        # is a tight spike, e.g. sinefunc3, or a broad spread, e.g. absfunc1).
        low = x[x < 10 ** lo_valley_log]
        converged_mode = float(np.median(low)) if low.size else np.nan
        if low.size:
            conv_hi = float(np.percentile(low, 98))       # upper edge of converged mass
            above = low[low > conv_hi]                    # top-2% tail up to the valley
            bottom_straggler = float(above.min()) if above.size else float(10 ** lo_valley_log)
            converged_cut = float(np.sqrt(conv_hi * bottom_straggler))
        else:
            converged_cut = float(10 ** lo_valley_log)

    frac_cluster = cluster.size / x.size
    return collapse_cut, {
        "method": method, "plateau_value": plateau_value,
        "converged_mode": converged_mode,
        "converged_cut": converged_cut,
        "valley_loss": float(10 ** hi_valley_log),
        "n_modes": int(len(prom)),
        "cluster_frac": float(frac_cluster), "margin": margin,
        "log_min": float(lo), "log_max": float(hi),
        "warn_no_plateau": frac_cluster < 0.02,
    }


# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #
def discover_metas(args):
    """Find, parse and load every matching .npy file into a list of metadata dicts."""
    paths = sorted(glob.glob(os.path.join(args.data_dir, args.glob)))
    metas = []
    for p in paths:
        m = parse_filename(p)
        if m is None:
            print(f"[skip] unparseable filename: {os.path.basename(p)}")
            continue
        m["losses"] = per_run_losses(p, args.loss_metric, args.last_n)
        metas.append(m)
    if not metas:
        raise SystemExit(
            f"No parseable .npy files in {args.data_dir!r} matching {args.glob!r}")
    return metas


def assign_baseline(metas):
    """Mark each file as baseline-width or not.

    Base width = the smallest per-layer width seen for that target (2 for f1-f3,
    4 for f4). A file is 'baseline' iff EVERY hidden layer is at that base width
    (i.e. nothing widened). This is where the collapse plateau is richest, so it
    gives the most stable threshold estimate.
    """
    base_width = {}
    for m in metas:
        t = m["target"]
        base_width[t] = min(base_width.get(t, min(m["widths"])), min(m["widths"]))
    for m in metas:
        bw = base_width[m["target"]]
        m["base_width"] = bw
        m["is_baseline"] = all(w == bw for w in m["widths"])


def _estimation_files(group_metas, key, anchor, anchor_info):
    """Files whose runs ESTIMATE the threshold for a group.

    With anchor 'baseline', use only the baseline-width files of the group (so a
    per_model group draws its threshold from that model's ~1000 baseline runs
    alone). Fall back to the whole group, with a warning, if no baseline file is
    present. With 'all', use every capacity. The threshold is always APPLIED to
    the whole group afterwards.
    """
    if anchor == "baseline":
        base = [m for m in group_metas if m["is_baseline"]]
        if base:
            anchor_info[key] = {
                "anchored_on": "baseline",
                "base_width": base[0]["base_width"],
                "n_baseline_files": len(base),
                "n_baseline_runs": int(sum(m["losses"].size for m in base)),
            }
            return base
        print(f"[warn] {key}: no baseline-width files found; "
              f"anchoring threshold on ALL capacities instead.")
        anchor_info[key] = {"anchored_on": "all (no baseline found)"}
    else:
        anchor_info[key] = {"anchored_on": "all"}
    return list(group_metas)


def build_thresholds(metas, args):
    """Estimate one collapse threshold per group (see group_key).

    Tags every file with its group key and returns
    (thresholds, anchor_info, est_files), where
      thresholds[key] = (collapse_cut, info),
      est_files[key]  = list of metas used to estimate that threshold.
    """
    method = args.threshold_method
    if method == "manual" and args.manual_threshold is None:
        raise SystemExit("--threshold-method manual requires --manual-threshold")

    def compute(pool):
        if method == "manual":
            thr, info = args.manual_threshold, {"method": "manual"}
        elif method == "plateau":
            thr, info = find_plateau_threshold(pool, margin=args.plateau_margin)
        else:
            thr, info = find_valley_threshold(pool)
        # Always attach converged/plateau anchors so the converged-cut line is
        # drawn for every method (valley/manual don't compute them natively).
        if "converged_cut" not in info:
            _, pinfo = find_plateau_threshold(pool, margin=args.plateau_margin)
            for k in ("converged_cut", "converged_mode", "plateau_value", "valley_loss"):
                info.setdefault(k, pinfo.get(k))
        return thr, info

    groups = defaultdict(list)
    for m in metas:
        m["group"] = group_key(m, args.threshold_scope)
        groups[m["group"]].append(m)

    anchor_info, est_files, thresholds = {}, {}, {}
    for key, group in groups.items():
        pool_metas = _estimation_files(group, key, args.threshold_anchor, anchor_info)
        est_files[key] = pool_metas
        pool = np.concatenate([m["losses"] for m in pool_metas])
        thresholds[key] = compute(pool)
    return thresholds, anchor_info, est_files


def score_runs(metas, thresholds):
    """Fill in collapse rate and the A/B/C split for every file, using that
    file's group threshold."""
    for m in metas:
        thr, info = thresholds[m["group"]]
        losses = m["losses"]
        n = losses.size
        n_col = int(np.sum(losses > thr))
        m["threshold"] = thr
        m["n_runs"] = n
        m["n_collapsed"] = n_col
        m["collapse_rate"] = n_col / n if n else np.nan

        # A = collapsed (> collapse cut), C = converged (<= converged cut),
        # B = intermediate in between.
        conv_cut = info.get("converged_cut", np.nan)
        m["converged_cut"] = conv_cut
        if n and np.isfinite(conv_cut):
            n_conv = int(np.sum(losses <= conv_cut))
            m["rate_converged_C"] = n_conv / n
            m["rate_intermediate_B"] = (n - n_conv - n_col) / n
        else:
            m["rate_converged_C"] = ""
            m["rate_intermediate_B"] = ""


def write_provenance(metas, thresholds, anchor_info, args, out_dir):
    """Write a self-documenting record of how each threshold was derived.

    The exact cut is estimated from the data (the earlier hand-picked constants
    came from an unseeded draw and are not reproducible). The plateau itself is
    seed-independent (approx the target's L2 energy), so we record it alongside
    the derived cuts and how many pooled runs actually sit on it. Returns the
    provenance dict so the console report can reuse it.
    """
    applied = defaultdict(list)
    for m in metas:
        applied[m["group"]].append(m["losses"])

    provenance = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "data_dir": os.path.abspath(args.data_dir),
        "glob": args.glob,
        "loss_metric": args.loss_metric,
        "threshold_method": args.threshold_method,
        "threshold_scope": args.threshold_scope,
        "threshold_anchor": args.threshold_anchor,
        "plateau_margin": args.plateau_margin,
        "note": ("Each threshold is estimated from its own group's plateau (see "
                 "threshold_scope), not the earlier hand-picked constants (which "
                 "came from unseeded runs and are not exactly reproducible). The "
                 "plateau is seed-independent; the empty band around the cut "
                 "means a few-percent shift in the cut changes the collapse "
                 "count by ~0."),
        "groups": {},
    }
    for key, (thr, info) in thresholds.items():
        pool = np.concatenate(applied[key])
        pool = pool[np.isfinite(pool)]
        pv = info.get("plateau_value", np.nan)
        # runs on the plateau spike (within the assumed spike band)
        on_plateau = (int(np.sum(pool >= pv * (1 - args.plateau_margin)))
                      if np.isfinite(pv) else None)
        provenance["groups"][key] = {
            "estimated_from": anchor_info.get(key, {}),
            "n_runs_applied": int(pool.size),
            "n_files_applied": int(len(applied[key])),
            "plateau_value": _num(pv),
            "collapse_cut": _num(thr),
            "converged_cut": _num(info.get("converged_cut", np.nan)),
            "n_on_plateau": on_plateau,
            "frac_on_plateau": _num(on_plateau / pool.size) if on_plateau is not None else None,
            "n_above_collapse_cut": int(np.sum(pool > thr)),
            "method": info.get("method"),
            "warn_no_plateau": bool(info.get("warn_no_plateau", False)),
        }
    path = os.path.join(out_dir, "threshold_provenance.json")
    with open(path, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"[write] {path}")
    return provenance


def write_csv(metas, args, out_dir):
    """Write the per-file collapse-rate summary."""
    path = os.path.join(out_dir, "collapse_rates.csv")
    fields = ["target", "model", "activation", "n_layers", "epochs", "fw_func",
              "fgi", "init", "widths_str", "capacity", "num_neurons", "n_widened",
              "n_runs", "n_collapsed", "collapse_rate", "rate_intermediate_B",
              "rate_converged_C", "converged_cut", "threshold", "loss_metric"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in sorted(metas, key=lambda z: (z["target"], z["capacity"])):
            row = {k: m.get(k, "") for k in fields}
            row["model"] = model_label(m)
            row["loss_metric"] = args.loss_metric
            w.writerow(row)
    print(f"[write] {path}")


def print_report(metas, thresholds, provenance):
    """Print the thresholds and per-file collapse rates to the console."""
    print("\n=== Thresholds (linear loss) ===")
    for key in sorted(thresholds):
        thr, info = thresholds[key]
        prov = provenance["groups"].get(key, {})
        if "plateau_value" in info:
            nop, npool = prov.get("n_on_plateau"), prov.get("n_runs_applied")
            plat = f"{nop}/{npool}" if nop is not None else "n/a"
            anch = prov.get("estimated_from", {})
            anch_str = anch.get("anchored_on", "all")
            if anch.get("n_baseline_runs"):
                anch_str += f"(w={anch.get('base_width')}, n={anch['n_baseline_runs']})"
            line = (f"  {key:>30}: plateau={info['plateau_value']:.4g}"
                    f"  collapse_cut={thr:.4g}"
                    f"  converged_cut={info.get('converged_cut', float('nan')):.4g}"
                    f"  on_plateau={plat}  est_from={anch_str}")
            if info.get("warn_no_plateau"):
                line += "  [WARN: <2% of runs on the plateau - no clear collapse mode]"
        else:
            line = f"  {key:>30}: cut={thr:.4g} [{info.get('method')}]"
        print(line)

    print("\n=== Collapse rates (A=collapsed  B=intermediate  C=converged) ===")
    for m in sorted(metas, key=lambda z: (z["target"], z["capacity"])):
        line = (f"  {m['target']:>8} | {model_label(m):<16} | {m['widths_str']:<20} "
                f"cap={m['capacity']:<4} nwid={m['n_widened']:<2} "
                f"-> collapse {m['collapse_rate']*100:5.1f}% "
                f"({m['n_collapsed']}/{m['n_runs']})")
        if m["rate_converged_C"] != "":
            line += (f"  | A={m['collapse_rate']*100:4.1f}% "
                     f"B={m['rate_intermediate_B']*100:4.1f}% "
                     f"C={m['rate_converged_C']*100:4.1f}%")
        print(line)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _plot_histograms(thresholds, est_files, args, out_png, report_intermediate):
    """One panel per threshold group: the pooled log10(loss) histogram (over the
    exact runs used to estimate the threshold) with the converged and collapse
    cuts drawn on top."""
    title_dict = {"absfunc1": r"$f_1(x) = |x|$",
                 "sinefunc2": r"$f_2(x) = x \cdot \sin(5x) $",
                 "sinefunc3": r"$f_3(x)=0.2 \cdot \sin(5x) + H(x)$",
                 "absfunc4" : r"$f_4(x_1,x_2) = (|x_1 + x_2|, |x_1 - x_2|)$"}
    
    keys = sorted(thresholds)
    ncol = min(3, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.8 * nrow),
                             squeeze=False, layout="constrained")
    for ax in axes.flat:
        ax.set_visible(False)

    log_scale = (args.hist_scale == "log")
    tx = (lambda v: np.log10(v)) if log_scale else (lambda v: v)

    for i, key in enumerate(keys):
        ax = axes.flat[i]
        ax.set_visible(True)
        thr, info = thresholds[key]
        pool = np.concatenate([m["losses"] for m in est_files[key]])
        pool = pool[np.isfinite(pool) & (pool > 0)]
        xvals = tx(pool)
        counts, edges, _ = ax.hist(xvals, bins=60, color="#4C72B0", alpha=0.85, zorder=1)

        ccut = info.get("converged_cut", np.nan)
        x0, x1 = xvals.min(), xvals.max()

        # Snap each drawn cut to a BIN BOUNDARY so it never crosses a bar
        # (counting still uses the true thr/ccut). Collapse -> left edge of the
        # first populated bin at/above the cut; converged -> right edge of the
        # last populated bin at/below the cut.
        def snap_up(c):
            for bi in range(len(counts)):
                if edges[bi] >= c: #  and counts[bi] > 0:
                    return edges[bi]
            return c

        def snap_down(c):
            for bi in range(len(counts) - 1, -1, -1):
                if edges[bi + 1] <= c: #  and counts[bi] > 0:
                    return edges[bi + 1]
            return c

        collapse_x = snap_up(tx(thr)) if (np.isfinite(thr) and thr > 0) else None
        conv_x = snap_down(tx(ccut)) if (np.isfinite(ccut) and ccut > 0) else None

        # Shade the three regions: converged (C) | intermediate (B) | collapsed (A).
        if conv_x is not None and report_intermediate:
            ax.axvspan(x0, conv_x, color="#2E7D32", alpha=0.06, zorder=0)
            ax.axvline(conv_x, color="#00A000", lw=1, ls=(0, (6, 3)), zorder=5,
                       label=f"converged cut = {ccut:.3g}")
        if collapse_x is not None:
            ax.axvspan(collapse_x, x1, color="#C62828", alpha=0.06, zorder=0)
            ax.axvline(collapse_x, color="crimson", lw=1, zorder=5,
                       label=f"collapse cut = {thr:.3g}")

        ax.set_title(f"{title_dict[key]}  (n={pool.size} runs)")
        ax.set_xlabel(r"$\log_{10}(\mathrm{%s\ loss})$" % args.loss_metric
                      if log_scale else f"{args.loss_metric} loss")
        ax.set_ylabel("count")
        ax.tick_params(
            axis='both',
            which='both',
            direction='in',
            top=False,
            right=False
        )
        ax.legend(fontsize=8, loc="upper left")

    scale_note = "log10 loss" if log_scale else "linear loss"
    # fig.suptitle("Loss distribution with converged (green) & collapse (red) cuts  "
    #              f"[C | B | A regions shaded, {scale_note}]", fontsize=12)
    fig.savefig(out_png) # , dpi=150)
    plt.close(fig)
    print(f"[write] {out_png}")


def _plot_collapse_curves(metas, args, out_png):
    """One panel per target: collapse_rate vs capacity, one line per model series."""
    title_dict = {"absfunc1": r"$f_1(x) = |x|$",
                 "sinefunc2": r"$f_2(x) = x \cdot \sin(5x) $",
                 "sinefunc3": r"$f_3(x)=0.2 \cdot \sin(5x) + H(x)$",
                 "absfunc4" : r"$f_4(x_1,x_2) = (|x_1 + x_2|, |x_1 - x_2|)$"}
    targets = sorted({m["target"] for m in metas})
    ncol = min(3, len(targets))
    nrow = int(np.ceil(len(targets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9, 1.7), # 5.6 * ncol, 4.2 * nrow),1.89
                             squeeze=False, sharey=True, layout="constrained")
    for ax in axes.flat:
        ax.set_visible(False)

    xlab = {"num_neurons": "total hidden neurons (capacity)",
            "capacity": "Model capacity",
            "n_widened": "# widened layers"}[args.x]

    for i, t in enumerate(targets):
        ax = axes.flat[i]
        ax.set_visible(True)
        series = defaultdict(list)
        for m in metas:
            if m["target"] == t:
                series[(m["series_key"], m["series_label"])].append(m)
        for (skey, slabel), members in sorted(
                series.items(), key=lambda kv: (model_order(kv[1][0]), kv[0][1])):
            members.sort(key=lambda z: z[args.x])
            xs = [z[args.x] for z in members]
            ys = [100 * z["collapse_rate"] for z in members]
            style = "o" if len(xs) < 2 else "-o"
            color = members[0].get("color") or model_color(members[0])
            ax.plot(xs, ys, style, ms=2, color=color, label=model_label(members[0]), alpha=0.8, linewidth=1., markeredgecolor=color, markeredgewidth=0.01)
        ax.set_title(title_dict[t])
        ax.set_xlabel(xlab)
        ax.set_ylabel(r"Collapse rate (\%)")
        ax.set_ylim(-3, 103)
        ax.tick_params(
            axis='both',
            which='both',
            direction='in',
            top=False,
            right=False
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7)

    fig.savefig(out_png) # , dpi=150)
    plt.close(fig)
    print(f"[write] {out_png}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=".", help="Directory containing .npy files.")
    ap.add_argument("--glob", default="*.npy", help="Filename glob within --data-dir.")
    ap.add_argument("--out-dir", default="./collapse_out", help="Where to write outputs.")
    ap.add_argument("--loss-metric", choices=["final", "min", "mean_lastN"],
                    default="final", help="How to reduce each run's loss curve to a scalar.")
    ap.add_argument("--last-n", type=int, default=10, help="Window for mean_lastN.")
    ap.add_argument("--x", choices=["capacity", "num_neurons", "n_widened"], default="capacity",
                    help="x-axis: #parameters, total hidden neurons, or number of widened layers.")
    ap.add_argument("--hist-scale", choices=["log", "linear"], default="log",
                    help="Histogram x-axis: 'log' plots log10(loss) (default; needed to "
                         "see the converged mode, which spans many orders of magnitude); "
                         "'linear' plots the original loss. Threshold detection is "
                         "unaffected -- it is always done in log.")
    ap.add_argument("--threshold-scope", choices=["per_model", "per_target", "global"],
                    default="per_target",
                    help="Group over which ONE collapse threshold is estimated. "
                         "'per_model': a separate threshold for each "
                         "(target, model) baseline -- (fgi=Custom, init=standard), "
                         "(fgi=NONE, init=standard), (fgi=NONE, init=rai) -- so each "
                         "model's baseline runs are NOT pooled with the other models'. "
                         "'per_target' (default, used for the paper): one threshold "
                         "per target (pools all models). "
                         "'global': one threshold for everything.")
    ap.add_argument("--threshold-anchor", choices=["baseline", "all"],
                    default="baseline",
                    help="Which runs ESTIMATE the threshold within each group. "
                         "'baseline' uses only the baseline-width config (every hidden "
                         "layer at the base width: 2 for f1-f3, 4 for f4), where the "
                         "collapse plateau is richest, then applies that one cut across "
                         "all capacities. 'all' pools every capacity to estimate. The "
                         "cut is always APPLIED to all files in the group.")
    ap.add_argument("--threshold-method", choices=["plateau", "valley", "manual"],
                    default="plateau",
                    help="plateau: cut just below the dead-network plateau (Lu et al., "
                         "counts only truly collapsed runs). valley: cut in the mid-gap "
                         "antimode (counts all not-converged runs). manual: use "
                         "--manual-threshold.")
    ap.add_argument("--plateau-margin", type=float, default=0.03,
                    help="Fractional margin below the plateau for --threshold-method plateau "
                         "(0.03 reproduces the hand-picked absfunc1 cut of 15.58).")
    ap.add_argument("--report-intermediate", action="store_true",
                    help="Also report the A/B/C breakdown: A=collapsed (>plateau cut), "
                         "C=converged (<=valley), B=intermediate in between.")
    ap.add_argument("--manual-threshold", type=float, default=None,
                    help="LINEAR loss cut used when --threshold-method manual.")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    metas = discover_metas(args)
    assign_baseline(metas)

    thresholds, anchor_info, est_files = build_thresholds(metas, args)
    score_runs(metas, thresholds)

    provenance = write_provenance(metas, thresholds, anchor_info, args, args.out_dir)
    write_csv(metas, args, args.out_dir)

    from os.path import normpath, basename
    target = basename(normpath(args.data_dir))
        
    _plot_histograms(thresholds, est_files, args,
                     os.path.join(args.out_dir, f"{target}_loss_histograms.pdf"), args.report_intermediate)
    # Paper figure: Fig. 2e (absfunc1) / Fig. S2a-c (sinefunc2, sinefunc3, absfunc4_relu20).
    _plot_collapse_curves(metas, args,
                          os.path.join(args.out_dir, f"{target}_collapse_rate_vs_capacity.pdf"))

    print_report(metas, thresholds, provenance)


if __name__ == "__main__":
    main()