# SUGAR – dying-ReLU toy regression experiments

Code for the toy-problem results of the paper: **Fig. 2a–e**, **Table 1** and **Supplementary Fig. S2a–c**.
The numerical source data of these display items are provided with the paper as Supplementary Data.

Deep, narrow ReLU MLPs are trained on simple regression targets. A run counts as *collapsed* when the
network ends up computing a constant function. We compare plain ReLU (He initialisation), ReLU with
randomized asymmetric initialisation (RAI, Lu et al. 2020), and SUGAR with the B-SiLU surrogate gradient
(He initialisation).

| Target (code name) | Function | Network |
|---|---|---|
| f1 (`absfunc1`) | \|x\| | 10 layers, base width 2 |
| f2 (`sinefunc2`) | x · sin(5x) | 10 layers, base width 2 |
| f3 (`sinefunc3`) | 0.2 · sin(5x) + H(x) | 10 layers, base width 2 |
| f4 (`absfunc4`) | (\|x1 + x2\|, \|x1 − x2\|) | 20 layers, base width 4 |

Code flags: `--fgi NONE` = standard ReLU backward pass; `--fgi Custom` = SUGAR with B-SiLU
(`relu_fgi.py`); `--rai` = RAI initialisation (`rai_init.py`).

## Files

| File | Purpose |
|---|---|
| `relu10_further_exp.py` | Trains one configuration (`--widths`) for `--n_runs` independent runs; seeded (`--seed 42`), data and initialisation identical across `--fgi` settings |
| `run_capacity_sweep_parallel_f1-3.sh`, `run_capacity_sweep_parallel_f1-3_rai.sh`, `run_capacity_sweep_parallel.sh` | Launch the capacity sweeps (f1–f3; f1–f3 with RAI; f4) in parallel |
| `collapse_rate.py` | Collapse rate vs. model capacity from the sweep results → Fig. 2e, Fig. S2a–c, Table 1 |
| `reLu10_script.py`, `reLu20_script.py` | Train the single example networks shown in Fig. 2a–c (10 layers) and Fig. 2d (20 layers) |
| `plot_collapse_split.py` | Plots Fig. 2a–d (one PDF per panel) from the example networks |
| `relu_fgi.py`, `rai_init.py` | SUGAR surrogate-gradient layers; RAI initialisation |
| `models/` | The eight trained example networks used in Fig. 2a–d, as PyTorch state dicts (optional, ~150 KB) |
| `requirements.txt` | Python dependencies |

## Environment

Python 3.12 with torch 2.13, numpy 2.5, scipy 1.18, matplotlib 3.11, SciencePlots 2.2. The figures use the
SciencePlots `science`/`nature` styles, which need a LaTeX installation. The published figures were made with
matplotlib 3.9.2; newer versions shift the axes of Fig. 2a–c by up to ~0.13 pt. The capacity sweep runs on the CPU;
the example-network scripts use a GPU when one is available.

```bash
pip install -r requirements.txt   # plus a LaTeX installation, or pass --no-latex to plot_collapse_split.py
```

All commands below are run from this folder.


## Fig. 2a–d: example networks

`plot_collapse_split.py` loads the networks from `models/`. Run it from this folder:

```bash
python plot_collapse_split.py --models-dir models --legend each --legend-ncol 2   # Fig. 2a–d
```

writes `comparison_f1.pdf`, `comparison_f2.pdf`, `comparison_f3.pdf` and `comparison_plot_f4.pdf` (Fig. 2a–d).
The input points of Fig. 2a–c are drawn with a fixed seed (`--seed 0`); Fig. 2d shows the first output y1 on a
100 × 100 grid. With matplotlib 3.9.2 the output matches the published panels. With the default
`--legend separate`, panels 2a–c have no legend and the legend is written to `comparison_legend.pdf`.

`models/` holds the networks used in the paper: for each target, `fgi_NONE` (plain ReLU, collapsed) and
`fgi_Custom` (SUGAR B-SiLU, not collapsed).

To train new example networks:

```bash
python reLu10_script.py --fgi <NONE|Custom> --fw_func NONE --target_func <absfunc1|sinefunc2|sinefunc3>
python reLu20_script.py --fgi <NONE|Custom> --fw_func NONE --target_func absfunc4
```

They are written to `models_retrained/` under the names `plot_collapse_split.py` expects;
pass `--models-dir models_retrained` to plot them. Training is seeded (`--seed`, default 42), so repeated runs
on the same device give identical networks; CPU and GPU runs differ. Whether a particular network collapses
depends on the seed, so a retrained ReLU network need not collapse, and vice versa. The networks in `models/`
come from earlier, unseeded runs.

## Fig. 2e, Supplementary Fig. S2, Table 1: capacity sweep

1. Train. Each configuration is one call of the form

   ```bash
   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 relu10_further_exp.py \
       --fgi <NONE|Custom> --target_func <absfunc1|sinefunc2|sinefunc3|absfunc4> \
       --widths <w1,...,wL> --n_runs 1000 --num_epochs 250 [--rai]
   ```

   Run these scripts to cover all published capacities without editing their settings:

   ```bash
   bash run_capacity_sweep_parallel_f1-3.sh       # f1–f3: ReLU and B-SiLU
   bash run_capacity_sweep_parallel_f1-3_rai.sh   # f1–f3: RAI
   bash run_capacity_sweep_parallel.sh           # f4: ReLU and B-SiLU
   bash run_capacity_sweep_parallel_rai.sh       # f4: RAI
   ```

   For f1–f3, each method covers 19 configurations: widths 2 → 3, then
   3 → 4 over nine hidden layers, without repeating the shared width-3
   configuration. For f4, each method covers 20 configurations: widths
   4 → 5 over 19 hidden layers. Each step widens one neuron, starting
   with the first hidden layer.

   Results are written to `results/` (one `.npy` per configuration, 1000 runs each), the final network of each
   configuration to `models_sweep/`, per-config logs to `outputs/` and console logs to `logs/`. This takes days of
   CPU time. For a quick smoke test, override the run count and epochs, e.g.
   `N_RUNS=4 EPOCHS=3 bash run_capacity_sweep_parallel_f1-3.sh`.

2. Collect the `.npy` files per target into `results/absfunc1`, `results/sinefunc2`, `results/sinefunc3`,
   `results/absfunc4_relu20`. Then run:

   ```bash
   python collapse_rate.py --data-dir results/absfunc1        --out-dir results/absfunc1/figures         # Fig. 2e
   python collapse_rate.py --data-dir results/sinefunc2       --out-dir results/sinefunc2/figures        # Fig. S2a
   python collapse_rate.py --data-dir results/sinefunc3       --out-dir results/sinefunc3/figures        # Fig. S2b
   python collapse_rate.py --data-dir results/absfunc4_relu20 --out-dir results/absfunc4_relu20/figures  # Fig. S2c
   ```

   Each call writes `<target>_collapse_rate_vs_capacity.pdf` (the figure), `collapse_rates.csv` (all points)
   and `threshold_provenance.json`. The base-width "B-SiLU (He-init)" rows of `collapse_rates.csv` are the
   SUGAR column of Table 1; the He-init and RAI columns of Table 1 are from Lu et al. (2020).

   The sweep results of the paper are not included in this code.
