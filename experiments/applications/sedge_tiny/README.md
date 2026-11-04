# SUGAR + S-Edge-Tiny

Code for **Table 2** of the paper. Numerical source data are supplied with
the paper as Supplementary Data. These experiments
compare ReLU with SUGAR B-SiLU, Leaky and NeLU on the 35-class Speech Commands
v0.02 dataset. SUGAR uses ReLU in the forward pass and the selected surrogate
derivative in the backward pass. The ReLU control uses ReLU in both directions.

## Setup

From this folder, create a fresh Python environment and install the package:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Training uses Linux and a CUDA GPU. The reported runs used Python 3.9,
PyTorch/TorchAudio 2.8.0 with CUDA 12.8, and NumPy 2.0.2 on NVIDIA A100 GPUs.
The default is 16 CPU threads and 16 data-loader workers. Allow at least 48 GB
of host memory for the in-memory audio data. Results may vary across software,
hardware and thread counts even when the initial weights and seeds match.
Four reported runs used 128 CPU threads; the remaining 41 used 16, as recorded
per run in the Supplementary Data. Use `--cpu-threads 128` to match those four runs.

## Data

Download and verify the original audio archive:

```bash
python -m sedge.dataset
```

This extracts the WAV files and split lists to `datasets/speech_commands/`.
Use `--archive` to reuse a downloaded `speech_commands_v0.02.tar.gz`, or point
training's `--dataset-dir` at an already extracted copy. Training uses 84,843
examples, validation 9,981, and testing 11,005. Audio is zero-padded to 16,000
samples and normalized using the padded training set's mean and sample SD.
Validation and test ordering follows the supplied lists; training paths are
sorted before shuffling. No augmentation is used.

The five initial states in `data/initialization/` are shared across variants
within each seed and restored by default. Their hashes and the experiment
metadata are in `data/provenance.json`. Generating new weights from the same
seed is not a substitute for these archived initial states.

## Reproduce the experiments

```bash
python -m sedge.experiment
```

This runs **9 configurations x 5 seeds = 45 runs**, with seeds
**1, 10, 20, 25, 42**. Each uses batch size 16, 40 epochs and 212,120 AdamW
updates, batch normalization, no dropout, and the original decoder
orthogonality penalty (coefficient 0.1). The original parameter groups have
nominal learning rates 0.002/0.008 and weight decay 0/0.04. The scalar cosine
schedule sets both groups to the same learning rate: zero initially, a
100-update warmup to 0.008, then cosine decay. Checkpoints are selected by
lowest validation loss, never test accuracy.

For one configuration:

```bash
python -m sedge.train --surrogate nelu --alpha 0.005 --seed 1
python -m sedge.train --surrogate relu --alpha 0 --seed 1
```

Add `--dry-run` to the sweep to inspect its commands without loading data.
To split it across GPUs, launch non-overlapping shards, for example
`--shard 0 --n-shards 2 --device cuda:0` and
`--shard 1 --n-shards 2 --device cuda:1`. Each shard runs sequentially on its
selected GPU; reserve the CPU and memory resources for each process.

Runs write `output/<surrogate>_alpha_<alpha>_seed_<seed>/`, containing the
configuration, epoch history, best-validation-loss checkpoint and final test
result. Existing run directories are never overwritten; use a new
`--output-dir` for another repetition. The fixed protocol reproduces the
reported experiment, not a guarantee of bitwise-identical training.

## Recreate Table 2

Training generates `history.csv` and `result.csv` in each run directory.
After completing the sweep, aggregate the results with:

```bash
python -m sedge.summarize
python -m sedge.summarize --format latex
```

Both commands read `output/` by default; pass another directory as the
positional argument if needed. Aggregation requires exactly one completed run
per configuration and seed and uses the sample standard deviation (`ddof=1`).
Add `--output table2.csv` to save the summary without overwriting an existing
file. The published S-Edge reference (90.50%) is a literature result, not one
of these runs.

Lightweight protocol checks run with `python -m unittest discover -s tests`.

The model is adapted from [S-Edge](https://github.com/embedded-machine-learning/S-Edge)
by Bittner et al., [Machine Learning (2025)](https://doi.org/10.1007/s10994-025-06807-z).
Third-party attribution and licenses are provided in `THIRD_PARTY.md` and `licenses/`.
