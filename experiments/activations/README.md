# SUGAR + VGG-16 and ResNet-18

Code for **Fig. 3a-f** of the paper: validation loss for ResNet-18, test
accuracy for VGG-16, and layer-wise activity profiles for VGG-16 on CIFAR-100.
The numerical source data are supplied with the paper as Supplementary Data.
The original checkpoints and datasets are not included here.

## Setup

Run the following commands from `experiments/activations/`, in a fresh Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requires Python >= 3.10. Dependencies are listed in `pyproject.toml`.
Training uses Linux and CUDA GPUs, including `torch.compile` for the surrogate
layers. Plotting and configuration checks also work on a CPU. CIFAR-100 is
downloaded automatically to `data/`; use `--data-dir` to reuse an existing copy.
Training results can vary across PyTorch/CUDA versions and hardware even with
the same seeds. The plotting code was checked with Matplotlib 3.10.8.
Analysis records activation-input gradients with tensor hooks, including when
SUGAR layers are compiled. Lightweight checks run with
`python -m unittest discover -s tests`.

## Reproduce the experiments

```bash
python -m cnn.experiment
python -m cnn.experiment --config cnn/configs/activations.yaml
```

The first command runs the CIFAR-100 portion of the original training sweep:
210 runs across both models, SUGAR and non-SUGAR activations, and seeds
**1, 10, 20, 25, 42**. The second runs the two VGG-16 activity-profile
experiments (ReLU and SUGAR B-SiLU, seed **1**). These are separate runs because
collecting activity statistics changes the random-number sequence during
training. Configurations with an existing `.pt` result are skipped.

To reproduce the CIFAR-10 results in Supplementary Table S1, copy
`cnn/configs/training.yaml` to `cnn/configs/cifar10.yaml` and change
`datasets: [cifar100]` to `datasets: [cifar10]`. Then run:

```bash
python -m cnn.experiment --config cnn/configs/cifar10.yaml
```

Both models use 50 epochs for CIFAR-10; the existing CIFAR-100 epoch settings
in the copied configuration do not apply to this dataset. CIFAR-10 is downloaded
automatically, and results are saved under `results/cifar10/`.

VGG-16 uses 50 epochs and ResNet-18 uses 100 epochs for CIFAR-100. Both sweeps use batch
size 128, a seeded 10% validation split, and SGD
with learning rate 0.001, momentum 0.9, Nesterov acceleration and weight decay
0.0001. The learning-rate milestone is 100 (gamma 0.1). Training images are
randomly flipped horizontally and normalized; validation and test images are
only normalized. Both models use batch normalization; VGG-16 also uses
dropout (0.5). The original initialization, surrogate definitions and
analysis passes are retained in `cnn/train.py`, `cnn/sugar.py` and the model
files. `fgi` in code and result paths means SUGAR; `param_swish` means B-SiLU
with its fixed parameter 1.67. Non-SUGAR runs use the respective activation
in both forward and backward passes; SUGAR keeps ReLU in the forward pass.

By default, runs are distributed over GPUs 0 and 1, with two processes per GPU.
For one GPU, add `--gpus 0` to either command. To inspect the run list without
downloading data or training, add `--dry-run`. For a single configuration:

```bash
python -m cnn.experiment --model vgg16 --activation param_swish --mode sugar --seed 1 --gpus 0
```

`--shard 0 --n-shards 2` and `--shard 1 --n-shards 2` partition a sweep
between two launches; use the same configuration and filters for both.
Do not launch overlapping configurations into the same output directory.

Results are saved as
`results/cifar100/<model>/<fgi|non_fgi>/<activation>/[alpha_<value>/]seed_<seed>/run_acc_<accuracy>.pt`.
They contain validation-loss histories, test accuracy, model weights and
activity measurements. The activity sweep uses `analyze_freq_results/`
instead; `--results-dir` overrides either destination.

## Create Figure 3

After both sweeps finish:

```bash
python -m cnn.plot
```

This reads `results/` and `analyze_freq_results/` and writes vector PDFs to
`plots/`:

| File | Panels |
|---|---|
| `cifar100_resnet18_loss_curves.pdf` | a, b |
| `VGG16_CIFAR100_accuracy_c.pdf` | c |
| `vgg16_cifar100_activations_combined.pdf` | d, e, f |
| `fig3_a.pdf` through `fig3_f.pdf` | Individual panels, without panel letters |

The three combined plots also get PNG previews. Existing archived results
can be used with `--results-dir` and `--analysis-dir`; `--output-dir` changes
the plot destination. Only load trusted `.pt` files: they contain Python
pickle data. Both the original and packaged checkpoint formats are supported.
Plotting requires exactly one result per expected seed and all 100 validation
epochs for the ResNet-18 loss curves, rather than silently averaging incomplete
or duplicate runs.

## Figure details

- Panels a-b show the mean validation loss with a 95% interval calculated as
  mean +/- 1.96 times the sample standard deviation divided by sqrt(5).
  NeLU uses alpha 0.01 and the `nelu` results from the training sweep.
  The combined figure's shared legend reports SUGAR mean test accuracies,
  with ReLU as the baseline. Individual exports instead show each panel's
  own mean test accuracies: non-SUGAR in a and SUGAR in b.
- Panel c uses the population standard deviation (`ddof=0`), matching the
  original figure aggregation. ReLU is the shared baseline.
- Panels d-f use **stored epoch key 40** from the activity sweep. Keys are
  zero-based, so this measurement follows the 41st training epoch; this is
  the original data selection, not a new measurement at epoch 40.
- Activity counts accumulate positive responses at each neuron/spatial
  position over the 45,000 training images. Panels d-e use 100 histogram
  bins, each normalized to unit total frequency. Panel f uses 100 common
  bins over 0-45,001, smooths with the kernel `[1, 2, 3, 2, 1] / 9`, and
  divides both curves by their shared peak within each layer. Its heights
  compare the two models within a layer, not absolute frequencies across
  layers; FC is shown at the top.
