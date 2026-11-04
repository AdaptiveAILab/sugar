# SUGAR + MobileNetV2

Code accompanying the results given in Table 3 in the paper. The experiments
apply SUGAR variants to a ImageNet pretrained MobileNetV2 backbone by replacing  
its `ReLU6` function with surrogate-gradient activations during transfer
learning. Two image-classification benchmarks are used: **HAM10000** (skin
lesion classification, 7 classes) and **Oxford102** (flower classification,
102 classes).

Each sweep runs **5 SUGAR configurations × 5 random seeds = 25 runs per
dataset**, with results aggregated to `results/<dataset>.txt`.

## Setup

From the repository root, enter this experiment directory:

```bash
cd experiments/applications/mobilenetv2
```

Then create and activate a fresh environment. Pick one:

```bash
# venv (built-in)
python -m venv .venv && source .venv/bin/activate

# conda
conda create -n sugar-mobilenet python=3.11 -y && conda activate sugar-mobilenet

# uv
uv venv && source .venv/bin/activate
```

Then install the project and its dependencies:

```bash
pip install -e .
```

Requires Python ≥ 3.10. Dependencies (torch, torchvision, numpy, scipy,
Pillow) are declared in `pyproject.toml`.

## Data

Place the datasets inside `experiments/applications/mobilenetv2/data/` as follows:

```
data/
├── ham10000/
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/    # *.jpg
│   └── HAM10000_images_part_2/    # *.jpg
└── oxford102/
    ├── jpg/                       # *.jpg, extracted from 102flowers.tar
    ├── imagelabels.mat
    └── setid.mat
```

- **HAM10000**: download from the Harvard Dataverse (DOI
  [10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T)) and
  extract both image-part archives into `data/ham10000/`. 
  We only need the above given files.
- **Oxford102**: download the three files from the
  [VGG flowers page](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
  by clicking `Dataset images`, `The image labels` and `The data splits` to download `102flowers.tar`, `imagelabels.mat`, `setid.mat` respectively.
Then extract the tarball inside `data/oxford102/` so that the images sit at `data/oxford102/jpg/`.

## Reproduce the results

Each command runs the full 25-run sweep for the given dataset:

```bash
python -m mobilenet.experiment --dataset ham10000
python -m mobilenet.experiment --dataset oxford102
```

Aggregated mean/std accuracies are appended to `results/ham10000.txt` and
`results/oxford102.txt` respectively. Per-run logs and best checkpoints land
under `output/<dataset>/<sugar>_<alpha>_<seed>/`.

To run a single training run instead of a sweep:

```bash
python -m mobilenet.train --dataset ham10000 --sugar nelu --alpha 0.05 --seed 42
```
