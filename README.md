<p align="center">
  <img src="assets/SUGAR_light.png#gh-light-mode-only"
       alt="SUGAR" width="100%">
  <img src="assets/SUGAR_dark.png#gh-dark-mode-only"
       alt="SUGAR" width="100%">
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)


Experimental code for the paper **The Resurrection of the ReLU**

Surrogate gradient for ReLU (SUGAR) keeps ReLU in the forward pass and uses a surrogate gradient in the backward pass, allowing learning signals to pass through inactive units.

## Minimal implementation

Using only PyTorch, replace `nn.ReLU()` in your model with `SUGAR_NeLU()`.
The forward pass remains ReLU and the backward pass uses the derivative of NeLU
with `alpha=0.01`.

```python
import torch
from torch import nn
import torch.nn.functional as F

class SUGAR_NeLU(nn.Module):
    def forward(self, x, alpha=0.01):
        gx = torch.where(x > 0, x, -alpha * torch.reciprocal(1 + x.square()))
        return gx - gx.detach() + F.relu(x).detach()
```


A tiny CNN for 28 × 28 grayscale images:

```python
model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),
    SUGAR_NeLU(),  # use instead of nn.ReLU()
    nn.MaxPool2d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(8 * 14 * 14, 10),
)
```

## Optional package

For an interface to apply SUGAR to your own models, use
[sugar-pytorch](https://github.com/AdaptiveAILab/sugar-pytorch), included as the
`sugar/` submodule. From this repository's root:

```bash
git submodule update --init sugar
pip install -e ./sugar
```

See the linked repository for usage. The experiments below use their own
implementations and do not require this package.

## Repository overview

| Experiment | Paper results | Instructions |
|---|---|---|
| Toy regression and network collapse | Figure 2, Table 1, Supplementary Figure S2 | [Toy experiments](experiments/toy/README.md) |
| CNN generalization and activation profiles | Figure 3, Supplementary Tables S1–S2 | [VGG-16 and ResNet-18](experiments/activations/README.md) |
| Audio classification | Table 2 | [S-Edge-Tiny](experiments/applications/sedge_tiny/README.md) |
| Transfer learning | Table 3 | [MobileNetV2](experiments/applications/mobilenetv2/README.md) |

## Reproducing the experiments

Choose an experiment above and follow its README for installation, dataset
preparation, training and result generation. Use a separate environment for
each experiment, since dependencies differ. Run commands from the directory
specified in each guide.

Datasets are downloaded separately as described in the guides. Note that training
results may slightly vary across hardware and software versions, even with fixed seeds.

## Figure generation

The [`plots/`](plots/README.md) directory contains the notebook and script for
Figure 1d and Supplementary Figure S1, showing activation functions and their
backward rules. These require no training or downloaded datasets; see the
linked README for setup and execution instructions.

## Citation

If you find this repository helpful and use SUGAR in your research, please cite:

- Coşku Can Horuz, Geoffrey Kasenbacher et al. (2025). **The Resurrection of the ReLU**. arXiv preprint [arXiv:2505.22074](https://arxiv.org/abs/2505.22074).

This work has been accepted for publication in *Communications AI & Computing*. The citation will be updated once the final journal version becomes available.

```bibtex
@misc{Horuz2025Resurrection,
    author        = {Co{\c{s}}ku Can Horuz and Geoffrey Kasenbacher
                     and Saya Higuchi and Sebastian Kairat
                     and Jendrik Stoltz and Moritz Pesl
                     and Bernhard A. Moser and Christoph Linse
                     and Thomas Martinetz and Sebastian Otte},
    title         = {The Resurrection of the ReLU},
    year          = {2025},
    eprint        = {2505.22074},
    archivePrefix = {arXiv},
    primaryClass  = {cs.LG},
    doi           = {10.48550/arXiv.2505.22074}
}
```
