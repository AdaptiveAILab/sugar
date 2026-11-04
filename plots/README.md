# Figure plots

These plots require no training or downloaded datasets.
Install the Python dependencies from the repository root:

```bash
pip install -r plots/requirements.txt
```

## Figure 1d

[SUGAR-Surrogates.ipynb](SUGAR-Surrogates.ipynb) generates the forward and
backward curves for plain ReLU, SUGAR NeLU (`alpha=0.2`) and SUGAR B-SiLU
(`alpha=1.67`). It produces the three plots used in Figure 1d.

From the repository root:

```bash
cd plots
jupyter notebook SUGAR-Surrogates.ipynb
```

Run all cells in order. The notebook's plotting style also requires a LaTeX
installation.
The PDFs are saved in `plots/` as `relu_grad_stack.pdf`,
`sugar_grad_stack.pdf` and `sugar_bsilu_grad_stack.pdf`.

## Supplementary Figure S1

`activation_functions.py` generates **Supplementary Figure S1** from the paper:
activation functions and their derivatives.

From the repository root, run:

```bash
python plots/activation_functions.py
```

The figure is saved as `plots/activation_functions.pdf`.

The script evaluates 1,000 inputs between -3 and 3. B-SiLU uses `alpha=1.67`,
and Leaky ReLU uses a negative slope of `0.2`. Derivatives are calculated with
PyTorch autograd.
