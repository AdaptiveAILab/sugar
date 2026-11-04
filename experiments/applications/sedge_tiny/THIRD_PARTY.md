# Third-Party Code

The files in `sedge/model/` and the audio-loading and training operations are
adapted from [S-Edge](https://github.com/embedded-machine-learning/S-Edge)
by Matthias Bittner, Daniel Schnoll, Matthias Wess and Axel Jantsch:
*Efficient and interpretable raw audio classification with diagonal state
space models*, Machine Learning 114, 175 (2025),
[doi:10.1007/s10994-025-06807-z](https://doi.org/10.1007/s10994-025-06807-z).
S-Edge is distributed under Apache-2.0; its license is included in
`licenses/Apache-2.0.txt`. These files have been reorganized and simplified
for the SUGAR experiments; unused initialization choices were removed.

S-Edge's state-space implementation builds on
[S5-PyTorch](https://github.com/i404788/s5-pytorch), distributed under MPL-2.0.
The derived model source is provided here in editable source form with the
MPL-2.0 license in `licenses/MPL-2.0.txt`. This source code form is subject to
the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not
distributed with this file, obtain one at https://mozilla.org/MPL/2.0/.

`sedge/scheduler.py` is adapted from
[pytorch-cosine-annealing-with-warmup](https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup),
Copyright (c) 2022 Naoki Katsura, under the MIT license in
`licenses/cosine-annealing-MIT.txt`. Formatting and comments were simplified;
the scalar schedule is retained.

These notices apply to this experiment package and do not change the licenses
of other experiments in the repository.
