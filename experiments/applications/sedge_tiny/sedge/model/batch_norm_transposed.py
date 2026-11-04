# Adapted from S-Edge for the SUGAR experiments; packaging and comments simplified.
# See THIRD_PARTY.md and licenses/ for Apache-2.0 and MPL-2.0 notices.

import torch


class ComplexBatchNorm1dT(torch.nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.bn_real = torch.nn.BatchNorm1d(d_in)
        self.bn_imag = torch.nn.BatchNorm1d(d_in)

    def forward(self, x):
        real = self.bn_real.forward(x.real.permute(0, 2, 1)).permute(0, 2, 1)
        imag = self.bn_imag.forward(x.imag.permute(0, 2, 1)).permute(0, 2, 1)
        return real + 1j * imag


class BatchNorm1dT(torch.nn.BatchNorm1d):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 1)).permute(0, 2, 1)
