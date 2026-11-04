# Adapted from S-Edge for the SUGAR experiments; packaging and comments simplified.
# See THIRD_PARTY.md and licenses/ for Apache-2.0 and MPL-2.0 notices.

import torch


class ChannelDropout(torch.nn.Dropout1d):
    def __init__(self, p=0.5, inplace=False):
        super().__init__(p, inplace)

    def forward(self, x):
        if self.p == 0 or not self.training:
            return x
        return super().forward(x.permute(0, 2, 1)).permute(0, 2, 1)
