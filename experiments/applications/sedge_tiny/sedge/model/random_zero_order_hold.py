# Adapted from S-Edge for the SUGAR experiments; packaging and comments simplified.
# See THIRD_PARTY.md and licenses/ for Apache-2.0 and MPL-2.0 notices.

import torch
import math
import random


class RandomZeroOrderHold(torch.nn.Module):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if self.training:
            random_val = random.uniform(0, 1)
            random_val /= self.prob
            if random_val < 0.001:
                return x
            downscale = 2 ** math.floor(-math.log2(random_val))
            if downscale < 1:
                return x
            if downscale > 128:
                downscale = 128
            for offset in range(1, int(downscale)):
                x[:, offset :: int(downscale), :] = x[:, :: int(downscale), :]
            return x
        else:
            return x

    def set_step_scale(self, step_scale, previous_step_scale=None):
        pass

    def get_number_of_MACs(self):
        return 0

    def get_number_of_parameters(self):
        return 0
