# Adapted from S-Edge for the SUGAR experiments; packaging and comments simplified.
# See THIRD_PARTY.md and licenses/ for Apache-2.0 and MPL-2.0 notices.

import numpy as np
import torch


def make_linear_eigenvalues(N, symmetric=True):
    if symmetric:
        Lambda = -0.5 + 1j * np.arange(-N // 2, N // 2)
    else:
        Lambda = -0.5 + 1j * np.linspace(0, N // 2, N)
    lambda_real = np.expand_dims(Lambda.real, axis=1)
    lambda_imag = np.expand_dims(Lambda.imag, axis=1)
    Lambda = np.concatenate((lambda_real, lambda_imag), axis=1)
    Lambda = torch.tensor(Lambda, dtype=torch.float)
    return Lambda


def uniform(shape, dtype=torch.float, minval=0.0, maxval=1.0, device=None):
    src = torch.rand(shape, dtype=dtype, device=device)
    if minval == 0 and maxval == 1.0:
        return src
    else:
        return src * (maxval - minval) + minval


def log_step_initializer(dt_min=0.001, dt_max=0.1):

    def init(shape):
        return uniform(shape, minval=np.log(dt_min), maxval=np.log(dt_max))

    return init


def init_log_steps(H, dt_min, dt_max):
    log_steps = []
    for i in range(H):
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(shape=(1,))
        log_steps.append(log_step)
    return torch.tensor(log_steps)
