# Adapted from S-Edge for the SUGAR experiments; packaging and comments simplified.
# See THIRD_PARTY.md and licenses/ for Apache-2.0 and MPL-2.0 notices.

from typing import Optional
import torch
from torch.nn import functional as F
import numpy as np
from .associative_scan import apply_ssm
from .init import make_linear_eigenvalues, init_log_steps


def as_complex(t: torch.Tensor, dtype=torch.complex64):
    assert t.shape[-1] == 2, "as_complex can only be done on tensors with shape=(...,2)"
    nt = torch.complex(t[..., 0], t[..., 1])
    if nt.dtype != dtype:
        nt = nt.type(dtype)
    return nt


def discretize_zoh(Lambda, B, B_bias, Delta, bias):
    if bias:
        B_concat = torch.cat((B, B_bias.unsqueeze(1)), dim=-1)
    else:
        B_concat = B
    Lambda_bar = torch.exp(Lambda * Delta)
    B_bar = ((Lambda_bar - 1) / Lambda)[..., None] * B_concat
    return (Lambda_bar, B_bar)


class SSM(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_state: int,
        d_out: int,
        dt_min: float,
        dt_max: float,
        step_scale: float = 1.0,
        input_bias=False,
        bias_init="zero",
        output_bias=False,
        complex_output=False,
        B_C_init="orthogonal",
        ensure_stability="abs",
        symmetric=False,
    ):
        super().__init__()
        self.symmetric = symmetric
        self.Lambda = torch.nn.Parameter(
            make_linear_eigenvalues(d_state, symmetric=self.symmetric)
        )
        self.log_step = torch.nn.Parameter(init_log_steps(d_state, dt_min, dt_max))
        self.discretize = discretize_zoh
        self.input_bias = input_bias
        self.output_bias = output_bias
        self.complex_output = complex_output
        self.step_scale = step_scale
        self.ensure_stability = ensure_stability
        if self.input_bias:
            if bias_init == "zero":
                self.B_bias = torch.nn.Parameter(
                    torch.zeros(d_state, 2, dtype=torch.float)
                )
            elif bias_init == "uniform":
                self.B_bias = torch.nn.Parameter(
                    torch.rand(d_state, 2, dtype=torch.float)
                )
        else:
            self.B_bias = torch.nn.Parameter(
                torch.zeros(d_state, 2, dtype=torch.float), requires_grad=False
            )
        if self.output_bias:
            if bias_init == "zero":
                self.C_bias = torch.nn.Parameter(
                    torch.zeros(d_out, 2, dtype=torch.float)
                )
            elif bias_init == "uniform":
                self.C_bias = torch.nn.Parameter(
                    torch.rand(d_out, 2, dtype=torch.float)
                )
        else:
            self.C_bias = torch.nn.Parameter(
                torch.zeros(d_out, 2, dtype=torch.float), requires_grad=False
            )
        if B_C_init != "orthogonal":
            raise ValueError("Only the Table 2 orthogonal initialization is supported")
        gain = np.sqrt(4 / 12)
        B_r = torch.empty(d_state, d_in)
        B_i = torch.empty(d_state, d_in)
        if d_in == 1:
            B_r = torch.nn.init.normal_(B_r, std=gain)
            B_i = torch.nn.init.normal_(B_i, std=gain)
        else:
            B_r = torch.nn.init.orthogonal_(B_r.T, gain=gain).T
            B_i = torch.nn.init.orthogonal_(B_i.T, gain=gain).T
        self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
        C_r = torch.empty(d_out, d_state)
        C_r = torch.nn.init.orthogonal_(C_r.T, gain=gain).T
        C_i = torch.empty(d_out, d_state)
        C_i = torch.nn.init.orthogonal_(C_i.T, gain=gain).T
        self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

    def initial_state(self, batch_size: Optional[int]):
        batch_shape = (batch_size,) if batch_size is not None else ()
        return torch.zeros((*batch_shape, self.C.shape[-2]))

    def forward_rnn(self, signal, prev_state):
        Lambda_c = as_complex(self.Lambda)
        if self.ensure_stability == "relu":
            Lambda_c = torch.complex(-F.relu(-Lambda_c.real), Lambda_c.imag)
        elif self.ensure_stability == "abs":
            Lambda_c = torch.complex(-torch.abs(Lambda_c.real), Lambda_c.imag)
        else:
            raise NotImplementedError("Only relu and abs stability are implemented")
        B_c = as_complex(self.B)
        B_bias_c = as_complex(self.B_bias)
        C_c = as_complex(self.C)
        C_bias_c = as_complex(self.C_bias)
        cinput_sequence = signal.type(C_c.dtype)
        step = self.step_scale * torch.exp(self.log_step)
        Lambda_bar, B_bars = self.discretize(
            Lambda_c, B_c, B_bias_c, step, self.input_bias
        )
        if self.input_bias:
            B_bar = B_bars[:, 0:-1]
            B_bias_bar = B_bars[:, -1]
        else:
            B_bar = B_bars
            B_bias_bar = torch.zeros_like(B_bar[:, 0])
        Bu = B_bar @ cinput_sequence + B_bias_bar
        x = Lambda_bar * prev_state + Bu
        y = C_c @ x + C_bias_c
        if self.complex_output:
            y_out = y
        else:
            y_out = y.real
        return (y_out, x)

    def forward(self, signal):
        with torch.no_grad():
            if self.ensure_stability == "relu":
                self.Lambda.data[:, 0] = -F.relu(-self.Lambda.data[:, 0])
            elif self.ensure_stability == "abs":
                self.Lambda.data[:, 0] = -torch.abs(self.Lambda.data[:, 0])
            if not self.symmetric:
                self.Lambda.data[:, 1] = torch.abs(self.Lambda.data[:, 1])
        Lambda = as_complex(self.Lambda)
        step = self.step_scale * torch.exp(self.log_step)
        B_c = as_complex(self.B)
        B_bias_c = as_complex(self.B_bias)
        C_c = as_complex(self.C)
        C_bias_c = as_complex(self.C_bias)
        Lambda_bars, B_bars = self.discretize(
            Lambda, B_c, B_bias_c, step, self.input_bias
        )
        if self.input_bias:
            B_bar = B_bars[:, 0:-1]
            B_bias_bar = B_bars[:, -1]
        else:
            B_bar = B_bars
            B_bias_bar = torch.zeros_like(B_bars[:, 0])
        return apply_ssm(
            Lambda_bars, B_bar, B_bias_bar, C_c, C_bias_c, signal, self.complex_output
        )
