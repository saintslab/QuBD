import numpy as np
import math
from math import ceil, floor
import torch
import torch.nn as nn
from torch import Tensor

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ClampSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lo, hi):
        return torch.clamp(x, min=lo, max=hi)

    @staticmethod
    def backward(ctx, g):
        return g, None, None


class ReluSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # normal ReLU in the forward
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pretend d(ReLU)/dx = 1 for all x
        grad_input = grad_output
        return grad_input


def torch_quantile(
    tensor: Tensor,
    q,
    dim = None,
    *,
    keepdim: bool = False,
    interpolation: str = "linear",
    out= None,
) -> Tensor:
    r"""Improved ``torch.quantile`` for one scalar quantile.
    Arguments
    ---------
    tensor: ``Tensor``
        See ``torch.quantile``.
    q: ``float``
        See ``torch.quantile``. Supports only scalar values currently.
    dim: ``int``, optional
        See ``torch.quantile``.
    keepdim: ``bool``
        See ``torch.quantile``. Supports only ``False`` currently.
        Defaults to ``False``.
    interpolation: ``{"linear", "lower", "higher", "midpoint", "nearest"}``
        See ``torch.quantile``. Defaults to ``"linear"``.
    out: ``Tensor``, optional
        See ``torch.quantile``. Currently not supported.
    Notes
    -----
    Uses ``torch.kthvalue``. Better than ``torch.quantile`` since:
    #. it has no :math:`2^{24}` tensor `size limit <https://github.com/pytorch/pytorch/issues/64947#issuecomment-2304371451>`_;
    #. it is much faster, at least on big tensor sizes.
    """
    # Sanitization of: q
    q_float = float(q)  # May raise an (unpredictible) error
    if not 0 <= q_float <= 1:
        msg = f"Only values 0<=q<=1 are supported (got {q_float!r})"
        raise ValueError(msg)
    # Sanitization of: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        tensor = tensor.reshape((-1, *(1,) * (tensor.ndim - 1)))
    # Sanitization of: inteporlation
    idx_float = q_float * (tensor.shape[dim] - 1)
    if interpolation == "nearest":
        idxs = [round(idx_float)]
    elif interpolation == "lower":
        idxs = [floor(idx_float)]
    elif interpolation == "higher":
        idxs = [ceil(idx_float)]
    elif interpolation in {"linear", "midpoint"}:
        low = floor(idx_float)
        idxs = [low] if idx_float == low else [low, low + 1]
        weight = idx_float - low if interpolation == "linear" else 0.5
    else:
        msg = (
            "Currently supported interpolations are {'linear', 'lower', 'higher', "
            f"'midpoint', 'nearest'}} (got {interpolation!r})"
        )
        raise ValueError(msg)
    # Sanitization of: out
    if out is not None:
        msg = f"Only None value is currently supported for out (got {out!r})"
        raise ValueError(msg)
    # Logic
    outs = [torch.kthvalue(tensor, idx + 1, dim, keepdim=True)[0] for idx in idxs]
    out = outs[0] if len(outs) == 1 else outs[0].lerp(outs[1], torch.tensor(weight, device=outs[0].device))
    # Rectification of: keepdim
    if keepdim:
        return out
    return out.squeeze() if dim_was_none else out.squeeze(dim)

class UniformSymmetricQuantizer(nn.Module):
    def __init__(self, bit_width: int):
        super(UniformSymmetricQuantizer, self).__init__()
        self.bit_width = bit_width

    def _get_absmax(self, w):
        absmax = torch_quantile(w.abs(), 0.99)
        return absmax

    def _get_step_size(self, absmax, Q):
        eps = 1e-8
        return absmax / Q  + eps

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # Dont track grad for quantile absmax
        with torch.no_grad():
            absmax = self._get_absmax(w)

        # Quantization parameters
        B = self.bit_width

        # Step size
        Q = torch.pow(torch.tensor(2.0, device=w.device, dtype=w.dtype), B - 1.0) - 1.0
        s = self._get_step_size(absmax, Q)
        
        # Quantize
        q = ClampSTE.apply(RoundSTE.apply(w / s), -Q, Q)
        w_hat = s * q 
        return w_hat

class DeadZoneLDZCompander(nn.Module):
    def __init__(self, fixed_bit_val=4, max_bits=8, init_bit_logit=3.0, init_deadzone_logit=3.0, learnable_bit=True,
                 learnable_deadzone=True):
        super().__init__()

        # Learnable params
        self.logit_dz = nn.Parameter(torch.tensor(init_deadzone_logit), requires_grad=learnable_deadzone)
        self.logit_bit = nn.Parameter(torch.tensor(init_bit_logit), requires_grad=learnable_bit)

        # Register buffers
        self.register_buffer("fixed_bit", torch.tensor(float(fixed_bit_val)))
        self.register_buffer("max_bits", torch.tensor(float(max_bits)))
        self.register_buffer("min_bits", torch.tensor(2.0))
        self.register_buffer("learnable_bit", torch.tensor(float(learnable_bit)))
        self.register_buffer("learnable_deadzone", torch.tensor(float(learnable_deadzone)))

        # flags
        self.learnable_bit_flag = learnable_bit
        self.learnable_deadzone_flag = learnable_deadzone

    def _get_bitwidth(self):
        """ Uses Learnable bit else fixed bit """
        if self.learnable_bit_flag:
            bit_budget = self.max_bits - self.min_bits
            b_cont = bit_budget * torch.tanh(self.logit_bit).abs() + self.min_bits
            B = RoundSTE.apply(b_cont)
        else:
            B = self.fixed_bit
        return B

    def _get_deadzone(self, absmax):
        return 2.0 * absmax * (1 - torch.tanh(self.logit_dz).abs())

    def _get_absmax(self, w):
        absmax = torch_quantile(w.abs(), 0.99)
        return absmax

    def _get_step_size(self, absmax, d, Q):
        eps = 1e-8
        return (absmax - d / 2.0) / (Q - 0.5) + eps

    @torch.no_grad()
    def get_bitwidth(self):
        return self._get_bitwidth().clone().cpu()

    @torch.no_grad()
    def get_deadzone_logit(self):
        return self.logit_dz.detach().clone().cpu()

    @torch.no_grad()
    def get_bitwidth_logit(self):
        return self.logit_bit.detach().clone().cpu()

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # Dont track grad for quantile absmax
        with torch.no_grad():
            absmax = self._get_absmax(w)

        # Quantization parameters
        B = self._get_bitwidth()
        d = self._get_deadzone(absmax)

        # Step size
        Q = torch.pow(torch.tensor(2.0, device=w.device, dtype=w.dtype), B - 1.0) - 1.0
        s = self._get_step_size(absmax, d, Q)
        
        # Quantize
        shift = d/2 - s/2
        w_comp = torch.sign(w) * ReluSTE.apply(w.abs() - shift)
        q = ClampSTE.apply(RoundSTE.apply(w_comp / s), -Q, Q)
        w_hat = s * q + shift * torch.sign(q)
        return w_hat
