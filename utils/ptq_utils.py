from math import ceil, floor

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch import Tensor


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs


def torch_quantile(
    tensor: Tensor,
    q,
    dim=None,
    *,
    keepdim: bool = False,
    interpolation: str = "linear",
    out=None,
) -> Tensor:
    """Improved ``torch.quantile`` with no 2^24 size limit and faster execution."""
    q_float = float(q)
    if not 0 <= q_float <= 1:
        raise ValueError(f"Only values 0<=q<=1 are supported (got {q_float!r})")
    if dim_was_none := dim is None:
        dim = 0
        tensor = tensor.reshape((-1, *(1,) * (tensor.ndim - 1)))
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
        raise ValueError(
            f"Supported interpolations: {{'linear','lower','higher','midpoint','nearest'}} (got {interpolation!r})"
        )
    if out is not None:
        raise ValueError(f"Only None is supported for out (got {out!r})")
    outs = [torch.kthvalue(tensor, idx + 1, dim, keepdim=True)[0] for idx in idxs]
    out = outs[0] if len(outs) == 1 else outs[0].lerp(outs[1], torch.tensor(weight, device=outs[0].device))
    return out if keepdim else (out.squeeze() if dim_was_none else out.squeeze(dim))


class UniformQuantizer(nn.Module):
    """Uniform symmetric quantizer with per-tensor scale factor."""

    def __init__(self, bit_width: int):
        super().__init__()
        self.bit_width = bit_width

    def set_bits(self, new_bit_width: int) -> None:
        self.bit_width = new_bit_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_val = torch_quantile(x.abs(), 0.99)
        s = max_val / (2 ** (self.bit_width - 1) - 1)
        return RoundSTE.apply(x / s) * s


class UniformQuantizer_per_channel(nn.Module):
    """Uniform symmetric quantizer with per-output-channel scale factor."""

    def __init__(self, bit_width: int):
        super().__init__()
        self.bit_width = bit_width

    def set_bits(self, new_bit_width: int) -> None:
        self.bit_width = new_bit_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            max_val = x.abs().view(x.shape[0], -1).max(dim=1).values
            s = (max_val / (2 ** (self.bit_width - 1) - 1)).view(-1, 1, 1, 1)
        elif x.dim() == 2:
            max_val = x.abs().max(dim=1).values
            s = (max_val / (2 ** (self.bit_width - 1) - 1)).view(-1, 1)
        else:
            max_val = x.abs().max()
            s = max_val / (2 ** (self.bit_width - 1) - 1)
        return RoundSTE.apply(x / s) * s


class FakeQuantParametrization(nn.Module):
    """Parametrization that applies fake quantization when enabled."""

    def __init__(self, quantizer, enabled=True):
        super().__init__()
        self.quantizer = quantizer
        self.enabled = enabled

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return W
        return self.quantizer(W)


def attach_weight_quantizers(model, exclude_layers, quantizer, enabled=True, verbose=False) -> None:
    """Attaches fake-quant parametrizations to all eligible weight layers."""
    for name, module in model.named_modules():
        if not any(target in name for target in exclude_layers):
            if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
                parametrize.register_parametrization(
                    module, "weight", FakeQuantParametrization(quantizer=quantizer, enabled=enabled)
                )
                if verbose:
                    print(f"Attached weight quantizer to layer: {name}")


def detach_weight_quantizers(model, leave_parametrized=False, verbose=False) -> None:
    """Removes fake-quant parametrizations from a model."""
    for name, module in model.named_modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=leave_parametrized)
            if verbose:
                print(f"Detached weight quantizer from layer: {name}")


def toggle_quantization(model, enabled: bool) -> None:
    """Activates or deactivates quantization for all attached quantizers."""
    for _, submodule in model.named_modules():
        if hasattr(submodule, "parametrizations"):
            for _, param_list in submodule.parametrizations.items():
                for p in param_list:
                    if isinstance(p, FakeQuantParametrization):
                        p.enabled = enabled
