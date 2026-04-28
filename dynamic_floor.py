"""DyF: Dynamic Floor — element-wise norm replacement.

Three kernels share the form  f_a(u) = ψ(u - a) + a:

    silu (soft):  ψ(v) = v · σ(v)             — Swish/SiLU
    gelu (soft):  ψ(v) = GELU(v)              — Gaussian-CDF gating
    hard:         ψ(v) = max(0, v) = ReLU(v)  → f_a(u) = max(a, u)

Final layer:

    DyF(x)_k = γ_k · f_a(x_k) + β_k

Parameter counts per module:
    a       — scalar (1 parameter, shared across channels)
    γ, β    — per-channel (C parameters each)
    total   — 2C + 1

Properties (see docs/spec.md):
  P1  Element-wise (no token statistics).
  P2  Positive unbounded — outliers preserved.
  P3  Negative bounded — saturates to a.
  P4  2C+1 affine DOF; > LN's 2C at inference.
  P5  Includes shifted ReLU, identity, Swish/GELU as special cases.

Drop-in for nn.LayerNorm in pre-norm Transformers / ConvNeXt.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerNorm2d


# Valid values for the `kernel` argument. "hard" is also accepted via a
# legacy `hard=True` kwarg for backwards compat with earlier iterations.
KERNELS = ("silu", "gelu", "hard")


class DynamicFloor(nn.Module):
    """Element-wise Dynamic Floor activation.

    Args:
        normalized_shape: int or tuple — channel count, used for per-channel
            (γ, β). `a` is a scalar regardless of this. Matches LayerNorm's
            API for drop-in swap.
        channels_last: True for (..., C) tensors (ViT, ConvNeXt LN);
            False for (N, C, H, W) (LayerNorm2d).
        kernel: which ψ to use — "silu", "gelu", or "hard".
        a_init: initial saturation floor.
        gamma_init: initial γ.
        beta_init: initial β.
        hard: legacy kwarg; if True, overrides `kernel` to "hard".
    """

    def __init__(
        self,
        normalized_shape,
        channels_last: bool,
        kernel: str = "silu",
        a_init: float = 0.0,
        gamma_init: float = 1.0,
        beta_init: float = 0.0,
        hard: bool = False,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.channels_last = channels_last
        if hard:
            kernel = "hard"
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
        self.kernel = kernel
        self.a_init = a_init
        self.gamma_init = gamma_init
        self.beta_init = beta_init

        c = self.normalized_shape[-1]
        # `a` is a scalar — one floor per module. γ, β remain per-channel.
        self.a = nn.Parameter(torch.tensor(float(a_init)))
        self.weight = nn.Parameter(torch.full((c,), gamma_init))
        self.bias = nn.Parameter(torch.full((c,), beta_init))

    def _floor(self, u: torch.Tensor) -> torch.Tensor:
        # All three kernels follow the same shape: ψ(u-a) + a.
        # We use the fused PyTorch ops (F.silu / F.relu / F.gelu) rather
        # than spelling them out — they're faster (single CUDA kernel)
        # and autograd-cheaper (intermediate sigmoid/cdf not stored).
        # `self.a` is scalar; broadcasts against u of any shape.
        a = self.a
        d = u - a
        if self.kernel == "silu":
            return F.silu(d) + a
        if self.kernel == "gelu":
            return F.gelu(d) + a
        # hard: max(a, u) ≡ ReLU(u-a) + a
        return F.relu(d) + a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._floor(x)
        if self.channels_last:
            return y * self.weight + self.bias
        # (N, C, H, W) layout
        return y * self.weight[None, :, None, None] + self.bias[None, :, None, None]

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, "
            f"channels_last={self.channels_last}, kernel={self.kernel}, "
            f"a_init={self.a_init}, gamma_init={self.gamma_init}, "
            f"beta_init={self.beta_init}"
        )


def convert_ln_to_dyf(
    module: nn.Module,
    *,
    kernel: str = "silu",
    a_init: float = 0.0,
    gamma_init: float = 1.0,
    beta_init: float = 0.0,
    hard: bool = False,
) -> nn.Module:
    """Recursively replace every nn.LayerNorm in `module` with DynamicFloor.

    Mirrors `convert_ln_to_dyt` from dynamic_tanh.py for parity.
    """
    out = module
    if isinstance(module, nn.LayerNorm):
        out = DynamicFloor(
            module.normalized_shape,
            channels_last=not isinstance(module, LayerNorm2d),
            kernel=kernel,
            a_init=a_init,
            gamma_init=gamma_init,
            beta_init=beta_init,
            hard=hard,
        )
    for name, child in module.named_children():
        out.add_module(
            name,
            convert_ln_to_dyf(
                child,
                kernel=kernel,
                a_init=a_init,
                gamma_init=gamma_init,
                beta_init=beta_init,
                hard=hard,
            ),
        )
    del module
    return out
