"""DyS: Dynamic Saturation — bounded element-wise norm replacement.

Two kernels share the form  g(u) = u · h(u)  where h(u) is an S-curve's
derivative ("bell"):

    dsilu:  g(u) = u · σ(u)·(1 - σ(u))      — sigmoid bell × identity
                                               σ' = σ(1-σ)
    dgelu:  g(u) = u · φ(u)                  — Gaussian bell × identity
                                               φ(u) = (2π)^(-1/2)·exp(-u²/2)

Both are odd, bounded, decay to 0 in BOTH tails. The naming refers to
which S-curve's derivative provides the bell:
    dsilu ↔ derivative of sigmoid (peak 0.25 at u=0 for σ', then ×u)
    dgelu ↔ derivative of erf-based CDF (= φ, the std normal PDF, then ×u)

Note: this is NOT the full d/du[GELU(u)] = Φ(u) + u·φ(u), which is
unbounded (→ u for large u). We deliberately drop the Φ(u) "DC" term
to keep g(u) bounded in both tails — that's what gives DyS its
saturation character vs DyF's positive-unbounded.

The full layer applies a learned scalar `a` as an input-side scale:

    DyS(x)_k = γ_k · g(x_k / a) + β_k

Per-module:
    a       — scalar (1 parameter)
    γ, β    — per-channel (C each)
    total   — 2C + 1

Properties (vs DyF):
  - element-wise: yes (same as DyF)
  - bounded BOTH tails: g(u) → 0 as |u| → ∞   ← departure from DyF
  - odd function: g(-u) = -g(u)              ← departure from DyF
  - peak at |u| ≈ 1.54 (silu) / |u| ≈ 0.75 (gelu); |g_max| ≈ 0.22 / 0.36
  - a controls input scale: small a → saturation, large a → linear-like
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerNorm2d


KERNELS = ("dsilu", "dgelu")

# d/du[GELU(u)] = Φ(u) + u · φ(u), where φ(u) = exp(-u²/2)/√(2π).
# Precise GELU (not the tanh approximation) since we want a sharp,
# well-defined derivative for the activation.
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)


def _dsilu(u: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(u)
    return u * s * (1.0 - s)


def _dgelu(u: torch.Tensor) -> torch.Tensor:
    # u · φ(u), where φ(u) = (2π)^(-1/2) · exp(-u²/2). The "bell × u"
    # GELU analog of dsilu. Bounded in both tails (peak |u·φ| ≈ 0.242
    # at u=±1).
    pdf = _INV_SQRT_2PI * torch.exp(-0.5 * u * u)
    return u * pdf


class DynamicSaturation(nn.Module):
    """Element-wise Dynamic Saturation activation.

    Args:
        normalized_shape: int or tuple — channel count (last dim).
            Used for per-channel γ, β. `a` is scalar regardless.
        channels_last: True for (..., C); False for (N, C, H, W).
        kernel: "dsilu" or "dgelu".
        a_init: initial scale (default 1.0). Recommended range [0.5, 2.0].
            See module docstring + spec for sensitivity discussion.
        gamma_init, beta_init: per-channel affine init.
    """

    def __init__(
        self,
        normalized_shape,
        channels_last: bool,
        kernel: str = "dgelu",
        a_init: float = 1.0,
        gamma_init: float = 1.0,
        beta_init: float = 0.0,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.channels_last = channels_last
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
        self.kernel = kernel
        self.a_init = a_init
        self.gamma_init = gamma_init
        self.beta_init = beta_init

        c = self.normalized_shape[-1]
        # Scalar `a` — one input-scale per module. Floats only to avoid
        # rounding the init away under bf16 down-conversion later.
        self.a = nn.Parameter(torch.tensor(float(a_init)))
        self.weight = nn.Parameter(torch.full((c,), gamma_init))
        self.bias = nn.Parameter(torch.full((c,), beta_init))

    def _kernel(self, u: torch.Tensor) -> torch.Tensor:
        if self.kernel == "dsilu":
            return _dsilu(u)
        return _dgelu(u)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute kernel at fp32 precision for numerical safety: bf16
        # erf/exp lose precision near the peak. Cast result back to x's
        # dtype so AMP autocast accounting stays consistent.
        x_fp = x.float()
        # Promote `a` to fp32 too; division below would otherwise warn
        # about dtype mismatch under autocast.
        u = x_fp / self.a.float()
        y = self._kernel(u).to(x.dtype)
        if self.channels_last:
            return y * self.weight + self.bias
        return y * self.weight[None, :, None, None] + self.bias[None, :, None, None]

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, "
            f"channels_last={self.channels_last}, kernel={self.kernel}, "
            f"a_init={self.a_init}, gamma_init={self.gamma_init}, "
            f"beta_init={self.beta_init}"
        )


def convert_ln_to_dys(
    module: nn.Module,
    *,
    kernel: str = "dgelu",
    a_init: float = 1.0,
    gamma_init: float = 1.0,
    beta_init: float = 0.0,
) -> nn.Module:
    """Recursively replace every nn.LayerNorm in `module` with DynamicSaturation."""
    out = module
    if isinstance(module, nn.LayerNorm):
        out = DynamicSaturation(
            module.normalized_shape,
            channels_last=not isinstance(module, LayerNorm2d),
            kernel=kernel,
            a_init=a_init,
            gamma_init=gamma_init,
            beta_init=beta_init,
        )
    for name, child in module.named_children():
        out.add_module(
            name,
            convert_ln_to_dys(
                child,
                kernel=kernel,
                a_init=a_init,
                gamma_init=gamma_init,
                beta_init=beta_init,
            ),
        )
    del module
    return out
