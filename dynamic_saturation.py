"""DyS: Dynamic Saturation — bounded element-wise norm replacement.

Two kernels share the form  g(u) = u · h(u)  where h is an S-curve's
derivative ("bell"). Both are odd, bounded, decay to 0 in BOTH tails.

    dsilu:  g(u) = u · σ(u)·(1 - σ(u))     peak ≈ 0.224 at |u| ≈ 1.54
    dgelu:  g(u) = u · φ(u)                peak ≈ 0.242 at |u| ≈ 1.00
                       (φ = standard normal PDF, no √(2π) prefactor)

Earlier we multiplied by a normalization constant K (4 for dsilu,
√(2π) for dgelu) so g'(0) = 1, then paired with γ_init=1.626/2.279
to get output std=1. The K factor is mathematically equivalent to
folding it into γ (γ' = γ·K), so we drop K and absorb it into γ_init.

Why drop K: AdamW's weight_decay acts as `γ ← γ·(1 - lr·wd)`, with
penalty proportional to |γ|. With K and small γ_init=1.6, decay
pressure is tiny. Without K and γ_init=6+, decay pressure is real
and may shift γ trajectory. Forward output is identical either way,
but training dynamics differ.

Resulting γ_init for output std=1 (with a_init = sqrt(l+1) sqrt schedule
giving u ~ N(0,1)):
    dsilu: σ(g) ≈ 0.154 → γ_init = 1/0.154 = 6.494
    dgelu: σ(g) ≈ 0.175 → γ_init = 1/0.175 = 5.711

Note: dgelu kernel u · φ(u) is NOT the full d/du[GELU(u)] = Φ(u) +
u·φ(u). We drop Φ(u) so g stays bounded (→ 0) in both tails — that's
what gives DyS its saturation character vs DyF's positive-unbounded.

The full layer applies a learned scalar `a` as an input-side scale:

    DyS(x)_k = γ_k · g(x_k / a) + β_k

Per-module:
    a       — scalar (1 parameter, broadcast across all channels)
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

# Raw, un-normalized kernels. The K constant (4 for dsilu, √(2π) for
# dgelu) that used to live here has been absorbed into γ_init so that
# AdamW weight_decay acts on the full effective scale.
_INV_SQRT_2PI = 1.0 / (2.0 * 3.141592653589793) ** 0.5


def _dsilu(u: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(u)
    return u * s * (1.0 - s)


def _dgelu(u: torch.Tensor) -> torch.Tensor:
    # u · φ(u), where φ(u) = exp(-u²/2) / √(2π). Peak ≈ 0.242 at u=±1.
    return u * _INV_SQRT_2PI * torch.exp(-0.5 * u * u)


class DynamicSaturation(nn.Module):
    """Element-wise Dynamic Saturation activation.

    Args:
        normalized_shape: int or tuple — channel count (last dim).
            Used for per-channel γ, β. `a` is scalar regardless.
        channels_last: True for (..., C); False for (N, C, H, W).
        kernel: "dsilu" or "dgelu".
        a_init: initial scale (scalar). The sqrt-schedule in
            `convert_ln_to_dys` overrides this on a per-site basis.
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
        # Scalar `a` — one input scale per module, broadcast across
        # channels. The sqrt schedule sets a different scalar per site
        # (l-th site → sqrt(l+1)). γ, β remain per-channel.
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
        # `a` is scalar; broadcasts against u of any shape.
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
    a_init_schedule: str = "constant",
    gamma_init: float = 1.0,
    beta_init: float = 0.0,
) -> nn.Module:
    """Recursively replace every nn.LayerNorm in `module` with DynamicSaturation.

    a_init_schedule:
        "constant"  — every site gets a_init (default).
        "sqrt"      — site k (in residual-stream order) gets
                      a_init * sqrt(k + 1). Motivation: in pre-LN
                      transformers Var(x_l) ≈ l + 1, so the input scale
                      to the l-th norm is sqrt(l+1). Setting a per-site
                      to that scale puts the input squarely in the
                      transition region of g(u)=u·K·σ'(u) (which is
                      meaningful for |u| ≲ 2-3).

    Sites are numbered 0..N-1 in the order they appear during
    `named_children` recursion. For timm ViT this matches the residual-
    stream sublayer order: blocks.0.norm1=0, blocks.0.norm2=1, …,
    blocks.{L-1}.norm2=2L-1, fc_norm=2L.
    """
    if a_init_schedule not in ("constant", "sqrt"):
        raise ValueError(f"a_init_schedule must be 'constant' or 'sqrt', got {a_init_schedule!r}")

    # Pre-pass: count LN sites in deterministic visit order. We do this
    # before the in-place swap so the index assignment is stable, even
    # though the swap itself happens during the second pass below.
    sites: list[str] = []

    def _scan(mod, prefix=""):
        for name, child in mod.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.LayerNorm):
                sites.append(full)
            else:
                _scan(child, full)

    _scan(module)
    site_to_idx = {name: i for i, name in enumerate(sites)}

    def _convert(mod, prefix=""):
        out = mod
        if isinstance(mod, nn.LayerNorm):
            # The caller passes us a leaf LN; we don't get the prefix
            # for the root call, so root LN (rare) gets idx 0 by the
            # site_to_idx default.
            full = prefix
            idx = site_to_idx.get(full, 0)
            if a_init_schedule == "sqrt":
                a = a_init * math.sqrt(idx + 1)
            else:
                a = a_init
            out = DynamicSaturation(
                mod.normalized_shape,
                channels_last=not isinstance(mod, LayerNorm2d),
                kernel=kernel,
                a_init=a,
                gamma_init=gamma_init,
                beta_init=beta_init,
            )
            return out
        for name, child in mod.named_children():
            full = f"{prefix}.{name}" if prefix else name
            mod.add_module(name, _convert(child, full))
        return mod

    return _convert(module)
