"""LSU: LN-Sigmoid Unit (parameter-free).

Element-wise activation that fuses RMS-normalization into a SiLU-like
multiplicative gate, with no learnable parameters:

    LSU(x) = x · (1 + 0.5 · RMS(x)) / 2

where RMS(x)_k = x_k / sqrt(mean(x²))  (token-wise normalization,
                                        no mean-centering).

Derivation:
    SiLU(x) = x · σ(x) = x · (1 + tanh(x/2)) / 2

By DyT-LN duality (Zhu et al. 2025), tanh(x/2) approximates LN(x)/scale.
Substituting and dropping the mean-centering term (since β is absorbable
by the next Linear's column scale, and mean-shift gives no useful signal
when input mean=0):

    LSU(x) = x · (1 + γ · RMS(x)) / 2
    γ = 0.5 (fixed; matches SiLU in correlation 0.986 at x~N(0,1),
             prevents 16% channel sign-flip that γ=1 would cause)

Properties:
  - parameter-free (no γ, β to learn — γ baked at 0.5)
  - drop-in replacement for SiLU/GELU in FFN
  - fuses normalization into activation (token-aware)
  - 1 reduction per token (sum of squares only) — faster than LN
"""
from __future__ import annotations

import torch
import torch.nn as nn


_GAMMA = 0.5  # fixed by DyT-LN duality and channel sign-flip analysis


class LSU(nn.Module):
    """Parameter-free LN-Sigmoid Unit activation.

    Drop-in replacement for nn.SiLU / nn.GELU. Operates token-wise
    over the last dim (channels), via PyTorch's fused nn.RMSNorm with
    elementwise_affine=False (no learnable scale; γ baked at 0.5
    inline, β=0).

    Computation:
        rms_x = RMSNorm(x, affine=False)       # x / sqrt(mean(x²) + eps)
        out   = x * (1 + 0.5 · rms_x) / 2

    Why nn.RMSNorm and not a hand-rolled reduction:
        - PyTorch RMSNorm dispatches to a fused CUDA kernel (and a fused
          inductor lowering under torch.compile).
        - One reduction (sum-of-squares) per token, vs LayerNorm's two.
        - elementwise_affine=False removes the γ parameter entirely
          (we don't need it — γ is baked into the 0.5 multiplier).

    Args:
        normalized_shape: int or tuple — last-dim feature count.
            Required by nn.RMSNorm. Pass C for (..., C) layout.
        eps: numerical floor for RMS denominator.
    """

    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.rms = nn.RMSNorm(
            normalized_shape, eps=eps, elementwise_affine=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # nn.RMSNorm runs in the input dtype but uses fp32 accumulation
        # internally for numerical stability under bf16 AMP.
        rms_x = self.rms(x)
        return x * (1.0 + 0.5 * rms_x) / 2.0

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, gamma=0.5"


def _infer_channel_dim(parent: nn.Module, act_name: str) -> int:
    """Find the channel dimension that flows into the activation.

    For ViT/transformer FFN: the activation sits between two Linear
    layers (mlp.fc1 → act → mlp.fc2). The activation's input width is
    fc1.out_features. We look up that sibling so we can build LSU's
    nn.RMSNorm with the right normalized_shape.

    Falls back to scanning the parent's children for the closest
    preceding Linear if 'fc1' is not present (e.g. some
    SwiGLU-style blocks).
    """
    # Common case: timm.layers.Mlp has .fc1 sibling
    fc1 = getattr(parent, 'fc1', None)
    if isinstance(fc1, nn.Linear):
        return fc1.out_features
    # Fallback: pick the last Linear declared before `act_name`
    last_linear = None
    for name, mod in parent.named_children():
        if name == act_name:
            break
        if isinstance(mod, nn.Linear):
            last_linear = mod
    if last_linear is not None:
        return last_linear.out_features
    raise RuntimeError(
        f"Cannot infer channel dim for activation '{act_name}' under "
        f"{type(parent).__name__}: no fc1 or preceding Linear found."
    )


def convert_silu_to_lsu(module: nn.Module) -> nn.Module:
    """Recursively replace every nn.SiLU / nn.GELU with LSU.

    Each activation site is matched with the channel dim of its
    preceding Linear so LSU's RMSNorm gets the correct normalized_shape.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.SiLU, nn.GELU)):
            c = _infer_channel_dim(module, name)
            module.add_module(name, LSU(normalized_shape=c))
        else:
            convert_silu_to_lsu(child)
    return module


def convert_ln_to_lsu(module: nn.Module) -> nn.Module:
    """Recursively replace every nn.LayerNorm with LSU.

    Alternative integration: place LSU at the LN site (DyT/DyF/DyS
    paradigm — replace normalization layers). Reuses LN's
    normalized_shape for LSU's RMSNorm.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            module.add_module(name, LSU(normalized_shape=child.normalized_shape))
        else:
            convert_ln_to_lsu(child)
    return module
