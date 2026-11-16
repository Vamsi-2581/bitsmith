"""SmoothQuant: move activation outliers into the weights with a per-channel scale.

For input channel ``j``::

    s_j = max|X_j| ** alpha / max|W_:,j| ** (1 - alpha)
    Y = X W^T = (X / s) (W * s)^T          (exact in real arithmetic)

Dividing activations by ``s`` flattens outlier channels so per-token INT8 activation
quantization loses less, while the weights absorb a range they can easily afford.

``calibrate_and_quantize(method="smoothquant")`` uses :func:`smoothquant_search` by
default, which may leave a layer unsmoothed (ADR 0018); pass ``alpha`` for a fixed value.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

_FLOOR = 1e-5


def smoothing_factors(act_absmax: Tensor, weight: Tensor, alpha: float = 0.5) -> Tensor:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    w_absmax = weight.detach().abs().amax(dim=0).to(torch.float32)
    a = act_absmax.to(torch.float32).clamp_min(_FLOOR)
    s = a.pow(alpha) / w_absmax.clamp_min(_FLOOR).pow(1.0 - alpha)
    return s.clamp_min(_FLOOR)


def smooth_weight(weight: Tensor, s: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(weight * s, 1 / s)``; feed activations through the second factor."""
    smoothed = (weight.to(torch.float32) * s.unsqueeze(0)).to(weight.dtype)
    return smoothed, 1.0 / s


ALPHAS = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class SmoothSearch:
    """Best smoothing for one layer. ``scale is None`` means smoothing did not help."""

    scale: Tensor | None
    alpha: float | None
    loss: float  # mean squared output error on the calibration samples, chosen option
    baseline_loss: float  # the same without smoothing


def smoothquant_search(
    weight: Tensor,
    act_absmax: Tensor,
    x: Tensor,
    *,
    scheme: str = "int8",
    group_size: int = 128,
    act_bits: int | None = 8,
    alphas: tuple[float, ...] = ALPHAS,
) -> SmoothSearch:
    """Pick no smoothing or the ``alpha`` with the lowest quantized output error on ``x``.

    A fixed ``alpha`` can hurt: for a layer whose input has rarely firing channels (the
    input of an MLP down projection), calibration sees a small maximum, ``1 / s`` gets
    large, and when the channel fires later it dominates the token's INT8 range. Comparing
    against no smoothing on the calibration samples catches this without knowing the
    model's structure.
    """
    from bitsmith.nn.codecs import make_codec
    from bitsmith.quant import quantize_activations_per_token

    codec = make_codec(scheme, group_size)
    w = weight.detach().to(torch.float32)
    xf = x.to(torch.float32)
    y_ref = xf @ w.T

    def loss(scale: Tensor | None) -> float:
        if scale is None:
            ws, xs = w, xf
        else:
            ws_t, inv = smooth_weight(w, scale)
            ws, xs = ws_t, xf * inv.to(torch.float16).float()  # 1/s is stored in fp16
        wq = codec.decode(codec.encode(ws), *w.shape)
        xq = quantize_activations_per_token(xs, act_bits) if act_bits else xs
        return float((xq @ wq.T - y_ref).pow(2).mean())

    base = loss(None)
    best: tuple[float, float | None, Tensor | None] = (base, None, None)
    for alpha in alphas:
        s = smoothing_factors(act_absmax, weight, alpha)
        cand = loss(s)
        if cand < best[0]:
            best = (cand, alpha, s)
    return SmoothSearch(scale=best[2], alpha=best[1], loss=best[0], baseline_loss=base)
