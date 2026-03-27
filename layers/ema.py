"""
Exponential Moving Average decomposition.
Source: xPatch vectorized O(1) implementation, device-agnostic fix applied.

The original xPatch code had `.to('cuda')` hardcoded.  Replaced with
`.to(x.device)` so both classes work on CPU, multi-GPU, and MPS without
modification.

EMA      : returns the trend (smoothed) signal.
EMADecomp: wraps EMA and returns (seasonal, trend) pair for use by the
           dual-stream backbone.
"""

import torch
import torch.nn as nn


class EMA(nn.Module):
    """
    Vectorized Exponential Moving Average.
    Returns the TREND (smoothed) signal, shape [B, L, C].

    EMA[t] = alpha * sum_{k=0}^{t} (1-alpha)^{t-k} * x[k]
             + (1-alpha)^{t+1} * x[0]

    The decay weights are pre-computed once and the update becomes a single
    torch.cumsum call, giving O(1) Python-loop iterations.

    Parameters
    ----------
    alpha : smoothing factor in (0, 1).
            Close to 1 -> trend tracks input closely (short memory).
            Close to 0 -> trend changes slowly          (long memory).
            0.3 is the xPatch default and a good starting point.

    To make alpha learnable, replace self.alpha = alpha with:
        self.alpha = nn.Parameter(torch.tensor(alpha))
    and add self.alpha.data.clamp_(0, 1) at the top of forward().
    """
    def __init__(self, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        _, t, _ = x.shape
        powers  = torch.flip(torch.arange(t, dtype=torch.double), dims=(0,))
        weights = torch.pow(1.0 - self.alpha, powers).to(x.device)
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)   # broadcast over B and C
        divisor = divisor.reshape(1, t, 1)
        trend   = torch.cumsum(x.double() * weights, dim=1)
        trend   = torch.div(trend, divisor)
        return trend.float()


class EMADecomp(nn.Module):
    """
    Seasonal-trend decomposition using EMA.

    Returns (seasonal, trend) where:
        trend    = EMA(x)
        seasonal = x - trend

    Ablation A1: compare use_decomp=True vs use_decomp=False.
    Claim: explicit decomposition provides a better inductive bias than
    letting the dual streams implicitly separate the signal, because each
    stream then receives the signal matched to its structural assumption:
        trend stream    -> smooth, slowly-varying  -> Linear projection
        seasonal stream -> zero-mean periodic residual -> patch + DWConv

    Expected to matter most on low-forecastability datasets (Solar, ETT) and
    least on Traffic where dominant structure is spatial correlation.
    """
    def __init__(self, alpha: float = 0.3):
        super().__init__()
        self.ema = EMA(alpha)

    def forward(self, x: torch.Tensor):
        trend    = self.ema(x)       # [B, L, C]
        seasonal = x - trend         # [B, L, C]
        return seasonal, trend
