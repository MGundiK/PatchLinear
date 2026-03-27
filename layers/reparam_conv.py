"""
Structural reparameterisation for 1-D depthwise convolution.
Adapted from ModernTCN (originally for 1-D over the patch/temporal axis).

_fuse_bn          : module-level utility used only by ReparamDWConv.merge_kernel().
                    It has no meaning outside that context so it lives here
                    rather than in a shared utils module.
ReparamDWConv     : depthwise Conv1d with optional multi-branch training and
                    single-branch inference via kernel fusion.

Why structural reparameterisation?
-----------------------------------
During training, two parallel branches are maintained:
  large branch  Conv1d(kernel=large_k) + BN   -- the primary filter
  small branch  Conv1d(kernel=small_k) + BN   -- regulariser

The small branch penalises overfitting to high-frequency noise during
training without adding any inference cost, because after training both
branches can be fused into a single Conv1d+bias via _fuse_bn.

At inference: call merge_kernel() once.  The fused kernel is mathematically
identical to running both branches separately, so there is zero accuracy loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fuse_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d):
    """
    Absorb a BatchNorm layer into its preceding Conv1d analytically.

    The BatchNorm transform is:
        y = (x - mu) / sqrt(var + eps) * gamma + beta

    Substituting x = W * input + b_conv and rearranging:
        y = (W * gamma / std) * input + (beta - mu * gamma / std)

    Returns
    -------
    fused_weight : tensor, same shape as conv.weight
    fused_bias   : tensor, shape [out_channels]
    """
    std = (bn.running_var + bn.eps).sqrt()
    t   = (bn.weight / std).reshape(-1, 1, 1)   # per-channel scale
    fused_weight = conv.weight * t
    fused_bias   = bn.bias - bn.running_mean * bn.weight / std
    return fused_weight, fused_bias


class ReparamDWConv(nn.Module):
    """
    Feature-independent (groups=channels) depthwise Conv1d with structural
    reparameterisation inherited from ModernTCN.

    Input / output shape: [B, channels, N]
    Same-padding is applied so the temporal length N is preserved.

    Parameters
    ----------
    channels : number of input (and output) channels; equals d_model here
               because we use this after patch embedding.
    large_k  : primary kernel size.  This is the A6 hyperparameter.
               Determines the Effective Receptive Field (ERF) in patch space:
                 ERF_patches = large_k
                 ERF_timesteps = (large_k - 1) * stride + patch_len
    small_k  : auxiliary kernel for the regularising branch.
               Set to None to disable the small branch entirely.
    merged   : if True, skip straight to the single-branch inference form.
               Used after calling merge_kernel() or for loading saved models.

    Workflow
    --------
    1. Train with merged=False (default).
    2. After training, call merge_kernel() once.
    3. The module now behaves as a plain Conv1d at inference.
    """
    def __init__(self, channels: int, large_k: int,
                 small_k: int = 3, merged: bool = False):
        super().__init__()
        self.large_k = large_k
        self.small_k = small_k

        if merged:
            self.reparam = nn.Conv1d(
                channels, channels, large_k,
                padding=large_k // 2, groups=channels, bias=True,
            )
        else:
            self.large_branch = nn.Sequential(
                nn.Conv1d(channels, channels, large_k,
                          padding=large_k // 2, groups=channels, bias=False),
                nn.BatchNorm1d(channels),
            )
            if small_k is not None:
                assert small_k <= large_k, \
                    f"small_k ({small_k}) must be <= large_k ({large_k})"
                self.small_branch = nn.Sequential(
                    nn.Conv1d(channels, channels, small_k,
                              padding=small_k // 2, groups=channels, bias=False),
                    nn.BatchNorm1d(channels),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'reparam'):
            return self.reparam(x)
        out = self.large_branch(x)
        if hasattr(self, 'small_branch'):
            out = out + self.small_branch(x)
        return out

    def merge_kernel(self) -> None:
        """
        Fuse both branches into a single Conv1d+bias for fast inference.
        Call once after training.  Modifies the module in-place.
        """
        lk, lb = _fuse_bn(self.large_branch[0], self.large_branch[1])
        if hasattr(self, 'small_branch'):
            sk, sb = _fuse_bn(self.small_branch[0], self.small_branch[1])
            # Pad the small kernel to the same size as the large kernel before adding
            pad = (self.large_k - self.small_k) // 2
            lk  = lk + F.pad(sk, [pad, pad])
            lb  = lb + sb
        self.reparam = nn.Conv1d(
            lk.shape[0], lk.shape[0], self.large_k,
            padding=self.large_k // 2, groups=lk.shape[0], bias=True,
        )
        self.reparam.weight.data = lk
        self.reparam.bias.data   = lb
        del self.large_branch
        if hasattr(self, 'small_branch'):
            del self.small_branch
