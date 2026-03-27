"""
PatchLinear: a lightweight long-term multivariate time series forecasting model.

Combines:
  XLinear  -- Linear(seq_len, d_model) global temporal filter bank,
              global token + TGM/VGM cross-channel mechanism,
              sigmoid gating throughout.
  xPatch   -- EMA seasonal-trend decomposition,
              patching + depthwise-separable CNN for the seasonal stream,
              arctangent loss (in exp_main).
  ModernTCN -- large-kernel depthwise convolution along the patch axis (A6),
               structural reparameterisation (optional),
               pointwise ConvFFN for cross-feature mixing.

Ablation switches  (all bool, all exposed as CLI args in run.py)
-----------------------------------------------------------------
A1  use_decomp         True / False   -- EMA decomposition
A2a use_seas_stream    False          -- trend-only (disable seasonal CNN)
A2b use_trend_stream   False          -- seasonal-only (disable linear proj)
A3  use_fusion_gate    True / False   -- input-dependent stream fusion
A4a use_cross_channel  True / False   -- VGM cross-channel interaction
A4b use_alpha_gate     True / False   -- per-channel mixing coefficient
A6  dw_kernel          3 / 7 / 13    -- DWConv kernel size (ERF control)
"""

import torch
import torch.nn as nn

from layers.revin       import RevIN
from layers.ema         import EMADecomp
from layers.backbone    import Backbone


class Model(nn.Module):
    """
    Top-level PatchLinear model.

    The class is named Model (not PatchLinear) so it is compatible with the
    standard Time Series Library convention of importing Model from each
    model module.  The module file is named PatchLinear.py and the model is
    registered in exp_main as 'PatchLinear'.

    Required configs attributes
    ---------------------------
    seq_len           int     lookback window L (e.g. 96)
    pred_len          int     forecast horizon  (96 / 192 / 336 / 720)
    enc_in            int     number of input channels C
    d_model           int     embedding dimension (64 or 128)
    t_ff              int     TGM hidden dim (2 * d_model recommended)
    c_ff              int     VGM hidden dim (max(16, min(C // 4, 128)))
    patch_len         int     patch length                    (16)
    stride            int     patch stride                    (8)
    dw_kernel         int     DWConv kernel over patch axis   (3 / 7 / 13)
    small_kernel      int     small branch for reparameterisation (3)
    alpha             float   EMA smoothing factor            (0.3)
    t_dropout         float
    c_dropout         float
    embed_dropout     float
    head_dropout      float
    use_reparam       bool    structural reparameterisation
    use_decomp        bool    A1
    use_trend_stream  bool    A2b  (default True)
    use_seas_stream   bool    A2a  (default True)
    use_fusion_gate   bool    A3
    use_cross_channel bool    A4a
    use_alpha_gate    bool    A4b
    """
    def __init__(self, configs):
        super().__init__()

        # ── Normalisation ──────────────────────────────────────────────────────
        self.revin = RevIN(configs.enc_in, affine=True, subtract_last=False)

        # ── EMA decomposition (A1) ─────────────────────────────────────────────
        self.use_decomp = configs.use_decomp
        if self.use_decomp:
            self.decomp = EMADecomp(alpha=configs.alpha)

        # ── Backbone ───────────────────────────────────────────────────────────
        self.backbone = Backbone(
            seq_len           = configs.seq_len,
            d_model           = configs.d_model,
            channel           = configs.enc_in,
            t_ff              = configs.t_ff,
            c_ff              = configs.c_ff,
            patch_len         = configs.patch_len,
            stride            = configs.stride,
            dw_kernel         = configs.dw_kernel,
            small_kernel      = getattr(configs, 'small_kernel', 3),
            use_reparam       = getattr(configs, 'use_reparam', False),
            t_dropout         = configs.t_dropout,
            c_dropout         = configs.c_dropout,
            embed_dropout     = configs.embed_dropout,
            use_trend_stream  = configs.use_trend_stream,
            use_seas_stream   = configs.use_seas_stream,
            use_fusion_gate   = configs.use_fusion_gate,
            use_cross_channel = configs.use_cross_channel,
            use_alpha_gate    = configs.use_alpha_gate,
        )

        # ── Prediction head (channel-independent linear map) ───────────────────
        self.head = nn.Sequential(
            nn.Dropout(configs.head_dropout),
            nn.Linear(2 * configs.d_model, configs.pred_len),
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        Parameters
        ----------
        x_enc : [B, L, C]   lookback window (required)
        other args are accepted for compatibility with TSLib data loaders
        but are not used.

        Returns
        -------
        [B, pred_len, C]
        """
        # Normalise
        x = self.revin(x_enc, 'norm')                      # [B, L, C]

        # Decompose
        if self.use_decomp:
            seasonal, trend = self.decomp(x)               # [B, L, C] each
        else:
            seasonal = trend = x

        # Backbone expects [B, C, L]
        out  = self.backbone(
            seasonal.permute(0, 2, 1),
            trend.permute(0, 2, 1),
        )                                                   # [B, C, 2*d_model]

        # Predict and permute back
        pred = self.head(out).permute(0, 2, 1)             # [B, pred_len, C]

        # Denormalise
        return self.revin(pred, 'denorm')

    def structural_reparam(self) -> None:
        """
        Fuse reparameterised training branches for fast inference.
        Call once after training is complete.
        """
        self.backbone.merge_kernel()
