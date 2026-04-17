"""
PatchLinear — imputation-capable model.

Adds task_name flag:
  'long_term_forecast'  (default) — pred_len output, same as before
  'imputation'          — seq_len output, reconstruction head

The backbone is identical for both tasks. Only the head and forward
pass differ. This keeps the imputation model directly comparable to
the forecasting model in ablation studies.

Imputation forward pass:
  1. Apply mask to input  (done outside the model, in exp_imputation)
  2. RevIN normalise the ORIGINAL unmasked input statistics
  3. Run backbone on masked input
  4. Reconstruction head outputs [B, seq_len, C]
  5. RevIN denormalise
  6. Loss computed only on masked positions in exp_imputation
"""

import torch
import torch.nn as nn

from layers.revin    import RevIN
from layers.ema      import EMADecomp
from layers.backbone import Backbone


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, 'task_name', 'long_term_forecast')
        self.seq_len   = configs.seq_len

        self.revin      = RevIN(configs.enc_in, affine=True, subtract_last=False)
        self.use_decomp = configs.use_decomp
        if self.use_decomp:
            self.decomp = EMADecomp(alpha=configs.alpha)

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

        # Forecasting head: [B, C, 2d] -> [B, C, pred_len]
        if self.task_name == 'long_term_forecast':
            self.head = nn.Sequential(
                nn.Dropout(configs.head_dropout),
                nn.Linear(2 * configs.d_model, configs.pred_len),
            )
        # Imputation head: [B, C, 2d] -> [B, C, seq_len]
        elif self.task_name == 'imputation':
            self.head = nn.Sequential(
                nn.Dropout(configs.head_dropout),
                nn.Linear(2 * configs.d_model, configs.seq_len),
            )
        else:
            raise ValueError(f"Unknown task_name: {self.task_name}")

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                mask=None):
        """
        Parameters
        ----------
        x_enc    : [B, L, C]  — for imputation this is the MASKED input
                                (masked positions set to 0)
        mask     : [B, L, C]  — 1 = observed, 0 = masked (imputation only,
                                not used inside the model but accepted for
                                API compatibility with exp_imputation)

        Returns
        -------
        [B, L, C]  for imputation   (L = seq_len)
        [B, H, C]  for forecasting  (H = pred_len)
        """
        # Normalise using statistics of the (possibly masked) input
        x = self.revin(x_enc, 'norm')

        if self.use_decomp:
            seasonal, trend = self.decomp(x)
        else:
            seasonal = trend = x

        out  = self.backbone(
            seasonal.permute(0, 2, 1),
            trend.permute(0, 2, 1),
        )                                        # [B, C, 2d]
        pred = self.head(out).permute(0, 2, 1)  # [B, L or H, C]
        return self.revin(pred, 'denorm')

    def structural_reparam(self):
        self.backbone.merge_kernel()
