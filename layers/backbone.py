"""
Core building blocks of PatchLinear that sit between the input normalization
and the prediction head.

GatingBlock        : shared MLP-sigmoid gate primitive (XLinear).
SeasonalConvStream : xPatch patching + ModernTCN large-kernel DWConv (A6).
Backbone           : dual-stream encoding + TGM + VGM + alpha gate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reparam_conv import ReparamDWConv


# ---------------------------------------------------------------------------
# GatingBlock  (XLinear, unchanged)
# ---------------------------------------------------------------------------
class GatingBlock(nn.Module):
    """
    Input-dependent feature selector used as both TGM and VGM.

        output = x  *  sigmoid( Linear2( ReLU( Linear1(x) ) ) )

    The sigmoid output constrains gate values to [0, 1] so features are
    suppressed rather than sign-flipped.  Using the same primitive for both
    temporal and cross-channel gating reduces the number of design choices
    that need independent justification.

    Parameters
    ----------
    d_model : input (and output) feature dimension
    d_ff    : hidden dimension of the two-layer MLP
    dropout : applied between the two linear layers
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


# ---------------------------------------------------------------------------
# SeasonalConvStream
# ---------------------------------------------------------------------------
class SeasonalConvStream(nn.Module):
    """
    Seasonal (periodic) encoder combining xPatch patching with ModernTCN's
    large-kernel depthwise convolution along the patch/temporal axis.

    Pipeline
    --------
    1. ReplicationPad + unfold  ->  patches [B*C, N, patch_len]
    2. Linear embed             ->  [B*C, N, d_model]
    3. Permute                  ->  [B*C, d_model, N]
    4. Large-kernel DWConv (groups=d_model, kernel=dw_kernel along N)
       This is ModernTCN's DWConv applied to the patched sequence.
       groups=d_model makes the convolution feature-independent: each of
       the d_model features has its own 1-D temporal filter.
    5. Pointwise ConvFFN (groups=1) -- ModernTCN ConvFFN1
       Mixes all d_model features at each patch position.
    6. Flatten + Linear         ->  [B*C, d_model]

    Effective Receptive Field (ERF) in raw timesteps:
        ERF = (dw_kernel - 1) * stride + patch_len

        dw_kernel=3,  stride=8, patch_len=16  =>  ERF ~32   (local)
        dw_kernel=7,  stride=8, patch_len=16  =>  ERF ~64   (medium)
        dw_kernel=13, stride=8, patch_len=16  =>  ERF ~112  (full input)

    ERF grows linearly with dw_kernel (ModernTCN Sec. 5.2), making kernel
    size a direct and interpretable control over temporal context.

    Ablation A6
    -----------
    Compare dw_kernel in {3, 7, 13} with all other settings fixed.
    Claim: larger kernel helps on datasets with long periodic structure
    spanning multiple patches (ETTm: 15-min sampling, 96-step daily cycle;
    Weather: 10-min sampling, multi-hour cycles).
    Expected negligible benefit on Exchange (near-random walk, no periodicity).

    Parameters
    ----------
    seq_len     : lookback window length L
    d_model     : output embedding dimension
    patch_len   : length of each patch in raw timesteps
    stride      : step between consecutive patch starts
    dw_kernel   : kernel size for the DWConv along the patch axis (A6)
    small_kernel: small branch kernel for structural reparameterisation
    use_reparam : if True, use ReparamDWConv instead of plain Conv1d
    dropout     : applied before the final output projection
    """
    def __init__(
        self,
        seq_len:      int,
        d_model:      int,
        patch_len:    int   = 16,
        stride:       int   = 8,
        dw_kernel:    int   = 7,
        small_kernel: int   = 3,
        use_reparam:  bool  = False,
        dropout:      float = 0.0,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        # End-padding ensures the last (potentially incomplete) patch is populated
        self.pad  = nn.ReplicationPad1d((0, stride))
        N         = (seq_len - patch_len) // stride + 1 + 1   # +1 for pad patch

        # Step 2: per-patch linear embedding (shared across all patches)
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.bn_embed    = nn.BatchNorm1d(d_model)

        # Step 4: large-kernel DWConv along N (A6 / ModernTCN ERF argument)
        if use_reparam:
            self.dw_conv = ReparamDWConv(d_model, dw_kernel, small_kernel)
        else:
            self.dw_conv = nn.Conv1d(
                d_model, d_model,
                kernel_size=dw_kernel,
                padding=dw_kernel // 2,
                groups=d_model,
                bias=False,
            )
        self.bn_dw = nn.BatchNorm1d(d_model)

        # Step 5: pointwise ConvFFN (ModernTCN ConvFFN1, cross-feature mixing)
        self.pw_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.bn_pw   = nn.BatchNorm1d(d_model)

        # Step 6: collapse temporal dimension to d_model
        self.out_proj = nn.Linear(d_model * N, d_model)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        B, C, L = x.shape
        x = x.reshape(B * C, L)

        # Pad and unfold into patches
        x = self.pad(x)                                    # [B*C, L+stride]
        x = x.unfold(-1, self.patch_len, self.stride)      # [B*C, N, patch_len]

        # Embed patches
        x = self.patch_embed(x)                            # [B*C, N, d_model]
        x = F.gelu(x)
        x = x.permute(0, 2, 1)                            # [B*C, d_model, N]
        x = self.bn_embed(x)

        # Large-kernel DWConv along temporal/patch axis
        x = self.dw_conv(x)                               # [B*C, d_model, N]
        x = F.gelu(x)
        x = self.bn_dw(x)

        # Pointwise ConvFFN
        x = self.pw_conv(x)                               # [B*C, d_model, N]
        x = F.gelu(x)
        x = self.bn_pw(x)

        # Flatten and project
        x = x.flatten(start_dim=1)                        # [B*C, d_model*N]
        x = self.drop(x)
        x = self.out_proj(x)                              # [B*C, d_model]
        return x.reshape(B, C, -1)                        # [B, C, d_model]

    def merge_kernel(self) -> None:
        """Delegate structural reparameterisation to the DWConv if active."""
        if isinstance(self.dw_conv, ReparamDWConv):
            self.dw_conv.merge_kernel()


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
class Backbone(nn.Module):
    """
    [B, C, L]  ->  [B, C, 2 * d_model]

    Stage 1  Dual-stream temporal encoding  (channel-independent)
    ──────────────────────────────────────────────────────────────
    trend_proj      Linear(L -> d_model).
                    Each of the d_model outputs is a learned weighted sum over
                    ALL L timesteps simultaneously — global ERF by construction,
                    zero padding bias.  Receives the EMA trend component.

    seasonal_stream SeasonalConvStream (see above).
                    Receives the EMA seasonal (residual) component.

    Stage 2  Stream fusion  (A3)
    ─────────────────────────────
    gate  = sigmoid( Linear( [t_emb || s_emb] ) )   shape [B, C, d]
    emb   = gate * s_emb + (1 - gate) * t_emb

    Claim: input-dependent interpolation outperforms static addition because
    the relative dominance of trend vs. seasonal varies across windows and
    datasets.  A3 ablation: replace fusion_gate with simple addition.

    Stage 3  Temporal gating  (TGM, XLinear unchanged)
    ────────────────────────────────────────────────────
    One learnable global token per channel (shape [1, C, d]).
    GatingBlock on [emb || glob_token] along the 2*d_model feature axis.
    This stage is channel-independent: channels do not interact here.
    Output split: origin [B,C,d]  +  glob_updated [B,C,d]

    Stage 4  Cross-channel gating  (VGM + alpha, XLinear + new alpha)
    ───────────────────────────────────────────────────────────────────
    VGM  (A4a): GatingBlock on [emb || glob_updated] along the 2*C channel axis.
                Every channel's global token can attend to every other channel's
                embedding.  Crucially, interaction flows through the global token
                bottleneck only — the raw temporal features are never mixed
                across channels, preventing cross-channel noise from corrupting
                temporal representations.

    alpha (A4b): Scalar in (0, 1) per channel, conditioned on origin via a
                 single Linear(d_model -> 1) + sigmoid.
                    alpha -> 0: channel-independent behaviour for that channel.
                    alpha -> 1: full cross-channel mixing.
                 This allows the model to suppress cross-channel contributions
                 for channels that are currently in an unusual regime.

    Interpretability test (A5)
    ───────────────────────────
    After training, call get_alpha_values() on the test set and average alpha
    per channel.  Expected pattern:
        Traffic   -> high alpha  (862 sensors, dense spatial correlations)
        Exchange  -> low alpha   (8 currency pairs, unstable correlations)
    This constitutes a qualitative sanity check for A4b independent of MSE.

    Parameters
    ----------
    All flags default to the full-model configuration.  Set any flag to False
    to produce the corresponding ablation variant.
    """
    def __init__(
        self,
        seq_len:           int,
        d_model:           int,
        channel:           int,
        t_ff:              int,
        c_ff:              int,
        patch_len:         int   = 16,
        stride:            int   = 8,
        dw_kernel:         int   = 7,
        small_kernel:      int   = 3,
        use_reparam:       bool  = False,
        t_dropout:         float = 0.0,
        c_dropout:         float = 0.0,
        embed_dropout:     float = 0.0,
        use_trend_stream:  bool  = True,   # A2b: set False for seasonal-only
        use_seas_stream:   bool  = True,   # A2a: set False for trend-only
        use_fusion_gate:   bool  = True,   # A3
        use_cross_channel: bool  = True,   # A4a
        use_alpha_gate:    bool  = True,   # A4b
    ):
        super().__init__()
        self.d_model           = d_model
        self.channel           = channel
        self.use_trend_stream  = use_trend_stream
        self.use_seas_stream   = use_seas_stream
        self.use_cross_channel = use_cross_channel
        self.use_alpha_gate    = use_alpha_gate
        self.use_fusion_gate   = use_fusion_gate

        assert use_trend_stream or use_seas_stream, \
            "At least one stream must be active."

        # ── Trend stream (XLinear linear projection) ──────────────────────────
        if use_trend_stream:
            self.trend_proj = nn.Sequential(
                nn.Linear(seq_len, d_model),
                nn.Dropout(embed_dropout),
            )

        # ── Seasonal stream (xPatch patching + ModernTCN DWConv) ──────────────
        if use_seas_stream:
            self.seasonal_stream = SeasonalConvStream(
                seq_len, d_model, patch_len, stride,
                dw_kernel=dw_kernel,
                small_kernel=small_kernel,
                use_reparam=use_reparam,
                dropout=embed_dropout,
            )

        # ── Stream fusion gate (A3) ────────────────────────────────────────────
        if use_fusion_gate and use_trend_stream and use_seas_stream:
            self.fusion_gate = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.Sigmoid(),
            )

        # ── Global token: one learnable vector per channel (XLinear) ──────────
        self.glob_token = nn.Parameter(torch.zeros(1, channel, d_model))

        # ── TGM: temporal gating (XLinear, unchanged) ─────────────────────────
        self.tgm = GatingBlock(2 * d_model, t_ff, t_dropout)

        # ── VGM + alpha gate ───────────────────────────────────────────────────
        if use_cross_channel:
            self.vgm = GatingBlock(2 * channel, c_ff, c_dropout)
            if use_alpha_gate:
                self.alpha_gate = nn.Linear(d_model, 1)

    # ── private helper ─────────────────────────────────────────────────────────
    def _encode(self, seasonal: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
        """Dual stream + optional fusion -> emb [B, C, d_model]."""
        if self.use_trend_stream and self.use_seas_stream:
            t_emb = self.trend_proj(trend)
            s_emb = self.seasonal_stream(seasonal)
            if self.use_fusion_gate:
                gate = self.fusion_gate(torch.cat([t_emb, s_emb], dim=-1))
                return gate * s_emb + (1.0 - gate) * t_emb
            return t_emb + s_emb
        if self.use_trend_stream:
            return self.trend_proj(trend)
        return self.seasonal_stream(seasonal)

    # ── public forward ─────────────────────────────────────────────────────────
    def forward(self, seasonal: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        seasonal, trend : [B, C, L]

        Returns
        -------
        [B, C, 2 * d_model]
        """
        B   = seasonal.shape[0]
        emb = self._encode(seasonal, trend)                      # [B, C, d]

        # TGM
        glob     = self.glob_token.expand(B, -1, -1)             # [B, C, d]
        en_gated = self.tgm(torch.cat([emb, glob], dim=-1))      # [B, C, 2d]
        origin       = en_gated[:, :, : self.d_model]            # [B, C, d]
        glob_updated = en_gated[:, :, self.d_model :]            # [B, C, d]

        # VGM + alpha
        if self.use_cross_channel:
            ex_in      = torch.cat([emb, glob_updated], dim=1)           # [B, 2C, d]
            ex_out     = self.vgm(ex_in.permute(0, 2, 1))                # [B, d, 2C]
            glob_cross = ex_out[:, :, self.channel:].permute(0, 2, 1)    # [B, C, d]
            if self.use_alpha_gate:
                alpha      = torch.sigmoid(self.alpha_gate(origin))      # [B, C, 1]
                glob_cross = alpha * glob_cross
        else:
            glob_cross = torch.zeros_like(origin)

        return torch.cat([origin, glob_cross], dim=-1)                    # [B, C, 2d]

    def get_alpha_values(self, seasonal: torch.Tensor, trend: torch.Tensor):
        """
        Forward pass that also returns alpha per channel.
        Used for the interpretability analysis (A4b / A5).

        Returns
        -------
        output : [B, C, 2*d_model]
        alpha  : [B, C, 1]  -- per-channel mixing coefficient
        """
        if not (self.use_cross_channel and self.use_alpha_gate):
            raise RuntimeError(
                "Alpha gate is not active (use_cross_channel and use_alpha_gate "
                "must both be True)."
            )
        B   = seasonal.shape[0]
        emb = self._encode(seasonal, trend)
        glob     = self.glob_token.expand(B, -1, -1)
        en_gated = self.tgm(torch.cat([emb, glob], dim=-1))
        origin       = en_gated[:, :, : self.d_model]
        glob_updated = en_gated[:, :, self.d_model :]
        ex_out     = self.vgm(
            torch.cat([emb, glob_updated], dim=1).permute(0, 2, 1)
        )
        glob_cross = ex_out[:, :, self.channel:].permute(0, 2, 1)
        alpha      = torch.sigmoid(self.alpha_gate(origin))               # [B, C, 1]
        output     = torch.cat([origin, alpha * glob_cross], dim=-1)
        return output, alpha

    def merge_kernel(self) -> None:
        """Delegate structural reparameterisation to the seasonal stream."""
        if self.use_seas_stream:
            self.seasonal_stream.merge_kernel()
