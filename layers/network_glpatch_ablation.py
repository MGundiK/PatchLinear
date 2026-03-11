import torch
from torch import nn


class InterPatchGating(nn.Module):
    """GLCN-inspired inter-patch gating module."""
    def __init__(self, patch_num, reduction=4):
        super(InterPatchGating, self).__init__()
        hidden = max(patch_num // reduction, 2)
        self.mlp = nn.Sequential(
            nn.Linear(patch_num, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, patch_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = x.mean(dim=2)
        w = self.mlp(w).unsqueeze(2)
        return x * w


class GLPatchNetworkAblation(nn.Module):
    """
    GLPatch v8 with ablation flags for systematic experiments.

    Ablation controls:
        use_gating:       Enable inter-patch gating (True/False)
        use_fusion:       Enable adaptive stream fusion (True/False)
        gate_position:    'pre_depthwise' | 'pre_pointwise' | 'post_pointwise'
        res_alpha_init:   Initial value for gating residual blend (0.05)
        gate_hidden_dim:  Bottleneck dim for fusion gate (32), -1 for full-rank
        gate_min:         Lower bound of gate constraint (0.1)
        gate_max:         Upper bound of gate constraint (0.9)
        gate_reduction:   Reduction ratio for gating MLP (4)
    """
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 use_gating=True, use_fusion=True,
                 gate_position='pre_pointwise',
                 res_alpha_init=0.05,
                 gate_hidden_dim=32,
                 gate_min=0.1, gate_max=0.9,
                 gate_reduction=4):
        super(GLPatchNetworkAblation, self).__init__()

        # Store ablation config
        self.use_gating = use_gating
        self.use_fusion = use_fusion
        self.gate_position = gate_position
        self.gate_min = gate_min
        self.gate_max = gate_max

        # Parameters
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1

        # ================================================================
        # Non-linear Stream (Seasonality)
        # ================================================================

        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        self.fc2 = nn.Linear(self.dim, patch_len)

        # Inter-Patch Gating (conditional)
        if use_gating:
            self.inter_patch_gate = InterPatchGating(self.patch_num, reduction=gate_reduction)
            self.res_alpha = nn.Parameter(torch.tensor(float(res_alpha_init)))

        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # ================================================================
        # Linear Stream (Trend) — identical to xPatch
        # ================================================================
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # ================================================================
        # Stream Fusion (conditional)
        # ================================================================
        if use_fusion:
            if gate_hidden_dim == -1:
                # Full-rank gate (no bottleneck) — for ablation
                self.gate_fc = nn.Linear(pred_len * 2, pred_len)
                nn.init.normal_(self.gate_fc.weight, std=0.01)
                nn.init.zeros_(self.gate_fc.bias)
                self.full_rank = True
            else:
                gate_hidden = min(gate_hidden_dim, pred_len)
                self.gate_compress_s = nn.Linear(pred_len, gate_hidden)
                self.gate_compress_t = nn.Linear(pred_len, gate_hidden)
                self.gate_expand = nn.Linear(gate_hidden, pred_len)
                nn.init.normal_(self.gate_compress_s.weight, std=0.01)
                nn.init.normal_(self.gate_compress_t.weight, std=0.01)
                nn.init.normal_(self.gate_expand.weight, std=0.01)
                nn.init.zeros_(self.gate_compress_s.bias)
                nn.init.zeros_(self.gate_compress_t.bias)
                nn.init.zeros_(self.gate_expand.bias)
                self.full_rank = False

        self.fc8 = nn.Linear(pred_len, pred_len)

    def _apply_gating(self, s):
        """Apply inter-patch gating with residual blend."""
        s_base = s
        s_gated = self.inter_patch_gate(s)
        return s_base + self.res_alpha * (s_gated - s_base)

    def forward(self, s, t):
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        B, C, I = s.shape
        s = torch.reshape(s, (B * C, I))
        t = torch.reshape(t, (B * C, I))

        # ---- Non-linear Stream ----

        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Patch Embedding
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        # Gating position: pre_depthwise
        if self.use_gating and self.gate_position == 'pre_depthwise':
            s = self._apply_gating(s)

        # CNN Depthwise
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # Residual Stream
        res = self.fc2(res)
        s = s + res

        # Gating position: pre_pointwise (default, v8)
        if self.use_gating and self.gate_position == 'pre_pointwise':
            s = self._apply_gating(s)

        # CNN Pointwise
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # Gating position: post_pointwise
        if self.use_gating and self.gate_position == 'post_pointwise':
            s = self._apply_gating(s)

        # Flatten Head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        # ---- Linear Stream ----
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)
        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)
        t = self.fc7(t)

        # ---- Stream Fusion ----
        if self.use_fusion:
            if self.full_rank:
                gate = torch.sigmoid(self.gate_fc(torch.cat([s, t], dim=-1)))
            else:
                gate = torch.sigmoid(
                    self.gate_expand(
                        self.gate_compress_s(s) + self.gate_compress_t(t)
                    )
                )
            # Apply constraint
            gate_range = self.gate_max - self.gate_min
            gate = gate * gate_range + self.gate_min

            x = gate * s + (1 - gate) * t
        else:
            # Static fusion (xPatch style): simple addition
            x = s + t

        x = self.fc8(x)

        x = torch.reshape(x, (B, C, self.pred_len))
        x = x.permute(0, 2, 1)

        return x
