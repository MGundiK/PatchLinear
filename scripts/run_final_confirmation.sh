#!/usr/bin/env bash
# =============================================================================
# Final 3-seed confirmation — all single-seed winners vs xPatch Table 14
# Covers: ETTh1, ETTh2, ETTm1, ETTm2, Weather, Exchange, Solar + ILI deep search
# =============================================================================
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG_DIR="logs/final_confirm"
mkdir -p "$LOG_DIR"

run3() {
  local TAG=$1; shift
  echo ">>> ${TAG}"
  for SEED in 2021 2022 2023; do
    local LOG="${LOG_DIR}/${TAG}_s${SEED}.log"
    python -u run.py --is_training 1 "$@" \
      --model PatchLinear --seed ${SEED} --des Exp \
      --model_id "${TAG}_s${SEED}" \
      > "$LOG" 2>&1 || true
    grep -m1 "mse:" "$LOG" | tail -1 || echo "  [s${SEED} FAILED]"
  done
}

BASE_ETT="--root_path ./dataset/ --features M
  --lradj sigmoid --train_epochs 100 --patience 10 --enc_in 7
  --alpha 0.3 --batch_size 2048 --learning_rate 0.0005"

# ═══════════════════════════════════════════════════════════════════
# GROUP 1 — ETTh1 (3 pred_lens; best L/p/k/d from deep search)
# ═══════════════════════════════════════════════════════════════════
echo "=== ETTh1 ==="

# H96: L=512, p=16, k=7, d=64 — already has 3-seed from main table
# H192: d=96 gives 0.377 (vs xP 0.376) — NEW config needs confirmation
run3 ETTh1_pl192_L336_p24k3_d96 $BASE_ETT \
  --data_path ETTh1.csv --data ETTh1 \
  --seq_len 336 --label_len 48 --pred_len 192 \
  --patch_len 24 --stride 12 --dw_kernel 3 \
  --d_model 96 --t_ff 192 --c_ff 16

# H336: d=64 is still best (0.402 vs xP 0.391, gap irreducible) — skip
# H720: d=64 is still best (0.463 vs xP 0.442, gap irreducible) — skip

# ═══════════════════════════════════════════════════════════════════
# GROUP 2 — ETTh2 (2 pred_lens need confirmation)
# ═══════════════════════════════════════════════════════════════════
echo "=== ETTh2 ==="

# H96: 0.226 ties xPatch — L=720, p=32, k=3, d=64
run3 ETTh2_pl96_L720_p32k3_d64 $BASE_ETT \
  --data_path ETTh2.csv --data ETTh2 \
  --seq_len 720 --label_len 48 --pred_len 96 \
  --patch_len 32 --stride 16 --dw_kernel 3 \
  --d_model 64 --t_ff 128 --c_ff 16

# H192: L=720, p=16, k=7, d=64 — already has 3-seed; skip

# H336: d=96 gives 0.3118 (beats xP 0.312) — NEW config
run3 ETTh2_pl336_L512_p32k13_d96 $BASE_ETT \
  --data_path ETTh2.csv --data ETTh2 \
  --seq_len 512 --label_len 48 --pred_len 336 \
  --patch_len 32 --stride 16 --dw_kernel 13 \
  --d_model 96 --t_ff 192 --c_ff 16

# H720: 0.390 vs xP 0.384 (gap +0.006) — skip (not close enough)

# ═══════════════════════════════════════════════════════════════════
# GROUP 3 — ETTm1 (H96 beats xPatch — L=336, p=24, k=3)
# ═══════════════════════════════════════════════════════════════════
echo "=== ETTm1 ==="
run3 ETTm1_pl96_L336_p24k3_d64 $BASE_ETT \
  --data_path ETTm1.csv --data ETTm1 \
  --seq_len 336 --label_len 48 --pred_len 96 \
  --patch_len 24 --stride 12 --dw_kernel 3 \
  --d_model 64 --t_ff 128 --c_ff 16

# H192/336/720 already beat xPatch with L=336, p=16, k=7 — use existing 3-seed

# ═══════════════════════════════════════════════════════════════════
# GROUP 4 — ETTm2 (H336 and H720)
# ═══════════════════════════════════════════════════════════════════
echo "=== ETTm2 ==="
BASE_ETTm2="--root_path ./dataset/ --features M
  --lradj sigmoid --train_epochs 100 --patience 10 --enc_in 7
  --alpha 0.3 --batch_size 2048 --learning_rate 0.0001"

# H336: 0.263 beats xP 0.264 — L=336, p=16, k=3
run3 ETTm2_pl336_L336_p16k3_d64 $BASE_ETTm2 \
  --data_path ETTm2.csv --data ETTm2 \
  --seq_len 336 --label_len 48 --pred_len 336 \
  --patch_len 16 --stride 8 --dw_kernel 3 \
  --d_model 64 --t_ff 128 --c_ff 16

# H720: 0.337 beats xP 0.338 — L=720, p=8, k=3
run3 ETTm2_pl720_L720_p8k3_d64 $BASE_ETTm2 \
  --data_path ETTm2.csv --data ETTm2 \
  --seq_len 720 --label_len 48 --pred_len 720 \
  --patch_len 8 --stride 4 --dw_kernel 3 \
  --d_model 64 --t_ff 128 --c_ff 16

# ═══════════════════════════════════════════════════════════════════
# GROUP 5 — Weather (all 4 beat xPatch — already confirmed? Check)
# L=512/512/336/720, p=16, k=7, d=64 — use existing 3-seed from main table
# ═══════════════════════════════════════════════════════════════════
# Weather uses L=96 in main table. Need to confirm best-L configs.
echo "=== Weather ==="
BASE_WEA="--root_path ./dataset/ --features M
  --lradj sigmoid --train_epochs 100 --patience 10 --enc_in 21
  --alpha 0.3 --batch_size 2048 --learning_rate 0.0005
  --d_model 64 --t_ff 128 --c_ff 16"

run3 Weather_pl96_L512_p16k7  $BASE_WEA \
  --data_path weather.csv --data custom \
  --seq_len 512 --label_len 48 --pred_len 96  --patch_len 16 --stride 8 --dw_kernel 7

run3 Weather_pl192_L512_p16k7 $BASE_WEA \
  --data_path weather.csv --data custom \
  --seq_len 512 --label_len 48 --pred_len 192 --patch_len 16 --stride 8 --dw_kernel 7

run3 Weather_pl336_L336_p16k7 $BASE_WEA \
  --data_path weather.csv --data custom \
  --seq_len 336 --label_len 48 --pred_len 336 --patch_len 16 --stride 8 --dw_kernel 7

run3 Weather_pl720_L720_p16k7 $BASE_WEA \
  --data_path weather.csv --data custom \
  --seq_len 720 --label_len 48 --pred_len 720 --patch_len 16 --stride 8 --dw_kernel 7

# ═══════════════════════════════════════════════════════════════════
# GROUP 6 — ILI (3 wins need confirmation; pl=24 skip)
# Best config: d=64, c=8, L=104, lr=0.005 for pl=36/48/60
# ═══════════════════════════════════════════════════════════════════
echo "=== ILI ==="
BASE_ILI="--root_path ./dataset/ --features M
  --lradj type3 --train_epochs 100 --patience 10 --enc_in 7
  --alpha 0.3 --batch_size 32 --learning_rate 0.005
  --seq_len 104 --label_len 18
  --patch_len 6 --stride 3 --dw_kernel 3
  --d_model 64 --t_ff 128 --c_ff 8
  --use_decomp 0 --use_seas_stream 0 --use_fusion_gate 0"

# pl=24: d=48, c=32 (best single seed 1.305, still behind xP 1.188 — confirm anyway)
run3 ILI_pl24_L104_d48_c32_lr01 \
  --root_path ./dataset/ --features M --data_path national_illness.csv --data custom \
  --lradj type3 --train_epochs 100 --patience 10 --enc_in 7 \
  --alpha 0.3 --batch_size 32 --learning_rate 0.01 \
  --seq_len 104 --label_len 18 --pred_len 24 \
  --patch_len 6 --stride 3 --dw_kernel 3 \
  --d_model 48 --t_ff 96 --c_ff 32 \
  --use_decomp 0 --use_seas_stream 0 --use_fusion_gate 0

# pl=36: 1.217 beats xP 1.226
run3 ILI_pl36_L104_d64_c8_lr005 $BASE_ILI \
  --data_path national_illness.csv --data custom --pred_len 36

# pl=48: 1.223 beats xP 1.254
run3 ILI_pl48_L104_d64_c8_lr005 $BASE_ILI \
  --data_path national_illness.csv --data custom --pred_len 48

# pl=60: 1.412 beats xP 1.455
run3 ILI_pl60_L104_d64_c8_lr005 $BASE_ILI \
  --data_path national_illness.csv --data custom --pred_len 60

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
python3 - << 'PYEOF'
import re, os, numpy as np
from collections import defaultdict

log_dir = "logs/final_confirm"
xpatch = {
    'ETTh1_pl192':(0.376,0.395), 'ETTh2_pl96':(0.226,0.297),
    'ETTh2_pl336':(0.312,0.360), 'ETTm1_pl96':(0.275,0.330),
    'ETTm2_pl336':(0.264,0.315), 'ETTm2_pl720':(0.338,0.363),
    'Weather_pl96':(0.146,0.185),'Weather_pl192':(0.189,0.227),
    'Weather_pl336':(0.218,0.260),'Weather_pl720':(0.291,0.315),
    'ILI_pl24':(1.188,0.638),'ILI_pl36':(1.226,0.653),
    'ILI_pl48':(1.254,0.686),'ILI_pl60':(1.455,0.773),
}

results = defaultdict(list)
for fname in sorted(os.listdir(log_dir)):
    if not fname.endswith('.log'): continue
    tag = re.sub(r'_s\d{4}\.log$', '', fname)
    for line in open(os.path.join(log_dir, fname)):
        m = re.search(r'mse:\s*([\d.]+).*?mae:\s*([\d.]+)', line)
        if m:
            results[tag].append((float(m.group(1)), float(m.group(2))))

print(f"\n{'Config':<35} {'MSE mean':>9} {'std':>7}  {'MAE mean':>9}  {'xP MSE':>7}  {'Δ':>7}  result")
print("─"*95)
for tag, vals in sorted(results.items()):
    if len(vals) >= 2:
        mses = [v[0] for v in vals]
        maes = [v[1] for v in vals]
        mm = np.mean(mses); ms = np.std(mses)
        am = np.mean(maes)
        # Find matching xpatch key
        xk = None
        for k in xpatch:
            ds, pl = k.split('_pl')
            if ds in tag and f'pl{pl}' in tag:
                xk = k; break
        if xk:
            xm, xa = xpatch[xk]
            delta = mm - xm
            win = '★ WIN' if round(mm,3)<=round(xm,3) else ('≈tie' if delta<0.002 else '—')
            seeds = len(vals)
            print(f"{tag:<35} {mm:>9.3f} {ms:>7.4f}  {am:>9.3f}  {xm:>7.3f}  {delta:>+7.3f}  {win}  ({seeds} seeds)")
        else:
            print(f"{tag:<35} {mm:>9.3f} {ms:>7.4f}  {am:>9.3f}  {'?':>7}  {'?':>7}")
PYEOF
