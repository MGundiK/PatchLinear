#!/usr/bin/env bash
# =============================================================================
# PatchLinear — Few-Shot Forecasting (10% training data)
# Protocol: TimeMixer++ Section 4.1.5
#   - Train on 10% of available timesteps per dataset
#   - Test on full test set
#   - Averaged over 4 prediction lengths {96, 192, 336, 720}
#   - Datasets: ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity
#   - Report: ETT(Avg), Weather, ECL (Electricity)
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/logs/fewshot"
mkdir -p "$LOG_DIR"
RESULT="$SCRIPT_DIR/result_fewshot.txt"
> "$RESULT"

run_one() {
  local TAG=$1; shift
  local LOG="${LOG_DIR}/${TAG}.log"
  echo ">>> ${TAG}"
  python -u run.py "$@" > "$LOG" 2>&1
  grep "mse:" "$LOG" | tail -1 | tee -a "$RESULT" || echo "  [FAILED]"
}

SEED=2021

# Common flags — identical to full training except --few_shot_ratio 0.1
BASE="--is_training 1 --root_path ./dataset/ --features M
      --seq_len 96 --label_len 48 --enc_in 7
      --d_model 64 --t_ff 128 --c_ff 16
      --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3
      --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1
      --use_cross_channel 1 --use_alpha_gate 1
      --lradj sigmoid --train_epochs 100 --patience 10
      --model PatchLinear --seed ${SEED} --des Exp
      --few_shot_ratio 0.1"

# ── ETTh1 ─────────────────────────────────────────────────────────────────────
for PL in 96 192 336 720; do
  run_one "ETTh1_fs_pl${PL}" $BASE \
    --data_path ETTh1.csv --data ETTh1 \
    --batch_size 2048 --learning_rate 0.0005 --pred_len ${PL} \
    --model_id "ETTh1_fs_pl${PL}"
done

# ── ETTh2 ─────────────────────────────────────────────────────────────────────
for PL in 96 192 336 720; do
  run_one "ETTh2_fs_pl${PL}" $BASE \
    --data_path ETTh2.csv --data ETTh2 \
    --batch_size 2048 --learning_rate 0.0005 --pred_len ${PL} \
    --model_id "ETTh2_fs_pl${PL}"
done

# ── ETTm1 ─────────────────────────────────────────────────────────────────────
for PL in 96 192 336 720; do
  run_one "ETTm1_fs_pl${PL}" $BASE \
    --data_path ETTm1.csv --data ETTm1 \
    --batch_size 2048 --learning_rate 0.0005 --pred_len ${PL} \
    --model_id "ETTm1_fs_pl${PL}"
done

# ── ETTm2 ─────────────────────────────────────────────────────────────────────
for PL in 96 192 336 720; do
  run_one "ETTm2_fs_pl${PL}" $BASE \
    --data_path ETTm2.csv --data ETTm2 \
    --batch_size 2048 --learning_rate 0.0001 --pred_len ${PL} \
    --model_id "ETTm2_fs_pl${PL}"
done

# ── Weather ───────────────────────────────────────────────────────────────────
for PL in 96 192 336 720; do
  run_one "Weather_fs_pl${PL}" \
    --is_training 1 --root_path ./dataset/ --features M \
    --seq_len 96 --label_len 48 --enc_in 21 \
    --d_model 64 --t_ff 128 --c_ff 16 \
    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
    --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
    --use_cross_channel 1 --use_alpha_gate 1 \
    --batch_size 2048 --learning_rate 0.0005 --pred_len ${PL} \
    --lradj sigmoid --train_epochs 100 --patience 10 \
    --model PatchLinear --seed ${SEED} --des Exp \
    --few_shot_ratio 0.1 \
    --data_path weather.csv --data custom \
    --model_id "Weather_fs_pl${PL}"
done

# ── Electricity (ECL) ──────────────────────────────────────────────────────────
for PL in 96 192 336 720; do
  run_one "ECL_fs_pl${PL}" \
    --is_training 1 --root_path ./dataset/ --features M \
    --seq_len 96 --label_len 48 --enc_in 321 \
    --d_model 256 --t_ff 512 --c_ff 128 \
    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
    --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
    --use_cross_channel 1 --use_alpha_gate 1 \
    --batch_size 256 --learning_rate 0.01 --pred_len ${PL} \
    --lradj sigmoid --train_epochs 100 --patience 10 \
    --model PatchLinear --seed ${SEED} --des Exp \
    --few_shot_ratio 0.1 \
    --data_path electricity.csv --data custom \
    --model_id "ECL_fs_pl${PL}"
done

echo ""
echo "=== FEW-SHOT SUMMARY (avg across 4 horizons) ==="
python3 << 'PYEOF'
import re, os, numpy as np
log_dir = "logs/fewshot"
datasets = {
    'ETT(Avg)': ['ETTh1','ETTh2','ETTm1','ETTm2'],
    'Weather': ['Weather'],
    'ECL': ['ECL'],
}
for label, dsets in datasets.items():
    mse_all=[]; mae_all=[]
    for ds in dsets:
        for pl in [96,192,336,720]:
            f=f"{log_dir}/{ds}_fs_pl{pl}.log"
            if not os.path.exists(f): continue
            for line in open(f):
                m=re.search(r'mse:([\d.]+),?\s*mae:([\d.]+)',line)
                if m: mse_all.append(float(m.group(1))); mae_all.append(float(m.group(2)))
    if mse_all:
        print(f"{label}: MSE={np.mean(mse_all):.3f} MAE={np.mean(mae_all):.3f}")
PYEOF
