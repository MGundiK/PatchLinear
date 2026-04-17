#!/usr/bin/env bash
# =============================================================================
# PatchLinear — Imputation experiments
#
# Protocol: TimeMixer++ Table 4
#   seq_len   = 1024
#   mask_rate = {0.125, 0.25, 0.375, 0.5}
#   datasets  = ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Weather
#   metric    = MSE/MAE on masked positions, averaged across 4 mask rates
#
# Updates from multivariate tuning:
#   Electricity: c_ff updated 80→128 (cross-channel tuning carries over)
#   d_model kept at 64 (imputation is a different task — not re-tuned)
#
# Usage:
#   bash scripts/run_imputation.sh
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs/imputation"
mkdir -p "$LOG_DIR"
RESULT="$SCRIPT_DIR/result_imputation.txt"
> "$RESULT"

run_one() {
  local TAG=$1; shift
  local LOG="${LOG_DIR}/${TAG}.log"
  echo ">>> ${TAG}"
  python -u run_imputation.py "$@" > "$LOG" 2>&1
  grep "mse:" "$LOG" | tail -1 | tee -a "$RESULT" || true
}

MASK_RATES=(0.125 0.25 0.375 0.5)

# ── ETTh1 ─────────────────────────────────────────────────────────────────────
for MR in "${MASK_RATES[@]}"; do
  run_one "ETTh1_mr${MR}" \
    --is_training 1 --root_path ./dataset/ \
    --data_path ETTh1.csv --data ETTh1 --features M \
    --seq_len 1024 --enc_in 7 --mask_rate $MR \
    --d_model 64 --t_ff 128 --c_ff 16 \
    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
    --batch_size 16 --learning_rate 0.001 \
    --lradj sigmoid --train_epochs 100 --patience 10 \
    --model PatchLinear --seed 2021 --des Exp \
    --model_id "ETTh1_imputation_mr${MR}"
done

# ── ETTh2 ─────────────────────────────────────────────────────────────────────
for MR in "${MASK_RATES[@]}"; do
  run_one "ETTh2_mr${MR}" \
    --is_training 1 --root_path ./dataset/ \
    --data_path ETTh2.csv --data ETTh2 --features M \
    --seq_len 1024 --enc_in 7 --mask_rate $MR \
    --d_model 64 --t_ff 128 --c_ff 16 \
    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
    --batch_size 16 --learning_rate 0.001 \
    --lradj sigmoid --train_epochs 100 --patience 10 \
    --model PatchLinear --seed 2021 --des Exp \
    --model_id "ETTh2_imputation_mr${MR}"
done

# ── ETTm1 ─────────────────────────────────────────────────────────────────────
for MR in "${MASK_RATES[@]}"; do
  run_one "ETTm1_mr${MR}" \
    --is_training 1 --root_path ./dataset/ \
    --data_path ETTm1.csv --data ETTm1 --features M \
    --seq_len 1024 --enc_in 7 --mask_rate $MR \
    --d_model 64 --t_ff 128 --c_ff 16 \
    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
    --batch_size 16 --learning_rate 0.001 \
    --lradj sigmoid --train_epochs 100 --patience 10 \
    --model PatchLinear --seed 2021 --des Exp \
    --model_id "ETTm1_imputation_mr${MR}"
done

# ── ETTm2 ─────────────────────────────────────────────────────────────────────
for MR in "${MASK_RATES[@]}"; do
  run_one "ETTm2_mr${MR}" \
    --is_training 1 --root_path ./dataset/ \
    --data_path ETTm2.csv --data ETTm2 --features M \
    --seq_len 1024 --enc_in 7 --mask_rate $MR \
    --d_model 64 --t_ff 128 --c_ff 16 \
    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
    --batch_size 16 --learning_rate 0.001 \
    --lradj sigmoid --train_epochs 100 --patience 10 \
    --model PatchLinear --seed 2021 --des Exp \
    --model_id "ETTm2_imputation_mr${MR}"
done

# ── Weather ───────────────────────────────────────────────────────────────────
for MR in "${MASK_RATES[@]}"; do
  run_one "Weather_mr${MR}" \
    --is_training 1 --root_path ./dataset/ \
    --data_path weather.csv --data custom --features M \
    --seq_len 1024 --enc_in 21 --mask_rate $MR \
    --d_model 64 --t_ff 128 --c_ff 16 \
    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
    --batch_size 16 --learning_rate 0.001 \
    --lradj sigmoid --train_epochs 100 --patience 10 \
    --model PatchLinear --seed 2021 --des Exp \
    --model_id "Weather_imputation_mr${MR}"
done

# ── Electricity (c_ff=128 from multivariate tuning) ───────────────────────────
for MR in "${MASK_RATES[@]}"; do
  run_one "Electricity_mr${MR}" \
    --is_training 1 --root_path ./dataset/ \
    --data_path electricity.csv --data custom --features M \
    --seq_len 1024 --enc_in 321 --mask_rate $MR \
    --d_model 64 --t_ff 128 --c_ff 128 \
    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
    --batch_size 8 --learning_rate 0.001 \
    --lradj sigmoid --train_epochs 100 --patience 10 \
    --model PatchLinear --seed 2021 --des Exp \
    --model_id "Electricity_imputation_mr${MR}"
done

echo ""
echo "=== IMPUTATION RESULTS SUMMARY ==="
echo "Results saved to $RESULT"
echo ""
echo "Average MSE per dataset (across 4 mask rates):"
for DS in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity; do
  MSE_SUM=0; N=0
  for MR in "${MASK_RATES[@]}"; do
    MSE=$(grep "mse:" "$LOG_DIR/${DS}_mr${MR}.log" 2>/dev/null | tail -1 | \
          grep -oP 'mse:\K[\d.]+' || echo "0")
    MSE_SUM=$(python3 -c "print($MSE_SUM+$MSE)")
    N=$((N+1))
  done
  AVG=$(python3 -c "print(round($MSE_SUM/$N,4))")
  echo "  $DS: $AVG"
done
