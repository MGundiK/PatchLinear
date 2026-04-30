#!/usr/bin/env bash
# =============================================================================
# PatchLinear — M4 Short-Term Forecasting
# Each call to run_m4.py handles one frequency (train + predict + evaluate).
# All 6 freq CSVs accumulate in --forecast_dir; OWA is computed after each run.
# =============================================================================
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

model_name=PatchLinear
d_model=64
t_ff=128
c_ff=16
patch_len=8
stride=4
dw_kernel=3
batch_size=16
learning_rate=0.001
train_epochs=30
patience=10
seed=2021

LOG_DIR=logs/m4
mkdir -p "$LOG_DIR"

run_freq() {
  local FREQ=$1
  echo ">>> M4_${FREQ}"
  python -u run_m4.py \
    --model            "${model_name}" \
    --seasonal_patterns "${FREQ}" \
    --root_path        ./dataset/m4/ \
    --d_model          "${d_model}" \
    --t_ff             "${t_ff}" \
    --c_ff             "${c_ff}" \
    --patch_len        "${patch_len}" \
    --stride           "${stride}" \
    --dw_kernel        "${dw_kernel}" \
    --use_decomp       1 \
    --use_trend_stream 1 \
    --use_seas_stream  1 \
    --use_fusion_gate  1 \
    --use_cross_channel 0 \
    --use_alpha_gate   0 \
    --batch_size       "${batch_size}" \
    --learning_rate    "${learning_rate}" \
    --train_epochs     "${train_epochs}" \
    --seed             "${seed}" \
    2>&1 | tee -a "${LOG_DIR}/M4_${FREQ}.log"
  echo ""
}

# Run each frequency — order matches M4Summary grouping
# (Yearly/Quarterly/Monthly → main group; Weekly/Daily/Hourly → Others)
run_freq Monthly
run_freq Yearly
run_freq Quarterly
run_freq Weekly
run_freq Daily
run_freq Hourly

echo "=================================================="
echo "All M4 frequencies complete."
echo "Final results in logs/m4/results.txt"
echo "=================================================="
cat logs/m4/results.txt 2>/dev/null || true
