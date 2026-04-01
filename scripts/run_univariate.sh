#!/usr/bin/env bash
# =============================================================================
# PatchLinear — Univariate long-term forecasting (features=S)
#
# Uses same datasets and horizons as the multivariate benchmark.
# enc_in=1, features=S: each channel forecasted independently.
# No changes to architecture or hyperparameters needed.
#
# Usage:
#   bash scripts/run_univariate.sh
#   bash scripts/run_univariate.sh --seeds
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

MULTI_SEED=0
for arg in "$@"; do
  case $arg in --seeds) MULTI_SEED=1 ;; esac
done

LOG_DIR="$SCRIPT_DIR/logs/univariate"
mkdir -p "$LOG_DIR"
SEEDS=(2021 2022 2023)

run_one() {
  local TAG=$1; shift
  local LOG="${LOG_DIR}/${TAG}.log"
  echo ">>> ${TAG}"
  python -u run.py "$@" > "$LOG" 2>&1
  grep "mse:" "$LOG" | tail -1 || true
}

# =============================================================================
# DATASETS
# Fields: name  data_path  data_type  batch  lr
# enc_in=1 for all (univariate — model sees one channel at a time)
# seq_len=96 for all standard datasets
# =============================================================================
declare -a DATASETS=(
  "ETTh1       ETTh1.csv            ETTh1      2048  0.0005"
  "ETTh2       ETTh2.csv            ETTh2      2048  0.0005"
  "ETTm1       ETTm1.csv            ETTm1      2048  0.0005"
  "ETTm2       ETTm2.csv            ETTm2      2048  0.0001"
  "Weather     weather.csv          custom     2048  0.0005"
  "Traffic     traffic.csv          custom     256   0.005"
  "Electricity electricity.csv      custom     256   0.005"
  "Exchange    exchange_rate.csv    custom     32    0.00001"
  "Solar       solar.txt            Solar      512   0.005"
)

for dataset_entry in "${DATASETS[@]}"; do
  read -r NAME DATA_PATH DATA_TYPE BATCH LR <<< "$dataset_entry"

  for PRED_LEN in 96 192 336 720; do

    COMMON=(
      --is_training 1
      --root_path ./dataset/
      --data_path "${DATA_PATH}"
      --data "${DATA_TYPE}"
      --features S
      --seq_len 96
      --label_len 48
      --pred_len "${PRED_LEN}"
      --enc_in 1
      --d_model 64
      --t_ff 128
      --c_ff 16
      --patch_len 16
      --stride 8
      --dw_kernel 7
      --alpha 0.3
      --batch_size "${BATCH}"
      --learning_rate "${LR}"
      --lradj sigmoid
      --train_epochs 100
      --patience 10
      --model PatchLinear
      --des Exp
    )

    if [ "$MULTI_SEED" -eq 1 ]; then
      for SEED in "${SEEDS[@]}"; do
        run_one "${NAME}_pl${PRED_LEN}_uni_s${SEED}" \
          --model_id "${NAME}_${PRED_LEN}_uni_s${SEED}" \
          --seed "${SEED}" "${COMMON[@]}"
      done
    else
      run_one "${NAME}_pl${PRED_LEN}_uni" \
        --model_id "${NAME}_${PRED_LEN}_uni" \
        --seed 2021 "${COMMON[@]}"
    fi

  done
  echo "${NAME} univariate done"
done

# ILI univariate
for PRED_LEN in 24 36 48 60; do
  run_one "ILI_pl${PRED_LEN}_uni" \
    --is_training 1 --root_path ./dataset/ \
    --data_path national_illness.csv --data custom \
    --features S --seq_len 36 --label_len 18 \
    --pred_len "${PRED_LEN}" --enc_in 1 \
    --d_model 64 --t_ff 128 --c_ff 16 \
    --patch_len 6 --stride 3 --dw_kernel 3 --alpha 0.3 \
    --use_decomp 0 --use_seas_stream 0 --use_fusion_gate 0 \
    --batch_size 32 --learning_rate 0.01 --lradj type3 \
    --train_epochs 100 --patience 10 \
    --model PatchLinear --seed 2021 --des Exp \
    --model_id "ILI_${PRED_LEN}_uni"
done

echo ""
echo "Univariate complete. Results in ${SCRIPT_DIR}/result.txt"

if [ -d "/content/drive/MyDrive" ]; then
  STAMP=$(date +%Y%m%d_%H%M%S)
  cp "${SCRIPT_DIR}/result.txt" \
     "/content/drive/MyDrive/PatchLinear_univariate_${STAMP}.txt"
  echo "Backed up to Drive"
fi
