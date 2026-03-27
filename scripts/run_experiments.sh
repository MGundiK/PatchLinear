#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Unified experiment runner for ablation study.
#
# Usage:
#   bash run_experiments.sh             # full model only (baseline)
#   bash run_experiments.sh --ablations # full model + all ablations
#   bash run_experiments.sh --seeds     # full model, multiple seeds
#
# Structure:
#   - DATASETS array defines all datasets with their per-dataset hyperparams
#   - ABLATIONS dict defines all ablation configurations (one flag change each)
#   - run_single() launches one python process and logs output
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── parse script-level flags ─────────────────────────────────────────────────
RUN_ABLATIONS=0
MULTI_SEED=0
for arg in "$@"; do
  case $arg in
    --ablations) RUN_ABLATIONS=1 ;;
    --seeds)     MULTI_SEED=1 ;;
  esac
done

# ── global defaults ───────────────────────────────────────────────────────────
SEQ_LEN=96
MODEL_NAME=PatchLinear
MA_TYPE=ema
ALPHA=0.3

# d_model and kernel defaults — kept small for lightweight budget
D_MODEL=64
T_FF=128        # 2 * d_model
PATCH_LEN=16
STRIDE=8
DW_KERNEL=7     # A6 default (medium range)

TRAIN_EPOCHS=100
PATIENCE=10
LRADJ=sigmoid

LOG_DIR=./logs
mkdir -p "$LOG_DIR"

# ── helper: c_ff = max(16, min(enc_in // 4, 128)) ────────────────────────────
c_ff_for() {
  local enc_in=$1
  local v=$(( enc_in / 4 ))
  [ $v -lt 16 ]  && v=16
  [ $v -gt 128 ] && v=128
  echo $v
}

# ── run_single(tag, dataset_args..., ablation_args...) ───────────────────────
run_single() {
  local tag="$1"; shift
  local log_file="${LOG_DIR}/${tag}.log"
  echo ">>> $tag"
  python -u run.py "$@" > "$log_file" 2>&1
  # Extract and echo the final mse/mae line for quick monitoring
  tail -3 "$log_file" | grep -E "mse:|Epoch" || true
}

# ─────────────────────────────────────────────────────────────────────────────
# DATASETS  (name, data_path, data_type, enc_in, batch_size, lr, pred_lens)
# Per-dataset hyperparameters taken from xPatch as starting point.
# ─────────────────────────────────────────────────────────────────────────────
declare -a DATASETS=(
  # name       data_path            data_type  enc_in  batch lr       pred_lens
  "ETTh1       ETTh1.csv            ETTh1      7       2048  0.0005   96,192,336,720"
  "ETTh2       ETTh2.csv            ETTh2      7       2048  0.0005   96,192,336,720"
  "ETTm1       ETTm1.csv            ETTm1      7       2048  0.0005   96,192,336,720"
  "ETTm2       ETTm2.csv            ETTm2      7       2048  0.0001   96,192,336,720"
  "Weather     weather.csv          custom     21      2048  0.0005   96,192,336,720"
  "Traffic     traffic.csv          custom     862     96    0.005    96,192,336,720"
  "Electricity electricity.csv      custom     321     256   0.005    96,192,336,720"
  "Exchange    exchange_rate.csv    custom     8       32    0.00001  96,192,336,720"
  "Solar       solar.txt            Solar      137     512   0.005    96,192,336,720"
)

# ─────────────────────────────────────────────────────────────────────────────
# ABLATION CONFIGURATIONS
# Each entry: "name  flag=value"
# The full model has all flags at their default (1 / 7).
# Each ablation changes exactly one flag so the source of any
# performance change is unambiguous.
# ─────────────────────────────────────────────────────────────────────────────
declare -a ABLATION_CONFIGS=(
  # A1: decomposition
  "A1_no_decomp         --use_decomp 0"

  # A2: stream ablations
  "A2a_trend_only       --use_seas_stream 0 --use_fusion_gate 0"
  "A2b_seasonal_only    --use_trend_stream 0 --use_fusion_gate 0"

  # A3: fusion gate
  "A3_no_fusion_gate    --use_fusion_gate 0"

  # A4a: cross-channel
  "A4a_no_cross_ch      --use_cross_channel 0"

  # A4b: alpha gate  (keep cross-channel, disable alpha)
  "A4b_no_alpha         --use_alpha_gate 0"

  # A6: kernel size  (compare against default dw_kernel=7)
  "A6_dw_k3             --dw_kernel 3"
  "A6_dw_k13            --dw_kernel 13"
)

# ─────────────────────────────────────────────────────────────────────────────
# Seeds for multi-seed runs
# Only run after you have confirmed the full model is competitive.
# ─────────────────────────────────────────────────────────────────────────────
SEEDS=(2021 2022 2023)

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
for dataset_entry in "${DATASETS[@]}"; do
  read -r NAME DATA_PATH DATA_TYPE ENC_IN BATCH_SIZE LR PRED_LENS_CSV <<< "$dataset_entry"
  C_FF=$(c_ff_for "$ENC_IN")
  IFS=',' read -ra PRED_LENS <<< "$PRED_LENS_CSV"

  for PRED_LEN in "${PRED_LENS[@]}"; do

    # Common args shared by every run for this dataset+pred_len
    COMMON=(
      --is_training 1
      --root_path ./dataset/
      --data_path "${DATA_PATH}"
      --data "${DATA_TYPE}"
      --features M
      --seq_len "${SEQ_LEN}"
      --pred_len "${PRED_LEN}"
      --label_len 48
      --enc_in "${ENC_IN}"
      --d_model "${D_MODEL}"
      --t_ff "${T_FF}"
      --c_ff "${C_FF}"
      --patch_len "${PATCH_LEN}"
      --stride "${STRIDE}"
      --dw_kernel "${DW_KERNEL}"
      --alpha "${ALPHA}"
      --batch_size "${BATCH_SIZE}"
      --learning_rate "${LR}"
      --lradj "${LRADJ}"
      --train_epochs "${TRAIN_EPOCHS}"
      --patience "${PATIENCE}"
      --des Exp
    )

    # ── Full model (baseline) ─────────────────────────────────────────────────
    if [ "$MULTI_SEED" -eq 1 ]; then
      for SEED in "${SEEDS[@]}"; do
        TAG="${NAME}_pl${PRED_LEN}_full_s${SEED}"
        run_single "$TAG" \
          --model_id "${NAME}_${PRED_LEN}_full_s${SEED}" \
          --seed "${SEED}" \
          "${COMMON[@]}"
      done
    else
      TAG="${NAME}_pl${PRED_LEN}_full"
      run_single "$TAG" \
        --model_id "${NAME}_${PRED_LEN}_full" \
        --seed 2021 \
        "${COMMON[@]}"
    fi

    # ── Ablations ─────────────────────────────────────────────────────────────
    if [ "$RUN_ABLATIONS" -eq 1 ]; then
      for ablation_entry in "${ABLATION_CONFIGS[@]}"; do
        ABLATION_NAME="${ablation_entry%%  *}"
        ABLATION_ARGS="${ablation_entry#*  }"
        TAG="${NAME}_pl${PRED_LEN}_${ABLATION_NAME}"
        # shellcheck disable=SC2086
        run_single "$TAG" \
          --model_id "${NAME}_${PRED_LEN}_${ABLATION_NAME}" \
          --seed 2021 \
          "${COMMON[@]}" \
          $ABLATION_ARGS
      done
    fi

  done  # pred_len
done  # dataset

echo ""
echo "All experiments complete.  Results in result.txt"
