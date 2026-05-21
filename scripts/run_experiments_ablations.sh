#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Unified experiment runner for ablation study.
#
# Usage:
#   bash run_experiments.sh                      # full model only (all datasets)
#   bash run_experiments.sh --ablations          # full model + all ablations
#   bash run_experiments.sh --ablations-only     # ablations only  (skip full runs)
#   bash run_experiments.sh --seeds              # full model, multiple seeds
#   bash run_experiments.sh --from=ILI           # resume: skip datasets before ILI
#
# Flags can be combined, e.g.:
#   bash run_experiments.sh --from=ILI           # finish ILI full run
#   bash run_experiments.sh --ablations-only     # then run all ablations
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── parse script-level flags ─────────────────────────────────────────────────
RUN_ABLATIONS=0
SKIP_FULL=0
MULTI_SEED=0
FROM_DATASET=""
for arg in "$@"; do
  case $arg in
    --ablations)        RUN_ABLATIONS=1 ;;
    --ablations-only)   RUN_ABLATIONS=1; SKIP_FULL=1 ;;
    --seeds)            MULTI_SEED=1 ;;
    --from=*)           FROM_DATASET="${arg#--from=}" ;;
  esac
done

# ── global defaults ───────────────────────────────────────────────────────────
ALPHA=0.3
TRAIN_EPOCHS=100
PATIENCE=10
# lradj is now per-dataset (see DATASETS array; ILI uses type3, all others sigmoid)

LOG_DIR=./logs
mkdir -p "$LOG_DIR"

# ── run_single(tag, args...) ──────────────────────────────────────────────────
run_single() {
  local tag="$1"; shift
  local log_file="${LOG_DIR}/${tag}.log"
  echo ">>> $tag"
  python -u run.py "$@" > "$log_file" 2>&1
  tail -3 "$log_file" | grep -E "mse:|Epoch" || true
}

# ─────────────────────────────────────────────────────────────────────────────
# DATASETS
# Columns: name  data_path  data_type  enc_in  batch  lr  pred_lens
#          seq_len  d_model  t_ff  patch_len  stride  dw_kernel  c_ff
#
# Per-dataset hyperparameters (Table 4). Bold = tuned from default.
# Default: d=64, tff=128, p=16, s=8, k=7, cff=16, lr=5e-4.
# ─────────────────────────────────────────────────────────────────────────────
declare -a DATASETS=(
  # name        data_path             data_type  enc_in  batch   lr          pred_lens        seq_len  d    tff  p   s  k  cff
  "ETTh1        ETTh1.csv             ETTh1      7       2048    0.0005      96,192,336,720   96       64   128  16  8  7  16  sigmoid"
  "ETTh2        ETTh2.csv             ETTh2      7       2048    0.0005      96,192,336,720   96       64   128  16  8  7  16  sigmoid"
  "ETTm1        ETTm1.csv             ETTm1      7       2048    0.0005      96,192,336,720   96       64   128  16  8  7  16  sigmoid"
  "ETTm2        ETTm2.csv             ETTm2      7       2048    0.0001      96,192,336,720   96       64   128  16  8  7  16  sigmoid"
  "Weather      weather.csv           custom     21      2048    0.0005      96,192,336,720   96       64   128  16  8  7  16  sigmoid"
  "Traffic      traffic.csv           custom     862     64      0.005       96,192,336,720   96       256  512  16  8  7  128  sigmoid"
  "Electricity  electricity.csv       custom     321     256     0.01        96,192,336,720   96       256  512  16  8  7  128  sigmoid"
  "Exchange     exchange_rate.csv     custom     8       32      0.000002    96,192,336,720   96       64   128  8   4  3  16   sigmoid"
  "Solar        solar.txt             Solar      137     512     0.0008      96,192,336,720   96       192  384  16  8  7  34  sigmoid"
  "ILI          national_illness.csv  custom     7       32      0.01        24,36,48,60      36       64   128  6   3  3  8   type3"
)

# ─────────────────────────────────────────────────────────────────────────────
# ABLATION CONFIGURATIONS
# Each entry: "name  flag(s)"  — changes exactly one design decision.
# Logical order follows the forward pass:
#   A1  decomposition → A2 streams → A3 fusion gate → A3b TGM
#   → A4a cross-channel VGM → A4b alpha gate → A6 kernel size
# ─────────────────────────────────────────────────────────────────────────────
declare -a ABLATION_CONFIGS=(
  # A1: EMA decomposition
  "A1_no_decomp         --use_decomp 0"

  # A2: individual stream ablations
  "A2a_trend_only       --use_seas_stream 0 --use_fusion_gate 0"
  "A2b_seasonal_only    --use_trend_stream 0 --use_fusion_gate 0"

  # A3: input-dependent fusion gate
  "A3_no_fusion_gate    --use_fusion_gate 0"

  # A3b: temporal global module (TGM)
  # Sits between the fusion gate (A3) and the cross-channel VGM (A4a).
  # Without TGM, glob_updated falls back to the raw learned g_0 — a static
  # parameter shared across all batch instances.  The VGM still runs but is
  # driven by a fixed token rather than input-conditioned information.
  # Expected to hurt most on Traffic / Electricity (dynamic cross-channel
  # patterns) and least on Exchange (alpha already suppresses cross-channel).
  "A3b_no_tgm           --use_tgm 0"

  # A4a: cross-channel VGM
  "A4a_no_cross_ch      --use_cross_channel 0"

  # A4b: per-channel alpha mixing gate (keep cross-channel, disable alpha)
  "A4b_no_alpha         --use_alpha_gate 0"

  # A6: DWConv kernel size (compare against per-dataset default k)
  "A6_dw_k3             --dw_kernel 3"
  "A6_dw_k13            --dw_kernel 13"
)

# ─────────────────────────────────────────────────────────────────────────────
# Seeds for multi-seed runs
# ─────────────────────────────────────────────────────────────────────────────
SEEDS=(2021 2022 2023)

# ─────────────────────────────────────────────────────────────────────────────
# Resume support: --from=NAME skips every dataset before NAME.
# ─────────────────────────────────────────────────────────────────────────────
PAST_RESUME=0
[ -z "$FROM_DATASET" ] && PAST_RESUME=1

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
for dataset_entry in "${DATASETS[@]}"; do
  read -r NAME DATA_PATH DATA_TYPE ENC_IN BATCH_SIZE LR PRED_LENS_CSV \
       SEQ_LEN D_MODEL T_FF PATCH_LEN STRIDE DW_KERNEL C_FF LRADJ
       <<< "$dataset_entry"

  # ── resume-from guard ───────────────────────────────────────────────────────
  if [ "$PAST_RESUME" -eq 0 ]; then
    [ "$NAME" = "$FROM_DATASET" ] && PAST_RESUME=1 || continue
  fi

  IFS=',' read -ra PRED_LENS <<< "$PRED_LENS_CSV"

  # label_len must not exceed seq_len; cap at min(48, seq_len // 2)
  LABEL_LEN=$(( SEQ_LEN / 2 ))
  [ "$LABEL_LEN" -gt 48 ] && LABEL_LEN=48

  for PRED_LEN in "${PRED_LENS[@]}"; do

    COMMON=(
      --is_training 1
      --root_path ./dataset/
      --data_path "${DATA_PATH}"
      --data "${DATA_TYPE}"
      --features M
      --seq_len "${SEQ_LEN}"
      --pred_len "${PRED_LEN}"
      --label_len "${LABEL_LEN}"
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
    if [ "$SKIP_FULL" -eq 0 ]; then
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
    fi

    # ── Ablations ─────────────────────────────────────────────────────────────
    if [ "$RUN_ABLATIONS" -eq 1 ]; then
      for ablation_entry in "${ABLATION_CONFIGS[@]}"; do
        ABLATION_NAME="${ablation_entry%%  *}"
        ABLATION_ARGS="${ablation_entry#*  }"

        # A6 kernel ablations: skip if this dataset's default k already
        # matches the ablation value — the comparison would be vacuous.
        if [[ "$ABLATION_NAME" == "A6_dw_k3"  && "$DW_KERNEL" == "3"  ]]; then continue; fi
        if [[ "$ABLATION_NAME" == "A6_dw_k13" && "$DW_KERNEL" == "13" ]]; then continue; fi

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
