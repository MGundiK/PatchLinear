#!/usr/bin/env bash
# =============================================================================
# PatchLinear — Ablation Study
#
# Ablations (one component disabled per run vs full model):
#   A1   : --use_decomp 0
#   A2a  : --use_seas_stream 0      (trend-only)
#   A2b  : --use_trend_stream 0     (seas-only)
#   A3   : --use_fusion_gate 0
#   A3b  : --use_tgm 0              (needs distinct model_id — not in setting string)
#   A4a  : --use_cross_channel 0    (implies --use_alpha_gate 0)
#   A4b  : --use_alpha_gate 0
#   A6s  : --dw_kernel 3            (small ERF)
#   A6l  : --dw_kernel 13           (large ERF)
#
# Usage:
#   bash scripts/run_ablations.sh A1        # single ablation, all datasets
#   bash scripts/run_ablations.sh all       # all ablations (very long)
#
# Seeds: single seed 2021 for ablations (paper practice)
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ABLATION="${1:-all}"
SEED=2021
LOG_ROOT="logs/ablations"
mkdir -p "$LOG_ROOT"

# ── Dataset configurations ────────────────────────────────────────────────────
# Format: NAME|DATA_PATH|DATA_TYPE|ENC_IN|D|TFF|CFF|P|S|K|BS|LR|LRADJ|PRED_LENS|SEQ_LEN|EXTRA_FLAGS
DATASETS=(
  "ETTh1|ETTh1.csv|ETTh1|7|64|128|16|16|8|7|2048|0.0005|sigmoid|96,192,336,720|96|"
  "ETTh2|ETTh2.csv|ETTh2|7|64|128|16|16|8|7|2048|0.0005|sigmoid|96,192,336,720|96|"
  "ETTm1|ETTm1.csv|ETTm1|7|64|128|16|16|8|7|2048|0.0005|sigmoid|96,192,336,720|96|"
  "ETTm2|ETTm2.csv|ETTm2|7|64|128|16|16|8|7|2048|0.0001|sigmoid|96,192,336,720|96|"
  "Weather|weather.csv|custom|21|64|128|16|16|8|7|2048|0.0005|sigmoid|96,192,336,720|96|"
  "Traffic|traffic.csv|custom|862|256|512|128|16|8|7|64|0.005|sigmoid|96,192,336,720|96|"
  "Electricity|electricity.csv|custom|321|256|512|128|16|8|7|256|0.01|sigmoid|96,192,336,720|96|"
  "Exchange|exchange_rate.csv|custom|8|64|128|16|8|4|3|32|0.000002|sigmoid|96,192,336,720|96|"
  "Solar|solar.txt|Solar|137|192|384|34|16|8|7|512|0.0008|sigmoid|96,192,336,720|96|"
  # ILI: full model = trend-only (decomp=0, seas=0, fg=0 are already off)
  # Only A3b/A4a/A4b/A6 ablations make sense for ILI
  "ILI|national_illness.csv|custom|7|64|128|8|6|3|3|32|0.01|type3|24,36,48,60|36|--use_decomp 0 --use_seas_stream 0 --use_fusion_gate 0"
)

run_one() {
  local LABEL=$1; shift
  local LOG="$LOG_ROOT/${LABEL}.log"
  python -u run.py "$@" > "$LOG" 2>&1
  grep "mse:" "$LOG" | tail -1 || true
  echo "  → $LABEL"
}

run_ablation() {
  local ABL=$1
  local EXTRA_FLAGS=$2
  local MODEL_ID_PREFIX=$3   # empty for most; "A3b" for TGM ablation

  echo ""
  echo "================================================================"
  echo "ABLATION: ${ABL}"
  echo "================================================================"
  mkdir -p "$LOG_ROOT/$ABL"

  for ENTRY in "${DATASETS[@]}"; do
    IFS='|' read -r NAME DPATH DTYPE ENC D TFF CFF P S K BS LR LRADJ PREDS SEQ DATASET_FLAGS <<< "$ENTRY"

    # A3b only makes sense with cross-channel active; ILI ablations are limited
    # Skip A2a/A2b/A3 for ILI (already in those states)
    if [[ "$NAME" == "ILI" ]]; then
      case "$ABL" in
        A1|A2a|A2b|A3) echo "  [skip] $NAME — component already off in ILI full model"; continue ;;
      esac
    fi

    IFS=',' read -ra PRED_LENS <<< "$PREDS"
    for PL in "${PRED_LENS[@]}"; do
      # Construct model_id — for A3b use prefix to avoid checkpoint collision
      if [[ -n "$MODEL_ID_PREFIX" ]]; then
        MID="${MODEL_ID_PREFIX}_${NAME}_${PL}_s${SEED}"
      else
        MID="${ABL}_${NAME}_${PL}_s${SEED}"
      fi

      # shellcheck disable=SC2086
      run_one "$ABL/${NAME}_pl${PL}" \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path "$DPATH" \
        --data "$DTYPE" \
        --features M \
        --seq_len "$SEQ" \
        --label_len $((SEQ / 2)) \
        --pred_len "$PL" \
        --enc_in "$ENC" \
        --d_model "$D" \
        --t_ff "$TFF" \
        --c_ff "$CFF" \
        --patch_len "$P" \
        --stride "$S" \
        --dw_kernel "$K" \
        --alpha 0.3 \
        --batch_size "$BS" \
        --learning_rate "$LR" \
        --lradj "$LRADJ" \
        --train_epochs 100 \
        --patience 10 \
        --model PatchLinear \
        --seed "$SEED" \
        --des Exp \
        --model_id "$MID" \
        $DATASET_FLAGS \
        $EXTRA_FLAGS
    done
  done

  # Per-dataset summary
  echo ""
  echo "--- $ABL summary ---"
  for ENTRY in "${DATASETS[@]}"; do
    IFS='|' read -r NAME DPATH DTYPE ENC D TFF CFF P S K BS LR LRADJ PREDS SEQ DATASET_FLAGS <<< "$ENTRY"
    IFS=',' read -ra PRED_LENS <<< "$PREDS"
    MSE_SUM=0; N=0
    for PL in "${PRED_LENS[@]}"; do
      LOG="$LOG_ROOT/$ABL/${NAME}_pl${PL}.log"
      if [[ -f "$LOG" ]]; then
        MSE=$(grep "mse:" "$LOG" | tail -1 | grep -oP 'mse:\s*\K[\d.]+' || echo "0")
        MSE_SUM=$(python3 -c "print($MSE_SUM + $MSE)")
        N=$((N+1))
      fi
    done
    if [[ $N -gt 0 ]]; then
      AVG=$(python3 -c "print(f'{$MSE_SUM/$N:.4f}')")
      printf "  %-12s avg MSE: %s  (%d pred_lens)\n" "$NAME" "$AVG" "$N"
    fi
  done
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "$ABLATION" in
  A1)   run_ablation "A1"   "--use_decomp 0"                         "" ;;
  A2a)  run_ablation "A2a"  "--use_seas_stream 0"                    "" ;;
  A2b)  run_ablation "A2b"  "--use_trend_stream 0"                   "" ;;
  A3)   run_ablation "A3"   "--use_fusion_gate 0"                    "" ;;
  A3b)  run_ablation "A3b"  "--use_tgm 0"                           "A3b" ;;
  A4a)  run_ablation "A4a"  "--use_cross_channel 0 --use_alpha_gate 0" "" ;;
  A4b)  run_ablation "A4b"  "--use_alpha_gate 0"                    "" ;;
  A6s)  run_ablation "A6s"  "--dw_kernel 3"                         "" ;;
  A6l)  run_ablation "A6l"  "--dw_kernel 13"                        "" ;;
  all)
    run_ablation "A1"   "--use_decomp 0"                          ""
    run_ablation "A2a"  "--use_seas_stream 0"                     ""
    run_ablation "A2b"  "--use_trend_stream 0"                    ""
    run_ablation "A3"   "--use_fusion_gate 0"                     ""
    run_ablation "A3b"  "--use_tgm 0"                            "A3b"
    run_ablation "A4a"  "--use_cross_channel 0 --use_alpha_gate 0" ""
    run_ablation "A4b"  "--use_alpha_gate 0"                     ""
    run_ablation "A6s"  "--dw_kernel 3"                          ""
    run_ablation "A6l"  "--dw_kernel 13"                         ""
    ;;
  *)
    echo "Unknown ablation: $ABLATION"
    echo "Valid: A1 A2a A2b A3 A3b A4a A4b A6s A6l all"
    exit 1 ;;
esac

echo ""
echo "Done: $ABLATION"
