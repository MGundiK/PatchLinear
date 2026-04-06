#!/usr/bin/env bash
# =============================================================================
# ILI comprehensive hyperparameter search — Phase 1
# All configs run with 3 seeds on H=36 (hardest horizon, biggest gap)
# Phase 2 (manual): top 10 configs → all 4 horizons × 3 seeds
#
# Usage: bash scripts/run_ili_search.sh
# Results: logs/ili_search/ and result_ili_search.txt
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs/ili_search"
mkdir -p "$LOG_DIR"
RESULT="$SCRIPT_DIR/result_ili_search.txt"
> "$RESULT"

PL=36

# Base ILI args shared across all runs
BASE_ARGS="--is_training 1 --root_path ./dataset/
  --data_path national_illness.csv --data custom
  --features M --seq_len 36 --label_len 18
  --pred_len ${PL} --enc_in 7
  --batch_size 32 --lradj type3
  --train_epochs 100 --patience 10
  --model PatchLinear --des Exp"

run3() {
  # run3 TAG [extra args]
  local TAG=$1; shift
  local MSE_SUM=0; local MAE_SUM=0; local N=0
  for SEED in 2021 2022 2023; do
    local LOG="${LOG_DIR}/${TAG}_s${SEED}.log"
    python -u run.py $BASE_ARGS --seed ${SEED} \
      --model_id "ILI_${TAG}_s${SEED}" "$@" > "$LOG" 2>&1
    local LINE=$(grep "mse:" "$LOG" | tail -1)
    local MSE=$(echo "$LINE" | grep -oP 'mse:\K[\d.]+' || echo "9.999")
    local MAE=$(echo "$LINE" | grep -oP 'mae:\K[\d.]+' || echo "9.999")
    MSE_SUM=$(python3 -c "print($MSE_SUM+$MSE)")
    MAE_SUM=$(python3 -c "print($MAE_SUM+$MAE)")
    N=$((N+1))
  done
  local MEAN_MSE=$(python3 -c "print(round($MSE_SUM/$N,4))")
  local MEAN_MAE=$(python3 -c "print(round($MAE_SUM/$N,4))")
  echo "${MEAN_MSE} ${MEAN_MAE} ${TAG}"
  echo "${MEAN_MSE} ${MEAN_MAE} ${TAG}" >> "$RESULT"
}

# =============================================================================
# SECTION 1: Base param sweep
# (use_decomp=0, use_seas_stream=0, use_fusion_gate=0 — CNN irrelevant)
# =============================================================================
echo "=== SECTION 1: d_model sweep ===" | tee -a "$RESULT"
FIXED="--patch_len 6 --stride 3 --dw_kernel 3 --alpha 0.3
  --use_decomp 0 --use_seas_stream 0 --use_fusion_gate 0
  --use_cross_channel 1 --use_alpha_gate 1 --learning_rate 0.01"
for D in 16 32 48 64 96 128 192; do
  T=$((D*2)); run3 "d${D}_t${T}_c16_cc1_ag1_lr001" $FIXED \
    --d_model ${D} --t_ff ${T} --c_ff 16
done

echo "=== SECTION 2: t_ff sweep (d=64) ===" | tee -a "$RESULT"
for T in 32 48 64 96 128 192 256 320; do
  [ $T -eq 128 ] && continue  # already in d_model sweep
  run3 "d64_t${T}_c16_cc1_ag1_lr001" $FIXED \
    --d_model 64 --t_ff ${T} --c_ff 16
done

echo "=== SECTION 3: c_ff sweep (d=64, t=128) ===" | tee -a "$RESULT"
for C in 1 2 4 7 8 14 28 32; do
  [ $C -eq 16 ] && continue
  run3 "d64_t128_c${C}_cc1_ag1_lr001" $FIXED \
    --d_model 64 --t_ff 128 --c_ff ${C}
done

echo "=== SECTION 4: cc/ag flag combos (d=64) ===" | tee -a "$RESULT"
for CC in 0 1; do for AG in 0 1; do
  [ $CC -eq 1 ] && [ $AG -eq 1 ] && continue  # baseline
  run3 "d64_t128_c16_cc${CC}_ag${AG}_lr001" $FIXED \
    --d_model 64 --t_ff 128 --c_ff 16 \
    --use_cross_channel ${CC} --use_alpha_gate ${AG}
done; done

echo "=== SECTION 5: lr sweep (d=64, t=128, c=16) ===" | tee -a "$RESULT"
for LR in 0.001 0.003 0.005 0.008 0.015 0.02 0.03 0.05; do
  run3 "d64_t128_c16_cc1_ag1_lr${LR}" $FIXED \
    --d_model 64 --t_ff 128 --c_ff 16 --learning_rate ${LR}
done

echo "=== SECTION 6: d_model × t_ff × lr combos ===" | tee -a "$RESULT"
for D in 32 96; do
  for T_RATIO in 1 2 4; do
    T=$((D*T_RATIO))
    for LR in 0.005 0.01 0.02; do
      run3 "d${D}_t${T}_c16_cc1_ag1_lr${LR}" $FIXED \
        --d_model ${D} --t_ff ${T} --c_ff 16 --learning_rate ${LR}
    done
  done
done

# =============================================================================
# SECTION 7: Architecture flag combinations
# Try all 7 non-baseline combos of (decomp, seas, fusion)
# For configs with seas=1, patch params matter — try 2 patch configs
# =============================================================================
echo "=== SECTION 7: Architecture combos ===" | tee -a "$RESULT"
BASE_ARCH="--d_model 64 --t_ff 128 --c_ff 16
  --use_cross_channel 1 --use_alpha_gate 1 --learning_rate 0.01"

for DEC in 0 1; do
  for SEAS in 0 1; do
    for FUS in 0 1; do
      [ $DEC -eq 0 ] && [ $SEAS -eq 0 ] && [ $FUS -eq 0 ] && continue  # baseline
      if [ $SEAS -eq 0 ]; then
        # patch params irrelevant
        run3 "arch_dec${DEC}_s${SEAS}_f${FUS}_p6" \
          $BASE_ARCH \
          --patch_len 6 --stride 3 --dw_kernel 3 --alpha 0.3 \
          --use_decomp ${DEC} --use_seas_stream ${SEAS} --use_fusion_gate ${FUS}
      else
        # patch params matter — try 2 configs
        for PC in "6 3 3" "4 2 3" "9 3 3"; do
          PL_=$( echo $PC | cut -d' ' -f1)
          ST_=$(echo $PC | cut -d' ' -f2)
          DW_=$(echo $PC | cut -d' ' -f3)
          run3 "arch_dec${DEC}_s${SEAS}_f${FUS}_p${PL_}s${ST_}dk${DW_}" \
            $BASE_ARCH \
            --patch_len ${PL_} --stride ${ST_} --dw_kernel ${DW_} --alpha 0.3 \
            --use_decomp ${DEC} --use_seas_stream ${SEAS} --use_fusion_gate ${FUS}
        done
      fi
    done
  done
done

# =============================================================================
# SECTION 8: Best arch combos × d_model × lr
# Run the most promising arch combos with varying d_model and lr
# =============================================================================
echo "=== SECTION 8: Promising arch × d_model × lr ===" | tee -a "$RESULT"
for DEC_SEAS_FUS in "1 1 0" "1 1 1" "0 1 0"; do
  DEC=$(echo $DEC_SEAS_FUS | cut -d' ' -f1)
  SEAS=$(echo $DEC_SEAS_FUS | cut -d' ' -f2)
  FUS=$(echo $DEC_SEAS_FUS | cut -d' ' -f3)
  for D in 32 64 96; do
    T=$((D*2))
    for LR in 0.005 0.01 0.02; do
      run3 "arch_dec${DEC}_s${SEAS}_f${FUS}_d${D}_lr${LR}" \
        --d_model ${D} --t_ff ${T} --c_ff 16 \
        --patch_len 6 --stride 3 --dw_kernel 3 --alpha 0.3 \
        --use_decomp ${DEC} --use_seas_stream ${SEAS} --use_fusion_gate ${FUS} \
        --use_cross_channel 1 --use_alpha_gate 1 --learning_rate ${LR}
    done
  done
done

# =============================================================================
# Sort results
# =============================================================================
echo ""
echo "=== TOP 15 CONFIGS BY MSE ===" | tee -a "$RESULT"
sort -n "$RESULT" | grep "^[0-9]" | head -15 | tee -a "$RESULT"
echo ""
echo "Baseline (d64,t128,c16,cc1,ag1,lr0.01,no_seas): ~1.454"
echo "xPatch target: 1.315"
