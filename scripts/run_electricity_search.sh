#!/usr/bin/env bash
# =============================================================================
# Electricity hyperparameter search — Phase 1
# Single seed (2021), H=192 and H=720 only
#
# Current best: H=192 BOLD (0.156), H=720 ul (0.211)
# Targets: keep H=192 bold, improve H=720 (CARD=0.197 is best)
#
# All params are active (full arch: decomp=1, seas=1, fusion=1)
# Already tuned: lr=0.01, c_ff=128, patch=16/s=8, dk=7, batch=256
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/logs/elec_search"
mkdir -p "$LOG_DIR"
RESULT="$SCRIPT_DIR/result_elec_search.txt"
> "$RESULT"

SEED=2021

BASE="--is_training 1 --root_path ./dataset/
  --data_path electricity.csv --data custom
  --features M --seq_len 96 --label_len 48 --enc_in 321
  --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1
  --use_cross_channel 1 --use_alpha_gate 1
  --lradj sigmoid --train_epochs 100 --patience 10
  --model PatchLinear --seed ${SEED} --des Exp"

run2() {
  local TAG=$1; shift
  for PL in 192 720; do
    local LOG="${LOG_DIR}/${TAG}_pl${PL}.log"
    python -u run.py $BASE --pred_len ${PL} \
      --model_id "Elec_${TAG}_pl${PL}" "$@" > "$LOG" 2>&1
    MSE=$(grep "mse:" "$LOG" | tail -1 | grep -oP 'mse:\K[\d.]+' || echo "FAILED")
    MAE=$(grep "mse:" "$LOG" | tail -1 | grep -oP 'mae:\K[\d.]+' || echo "FAILED")
    echo ">>> Elec pl${PL} ${TAG}: mse=${MSE} mae=${MAE}"
    echo "${PL} ${MSE} ${MAE} ${TAG}" >> "$RESULT"
  done
}

DFLT="--d_model 64 --t_ff 128 --c_ff 128
  --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3
  --batch_size 256 --learning_rate 0.01"

# =============================================================================
echo "=== SECTION 1: d_model sweep (t_ff=2d, others default) ==="
for D in 32 48 96 128 160; do
  T=$((D*2))
  run2 "d${D}_t${T}" --d_model ${D} --t_ff ${T} --c_ff 128 \
    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
    --batch_size 256 --learning_rate 0.01
done

# =============================================================================
echo "=== SECTION 2: t_ff sweep (d=64) ==="
for T in 64 96 192 256 320; do
  run2 "d64_t${T}" --d_model 64 --t_ff ${T} --c_ff 128 \
    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
    --batch_size 256 --learning_rate 0.01
done

# =============================================================================
echo "=== SECTION 3: patch/stride ==="
for PS in "8 4" "24 12" "32 16"; do
  P=$(echo $PS | cut -d' ' -f1); S=$(echo $PS | cut -d' ' -f2)
  run2 "p${P}s${S}" $DFLT --patch_len ${P} --stride ${S}
done

# =============================================================================
echo "=== SECTION 4: dw_kernel ==="
for DK in 3 13; do
  run2 "dk${DK}" $DFLT --dw_kernel ${DK}
done

# =============================================================================
echo "=== SECTION 5: c_ff variants ==="
for C in 64 96 160 192 256; do
  run2 "c${C}" --d_model 64 --t_ff 128 --c_ff ${C} \
    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
    --batch_size 256 --learning_rate 0.01
done

# =============================================================================
echo "=== SECTION 6: cc/ag flag combos ==="
for CC in 0 1; do for AG in 0 1; do
  [[ $CC -eq 1 && $AG -eq 1 ]] && continue
  run2 "cc${CC}_ag${AG}" $DFLT \
    --use_cross_channel ${CC} --use_alpha_gate ${AG}
done; done

# =============================================================================
echo "=== SECTION 7: Architecture flag combos ==="
for DEC in 0 1; do for SEAS in 0 1; do for FUS in 0 1; do
  [[ $DEC -eq 1 && $SEAS -eq 1 && $FUS -eq 1 ]] && continue
  run2 "dec${DEC}_s${SEAS}_f${FUS}" \
    --d_model 64 --t_ff 128 --c_ff 128 \
    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
    --batch_size 256 --learning_rate 0.01 \
    --use_decomp ${DEC} --use_seas_stream ${SEAS} --use_fusion_gate ${FUS} \
    --use_cross_channel 1 --use_alpha_gate 1
done; done; done

# =============================================================================
echo "=== SECTION 8: Promising d_model × patch combos ==="
for D_P in "96 8 4" "96 24 12" "128 16 8" "64 8 4 3"; do
  D=$(echo $D_P | cut -d' ' -f1)
  P=$(echo $D_P | cut -d' ' -f2)
  S=$(echo $D_P | cut -d' ' -f3)
  DK=$(echo $D_P | cut -d' ' -f4); DK=${DK:-7}
  T=$((D*2))
  run2 "d${D}_p${P}s${S}_dk${DK}" \
    --d_model ${D} --t_ff ${T} --c_ff 128 \
    --patch_len ${P} --stride ${S} --dw_kernel ${DK} --alpha 0.3 \
    --batch_size 256 --learning_rate 0.01
done

# =============================================================================
echo ""
echo "=== TOP 10 by H=192 MSE ==="
grep "^192 " "$RESULT" | sort -k2 -n | head -10
echo ""
echo "=== TOP 10 by H=720 MSE ==="
grep "^720 " "$RESULT" | sort -k2 -n | head -10

echo ""
echo "Targets: H=192 bold=0.156 (keep), H=720 ul=0.211 (improve toward CARD=0.197)"
