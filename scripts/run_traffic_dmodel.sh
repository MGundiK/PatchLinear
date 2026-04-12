#!/usr/bin/env bash
# =============================================================================
# Traffic d_model sweep — single seed s2021, H=96 and H=720
#
# Current: 0 bold, 0 ul. Gap to 2nd place: 0.075 (H=96), 0.087 (H=720)
# Target: get into underline range (CARD H=96=0.419, iTransf H=720=0.490)
#
# Key lesson from Electricity/Solar: d_model scaling is the biggest lever
# Traffic: enc_in=862, batch=96 (memory constrained)
# Note: large d_model may require smaller batch — try batch=32 for d>=192
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/logs/traffic_dmodel"
mkdir -p "$LOG_DIR"

SEED=2021

run_one() {
  local TAG=$1; local PL=$2; shift 2
  local LOG="${LOG_DIR}/${TAG}_pl${PL}.log"
  echo ">>> Traffic pl${PL} ${TAG}"
  python -u run.py "$@" --pred_len ${PL} --seed ${SEED} \
    --model_id "Traffic_${TAG}_pl${PL}" > "$LOG" 2>&1
  grep "mse:" "$LOG" | tail -1 || echo "  [FAILED — check ${LOG}]"
}

BASE="--is_training 1 --root_path ./dataset/
  --data_path traffic.csv --data custom
  --features M --seq_len 96 --label_len 48 --enc_in 862
  --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1
  --use_cross_channel 1 --use_alpha_gate 1
  --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3
  --learning_rate 0.005
  --lradj sigmoid --train_epochs 100 --patience 10
  --model PatchLinear --des Exp"

echo "=== d_model sweep (t_ff=2d, c_ff=128, batch=96) ==="
for D in 96 128 160 192; do
  T=$((D*2))
  for PL in 96 720; do
    run_one "d${D}_t${T}_b96" ${PL} $BASE \
      --d_model ${D} --t_ff ${T} --c_ff 128 --batch_size 96
  done
done

echo "=== d_model=256 with smaller batches (memory) ==="
for BATCH in 64 32; do
  for PL in 96 720; do
    run_one "d256_t512_b${BATCH}" ${PL} $BASE \
      --d_model 256 --t_ff 512 --c_ff 128 --batch_size ${BATCH}
  done
done

echo "=== c_ff scaling with large d_model (d=128) ==="
for C in 192 256; do
  for PL in 96 720; do
    run_one "d128_c${C}_b96" ${PL} $BASE \
      --d_model 128 --t_ff 256 --c_ff ${C} --batch_size 96
  done
done

echo ""
echo "=== Results summary ==="
echo "H=96 ranking (target 2nd place: CARD 0.419):"
for f in "$LOG_DIR"/*_pl96.log; do
  MSE=$(grep "mse:" "$f" | tail -1 | grep -oP 'mse:\K[\d.]+' || echo "9.999")
  printf "  %s %s\n" "$MSE" "$(basename $f _pl96.log)"
done | sort -n

echo ""
echo "H=720 ranking (target 2nd place: iTransf 0.490):"
for f in "$LOG_DIR"/*_pl720.log; do
  MSE=$(grep "mse:" "$f" | tail -1 | grep -oP 'mse:\K[\d.]+' || echo "9.999")
  printf "  %s %s\n" "$MSE" "$(basename $f _pl720.log)"
done | sort -n

echo ""
echo "Baseline: H=96=0.494 H=720=0.577"
echo "Targets:  H=96<0.419 (CARD 2nd) H=720<0.490 (iTransf 2nd)"
