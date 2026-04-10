#!/usr/bin/env bash
# Solar search — Section 6 only: d_model × patch combos
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/logs/solar_search"
mkdir -p "$LOG_DIR"

SEED=2021

run_one() {
  local TAG=$1; local PL=$2; shift 2
  local LOG="${LOG_DIR}/${TAG}_pl${PL}.log"
  echo ">>> Solar pl${PL} ${TAG}"
  python -u run.py "$@" --pred_len ${PL} \
    --model_id "Solar_${TAG}_pl${PL}" > "$LOG" 2>&1
  grep "mse:" "$LOG" | tail -1 || echo "  [FAILED — check ${LOG}]"
}

BASE="--is_training 1 --root_path ./dataset/
  --data_path solar.txt --data Solar
  --features M --seq_len 96 --label_len 48 --enc_in 137
  --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1
  --use_cross_channel 1 --use_alpha_gate 1
  --alpha 0.3 --batch_size 512 --learning_rate 0.0008
  --lradj sigmoid --train_epochs 100 --patience 10
  --model PatchLinear --seed ${SEED} --des Exp"

# d128 + p8s4 + dk7
run_one "d128_c34_p8s4_dk7"   96  $BASE --d_model 128 --t_ff 256 --c_ff 34 --patch_len 8  --stride 4  --dw_kernel 7
run_one "d128_c34_p8s4_dk7"   720 $BASE --d_model 128 --t_ff 256 --c_ff 34 --patch_len 8  --stride 4  --dw_kernel 7

# d256 + p8s4 + dk7
run_one "d256_c34_p8s4_dk7"   96  $BASE --d_model 256 --t_ff 512 --c_ff 34 --patch_len 8  --stride 4  --dw_kernel 7
run_one "d256_c34_p8s4_dk7"   720 $BASE --d_model 256 --t_ff 512 --c_ff 34 --patch_len 8  --stride 4  --dw_kernel 7

# d128 + p16s8 + dk13
run_one "d128_c34_p16s8_dk13" 96  $BASE --d_model 128 --t_ff 256 --c_ff 34 --patch_len 16 --stride 8  --dw_kernel 13
run_one "d128_c34_p16s8_dk13" 720 $BASE --d_model 128 --t_ff 256 --c_ff 34 --patch_len 16 --stride 8  --dw_kernel 13

# d192 + p8s4 (combine best d_model with finer patch)
run_one "d192_c34_p8s4_dk7"   96  $BASE --d_model 192 --t_ff 384 --c_ff 34 --patch_len 8  --stride 4  --dw_kernel 7
run_one "d192_c34_p8s4_dk7"   720 $BASE --d_model 192 --t_ff 384 --c_ff 34 --patch_len 8  --stride 4  --dw_kernel 7

echo ""
echo "=== Section 6 complete ==="
echo ""
echo "H=96 ranking:"
for f in "$LOG_DIR"/*_pl96.log; do
  MSE=$(grep "mse:" "$f" | tail -1 | grep -oP 'mse:\K[\d.]+' || echo "9.999")
  printf "  %s %s\n" "$MSE" "$(basename $f _pl96.log)"
done | sort -n

echo ""
echo "H=720 ranking:"
for f in "$LOG_DIR"/*_pl720.log; do
  MSE=$(grep "mse:" "$f" | tail -1 | grep -oP 'mse:\K[\d.]+' || echo "9.999")
  printf "  %s %s\n" "$MSE" "$(basename $f _pl720.log)"
done | sort -n

echo ""
echo "Targets: H96 < 0.189 (TimeMixer)  H720 < 0.249 (xPatch ul)"
