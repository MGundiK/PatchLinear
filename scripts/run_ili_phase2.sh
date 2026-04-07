#!/usr/bin/env bash
# ILI Phase 2 — top 5 configs, all 4 horizons, 3 seeds each
# Run after run_ili_search.sh to validate top configs
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/logs/ili_phase2"
mkdir -p "$LOG_DIR"

run_one() {
  local TAG=$1; shift
  local LOG="${LOG_DIR}/${TAG}.log"
  echo ">>> ${TAG}"
  python -u run.py "$@" > "$LOG" 2>&1
  grep "mse:" "$LOG" | tail -1 || true
}

BASE="--is_training 1 --root_path ./dataset/
  --data_path national_illness.csv --data custom
  --features M --seq_len 36 --label_len 18 --enc_in 7
  --patch_len 6 --stride 3 --dw_kernel 3 --alpha 0.3
  --use_decomp 0 --use_seas_stream 0 --use_fusion_gate 0
  --use_cross_channel 1 --use_alpha_gate 1 --learning_rate 0.01
  --batch_size 32 --lradj type3 --train_epochs 100 --patience 10
  --model PatchLinear --des Exp"

for SEED in 2021 2022 2023; do
  for PL in 24 36 48 60; do

    # 1. c_ff=8 (best by mean)
    run_one "ILI_pl${PL}_c8_s${SEED}" $BASE \
      --d_model 64 --t_ff 128 --c_ff 8 \
      --pred_len ${PL} --seed ${SEED} --model_id "ILI_${PL}_c8_s${SEED}"

    # 2. d_model=96, t_ff=192 (best stability)
    run_one "ILI_pl${PL}_d96t192_s${SEED}" $BASE \
      --d_model 96 --t_ff 192 --c_ff 16 \
      --pred_len ${PL} --seed ${SEED} --model_id "ILI_${PL}_d96t192_s${SEED}"

    # 3. c_ff=1 (minimal cross-channel)
    run_one "ILI_pl${PL}_c1_s${SEED}" $BASE \
      --d_model 64 --t_ff 128 --c_ff 1 \
      --pred_len ${PL} --seed ${SEED} --model_id "ILI_${PL}_c1_s${SEED}"

    # 4. d_model=96, t_ff=96 (best upside)
    run_one "ILI_pl${PL}_d96t96_s${SEED}" $BASE \
      --d_model 96 --t_ff 96 --c_ff 16 \
      --pred_len ${PL} --seed ${SEED} --model_id "ILI_${PL}_d96t96_s${SEED}"

    # 5. c_ff=28 (4×enc_in)
    run_one "ILI_pl${PL}_c28_s${SEED}" $BASE \
      --d_model 64 --t_ff 128 --c_ff 28 \
      --pred_len ${PL} --seed ${SEED} --model_id "ILI_${PL}_c28_s${SEED}"

  done
  echo "Seed ${SEED} done"
done

echo "Phase 2 complete — 5 configs × 4 horizons × 3 seeds = 60 runs"
