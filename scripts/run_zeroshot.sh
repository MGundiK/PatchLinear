#!/usr/bin/env bash
# =============================================================================
# PatchLinear — Zero-Shot Forecasting
# Protocol: TimeMixer++ Section 4.1.6
#   - Train on dataset Da (full training set)
#   - Evaluate on dataset Db test set without any fine-tuning
#   - All ETT datasets share enc_in=7, same structure → transfer feasible
#   - RevIN normalises per-sample → handles different statistics automatically
#   - Transfers: ETTh1→ETTh2, ETTh1→ETTm2, ETTh2→ETTh1,
#                ETTm1→ETTh2, ETTm1→ETTm2, ETTm2→ETTm1
#   - Averaged over 4 prediction lengths {96, 192, 336, 720}
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/logs/zeroshot"
mkdir -p "$LOG_DIR"
RESULT="$SCRIPT_DIR/result_zeroshot.txt"
> "$RESULT"

SEED=2021

train_one() {
  local SRC=$1; local SRC_PATH=$2; local DATA_TYPE=$3; local PL=$4
  local LOG="${LOG_DIR}/train_${SRC}_pl${PL}.log"
  echo "  Training ${SRC} pl=${PL}"
  python -u run.py \
    --is_training 1 --root_path ./dataset/ --features M \
    --data_path ${SRC_PATH} --data ${DATA_TYPE} \
    --seq_len 96 --label_len 48 --pred_len ${PL} --enc_in 7 \
    --d_model 64 --t_ff 128 --c_ff 16 \
    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
    --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
    --use_cross_channel 1 --use_alpha_gate 1 \
    --batch_size 2048 --learning_rate 0.0005 \
    --lradj sigmoid --train_epochs 100 --patience 10 \
    --model PatchLinear --seed ${SEED} --des Exp \
    --model_id "ZS_${SRC}_pl${PL}" > "$LOG" 2>&1
}

eval_zeroshot() {
  local SRC=$1; local TGT=$2; local TGT_PATH=$3; local TGT_TYPE=$4; local PL=$5
  local LOG="${LOG_DIR}/eval_${SRC}_to_${TGT}_pl${PL}.log"
  echo "  Zero-shot ${SRC}→${TGT} pl=${PL}"
  python -u run.py \
    --is_training 0 --root_path ./dataset/ --features M \
    --data_path ${TGT_PATH} --data ${TGT_TYPE} \
    --seq_len 96 --label_len 48 --pred_len ${PL} --enc_in 7 \
    --d_model 64 --t_ff 128 --c_ff 16 \
    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
    --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
    --use_cross_channel 1 --use_alpha_gate 1 \
    --batch_size 2048 --learning_rate 0.0005 \
    --lradj sigmoid --train_epochs 100 --patience 10 \
    --model PatchLinear --seed ${SEED} --des Exp \
    --model_id "ZS_${SRC}_pl${PL}" > "$LOG" 2>&1
  grep "mse:" "$LOG" | tail -1 | tee -a "$RESULT" || echo "  [FAILED — check ${LOG}]"
}

# Dataset registry
declare -A DS_PATH=( [ETTh1]=ETTh1.csv [ETTh2]=ETTh2.csv [ETTm1]=ETTm1.csv [ETTm2]=ETTm2.csv )
declare -A DS_TYPE=( [ETTh1]=ETTh1 [ETTh2]=ETTh2 [ETTm1]=ETTm1 [ETTm2]=ETTm2 )

# ── Step 1: Train all source datasets ─────────────────────────────────────────
echo "=== STEP 1: Training source models ==="
for SRC in ETTh1 ETTh2 ETTm1 ETTm2; do
  for PL in 96 192 336 720; do
    train_one ${SRC} ${DS_PATH[$SRC]} ${DS_TYPE[$SRC]} ${PL}
  done
  echo "  ${SRC} done"
done

# ── Step 2: Zero-shot evaluation ───────────────────────────────────────────────
echo ""
echo "=== STEP 2: Zero-shot evaluation ==="

TRANSFERS=(
  "ETTh1 ETTh2"
  "ETTh1 ETTm2"
  "ETTh2 ETTh1"
  "ETTm1 ETTh2"
  "ETTm1 ETTm2"
  "ETTm2 ETTm1"
)

for PAIR in "${TRANSFERS[@]}"; do
  SRC=$(echo $PAIR | cut -d' ' -f1)
  TGT=$(echo $PAIR | cut -d' ' -f2)
  echo ">>> ${SRC} → ${TGT}"
  for PL in 96 192 336 720; do
    eval_zeroshot ${SRC} ${TGT} ${DS_PATH[$TGT]} ${DS_TYPE[$TGT]} ${PL}
  done
done

echo ""
echo "=== ZERO-SHOT SUMMARY (avg across 4 horizons) ==="
python3 << 'PYEOF'
import re, os, numpy as np
log_dir = "logs/zeroshot"
transfers = [
    ("ETTh1","ETTh2"),("ETTh1","ETTm2"),("ETTh2","ETTh1"),
    ("ETTm1","ETTh2"),("ETTm1","ETTm2"),("ETTm2","ETTm1"),
]
for src,tgt in transfers:
    mse_all=[]; mae_all=[]
    for pl in [96,192,336,720]:
        f=f"{log_dir}/eval_{src}_to_{tgt}_pl{pl}.log"
        if not os.path.exists(f): continue
        for line in open(f):
            m=re.search(r'mse:([\d.]+),?\s*mae:([\d.]+)',line)
            if m: mse_all.append(float(m.group(1))); mae_all.append(float(m.group(2)))
    if mse_all:
        print(f"  {src}→{tgt}: MSE={np.mean(mse_all):.3f} MAE={np.mean(mae_all):.3f}")
PYEOF
