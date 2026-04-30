#!/usr/bin/env bash
# =============================================================================
# PatchLinear — PEMS Traffic Forecasting
# Datasets: PEMS03 (358 sensors), PEMS04 (307), PEMS07 (883), PEMS08 (170)
# Standard horizons: 12 / 24 / 48 / 96 steps
# Files must be at ./dataset/PEMS/PEMS0{3,4,7,8}.npz
#
# PatchLinear config rationale:
#   - d_model=256, c_ff=128: PEMS has 170-883 sensors, needs larger model
#   - patch_len=12, stride=6: 5-minute data, 12-step patch = 1 hour context
#   - dw_kernel=7: moderate receptive field for traffic periodicity
#   - lr=5e-3: consistent with our Traffic long-term config
#   - label_len=0: PEMS uses direct multi-step (no decoder warmup)
# =============================================================================
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

model_name=PatchLinear
seq_len=96
d_model=256
t_ff=512
c_ff=128
patch_len=12
stride=6
dw_kernel=7
batch_size=32
learning_rate=0.005
train_epochs=50
patience=10
seed=2021

LOG_DIR=logs/pems
mkdir -p "$LOG_DIR"

run_pems() {
  local DATASET=$1
  local ENC_IN=$2

  echo "=================================================="
  echo ">>> ${DATASET}  enc_in=${ENC_IN}"
  echo "=================================================="

  for PRED in 12 24 48 96; do
    TAG="${DATASET}_pl${PRED}"
    echo "  pred_len=${PRED}"
    python -u run.py \
      --is_training    1 \
      --root_path      ./dataset/PEMS/ \
      --data_path      "${DATASET}.npz" \
      --model_id       "${TAG}" \
      --model          "${model_name}" \
      --data           PEMS \
      --features       M \
      --seq_len        "${seq_len}" \
      --label_len      0 \
      --pred_len       "${PRED}" \
      --enc_in         "${ENC_IN}" \
      --d_model        "${d_model}" \
      --t_ff           "${t_ff}" \
      --c_ff           "${c_ff}" \
      --patch_len      "${patch_len}" \
      --stride         "${stride}" \
      --dw_kernel      "${dw_kernel}" \
      --use_decomp     1 \
      --use_trend_stream 1 \
      --use_seas_stream  1 \
      --use_fusion_gate  1 \
      --use_cross_channel 1 \
      --use_alpha_gate    1 \
      --batch_size     "${batch_size}" \
      --learning_rate  "${learning_rate}" \
      --lradj          sigmoid \
      --train_epochs   "${train_epochs}" \
      --patience       "${patience}" \
      --seed           "${seed}" \
      --des            Exp \
      2>&1 | tee "${LOG_DIR}/${TAG}.log"

    grep "mse:" "${LOG_DIR}/${TAG}.log" | tail -1 \
      || echo "  [check ${LOG_DIR}/${TAG}.log]"
    echo ""
  done
}

run_pems PEMS03 358
run_pems PEMS04 307
run_pems PEMS07 883
run_pems PEMS08 170

echo "=================================================="
echo "SUMMARY — best MSE per dataset per horizon"
echo "=================================================="
python3 - << 'PYEOF'
import re, os
from collections import defaultdict

log_dir = "logs/pems"
datasets = {"PEMS03": 358, "PEMS04": 307, "PEMS07": 883, "PEMS08": 170}
horizons = [12, 24, 48, 96]

print(f"{'Dataset':<10}", end="")
for h in horizons:
    print(f"  H={h:>2} MSE   MAE", end="")
print()
print("-" * 80)

for ds in datasets:
    print(f"{ds:<10}", end="")
    for h in horizons:
        fname = f"{log_dir}/{ds}_pl{h}.log"
        mse = mae = None
        if os.path.exists(fname):
            for line in open(fname):
                m = re.search(r'mse:\s*([\d.]+).*?mae:\s*([\d.]+)', line)
                if m:
                    mse, mae = float(m.group(1)), float(m.group(2))
        if mse:
            print(f"  {mse:.3f}  {mae:.3f}", end="")
        else:
            print(f"  {'—':>5}  {'—':>5}", end="")
    print()
PYEOF
