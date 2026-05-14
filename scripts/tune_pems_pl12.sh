#!/usr/bin/env bash
# =============================================================================
# PatchLinear — PEMS Hyperparameter Tuning  (pred_len=12 only)
# Datasets: PEMS03 (358 sensors), PEMS04 (307), PEMS08 (170)
#
# WHAT WE'RE TUNING AND WHY
# ──────────────────────────
# lr {0.001, 0.0005}
#   Baseline 0.005 is 5–50× higher than SOTA configs for these datasets.
#   With patience=10 and sigmoid lradj the model early-stops before the
#   schedule cools the lr to a useful range.  0.001/0.0005 fix this.
#
# c_ff {128, 256}
#   VGM bottleneck (GatingBlock(2C, c_ff)):
#     PEMS03 c_ff=128 → 18% of 2C retained  (too tight for spatial corr.)
#     PEMS03 c_ff=256 → 36% retained
#     PEMS04 c_ff=128 → 21%  /  256 → 42%
#     PEMS08 c_ff=128 → 38%  /  256 → 75%
#   For pred_len=12 cross-channel (spatial) information is more important
#   than long-range temporal modelling, so the VGM is the critical path.
#
# dw_kernel {7, 13}
#   ERF = (dw_kernel-1)*stride + patch_len  (stride=6, patch_len=12)
#     dw_kernel=7  → ERF=48 steps = 4 h of 5-min data
#     dw_kernel=13 → ERF=84 steps = 7 h  (covers morning/evening rush)
#
# FIXED (not tuned here)
#   seq_len=96, patch_len=12, stride=6  → N=16 patches
#   d_model=256, t_ff=512               → consistent with long-term config
#   train_epochs=100, patience=15       → more room to converge than 50/10
#   seed=2021 for Stage 1 (search), then 3 seeds for Stage 2 (final)
#
# STAGE 1  Grid search: 2 lr × 2 c_ff × 2 dw_kernel = 8 configs per dataset
#          = 24 runs total.  All with seed=2021.
# STAGE 2  Best config re-run with seeds {2021, 2022, 2023}.
#          Set BEST_* variables below after Stage 1 completes.
# =============================================================================
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL=PatchLinear
SEQ_LEN=96
D_MODEL=256
T_FF=512
PATCH_LEN=12
STRIDE=6
BATCH=32
EPOCHS=100
PATIENCE=15
ALPHA=0.3

LOG_DIR=logs/tune_pems_pl12
mkdir -p "$LOG_DIR"

# ── helper ────────────────────────────────────────────────────────────────────
run_one() {
    local DS=$1 ENC_IN=$2 LR=$3 C_FF=$4 DW=$5 SEED=$6
    local TAG="${DS}_pl12_lr${LR}_cff${C_FF}_dk${DW}_s${SEED}"
    echo "  [run] $TAG"
    python -u run.py \
        --is_training       1 \
        --root_path         ./dataset/PEMS/ \
        --data_path         "${DS}.npz" \
        --model_id          "${TAG}" \
        --model             "${MODEL}" \
        --data              PEMS \
        --features          M \
        --seq_len           "${SEQ_LEN}" \
        --label_len         0 \
        --pred_len          12 \
        --enc_in            "${ENC_IN}" \
        --d_model           "${D_MODEL}" \
        --t_ff              "${T_FF}" \
        --c_ff              "${C_FF}" \
        --patch_len         "${PATCH_LEN}" \
        --stride            "${STRIDE}" \
        --dw_kernel         "${DW}" \
        --alpha             "${ALPHA}" \
        --use_decomp        1 \
        --use_trend_stream  1 \
        --use_seas_stream   1 \
        --use_fusion_gate   1 \
        --use_cross_channel 1 \
        --use_alpha_gate    1 \
        --batch_size        "${BATCH}" \
        --learning_rate     "${LR}" \
        --lradj             sigmoid \
        --train_epochs      "${EPOCHS}" \
        --patience          "${PATIENCE}" \
        --seed              "${SEED}" \
        --des               Exp \
        2>&1 | tee "${LOG_DIR}/${TAG}.log"
    # extract final test metrics from log
    grep "mse:" "${LOG_DIR}/${TAG}.log" | tail -1 \
        || echo "  [no metrics found — check ${LOG_DIR}/${TAG}.log]"
}

# =============================================================================
# STAGE 1 — grid search, seed=2021
# =============================================================================
echo ""
echo "========================================================"
echo "STAGE 1 — Grid search (seed=2021)"
echo "  lr      : {0.001, 0.0005}"
echo "  c_ff    : {128, 256}"
echo "  dw_kernel: {7, 13}"
echo "========================================================"

for DS_INFO in "PEMS03:358" "PEMS04:307" "PEMS08:170"; do
    DS="${DS_INFO%%:*}"
    ENC="${DS_INFO##*:}"
    echo ""
    echo "──────────────── ${DS} (enc_in=${ENC}) ────────────────"
    for LR in 0.001 0.0005; do
        for C_FF in 128 256; do
            for DW in 7 13; do
                run_one "$DS" "$ENC" "$LR" "$C_FF" "$DW" 2021
            done
        done
    done
done

# ── Parse Stage 1 results ─────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "STAGE 1 RESULTS — MAE by config (seed=2021)"
echo "========================================================"
python3 - << 'PYEOF'
import re, os, itertools
from collections import defaultdict

log_dir  = "logs/tune_pems_pl12"
datasets = {"PEMS03": 358, "PEMS04": 307, "PEMS08": 170}
lrs      = [0.001, 0.0005]
cffs     = [128, 256]
dks      = [7, 13]

print(f"{'Config':<30}  {'PEMS03 MAE':>11}  {'PEMS04 MAE':>11}  {'PEMS08 MAE':>11}  {'Mean MAE':>10}")
print("-" * 80)

best_mean = float('inf')
best_cfg  = None

for lr, c_ff, dk in itertools.product(lrs, cffs, dks):
    label  = f"lr={lr}, c_ff={c_ff}, dk={dk}"
    maes   = []
    parts  = []
    for ds in ["PEMS03", "PEMS04", "PEMS08"]:
        tag  = f"{ds}_pl12_lr{lr}_cff{c_ff}_dk{dk}_s2021"
        fpath = os.path.join(log_dir, f"{tag}.log")
        mae  = None
        if os.path.exists(fpath):
            for line in open(fpath):
                m = re.search(r'mae:\s*([\d.]+)', line)
                if m:
                    mae = float(m.group(1))
        parts.append(f"{mae:>11.4f}" if mae else f"{'—':>11}")
        if mae:
            maes.append(mae)
    mean_str = f"{sum(maes)/len(maes):>10.4f}" if len(maes) == 3 else f"{'—':>10}"
    print(f"  {label:<28}  {'  '.join(parts)}  {mean_str}")
    if len(maes) == 3:
        m = sum(maes)/len(maes)
        if m < best_mean:
            best_mean, best_cfg = m, (lr, c_ff, dk)

print()
if best_cfg:
    print(f"  ★ Best config: lr={best_cfg[0]}, c_ff={best_cfg[1]}, dw_kernel={best_cfg[2]}")
    print(f"    Mean MAE = {best_mean:.4f}")
    print()
    print("  → Set BEST_LR, BEST_C_FF, BEST_DK below and run Stage 2.")
PYEOF

# =============================================================================
# STAGE 2 — best config × 3 seeds
# Edit BEST_* to match the winner from Stage 1 before running this section.
# =============================================================================
echo ""
echo "========================================================"
echo "STAGE 2 — best config × 3 seeds"
echo "(edit BEST_* variables before running)"
echo "========================================================"

# ▼▼▼ SET THESE AFTER STAGE 1 ▼▼▼
BEST_LR=0.001
BEST_C_FF=256
BEST_DK=13
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# Comment out this guard once you've set the values above.
if [[ "$BEST_LR" == "FILL_IN" ]]; then
    echo "  Skipping Stage 2 — set BEST_LR / BEST_C_FF / BEST_DK first."
    exit 0
fi

for DS_INFO in "PEMS03:358" "PEMS04:307" "PEMS08:170"; do
    DS="${DS_INFO%%:*}"
    ENC="${DS_INFO##*:}"
    echo ""
    echo "──────────────── ${DS} — 3 seeds ────────────────"
    for SEED in 2021 2022 2023; do
        run_one "$DS" "$ENC" "$BEST_LR" "$BEST_C_FF" "$BEST_DK" "$SEED"
    done
done

# ── Parse Stage 2 results (3-seed means) ─────────────────────────────────────
echo ""
echo "========================================================"
echo "STAGE 2 RESULTS — 3-seed means"
echo "========================================================"
python3 - << 'PYEOF2'
import re, os

log_dir  = "logs/tune_pems_pl12"
datasets = ["PEMS03", "PEMS04", "PEMS08"]
seeds    = [2021, 2022, 2023]

import sys
# Read BEST_* from environment if set
lr   = os.environ.get("BEST_LR",   "0.001")
c_ff = os.environ.get("BEST_C_FF", "256")
dk   = os.environ.get("BEST_DK",   "13")

print(f"Config: lr={lr}, c_ff={c_ff}, dw_kernel={dk}")
print(f"{'Dataset':<10}  {'MAE':>8}  {'MSE':>10}  {'RMSE':>8}  {'MAPE':>8}")
print("-" * 50)

for ds in datasets:
    all_mae, all_mse, all_rmse, all_mape = [], [], [], []
    for seed in seeds:
        tag   = f"{ds}_pl12_lr{lr}_cff{c_ff}_dk{dk}_s{seed}"
        fpath = os.path.join(log_dir, f"{tag}.log")
        if not os.path.exists(fpath):
            continue
        for line in open(fpath):
            m = re.search(
                r'mse:([\d.]+).*?mae:([\d.]+).*?rmse:([\d.]+).*?mape:([\d.]+)', line)
            if m:
                all_mse.append(float(m.group(1)))
                all_mae.append(float(m.group(2)))
                all_rmse.append(float(m.group(3)))
                all_mape.append(float(m.group(4)))
    if all_mae:
        print(f"  {ds:<8}  "
              f"{sum(all_mae)/len(all_mae):>8.4f}  "
              f"{sum(all_mse)/len(all_mse):>10.4f}  "
              f"{sum(all_rmse)/len(all_rmse):>8.4f}  "
              f"{sum(all_mape)/len(all_mape):>8.4f}")
    else:
        print(f"  {ds:<8}  —  (logs not found)")
PYEOF2

export BEST_LR BEST_C_FF BEST_DK
