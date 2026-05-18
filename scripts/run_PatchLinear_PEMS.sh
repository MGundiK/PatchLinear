#!/usr/bin/env bash
# =============================================================================
# PatchLinear — PEMS Traffic Forecasting (tuned config, pred_len=12 only)
# Datasets : PEMS03 (358), PEMS04 (307), PEMS07 (883), PEMS08 (170)
# Files must be at ./dataset/PEMS/PEMS0{3,4,7,8}.npz
# Seeds    : 2021 / 2022 / 2023
# Config   : lr=0.001, c_ff=128, dw_kernel=13, epochs=100, patience=15
# PatchLinear config rationale:
#   - d_model=256, c_ff=128: PEMS has 170-883 sensors, needs larger model
#   - patch_len=12, stride=6: 5-minute data, 12-step patch = 1 hour context
#   - dw_kernel=13: moderate receptive field for traffic periodicity
#   - lr=0.001: consistent with our Traffic long-term config
#   - label_len=0: PEMS uses direct multi-step (no decoder warmup)
# =============================================================================
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL=PatchLinear
SEQ_LEN=96
D_MODEL=256
T_FF=512
C_FF=128
PATCH_LEN=12
STRIDE=6
DW_KERNEL=13
BATCH=32
LR=0.001
EPOCHS=100
PATIENCE=15
ALPHA=0.3

LOG_DIR=logs/pems_tuned
mkdir -p "$LOG_DIR"

for DS_INFO in "PEMS03:358" "PEMS04:307" "PEMS07:883" "PEMS08:170"; do
    DS="${DS_INFO%%:*}"
    ENC="${DS_INFO##*:}"
    echo "========================================================"
    echo ">>> ${DS}  enc_in=${ENC}"
    echo "========================================================"
    for SEED in 2021 2022 2023; do
        TAG="${DS}_pl12_lr${LR}_cff${C_FF}_dk${DW_KERNEL}_s${SEED}"
        echo "  seed=${SEED}  tag=${TAG}"
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
            --enc_in            "${ENC}" \
            --d_model           "${D_MODEL}" \
            --t_ff              "${T_FF}" \
            --c_ff              "${C_FF}" \
            --patch_len         "${PATCH_LEN}" \
            --stride            "${STRIDE}" \
            --dw_kernel         "${DW_KERNEL}" \
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
        grep "mse:" "${LOG_DIR}/${TAG}.log" | tail -1
        echo ""
    done
done

# ── 3-seed summary ────────────────────────────────────────────────────────────
echo "========================================================"
echo "SUMMARY — 3-seed means (pred_len=12)"
echo "========================================================"
python3 - << 'PYEOF'
import re, os

log_dir  = "logs/pems_tuned"
lr, c_ff, dk = 0.001, 128, 13

print(f"  {'Dataset':<8}  {'MAE':>8}  {'MSE':>10}  {'RMSE':>8}  {'MAPE%':>8}  {'n':>3}")
print("  " + "-" * 50)

for ds in ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]:
    rows = []
    for seed in [2021, 2022, 2023]:
        tag   = f"{ds}_pl12_lr{lr}_cff{c_ff}_dk{dk}_s{seed}"
        fpath = os.path.join(log_dir, f"{tag}.log")
        if not os.path.exists(fpath):
            continue
        for line in open(fpath):
            m = re.search(
                r'mse:([\d.]+).*?mae:([\d.]+).*?rmse:([\d.]+).*?mape:([\d.]+)', line)
            if m:
                rows.append(dict(
                    mse=float(m.group(1)), mae=float(m.group(2)),
                    rmse=float(m.group(3)), mape=float(m.group(4))))
    if rows:
        n   = len(rows)
        avg = {k: sum(r[k] for r in rows)/n for k in rows[0]}
        print(f"  {ds:<8}  {avg['mae']:>8.4f}  {avg['mse']:>10.4f}"
              f"  {avg['rmse']:>8.4f}  {avg['mape']*100:>7.4f}%  {n:>3}")
    else:
        print(f"  {ds:<8}  — no logs found")
PYEOF
