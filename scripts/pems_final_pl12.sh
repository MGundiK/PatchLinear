#!/usr/bin/env bash
# =============================================================================
# PatchLinear — PEMS Final Runs (pred_len=12)
# Best config: lr=0.001, c_ff=128, dw_kernel=13
# Datasets: PEMS03 (358), PEMS04 (307), PEMS08 (170)
# Seeds: 2021, 2022, 2023
# 9 runs total.
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

LOG_DIR=logs/pems_final_pl12
mkdir -p "$LOG_DIR"

for DS_INFO in "PEMS03:358" "PEMS04:307" "PEMS08:170"; do
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
echo "FINAL RESULTS — 3-seed means"
echo "========================================================"
python3 - << 'PYEOF'
import re, os

log_dir = "logs/pems_final_pl12"
lr, c_ff, dk = 0.001, 128, 13

print(f"Config: lr={lr}, c_ff={c_ff}, dw_kernel={dk}, epochs=100, patience=15")
print()
print(f"  {'Dataset':<8}  {'MAE':>8}  {'MSE':>10}  {'RMSE':>8}  {'MAPE%':>8}  {'Seeds found'}")
print("  " + "-" * 58)

for ds in ["PEMS03", "PEMS04", "PEMS08"]:
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
                rows.append({
                    'mse':  float(m.group(1)),
                    'mae':  float(m.group(2)),
                    'rmse': float(m.group(3)),
                    'mape': float(m.group(4)),
                })
    if rows:
        n = len(rows)
        avg = {k: sum(r[k] for r in rows) / n for k in rows[0]}
        print(f"  {ds:<8}  "
              f"{avg['mae']:>8.4f}  "
              f"{avg['mse']:>10.4f}  "
              f"{avg['rmse']:>8.4f}  "
              f"{avg['mape']*100:>7.4f}%  "
              f"{n}/3")
    else:
        print(f"  {ds:<8}  — no logs found in {log_dir}/")
PYEOF
