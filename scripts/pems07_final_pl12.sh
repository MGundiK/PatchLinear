#!/usr/bin/env bash
# =============================================================================
# PatchLinear — PEMS07 Final Run (pred_len=12)
# Best config: lr=0.001, c_ff=128, dw_kernel=13
# enc_in=883 (883 sensors)
# Seeds: 2021, 2022, 2023
# =============================================================================
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DS=PEMS07
ENC_IN=883
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

echo "========================================================"
echo ">>> ${DS}  enc_in=${ENC_IN}"
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
        --enc_in            "${ENC_IN}" \
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

echo "========================================================"
echo "PEMS07 RESULTS — 3-seed mean"
echo "========================================================"
python3 - << 'PYEOF'
import re, os

log_dir  = "logs/pems_final_pl12"
ds       = "PEMS07"
lr, c_ff, dk = 0.001, 128, 13

rows = []
for seed in [2021, 2022, 2023]:
    tag   = f"{ds}_pl12_lr{lr}_cff{c_ff}_dk{dk}_s{seed}"
    fpath = os.path.join(log_dir, f"{tag}.log")
    if not os.path.exists(fpath):
        print(f"  missing: {fpath}")
        continue
    for line in open(fpath):
        m = re.search(
            r'mse:([\d.]+).*?mae:([\d.]+).*?rmse:([\d.]+).*?mape:([\d.]+)', line)
        if m:
            rows.append(dict(
                seed=seed,
                mse=float(m.group(1)), mae=float(m.group(2)),
                rmse=float(m.group(3)), mape=float(m.group(4)),
            ))

print(f"  {'Seed':<6}  {'MAE':>8}  {'MSE':>10}  {'RMSE':>8}  {'MAPE%':>8}")
print("  " + "-" * 46)
for r in rows:
    print(f"  {r['seed']:<6}  {r['mae']:>8.4f}  {r['mse']:>10.4f}"
          f"  {r['rmse']:>8.4f}  {r['mape']*100:>7.4f}%")

if rows:
    n   = len(rows)
    avg = {k: sum(r[k] for r in rows)/n for k in ['mae','mse','rmse','mape']}
    print("  " + "-" * 46)
    print(f"  {'Mean':<6}  {avg['mae']:>8.4f}  {avg['mse']:>10.4f}"
          f"  {avg['rmse']:>8.4f}  {avg['mape']*100:>7.4f}%")
    print()
    timemixer = dict(mae=20.57, rmse=33.59, mape=8.62)
    print("  vs TimeMixer:")
    for m, tm in timemixer.items():
        pl  = avg[m] if m != 'mape' else avg['mape']*100
        sym = "✓ BEST" if pl < tm else "  2nd"
        print(f"    {m.upper()}: PL={pl:.4f}  TM={tm:.2f}  {sym}")
PYEOF
