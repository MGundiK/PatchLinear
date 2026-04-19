#!/usr/bin/env bash
# =============================================================================
# PatchLinear — Long Sequence Length Sweep
# Protocol: xPatch Table 14
#   L in {96,192,336,512,720} for standard datasets
#   L in {36,104,148} for ILI
#   Single seed (2021) for sweep; best L confirmed with 3 seeds
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/logs/longseq"
mkdir -p "$LOG_DIR"
RESULT="$SCRIPT_DIR/result_longseq.txt"
> "$RESULT"
SEED=2021

run_one() {
  local TAG=$1; shift
  local LOG="${LOG_DIR}/${TAG}.log"
  echo ">>> ${TAG}"
  python -u run.py "$@" > "$LOG" 2>&1
  local RES
  RES=$(grep "mse:" "$LOG" | tail -1 || echo "  FAILED")
  echo "  ${RES}"
  echo "${TAG}: ${RES}" >> "$RESULT"
}

# ── ETTh1 ──────────────────────────────────────────────────────────────────
for L in 192 336 512 720; do
  for PL in 96 192 336 720; do
    run_one "ETTh1_L${L}_pl${PL}" \
      --is_training 1 --root_path ./dataset/ \
      --data_path ETTh1.csv --data ETTh1 --features M \
      --seq_len ${L} --label_len 48 --pred_len ${PL} --enc_in 7 \
      --d_model 64 --t_ff 128 --c_ff 16 \
      --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
      --batch_size 2048 --learning_rate 0.0005 \
      --lradj sigmoid --train_epochs 100 --patience 10 \
      --model PatchLinear --seed ${SEED} --des Exp \
      --model_id "ETTh1_L${L}_pl${PL}"
  done
done

# ── ETTh2 ──────────────────────────────────────────────────────────────────
for L in 192 336 512 720; do
  for PL in 96 192 336 720; do
    run_one "ETTh2_L${L}_pl${PL}" \
      --is_training 1 --root_path ./dataset/ \
      --data_path ETTh2.csv --data ETTh2 --features M \
      --seq_len ${L} --label_len 48 --pred_len ${PL} --enc_in 7 \
      --d_model 64 --t_ff 128 --c_ff 16 \
      --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
      --batch_size 2048 --learning_rate 0.0005 \
      --lradj sigmoid --train_epochs 100 --patience 10 \
      --model PatchLinear --seed ${SEED} --des Exp \
      --model_id "ETTh2_L${L}_pl${PL}"
  done
done

# ── ETTm1 ──────────────────────────────────────────────────────────────────
for L in 192 336 512 720; do
  for PL in 96 192 336 720; do
    run_one "ETTm1_L${L}_pl${PL}" \
      --is_training 1 --root_path ./dataset/ \
      --data_path ETTm1.csv --data ETTm1 --features M \
      --seq_len ${L} --label_len 48 --pred_len ${PL} --enc_in 7 \
      --d_model 64 --t_ff 128 --c_ff 16 \
      --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
      --batch_size 2048 --learning_rate 0.0005 \
      --lradj sigmoid --train_epochs 100 --patience 10 \
      --model PatchLinear --seed ${SEED} --des Exp \
      --model_id "ETTm1_L${L}_pl${PL}"
  done
done

# ── ETTm2 ──────────────────────────────────────────────────────────────────
for L in 192 336 512 720; do
  for PL in 96 192 336 720; do
    run_one "ETTm2_L${L}_pl${PL}" \
      --is_training 1 --root_path ./dataset/ \
      --data_path ETTm2.csv --data ETTm2 --features M \
      --seq_len ${L} --label_len 48 --pred_len ${PL} --enc_in 7 \
      --d_model 64 --t_ff 128 --c_ff 16 \
      --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
      --batch_size 2048 --learning_rate 0.0001 \
      --lradj sigmoid --train_epochs 100 --patience 10 \
      --model PatchLinear --seed ${SEED} --des Exp \
      --model_id "ETTm2_L${L}_pl${PL}"
  done
done

# ── Weather ────────────────────────────────────────────────────────────────
for L in 192 336 512 720; do
  for PL in 96 192 336 720; do
    run_one "Weather_L${L}_pl${PL}" \
      --is_training 1 --root_path ./dataset/ \
      --data_path weather.csv --data custom --features M \
      --seq_len ${L} --label_len 48 --pred_len ${PL} --enc_in 21 \
      --d_model 64 --t_ff 128 --c_ff 16 \
      --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
      --batch_size 2048 --learning_rate 0.0005 \
      --lradj sigmoid --train_epochs 100 --patience 10 \
      --model PatchLinear --seed ${SEED} --des Exp \
      --model_id "Weather_L${L}_pl${PL}"
  done
done

# ── Traffic ────────────────────────────────────────────────────────────────
for L in 192 336 512 720; do
  for PL in 96 192 336 720; do
    run_one "Traffic_L${L}_pl${PL}" \
      --is_training 1 --root_path ./dataset/ \
      --data_path traffic.csv --data custom --features M \
      --seq_len ${L} --label_len 48 --pred_len ${PL} --enc_in 862 \
      --d_model 256 --t_ff 512 --c_ff 128 \
      --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
      --batch_size 64 --learning_rate 0.005 --patience 5 \
      --lradj sigmoid --train_epochs 100 \
      --model PatchLinear --seed ${SEED} --des Exp \
      --model_id "Traffic_L${L}_pl${PL}"
  done
done

# ── Electricity ────────────────────────────────────────────────────────────
for L in 192 336 512 720; do
  for PL in 96 192 336 720; do
    run_one "Elec_L${L}_pl${PL}" \
      --is_training 1 --root_path ./dataset/ \
      --data_path electricity.csv --data custom --features M \
      --seq_len ${L} --label_len 48 --pred_len ${PL} --enc_in 321 \
      --d_model 256 --t_ff 512 --c_ff 128 \
      --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
      --batch_size 256 --learning_rate 0.01 \
      --lradj sigmoid --train_epochs 100 --patience 10 \
      --model PatchLinear --seed ${SEED} --des Exp \
      --model_id "Elec_L${L}_pl${PL}"
  done
done

# ── Exchange ────────────────────────────────────────────────────────────────
for L in 192 336 512 720; do
  for PL in 96 192 336 720; do
    run_one "Exchange_L${L}_pl${PL}" \
      --is_training 1 --root_path ./dataset/ \
      --data_path exchange_rate.csv --data custom --features M \
      --seq_len ${L} --label_len 48 --pred_len ${PL} --enc_in 8 \
      --d_model 64 --t_ff 128 --c_ff 16 \
      --patch_len 8 --stride 4 --dw_kernel 3 --alpha 0.3 \
      --batch_size 32 --learning_rate 0.000002 \
      --lradj sigmoid --train_epochs 100 --patience 10 \
      --model PatchLinear --seed ${SEED} --des Exp \
      --model_id "Exchange_L${L}_pl${PL}"
  done
done

# ── Solar ──────────────────────────────────────────────────────────────────
for L in 192 336 512 720; do
  for PL in 96 192 336 720; do
    run_one "Solar_L${L}_pl${PL}" \
      --is_training 1 --root_path ./dataset/ \
      --data_path solar_AL.txt --data Solar --features M \
      --seq_len ${L} --label_len 48 --pred_len ${PL} --enc_in 137 \
      --d_model 192 --t_ff 384 --c_ff 34 \
      --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
      --batch_size 512 --learning_rate 0.0008 \
      --lradj sigmoid --train_epochs 100 --patience 10 \
      --model PatchLinear --seed ${SEED} --des Exp \
      --model_id "Solar_L${L}_pl${PL}"
  done
done

# ── ILI ────────────────────────────────────────────────────────────────────
for L in 36 104 148; do
  for PL in 24 36 48 60; do
    run_one "ILI_L${L}_pl${PL}" \
      --is_training 1 --root_path ./dataset/ \
      --data_path national_illness.csv --data custom --features M \
      --seq_len ${L} --label_len 18 --pred_len ${PL} --enc_in 7 \
      --d_model 64 --t_ff 128 --c_ff 8 \
      --patch_len 6 --stride 3 --dw_kernel 3 --alpha 0.3 \
      --use_decomp 0 --use_seas_stream 0 --use_fusion_gate 0 \
      --batch_size 32 --learning_rate 0.01 \
      --lradj type3 --train_epochs 100 --patience 10 \
      --model PatchLinear --seed ${SEED} --des Exp \
      --model_id "ILI_L${L}_pl${PL}"
  done
done

echo ""
echo "=== LONG SEQ SWEEP SUMMARY ==="
python3 - << 'PYEOF'
import re, os, numpy as np

log_dir = "logs/longseq"
datasets = {
    'ETTh1': (['96','192','336','512','720'], ['96','192','336','720']),
    'ETTh2': (['96','192','336','512','720'], ['96','192','336','720']),
    'ETTm1': (['96','192','336','512','720'], ['96','192','336','720']),
    'ETTm2': (['96','192','336','512','720'], ['96','192','336','720']),
    'Weather': (['96','192','336','512','720'], ['96','192','336','720']),
    'Traffic': (['96','192','336','512','720'], ['96','192','336','720']),
    'Elec':   (['96','192','336','512','720'], ['96','192','336','720']),
    'Exchange': (['96','192','336','512','720'], ['96','192','336','720']),
    'Solar':  (['96','192','336','512','720'], ['96','192','336','720']),
    'ILI':    (['36','104','148'], ['24','36','48','60']),
}

print(f"{'Dataset':<12} {'Best L':>7} {'Best avg MSE':>13}  Per-L averages")
print("─"*70)
for ds, (Ls, PLs) in datasets.items():
    best_L, best_mse = None, float('inf')
    L_avgs = {}
    for L in Ls:
        mses = []
        for pl in PLs:
            f = f"{log_dir}/{ds}_L{L}_pl{pl}.log"
            if not os.path.exists(f): continue
            for line in open(f):
                m = re.search(r'mse:([\d.]+)', line)
                if m: mses.append(float(m.group(1)))
        if mses:
            avg = np.mean(mses)
            L_avgs[L] = avg
            if avg < best_mse:
                best_mse = avg; best_L = L
    if best_L:
        avgs_str = '  '.join(f"L{l}:{v:.3f}" for l,v in L_avgs.items())
        print(f"{ds:<12} {best_L:>7} {best_mse:>13.3f}  {avgs_str}")
PYEOF
