#!/usr/bin/env bash
# =============================================================================
# PatchLinear — SWA + cosine warm-restart PILOT
#
# Compares 4 variants per (dataset, horizon):
#   Base : current setup (sigmoid LR, 100 epochs, no SWA) — fresh baseline
#   OptA : short train (30 ep) + SWA at 60% -> SWA starts at epoch 18
#   OptB : long train  (100 ep) + SWA at 20% -> SWA starts at epoch 20
#          (early stopping is suspended once SWA activates, so 100 is a ceiling)
#   OptC : cosine_warm (T_0=5, T_mult=2, 35 ep) + SWA at 43% -> starts at ep 15
#
# Datasets: ETTh1, ETTh2, ETTm1, ETTm2, Weather, Exchange
#   (Traffic / Electricity / Solar / ILI intentionally excluded — too slow
#    or too different to mix into a clean A/B/C comparison.)
# Horizons: 96, 720 (short + long). Seed: 2021.
#
# Usage:
#   bash scripts/run_pilot_swa.sh
#
# Outputs:
#   logs/pilot/*.log       per-run stdout/stderr
#   result.txt             appended-to; pilot rows have _Base / _OptA / etc.
#   checkpoints/*/history.csv per-epoch wMAE/MSE/drift (if --keep_checkpoint)
# =============================================================================

set -uo pipefail   # -e removed so one failing run doesn't kill the script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs/pilot"
mkdir -p "$LOG_DIR"

PILOT_TAG="pilot_$(date +%Y%m%d_%H%M%S)"
echo "Pilot tag: ${PILOT_TAG}"
echo "Logs:      ${LOG_DIR}"
echo ""

run_one() {
  local TAG=$1; shift
  local LOG="${LOG_DIR}/${TAG}.log"
  echo ">>> ${TAG}"
  if python -u run.py "$@" > "$LOG" 2>&1; then
    grep "mse:" "$LOG" | tail -1 || echo "  (no mse line found in log)"
  else
    echo "  !! FAILED  (see ${LOG})"
  fi
}

# ── horizons to pilot ────────────────────────────────────────────────────────
HORIZONS=(96 720)

# ── variant flag bundles ────────────────────────────────────────────────────
# Each bundle overrides --lradj / --train_epochs / SWA flags and sets --des so
# settings (and therefore checkpoint dirs & result.txt rows) don't collide.
BASE_OVR="--lradj sigmoid      --train_epochs 100 --patience 10 --des Base"
OPTA_OVR="--lradj sigmoid      --train_epochs 30  --patience 10 --des OptA \
          --use_swa --swa_start_frac 0.60"
OPTB_OVR="--lradj sigmoid      --train_epochs 100 --patience 10 --des OptB \
          --use_swa --swa_start_frac 0.20"
OPTC_OVR="--lradj cosine_warm  --cosine_T_0 5 --cosine_T_mult 2 \
          --train_epochs 35  --patience 10 --des OptC \
          --use_swa --swa_start_frac 0.43"

COMMON="--is_training 1 --root_path ./dataset/ --features M \
        --seq_len 96 --label_len 48 \
        --d_model 64 --t_ff 128 \
        --model PatchLinear --seed 2021"

# ── per-dataset static flags (no scheduler / no SWA flags — those come from OVR) ─
DS_ETTh1="--data_path ETTh1.csv --data ETTh1 --enc_in 7 \
          --c_ff 16 --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
          --batch_size 2048 --learning_rate 0.0005"

DS_ETTh2="--data_path ETTh2.csv --data ETTh2 --enc_in 7 \
          --c_ff 16 --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
          --batch_size 2048 --learning_rate 0.0005"

DS_ETTm1="--data_path ETTm1.csv --data ETTm1 --enc_in 7 \
          --c_ff 16 --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
          --batch_size 2048 --learning_rate 0.0005"

DS_ETTm2="--data_path ETTm2.csv --data ETTm2 --enc_in 7 \
          --c_ff 16 --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
          --batch_size 2048 --learning_rate 0.0001"

DS_WEATHER="--data_path weather.csv --data custom --enc_in 21 \
            --c_ff 16 --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
            --batch_size 2048 --learning_rate 0.0005"

# Exchange has lr=2e-6; default swa_lr would be 1e-7 (basically zero).
# Force a higher swa_lr so SWA sees weight motion worth averaging.
DS_EXCHANGE="--data_path exchange_rate.csv --data custom --enc_in 8 \
             --c_ff 16 --patch_len 8 --stride 4 --dw_kernel 3 --alpha 0.3 \
             --batch_size 32 --learning_rate 0.000002 \
             --swa_lr 0.0000005"

# Dataset iteration list: (name, static-flags)
DATASETS=(
  "ETTh1   $DS_ETTh1"
  "ETTh2   $DS_ETTh2"
  "ETTm1   $DS_ETTm1"
  "ETTm2   $DS_ETTm2"
  "Weather $DS_WEATHER"
  "Exchange $DS_EXCHANGE"
)

VARIANTS=("Base" "OptA" "OptB" "OptC")

# ── main loop ───────────────────────────────────────────────────────────────
for ENTRY in "${DATASETS[@]}"; do
  # Split first word (name) from the rest (flags)
  NAME="${ENTRY%% *}"
  FLAGS="${ENTRY#* }"

  echo ""
  echo "============================================================"
  echo "  Dataset: ${NAME}"
  echo "============================================================"

  for PL in "${HORIZONS[@]}"; do
    for VAR in "${VARIANTS[@]}"; do
      case "$VAR" in
        Base) OVR="$BASE_OVR" ;;
        OptA) OVR="$OPTA_OVR" ;;
        OptB) OVR="$OPTB_OVR" ;;
        OptC) OVR="$OPTC_OVR" ;;
      esac
      TAG="${NAME}_pl${PL}_${VAR}"
      run_one "$TAG" --model_id "$TAG" $COMMON $FLAGS $OVR --pred_len "$PL"
    done
  done
done

# ── optional: Solar (bigger; uncomment to include) ──────────────────────────
# DS_SOLAR="--data_path solar.txt --data Solar --enc_in 137 \
#           --d_model 192 --t_ff 384 \
#           --c_ff 34 --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
#           --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
#           --use_cross_channel 1 --use_alpha_gate 1 \
#           --batch_size 512 --learning_rate 0.0008"
# for PL in "${HORIZONS[@]}"; do
#   for VAR in "${VARIANTS[@]}"; do
#     case "$VAR" in Base) OVR="$BASE_OVR";; OptA) OVR="$OPTA_OVR";;
#                    OptB) OVR="$OPTB_OVR";; OptC) OVR="$OPTC_OVR";; esac
#     TAG="Solar_pl${PL}_${VAR}"
#     # Solar overrides d_model/t_ff so drop those from $COMMON:
#     COMMON_NO_DMODEL="--is_training 1 --root_path ./dataset/ --features M \
#                       --seq_len 96 --label_len 48 --model PatchLinear --seed 2021"
#     run_one "$TAG" --model_id "$TAG" $COMMON_NO_DMODEL $DS_SOLAR $OVR --pred_len "$PL"
#   done
# done

# ── summary ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  PILOT SUMMARY  (${PILOT_TAG})"
echo "============================================================"

python3 - <<'PY'
import re, os, collections

# Parse result.txt for pilot-tagged rows (des is Base/OptA/OptB/OptC)
pattern_setting = re.compile(
    r'^(?P<mid>[^_]+(?:_[^_]+)*?)_PatchLinear_.+_'
    r'(?P<des>Base|OptA|OptB|OptC)_\d+(?:_cos)?(?:_swa)?$'
)
pattern_metrics = re.compile(r'mse:([\d.]+),\s*mae:([\d.]+)')

if not os.path.exists('result.txt'):
    print("No result.txt found — nothing to summarise.")
    raise SystemExit(0)

rows = []
with open('result.txt') as f:
    lines = [ln.strip() for ln in f if ln.strip()]
i = 0
while i < len(lines) - 1:
    m = pattern_setting.match(lines[i])
    if m:
        mm = pattern_metrics.search(lines[i+1])
        if mm:
            mid = m.group('mid')
            des = m.group('des')
            mse, mae = float(mm.group(1)), float(mm.group(2))
            # mid is like "ETTh1_pl96_Base" — strip trailing "_<VAR>" to get ds+pl
            base = re.sub(r'_(Base|OptA|OptB|OptC)$', '', mid)
            rows.append((base, des, mse, mae))
    i += 1

# Take only the LAST occurrence of each (base, des) pair (most recent run)
last = {}
for base, des, mse, mae in rows:
    last[(base, des)] = (mse, mae)

# Pivot
bases = sorted({b for (b, _) in last})
variants = ['Base', 'OptA', 'OptB', 'OptC']

if not bases:
    print("No pilot rows found in result.txt yet.")
    raise SystemExit(0)

# Header
header = f"{'dataset/horizon':<22}"
for v in variants:
    header += f"  {v+' MSE':>11}  {v+' MAE':>11}"
header += "  winner(MSE)"
print(header)
print('-' * len(header))

for b in bases:
    line = f"{b:<22}"
    mses = {}
    for v in variants:
        if (b, v) in last:
            mse, mae = last[(b, v)]
            line += f"  {mse:>11.5f}  {mae:>11.5f}"
            mses[v] = mse
        else:
            line += f"  {'—':>11}  {'—':>11}"
    if mses:
        winner = min(mses, key=mses.get)
        base_mse = mses.get('Base')
        if base_mse and winner != 'Base':
            delta = (mses[winner] - base_mse) / base_mse * 100
            line += f"  {winner} ({delta:+.1f}%)"
        else:
            line += f"  {winner}"
    print(line)

# Aggregate: how often each variant wins
win_counts = collections.Counter()
for b in bases:
    mses = {v: last[(b, v)][0] for v in variants if (b, v) in last}
    if len(mses) >= 2:
        win_counts[min(mses, key=mses.get)] += 1

print()
print("Win counts (best MSE per dataset/horizon):")
for v in variants:
    print(f"  {v}: {win_counts[v]}")
PY

echo ""
echo "Done."
