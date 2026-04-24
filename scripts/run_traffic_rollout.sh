#!/usr/bin/env bash
# =============================================================================
# PatchLinear — Traffic MseSelect ROLLOUT
#
# Final Traffic numbers with the fix: train_loss=wmae, early_stop_metric=mse.
# This preserves the training trajectory (identical to Base) and only changes
# which checkpoint is saved. Pilot established this gives:
#   pl=96   : −3.9% MSE, +0.4% MAE (3-seed mean)
#   pl=720  : +0.4% MSE (neutral), −0.2% MAE (1 seed)
# Rollout: all 4 horizons × 3 seeds × MseSelect only.
#
# Total: 12 runs. Resume-aware → skips the 3 pl=96 MseSelect runs already done.
# New work: 9 runs (≈27–30h on T4).
#
# For reference / comparison, Base numbers for all horizons × 3 seeds should
# come from your main experiment run (run_experiments.sh --seeds). This
# script does NOT re-run Base — if you don't already have Base results for a
# given (horizon, seed) cell, run_experiments.sh first.
#
# Usage (Colab):
#   %%bash
#   mkdir -p /content/PatchLinear/logs
#   cd /content/PatchLinear
#   bash scripts/run_traffic_rollout.sh 2>&1 | tee logs/traffic_rollout_master.log
# =============================================================================

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs/traffic_rollout"
mkdir -p "$LOG_DIR"

PILOT_TAG="traffic_rollout_$(date +%Y%m%d_%H%M%S)"
echo "Pilot tag: ${PILOT_TAG}"
echo "Logs:      ${LOG_DIR}"
echo ""

# ── resume helper ──────────────────────────────────────────────────────────
already_done() {
  local MODEL_ID=$1
  local VARIANT=$2
  if [ ! -f result.txt ]; then return 1; fi
  grep -q "^${MODEL_ID}_.*_${VARIANT}_[0-9]\+\$" result.txt
}

run_one() {
  local TAG=$1; shift
  local MODEL_ID=$1; shift
  local VARIANT=$1; shift
  local LOG="${LOG_DIR}/${TAG}.log"

  if already_done "$MODEL_ID" "$VARIANT"; then
    local EXISTING
    EXISTING=$(grep -A1 "^${MODEL_ID}_.*_${VARIANT}_[0-9]\+\$" result.txt \
               | grep "mse:" | tail -1)
    echo ">>> ${TAG}  [SKIP — already in result.txt]"
    echo "    ${EXISTING}"
    return 0
  fi

  echo ">>> ${TAG}"
  if python -u run.py --model_id "$MODEL_ID" "$@" > "$LOG" 2>&1; then
    grep "mse:" "$LOG" | tail -1 || echo "  (no mse line found in log)"
  else
    echo "  !! FAILED  (see ${LOG})"
  fi
}

HORIZONS=(96 192 336 720)
SEEDS=(2021 2022 2023)
VARIANT="MseSelect"

# Traffic config mirrors run_experiments.sh
TRAFFIC_COMMON="--is_training 1 --root_path ./dataset/ --features M \
                --seq_len 96 --label_len 48 \
                --data_path traffic.csv --data custom --enc_in 862 \
                --d_model 256 --t_ff 512 --c_ff 128 \
                --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
                --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
                --use_cross_channel 1 --use_alpha_gate 1 \
                --batch_size 64 --learning_rate 0.005 \
                --lradj sigmoid --train_epochs 100 --patience 5 \
                --model PatchLinear"

OVR="--des MseSelect --train_loss wmae --early_stop_metric mse"

echo "============================================================"
echo "  Traffic MseSelect rollout — 4 horizons × 3 seeds = 12 cells"
echo "============================================================"

for SEED in "${SEEDS[@]}"; do
  for PL in "${HORIZONS[@]}"; do
    TAG="Traffic_pl${PL}_s${SEED}_${VARIANT}"
    MODEL_ID="Traffic_${PL}_s${SEED}_${VARIANT}"
    run_one "$TAG" "$MODEL_ID" "$VARIANT" --seed "$SEED" \
            $TRAFFIC_COMMON $OVR --pred_len "$PL"
  done
done

# ── summary ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ROLLOUT SUMMARY  (${PILOT_TAG})"
echo "============================================================"

python3 - <<'PY'
import re, os, statistics

# Match Traffic rows with a Variant of Base or MseSelect, any seed
pat_set = re.compile(
    r'^Traffic_(?P<pl>\d+)_s(?P<seed>\d+)_(?P<var>Base|MseSelect)'
    r'_PatchLinear_.+_(?P=var)_\d+$'
)
pat_met = re.compile(r'mse:([\d.]+),\s*mae:([\d.]+)')

if not os.path.exists('result.txt'):
    print("No result.txt — nothing to summarise.")
    raise SystemExit(0)

rows = []
with open('result.txt') as f:
    lines = [ln.strip() for ln in f if ln.strip()]

i = 0
while i < len(lines) - 1:
    m = pat_set.match(lines[i])
    if m:
        mm = pat_met.search(lines[i+1])
        if mm:
            rows.append((
                int(m.group('pl')),
                int(m.group('seed')),
                m.group('var'),
                float(mm.group(1)),
                float(mm.group(2)),
            ))
    i += 1

last = {}
for pl, seed, var, mse, mae in rows:
    last[(pl, seed, var)] = (mse, mae)

horizons = sorted({pl for (pl, _, _) in last})
variants = ['Base', 'MseSelect']

if not horizons:
    print("No Traffic rows found.")
    raise SystemExit(0)

# ── per-seed table ─────────────────────────────────────────────────────────
seeds = sorted({s for (_, s, _) in last})
for seed in seeds:
    print(f"\n  seed {seed}")
    header = f"    {'horizon':<8}"
    for v in variants:
        header += f"  {v+' MSE':>12}  {v+' MAE':>12}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for pl in horizons:
        line = f"    pl={pl:<5}"
        for v in variants:
            if (pl, seed, v) in last:
                mse, mae = last[(pl, seed, v)]
                line += f"  {mse:>12.6f}  {mae:>12.6f}"
            else:
                line += f"  {'—':>12}  {'—':>12}"
        print(line)

# ── mean across seeds with deltas ──────────────────────────────────────────
print(f"\n  Mean across seeds {seeds}")
header = f"    {'horizon':<8}"
for v in variants:
    header += f"  {v+' MSE':>12}  {v+' MAE':>12}"
header += "  Δ MSE    Δ MAE"
print(header)
print("    " + "-" * (len(header) - 4))

for pl in horizons:
    means = {}
    for v in variants:
        mses = [last[(pl, s, v)][0] for s in seeds if (pl, s, v) in last]
        maes = [last[(pl, s, v)][1] for s in seeds if (pl, s, v) in last]
        if mses:
            means[v] = (statistics.mean(mses), statistics.mean(maes))
    line = f"    pl={pl:<5}"
    for v in variants:
        if v in means:
            line += f"  {means[v][0]:>12.6f}  {means[v][1]:>12.6f}"
        else:
            line += f"  {'—':>12}  {'—':>12}"
    if 'Base' in means and 'MseSelect' in means:
        b_mse, b_mae = means['Base']
        m_mse, m_mae = means['MseSelect']
        dm = (m_mse - b_mse) / b_mse * 100
        da = (m_mae - b_mae) / b_mae * 100
        line += f"  {dm:+6.1f}%  {da:+6.1f}%"
    print(line)

# ── seed stability of MSE improvement at each horizon ─────────────────────
print(f"\n  MseSelect MSE improvement per seed (negative = better than Base):")
print(f"    {'horizon':<8}  " + "  ".join(f"s{s:>4}" for s in seeds) + "     mean")
print("    " + "-" * 50)
for pl in horizons:
    per_seed_deltas = []
    line = f"    pl={pl:<5}"
    for s in seeds:
        if (pl, s, 'Base') in last and (pl, s, 'MseSelect') in last:
            b = last[(pl, s, 'Base')][0]
            m = last[(pl, s, 'MseSelect')][0]
            d = (m - b) / b * 100
            per_seed_deltas.append(d)
            line += f"  {d:+5.1f}%"
        else:
            line += f"  {'—':>6}"
    if per_seed_deltas:
        line += f"    {statistics.mean(per_seed_deltas):+5.1f}%"
    print(line)
PY

# ── Drive backup (Colab) ────────────────────────────────────────────────────
if [ -d "/content/drive/MyDrive" ]; then
  STAMP=$(date +%Y%m%d_%H%M%S)
  cp "${SCRIPT_DIR}/result.txt" \
     "/content/drive/MyDrive/PatchLinear_traffic_rollout_${STAMP}.txt"
  cp -r "${LOG_DIR}" \
     "/content/drive/MyDrive/PatchLinear_traffic_rollout_logs_${STAMP}/"
  echo "Backed up to Drive (stamp: ${STAMP})"
fi

echo ""
echo "Done."
