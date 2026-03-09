#!/bin/bash

# GLPatch v8 — MULTI-SEED: TRAFFIC ONLY
# Separate script due to ~5 hours per seed.
# Run after the main multiseed script, or in parallel on another session.
#
# Estimated: ~15 hours total (3 seeds × 4 horizons × ~75 min each)

ma_type=ema
alpha=0.3
beta=0.3
model_name=GLPatch
seq_len=96

LOGDIR="./logs/multiseed"

echo ""
echo "========== [$(date '+%H:%M:%S')] TRAFFIC MULTI-SEED START =========="
echo ""

for SEED in 2022 2023 2024; do

  echo "################################################################"
  echo "  SEED ${SEED} — [$(date '+%H:%M:%S')]"
  echo "################################################################"

  sed -i "s/^fix_seed = .*/fix_seed = ${SEED}/" run.py
  echo "  Set fix_seed = ${SEED} in run.py"
  grep "^fix_seed" run.py
  echo ""

  sdir="${LOGDIR}/seed${SEED}/Traffic"
  mkdir -p ${sdir}

  for pred_len in 96 192 336 720; do
    echo ">>> [$(date '+%H:%M:%S')] Traffic seed=${SEED} pl=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path traffic.csv \
      --model_id Traffic_s${SEED}_${pred_len}_${ma_type} --model $model_name --data custom \
      --features M --seq_len $seq_len --pred_len $pred_len --enc_in 862 \
      --des 'Exp' --itr 1 --batch_size 96 --learning_rate 0.002 \
      --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pred_len}.log
  done

  echo "=== Traffic seed=${SEED} complete ==="
  for pred_len in 96 192 336 720; do
    result=$(grep "mse:" ${sdir}/${pred_len}.log | tail -1)
    echo "  pl=${pred_len}: ${result}"
  done
  echo ""

done

# Restore default seed
sed -i "s/^fix_seed = .*/fix_seed = 2021/" run.py
echo "Restored fix_seed = 2021 in run.py"

echo ""
echo "========== [$(date '+%H:%M:%S')] TRAFFIC MULTI-SEED COMPLETE =========="

# Summary
echo ""
echo "Traffic results across seeds:"
for pred_len in 96 192 336 720; do
  echo -n "  pl=${pred_len}:"
  for seed in 2021 2022 2023 2024; do
    if [ "$seed" = "2021" ]; then
      logfile="./logs/final_r2/Traffic/${pred_len}.log"
    else
      logfile="${LOGDIR}/seed${seed}/Traffic/${pred_len}.log"
    fi
    if [ -f "$logfile" ]; then
      mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
      mae=$(grep "mae:" "$logfile" | tail -1 | sed -n 's/.*mae:\([0-9.]*\).*/\1/p')
      if [ -n "$mse" ]; then
        mse_short=$(printf "%.4f" $mse)
        mae_short=$(printf "%.4f" $mae)
        echo -n "  s${seed}=(${mse_short},${mae_short})"
      else
        echo -n "  s${seed}=N/A"
      fi
    else
      echo -n "  s${seed}=---"
    fi
  done
  echo ""
done
