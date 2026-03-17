#!/bin/bash

# ============================================================
# MULTI-SEED AT BEST TABLE 14 CONFIGS
# ============================================================
# Runs 3 additional seeds (2022, 2023, 2024) at the best config
# per dataset. Seed 2021 is already done from previous runs.
# Final paper: report best 3 of 4 seeds average.
#
# Best configs (incorporating LR tuning results):
#   ETTh1:       sl=512, lr=0.0001,   batch=2048
#   ETTh2:       sl=720, lr=0.0001,   batch=2048
#   ETTm1:       sl=336, lr=0.0001,   batch=2048
#   ETTm2:       sl=720, lr=0.00005,  batch=2048
#   Weather:     sl=512, lr=0.0003,   batch=1024  ← IMPROVED LR
#   Traffic:     sl=720, lr=0.005,    batch=46
#   Electricity: sl=512, lr=0.001,    batch=192
#   Exchange:    sl=96,  lr=0.000005, batch=32
#   Solar:       sl=720, lr=0.005,    batch=256
#   ILI:         sl=36,  lr=0.01,     ma_type=reg, batch=32
#
# Estimated runtime:
#   Small datasets (ETT×4, Weather, Exchange): ~4 hrs per seed × 3 = ~12 hrs
#   Large datasets (Traffic, Electricity, Solar): ~6 hrs per seed × 3 = ~18 hrs
#   ILI: ~15 min per seed × 3 = ~45 min
#   Total: ~30 hrs (run in parallel where possible)
#
# IMPORTANT: This script modifies fix_seed in run.py using sed.
# It restores to 2021 at the end.

LOGDIR="./logs/multiseed_t14"
mkdir -p ${LOGDIR}

alpha=0.3
beta=0.3
model_name=GLPatch

echo ""
echo "========== [$(date '+%H:%M:%S')] MULTI-SEED TABLE 14 — ALL DATASETS =========="
echo ""

for seed in 2022 2023 2024; do
  echo ""
  echo "################################################################"
  echo "  SEED = ${seed} — [$(date '+%H:%M:%S')]"
  echo "################################################################"
  echo ""

  # Set seed in run.py
  sed -i "s/fix_seed = [0-9]*/fix_seed = ${seed}/" run.py
  echo "  Set fix_seed = ${seed}"

  
  # ============================================================
  # Traffic: sl=720, lr=0.005
  # ============================================================
  ds=Traffic; sl=720; lr=0.005; batch=46; ma=ema
  sdir="${LOGDIR}/${ds}/seed${seed}"
  mkdir -p ${sdir}
  echo ">>> [$(date '+%H:%M:%S')] ${ds} seed=${seed} sl=${sl} lr=${lr}"
  for pl in 96 192 336 720; do
    echo "  [$(date '+%H:%M:%S')] ${ds} seed=${seed} pl=${pl}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path traffic.csv \
      --model_id ms_${ds}_s${seed}_${pl} --model $model_name --data custom \
      --features M --seq_len $sl --pred_len $pl --enc_in 862 \
      --des 'Exp' --itr 1 --batch_size $batch --learning_rate $lr \
      --lradj 'sigmoid' --ma_type $ma --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pl}.log
  done
  echo "  === ${ds} seed=${seed} complete ==="
  for pl in 96 192 336 720; do
    result=$(grep "mse:" ${sdir}/${pl}.log | tail -1)
    echo "    pl=${pl}: ${result}"
  done

  echo ""
  echo "========== SEED ${seed} ALL DATASETS COMPLETE [$(date '+%H:%M:%S')] =========="
  echo ""
done

# Restore seed
sed -i "s/fix_seed = [0-9]*/fix_seed = 2021/" run.py
echo "Restored fix_seed = 2021"

# ============================================================
# PRINT ALL RESULTS
# ============================================================
echo ""
echo "========== MULTI-SEED RESULTS SUMMARY =========="
echo ""

for ds in  Traffic; do
  if [ "$ds" = "ILI" ]; then
    pls="24 36 48 60"
  else
    pls="96 192 336 720"
  fi

  echo "${ds}:"
  echo "  seed  |$(for pl in $pls; do printf "  %8s" "pl=${pl}"; done)"
  echo "  ------+$(for pl in $pls; do printf "  --------"; done)"

  for seed in 2021 2022 2023 2024; do
    sdir="${LOGDIR}/${ds}/seed${seed}"
    printf "  %5s |" $seed
    for pl in $pls; do
      logfile="${sdir}/${pl}.log"
      if [ -f "$logfile" ]; then
        mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
        [ -n "$mse" ] && printf "  %8s" $(printf "%.4f" $mse) || printf "      N/A"
      else
        # Seed 2021 results are in table14 logs, not multiseed
        printf "  (prev) "
      fi
    done
    echo ""
  done
  echo ""
done

echo ""
echo "NOTE: Seed 2021 results are from previous table14 runs (not in ${LOGDIR})."
echo "Combine seed 2021 from those logs with seeds 2022-2024 from this run."
echo "Then compute best-3-of-4 average for final paper values."
