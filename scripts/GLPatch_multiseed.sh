#!/bin/bash

# GLPatch v8 — MULTI-SEED VERIFICATION (all datasets except Traffic)
# Runs 3 additional seeds (2022, 2023, 2024) for all datasets.
# Seed 2021 is the default (already run in final_r2).
#
# Uses sed to change fix_seed automatically — no manual editing needed.
#
# Estimated runtime per seed (~1.5-2 hours):
#   ETTh1:       ~5 min
#   ETTh2:       ~5 min
#   ETTm1:       ~10 min
#   ETTm2:       ~15 min
#   Weather:     ~20 min
#   Exchange:    ~5 min
#   ILI:         ~2 min
#   Electricity: ~30 min
#   Solar:       ~45 min
# Total: ~4.5-6 hours for 3 seeds

ma_type=ema
alpha=0.3
beta=0.3
model_name=GLPatch
seq_len=96

LOGDIR="./logs/multiseed"

# ============================================================
# FUNCTION: run one dataset at one seed
# ============================================================
run_dataset() {
  local ds=$1
  local lr=$2
  local data_path=$3
  local data_flag=$4
  local enc_in=$5
  local batch=$6
  local seed=$7
  local lradj="${8:-sigmoid}"
  local extra_args="${9:-}"

  local sdir="${LOGDIR}/seed${seed}/${ds}"
  mkdir -p ${sdir}

  local pred_lens="96 192 336 720"
  local sl=$seq_len
  if [ "$ds" = "ILI" ]; then
    pred_lens="24 36 48 60"
    sl=36
  fi

  echo ">>> [$(date '+%H:%M:%S')] ${ds} seed=${seed} lr=${lr}"
  for pred_len in $pred_lens; do
    echo "  [$(date '+%H:%M:%S')] ${ds} seed=${seed} pl=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path ${data_path} \
      --model_id ${ds}_s${seed}_${pred_len}_${ma_type} --model $model_name --data ${data_flag} \
      --features M --seq_len $sl --pred_len $pred_len --enc_in $enc_in \
      --des 'Exp' --itr 1 --batch_size $batch --learning_rate $lr \
      --lradj ${lradj} --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      ${extra_args} \
      2>&1 | tee ${sdir}/${pred_len}.log
  done

  echo "  === ${ds} seed=${seed} complete ==="
  for pred_len in $pred_lens; do
    result=$(grep "mse:" ${sdir}/${pred_len}.log | tail -1)
    echo "    pl=${pred_len}: ${result}"
  done
  echo ""
}

# ============================================================
# MAIN
# ============================================================
echo ""
echo "========== [$(date '+%H:%M:%S')] MULTI-SEED BENCHMARK START =========="
echo "Datasets: ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange ILI Electricity Solar"
echo "Seeds: 2022, 2023, 2024 (seed 2021 already done)"
echo "Logs: ${LOGDIR}/seed<N>/<dataset>/<pred_len>.log"
echo ""

for SEED in 2022 2023 2024; do

  echo ""
  echo "################################################################"
  echo "  SEED ${SEED} — [$(date '+%H:%M:%S')]"
  echo "################################################################"

  # Change seed in run.py automatically
  sed -i "s/^fix_seed = .*/fix_seed = ${SEED}/" run.py
  echo "  Set fix_seed = ${SEED} in run.py"
  grep "^fix_seed" run.py
  echo ""

  # ---- ETT datasets (fast) ----
  run_dataset "ETTh1" 0.0005 "ETTh1.csv" "ETTh1" 7 2048 $SEED
  run_dataset "ETTh2" 0.0007 "ETTh2.csv" "ETTh2" 7 2048 $SEED
  run_dataset "ETTm1" 0.0007 "ETTm1.csv" "ETTm1" 7 2048 $SEED
  run_dataset "ETTm2" 0.00005 "ETTm2.csv" "ETTm2" 7 2048 $SEED

  # ---- Weather (medium) ----
  run_dataset "Weather" 0.0005 "weather.csv" "custom" 21 2048 $SEED

  # ---- Exchange (fast) ----
  run_dataset "Exchange" 0.000005 "exchange_rate.csv" "custom" 8 32 $SEED

  # ---- ILI (fast, different config) ----
  run_dataset "ILI" 0.02 "national_illness.csv" "custom" 7 32 $SEED \
    "type3" "--label_len 18 --patch_len 6 --stride 3"

  # ---- Electricity (medium-slow) ----
  run_dataset "Electricity" 0.005 "electricity.csv" "custom" 321 256 $SEED

  # ---- Solar (medium-slow) ----
  run_dataset "Solar" 0.01 "solar.txt" "Solar" 137 512 $SEED

  echo ""
  echo "========== [$(date '+%H:%M:%S')] SEED ${SEED} COMPLETE =========="
  echo ""

done

# Restore default seed
sed -i "s/^fix_seed = .*/fix_seed = 2021/" run.py
echo "Restored fix_seed = 2021 in run.py"

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "================================================================"
echo "  MULTI-SEED RESULTS SUMMARY (all datasets except Traffic)"
echo "================================================================"

for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange ILI Electricity Solar; do
  echo ""
  echo "${ds}:"

  if [ "$ds" = "ILI" ]; then
    pred_lens="24 36 48 60"
  else
    pred_lens="96 192 336 720"
  fi

  for pred_len in $pred_lens; do
    echo -n "  pl=${pred_len}:"
    for seed in 2021 2022 2023 2024; do
      if [ "$seed" = "2021" ]; then
        logfile="./logs/final_r2/${ds}/${pred_len}.log"
      else
        logfile="${LOGDIR}/seed${seed}/${ds}/${pred_len}.log"
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
done

echo ""
echo "================================================================"
echo "  DONE — 9 datasets × 3 seeds × 4 horizons = 108 runs"
echo "  Log files: ${LOGDIR}/seed<SEED>/<dataset>/<pred_len>.log"
echo "================================================================"
