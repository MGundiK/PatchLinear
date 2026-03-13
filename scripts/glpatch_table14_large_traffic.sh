#!/bin/bash

# ============================================================
# GLPatch TABLE 14 — LARGE DATASETS
# ============================================================
# Matches xPatch Table 14 configs:
#   Traffic:     sl=720, lr=0.005, batch=46
#   Electricity: sl=720, lr=0.001, batch=128
#   Solar:       sl=720, lr=0.005, batch=256
#
# Also searches sl={336,512,720} to find GLPatch's best.
# Estimated: ~15-20 hours total

ma_type=ema
alpha=0.3
beta=0.3
model_name=GLPatch
LOGDIR="./logs/table14"

echo ""
echo "========== [$(date '+%H:%M:%S')] GLPatch TABLE 14 — LARGE DATASETS =========="
echo ""




# ============================================================
# Traffic: xPatch T14 uses sl=720, lr=0.005, batch=46
# NOTE: This is VERY slow (~3+ hours per seq_len)
# ============================================================
for sl in 336 512 720; do
  sdir="${LOGDIR}/sl${sl}/Traffic"
  mkdir -p ${sdir}

  if [ $sl -le 336 ]; then batch=96;
  elif [ $sl -le 512 ]; then batch=64;
  else batch=46; fi

  echo ">>> [$(date '+%H:%M:%S')] Traffic sl=${sl} lr=0.005 batch=${batch}"
  for pred_len in 96 192 336 720; do
    echo "  [$(date '+%H:%M:%S')] Traffic sl=${sl} pl=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path traffic.csv \
      --model_id t14_Traffic_sl${sl}_${pred_len}_${ma_type} --model $model_name --data custom \
      --features M --seq_len $sl --pred_len $pred_len --enc_in 862 \
      --des 'Exp' --itr 1 --batch_size $batch --learning_rate 0.005 \
      --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pred_len}.log
  done

  echo "  === Traffic sl=${sl} complete ==="
  for pred_len in 96 192 336 720; do
    result=$(grep "mse:" ${sdir}/${pred_len}.log | tail -1)
    echo "    pl=${pred_len}: ${result}"
  done
  echo ""
done

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "========== [$(date '+%H:%M:%S')] TABLE 14 LARGE DATASETS COMPLETE =========="
echo ""

for ds in Traffic; do
  echo "${ds}:"
  echo "  seq_len |     96      192      336      720"
  echo "  --------+------------------------------------"
  for sl in 336 512 720; do
    dir="${LOGDIR}/sl${sl}/${ds}"
    printf "  %6d  |" $sl
    for pred_len in 96 192 336 720; do
      logfile="${dir}/${pred_len}.log"
      if [ -f "$logfile" ]; then
        mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
        [ -n "$mse" ] && printf "  %7s" $(printf "%.4f" $mse) || printf "     N/A"
      else
        printf "     N/A"
      fi
    done
    echo ""
  done
  echo ""
done
