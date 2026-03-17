#!/bin/bash

# ============================================================
# INTERMEDIATE LR TUNING — Fix MAE regression
# ============================================================
# Problem: lr=0.0001 at longer seq_lens improved MSE but hurt MAE.
# Solution: test intermediate LRs between 0.0001 and original LR.
#
# ETTh2:   sl=512, test lr={0.0002, 0.0003, 0.0005} (original was 0.0007)
# ETTm1:   sl=336, test lr={0.0002, 0.0003, 0.0005} (original was 0.0007)
# ETTm2:   sl=720, test lr={0.00002, 0.00003, 0.00005} (original was 0.00005)
# Weather: sl=512, test lr={0.0002, 0.0003, 0.0005} (original was 0.0005)
#
# Each: 3 LRs × 4 pred_lens = 12 runs per dataset
# Total: 48 runs, estimated ~2-3 hours

ma_type=ema
alpha=0.3
beta=0.3
model_name=GLPatch
LOGDIR="./logs/lr_tuning_t14"

echo ""
echo "========== [$(date '+%H:%M:%S')] INTERMEDIATE LR TUNING =========="
echo ""

# ============================================================
# ETTh2: sl=512, lr={0.0002, 0.0003, 0.0005}
# Known: lr=0.0001 → MSE good, MAE bad
#        lr=0.0007 → MAE good at sl=96
# ============================================================
for lr in 0.0002 0.0003 0.0005; do
  sdir="${LOGDIR}/ETTh2/sl512_lr${lr}"
  mkdir -p ${sdir}

  echo ">>> [$(date '+%H:%M:%S')] ETTh2 sl=512 lr=${lr}"
  for pred_len in 96 192 336 720; do
    echo "  [$(date '+%H:%M:%S')] ETTh2 sl=512 lr=${lr} pl=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path ETTh2.csv \
      --model_id lrt_ETTh2_sl512_lr${lr}_${pred_len} --model $model_name --data ETTh2 \
      --features M --seq_len 512 --pred_len $pred_len --enc_in 7 \
      --des 'Exp' --itr 1 --batch_size 2048 --learning_rate $lr \
      --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pred_len}.log
  done

  echo "  === ETTh2 sl=512 lr=${lr} complete ==="
  for pred_len in 96 192 336 720; do
    result=$(grep "mse:\|mae:" ${sdir}/${pred_len}.log | tail -1)
    echo "    pl=${pred_len}: ${result}"
  done
  echo ""
done

# ============================================================
# ETTm1: sl=336, lr={0.0002, 0.0003, 0.0005}
# Known: lr=0.0001 → MSE 0.2798/0.3168/0.3547/0.4189
#        lr=0.0007 → MAE better at sl=336
# ============================================================
for lr in 0.0002 0.0003 0.0005; do
  sdir="${LOGDIR}/ETTm1/sl336_lr${lr}"
  mkdir -p ${sdir}

  echo ">>> [$(date '+%H:%M:%S')] ETTm1 sl=336 lr=${lr}"
  for pred_len in 96 192 336 720; do
    echo "  [$(date '+%H:%M:%S')] ETTm1 sl=336 lr=${lr} pl=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path ETTm1.csv \
      --model_id lrt_ETTm1_sl336_lr${lr}_${pred_len} --model $model_name --data ETTm1 \
      --features M --seq_len 336 --pred_len $pred_len --enc_in 7 \
      --des 'Exp' --itr 1 --batch_size 2048 --learning_rate $lr \
      --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pred_len}.log
  done

  echo "  === ETTm1 sl=336 lr=${lr} complete ==="
  for pred_len in 96 192 336 720; do
    result=$(grep "mse:\|mae:" ${sdir}/${pred_len}.log | tail -1)
    echo "    pl=${pred_len}: ${result}"
  done
  echo ""
done

# ============================================================
# ETTm2: sl=720, lr={0.00002, 0.00003, 0.00005}
# Known: lr=0.0001 → MSE 0.1501/0.2115/0.2666/0.3361
#        lr=0.00005 → MAE was better at sl=336/512
# This dataset uses much lower LRs — test finer range
# ============================================================
for lr in 0.00002 0.00003 0.00005; do
  sdir="${LOGDIR}/ETTm2/sl720_lr${lr}"
  mkdir -p ${sdir}

  echo ">>> [$(date '+%H:%M:%S')] ETTm2 sl=720 lr=${lr}"
  for pred_len in 96 192 336 720; do
    echo "  [$(date '+%H:%M:%S')] ETTm2 sl=720 lr=${lr} pl=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path ETTm2.csv \
      --model_id lrt_ETTm2_sl720_lr${lr}_${pred_len} --model $model_name --data ETTm2 \
      --features M --seq_len 720 --pred_len $pred_len --enc_in 7 \
      --des 'Exp' --itr 1 --batch_size 2048 --learning_rate $lr \
      --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pred_len}.log
  done

  echo "  === ETTm2 sl=720 lr=${lr} complete ==="
  for pred_len in 96 192 336 720; do
    result=$(grep "mse:\|mae:" ${sdir}/${pred_len}.log | tail -1)
    echo "    pl=${pred_len}: ${result}"
  done
  echo ""
done

# ============================================================
# Weather: sl=512, lr={0.0002, 0.0003, 0.0005}
# Known: lr=0.0001 → MSE 0.1471/0.1887/0.2215/0.2953
#        lr=0.0005 → was better at sl=336 for some settings
# ============================================================
for lr in 0.0002 0.0003 0.0005; do
  sdir="${LOGDIR}/Weather/sl512_lr${lr}"
  mkdir -p ${sdir}

  echo ">>> [$(date '+%H:%M:%S')] Weather sl=512 lr=${lr}"
  for pred_len in 96 192 336 720; do
    echo "  [$(date '+%H:%M:%S')] Weather sl=512 lr=${lr} pl=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path weather.csv \
      --model_id lrt_Weather_sl512_lr${lr}_${pred_len} --model $model_name --data custom \
      --features M --seq_len 512 --pred_len $pred_len --enc_in 21 \
      --des 'Exp' --itr 1 --batch_size 1024 --learning_rate $lr \
      --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pred_len}.log
  done

  echo "  === Weather sl=512 lr=${lr} complete ==="
  for pred_len in 96 192 336 720; do
    result=$(grep "mse:\|mae:" ${sdir}/${pred_len}.log | tail -1)
    echo "    pl=${pred_len}: ${result}"
  done
  echo ""
done

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "========== [$(date '+%H:%M:%S')] LR TUNING COMPLETE =========="
echo ""
echo "xPatch T14 reference values (MSE/MAE):"
echo "  ETTh2:   96=(0.226/0.297) 192=(0.275/0.330) 336=(0.312/0.360) 720=(0.384/0.418)"
echo "  ETTm1:   96=(0.275/0.330) 192=(0.315/0.355) 336=(0.355/0.376) 720=(0.419/0.411)"
echo "  ETTm2:   96=(0.153/0.240) 192=(0.213/0.280) 336=(0.264/0.315) 720=(0.338/0.363)"
echo "  Weather: 96=(0.146/0.185) 192=(0.189/0.227) 336=(0.218/0.260) 720=(0.291/0.315)"
echo ""

for ds in ETTh2 ETTm1 ETTm2 Weather; do
  echo "${ds}:"
  # Determine configs
  if [ "$ds" = "ETTh2" ]; then
    sl=512; lrs="0.0001 0.0002 0.0003 0.0005 0.0007"
  elif [ "$ds" = "ETTm1" ]; then
    sl=336; lrs="0.0001 0.0002 0.0003 0.0005 0.0007"
  elif [ "$ds" = "ETTm2" ]; then
    sl=720; lrs="0.00002 0.00003 0.00005 0.0001"
  elif [ "$ds" = "Weather" ]; then
    sl=512; lrs="0.0001 0.0002 0.0003 0.0005"
  fi

  echo "  sl=${sl}:"
  printf "  %-12s|" "lr"
  for pl in 96 192 336 720; do
    printf "  %8s %8s" "${pl}mse" "${pl}mae"
  done
  echo ""
  echo "  ------------|$(printf '  -------- --------%.0s' 1 2 3 4)"

  for lr in $lrs; do
    # Check both table14 logs and lr_tuning logs
    logdir1="./logs/table14/sl${sl}/${ds}"
    logdir2="./logs/lr_tuning_t14/${ds}/sl${sl}_lr${lr}"

    printf "  %-12s|" "lr=${lr}"
    for pl in 96 192 336 720; do
      # Try lr_tuning dir first, then table14 dir
      found=0
      for logdir in "$logdir2" "$logdir1"; do
        logfile="${logdir}/${pl}.log"
        if [ -f "$logfile" ]; then
          mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
          mae=$(grep "mae:" "$logfile" | tail -1 | sed -n 's/.*mae:\([0-9.]*\).*/\1/p')
          if [ -n "$mse" ] && [ -n "$mae" ]; then
            printf "  %8s %8s" $(printf "%.4f" $mse) $(printf "%.4f" $mae)
            found=1
            break
          fi
        fi
      done
      if [ $found -eq 0 ]; then
        printf "      N/A      N/A"
      fi
    done
    echo ""
  done
  echo ""
done

echo "Log files: ${LOGDIR}/<dataset>/sl<SL>_lr<LR>/<pred_len>.log"
