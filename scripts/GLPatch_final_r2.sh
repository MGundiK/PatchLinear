#!/bin/bash

# GLPatch v8 — FINAL BENCHMARK (R2 updates)
# Now with per-run log files + result extraction from logs (not result.txt)
#
# Changes from R1 final:
#   ETTh2:       0.0005 → 0.0007
#   ETTm1:       0.0005 → 0.0007
#   ETTm2:       0.0001 → 0.00005
#   Traffic:     0.005  → 0.002
#   Solar:       0.005  → 0.01
#
# Unchanged (use existing results):
#   ETTh1:       0.0005
#   Weather:     0.0005
#   Exchange:    0.000005
#   ILI:         0.02
#   Electricity: 0.005
#
# Logs saved to: ./logs/final_r2/<dataset>/<pred_len>.log

ma_type=ema
alpha=0.3
beta=0.3
model_name=GLPatch
seq_len=96

LOGDIR="./logs/final_r2"

echo ""
echo "========== [$(date '+%H:%M:%S')] STARTING FINAL BENCHMARK R2 =========="
echo "Logs: ${LOGDIR}/<dataset>/<pred_len>.log"
echo ""

# ================================================================
# ETTh2 — lr=0.0007 (was 0.0005)
# ================================================================
ds="ETTh2"; lr=0.0007
mkdir -p ${LOGDIR}/${ds}
echo "========== [$(date '+%H:%M:%S')] ${ds} lr=${lr} =========="
for pred_len in 96 192 336 720; do
  echo ">>> [$(date '+%H:%M:%S')] ${ds} pred_len=${pred_len}"
  python -u run.py \
    --is_training 1 --root_path ./dataset/ --data_path ETTh2.csv \
    --model_id ${ds}_final_${pred_len}_${ma_type} --model $model_name --data ETTh2 \
    --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 \
    --des 'Exp' --itr 1 --batch_size 2048 --learning_rate $lr \
    --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
    --use_amp --num_workers 2 \
    2>&1 | tee ${LOGDIR}/${ds}/${pred_len}.log
done
echo "=== ${ds} complete ==="
for pred_len in 96 192 336 720; do
  result=$(grep "mse:" ${LOGDIR}/${ds}/${pred_len}.log | tail -1)
  echo "  ${ds} pl=${pred_len}: ${result}"
done
echo ""

# ================================================================
# ETTm1 — lr=0.0007 (was 0.0005)
# ================================================================
ds="ETTm1"; lr=0.0007
mkdir -p ${LOGDIR}/${ds}
echo "========== [$(date '+%H:%M:%S')] ${ds} lr=${lr} =========="
for pred_len in 96 192 336 720; do
  echo ">>> [$(date '+%H:%M:%S')] ${ds} pred_len=${pred_len}"
  python -u run.py \
    --is_training 1 --root_path ./dataset/ --data_path ETTm1.csv \
    --model_id ${ds}_final_${pred_len}_${ma_type} --model $model_name --data ETTm1 \
    --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 \
    --des 'Exp' --itr 1 --batch_size 2048 --learning_rate $lr \
    --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
    --use_amp --num_workers 2 \
    2>&1 | tee ${LOGDIR}/${ds}/${pred_len}.log
done
echo "=== ${ds} complete ==="
for pred_len in 96 192 336 720; do
  result=$(grep "mse:" ${LOGDIR}/${ds}/${pred_len}.log | tail -1)
  echo "  ${ds} pl=${pred_len}: ${result}"
done
echo ""

# ================================================================
# ETTm2 — lr=0.00005 (was 0.0001)
# ================================================================
ds="ETTm2"; lr=0.00005
mkdir -p ${LOGDIR}/${ds}
echo "========== [$(date '+%H:%M:%S')] ${ds} lr=${lr} =========="
for pred_len in 96 192 336 720; do
  echo ">>> [$(date '+%H:%M:%S')] ${ds} pred_len=${pred_len}"
  python -u run.py \
    --is_training 1 --root_path ./dataset/ --data_path ETTm2.csv \
    --model_id ${ds}_final_${pred_len}_${ma_type} --model $model_name --data ETTm2 \
    --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 \
    --des 'Exp' --itr 1 --batch_size 2048 --learning_rate $lr \
    --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
    --use_amp --num_workers 2 \
    2>&1 | tee ${LOGDIR}/${ds}/${pred_len}.log
done
echo "=== ${ds} complete ==="
for pred_len in 96 192 336 720; do
  result=$(grep "mse:" ${LOGDIR}/${ds}/${pred_len}.log | tail -1)
  echo "  ${ds} pl=${pred_len}: ${result}"
done
echo ""

# ================================================================
# Solar — lr=0.01 (was 0.005)
# ================================================================
ds="Solar"; lr=0.01
mkdir -p ${LOGDIR}/${ds}
echo "========== [$(date '+%H:%M:%S')] ${ds} lr=${lr} =========="
for pred_len in 96 192 336 720; do
  echo ">>> [$(date '+%H:%M:%S')] ${ds} pred_len=${pred_len}"
  python -u run.py \
    --is_training 1 --root_path ./dataset/ --data_path solar.txt \
    --model_id ${ds}_final_${pred_len}_${ma_type} --model $model_name --data Solar \
    --features M --seq_len $seq_len --pred_len $pred_len --enc_in 137 \
    --des 'Exp' --itr 1 --batch_size 512 --learning_rate $lr \
    --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
    --use_amp --num_workers 2 \
    2>&1 | tee ${LOGDIR}/${ds}/${pred_len}.log
done
echo "=== ${ds} complete ==="
for pred_len in 96 192 336 720; do
  result=$(grep "mse:" ${LOGDIR}/${ds}/${pred_len}.log | tail -1)
  echo "  ${ds} pl=${pred_len}: ${result}"
done
echo ""

# ================================================================
# Traffic — lr=0.002 (was 0.005)
# NOTE: Slowest dataset — run last
# ================================================================
ds="Traffic"; lr=0.002
mkdir -p ${LOGDIR}/${ds}
echo "========== [$(date '+%H:%M:%S')] ${ds} lr=${lr} =========="
for pred_len in 96 192 336 720; do
  echo ">>> [$(date '+%H:%M:%S')] ${ds} pred_len=${pred_len}"
  python -u run.py \
    --is_training 1 --root_path ./dataset/ --data_path traffic.csv \
    --model_id ${ds}_final_${pred_len}_${ma_type} --model $model_name --data custom \
    --features M --seq_len $seq_len --pred_len $pred_len --enc_in 862 \
    --des 'Exp' --itr 1 --batch_size 96 --learning_rate $lr \
    --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
    --use_amp --num_workers 2 \
    2>&1 | tee ${LOGDIR}/${ds}/${pred_len}.log
done
echo "=== ${ds} complete ==="
for pred_len in 96 192 336 720; do
  result=$(grep "mse:" ${LOGDIR}/${ds}/${pred_len}.log | tail -1)
  echo "  ${ds} pl=${pred_len}: ${result}"
done
echo ""

# ================================================================
# FINAL SUMMARY — extract all results from log files
# ================================================================
echo ""
echo "========== [$(date '+%H:%M:%S')] FINAL BENCHMARK R2 COMPLETE =========="
echo ""
echo "RESULTS SUMMARY (extracted from log files):"
echo "============================================"
for ds in ETTh2 ETTm1 ETTm2 Solar Traffic; do
  echo ""
  echo "${ds}:"
  for pred_len in 96 192 336 720; do
    logfile="${LOGDIR}/${ds}/${pred_len}.log"
    if [ -f "$logfile" ]; then
      result=$(grep "mse:" "$logfile" | tail -1)
      if [ -n "$result" ]; then
        echo "  pl=${pred_len}: ${result}"
      else
        echo "  pl=${pred_len}: *** NO RESULT FOUND IN LOG ***"
      fi
    else
      echo "  pl=${pred_len}: *** LOG FILE MISSING ***"
    fi
  done
done

echo ""
echo "FINAL LR MAP (for paper):"
echo "  ETTh1:       0.0005   (unchanged)"
echo "  ETTh2:       0.0007   (above)"
echo "  ETTm1:       0.0007   (above)"
echo "  ETTm2:       0.00005  (above)"
echo "  Weather:     0.0005   (unchanged)"
echo "  Exchange:    0.000005 (unchanged)"
echo "  ILI:         0.02     (unchanged)"
echo "  Electricity: 0.005    (unchanged)"
echo "  Traffic:     0.002    (above)"
echo "  Solar:       0.01     (above)"
echo ""
echo "Log files: ${LOGDIR}/<dataset>/<pred_len>.log"
