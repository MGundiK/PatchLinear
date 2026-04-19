#!/usr/bin/env bash
# Zero-shot evaluation only
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG_DIR="logs/zeroshot"
mkdir -p "$LOG_DIR"
> result_zeroshot.txt

TRANSFERS=(
  "ETTh1 ETTh2"
  "ETTh1 ETTm2"
  "ETTh2 ETTh1"
  "ETTm1 ETTh2"
  "ETTm1 ETTm2"
  "ETTm2 ETTm1"
)

for PAIR in "${TRANSFERS[@]}"; do
  SRC=$(echo "$PAIR"|cut -d' ' -f1)
  TGT=$(echo "$PAIR"|cut -d' ' -f2)
  echo ">>> ${SRC} → ${TGT}"
  for PL in 96 192 336 720; do
    LOG="${LOG_DIR}/eval_${SRC}_to_${TGT}_pl${PL}.log"
    echo "  Eval ${SRC}→${TGT} pl=${PL}"
    python eval_zeroshot.py \
      --src "${SRC}" --tgt "${TGT}" --pred_len "${PL}" \
      > "$LOG" 2>&1 || true   # don't abort on error
    # Show result or first error line
    if grep -q "mse:" "$LOG"; then
      grep "mse:" "$LOG" | tail -1
    else
      echo "  [FAILED] $(head -5 $LOG | tail -1)"
    fi
  done
done

echo ""
echo "=== ZERO-SHOT SUMMARY ==="
python3 - << 'PYEOF'
import re, os, numpy as np
log_dir = "logs/zeroshot"
transfers = [
    ("ETTh1","ETTh2"),("ETTh1","ETTm2"),("ETTh2","ETTh1"),
    ("ETTm1","ETTh2"),("ETTm1","ETTm2"),("ETTm2","ETTm1"),
]
for src,tgt in transfers:
    mse_all=[]; mae_all=[]
    for pl in [96,192,336,720]:
        f=f"{log_dir}/eval_{src}_to_{tgt}_pl{pl}.log"
        if not os.path.exists(f): continue
        for line in open(f):
            m=re.search(r'mse:([\d.]+)\s+mae:([\d.]+)',line)
            if m: mse_all.append(float(m.group(1))); mae_all.append(float(m.group(2)))
    if mse_all:
        print(f"  {src}→{tgt}: MSE={np.mean(mse_all):.3f} MAE={np.mean(mae_all):.3f}")
    else:
        print(f"  {src}→{tgt}: no results")
PYEOF
