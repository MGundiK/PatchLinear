"""
run_m4.py — M4 Short-Term Forecasting for PatchLinear

Two usage modes:
  # Run a single frequency (called by shell script, one call per freq):
  python run_m4.py --seasonal_patterns Monthly

  # Run all 6 frequencies in sequence (standalone):
  python run_m4.py

Setup:
  1. exp/exp_m4.py        — M4 experiment class
  2. utils/m4_summary.py  — SMAPE / OWA evaluator
  3. utils/losses.py      — smape_loss / mase_loss
  4. dataset/m4/          — training.npz, test.npz, M4-info.csv,
                            submission-Naive2.csv
"""

import argparse, os, random
import numpy as np
import torch

# ── argument parser ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()

parser.add_argument('--seed',          type=int,   default=2021)
parser.add_argument('--root_path',     type=str,   default='./dataset/m4/')
parser.add_argument('--checkpoints',   type=str,   default='./checkpoints/m4/')
parser.add_argument('--forecast_dir',  type=str,   default='./results/m4_forecasts/')
parser.add_argument('--train_epochs',  type=int,   default=30)
parser.add_argument('--batch_size',    type=int,   default=16)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--num_workers',   type=int,   default=4)
parser.add_argument('--gpu',           type=int,   default=0)
parser.add_argument('--use_gpu',       type=bool,  default=True)

# Accepted but ignored — lets shell script pass --model without crashing
parser.add_argument('--model',         type=str,   default='PatchLinear')

# If given: run only this frequency. If omitted: run all 6.
parser.add_argument('--seasonal_patterns', type=str, default=None,
                    help='Single M4 frequency (Yearly/Quarterly/Monthly/'
                         'Weekly/Daily/Hourly). Omit to run all 6.')

# Model arch
parser.add_argument('--d_model',       type=int,   default=64)
parser.add_argument('--t_ff',          type=int,   default=128)
parser.add_argument('--c_ff',          type=int,   default=16)
parser.add_argument('--patch_len',     type=int,   default=8)
parser.add_argument('--stride',        type=int,   default=4)
parser.add_argument('--dw_kernel',     type=int,   default=3)
parser.add_argument('--alpha',         type=float, default=0.3)
parser.add_argument('--t_dropout',     type=float, default=0.1)
parser.add_argument('--c_dropout',     type=float, default=0.1)
parser.add_argument('--embed_dropout', type=float, default=0.1)
parser.add_argument('--head_dropout',  type=float, default=0.0)

# Ablation flags
# Decomp + both streams + fusion ON: M4 has trend (esp. Yearly/Quarterly)
# Cross-channel + alpha OFF: M4 is univariate (C=1), these do nothing
parser.add_argument('--use_decomp',        type=int, default=1)
parser.add_argument('--use_trend_stream',  type=int, default=1)
parser.add_argument('--use_seas_stream',   type=int, default=1)
parser.add_argument('--use_fusion_gate',   type=int, default=1)
parser.add_argument('--use_cross_channel', type=int, default=0)
parser.add_argument('--use_alpha_gate',    type=int, default=0)
parser.add_argument('--use_reparam',       type=int, default=0)

args = parser.parse_args()

# ── seeds & flag conversion ───────────────────────────────────────────────────
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
for f in ['use_decomp','use_trend_stream','use_seas_stream',
          'use_fusion_gate','use_cross_channel','use_alpha_gate','use_reparam']:
    setattr(args, f, bool(getattr(args, f)))
args.use_gpu = args.use_gpu and torch.cuda.is_available()

# ── imports ───────────────────────────────────────────────────────────────────
from exp.exp_m4 import Exp_M4
from utils.m4_summary import M4Summary

# ── frequency table ───────────────────────────────────────────────────────────
ALL_FREQUENCIES = [
    ('Yearly',    6),
    ('Quarterly', 8),
    ('Monthly',  18),
    ('Weekly',   13),
    ('Daily',    14),
    ('Hourly',   48),
]

if args.seasonal_patterns is not None:
    pred_map = dict(ALL_FREQUENCIES)
    if args.seasonal_patterns not in pred_map:
        raise ValueError(f"Unknown frequency '{args.seasonal_patterns}'. "
                         f"Choose from: {list(pred_map)}")
    frequencies_to_run = [(args.seasonal_patterns, pred_map[args.seasonal_patterns])]
else:
    frequencies_to_run = ALL_FREQUENCIES

os.makedirs('logs/m4', exist_ok=True)
os.makedirs(args.forecast_dir, exist_ok=True)

print("=" * 65)
print("PatchLinear — M4 Short-Term Forecasting")
print(f"Config: d={args.d_model}, p={args.patch_len}, k={args.dw_kernel}, "
      f"lr={args.learning_rate}, epochs={args.train_epochs}")
mode = f"single frequency ({args.seasonal_patterns})" if args.seasonal_patterns else "all 6 frequencies"
print(f"Mode: {mode}")
print("=" * 65)

# ── train + predict ───────────────────────────────────────────────────────────
for freq_name, pred_len in frequencies_to_run:
    seq_len = pred_len * 2          # standard M4, no arbitrary floor
    print(f"\n>>> {freq_name}  pred_len={pred_len}  seq_len={seq_len}")

    args.seasonal_patterns = freq_name
    args.pred_len  = pred_len
    args.seq_len   = seq_len
    args.label_len = 0

    setting = (f"M4_{freq_name}_pl{pred_len}"
               f"_d{args.d_model}_p{args.patch_len}_k{args.dw_kernel}"
               f"_s{args.seed}")

    exp = Exp_M4(args)
    exp.train(setting)
    exp.predict(setting, args.forecast_dir)

# ── evaluate (only frequencies with forecast CSVs present) ───────────────────
ready = [f for f, _ in ALL_FREQUENCIES
         if os.path.exists(os.path.join(args.forecast_dir, f'{f}_forecast.csv'))]

if not ready:
    print("\nNo forecast CSVs found — skipping evaluation.")
else:
    print(f"\n{'='*65}")
    print(f"EVALUATION — frequencies ready: {ready}")
    print(f"{'='*65}")

    summary = M4Summary(
        file_path=args.forecast_dir + os.sep,
        root_path=args.root_path,
    )
    smapes, owas, _, _ = summary.evaluate()

    print(f"\n{'Frequency':<14} {'SMAPE':>8} {'OWA':>8}")
    print("-" * 34)
    for key in ['Yearly', 'Quarterly', 'Monthly', 'Others', 'Average']:
        s = smapes.get(key, float('nan'))
        o = owas.get(key,   float('nan'))
        print(f"{key:<14} {s:>8.3f} {o:>8.3f}")

    results_path = 'logs/m4/results.txt'
    mode = 'a' if os.path.exists(results_path) else 'w'
    with open(results_path, mode) as f:
        if mode == 'w':
            f.write(f"PatchLinear M4 Results (seed={args.seed})\n")
            f.write(f"{'Frequency':<14} {'SMAPE':>8} {'OWA':>8}\n")
        for key in ['Yearly', 'Quarterly', 'Monthly', 'Others', 'Average']:
            s = smapes.get(key, float('nan'))
            o = owas.get(key,   float('nan'))
            f.write(f"{key:<14} {s:>8.3f} {o:>8.3f}\n")
    print(f"\nResults saved to {results_path}")
