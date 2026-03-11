#!/bin/bash

# ============================================================
# GATE VISUALIZATION — Standalone Pipeline
# ============================================================
# Trains GLPatch, extracts fusion gate values during test,
# and generates all visualizations.
#
# PREREQUISITE:
#   1. Run setup_ablation.sh first
#   2. Comment out checkpoint deletion in exp_main.py:
#      Find: os.remove(best_model_path)
#      Change to: # os.remove(best_model_path)
#
# Datasets: ETTh2, ETTm1, Weather, Electricity, Solar, Exchange
# Estimated: ~3 hours (24 models + gate extraction)
# Output: ./plots/gates/<dataset>/

ma_type=ema; alpha=0.3; beta=0.3
model=GLPatch_ablation
seq_len=96

declare -A DS_LR DS_PATH DS_FLAG DS_ENC DS_BATCH
DS_LR=( [ETTh2]=0.0007 [ETTm1]=0.0007 [Weather]=0.0005 [Electricity]=0.005 [Solar]=0.01 [Exchange]=0.000005 )
DS_PATH=( [ETTh2]=ETTh2.csv [ETTm1]=ETTm1.csv [Weather]=weather.csv [Electricity]=electricity.csv [Solar]=solar.txt [Exchange]=exchange_rate.csv )
DS_FLAG=( [ETTh2]=ETTh2 [ETTm1]=ETTm1 [Weather]=custom [Electricity]=custom [Solar]=Solar [Exchange]=custom )
DS_ENC=( [ETTh2]=7 [ETTm1]=7 [Weather]=21 [Electricity]=321 [Solar]=137 [Exchange]=8 )
DS_BATCH=( [ETTh2]=2048 [ETTm1]=2048 [Weather]=2048 [Electricity]=256 [Solar]=512 [Exchange]=32 )

echo ""
echo "========== [$(date '+%H:%M:%S')] GATE VISUALIZATION PIPELINE =========="
echo ""

# Check that os.remove is commented out
if grep -q "^        os.remove(best_model_path)" exp/exp_main.py; then
    echo "WARNING: os.remove(best_model_path) is NOT commented out in exp/exp_main.py"
    echo "Checkpoints will be deleted after training and gates cannot be extracted."
    echo ""
    echo "Fix: comment out that line, then re-run this script."
    echo "  sed -i 's/^        os.remove(best_model_path)/        # os.remove(best_model_path)/' exp/exp_main.py"
    echo ""
    read -p "Apply fix automatically? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sed -i 's/^        os.remove(best_model_path)/        # os.remove(best_model_path)/' exp/exp_main.py
        echo "Fixed."
    else
        echo "Aborting."
        exit 1
    fi
fi

# Train all models
for ds in ETTh2 ETTm1 Weather Electricity Solar Exchange; do
  lr=${DS_LR[$ds]}; data_path=${DS_PATH[$ds]}; data_flag=${DS_FLAG[$ds]}
  enc_in=${DS_ENC[$ds]}; batch=${DS_BATCH[$ds]}

  for pred_len in 96 192 336 720; do
    echo ">>> [$(date '+%H:%M:%S')] Training ${ds} pl=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path ${data_path} \
      --model_id gateviz_${ds}_${pred_len} --model $model --data ${data_flag} \
      --features M --seq_len $seq_len --pred_len $pred_len --enc_in $enc_in \
      --des 'Exp' --itr 1 --batch_size $batch --learning_rate $lr \
      --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_gating 1 --use_fusion 1 \
      --use_amp --num_workers 2 \
      2>&1 | tail -3
  done
  echo "=== ${ds} training complete ==="
  echo ""
done

echo ""
echo ">>> [$(date '+%H:%M:%S')] Extracting gates and generating plots..."
echo ""

# Extract gates and generate plots
python -u << 'PYEOF'
import os, sys, glob, torch
import numpy as np
from types import SimpleNamespace

sys.path.insert(0, '.')
from extract_gates import (extract_and_plot_gates, plot_cross_horizon_gates,
                           plot_alpha_summary)
from models.GLPatch_ablation import Model
from data_provider.data_factory import data_provider

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
out_base = './plots/gates'
os.makedirs(out_base, exist_ok=True)

ds_configs = {
    'ETTh2':       {'data_path':'ETTh2.csv','data':'ETTh2','enc_in':7,'batch':2048},
    'ETTm1':       {'data_path':'ETTm1.csv','data':'ETTm1','enc_in':7,'batch':2048},
    'Weather':     {'data_path':'weather.csv','data':'custom','enc_in':21,'batch':2048},
    'Electricity': {'data_path':'electricity.csv','data':'custom','enc_in':321,'batch':256},
    'Solar':       {'data_path':'solar.txt','data':'Solar','enc_in':137,'batch':512},
    'Exchange':    {'data_path':'exchange_rate.csv','data':'custom','enc_in':8,'batch':32},
}

alpha_summary = {}
gate_means = {}  # {(ds, pl): mean_gate}

for dataset, cfg in ds_configs.items():
    ds_out = os.path.join(out_base, dataset)
    os.makedirs(ds_out, exist_ok=True)
    gates_by_pl = {}

    for pred_len in [96, 192, 336, 720]:
        # Build model args
        args = SimpleNamespace(
            seq_len=96, pred_len=pred_len, enc_in=cfg['enc_in'],
            patch_len=16, stride=8, padding_patch='end',
            revin=1, ma_type='ema', alpha=0.3, beta=0.3,
            use_gating=1, use_fusion=1, gate_position='pre_pointwise',
            res_alpha_init=0.05, gate_hidden_dim=32,
            gate_min=0.1, gate_max=0.9, gate_reduction=4,
            data=cfg['data'], root_path='./dataset/',
            data_path=cfg['data_path'], features='M',
            target='OT', freq='h', embed='timeF',
            label_len=48, num_workers=2,
            batch_size=cfg['batch'],
        )

        # Find checkpoint
        matches = glob.glob(f"./checkpoints/gateviz_{dataset}_{pred_len}_*")
        if not matches:
            print(f"  SKIP {dataset} pl={pred_len}: no checkpoint")
            continue

        ckpt_path = os.path.join(matches[0], 'checkpoint.pth')
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {dataset} pl={pred_len}: checkpoint.pth deleted!")
            print(f"  (Comment out os.remove in exp_main.py and re-train)")
            continue

        print(f"\n>>> {dataset} pl={pred_len}")
        model = Model(args).float().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        test_data, test_loader = data_provider(args, flag='test')

        gates, res_alpha = extract_and_plot_gates(
            model, test_loader, device, dataset, pred_len, ds_out)

        if gates is not None:
            gates_by_pl[pred_len] = gates
            gate_means[(dataset, pred_len)] = (gates.mean(), gates.std())
        if res_alpha is not None:
            alpha_summary[(dataset, pred_len)] = res_alpha

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Cross-horizon comparison
    if len(gates_by_pl) > 1:
        plot_cross_horizon_gates(gates_by_pl, dataset, ds_out)

# ---- Summary plots ----
if alpha_summary:
    plot_alpha_summary(alpha_summary, out_base)

# ---- Print summary table ----
print(f"\n\n{'='*80}")
print(f"GATE ANALYSIS SUMMARY")
print(f"{'='*80}")
print(f"  {'Dataset':<15} {'PL':>4}  {'Mean g':>8}  {'Std g':>8}  {'α_res':>8}  Interpretation")
print(f"  {'-'*15} {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*25}")

for (ds, pl) in sorted(gate_means.keys()):
    mean_g, std_g = gate_means[(ds, pl)]
    alpha_val = alpha_summary.get((ds, pl), float('nan'))

    if mean_g > 0.55:
        interp = "seasonal-dominant"
    elif mean_g < 0.45:
        interp = "trend-dominant"
    else:
        interp = "balanced"

    print(f"  {ds:<15} {pl:>4}  {mean_g:>8.4f}  {std_g:>8.4f}  {alpha_val:>8.4f}  {interp}")

print(f"\nPlots: {out_base}/<dataset>/")
print(f"  gate_histogram_<H>.png     — distribution of gate values")
print(f"  gate_per_timestep_<H>.png  — mean gate vs forecast step")
print(f"  gate_heatmap_<H>.png       — gate across samples × timesteps")
print(f"  gate_across_horizons.png   — compare profiles across H")
print(f"  res_alpha_summary.png      — learned α_res bar chart")
PYEOF

echo ""
echo "========== [$(date '+%H:%M:%S')] GATE VISUALIZATION COMPLETE =========="
