import argparse
import os
import torch
import random
import numpy as np

# Seeds are set here and also accept --seed so multi-seed runs
# can be launched from the shell script without editing this file.
parser = argparse.ArgumentParser(description="Proposed Model")

# ── reproducibility ──────────────────────────────────────────────────────────
parser.add_argument("--seed", type=int, default=2021)

# ── basic ─────────────────────────────────────────────────────────────────────
parser.add_argument("--is_training",  type=int, default=1)
parser.add_argument("--model_id",     type=str, default="test")
parser.add_argument('--model', type=str, default='PatchLinear')
parser.add_argument("--checkpoints",  type=str, default="./checkpoints/")
parser.add_argument('--train_only', type=bool, default=False,
                    help='train on full data without validation/test split')
parser.add_argument('--keep_checkpoint', action='store_true', default=False,
    help='Keep checkpoint file after training (needed for zero-shot eval)')
parser.add_argument('--few_shot_ratio', type=float, default=1.0,
    help='Fraction of training timesteps to use (e.g. 0.1 = 10% few-shot)')

# ── data ──────────────────────────────────────────────────────────────────────
parser.add_argument("--data",       type=str, default="ETTh1")
parser.add_argument("--root_path",  type=str, default="./dataset/")
parser.add_argument("--data_path",  type=str, default="ETTh1.csv")
parser.add_argument("--features",   type=str, default="M",
                    help="M: multivariate->multivariate")
parser.add_argument("--target",     type=str, default="OT")
parser.add_argument("--freq",       type=str, default="h")
parser.add_argument("--embed",      type=str, default="timeF")

# ── forecasting ───────────────────────────────────────────────────────────────
parser.add_argument("--seq_len",   type=int, default=96)
parser.add_argument("--label_len", type=int, default=48)
parser.add_argument("--pred_len",  type=int, default=96)
parser.add_argument("--enc_in",    type=int, default=7)

# ── model architecture ────────────────────────────────────────────────────────
parser.add_argument("--d_model",    type=int,   default=64)
parser.add_argument("--t_ff",       type=int,   default=128)   # 2 * d_model
parser.add_argument("--c_ff",       type=int,   default=16)    # max(16,min(C//4,128))
parser.add_argument("--patch_len",  type=int,   default=16)
parser.add_argument("--stride",     type=int,   default=8)
parser.add_argument("--dw_kernel",  type=int,   default=7,     # A6
                    help="DWConv kernel size: 3 / 7 / 13")
parser.add_argument("--small_kernel", type=int, default=3)
parser.add_argument("--alpha",      type=float, default=0.3)   # EMA smoothing

# ── dropout ───────────────────────────────────────────────────────────────────
parser.add_argument("--t_dropout",     type=float, default=0.1)
parser.add_argument("--c_dropout",     type=float, default=0.1)
parser.add_argument("--embed_dropout", type=float, default=0.1)
parser.add_argument("--head_dropout",  type=float, default=0.1)

# ── ablation switches ─────────────────────────────────────────────────────────
parser.add_argument("--use_reparam",        type=int, default=0,
                    help="Structural reparameterisation (0/1)")
parser.add_argument("--use_decomp",         type=int, default=1,  # A1
                    help="EMA decomposition (0/1)")
parser.add_argument("--use_trend_stream",   type=int, default=1,  # A2b
                    help="Trend linear stream (0/1)")
parser.add_argument("--use_seas_stream",    type=int, default=1,  # A2a
                    help="Seasonal CNN stream (0/1)")
parser.add_argument("--use_fusion_gate",    type=int, default=1,  # A3
                    help="Input-dependent stream fusion gate (0/1)")
parser.add_argument("--use_cross_channel",  type=int, default=1,  # A4a
                    help="Cross-channel VGM (0/1)")
parser.add_argument("--use_alpha_gate",     type=int, default=1,  # A4b
                    help="Per-channel alpha mixing gate (0/1)")

# ── optimisation ──────────────────────────────────────────────────────────────
parser.add_argument("--num_workers",    type=int,   default=10)
parser.add_argument("--itr",           type=int,   default=1)
parser.add_argument("--train_epochs",  type=int,   default=100)
parser.add_argument("--batch_size",    type=int,   default=32)
parser.add_argument("--patience",      type=int,   default=10)
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--des",           type=str,   default="Exp")
parser.add_argument("--loss",          type=str,   default="mae")
parser.add_argument("--lradj",         type=str,   default="sigmoid",
    help="LR schedule: type1 | type2 | type3 | sigmoid | constant | 3..6 | cosine_warm")
parser.add_argument("--use_amp",       action="store_true", default=False)

# ── SWA (Stochastic Weight Averaging) ─────────────────────────────────────────
parser.add_argument("--use_swa",        action="store_true", default=False,
                    help="Enable Stochastic Weight Averaging (Izmailov et al. 2018)")
parser.add_argument("--swa_start_frac", type=float, default=0.75,
                    help="Fraction of train_epochs at which SWA begins (default 0.75)")
parser.add_argument("--swa_lr",         type=float, default=None,
                    help="Flat LR during SWA phase (default: 0.05 * learning_rate)")
parser.add_argument("--swa_anneal_epochs", type=int, default=1,
                    help="Epochs to anneal from current LR to swa_lr at SWA start")

# ── cosine warm restarts (used when --lradj cosine_warm) ──────────────────────
parser.add_argument("--cosine_T_0",     type=int,   default=5,
                    help="Epochs in the first cosine cycle")
parser.add_argument("--cosine_T_mult",  type=int,   default=2,
                    help="Cycle length multiplier after each restart")
parser.add_argument("--cosine_eta_min", type=float, default=1e-6,
                    help="Minimum LR at end of each cosine cycle")

# ── GPU ───────────────────────────────────────────────────────────────────────
parser.add_argument("--use_gpu",       type=bool, default=True)
parser.add_argument("--gpu",           type=int,  default=0)
parser.add_argument("--use_multi_gpu", action="store_true", default=False)
parser.add_argument("--devices",       type=str,  default="0,1,2,3")

args = parser.parse_args()

# ── set seeds ─────────────────────────────────────────────────────────────────
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ── convert int flags to bool ─────────────────────────────────────────────────
args.use_reparam       = bool(args.use_reparam)
args.use_decomp        = bool(args.use_decomp)
args.use_trend_stream  = bool(args.use_trend_stream)
args.use_seas_stream   = bool(args.use_seas_stream)
args.use_fusion_gate   = bool(args.use_fusion_gate)
args.use_cross_channel = bool(args.use_cross_channel)
args.use_alpha_gate    = bool(args.use_alpha_gate)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.devices    = args.devices.replace(" ", "")
    device_ids      = args.devices.split(",")
    args.device_ids = [int(x) for x in device_ids]
    args.gpu        = args.device_ids[0]

print("Args:", args)

from exp.exp_main import Exp_Main
Exp = Exp_Main


def _build_setting(args, ii):
    """Build run identifier; append suffixes so SWA/cosine runs don't collide."""
    base = (
        f"{args.model_id}_{args.model}_{args.data}_"
        f"ft{args.features}_sl{args.seq_len}_pl{args.pred_len}_"
        f"d{args.d_model}_dk{args.dw_kernel}_"
        f"dec{int(args.use_decomp)}_"
        f"ts{int(args.use_trend_stream)}_ss{int(args.use_seas_stream)}_"
        f"fg{int(args.use_fusion_gate)}_"
        f"cc{int(args.use_cross_channel)}_ag{int(args.use_alpha_gate)}_"
        f"{args.des}_{ii}"
    )
    suffix = ""
    if args.lradj == 'cosine_warm':
        suffix += "_cos"
    if args.use_swa:
        suffix += "_swa"
    return base + suffix


if args.is_training:
    for ii in range(args.itr):
        setting = _build_setting(args, ii)
        exp = Exp(args)
        print(f">>>>>>>start training: {setting}")
        exp.train(setting)
        print(f">>>>>>>testing: {setting}")
        exp.test(setting)
        torch.cuda.empty_cache()
else:
    ii = 0
    setting = _build_setting(args, ii)
    exp = Exp(args)
    print(f">>>>>>>testing: {setting}")
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
