"""
Lightweight gate extraction — run on any trained GLPatch model during test.

Usage (standalone, trains + extracts in one go):
  python extract_gates.py --data ETTh2 --data_path ETTh2.csv --enc_in 7 \
    --pred_len 96 --learning_rate 0.0007

Or import and call from exp_main after training:
  from extract_gates import extract_and_plot_gates
  extract_and_plot_gates(model, test_loader, device, 'ETTh2', 96, './plots/gates')
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def extract_and_plot_gates(model, test_loader, device, dataset_name, pred_len, out_dir):
    """
    Extract fusion gate values from a trained GLPatch model during test inference.
    Generates all gate visualization plots.

    Args:
        model: trained GLPatch model (Model wrapper or network)
        test_loader: test DataLoader
        device: torch device
        dataset_name: string for plot titles
        pred_len: prediction horizon
        out_dir: output directory for plots
    """
    os.makedirs(out_dir, exist_ok=True)

    # Find the network inside the wrapper
    net = model
    if hasattr(model, 'net'):
        net = model.net
    if hasattr(model, 'module'):
        if hasattr(model.module, 'net'):
            net = model.module.net

    # Get res_alpha
    res_alpha = None
    if hasattr(net, 'res_alpha'):
        res_alpha = net.res_alpha.detach().cpu().item()

    # Hook to capture gate values
    captured_gates = []

    def hook_fn(module, input, output):
        # output = raw logits from gate_expand
        # Reconstruct actual gate
        gate_min = getattr(net, 'gate_min', 0.1)
        gate_max = getattr(net, 'gate_max', 0.9)
        gate = torch.sigmoid(output) * (gate_max - gate_min) + gate_min
        captured_gates.append(gate.detach().cpu())

    # Register hook
    hook = None
    if hasattr(net, 'gate_expand'):
        hook = net.gate_expand.register_forward_hook(hook_fn)
    elif hasattr(net, 'gate_fc'):
        hook = net.gate_fc.register_forward_hook(hook_fn)
    else:
        print(f"  WARNING: No fusion gate found in model — skipping gate extraction")
        return None, None

    # Run test inference
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(device)
            _ = model(batch_x)

    # Remove hook
    hook.remove()

    # Concatenate: (total_samples * channels, pred_len)
    all_gates = torch.cat(captured_gates, dim=0).numpy()
    print(f"  Gates extracted: shape={all_gates.shape}, mean={all_gates.mean():.4f}, "
          f"std={all_gates.std():.4f}, res_alpha={res_alpha}")

    # ---- Plot 1: Histogram ----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_gates.flatten(), bins=80, density=True, alpha=0.7,
            color='steelblue', edgecolor='none')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.5, label='g=0.5 (equal)')
    ax.axvline(all_gates.mean(), color='orange', linewidth=2,
               label=f'mean={all_gates.mean():.3f}')
    ax.set_xlabel('Gate value g  (g>0.5 → seasonal, g<0.5 → trend)')
    ax.set_ylabel('Density')
    ax.set_title(f'{dataset_name} H={pred_len} — Fusion Gate Distribution')
    ax.set_xlim(0, 1); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'gate_histogram_{pred_len}.png'), dpi=150)
    plt.close(fig)

    # ---- Plot 2: Mean per timestep ----
    mean_g = all_gates.mean(axis=0)
    std_g = all_gates.std(axis=0)
    t = np.arange(1, pred_len + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, mean_g, 'o-', color='steelblue', linewidth=2, markersize=3)
    ax.fill_between(t, mean_g - std_g, mean_g + std_g, alpha=0.2, color='steelblue')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Forecast timestep t+k')
    ax.set_ylabel('Gate value g')
    ax.set_title(f'{dataset_name} H={pred_len} — Gate per Timestep (mean ± std)')
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'gate_per_timestep_{pred_len}.png'), dpi=150)
    plt.close(fig)

    # ---- Plot 3: Heatmap ----
    n_show = min(all_gates.shape[0], 200)
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(all_gates[:n_show], aspect='auto', cmap='RdYlBu_r',
                    vmin=0.1, vmax=0.9, interpolation='nearest')
    ax.set_xlabel('Forecast timestep')
    ax.set_ylabel('Sample')
    ax.set_title(f'{dataset_name} H={pred_len} — Gate Heatmap '
                 f'(red=seasonal, blue=trend)')
    plt.colorbar(im, ax=ax, label='g')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'gate_heatmap_{pred_len}.png'), dpi=150)
    plt.close(fig)

    return all_gates, res_alpha


def plot_cross_horizon_gates(gates_dict, dataset_name, out_dir):
    """
    Compare gate profiles across horizons.
    gates_dict: {pred_len: gates_array}
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (pl, gates) in enumerate(sorted(gates_dict.items())):
        mean_g = gates.mean(axis=0)
        x = np.linspace(0, 1, len(mean_g))
        ax.plot(x, mean_g, '-', color=colors[i % len(colors)], linewidth=2,
                label=f'H={pl} (mean={mean_g.mean():.3f})')

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Relative position (0=near, 1=far)')
    ax.set_ylabel('Gate value g')
    ax.set_title(f'{dataset_name} — Gate Profile Across Horizons')
    ax.set_ylim(0, 1); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'gate_across_horizons.png'), dpi=150)
    plt.close(fig)


def plot_alpha_summary(alpha_dict, out_dir):
    """
    Bar chart of learned α_res.
    alpha_dict: {(dataset, pred_len): value}
    """
    if not alpha_dict:
        return

    labels = [f'{ds}\nH={pl}' for (ds, pl) in sorted(alpha_dict.keys())]
    values = [alpha_dict[k] for k in sorted(alpha_dict.keys())]

    fig, ax = plt.subplots(figsize=(max(len(labels) * 0.6, 8), 4))
    bars = ax.bar(range(len(labels)), values, color='steelblue', alpha=0.8, width=0.7)
    ax.axhline(0.05, color='red', linestyle='--', linewidth=1.5, label='Init (0.05)')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel('Learned α_res')
    ax.set_title('Learned Gating Blend Factor Across Datasets/Horizons')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'res_alpha_summary.png'), dpi=150)
    plt.close(fig)
