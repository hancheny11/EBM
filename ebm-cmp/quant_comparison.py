"""
quant_comparison.py — Quantization comparison for ebm-cmp ResNet128 EBM model.

Mirrors the methodology from ebm-imp/quant_comparison.py and
ired_code_release_origin_claude:
- Runs Langevin sampling with original FP32 weights and 4 quantized variants
  (NF4, FP4, INT8, FP8) in parallel from the same initial state.
- Measures normalized L2 divergence of the x_mod trajectory at each step.
- Saves per-step metrics to quant_comparison.csv and a plot to quant_comparison.png.

Usage:
  python quant_comparison.py --num_steps 200 --step_lr 170 --batch_size 16
"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import models_pt
from models_pt import ResNet128

# Optional bitsandbytes for NF4/FP4
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    print("[compare-quant] Warning: bitsandbytes not found — NF4/FP4 will use simulated 4-bit quantization")

NOISE_STD = 0.005


# ---------------------------------------------------------------------------
# Tensor-level quantization helpers
# ---------------------------------------------------------------------------

def _quantize_tensor(t: torch.Tensor, quant_type: str) -> torch.Tensor:
    """Quantize a single tensor and dequantize back to original dtype."""
    orig_dtype = t.dtype

    if quant_type == 'fp8':
        t_fp8 = t.detach().to(torch.float8_e4m3fn)
        return t_fp8.to(orig_dtype).detach()

    elif quant_type == 'int8':
        t_f = t.detach().float()
        scale = t_f.abs().max() / 127.0
        t_int8 = (t_f / (scale + 1e-8)).round().clamp(-128, 127).to(torch.int8)
        t_dq = t_int8.float() * scale
        return t_dq.to(orig_dtype).detach()

    elif quant_type in ('nf4', 'fp4'):
        if BNB_AVAILABLE:
            t_f = t.detach().float().contiguous()
            if t_f.is_cuda and t_f.ndim >= 1:
                t_2d = t_f.reshape(1, -1) if t_f.ndim == 1 else t_f.reshape(t_f.shape[0], -1)
                try:
                    q, state = bnb.functional.quantize_4bit(t_2d, quant_type=quant_type)
                    t_dq = bnb.functional.dequantize_4bit(q, state, quant_type=quant_type)
                    return t_dq.reshape(t.shape).to(orig_dtype).detach()
                except Exception:
                    pass
        # Simulated 4-bit: uniform quantization to 16 levels
        t_f = t.detach().float()
        min_v, max_v = t_f.min(), t_f.max()
        t_norm = (t_f - min_v) / (max_v - min_v + 1e-8)
        t_q4 = (t_norm * 15.0).round() / 15.0
        t_dq = t_q4 * (max_v - min_v) + min_v
        return t_dq.to(orig_dtype).detach()

    raise ValueError(f"Unknown quant_type: {quant_type}")


def _get_tensor_data(v) -> torch.Tensor:
    """Extract the underlying tensor from a Parameter or plain tensor."""
    if isinstance(v, nn.Parameter):
        return v.data
    return v


def quantize_weights_dict(weights: dict, quant_type: str) -> dict:
    """Return a new weights dict with all conv and FC weight tensors quantized."""
    new_weights = {}
    for scope, grp in weights.items():
        new_grp = {}
        for key, val in grp.items():
            t = _get_tensor_data(val)
            if isinstance(t, torch.Tensor) and key in ('c', 'w') and t.ndim >= 2:
                q_data = _quantize_tensor(t, quant_type)
                # Preserve nn.Parameter wrapping so the model's forward sees same type
                if isinstance(val, nn.Parameter):
                    p = nn.Parameter(q_data, requires_grad=False)
                    new_grp[key] = p
                else:
                    new_grp[key] = q_data
            else:
                new_grp[key] = val
        new_weights[scope] = new_grp
    return new_weights


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def _save_quant_results(quant_metrics: dict, quant_types: list, results_dir: str = '.'):
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, 'quant_comparison.csv')
    steps = [m[0] for m in quant_metrics[quant_types[0]]]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step'] + [f'normalized_l2_{qt}' for qt in quant_types])
        for i, s in enumerate(steps):
            row = [s] + [f"{quant_metrics[qt][i][1]:.6f}" for qt in quant_types]
            writer.writerow(row)
    print(f"[compare-quant] Saved per-step metrics to {csv_path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    for qt in quant_types:
        x = [m[0] for m in quant_metrics[qt]]
        y = [m[1] for m in quant_metrics[qt]]
        ax.plot(x, y, linewidth=1.2, label=qt.upper())
    ax.set_xlabel('Langevin step')
    ax.set_ylabel('Normalized L2 divergence (x_mod)')
    ax.set_title('Quantization comparison — ebm-cmp ResNet128')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(results_dir, 'quant_comparison.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[compare-quant] Saved plot to {plot_path}")

    print(f"\n[compare-quant] Summary Statistics (Langevin steps):")
    for qt in quant_types:
        nl2_vals = [m[1] for m in quant_metrics[qt]]
        arr = np.array(nl2_vals)
        label = qt.upper().rjust(4)
        print(f"  {label} — Min: {arr.min():.6f}  Max: {arr.max():.6f}  Mean: {arr.mean():.6f}  Std: {arr.std():.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=200)
    parser.add_argument('--step_lr', type=float, default=170.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', type=str, default='.')
    parser.add_argument('--quant_types', type=str, default='nf4,fp4,int8,fp8')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    quant_types = [qt.strip() for qt in args.quant_types.split(',')]

    # Build model + weights (same as imagenet_demo_pt.py)
    model = ResNet128(num_channels=3, num_filters=64)
    weights = model.construct_weights("context_0")

    # Move weights to device
    def _move_weights(w, dev):
        out = {}
        for scope, grp in w.items():
            new_grp = {}
            for key, val in grp.items():
                t = _get_tensor_data(val)
                if torch.is_tensor(t):
                    moved = t.to(dev)
                    new_grp[key] = nn.Parameter(moved, requires_grad=False) if isinstance(val, nn.Parameter) else moved
                else:
                    new_grp[key] = val
            out[scope] = new_grp
        return out

    weights = _move_weights(weights, device)

    # Randomize weights (no checkpoint available)
    def _randomize_weights(w):
        for grp in w.values():
            for key, val in grp.items():
                t = _get_tensor_data(val)
                if not (torch.is_tensor(t) and t.is_floating_point()):
                    continue
                with torch.no_grad():
                    if t.ndim >= 2:
                        nn.init.kaiming_normal_(t)
                    else:
                        t.normal_(0.0, 0.02)

    _randomize_weights(weights)

    # Build quantized weight dicts
    quant_weights = {}
    for qt in quant_types:
        quant_weights[qt] = quantize_weights_dict(weights, qt)
        print(f"[compare-quant] Created {qt} quantized weights for comparison")

    # Labels (model ignores them when models_pt.CCLASS=False, but keep for API parity)
    ls = np.random.permutation(1000)[:args.batch_size]
    labels = torch.from_numpy(np.eye(1000, dtype=np.float32)[ls]).to(device)

    # Initial x in NHWC [B, H, W, C] uniform (mirrors imagenet_demo_pt.py)
    x_init = torch.rand((args.batch_size, 128, 128, 3), device=device)

    def langevin_step(x_nhwc, w):
        # Add noise + convert to NCHW for model (mirrors imagenet_demo_pt.py)
        x_noisy = x_nhwc.detach() + torch.randn_like(x_nhwc) * NOISE_STD
        x_noisy.requires_grad_(True)
        x_nchw = x_noisy.permute(0, 3, 1, 2).contiguous()
        energy = model.forward(x_nchw, w, label=labels, reuse=True, stop_at_grad=False, stop_batch=True)
        energy_scalar = energy.sum() if energy.ndim != 0 else energy
        (x_grad,) = torch.autograd.grad(energy_scalar, x_noisy, create_graph=False)
        x_next = (x_noisy.detach() - args.step_lr * x_grad.detach()).clamp(0.0, 1.0)
        return x_next.detach()

    # Warmup step
    x_fp = langevin_step(x_init, weights)
    x_q = {qt: langevin_step(x_init, quant_weights[qt]) for qt in quant_types}

    quant_metrics = {qt: [] for qt in quant_types}

    print("Running quantization comparison...")
    for i in range(args.num_steps):
        x_fp = langevin_step(x_fp, weights)
        for qt in quant_types:
            x_q[qt] = langevin_step(x_q[qt], quant_weights[qt])
            norm_l2 = (torch.norm(x_q[qt] - x_fp) / (torch.norm(x_fp) + 1e-8)).item()
            quant_metrics[qt].append((i, norm_l2))

        if (i + 1) % 20 == 0:
            vals = "  ".join(f"{qt.upper()}={quant_metrics[qt][-1][1]:.4f}" for qt in quant_types)
            print(f"  step {i+1:4d}/{args.num_steps}: {vals}")

    _save_quant_results(quant_metrics, quant_types, results_dir=args.results_dir)


if __name__ == '__main__':
    main()
