# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal sampling script for EqM using PyTorch DDP.
Modified to include PyTorch Profiler for Algorithmic Intensity analysis.
"""
import math
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from tqdm import tqdm
from models import EqM_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args
import wandb_utils
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
from contextlib import nullcontext
import csv

# ==========================================
# [Added] Import Profiler Components
# ==========================================
from torch.profiler import profile, record_function, ProfilerActivity

# ==========================================
# [Added] Import Memory Profiling Components
# ==========================================
import psutil
import gc


def get_gpu_memory_info(device):
    """Get GPU memory statistics in MB."""
    if not torch.cuda.is_available():
        return {}

    return {
        'allocated_mb': torch.cuda.memory_allocated(device) / 1024 / 1024,
        'reserved_mb': torch.cuda.memory_reserved(device) / 1024 / 1024,
        'max_allocated_mb': torch.cuda.max_memory_allocated(device) / 1024 / 1024,
        'max_reserved_mb': torch.cuda.max_memory_reserved(device) / 1024 / 1024,
    }


def get_cpu_memory_info():
    """Get CPU memory statistics in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
    }


def print_memory_stats(label, device, include_cpu=True):
    """Print memory statistics with a label."""
    gpu_mem = get_gpu_memory_info(device)
    print(f"\n[Memory @ {label}]")
    print(f"  GPU Allocated: {gpu_mem['allocated_mb']:.2f} MB")
    print(f"  GPU Reserved:  {gpu_mem['reserved_mb']:.2f} MB")
    print(f"  GPU Max Allocated: {gpu_mem['max_allocated_mb']:.2f} MB")
    print(f"  GPU Max Reserved:  {gpu_mem['max_reserved_mb']:.2f} MB")

    if include_cpu:
        cpu_mem = get_cpu_memory_info()
        print(f"  CPU RSS: {cpu_mem['rss_mb']:.2f} MB")
        print(f"  CPU VMS: {cpu_mem['vms_mb']:.2f} MB")

    return gpu_mem, get_cpu_memory_info() if include_cpu else {}


class MemoryTracker:
    """Track memory usage over time for profiling."""

    def __init__(self, device, enabled=True):
        self.device = device
        self.enabled = enabled
        self.snapshots = []
        self.step_memory = []  # Track per-step memory

    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        if self.enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def snapshot(self, label):
        """Take a memory snapshot."""
        if not self.enabled:
            return

        gpu_mem = get_gpu_memory_info(self.device)
        cpu_mem = get_cpu_memory_info()
        self.snapshots.append({
            'label': label,
            'gpu': gpu_mem,
            'cpu': cpu_mem,
            'timestamp': time()
        })

    def track_step(self, step):
        """Track memory for a sampling step."""
        if not self.enabled:
            return

        gpu_allocated = torch.cuda.memory_allocated(self.device) / 1024 / 1024
        self.step_memory.append((step, gpu_allocated))

    def print_summary(self):
        """Print memory profiling summary."""
        if not self.enabled or not self.snapshots:
            return

        print("\n" + "=" * 60)
        print(" MEMORY PROFILING SUMMARY ")
        print("=" * 60)

        # Print snapshots
        print("\n--- Memory Snapshots ---")
        print(f"{'Label':<30} {'GPU Alloc (MB)':<15} {'GPU Res (MB)':<15} {'CPU RSS (MB)':<15}")
        print("-" * 75)
        for snap in self.snapshots:
            print(f"{snap['label']:<30} {snap['gpu']['allocated_mb']:<15.2f} {snap['gpu']['reserved_mb']:<15.2f} {snap['cpu']['rss_mb']:<15.2f}")

        # Print peak memory
        if self.snapshots:
            max_gpu_alloc = max(s['gpu']['allocated_mb'] for s in self.snapshots)
            max_gpu_res = max(s['gpu']['reserved_mb'] for s in self.snapshots)
            max_cpu_rss = max(s['cpu']['rss_mb'] for s in self.snapshots)

            print("\n--- Peak Memory Usage ---")
            print(f"  Peak GPU Allocated: {max_gpu_alloc:.2f} MB")
            print(f"  Peak GPU Reserved:  {max_gpu_res:.2f} MB")
            print(f"  Peak CPU RSS:       {max_cpu_rss:.2f} MB")

        # Print step-wise stats if available
        if self.step_memory:
            step_mem_values = [m[1] for m in self.step_memory]
            print("\n--- Per-Step GPU Memory (Sampling Loop) ---")
            print(f"  Min:  {min(step_mem_values):.2f} MB")
            print(f"  Max:  {max(step_mem_values):.2f} MB")
            print(f"  Mean: {sum(step_mem_values)/len(step_mem_values):.2f} MB")

        # Final peak stats from CUDA
        if torch.cuda.is_available():
            print("\n--- CUDA Peak Stats (Session) ---")
            print(f"  Max Memory Allocated: {torch.cuda.max_memory_allocated(self.device) / 1024 / 1024:.2f} MB")
            print(f"  Max Memory Reserved:  {torch.cuda.max_memory_reserved(self.device) / 1024 / 1024:.2f} MB")

def create_npz_from_sample_folder(sample_dir, num):
    """
    Builds a single .npz file from a folder of .png samples.
    Only the first ``num`` samples are read so profiling runs can stay small.
    """
    samples = []
    # Only process files that exist
    files = sorted(glob(f"{sample_dir}/*.png"))[:num]
    for fpath in tqdm(files, desc="Building .npz file from samples"):
        sample_pil = Image.open(fpath)
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    if len(samples) > 0:
        samples = np.stack(samples)
        assert samples.shape == (len(samples), samples.shape[1], samples.shape[2], 3)
        npz_path = f"{sample_dir}.npz"
        np.savez(npz_path, arr_0=samples)
        print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
        return npz_path
    else:
        print("No samples found to create npz.")
        return None

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def quantize_model(model, quant_type):
    """Replace all nn.Linear layers with quantized versions.

    Supported quant_type values:
      - 'nf4': bitsandbytes 4-bit NormalFloat
      - 'fp4': bitsandbytes 4-bit FloatingPoint
      - 'int8': bitsandbytes 8-bit LLM.int8()
      - 'fp8': simulate FP8 weight-only via torch.float8_e4m3fn cast
    """
    import bitsandbytes as bnb
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            in_f, out_f = module.in_features, module.out_features
            has_bias = module.bias is not None

            if quant_type in ('nf4', 'fp4'):
                quant_linear = bnb.nn.Linear4bit(
                    in_f, out_f, bias=has_bias,
                    quant_type=quant_type
                )
                quant_linear.weight = bnb.nn.Params4bit(
                    module.weight.data, requires_grad=False, quant_type=quant_type
                ).cuda()
                if has_bias:
                    quant_linear.bias = module.bias
                setattr(model, name, quant_linear)

            elif quant_type == 'int8':
                quant_linear = bnb.nn.Linear8bitLt(
                    in_f, out_f, bias=has_bias, has_fp16_weights=False
                )
                quant_linear.weight = bnb.nn.Int8Params(
                    module.weight.data, requires_grad=False, has_fp16_weights=False
                ).cuda()
                if has_bias:
                    quant_linear.bias = module.bias
                setattr(model, name, quant_linear)

            elif quant_type == 'fp8':
                # Simulate FP8 weight-only quantization by casting weights
                # to float8_e4m3fn and back, losing precision in the process.
                orig_dtype = module.weight.data.dtype
                w_fp8 = module.weight.data.to(torch.float8_e4m3fn)
                module.weight.data = w_fp8.to(orig_dtype)
                # Module stays as nn.Linear — no structural change needed.
            else:
                raise ValueError(f"Unknown quant_type: {quant_type}")
        else:
            quantize_model(module, quant_type)
    return model


def cleanup():
    """
    End DDP training.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def main(args):
    """
    Trains a new EqM model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    n_gpus = torch.cuda.device_count()
    # disable flash for energy training
    if args.ebm != 'none':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    
    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = int(os.environ["LOCAL_RANK"])
    print(f"Found {n_gpus} GPUs, trying to use device index {device}")
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())
    
    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = EqM_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        uncond=args.uncond,
        ebm=args.ebm
    ).to(device)

    # Note that parameter initialization is done within the EqM constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    if args.ckpt is not None:
        ckpt_path = args.ckpt
        # Handle cases where find_model might fail if path is direct
        try:
            state_dict = find_model(ckpt_path)
        except:
            state_dict = torch.load(ckpt_path, map_location='cpu')
            
        if 'model' in state_dict.keys():
            model.load_state_dict(state_dict["model"])
            ema.load_state_dict(state_dict["ema"])
        else:
            model.load_state_dict(state_dict)
            ema.load_state_dict(state_dict)

        ema = ema.to(device)
        model = model.to(device)
    requires_grad(ema, False)

    # Quantization comparison setup
    quant_models = {}  # quant_type -> quantized ema model
    if args.compare_quant:
        for qt in args.compare_quant:
            ema_q = deepcopy(ema)
            ema_q = quantize_model(ema_q, qt)
            ema_q.eval()
            requires_grad(ema_q, False)
            quant_models[qt] = ema_q
            print(f"[compare-quant] Created {qt} quantized model for comparison")

    model = DDP(model, device_ids=[device])
    
    # Load VAE (ensure we don't need auth token issues, assuming pre-downloaded or public)
    try:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    except Exception as e:
        print(f"Warning: Could not load VAE from HuggingFace ({e}). Make sure you have access or cache.")
        return

    print(f"EqM Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Memory snapshot after VAE load
    if args.memory_profile and rank == 0:
        torch.cuda.synchronize()
        # Will be captured by tracker initialized later, so we print directly
        gpu_mem = get_gpu_memory_info(device)
        print(f"[Memory @ After VAE Load] GPU Allocated: {gpu_mem['allocated_mb']:.2f} MB, Reserved: {gpu_mem['reserved_mb']:.2f} MB")

    # ==========================================
    # [Added] Initialize Memory Tracker
    # ==========================================
    mem_tracker = MemoryTracker(device, enabled=(args.memory_profile and rank == 0))
    mem_tracker.reset_peak_stats()
    mem_tracker.snapshot("After Model Load")

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    # if args.ebm == 'none':
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    # train_steps = 0 # Unused in sampling
    # log_steps = 0   # Unused in sampling
    # running_loss = 0 # Unused in sampling
    # start_time = time() # Unused in sampling

    # Labels to condition the model with (feel free to change):
    ys = torch.randint(1000, size=(local_batch_size,), device=device)
    use_cfg = args.cfg_scale > 1.0
    # Create sampling noise:
    n = ys.size(0)
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Setup classifier-free guidance:
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([1000] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    quant_model_fns = {}  # quant_type -> model_fn
    quant_metrics = {}    # quant_type -> list of (step, normalized_l2)
    if args.compare_quant:
        for qt, ema_q in quant_models.items():
            if use_cfg:
                quant_model_fns[qt] = ema_q.forward_with_cfg
            else:
                quant_model_fns[qt] = ema_q.forward
            quant_metrics[qt] = []

    if rank == 0:
        os.makedirs(args.folder, exist_ok=True)
        
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / args.global_batch_size) * args.global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    
    # Main Sampling Loop
    iterations = int(total_samples // args.global_batch_size)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    
    # ==========================================
    # [Added] Configure Profiler
    # ==========================================
    # Only profile on rank 0 to keep output clean.
    # We use 'with_flops=True' to calculate algorithmic intensity components.
    profiler_ctx = nullcontext()
    if args.profile and rank == 0:
        print(">>> Profiling enabled. This will record FLOPS and CUDA time.")
        profiler_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True,
            with_stack=True,
            # schedule=torch.profiler.schedule(wait=0, warmup=0, active=1), # Optional: only profile 1st iter
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler')
        )

    with profiler_ctx as prof:
        for _ in pbar:
            # We wrap the inference part with torch.no_grad()
            with torch.no_grad():
                z = torch.randn(n, 4, latent_size, latent_size, device=device)
                y = torch.randint(0, args.num_classes, (n,), device=device)
                t = torch.ones((n,), device=device)
                
                if use_cfg:
                    z = torch.cat([z, z], 0)
                    y_null = torch.tensor([1000] * n, device=device)
                    y = torch.cat([y, y_null], 0)
                    t = torch.cat([t, t], 0)
                
                xt = z
                m = torch.zeros_like(xt)

                # Memory snapshot before sampling loop (first iteration only)
                if total == 0:
                    mem_tracker.snapshot("Before Sampling Loop")

                # Loop over sampling steps
                # We can add a record_function here to group all steps
                with record_function("Sampling_Loop"):
                    if args.cross_step:
                        # ==========================================
                        # Cross-step forward setting
                        # Pattern:
                        #                         fwd2 -> fwd3 -> ... -> fwdN
                        # fwd0 -> bwd0 -> fwd1 -> bwd1 -> bwd2 -> bwd3 -> ... -> bwdN
                        # Uses gradient from previous step for updates after initial steps,
                        # allowing forward pass to overlap with sample update.
                        # ==========================================

                        # Cold start: fwd0 + bwd0
                        with record_function("Model_Inference_Step_ColdStart"):
                            if args.sampler == 'gd':
                                out = model_fn(xt, t, y, args.cfg_scale)
                            else:
                                x_ = xt + args.stepsize * m * args.mu
                                out = model_fn(x_, t, y, args.cfg_scale)

                        if not torch.is_tensor(out):
                            out = out[0]

                        # Quantization comparison for cold start
                        if args.compare_quant and total == 0:
                            for qt, qfn in quant_model_fns.items():
                                with record_function(f"Quant_Comparison_ColdStart_{qt}"):
                                    if args.sampler == 'gd':
                                        out_q = qfn(xt, t, y, args.cfg_scale)
                                    else:
                                        out_q = qfn(x_, t, y, args.cfg_scale)
                                    if not torch.is_tensor(out_q):
                                        out_q = out_q[0]
                                    norm_l2 = (torch.norm(out_q - out) / torch.norm(out)).item()
                                    quant_metrics[qt].append((0, norm_l2))

                        if args.sampler == 'ngd':
                            m = out

                        xt = xt + out * args.stepsize
                        t += args.stepsize

                        # Main cross-step loop
                        for step in range(args.num_sampling_steps - 2):
                            # Store previous output
                            out_prev = out

                            # Forward pass (fwd -> bwd stays the same)
                            with record_function("Model_Inference_Step"):
                                if args.sampler == 'gd':
                                    out = model_fn(xt, t, y, args.cfg_scale)
                                else:
                                    x_ = xt + args.stepsize * m * args.mu
                                    out = model_fn(x_, t, y, args.cfg_scale)

                            if not torch.is_tensor(out):
                                out = out[0]

                            # Quantization comparison
                            if args.compare_quant and total == 0:
                                for qt, qfn in quant_model_fns.items():
                                    with record_function(f"Quant_Comparison_Step_{qt}"):
                                        if args.sampler == 'gd':
                                            out_q = qfn(xt, t, y, args.cfg_scale)
                                        else:
                                            out_q = qfn(x_, t, y, args.cfg_scale)
                                        if not torch.is_tensor(out_q):
                                            out_q = out_q[0]
                                        norm_l2 = (torch.norm(out_q - out) / torch.norm(out)).item()
                                        quant_metrics[qt].append((step + 1, norm_l2))

                            # Cross-step update logic
                            if step == 0:
                                # First iteration after cold start: use current gradient
                                update_out = out
                            else:
                                # Subsequent iterations: use previous gradient for overlap
                                update_out = out_prev

                            if args.sampler == 'ngd':
                                m = update_out

                            xt = xt + update_out * args.stepsize
                            t += args.stepsize

                            # Track memory periodically (every 50 steps, first batch only)
                            if total == 0 and step % 50 == 0:
                                mem_tracker.track_step(step)
                    else:
                        # ==========================================
                        # Sequential (original) sampling
                        # Pattern: fwd0 -> bwd0 -> fwd1 -> bwd1 -> ... -> fwdN -> bwdN
                        # ==========================================
                        for step in range(args.num_sampling_steps - 1):

                            # [Added] Record specific model inference
                            with record_function("Model_Inference_Step"):
                                if args.sampler == 'gd':
                                    out = model_fn(xt, t, y, args.cfg_scale)
                                else:
                                    x_ = xt + args.stepsize * m * args.mu
                                    out = model_fn(x_, t, y, args.cfg_scale)

                            if not torch.is_tensor(out):
                                out = out[0]

                            # Quantization comparison
                            if args.compare_quant and total == 0:
                                for qt, qfn in quant_model_fns.items():
                                    with record_function(f"Quant_Comparison_Step_{qt}"):
                                        if args.sampler == 'gd':
                                            out_q = qfn(xt, t, y, args.cfg_scale)
                                        else:
                                            out_q = qfn(x_, t, y, args.cfg_scale)
                                        if not torch.is_tensor(out_q):
                                            out_q = out_q[0]
                                        norm_l2 = (torch.norm(out_q - out) / torch.norm(out)).item()
                                        quant_metrics[qt].append((step, norm_l2))

                            if args.sampler == 'ngd':
                                m = out

                            xt = xt + out * args.stepsize
                            t += args.stepsize

                            # Track memory periodically (every 50 steps, first batch only)
                            if total == 0 and step % 50 == 0:
                                mem_tracker.track_step(step)

                if use_cfg:
                    xt, _ = xt.chunk(2, dim=0)
                
                # Memory snapshot after sampling loop (first iteration only)
                if total == 0:
                    mem_tracker.snapshot("After Sampling Loop")

                # Decode VAE
                with record_function("VAE_Decode"):
                    samples = vae.decode(xt / 0.18215).sample
                    samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

                # Memory snapshot after VAE decode (first iteration only)
                if total == 0:
                    mem_tracker.snapshot("After VAE Decode")

                for i, sample in enumerate(samples):
                    index = i * dist.get_world_size() + rank + total
                    Image.fromarray(sample).save(f"{args.folder}/{index:06d}.png")
            
            total += args.global_batch_size
            dist.barrier()
            
            # If profiling, we can step the schedule if used, or just break after one iter to save time
            if args.profile and rank == 0:
                # break # Uncomment if you only want to profile the first batch
                pass

    # ==========================================
    # [Added] Print Profiling Statistics
    # ==========================================
    if args.profile and rank == 0:
        print("\n" + "="*60)
        print(" PROFILING RESULTS ")
        print("="*60)
        
        # 1. Print Top CUDA Kernels by Time
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
        
        # 2. Calculate Total FLOPS
        events = prof.key_averages()
        total_flops = sum([e.flops for e in events])
        
        # 3. Intensity Calculation (Approximate)
        # Note: Profiler gives FLOPs accurately. Memory Traffic is harder.
        # We use Total CUDA Time to estimate "effective" intensity or just print FLOPS.
        print(f"\nTotal Floating Point Operations (FLOPS): {total_flops:.4e}")
        
        if total_flops == 0:
            print("WARNING: Total FLOPS is 0. Make sure you have a GPU available and PyTorch is built with CUDA support.")
        else:
            print(f"This represents the computational load for generating {args.global_batch_size} images over {args.num_sampling_steps} steps.")
            print("To get precise Algorithmic Intensity (FLOPS / Bytes), divide this FLOP count by the total DRAM bytes read/written")
            print("measured via 'ncu' (Nsight Compute) or estimated theoretically (Model Size * Steps + Activation IO).")

    # ==========================================
    # [Added] Print Memory Profiling Summary
    # ==========================================
    mem_tracker.snapshot("End of Sampling")
    mem_tracker.print_summary()

    # Quantization comparison results
    if args.compare_quant and rank == 0 and quant_metrics:
        # Build step list from first quant type (all share same steps)
        first_qt = args.compare_quant[0]
        steps = [m[0] for m in quant_metrics[first_qt]]

        # Save CSV with one column per quant type
        csv_path = f"{args.folder}/quant_comparison.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['step'] + [f'normalized_l2_{qt}' for qt in args.compare_quant]
            writer.writerow(header)
            for i, s in enumerate(steps):
                row = [s] + [f"{quant_metrics[qt][i][1]:.6f}" for qt in args.compare_quant]
                writer.writerow(row)
        print(f"[compare-quant] Saved per-step metrics to {csv_path}")

        # Generate plot with one line per quant type
        plt.figure(figsize=(10, 5))
        for qt in args.compare_quant:
            nl2_vals = [m[1] for m in quant_metrics[qt]]
            plt.plot(steps, nl2_vals, linewidth=0.8, label=qt.upper())
        plt.xlabel('Step')
        plt.ylabel('Normalized L2 Shift')
        plt.title('Quantization vs Full Precision')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = f"{args.folder}/quant_comparison.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[compare-quant] Saved plot to {plot_path}")

        # Print summary stats per quant type
        print(f"\n[compare-quant] Summary Statistics:")
        for qt in args.compare_quant:
            nl2_vals = [m[1] for m in quant_metrics[qt]]
            nl2_tensor = torch.tensor(nl2_vals)
            print(f"  {qt.upper():>5s} — Min: {nl2_tensor.min().item():.6f}  "
                  f"Max: {nl2_tensor.max().item():.6f}  "
                  f"Mean: {nl2_tensor.mean().item():.6f}  "
                  f"Std: {nl2_tensor.std().item():.6f}")

    if rank == 0:
        print(f"Creating .npz file with {args.num_fid_samples} samples")
        create_npz_from_sample_folder(args.folder, args.num_fid_samples)
        print("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will sample EqM-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom EqM checkpoint")
    parser.add_argument("--stepsize", type=float, default=0.0017,
                        help="step size eta")
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--folder", type=str, default='samples')
    parser.add_argument("--sampler", type=str, default='gd', choices=['gd', 'ngd'])
    parser.add_argument("--mu", type=float, default=0.3,
                        help="NAG-GD hyperparameter mu")
    parser.add_argument("--num-fid-samples", type=int, default=16)
    parser.add_argument("--uncond", type=bool, default=True,
                        help="disable/enable noise conditioning")
    parser.add_argument("--ebm", type=str, choices=["none", "l2", "dot", "mean"], default="none",
                        help="energy formulation")

    # [Added] Profiling Argument
    parser.add_argument("--profile", action='store_true', help="Enable PyTorch Profiler for FLOPs calculation")

    # [Added] Cross-step forward setting
    parser.add_argument("--cross-step", action='store_true',
                        help="Enable cross-step forward setting for overlapped computation")

    # [Added] Memory Profiling Argument
    parser.add_argument("--memory-profile", action='store_true',
                        help="Enable memory profiling for CPU and GPU")

    # [Added] Quantization Comparison
    parser.add_argument("--compare-quant", nargs='+',
                        choices=['nf4', 'fp4', 'int8', 'fp8'], default=None,
                        help="Quantization types to compare against full precision (e.g. --compare-quant nf4 fp4 int8 fp8)")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)