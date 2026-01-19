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
from contextlib import nullcontext

# ==========================================
# [Added] Import Profiler Components
# ==========================================
from torch.profiler import profile, record_function, ProfilerActivity

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
    model = DDP(model, device_ids=[device])
    
    # Load VAE (ensure we don't need auth token issues, assuming pre-downloaded or public)
    try:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    except Exception as e:
        print(f"Warning: Could not load VAE from HuggingFace ({e}). Make sure you have access or cache.")
        return

    print(f"EqM Parameters: {sum(p.numel() for p in model.parameters()):,}")

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
                
                # Loop over sampling steps
                # We can add a record_function here to group all steps
                with record_function("Sampling_Loop"):
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
                        if args.sampler == 'ngd':
                            m = out
                            
                        xt = xt + out * args.stepsize
                        t += args.stepsize

                if use_cfg:
                    xt, _ = xt.chunk(2, dim=0)
                
                # Decode VAE
                with record_function("VAE_Decode"):
                    samples = vae.decode(xt / 0.18215).sample
                    samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

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

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)