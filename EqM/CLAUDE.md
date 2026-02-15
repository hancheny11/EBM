# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Equilibrium Matching (EqM) is a generative modeling framework based on implicit energy-based models. It replaces time-conditional dynamics with a unified equilibrium landscape learned through gradient descent.

**Paper**: [Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models](https://arxiv.org/abs/2510.02300)

## Common Commands

### Environment Setup
```bash
conda env create -f environment.yml
conda activate EqM
```

### Training
```bash
# Standard EqM training (N = number of GPUs)
torchrun --nnodes=1 --nproc_per_node=N train.py --model EqM-XL/2 --data-path /path/to/imagenet/train

# With explicit energy (EqM-E) using dot product parameterization
torchrun --nnodes=1 --nproc_per_node=N train.py --model EqM-XL/2 --data-path /path/to/imagenet/train --ebm dot

# With wandb logging and dispersive loss
torchrun --nnodes=1 --nproc_per_node=N train.py --model EqM-XL/2 --data-path /path/to/imagenet/train --disp --wandb

# Resume from checkpoint
torchrun --nnodes=1 --nproc_per_node=N train.py --model EqM-XL/2 --data-path /path/to/imagenet/train --ckpt /path/to/model.pt
```

### Sampling
```bash
# Gradient descent sampling (recommended - novel EqM approach)
python sample_gd.py --model EqM-XL/2 --ckpt /path/to/model.pt

# Parallel sampling for FID evaluation
torchrun --nnodes=1 --nproc_per_node=N sample_gd.py --model EqM-XL/2 --num-fid-samples 50000 --ckpt /path/to/model.pt

# Traditional ODE/SDE sampling
python sample.py --model EqM-XL/2 --ckpt /path/to/model.pt
```

### Evaluation
Use generated samples with [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) for FID and Inception Score.

## Architecture

### Core Components

**EqM Model (`models.py`)**: Vision Transformer backbone with:
- Patch embedding, timestep embedder, label embedder for class conditioning
- Stack of `SiTBlock` transformer blocks with adaptive layer norm
- Model variants: EqM-XL/L/B/S with /2/4/8 patch sizes (12 configurations total)

**EBM Modes** (`--ebm` flag):
- `none` (default): Implicit energy
- `dot`: Dot product parameterization
- `l2`: Quadratic energy function
- `mean`: Mean parameterization

**Transport Framework (`transport/`)**: Unified interface for generative modeling
- Path types: Linear, GVP (Generalized Variance Preserving), VP (Variance Preserving)
- Model prediction types: VELOCITY (default), NOISE, SCORE
- Supports ODE, SDE, and gradient descent sampling

### Key Files

| File | Purpose |
|------|---------|
| `train.py` | DDP training with VAE latent encoding, EMA updates, wandb logging |
| `sample_gd.py` | Gradient descent sampling with NAG optimizer and profiling |
| `sample.py` | Traditional ODE/SDE integration sampling |
| `models.py` | EqM model definitions and 12 pre-configured variants |
| `transport/transport.py` | Core Transport class for different path types and samplers |
| `train_utils.py` | Argument parsing for transport/ODE/SDE configurations |

### Data Flow

1. Training uses ImageNet images encoded to VAE latent space (diffusers AutoencoderKL)
2. Model learns in 32x32 latent space (256x256 images with patch size /8)
3. Sampling generates latents, then decodes through VAE

## Development Notes

- Always use `torchrun` for training (DDP requirement)
- EMA decay is 0.9999 by default
- For wandb: set WANDB_KEY, ENTITY, PROJECT environment variables
- `sample_gd.py` includes PyTorch profiler integration
- `guided-diffusion/` contains baseline diffusion implementation (separate codebase)
