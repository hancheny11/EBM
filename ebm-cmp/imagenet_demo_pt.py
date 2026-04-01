"""
PyTorch rewrite of imagenet_demo.py

Changes:
- Removed TensorFlow placeholders/session/graph ops.
- Replaced TF flags with hardcoded default constants.
- Uses torch autograd to compute dE/dx and performs Langevin-style updates.
"""

from __future__ import annotations

import os
import os.path as osp
from typing import Any, Optional, Sequence, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn as nn

from models_pt import ResNet128

import time


# ---------------------------------------------------------------------
# Hardcoded defaults (replacing TF flags) - matches the original script.
# ---------------------------------------------------------------------
LOGDIR: str = "./logdir"
NUM_STEPS: int = 10
STEP_LR: float = 170.0
BATCH_SIZE: int = 16
EXP: str = "default"
RESUME_ITER: int = -1
SPEC_NORM: bool = True
CCLASS: bool = True
USE_ATTENTION: bool = False

IMAGE_SIZE: int = 128
NUM_CLASSES: int = 1000
NOISE_STD: float = 0.005


def rescale_im(im: np.ndarray) -> np.ndarray:
    """Match original behavior: [0,1] -> [0,255] uint8 with clipping."""
    return np.clip(im * 256, 0, 255).astype(np.uint8)


def _disable_param_grads(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def _randomize_model_params(model: torch.nn.Module) -> None:
    """Force random params (so the script doesn't depend on any checkpoint file)."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not p.is_floating_point():
                continue
            if p.ndim >= 2:
                nn.init.kaiming_normal_(p)
            else:
                nn.init.zeros_(p)



def _move_any_to_device(obj: Any, device: torch.device, *, dtype: Optional[torch.dtype] = None) -> Any:
    """Recursively move tensors/parameters/ndarrays inside nested structures to the given device."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        t = torch.from_numpy(obj)
        return t.to(device=device, dtype=dtype) if dtype is not None else t.to(device=device)
    if isinstance(obj, torch.nn.Parameter):
        t = obj.data
        obj.data = (t.to(device=device, dtype=dtype) if dtype is not None else t.to(device=device))
        return obj
    if torch.is_tensor(obj):
        return obj.to(device=device, dtype=dtype) if dtype is not None else obj.to(device=device)
    if isinstance(obj, dict):
        return {k: _move_any_to_device(v, device, dtype=dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_any_to_device(v, device, dtype=dtype) for v in obj)
    return obj


def _randomize_any_tensors(obj: Any) -> None:
    """Recursively initialize floating tensors/parameters in-place with random values."""
    if obj is None:
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _randomize_any_tensors(v)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _randomize_any_tensors(v)
        return
    # Handle Parameter or Tensor
    t = None
    if isinstance(obj, torch.nn.Parameter):
        t = obj.data
    elif torch.is_tensor(obj):
        t = obj
    if t is None or (not t.is_floating_point()):
        return
    with torch.no_grad():
        if t.ndim >= 2:
            nn.init.kaiming_normal_(t)
        else:
            # Small random values for 1D params (biases, BN scales, etc.)
            t.normal_(mean=0.0, std=0.02)


def _forward_energy(
    model: Any,
    x: torch.Tensor,
    labels: Union[torch.Tensor, np.ndarray],
    *,
    weights: Optional[Any] = None,
) -> torch.Tensor:
    """
    Call into the user's ResNet128 implementation.

    Supports two common styles:
    1) TF-like port: model.forward(x, weights, label=..., ...)
    2) Standard nn.Module: model(x, label=...) or model(x, labels)
    """
    if isinstance(labels, np.ndarray):
        labels_t = torch.from_numpy(labels).to(x.device)
    else:
        labels_t = labels.to(x.device)

    # Style (1): forward(x, weights, label=...)
    if hasattr(model, "forward") and weights is not None:
        try:
            return model.forward(
                x,
                weights,
                label=labels_t,
                reuse=True,
                stop_at_grad=False,
                stop_batch=True,
            )
        except TypeError:
            # If signature differs, retry with fewer args
            return model.forward(x, weights, label=labels_t)

    # Style (2): callable module
    try:
        return model(x, label=labels_t)
    except TypeError:
        return model(x, labels_t)


def _make_montage(x_hwc: np.ndarray, grid_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Create a single HWC image by tiling a batch of images.
    x_hwc: (B, H, W, C) in [0,1] float.
    Returns: (H*rows, W*cols, C) uint8.
    """
    b, h, w, c = x_hwc.shape

    if grid_hw is None:
        cols = int(np.ceil(np.sqrt(b)))
        rows = int(np.ceil(b / cols))
    else:
        rows, cols = grid_hw

    canvas = np.zeros((rows * h, cols * w, c), dtype=np.uint8)

    x_u8 = rescale_im(x_hwc)

    for idx in range(b):
        r = idx // cols
        col = idx % cols
        if r >= rows:
            break
        canvas[r * h : (r + 1) * h, col * w : (col + 1) * w, :] = x_u8[idx]

    return canvas


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    # Build model (assumes models.ResNet128 is already a PyTorch module / port).
    model = ResNet128(num_channels=3, num_filters=64)
    if isinstance(model, torch.nn.Module):
        model.to(device)
        _randomize_model_params(model)
        model.eval()
        _disable_param_grads(model)

    # Optional: preserve the original API if your model uses a separate weights dict.
    weights = None
    if hasattr(model, "construct_weights"):
        try:
            weights = model.construct_weights("context_0")
        except Exception:
            weights = None

    # If a separate weights structure is used (functional conv/BN weights),
    # make sure it lives on the same device as x/model and initialize it randomly.
    if weights is not None:
        weights = _move_any_to_device(weights, device, dtype=torch.float32)
        _randomize_any_tensors(weights)
        weights = _move_any_to_device(weights, device, dtype=torch.float32)

    # NOTE: No checkpoint loading for now.

    # We explicitly randomized parameters above to avoid any dependency on saved weights.
    logdir = osp.join(LOGDIR, EXP)


    # Sample labels (fixes original typo: lx -> ls) and init x.
    ls = np.random.permutation(NUM_CLASSES)[:BATCH_SIZE]
    labels = torch.from_numpy(np.eye(NUM_CLASSES, dtype=np.float32)[ls]).to(device)

    # Initialize sampling with Uniform(0,1)
    x_mod = torch.rand((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3), device=device)

    ims: Sequence[np.ndarray] = []

    from torch.profiler import profile, record_function, ProfilerActivity

    start_time = time.time()
    
    for i in range(NUM_STEPS):

        # if (i % 10) == 0:
        #     prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False)
        #     prof.start()

        # Add Gaussian noise each step (mirrors tf.random_normal in the graph).
        x = x_mod.detach() + torch.randn_like(x_mod) * NOISE_STD
        x.requires_grad_(True)

        # Convert to NCHW for typical PyTorch models; keep a fallback if your port expects NHWC.
        x_for_model = x
        if x_for_model.shape[-1] == 3:
            # NHWC -> NCHW
            x_for_model = x_for_model.permute(0, 3, 1, 2).contiguous()

        energy = _forward_energy(model, x_for_model, labels, weights=weights)

        # Ensure scalar for autograd.grad (TF grad of vector == sum of components).
        if energy.ndim != 0:
            energy_scalar = energy.sum()
        else:
            energy_scalar = energy

        (x_grad,) = torch.autograd.grad(energy_scalar, x, create_graph=False)

        with torch.no_grad():
            x_next = x - STEP_LR * x_grad
            x_next = x_next.clamp(0.0, 1.0)
            x_mod = x_next.detach()

        # Save montage for GIF (expects NHWC)
        x_hwc = x_mod.detach().cpu().numpy()
        ims.append(_make_montage(x_hwc, grid_hw=(4, 4)))

        # if (i % 10) == 0:
        #     prof.stop()
        #     with open(osp.join(logdir, f"prof_ops_step_{i}.txt"), "w") as f:
        #         print(prof.key_averages().table(sort_by="cuda_time_total"), file=f)

        current_time = time.time()
        elapsed_time = current_time - start_time
        # print(f"Step {i}: Total runtime = {elapsed_time:.7f}s")


    # os.makedirs(logdir, exist_ok=True)
    # imageio.mimwrite(osp.join(logdir, "sample.gif"), list(ims))


if __name__ == "__main__":
    main()
