
# imagenet_demo_pt.py — PyTorch rewrite (operator-faithful)
# ---------------------------------------------------------
# Mirrors the original TensorFlow sampling script:
# - Keeps NHWC tensors (128x128x3) like TF
# - Same FLAGS defaults and update math order
# - Uses models_pt.ResNet128 and utils_pt.FLAGS
#
# Usage:
#   python imagenet_demo_pt.py --num_steps 200 --step_lr 180 --batch_size 16
#
import argparse
import numpy as np
import imageio
import torch
import torch.nn.functional as F

# Use the PyTorch ports that mirror the TF ops
import models_pt
import utils_pt
from utils_pt import FLAGS

import time

from torch.profiler import profile, ProfilerActivity

def rescale_im(im: np.ndarray) -> np.ndarray:
    return np.clip(im * 256.0, 0, 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='cachedir')
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--step_lr', type=float, default=180.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--exp', type=str, default='default')
    parser.add_argument('--resume_iter', type=int, default=-1)
    parser.add_argument('--spec_norm', action='store_true', default=True)
    parser.add_argument('--no-spec_norm', dest='spec_norm', action='store_false')
    parser.add_argument('--cclass', action='store_true', default=True)
    parser.add_argument('--no-cclass', dest='cclass', action='store_false')
    parser.add_argument('--use_attention', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    utils_pt.set_seed(args.seed)

    # Mirror TF flags -> utils_pt.FLAGS
    FLAGS.spec_norm = bool(args.spec_norm)
    FLAGS.cclass = bool(args.cclass)
    FLAGS.use_attention = bool(args.use_attention)

    device = torch.device(args.device)

    # Build model + weights (same API as TF port)
    model = models_pt.ResNet128(num_filters=64, num_channels=3, train=False)
    weights = model.construct_weights(scope='')

    # Move weights to device
    for grp in weights.values():
        for k, t in list(grp.items()):
            if isinstance(t, torch.Tensor):
                grp[k] = t.to(device)

    # Random labels (one-hot), like original
    lx = np.random.permutation(1000)[:args.batch_size]
    labels_np = np.eye(1000, dtype=np.float32)[lx]
    labels = torch.from_numpy(labels_np).to(device)

    # Initialize sampling with uniform noise on [0,1]
    x_mod = np.random.uniform(0, 1, size=(args.batch_size, 128, 128, 3)).astype(np.float32)
    x_mod = torch.from_numpy(x_mod).to(device)

    ims = []

    
    # --- Begin step 0 math (match TF order) ---
    # x_mod = x_mod + Normal(0, 0.005)
    x_mod = x_mod + torch.randn_like(x_mod) * 0.005
    x_mod = x_mod.clone().detach().requires_grad_(True)

    # energy_noise = model.forward(x_mod, weights, label=LABEL, ...)
    # No stop_at_grad or stop_batch behavior change in PyTorch port; mirrored internally.
    energy_noise = model.forward(x_mod, weights, label=labels, reuse=True, stop_at_grad=False, stop_batch=True)

    x_grad = torch.autograd.grad(energy_noise.sum(), x_mod)[0]
    lr = float(args.step_lr)
    x_mod = (x_mod - lr * x_grad).clamp(0.0, 1.0).detach() 

    # x_grad = d(energy_noise)/dx
    # In TF they used tf.gradients(energy_noise, [x_mod])[0];
    # Here we backprop the summed energy to get gradient wrt x.
    # x_mod.requires_grad_(True)
    # energy_noise_sum = energy_noise.sum()
    # x_grad, = torch.autograd.grad(energy_noise_sum, x_mod, create_graph=False)

    # x_last = x_mod - (lr) * x_grad
    # lr = float(args.step_lr)
    # x_last = x_mod - lr * x_grad

    # x_mod = clip(x_last, 0, 1)
    # x_mod = x_last.clamp_(0.0, 1.0)
    # x_output = x_mod
    # --- End step 0 math ---

    start_time = time.time()

    # Run the remaining (num_steps) evaluations, recording the images each time
    for i in range(args.num_steps):

        # if i % 10 == 0:
        #     prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False)
        #     prof.start()

        x_mod = x_mod.clone().detach().requires_grad_(True)

        # Forward energy
        energy_noise = model.forward(x_mod, weights, label=labels, reuse=True, stop_at_grad=False, stop_batch=True)

        # Compute grad wrt current x
        # x_mod.requires_grad_(True)
        x_grad = torch.autograd.grad(energy_noise.sum(), x_mod)[0]
        x_mod = (x_mod - lr * x_grad).clamp(0.0, 1.0).detach()

        # energy_sum = energy_noise.sum()
        # x_grad, = torch.autograd.grad(energy_sum, x_mod, create_graph=False)

        # Langevin-like update (gradient descent on energy)
        # x_last = x_mod - lr * x_grad
        # x_mod = x_last.clamp(0.0, 1.0).detach()  # detach graph like TF run loop

        # Save a frame
        with torch.no_grad():
            frame = x_mod.detach().cpu().numpy()
            frame = rescale_im(frame).reshape((4, 4, 128, 128, 3)).transpose((0, 2, 1, 3, 4)).reshape((512, 512, 3))
            ims.append(frame)

        # if i % 10 == 0:
        #     prof.stop()
        #     with open(f"prof_ops_step_{i}.txt", "w") as f:
        #         print(prof.key_averages().table(sort_by="cuda_time_total"), file=f)


        current_time = time.time()
        elapsed_time = current_time - start_time
        print(f"Step {i}: Total runtime = {elapsed_time:.7f}s")

    imageio.mimwrite('sample.gif', ims)

if __name__ == '__main__':
    main()
