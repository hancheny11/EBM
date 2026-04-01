
"""
utils_pt.py — PyTorch port mirroring TF ops order (NHWC semantics)
------------------------------------------------------------------
This is a line-by-line, operator-faithful rewrite of the TensorFlow utilities.
The math order (conv -> optional scale/bias -> activation -> norms -> pool)
and attention equations are preserved. We only add NHWC<->NCHW transposes to
call torch.nn.functional ops, which expect NCHW.
"""

import os
import random
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class _FLAGS:
    spec_iter: int = 1
    spec_norm_val: float = 1.0
    downsample: bool = False
    spec_eval: bool = False
    norm: str = 'None'
    spec_norm: bool = True
    input_objects: int = 1
    cclass: bool = True
    use_attention: bool = False
    swish_act: bool = False

FLAGS = _FLAGS()


def set_seed(seed: int):
    import numpy, random
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

def swish(inp: torch.Tensor) -> torch.Tensor:
    return inp * torch.sigmoid(inp)

class ReplayBuffer:
    def __init__(self, size: int):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, ims: np.ndarray):
        batch_size = ims.shape[0]
        if self._next_idx >= len(self._storage):
            self._storage.extend(list(ims))
        else:
            if batch_size + self._next_idx < self._maxsize:
                self._storage[self._next_idx:self._next_idx + batch_size] = list(ims)
            else:
                split_idx = self._maxsize - self._next_idx
                self._storage[self._next_idx:] = list(ims)[:split_idx]
                self._storage[:batch_size - split_idx] = list(ims)[split_idx:]
        self._next_idx = (self._next_idx + ims.shape[0]) % self._maxsize

    def _encode_sample(self, idxes):
        ims = [self._storage[i] for i in idxes]
        return np.array(ims)

    def sample(self, batch_size: int) -> np.ndarray:
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


def _l2normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.pow(2).sum().sqrt() + eps)

def spectral_normed_weight(w: torch.Tensor,
                           u: torch.Tensor,
                           sigma_new: float,
                           iteration: int,
                           spec_eval: bool,
                           lower_bound: bool = False,
                           fc: bool = False):
    w_shape = list(w.shape)
    w_2d = w.view(-1, w_shape[-1])

    iters = 2 if fc else iteration
    u_hat = u
    v_hat = None
    for _ in range(iters):
        v_ = torch.matmul(u_hat, w_2d.t())
        v_hat = F.normalize(v_, p=2, dim=1, eps=1e-12)

        u_ = torch.matmul(v_hat, w_2d)
        u_hat = F.normalize(u_, p=2, dim=1, eps=1e-12)

    u_hat = u_hat.detach()
    v_hat = v_hat.detach()

    sigma = torch.matmul(torch.matmul(v_hat, w_2d), u_hat.t())

    if lower_bound:
        sigma = sigma + 1e-6
        w_norm = w_2d / sigma * torch.min(sigma, torch.ones_like(sigma)) * sigma_new
    else:
        w_norm = w_2d / sigma * sigma_new

    if not spec_eval:
        u = u_hat

    return w_norm.view(*w_shape), u

_u_registry = {}

def _get_or_create_u(key: str, out_dim: int, device: torch.device) -> torch.Tensor:
    if key not in _u_registry:
        _u_registry[key] = torch.randn(1, out_dim, device=device)
    return _u_registry[key]


def get_weight(name: str,
               shape: Tuple[int, ...],
               gain: float = np.sqrt(2.0),
               use_wscale: bool = False,
               fan_in: Optional[int] = None,
               spec_norm: bool = False,
               zero: bool = False,
               fc: bool = False,
               full_name: Optional[str] = None,
               device: Optional[torch.device] = None) -> torch.Tensor:
    if device is None:
        device = torch.device('cpu')

    if fan_in is None:
        fin = int(np.prod(shape[:-1]))
    else:
        fin = fan_in
    std = gain / (fin ** 0.5)

    if use_wscale:
        w = torch.randn(*shape, device=device) * std
    elif spec_norm:
        if zero:
            w = torch.randn(*shape, device=device) * 1e-10
        else:
            w = torch.randn(*shape, device=device)
        key = (full_name or name) + ":w"
        u = _get_or_create_u(key, shape[-1], device)
        w, u_new = spectral_normed_weight(
            w, u, sigma_new=float(FLAGS.spec_norm_val),
            iteration=int(FLAGS.spec_iter), spec_eval=bool(FLAGS.spec_eval),
            lower_bound=bool(zero), fc=bool(fc))
        _u_registry[key] = u_new
    else:
        if zero:
            w = torch.zeros(*shape, device=device)
        else:
            w = torch.empty(*shape, device=device)
            nn.init.xavier_uniform_(w)

    w.requires_grad_(True)
    return w

def pixel_norm(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    mean_sq = (x ** 2).mean(dim=(1, 2), keepdim=True)
    return x * torch.rsqrt(mean_sq + epsilon)


def batch_norm(inp: torch.Tensor, scale: Optional[torch.Tensor], bias: Optional[torch.Tensor], eps: float = 0.01):
    mean = inp.mean(dim=0, keepdim=True)
    var = ((inp - mean) ** 2).mean(dim=0, keepdim=True)
    out = (inp - mean) / torch.sqrt(var + eps)
    if bias is not None:
        out = out + bias
    if scale is not None:
        out = out * scale
    return out

def group_norm(inp: torch.Tensor, scale: Optional[torch.Tensor], bias: Optional[torch.Tensor], g: int = 32, eps: float = 1e-6, stop_batch: bool = False):
    n, h, w, c = inp.shape
    x = inp.view(n, h, w, c // g, g)
    mean = x.mean(dim=(1, 2, 4), keepdim=True)
    var = ((x - mean) ** 2).mean(dim=(1, 2, 4), keepdim=True)
    gain = torch.rsqrt(var + eps)
    out = gain * (x - mean)
    out = out.view(n, h, w, c)
    if scale is not None:
        out = out * scale
    if bias is not None:
        out = out + bias
    return out

def layer_norm(inp: torch.Tensor, scale: Optional[torch.Tensor], bias: Optional[torch.Tensor], eps: float = 1e-6):
    n, h, w, c = inp.shape
    mean = inp.mean(dim=(1, 2, 3), keepdim=True)
    var = ((inp - mean) ** 2).mean(dim=(1, 2, 3), keepdim=True)
    gain = torch.rsqrt(var + eps)
    out = gain * (inp - mean)
    if scale is not None:
        out = out * scale
    if bias is not None:
        out = out + bias
    return out


def _conv2d_nhwc(x: torch.Tensor, w: torch.Tensor, stride_hw: Tuple[int, int]) -> torch.Tensor:
    kh, kw, in_c, out_c = w.shape
    w_t = w.permute(3, 2, 0, 1).contiguous()
    x_t = x.permute(0, 3, 1, 2).contiguous()
    y_t = F.conv2d(x_t, w_t, bias=None, stride=stride_hw, padding=(kh//2, kw//2))
    y = y_t.permute(0, 2, 3, 1).contiguous()
    return y

def _conv2d_transpose_nhwc(x: torch.Tensor, w: torch.Tensor, out_hw: Tuple[int, int], stride_hw: Tuple[int, int]) -> torch.Tensor:
    kh, kw, in_c, out_c = w.shape
    w_tf = w.permute(0, 1, 3, 2).contiguous()
    w_pt = w_tf.permute(3, 2, 0, 1).contiguous()

    x_t = x.permute(0, 3, 1, 2).contiguous()
    y_t = F.conv_transpose2d(x_t, w_pt, bias=None, stride=stride_hw, padding=(kh//2, kw//2))
    y = y_t.permute(0, 2, 3, 1).contiguous()
    if y.shape[1] != out_hw[0] or y.shape[2] != out_hw[1]:
        y = F.interpolate(y.permute(0,3,1,2), size=out_hw, mode='nearest').permute(0,2,3,1).contiguous()
    return y

def hw_flatten(x: torch.Tensor) -> torch.Tensor:
    n, h, w, c = x.shape
    return x.view(n, h*w, c)

def conv_cond_concat(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    N, H, W, _ = x.shape
    yH, yW = y.shape[1], y.shape[2]
    if yH == 1 and yW == 1:
        y = y.expand(N, H, W, y.shape[3])
    return torch.cat([x, y / 10.0], dim=3)


def conv_block(inp: torch.Tensor,
               cweight: torch.Tensor,
               bweight: torch.Tensor,
               reuse: bool,
               scope: str,
               use_stride: bool = True,
               activation = F.leaky_relu,
               pn: bool = False,
               bn: bool = False,
               gn: bool = False,
               ln: bool = False,
               scale: Optional[torch.Tensor] = None,
               bias: Optional[torch.Tensor] = None,
               class_bias: Optional[torch.Tensor] = None,
               use_bias: bool = False,
               downsample: bool = False,
               stop_batch: bool = False,
               use_scale: bool = False,
               extra_bias: bool = False,
               average: bool = False,
               label: Optional[torch.Tensor] = None) -> torch.Tensor:
    stride_hw = (2, 2) if use_stride else (1, 1)
    if FLAGS.downsample:
        stride_hw = (1, 1)

    if not use_bias:
        b = 0.0
    else:
        b = bweight

    x = inp
    if extra_bias:
        b_extra = bias
        if label is not None:
            if b_extra.ndim == 1:
                b_extra = b_extra.view(1, -1)
            bias_batch = torch.matmul(label, b_extra)
            batch, dim = bias_batch.shape
            bias_reshaped = bias_batch.view(batch, 1, 1, dim)
        else:
            bias_reshaped = b_extra
        x = x + bias_reshaped

    y = _conv2d_nhwc(x, cweight, stride_hw)

    if use_scale:
        s = scale
        if label is not None:
            if s.ndim == 1:
                s = s.view(1, -1)
            scale_batch = torch.matmul(label, s) + class_bias
            batch, dim = scale_batch.shape
            s = scale_batch.view(batch, 1, 1, dim)
        y = y * s

    if use_bias:
        y = y + b

    if activation is not None:
        if activation is F.leaky_relu:
            y = F.leaky_relu(y, negative_slope=0.2)
        else:
            y = activation(y)

    if bn:
        y = batch_norm(y, scale, bias)
    if pn:
        y = pixel_norm(y)
    if gn:
        y = group_norm(y, scale, bias, stop_batch=stop_batch)
    if ln:
        y = layer_norm(y, scale, bias)

    if FLAGS.downsample and use_stride:
        y = F.avg_pool2d(y.permute(0,3,1,2), kernel_size=2, stride=2).permute(0,2,3,1).contiguous()

    return y


def conv_block_1d(inp: torch.Tensor,
                  cweight: torch.Tensor,
                  bweight: torch.Tensor,
                  reuse: bool,
                  scope: str,
                  activation = F.leaky_relu) -> torch.Tensor:
    N, L, C = inp.shape
    kw, _, C_out = cweight.shape
    x = inp.permute(0,2,1).contiguous()
    w = cweight.permute(2,1,0).contiguous()
    y = F.conv1d(x, w, bias=bweight, stride=1, padding=kw//2)
    y = y.permute(0,2,1).contiguous()
    if activation is not None:
        if activation is F.leaky_relu:
            y = F.leaky_relu(y, negative_slope=0.2)
        else:
            y = activation(y)
    return y


def conv_block_3d(inp: torch.Tensor,
                  cweight: torch.Tensor,
                  bweight: torch.Tensor,
                  reuse: bool,
                  scope: str,
                  use_stride: bool = True,
                  activation = F.leaky_relu,
                  pn: bool = False,
                  bn: bool = False,
                  gn: bool = False,
                  ln: bool = False,
                  scale: Optional[torch.Tensor] = None,
                  bias: Optional[torch.Tensor] = None,
                  use_bias: bool = False) -> torch.Tensor:
    kd, kh, kw, in_c, out_c = cweight.shape
    x = inp.permute(0,4,1,2,3).contiguous()
    w = cweight.permute(4,3,0,1,2).contiguous()
    stride = (1,2,2) if use_stride else (1,1,1)
    y = F.conv3d(x, w, bias=bweight if use_bias else None, stride=stride, padding=(kd//2, kh//2, kw//2))
    y = y.permute(0,2,3,4,1).contiguous()

    if activation is not None:
        y = F.leaky_relu(y, negative_slope=0.1) if activation is F.leaky_relu else activation(y)

    if bn:
        y = batch_norm(y, scale, bias)
    if pn:
        y = pixel_norm(y)
    if gn:
        y = group_norm(y, scale, bias)
    if ln:
        y = layer_norm(y, scale, bias)

    if FLAGS.downsample and use_stride:
        # This is a placeholder; the original 3D path isn't used by models.py
        pass
    return y


def attention(inp: torch.Tensor,
              q: torch.Tensor, q_b: torch.Tensor,
              k: torch.Tensor, k_b: torch.Tensor,
              v: torch.Tensor, v_b: torch.Tensor,
              gamma: torch.Tensor,
              reuse: bool, scope: str,
              stop_at_grad: bool = False,
              seperate: bool = False,
              scale: bool = False,
              train: bool = False,
              dropout: float = 0.0) -> torch.Tensor:
    conv_q = conv_block(inp, q, q_b, reuse=reuse, scope=scope, use_stride=False,
                        activation=None, use_bias=True, pn=False, bn=False, gn=False)
    conv_k = conv_block(inp, k, k_b, reuse=reuse, scope=scope, use_stride=False,
                        activation=None, use_bias=True, pn=False, bn=False, gn=False)
    conv_v = conv_block(inp, v, v_b, reuse=reuse, scope=scope, use_stride=False,
                        pn=False, bn=False, gn=False)

    c_num = float(conv_q.shape[-1])
    s = torch.bmm(hw_flatten(conv_q), hw_flatten(conv_k).transpose(1,2))

    if scale:
        s = s / (c_num ** 0.5)

    if train:
        s = F.dropout(s, p=0.1, training=True)

    beta = torch.softmax(s, dim=-1)
    o = torch.bmm(beta, hw_flatten(conv_v))
    o = o.view_as(inp)
    out = inp + gamma * o
    return out if not seperate else gamma * o


def attention_2d(inp: torch.Tensor,
                 q: torch.Tensor, q_b: torch.Tensor,
                 k: torch.Tensor, k_b: torch.Tensor,
                 v: torch.Tensor, v_b: torch.Tensor,
                 reuse: bool, scope: str,
                 stop_at_grad: bool = False,
                 seperate: bool = False,
                 scale: bool = False) -> torch.Tensor:
    N, H, W, C = inp.shape
    inp_compact = inp.view(N * FLAGS.input_objects * H, C)
    f_q = torch.matmul(inp_compact, q) + q_b
    f_k = torch.matmul(inp_compact, k) + k_b
    f_v = F.leaky_relu(torch.matmul(inp_compact, v) + v_b, negative_slope=0.2)

    f_q = f_q.view(N, H, W, -1)
    f_k = f_k.view(N, H, W, -1)
    f_v = f_v.view(N, H, W, C)

    s = torch.matmul(f_k, f_q.transpose(-1,-2))
    if scale:
        s = s / (32 ** 0.5)
    beta = torch.softmax(s, dim=-1)

    o = torch.matmul(beta, f_v).view_as(inp) + inp
    return o


def init_conv_weight(weights, scope, k, c_in, c_out, spec_norm=True, zero=False, scale=1.0, classes=1, device=None):
    if spec_norm:
        spec_norm = FLAGS.spec_norm

    cw = {}
    cw['c'] = get_weight('c', (k, k, c_in, c_out), spec_norm=spec_norm, zero=zero, full_name=f'{scope}/c', device=device)
    cw['b'] = torch.zeros(c_out, device=device, requires_grad=True)

    if classes != 1:
        cw['g']  = torch.ones(classes, c_out, device=device, requires_grad=True)
        cw['gb'] = torch.zeros(classes, c_in, device=device, requires_grad=True)
    else:
        cw['g']  = torch.ones(c_out, device=device, requires_grad=True)
        cw['gb'] = torch.zeros(c_in, device=device, requires_grad=True)

    cw['cb'] = torch.zeros(c_in, device=device, requires_grad=True)
    weights[scope] = cw

def init_convt_weight(weights, scope, k, c_in, c_out, spec_norm=True, zero=False, scale=1.0, classes=1, device=None):
    if spec_norm:
        spec_norm = FLAGS.spec_norm

    cw = {}
    cw['c'] = get_weight('c', (k, k, c_in, c_out), spec_norm=spec_norm, zero=zero, full_name=f'{scope}/c', device=device)
    cw['b'] = torch.zeros(c_in, device=device, requires_grad=True)

    if classes != 1:
        cw['g']  = torch.ones(classes, c_in, device=device, requires_grad=True)
        cw['gb'] = torch.zeros(classes, c_out, device=device, requires_grad=True)
    else:
        cw['g']  = torch.ones(c_in, device=device, requires_grad=True)
        cw['gb'] = torch.zeros(c_out, device=device, requires_grad=True)

    cw['cb'] = torch.zeros(c_in, device=device, requires_grad=True)
    weights[scope] = cw

def init_attention_weight(weights, scope, c_in, k, trainable_gamma=True, spec_norm=True, device=None):
    if spec_norm:
        spec_norm = FLAGS.spec_norm

    aw = {}
    k = int(k)
    aw['q']   = get_weight('atten_q', (1, 1, c_in, k), spec_norm=spec_norm, full_name=f'{scope}/atten_q', device=device)
    aw['q_b'] = torch.zeros(k, device=device, requires_grad=True)
    aw['k']   = get_weight('atten_k', (1, 1, c_in, k), spec_norm=spec_norm, full_name=f'{scope}/atten_k', device=device)
    aw['k_b'] = torch.zeros(k, device=device, requires_grad=True)
    aw['v']   = get_weight('atten_v', (1, 1, c_in, c_in), spec_norm=spec_norm, full_name=f'{scope}/atten_v', device=device)
    aw['v_b'] = torch.zeros(c_in, device=device, requires_grad=True)
    aw['gamma'] = torch.zeros(1, device=device, requires_grad=trainable_gamma)
    weights[scope] = aw

def init_fc_weight(weights, scope, c_in, c_out, spec_norm=True, device=None):
    fw = {}
    if spec_norm:
        spec_norm = FLAGS.spec_norm
    fw['w'] = get_weight('w', (c_in, c_out), spec_norm=spec_norm, fc=True, full_name=f'{scope}/w', device=device)
    fw['b'] = torch.zeros(c_out, device=device, requires_grad=True)
    weights[scope] = fw

def init_res_weight(weights, scope, k, c_in, c_out, hidden_dim=None, spec_norm=True, res_scale=1.0, classes=1, device=None):
    if hidden_dim is None:
        hidden_dim = c_in
    if spec_norm:
        spec_norm = FLAGS.spec_norm

    init_conv_weight(weights, scope + '_res_c1', k, c_in, c_out, spec_norm=spec_norm, scale=res_scale, classes=classes, device=device)
    init_conv_weight(weights, scope + '_res_c2', k, c_out, c_out, spec_norm=spec_norm, zero=True, scale=res_scale, classes=classes, device=device)
    if c_in != c_out:
        init_conv_weight(weights, scope + '_res_adaptive', k, c_in, c_out, spec_norm=spec_norm, scale=res_scale, classes=classes, device=device)


def smart_conv_block(inp, weights, reuse, scope, use_stride=True, **kwargs):
    w = weights[scope]
    return conv_block(inp, w['c'], w['b'], reuse, scope,
                      scale=w['g'], bias=w['gb'], class_bias=w['cb'],
                      use_stride=use_stride, **kwargs)

def smart_convt_block(inp, weights, reuse, scope, output_dim, upsample=True, label=None):
    w = weights[scope]

    cweight = w['c']
    bweight = w['b']
    scale   = w['g']
    bias    = w['gb']
    class_bias = w['cb']

    stride_hw = (2,2) if upsample else (1,1)

    if label is not None:
        if bias.ndim == 1:
            b_in = bias.view(1, -1)
        else:
            b_in = bias
        bias_batch = torch.matmul(label, b_in)
        batch, dim = bias_batch.shape
        bias_reshaped = bias_batch.view(batch, 1, 1, dim)
        inp = inp + bias_reshaped

    conv_output = _conv2d_transpose_nhwc(inp, cweight, (output_dim, output_dim), stride_hw)

    if label is not None:
        s = scale
        if s.ndim == 1:
            s = s.view(1,-1)
        scale_batch = torch.matmul(label, s) + class_bias
        batch, dim = scale_batch.shape
        s = scale_batch.view(batch, 1, 1, dim)
        conv_output = conv_output * s

    conv_output = F.leaky_relu(conv_output, negative_slope=0.2)
    return conv_output

def smart_res_block(inp, weights, reuse, scope, downsample=True, adaptive=True, stop_batch=False, upsample=False,
                    label=None, act=F.leaky_relu, dropout=False, train=False, **kwargs):
    c1 = smart_conv_block(inp, weights, reuse, scope + '_res_c1', use_stride=False,
                          activation=None, extra_bias=True, label=label, **kwargs)
    if dropout:
        c1 = F.dropout(c1, p=0.5, training=train)
    c1 = act(c1) if act is not None else c1
    c2 = smart_conv_block(c1, weights, reuse, scope + '_res_c2', use_stride=False,
                          activation=None, use_scale=True, extra_bias=True, label=label, **kwargs)

    if adaptive:
        c_bypass = smart_conv_block(inp, weights, reuse, scope + '_res_adaptive', use_stride=False, activation=None, **kwargs)
    else:
        c_bypass = inp

    res = c2 + c_bypass

    if upsample:
        n, h, w, c = res.shape
        res = torch.nn.functional.interpolate(res.permute(0,3,1,2), scale_factor=2, mode='nearest').permute(0,2,3,1).contiguous()
    elif downsample:
        res = torch.nn.functional.avg_pool2d(res.permute(0,3,1,2), kernel_size=2, stride=2).permute(0,2,3,1).contiguous()

    res = act(res) if act is not None else res
    return res

def smart_res_block_optim(inp, weights, reuse, scope, **kwargs):
    c1 = smart_conv_block(inp, weights, reuse, scope + '_res_c1', use_stride=False, activation=None, **kwargs)
    c1 = torch.nn.functional.leaky_relu(c1, negative_slope=0.2)
    c2 = smart_conv_block(c1, weights, reuse, scope + '_res_c2', use_stride=False, activation=None, **kwargs)

    inp_ds = torch.nn.functional.avg_pool2d(inp.permute(0,3,1,2), kernel_size=2, stride=2).permute(0,2,3,1).contiguous()
    c_bypass = smart_conv_block(inp_ds, weights, reuse, scope + '_res_adaptive', use_stride=False, activation=None, **kwargs)
    c2_ds = torch.nn.functional.avg_pool2d(c2.permute(0,3,1,2), kernel_size=2, stride=2).permute(0,2,3,1).contiguous()

    res = c2_ds + c_bypass
    return c2_ds

def groupsort(k=4):
    def sortact(inp: torch.Tensor):
        old_shape = inp.shape
        x = inp.view(-1, 4)
        x_sorted, _ = torch.sort(x, dim=1)
        return x_sorted.view(*old_shape)
    return sortact

def smart_atten_block(inp, weights, reuse, scope, **kwargs):
    w = weights[scope]
    return attention(inp, w['q'], w['q_b'], w['k'], w['k_b'], w['v'], w['v_b'], w['gamma'], reuse, scope, **kwargs)

def smart_fc_block(inp, weights, reuse, scope, use_bias=True):
    w = weights[scope]
    out = torch.matmul(inp, w['w'])
    if use_bias:
        out = out + w['b']
    return out


def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) for i, path in zip(labels, paths) for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images
