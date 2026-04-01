import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration values (replacing TensorFlow FLAGS).
# You can modify these from outside after import if you need different values.
# ---------------------------------------------------------------------------

SPEC_ITER = 1             # formerly FLAGS.spec_iter
SPEC_NORM_VAL = 1.0       # formerly FLAGS.spec_norm_val
GLOBAL_DOWNSAMPLE = False # formerly FLAGS.downsample
SPEC_EVAL = False         # formerly FLAGS.spec_eval

SWISH_ACT = False         # formerly FLAGS.swish_act
CCLASS = False            # formerly FLAGS.cclass
NORM = 'None'             # formerly FLAGS.norm
DATASOURCE = None         # formerly FLAGS.datasource
COMB_MASK = False         # formerly FLAGS.comb_mask
COND_FUNC = 1             # formerly FLAGS.cond_func
INPUT_OBJECTS = 1         # formerly FLAGS.input_objects
USE_ATTENTION = False     # formerly FLAGS.use_attention
SPEC_NORM = False         # formerly FLAGS.spec_norm (default)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_median(v: torch.Tensor) -> torch.Tensor:
    """
    Approximate median using the same top-k trick as the original TF code.
    """
    v = v.reshape(-1)
    m = v.numel() // 2
    if m == 0:
        return v.mean()
    values, _ = torch.topk(v, k=m)
    return values[m - 1]


def set_seed(seed: int):
    import numpy
    import random as pyrandom

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    pyrandom.seed(seed)


def swish(inp: torch.Tensor) -> torch.Tensor:
    return inp * torch.sigmoid(inp)


class ReplayBuffer(object):
    def __init__(self, size: int):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
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
        self._next_idx = (self._next_idx + batch_size) % self._maxsize

    def _encode_sample(self, idxes):
        ims = []
        for i in idxes:
            ims.append(self._storage[i])
        return np.array(ims)

    def sample(self, batch_size: int):
        """Sample a batch of experiences."""
        idxes = [random.randint(0, len(self._storage) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes)


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def _fan_in_from_shape(shape):
    if len(shape) <= 1:
        return shape[0]
    return int(np.prod(shape[:-1]))


def get_weight(
        name,
        shape,
        gain=np.sqrt(2),
        use_wscale=False,
        fan_in=None,
        spec_norm=False,
        zero=False,
        fc=False):
    """
    Create a torch Parameter roughly mimicking the TF initialization logic.
    For conv weights, `shape` is [k, k, c_in, c_out].
    For fc weights, `shape` is [c_in, c_out].
    """
    if fan_in is None:
        fan_in = _fan_in_from_shape(shape)
    std = gain / np.sqrt(fan_in)  # He init

    # Allocate tensor in a TF-like layout, but we'll rearrange for PyTorch where needed
    w = torch.empty(*shape)

    if zero:
        nn.init.zeros_(w)
    else:
        # Match the spirit of the original code:
        # - use_wscale: normal(0,1) scaled by std
        # - spec_norm: normal(0,1) then spectral normalization
        # - otherwise: xavier for conv / fc
        if use_wscale:
            nn.init.normal_(w, mean=0.0, std=1.0)
            w = w * std
        elif spec_norm:
            nn.init.normal_(w, mean=0.0, std=1.0)
        else:
            if len(shape) in (2, 4):
                nn.init.xavier_uniform_(w)
            else:
                nn.init.normal_(w, mean=0.0, std=std)

    if spec_norm:
        w = spectral_normed_weight(w, name, lower_bound=zero, iteration=2 if fc else 1, fc=fc)

    return nn.Parameter(w)


def pixel_norm(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Pixel-wise feature vector normalization.

    For 4D input (N, C, H, W) it normalizes over H and W.
    For 5D input (N, C, D, H, W) it normalizes over D, H, W.
    """
    if x.dim() == 4:
        dims = (2, 3)
    elif x.dim() == 5:
        dims = (2, 3, 4)
    else:
        dims = tuple(range(1, x.dim()))
    return x * torch.rsqrt(x.pow(2).mean(dim=dims, keepdim=True) + epsilon)


# helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        def sampler(x): return random.sample(x, nb_samples)
    else:
        def sampler(x): return x
    images = [(i, os.path.join(path, image))
              for i, path in zip(labels, paths)
              for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images


# The TF checkpoint restore helpers are very specific to TF's graph / Session
# model. In PyTorch you should instead use model.load_state_dict(torch.load(...))
# directly on your nn.Module. We keep the names for compatibility but raise
# a clear error instead of silently relying on TensorFlow.

def optimistic_restore(*args, **kwargs):
    raise NotImplementedError(
        "optimistic_restore is TensorFlow-specific in the original code. "
        "Please use torch.load and nn.Module.load_state_dict instead."
    )


def optimistic_remap_restore(*args, **kwargs):
    raise NotImplementedError(
        "optimistic_remap_restore is TensorFlow-specific in the original code. "
        "Please use torch.load and nn.Module.load_state_dict instead."
    )


def remap_restore(*args, **kwargs):
    raise NotImplementedError(
        "remap_restore is TensorFlow-specific in the original code. "
        "Please use torch.load and nn.Module.load_state_dict instead."
    )


# ---------------------------------------------------------------------------
# Network weight initializers
# ---------------------------------------------------------------------------

def init_conv_weight(
        weights,
        scope,
        k,
        c_in,
        c_out,
        spec_norm=True,
        zero=False,
        scale=1.0,
        classes=1):

    if spec_norm:
        spec_norm = SPEC_NORM

    conv_weights = {}
    # store as TF-style shape [k, k, c_in, c_out], but conv ops will permute
    conv_weights['c'] = get_weight(
        f'{scope}.c', [k, k, c_in, c_out], spec_norm=spec_norm, zero=zero)

    conv_weights['b'] = nn.Parameter(torch.zeros(c_out))

    if classes != 1:
        conv_weights['g'] = nn.Parameter(torch.ones(classes, c_out))
        conv_weights['gb'] = nn.Parameter(torch.zeros(classes, c_in))
    else:
        conv_weights['g'] = nn.Parameter(torch.ones(c_out))
        conv_weights['gb'] = nn.Parameter(torch.zeros(c_in))

    conv_weights['cb'] = nn.Parameter(torch.zeros(c_in))

    weights[scope] = conv_weights


def init_convt_weight(
        weights,
        scope,
        k,
        c_in,
        c_out,
        spec_norm=True,
        zero=False,
        scale=1.0,
        classes=1):

    if spec_norm:
        spec_norm = SPEC_NORM

    conv_weights = {}
    # We keep the same [k, k, c_in, c_out] shape; transposed conv will handle layout
    conv_weights['c'] = get_weight(
        f'{scope}.c', [k, k, c_in, c_out], spec_norm=spec_norm, zero=zero)

    conv_weights['b'] = nn.Parameter(torch.zeros(c_in))

    if classes != 1:
        conv_weights['g'] = nn.Parameter(torch.ones(classes, c_in))
        conv_weights['gb'] = nn.Parameter(torch.zeros(classes, c_out))
    else:
        conv_weights['g'] = nn.Parameter(torch.ones(c_in))
        conv_weights['gb'] = nn.Parameter(torch.zeros(c_out))

    conv_weights['cb'] = nn.Parameter(torch.zeros(c_in))

    weights[scope] = conv_weights


def init_attention_weight(
        weights,
        scope,
        c_in,
        k,
        trainable_gamma=True,
        spec_norm=True):

    if spec_norm:
        spec_norm = SPEC_NORM

    atten_weights = {}
    atten_weights['q'] = get_weight(
        f'{scope}.atten_q', [1, 1, c_in, k], spec_norm=spec_norm)
    atten_weights['q_b'] = nn.Parameter(torch.zeros(k))
    atten_weights['k'] = get_weight(
        f'{scope}.atten_k', [1, 1, c_in, k], spec_norm=spec_norm)
    atten_weights['k_b'] = nn.Parameter(torch.zeros(k))
    atten_weights['v'] = get_weight(
        f'{scope}.atten_v', [1, 1, c_in, c_in], spec_norm=spec_norm)
    atten_weights['v_b'] = nn.Parameter(torch.zeros(c_in))
    atten_weights['gamma'] = nn.Parameter(torch.zeros(1))

    weights[scope] = atten_weights


def init_fc_weight(weights, scope, c_in, c_out, spec_norm=True):
    fc_weights = {}

    if spec_norm:
        spec_norm = SPEC_NORM

    fc_weights['w'] = get_weight(
        f'{scope}.w', [c_in, c_out], spec_norm=spec_norm, fc=True)
    fc_weights['b'] = nn.Parameter(torch.zeros(c_out))

    weights[scope] = fc_weights


def init_res_weight(
        weights,
        scope,
        k,
        c_in,
        c_out,
        hidden_dim=None,
        spec_norm=True,
        res_scale=1.0,
        classes=1):

    if hidden_dim is None:
        hidden_dim = c_in

    if spec_norm:
        spec_norm = SPEC_NORM

    init_conv_weight(
        weights,
        scope + '_res_c1',
        k,
        c_in,
        c_out,
        spec_norm=spec_norm,
        classes=classes)
    init_conv_weight(
        weights,
        scope + '_res_c2',
        k,
        c_out,
        c_out,
        spec_norm=spec_norm,
        zero=True,
        classes=classes)

    if c_in != c_out:
        init_conv_weight(
            weights,
            scope + '_res_adaptive',
            k,
            c_in,
            c_out,
            spec_norm=spec_norm,
            classes=classes)


# ---------------------------------------------------------------------------
# Network forward helpers
# ---------------------------------------------------------------------------

def _conv2d_nhwc(inp: torch.Tensor, weight_tf: torch.Tensor, bias=None, stride=1):
    """
    Helper that assumes inp is NCHW internally but weight is stored in TF-style
    [k, k, c_in, c_out]. However, in this file we treat inp as NCHW everywhere,
    so stride is an int or (int, int).
    """
    # weight_tf: [k, k, c_in, c_out] -> [c_out, c_in, k, k]
    w = weight_tf.permute(3, 2, 0, 1)
    return F.conv2d(inp, w, bias=bias, stride=stride,
                    padding=w.shape[2] // 2)


def smart_conv_block(inp, weights, reuse, scope, use_stride=True, **kwargs):
    conv_weights = weights[scope]
    return conv_block(
        inp,
        conv_weights['c'],
        conv_weights['b'],
        reuse,
        scope,
        scale=conv_weights['g'],
        bias=conv_weights['gb'],
        class_bias=conv_weights['cb'],
        use_stride=use_stride,
        **kwargs)


def smart_convt_block(
        inp,
        weights,
        reuse,
        scope,
        output_dim,
        upsample=True,
        label=None):
    conv_weights = weights[scope]

    cweight = conv_weights['c']
    bweight = conv_weights['b']
    scale = conv_weights['g']
    bias = conv_weights['gb']
    class_bias = conv_weights['cb']

    stride = 2 if upsample else 1

    if label is not None:
        if bias.dim() == 1:
            bias_mat = bias.view(1, -1)
        else:
            bias_mat = bias
        bias_batch = label @ bias_mat
        bias_batch = bias_batch.view(bias_batch.size(0), bias_batch.size(1), 1, 1)
        inp = inp + bias_batch

    # Transposed conv: we upsample using nearest neighbor then do a regular conv
    if upsample:
        inp = F.interpolate(inp, scale_factor=2, mode='nearest')

    # cweight: [k, k, c_in, c_out] where c_in matches inp channels
    w = cweight.permute(3, 2, 0, 1)
    conv_output = F.conv2d(inp, w, bias=bweight, stride=1, padding=w.shape[2] // 2)

    if label is not None:
        if scale.dim() == 1:
            scale_mat = scale.view(1, -1)
        else:
            scale_mat = scale
        scale_batch = label @ scale_mat + class_bias
        scale_batch = scale_batch.view(scale_batch.size(0), scale_batch.size(1), 1, 1)
        conv_output = conv_output * scale_batch

    conv_output = F.leaky_relu(conv_output, negative_slope=0.1)

    return conv_output


def smart_res_block(
        inp,
        weights,
        reuse,
        scope,
        downsample=True,
        adaptive=True,
        stop_batch=False,
        upsample=False,
        label=None,
        act=F.leaky_relu,
        dropout=False,
        train=False,
        **kwargs):
    # c1
    c1 = smart_conv_block(
        inp,
        weights,
        reuse,
        scope + '_res_c1',
        use_stride=False,
        activation=None,
        extra_bias=True,
        label=label,
        **kwargs)

    if dropout and train:
        c1 = F.dropout(c1, p=0.5, training=train)

    c1 = act(c1)

    c2 = smart_conv_block(
        c1,
        weights,
        reuse,
        scope + '_res_c2',
        use_stride=False,
        activation=None,
        use_scale=True,
        extra_bias=True,
        label=label,
        **kwargs)

    if adaptive and (scope + '_res_adaptive') in weights:
        c_bypass = smart_conv_block(
            inp,
            weights,
            reuse,
            scope + '_res_adaptive',
            use_stride=False,
            activation=None,
            **kwargs)
    else:
        c_bypass = inp

    res = c2 + c_bypass

    if upsample:
        res = F.interpolate(res, scale_factor=2, mode='nearest')
    elif downsample:
        res = F.avg_pool2d(res, kernel_size=2, stride=2)

    res = act(res)

    return res


def smart_res_block_optim(inp, weights, reuse, scope, **kwargs):
    c1 = smart_conv_block(
        inp,
        weights,
        reuse,
        scope + '_res_c1',
        use_stride=False,
        activation=None,
        **kwargs)
    c1 = F.leaky_relu(c1, negative_slope=0.1)
    c2 = smart_conv_block(
        c1,
        weights,
        reuse,
        scope + '_res_c2',
        use_stride=False,
        activation=None,
        **kwargs)

    # average pooling on input and c2
    inp_ds = F.avg_pool2d(inp, kernel_size=2, stride=2)
    c_bypass = smart_conv_block(
        inp_ds,
        weights,
        reuse,
        scope + '_res_adaptive',
        use_stride=False,
        activation=None,
        **kwargs)
    c2_ds = F.avg_pool2d(c2, kernel_size=2, stride=2)

    res = c2_ds + c_bypass

    return res


def groupsort(k=4):
    def sortact(inp: torch.Tensor):
        orig_shape = inp.shape
        inp_flat = inp.view(-1, k)
        inp_sorted, _ = torch.sort(inp_flat, dim=-1)
        return inp_sorted.view(orig_shape)
    return sortact


def smart_atten_block(inp, weights, reuse, scope, **kwargs):
    w = weights[scope]
    return attention(
        inp,
        w['q'],
        w['q_b'],
        w['k'],
        w['k_b'],
        w['v'],
        w['v_b'],
        w['gamma'],
        reuse,
        scope,
        **kwargs)


def smart_fc_block(inp, weights, reuse, scope, use_bias=True):
    fc_weights = weights[scope]
    output = inp @ fc_weights['w']
    if use_bias:
        output = output + fc_weights['b']
    return output


# ---------------------------------------------------------------------------
# Convolution / normalization helpers
# ---------------------------------------------------------------------------

def conv_block(
        inp,
        cweight,
        bweight,
        reuse,
        scope,
        use_stride=True,
        activation=F.leaky_relu,
        pn=False,
        bn=False,
        gn=False,
        ln=False,
        scale=None,
        bias=None,
        class_bias=None,
        use_bias=False,
        downsample=False,
        stop_batch=False,
        use_scale=False,
        extra_bias=False,
        average=False,
        label=None):
    """Perform conv, optional normalization, nonlinearity, and optional pool.

    Expects inp to be NCHW and cweight to have shape [k, k, c_in, c_out].
    """
    stride = 2 if use_stride else 1

    if GLOBAL_DOWNSAMPLE:
        # In the TF code this disables stride and applies avg pooling later.
        stride_conv = 1
    else:
        stride_conv = stride

    # Extra bias is applied to the input as a conditional bias
    if extra_bias and bias is not None:
        if label is not None:
            if bias.dim() == 1:
                bias_mat = bias.view(1, -1)
            else:
                bias_mat = bias
            # label: [B, classes], bias_mat: [classes, C_in] -> [B, C_in]
            bias_batch = label @ bias_mat
            bias_batch = bias_batch.view(bias_batch.size(0), bias_batch.size(1), 1, 1)
        else:
            bias_batch = bias.view(1, -1, 1, 1)
        inp = inp + bias_batch

    # cweight is stored as [k, k, c_in, c_out]; convert for PyTorch conv
    w = cweight.permute(3, 2, 0, 1)
    b = bweight if use_bias else None
    conv_output = F.conv2d(inp, w, bias=b, stride=stride_conv,
                           padding=w.shape[2] // 2)

    if use_scale and scale is not None:
        if label is not None:
            if scale.dim() == 1:
                scale_mat = scale.view(1, -1)
            else:
                scale_mat = scale
            scale_batch = label @ scale_mat
            if class_bias is not None:
                scale_batch = scale_batch + class_bias
            scale_batch = scale_batch.view(scale_batch.size(0), scale_batch.size(1), 1, 1)
            conv_output = conv_output * scale_batch
        else:
            conv_output = conv_output * scale.view(1, -1, 1, 1)

    if activation is not None:
        conv_output = activation(conv_output)

    if bn:
        conv_output = batch_norm(conv_output, scale=None, bias=None)
    if pn:
        conv_output = pixel_norm(conv_output)
    if gn:
        conv_output = group_norm(conv_output, scale, bias, stop_batch=stop_batch)
    if ln:
        conv_output = layer_norm(conv_output, scale, bias)

    if GLOBAL_DOWNSAMPLE and use_stride:
        conv_output = F.avg_pool2d(conv_output, kernel_size=2, stride=2)

    return conv_output


def conv_block_1d(
        inp,
        cweight,
        bweight,
        reuse,
        scope,
        activation=F.leaky_relu):
    """
    1D convolution block.
    Expects inp to be [N, L, C_in] (TF-style); converts internally to NCL.
    """
    # Convert to NCL
    inp_ncl = inp.permute(0, 2, 1)
    # cweight: [k, c_in, c_out] -> [c_out, c_in, k]
    w = cweight.permute(2, 1, 0)
    conv_output = F.conv1d(inp_ncl, w, bias=bweight, stride=1, padding=w.shape[2] // 2)
    if activation is not None:
        conv_output = activation(conv_output)
    # Back to [N, L, C]
    return conv_output.permute(0, 2, 1)


def conv_block_3d(
        inp,
        cweight,
        bweight,
        reuse,
        scope,
        use_stride=True,
        activation=F.leaky_relu,
        pn=False,
        bn=False,
        gn=False,
        ln=False,
        scale=None,
        bias=None,
        use_bias=False):
    """
    3D convolution block.

    Expects inp to be [N, C, D, H, W] and cweight to be [k, k, k, c_in, c_out].
    """
    stride = (1, 2, 2) if use_stride else (1, 1, 1)

    # cweight: [k, k, k, c_in, c_out] -> [c_out, c_in, k, k, k]
    w = cweight.permute(4, 3, 0, 1, 2)
    b = bweight if use_bias else None

    conv_output = F.conv3d(inp, w, bias=b, stride=stride,
                           padding=(w.shape[2] // 2,
                                    w.shape[3] // 2,
                                    w.shape[4] // 2))

    if activation is not None:
        conv_output = activation(conv_output)

    if bn:
        conv_output = batch_norm(conv_output, scale, bias)
    if pn:
        conv_output = pixel_norm(conv_output)
    if gn:
        conv_output = group_norm(conv_output, scale, bias)
    if ln:
        conv_output = layer_norm(conv_output, scale, bias)

    if GLOBAL_DOWNSAMPLE and use_stride:
        conv_output = F.avg_pool3d(conv_output, kernel_size=(1, 2, 2), stride=(1, 2, 2))

    return conv_output


def group_norm(inp, scale, bias, g=32, eps=1e-6, stop_batch=False):
    """Applies group normalization assuming NCHW format."""
    N, C, H, W = inp.shape
    G = min(g, C)
    x = inp.view(N, G, C // G, H, W)
    mean = x.mean(dim=(2, 3, 4), keepdim=True)
    var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
    x_norm = (x - mean) * torch.rsqrt(var + eps)
    output = x_norm.view(N, C, H, W)

    if scale is not None:
        output = output * scale.view(1, -1, 1, 1)
    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)

    return output


def layer_norm(inp, scale, bias, eps=1e-6):
    """Layer normalization over all non-batch dimensions."""
    mean = inp.mean(dim=(1, 2, 3), keepdim=True)
    var = inp.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
    output = (inp - mean) * torch.rsqrt(var + eps)

    if scale is not None:
        output = output * scale.view(1, -1, 1, 1)
    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)

    return output


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis (channel axis)."""
    # x: [N, Cx, H, W], y: [N, Cy, H, W] or [N, Cy, 1, 1]
    if y.dim() == 4 and (y.shape[2] == 1 and y.shape[3] == 1):
        y = y.expand(-1, -1, x.shape[2], x.shape[3])
    elif y.shape[2:] != x.shape[2:]:
        y = F.interpolate(y, size=x.shape[2:], mode='nearest')
    # match TF behavior of scaling y by 1/10.
    y = y / 10.0
    return torch.cat([x, y], dim=1)


def hw_flatten(x: torch.Tensor) -> torch.Tensor:
    """Flatten spatial dimensions (N, C, H, W) -> (N, H*W, C)."""
    N, C, H, W = x.shape
    return x.view(N, C, H * W).permute(0, 2, 1)


def attention(
        inp,
        q,
        q_b,
        k,
        k_b,
        v,
        v_b,
        gamma,
        reuse,
        scope,
        stop_at_grad=False,
        seperate=False,
        scale=False,
        train=False,
        dropout=0.0):
    # 1x1 convs for q, k, v
    conv_q = conv_block(
        inp,
        q,
        q_b,
        reuse=reuse,
        scope=scope,
        use_stride=False,
        activation=None,
        use_bias=True,
        pn=False,
        bn=False,
        gn=False)
    conv_k = conv_block(
        inp,
        k,
        k_b,
        reuse=reuse,
        scope=scope,
        use_stride=False,
        activation=None,
        use_bias=True,
        pn=False,
        bn=False,
        gn=False)
    conv_v = conv_block(
        inp,
        v,
        v_b,
        reuse=reuse,
        scope=scope,
        use_stride=False,
        pn=False,
        bn=False,
        gn=False)

    c_num = float(conv_q.shape[1])
    s = torch.matmul(hw_flatten(conv_q), hw_flatten(conv_k).transpose(1, 2))

    if scale:
        s = s / (c_num ** 0.5)

    if train and dropout > 0.0:
        s = F.dropout(s, p=dropout, training=True)

    beta = F.softmax(s, dim=-1)
    o = torch.matmul(beta, hw_flatten(conv_v))
    N, HW, C = o.shape
    o = o.permute(0, 2, 1).view(N, C, inp.shape[2], inp.shape[3])

    out = inp + gamma.view(1, 1, 1, 1) * o

    if not seperate:
        return out
    else:
        return gamma.view(1, 1, 1, 1) * o


def attention_2d(
        inp,
        q,
        q_b,
        k,
        k_b,
        v,
        v_b,
        reuse,
        scope,
        stop_at_grad=False,
        seperate=False,
        scale=False):
    """
    This is a direct but simplified PyTorch port of the original TF code.
    It assumes `inp` has shape [B, T, O, C] and flattens the middle dims.
    """
    B, T, O, C = inp.shape
    inp_compact = inp.reshape(B * INPUT_OBJECTS * T, C)

    f_q = inp_compact @ q + q_b
    f_k = inp_compact @ k + k_b
    f_v = F.leaky_relu(inp_compact @ v + v_b, negative_slope=0.1)

    f_q = f_q.view(B, T, O, -1)
    f_k = f_k.view(B, T, O, -1)
    f_v = f_v.view(B, T, O, C)

    s = torch.matmul(f_k, f_q.transpose(2, 3))
    c_num = (32 ** 0.5)

    if scale:
        s = s / c_num

    beta = F.softmax(s, dim=-1)

    o = torch.matmul(beta, f_v)
    o = o.view_as(inp) + inp

    return o


def batch_norm(inp, scale, bias, eps=0.01):
    """
    Simple batch normalization over batch dimension only, similar to the TF code.
    Expects NCHW or NCDHW; normalization is done per-channel.
    """
    if inp.dim() == 4:
        dims = (0, 2, 3)
    elif inp.dim() == 5:
        dims = (0, 2, 3, 4)
    else:
        dims = (0,)

    mean = inp.mean(dim=dims, keepdim=True)
    var = inp.var(dim=dims, keepdim=True, unbiased=False)
    x_hat = (inp - mean) / torch.sqrt(var + eps)

    if scale is not None:
        if scale.dim() == 1:
            scale_reshaped = scale.view(1, -1, *([1] * (inp.dim() - 2)))
        else:
            scale_reshaped = scale
        x_hat = x_hat * scale_reshaped
    if bias is not None:
        if bias.dim() == 1:
            bias_reshaped = bias.view(1, -1, *([1] * (inp.dim() - 2)))
        else:
            bias_reshaped = bias
        x_hat = x_hat + bias_reshaped

    return x_hat


def normalize(inp, activation, reuse, scope):
    if NORM == 'batch_norm':
        out = batch_norm(inp, scale=None, bias=None)
        if activation is not None:
            out = activation(out)
        return out
    elif NORM == 'layer_norm':
        out = layer_norm(inp, scale=None, bias=None)
        if activation is not None:
            out = activation(out)
        return out
    elif NORM == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp
    else:
        # Fallback: just apply activation
        return activation(inp) if activation is not None else inp


# ---------------------------------------------------------------------------
# Loss functions and spectral normalization
# ---------------------------------------------------------------------------

def mse(pred, label):
    pred = pred.view(-1)
    label = label.view(-1)
    return torch.mean((pred - label) ** 2)


NO_OPS = 'NO_OPS'


def _l2normalize(v, eps=1e-12):
    return v / (v.pow(2).sum() ** 0.5 + eps)


def spectral_normed_weight(w, name, lower_bound=False, iteration=1, fc=False):
    """
    PyTorch implementation of the spectral normalization used in the TF code.
    This version does power iteration on the flattened weight matrix.
    """
    if fc:
        iteration = 2

    w_shape = w.shape
    w_mat = w.view(-1, w_shape[-1])  # [N, out_dim]

    u = torch.randn(1, w_mat.shape[1], device=w.device)
    u = F.normalize(u, dim=1, eps=1e-12)

    iters = SPEC_ITER if SPEC_ITER is not None else iteration
    for _ in range(iters):
        v = F.normalize(torch.matmul(u, w_mat.t()), dim=1, eps=1e-12)
        u = F.normalize(torch.matmul(v, w_mat), dim=1, eps=1e-12)

    sigma = torch.matmul(torch.matmul(v, w_mat), u.t())  # [1,1]
    sigma = sigma.squeeze()

    sigma_new = SPEC_NORM_VAL

    if lower_bound:
        sigma = sigma + 1e-6
        w_norm = w_mat / sigma * torch.min(sigma, torch.tensor(1.0, device=w.device)) * sigma_new
    else:
        w_norm = w_mat / sigma * sigma_new

    w_norm = w_norm.view(w_shape)
    return w_norm


def average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all towers.
    `tower_grads` is a list of lists of (grad, variable) tuples.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, v in grad_and_vars:
            if g is not None:
                grads.append(g.unsqueeze(0))
            else:
                print(g, v)
        if not grads:
            continue
        grad = torch.cat(grads, dim=0).mean(dim=0)
        v = grad_and_vars[0][1]
        average_grads.append((grad, v))
    return average_grads


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CubesNet(object):
    """Construct the convolutional network specified in MAML (PyTorch port)."""
    def __init__(self, num_filters=64, num_channels=3, label_size=6):

        self.channels = num_channels
        self.dim_hidden = num_filters
        self.img_size = 64
        self.label_size = label_size
        print("label_size ", self.label_size)

    def construct_weights(self, scope=''):
        weights = {}

        k = 5

        if not CCLASS:
            classes = 1
        else:
            classes = self.label_size

        init_conv_weight(weights, 'c1_pre', 3, self.channels, self.dim_hidden)
        init_res_weight(weights, 'res_optim', 3, self.dim_hidden, self.dim_hidden, classes=classes)
        init_res_weight(weights, 'res_1', 3, self.dim_hidden, 2 * self.dim_hidden, classes=classes)
        init_res_weight(weights, 'res_2', 3, 2 * self.dim_hidden, 2 * self.dim_hidden, classes=classes)
        init_res_weight(weights, 'res_3', 3, 2 * self.dim_hidden, 2 * self.dim_hidden, classes=classes)
        init_res_weight(weights, 'res_4', 3, 2 * self.dim_hidden, 4 * self.dim_hidden, classes=classes)
        init_fc_weight(weights, 'fc_dense', 4 * self.dim_hidden, 2 * self.dim_hidden)
        init_fc_weight(weights, 'fc5', 4 * self.dim_hidden, 1, spec_norm=False)

        return weights

    def forward(self, inp, weights, attention_mask, reuse=False, scope='',
                stop_grad=False, label=None, stop_at_grad=False, stop_batch=False):

        # Expect inp as [B, C, H, W]
        batch_size = inp.shape[0]
        channels = self.channels

        if COMB_MASK and attention_mask is not None:
            # attention_mask: [B, COND_FUNC, H, W]
            attention_mask = F.softmax(attention_mask, dim=1)
            inp_exp = inp.unsqueeze(1)  # [B,1,C,H,W]
            mask_exp = attention_mask.unsqueeze(2)  # [B,COND_FUNC,1,H,W]
            inp = (inp_exp * mask_exp).view(batch_size * COND_FUNC, channels, self.img_size, self.img_size)

        local_weights = {}
        for k, v in weights.items():
            if isinstance(v, dict):
                sub = {}
                for kk, vv in v.items():
                    sub[kk] = vv.detach() if stop_grad else vv
                local_weights[k] = sub
            else:
                local_weights[k] = v.detach() if stop_grad else v

        if not CCLASS:
            label = None

        if SWISH_ACT:
            act = swish
        else:
            act = lambda x: F.leaky_relu(x, negative_slope=0.1)

        x = smart_conv_block(inp, local_weights, reuse, 'c1_pre', use_stride=False, activation=act)

        hidden1 = smart_res_block(x, local_weights, reuse, 'res_optim', adaptive=False, label=label, act=act)
        hidden2 = smart_res_block(hidden1, local_weights, reuse, 'res_1', stop_batch=stop_batch, adaptive=True, label=label, act=act)
        hidden3 = smart_res_block(hidden2, local_weights, reuse, 'res_2', stop_batch=stop_batch, adaptive=False, label=label, act=act)
        hidden4 = smart_res_block(hidden3, local_weights, reuse, 'res_3', adaptive=False, downsample=False, stop_batch=stop_batch, label=label, act=act)
        hidden5 = smart_res_block(hidden4, local_weights, reuse, 'res_4', stop_batch=stop_batch, adaptive=True, label=label, act=act)

        hidden5 = act(hidden5)
        hidden6 = hidden5.mean(dim=(2, 3))
        energy = smart_fc_block(hidden6, local_weights, reuse, 'fc5')

        if COMB_MASK and attention_mask is not None:
            energy = energy.view(batch_size, COND_FUNC)
            energy = energy.sum(dim=1, keepdim=True)

        return energy


class CubesNetGen(object):
    """Generator network (PyTorch port)."""
    def __init__(self, num_filters=64, num_channels=3, label_size=6):
        self.channels = num_channels
        self.dim_hidden = num_filters
        self.img_size = 64
        self.label_size = label_size
        print("label_size ", self.label_size)

    def construct_weights(self, scope=''):
        weights = {}

        if not CCLASS:
            classes = 1
        else:
            classes = self.label_size

        init_fc_weight(weights, 'fc_dense', 2 * self.dim_hidden, 4 * 4 * 4 * self.dim_hidden)
        init_res_weight(weights, 'res_1', 3, 4 * self.dim_hidden, 2 * self.dim_hidden, classes=classes)
        init_res_weight(weights, 'res_2', 3, 2 * self.dim_hidden, 2 * self.dim_hidden, classes=classes)
        init_res_weight(weights, 'res_3', 3, 2 * self.dim_hidden, self.dim_hidden, classes=classes)
        init_res_weight(weights, 'res_4', 3, self.dim_hidden, self.dim_hidden, classes=classes)
        init_conv_weight(weights, 'c4_out', 3, self.dim_hidden, self.channels)

        return weights

    def forward(self, inp, weights, reuse=False, scope='',
                stop_grad=False, label=None, stop_at_grad=False, stop_batch=False):

        local_weights = {}
        for k, v in weights.items():
            if isinstance(v, dict):
                sub = {}
                for kk, vv in v.items():
                    sub[kk] = vv.detach() if stop_grad else vv
                local_weights[k] = sub
            else:
                local_weights[k] = v.detach() if stop_grad else v

        if SWISH_ACT:
            act = swish
        else:
            act = lambda x: F.leaky_relu(x, negative_slope=0.1)

        hidden1 = act(smart_fc_block(inp, local_weights, reuse, 'fc_dense'))
        B = inp.shape[0]
        hidden1 = hidden1.view(B, 4 * self.dim_hidden, 4, 4)

        hidden2 = smart_res_block(hidden1, local_weights, reuse, 'res_1', label=label, act=act, upsample=True)
        hidden3 = smart_res_block(hidden2, local_weights, reuse, 'res_2', adaptive=False, label=label, act=act, upsample=True)
        hidden4 = smart_res_block(hidden3, local_weights, reuse, 'res_3', label=label, act=act, upsample=True)
        hidden5 = smart_res_block(hidden4, local_weights, reuse, 'res_4', label=label, adaptive=False, act=act, upsample=True)
        output = smart_conv_block(hidden5, local_weights, reuse, 'c4_out', use_stride=False, activation=None)

        return output


class ResNet128(object):
    """Construct the convolutional network specified in MAML (PyTorch port)."""

    def __init__(self, num_channels=3, num_filters=64, train=False, classes=1000):

        self.channels = num_channels
        self.dim_hidden = num_filters
        self.dropout = train
        self.train = train
        self.classes = classes

        print("set classes to be", classes)

    def construct_weights(self, scope=''):
        weights = {}

        if not CCLASS:
            classes = 1
        else:
            classes = self.classes

        print("constructing weights with class number ", classes)

        init_conv_weight(weights, 'c1_pre', 3, self.channels, 64)
        init_res_weight(weights, 'res_optim', 3, 64, self.dim_hidden, classes=classes)
        init_res_weight(weights, 'res_3', 3, self.dim_hidden, 2 * self.dim_hidden, classes=classes)
        init_res_weight(weights, 'res_5', 3, 2 * self.dim_hidden, 4 * self.dim_hidden, classes=classes)
        init_res_weight(weights, 'res_7', 3, 4 * self.dim_hidden, 8 * self.dim_hidden, classes=classes)
        init_res_weight(weights, 'res_9', 3, 8 * self.dim_hidden, 8 * self.dim_hidden, classes=classes)
        init_res_weight(weights, 'res_10', 3, 8 * self.dim_hidden, 8 * self.dim_hidden, classes=classes)
        init_fc_weight(weights, 'fc5', 8 * self.dim_hidden, 1, spec_norm=False)
        init_attention_weight(weights, 'atten', self.dim_hidden, int(self.dim_hidden / 2), trainable_gamma=True)

        return weights

    def forward(self, inp, weights, reuse=False, scope='',
                stop_grad=False, label=None, stop_at_grad=False, stop_batch=False, latent=None):

        local_weights = {}
        for k, v in weights.items():
            if isinstance(v, dict):
                sub = {}
                for kk, vv in v.items():
                    sub[kk] = vv.detach() if stop_grad else vv
                local_weights[k] = sub
            else:
                local_weights[k] = v.detach() if stop_grad else v

        batch = inp.shape[0]

        if not CCLASS:
            label = None

        if SWISH_ACT:
            act = swish
        else:
            act = lambda x: F.leaky_relu(x, negative_slope=0.1)

        dropout = self.dropout
        train = self.train

        x = smart_conv_block(inp, local_weights, reuse, 'c1_pre', use_stride=False, activation=act)
        hidden1 = smart_res_block(x, local_weights, reuse, 'res_optim', label=label,
                                  dropout=dropout, train=train, downsample=True, adaptive=False)

        if USE_ATTENTION:
            hidden1 = smart_atten_block(hidden1, local_weights, reuse, 'atten', stop_at_grad=stop_at_grad)

        hidden2 = smart_res_block(hidden1, local_weights, reuse, 'res_3', stop_batch=stop_batch,
                                  downsample=True, adaptive=True, label=label,
                                  dropout=dropout, train=train, act=act)
        hidden3 = smart_res_block(hidden2, local_weights, reuse, 'res_5', stop_batch=stop_batch,
                                  downsample=True, adaptive=True, label=label,
                                  dropout=dropout, train=train, act=act)
        hidden4 = smart_res_block(hidden3, local_weights, reuse, 'res_7', stop_batch=stop_batch,
                                  label=label, dropout=dropout, train=train, act=act,
                                  downsample=True, adaptive=True)
        hidden5 = smart_res_block(hidden4, local_weights, reuse, 'res_9', stop_batch=stop_batch,
                                  label=label, dropout=dropout, train=train, act=act,
                                  downsample=True, adaptive=False)
        hidden6 = smart_res_block(hidden5, local_weights, reuse, 'res_10', stop_batch=stop_batch,
                                  label=label, dropout=dropout, train=train, act=act,
                                  downsample=False, adaptive=False)

        if SWISH_ACT:
            hidden6 = act(hidden6)
        else:
            hidden6 = F.relu(hidden6)

        hidden5 = hidden6.sum(dim=(2, 3))
        hidden6 = smart_fc_block(hidden5, local_weights, reuse, 'fc5')
        energy = hidden6

        return energy


class CubesPredict(object):
    def __init__(self, num_channels=3, num_filters=64):

        self.channels = num_channels
        self.dim_hidden = num_filters
        self.datasource = DATASOURCE

    def construct_weights(self, scope=''):
        weights = {}

        classes = 1

        init_conv_weight(weights, 'c1_pre', 1, self.channels, 64, spec_norm=False)
        init_conv_weight(weights, 'c1', 4, 64, self.dim_hidden, classes=classes, spec_norm=False)
        init_conv_weight(weights, 'c2', 4, self.dim_hidden, 2 * self.dim_hidden, classes=classes, spec_norm=False)
        init_conv_weight(weights, 'c3', 4, 2 * self.dim_hidden, 4 * self.dim_hidden, classes=classes, spec_norm=False)
        init_conv_weight(weights, 'c4', 4, 4 * self.dim_hidden, 4 * self.dim_hidden, classes=classes, spec_norm=False)
        init_fc_weight(weights, 'fc_dense_pos', 4 * self.dim_hidden, 2 * self.dim_hidden, spec_norm=False)
        init_fc_weight(weights, 'fc_dense_logit', 4 * self.dim_hidden, 2 * self.dim_hidden, spec_norm=False)
        init_fc_weight(weights, 'fc5_pos', 2 * self.dim_hidden, 2, spec_norm=False)
        init_fc_weight(weights, 'fc5_logit', 2 * self.dim_hidden, 1, spec_norm=False)

        return weights

    def forward(self, inp, weights, reuse=False, scope='',
                stop_grad=False, label=None, **kwargs):
        weights_local = {}
        for k, v in weights.items():
            if isinstance(v, dict):
                sub = {}
                for kk, vv in v.items():
                    sub[kk] = vv.detach() if stop_grad else vv
                weights_local[k] = sub
            else:
                weights_local[k] = v.detach() if stop_grad else v

        # Expect inp as [B, C, H, W] or [B, 64*64*C]
        if inp.dim() == 2:
            B = inp.shape[0]
            inp = inp.view(B, self.channels, 64, 64)

        if SWISH_ACT:
            act = swish
        else:
            act = lambda x: F.leaky_relu(x, negative_slope=0.1)

        h1 = smart_conv_block(inp, weights_local, reuse, 'c1_pre', use_stride=False, activation=act)
        h2 = smart_conv_block(h1, weights_local, reuse, 'c1', use_stride=True, downsample=True,
                              label=label, extra_bias=False, activation=act)
        h3 = smart_conv_block(h2, weights_local, reuse, 'c2', use_stride=True, downsample=True,
                              label=label, extra_bias=False, activation=act)
        h4 = smart_conv_block(h3, weights_local, reuse, 'c3', use_stride=True, downsample=True,
                              label=label, use_scale=False, extra_bias=False, activation=act)
        h5 = smart_conv_block(h4, weights_local, reuse, 'c4', use_stride=True, downsample=True,
                              label=label, use_scale=False, extra_bias=False, activation=act)

        # Global average pooling
        h5_mean = h5.mean(dim=(2, 3))
        h6_pos = act(smart_fc_block(h5_mean, weights_local, reuse, 'fc_dense_pos'))
        h6_logit = act(smart_fc_block(h5_mean, weights_local, reuse, 'fc_dense_logit'))
        pos = smart_fc_block(h6_pos, weights_local, reuse, 'fc5_pos')
        logit = smart_fc_block(h6_logit, weights_local, reuse, 'fc5_logit')

        return logit, pos
