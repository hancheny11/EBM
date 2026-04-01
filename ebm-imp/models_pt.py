
"""
models_pt.py — PyTorch port (operator-faithful) of models.py
-----------------------------------------------------------
Mirrors class ResNet128 with the same call structure:
  - construct_weights(): builds nested dict of parameters (like TF vars)
  - forward(inp, weights, ...): uses utils_pt.smart_* blocks in the same order
"""
from typing import Dict, Any, Optional
import copy
import torch
import torch.nn.functional as F

from utils_pt import FLAGS, swish
from utils_pt import init_conv_weight, init_res_weight, init_attention_weight, init_fc_weight
from utils_pt import smart_conv_block, smart_res_block, smart_res_block_optim, smart_atten_block, smart_fc_block


class ResNet128(object):
    """Construct the convolutional network specified in the original TF code."""
    def __init__(self, num_channels=3, num_filters=64, train=False):
        self.channels = num_channels
        self.dim_hidden = num_filters
        self.dropout = train
        self.train = train

    def construct_weights(self, scope: str = '') -> Dict[str, Dict[str, torch.Tensor]]:
        weights: Dict[str, Dict[str, torch.Tensor]] = {}
        classes = 1000

        init_conv_weight(weights, f'{scope}c1_pre', 3, self.channels, 64)
        init_res_weight(weights, f'{scope}res_optim', 3, 64, self.dim_hidden, classes=classes)
        init_res_weight(weights, f'{scope}res_3', 3, self.dim_hidden, 2*self.dim_hidden, classes=classes)
        init_res_weight(weights, f'{scope}res_5', 3, 2*self.dim_hidden, 4*self.dim_hidden, classes=classes)
        init_res_weight(weights, f'{scope}res_7', 3, 4*self.dim_hidden, 8*self.dim_hidden, classes=classes)
        init_res_weight(weights, f'{scope}res_9', 3, 8*self.dim_hidden, 8*self.dim_hidden, classes=classes)
        init_res_weight(weights, f'{scope}res_10', 3, 8*self.dim_hidden, 8*self.dim_hidden, classes=classes)
        init_fc_weight(weights, f'{scope}fc5', 8*self.dim_hidden, 1, spec_norm=False)

        init_attention_weight(weights, f'{scope}atten', self.dim_hidden, self.dim_hidden / 2., trainable_gamma=True)

        return weights

    def forward(self, inp: torch.Tensor, weights: Dict[str, Dict[str, torch.Tensor]], reuse: bool = False, scope: str = '',
                stop_grad: bool = False, label: Optional[torch.Tensor] = None, stop_at_grad: bool = False, stop_batch: bool = False):
        # weights = copy.deepcopy(weights)
        weights = {k: {kk: vv.clone() for kk, vv in v.items()} for k, v in weights.items()}
        
        batch = inp.shape[0]

        if not getattr(FLAGS, 'cclass', False):
            label = None

        if stop_grad:
            for k, v in list(weights.items()):
                if isinstance(v, dict):
                    v_copy = v.copy()
                    for k_sub, v_sub in list(v_copy.items()):
                        v_copy[k_sub] = v_sub.detach()
                    weights[k] = v_copy
                else:
                    weights[k] = v.detach()

        act = swish if getattr(FLAGS, 'swish_act', False) else F.leaky_relu
        dropout = self.dropout
        train = self.train

        inp = smart_conv_block(inp, weights, reuse, f'{scope}c1_pre', use_stride=False, activation=act)
        hidden1 = smart_res_block(inp, weights, reuse, f'{scope}res_optim', label=label, dropout=dropout, train=train, downsample=True, adaptive=False)

        if getattr(FLAGS, 'use_attention', False):
            hidden1 = smart_atten_block(hidden1, weights, reuse, f'{scope}atten', stop_at_grad=stop_at_grad)

        hidden2 = smart_res_block(hidden1, weights, reuse, f'{scope}res_3', stop_batch=stop_batch, downsample=True, adaptive=True, label=label, dropout=dropout, train=train, act=act)
        hidden3 = smart_res_block(hidden2, weights, reuse, f'{scope}res_5', stop_batch=stop_batch, downsample=True, adaptive=True, label=label, dropout=dropout, train=train, act=act)
        hidden4 = smart_res_block(hidden3, weights, reuse, f'{scope}res_7', stop_batch=stop_batch, label=label, dropout=dropout, train=train, act=act, downsample=True, adaptive=True)
        hidden5 = smart_res_block(hidden4, weights, reuse, f'{scope}res_9', stop_batch=stop_batch, label=label, dropout=dropout, train=train, act=act, downsample=True, adaptive=False)
        hidden6 = smart_res_block(hidden5, weights, reuse, f'{scope}res_10', stop_batch=stop_batch, label=label, dropout=dropout, train=train, act=act, downsample=False, adaptive=False)

        if getattr(FLAGS, 'swish_act', False):
            hidden6 = act(hidden6)
        else:
            hidden6 = F.relu(hidden6)

        hidden5 = hidden6.sum(dim=(1, 2))
        hidden6 = smart_fc_block(hidden5, weights, reuse, f'{scope}fc5')
        energy = hidden6
        return energy
