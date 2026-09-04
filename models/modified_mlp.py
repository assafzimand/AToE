"""Modified MLP (Wang et al.) with named layers.

The gated architecture from "Understanding and mitigating gradient flow
pathologies in physics-informed neural networks" (Wang, Teng & Perdikaris,
2021), as used for KS in the causal-training paper and the Expert's Guide:
two input encoders U and V, and every hidden layer's activation Z gates a
blend of them:

    U = act(W_u x̃ + b_u),  V = act(W_v x̃ + b_v)
    Z_k = act(W_k H_{k-1} + b_k)
    H_k = (1 - Z_k) ⊙ U + Z_k ⊙ V
    out = W_out H_n + b_out

Framework parity with FCNet:
  - hidden layers named ``network.layer_{i}`` (checkpoint/arch-inference
    compatible); encoders live at ``encoder_u`` / ``encoder_v``
  - Fourier-feature input embedding (``config['fourier_features']``,
    dim 'auto' sized to the first hidden width, periodic variant included),
    disabled for non-base experts exactly like FCNet
  - RWF hidden/encoder layers (``config['rwf']``); output layer stays a
    plain nn.Linear for output-scale stability
  - forward(return_activation=True), get_activation_dim, get_layer_names

Constraint: all hidden widths must be equal (the gate blends U/V of one
width across every layer) — [2, 128, 128, ..., 1] style architectures.
"""

import torch
import torch.nn as nn
from functools import partial
from typing import Dict, List, Optional

from models.rwf_layer import RWFLinear
from models.fourier_features import (FourierFeatureEmbedding,
                                     PeriodicSpatialFourierEmbedding)


class ModifiedMLP(nn.Module):
    def __init__(self, layers: List[int], activation: str, config: Dict,
                 is_base: bool = True):
        super().__init__()

        self.is_base = is_base

        problem = config['problem']
        problem_config = config[problem]
        spatial_dim = problem_config['spatial_dim']
        output_dim = problem_config['output_dim']

        if is_base:
            expected_input_dim = spatial_dim + 1  # x + t
            assert layers[0] == expected_input_dim, (
                f"Architecture input dimension {layers[0]} does not match "
                f"expected dimension {expected_input_dim} "
                f"(spatial_dim={spatial_dim} + 1 for time)"
            )
        assert layers[-1] == output_dim, (
            f"Architecture output dimension {layers[-1]} does not match "
            f"expected dimension {output_dim} (problem={problem})"
        )

        hidden = layers[1:-1]
        assert len(hidden) >= 1, "ModifiedMLP needs at least 1 hidden layer"
        assert len(set(hidden)) == 1, (
            f"ModifiedMLP requires equal hidden widths (the U/V gate blends "
            f"one width across all layers); got {hidden}"
        )
        width = hidden[0]

        self.layers = layers
        self.activation_name = activation
        self.config = config
        self.activation = self._get_activation(activation)

        # Fourier features: identical policy to FCNet (base models only).
        ff_cfg = config['fourier_features']
        use_ff = ff_cfg['enabled'] and is_base
        use_periodic = ff_cfg['periodic']
        self.ff_emb: Optional[nn.Module] = None
        effective_input_dim = layers[0]
        if use_ff:
            ff_dim = ff_cfg['dim']
            if ff_dim in ('auto', None):
                ff_dim = max(1, layers[1] // 2)
            ff_scale = ff_cfg['scale']
            if use_periodic:
                _lo, _hi = problem_config['spatial_domain'][0]
                L = _hi - _lo
                self.ff_emb = PeriodicSpatialFourierEmbedding(
                    spatial_dim, ff_dim, ff_scale, L)
            else:
                self.ff_emb = FourierFeatureEmbedding(layers[0], ff_dim, ff_scale)
            effective_input_dim = self.ff_emb.output_dim

        # RWF: hidden + encoder layers factorized, output layer plain.
        _rwf = config['rwf']
        use_rwf = _rwf['enabled']
        LinearCls_hidden = (partial(RWFLinear, mean=_rwf.get('mean', 1.0),
                                    std=_rwf.get('std', 0.1))
                            if use_rwf else nn.Linear)

        # The two gate encoders (embedded input -> width).
        self.encoder_u = LinearCls_hidden(effective_input_dim, width)
        self.encoder_v = LinearCls_hidden(effective_input_dim, width)

        # Hidden gate layers + output, named like FCNet for checkpoint and
        # arch-inference compatibility.
        n_layers = len(layers) - 1
        self.network = nn.ModuleDict()
        for i in range(n_layers):
            layer_name = f"layer_{i + 1}"
            is_output_layer = (i == n_layers - 1)
            in_dim = effective_input_dim if i == 0 else layers[i]
            out_dim = layers[i + 1]
            LinearCls = nn.Linear if is_output_layer else LinearCls_hidden
            self.network[layer_name] = LinearCls(in_dim, out_dim)

    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(),
        }
        if activation.lower() not in activations:
            raise ValueError(
                f"Unknown activation: {activation}. "
                f"Available: {list(activations.keys())}")
        return activations[activation.lower()]

    def forward(self, x: torch.Tensor, return_activation: bool = False):
        out = x
        if self.ff_emb is not None:
            out = self.ff_emb(out)

        u = self.activation(self.encoder_u(out))
        v = self.activation(self.encoder_v(out))

        layer_names = list(self.network.keys())
        for layer_name in layer_names[:-1]:
            z = self.activation(self.network[layer_name](out))
            out = (1.0 - z) * u + z * v

        last_hidden = out
        out = self.network[layer_names[-1]](out)

        if return_activation:
            return out, last_hidden
        return out

    def get_activation_dim(self) -> int:
        return self.layers[-2]

    def get_layer_names(self) -> List[str]:
        return list(self.network.keys())

    def __repr__(self) -> str:
        layers_str = " -> ".join(map(str, self.layers))
        return (
            f"ModifiedMLP(\n"
            f"  architecture: {layers_str} (+2 gate encoders)\n"
            f"  activation: {self.activation_name}\n"
            f"  layers: {self.get_layer_names()}\n"
            f")"
        )
