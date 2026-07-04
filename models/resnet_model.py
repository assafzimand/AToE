"""Residual neural network with the same interface as FCNet."""

import torch
import torch.nn as nn
from functools import partial
from typing import List, Dict, Optional
from models.rwf_layer import RWFLinear
from models.fourier_features import FourierFeatureEmbedding, PeriodicSpatialFourierEmbedding


class ResBlock(nn.Module):
    """Pre-activation residual block: x + act(Linear(act(Linear(x))))."""

    def __init__(self, dim: int, activation: nn.Module, LinearCls=None):
        super().__init__()
        if LinearCls is None:
            LinearCls = nn.Linear
        self.fc1 = LinearCls(dim, dim)
        self.fc2 = LinearCls(dim, dim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        return self.activation(residual + out)


class ResNetModel(nn.Module):
    """Residual neural network for PINN problems.

    Drop-in replacement for FCNet with identical constructor signature
    and public API (forward, get_activation_dim, get_layer_names).

    Architecture list ``[in, h, h, ..., h, out]`` is interpreted as:

    1. **Input projection** — ``Linear(in, h) + activation``
    2. **Interior pairs** — each pair of hidden layers becomes a
       ``ResBlock(h)``.  If the number of hidden layers (after the
       first) is odd, the leftover is a plain ``Linear(h, h) + act``.
    3. **Output projection** — ``Linear(h, out)`` (no activation).

    All hidden widths must be identical (validated in ``__init__``).
    """

    def __init__(self, layers: List[int], activation: str, config: Dict,
                 is_base: bool = True):
        super().__init__()

        self.is_base = is_base

        problem = config['problem']
        problem_config = config[problem]
        spatial_dim = problem_config['spatial_dim']
        output_dim = problem_config['output_dim']

        if is_base:
            expected_input_dim = spatial_dim + 1
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
        if len(hidden) < 2:
            raise ValueError(
                "ResNetModel requires at least 2 hidden "
                f"layers, got {len(hidden)}. "
                f"Architecture: {layers}"
            )
        if len(set(hidden)) != 1:
            raise ValueError(
                "ResNetModel requires all hidden widths "
                f"to be identical, got {hidden}. "
                f"Architecture: {layers}"
            )

        self.layers = layers
        self.activation_name = activation
        self.config = config

        self.activation = self._get_activation(activation)

        # RWF — config['rwf'] is a dict {enabled, mean, std}
        _rwf = config['rwf']
        use_rwf = _rwf['enabled']
        rwf_mean = _rwf.get('mean', 1.0)
        rwf_std = _rwf.get('std', 0.1)

        # Fourier Features: embed input before input_proj.
        # Disabled for non-base experts whose input is a parent activation.
        ff_cfg = config['fourier_features']
        use_ff = ff_cfg['enabled'] and is_base
        use_periodic = ff_cfg['periodic']
        self.ff_emb: Optional[FourierFeatureEmbedding] = None
        effective_input_dim = layers[0]
        if use_ff:
            ff_dim = ff_cfg['dim']
            ff_scale = ff_cfg['scale']
            if use_periodic:
                _lo, _hi = problem_config['spatial_domain'][0]
                L = _hi - _lo
                self.ff_emb = PeriodicSpatialFourierEmbedding(spatial_dim, ff_dim, ff_scale, L)
            else:
                self.ff_emb = FourierFeatureEmbedding(layers[0], ff_dim, ff_scale)
            effective_input_dim = self.ff_emb.output_dim

        h = hidden[0]
        LinearCls = partial(RWFLinear, mean=rwf_mean, std=rwf_std) if use_rwf else nn.Linear
        self.input_proj = LinearCls(effective_input_dim, h)

        n_blocks = len(hidden) // 2
        has_leftover = len(hidden) % 2 == 1

        self.res_blocks = nn.ModuleList(
            [ResBlock(h, self.activation, LinearCls=LinearCls) for _ in range(n_blocks)]
        )
        self.leftover = LinearCls(h, h) if has_leftover else None

        # Output projection always plain nn.Linear for output scale stability
        self.output_proj = nn.Linear(h, layers[-1])

    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU()
        }
        if activation.lower() not in activations:
            raise ValueError(
                f"Unknown activation: {activation}. "
                f"Available: {list(activations.keys())}"
            )
        return activations[activation.lower()]

    def forward(self, x: torch.Tensor, return_activation: bool = False):
        if self.ff_emb is not None:
            x = self.ff_emb(x)
        out = self.activation(self.input_proj(x))

        for block in self.res_blocks:
            out = block(out)

        if self.leftover is not None:
            out = self.activation(self.leftover(out))

        last_hidden = out
        out = self.output_proj(out)

        if return_activation:
            return out, last_hidden
        return out

    def get_activation_dim(self) -> int:
        return self.layers[-2]

    def get_layer_names(self) -> List[str]:
        names = ['input_proj']
        for i in range(len(self.res_blocks)):
            names.append(f'res_block_{i}')
        if self.leftover is not None:
            names.append('leftover')
        names.append('output_proj')
        return names

    def __repr__(self) -> str:
        layers_str = " -> ".join(map(str, self.layers))
        return (
            f"ResNetModel(\n"
            f"  architecture: {layers_str}\n"
            f"  activation: {self.activation_name}\n"
            f"  res_blocks: {len(self.res_blocks)}\n"
            f"  leftover: {self.leftover is not None}\n"
            f")"
        )
