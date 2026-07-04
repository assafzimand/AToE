"""PirateNet backbone for PINN expert networks.

Architecture (Wang et al., JMLR 2024, Eqs. 4.1-4.7):
    1. Fourier Feature embedding: FF(z) = [cos(Bz), sin(Bz)]
    2. Gating streams (fixed, computed once per forward):
           U = σ(W_U · FF(z) + b_U)
           V = σ(W_V · FF(z) + b_V)
    3. Input projection: x⁰ = σ(W_in · FF(z) + b_in)
    4. L residual blocks, each with 3 Dense layers and 2 UV-gating ops:
           f  = σ(W₁ · x + b₁)
           z₁ = f ⊙ U + (1-f) ⊙ V
           g  = σ(W₂ · z₁ + b₂)
           z₂ = g ⊙ U + (1-g) ⊙ V
           h  = σ(W₃ · z₂ + b₃)
           x_next = α·h + (1-α)·x       (α initialized at 0 → identity at start)
    5. Output projection: W_out · x^L (no activation)

When the requested number of hidden layers is not divisible by 3, we round
to the nearest block count (adding or removing at most 1 layer) so each
block always has exactly 3 Dense layers.

Optional:
    - RWF: applies W_eff = diag(exp(s)) @ W on hidden layers (enabled via config['rwf'])
    - Least-squares init of output layer (piratenet.ls_init=true, expensive)

Reference: Wang et al. (2024) "PirateNets: Physics-informed Deep Learning with
Residual Adaptive Networks." JMLR.

Constructor signature matches FCNet/ResNetModel for drop-in factory use.
Architecture: layers = [input_dim, h, h, ..., h, output_dim]
  - hidden_dim = layers[1] (uniform width required)
  - n_hidden = len(layers) - 2, rounded to nearest multiple of 3 (min 3)
  - n_blocks = n_hidden // 3

Fourier features are read from config['fourier_features'] (same key as FCNet/ResNet).
RWF is read from config['rwf'] (same key).
ls_init is read from config.get('piratenet', {}).get('ls_init', False).
"""

import torch
import torch.nn as nn
from functools import partial
from typing import List, Dict
from models.rwf_layer import RWFLinear
from models.fourier_features import FourierFeatureEmbedding, PeriodicSpatialFourierEmbedding


def _get_activation(name: str) -> nn.Module:
    activations = {
        'tanh': nn.Tanh(),
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'gelu': nn.GELU(),
        'elu': nn.ELU(),
        'leaky_relu': nn.LeakyReLU(),
    }
    name = name.lower()
    if name not in activations:
        raise ValueError(
            f"Unknown activation: {name}. Available: {list(activations.keys())}")
    return activations[name]


class PirateBlock(nn.Module):
    """Single PirateNet residual block (Eqs. 4.1-4.6).

    Contains 3 Dense layers, 2 UV-gating operations, and an adaptive
    residual skip with trainable α (initialized at 0 → identity mapping).
    """

    def __init__(self, h: int, activation: nn.Module, LinearCls):
        super().__init__()
        self.W1 = LinearCls(h, h)
        self.W2 = LinearCls(h, h)
        self.W3 = LinearCls(h, h)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.activation = activation

    def forward(self, x: torch.Tensor, U: torch.Tensor, V: torch.Tensor):
        f = self.activation(self.W1(x))
        z1 = f * U + (1 - f) * V
        g = self.activation(self.W2(z1))
        z2 = g * U + (1 - g) * V
        h = self.activation(self.W3(z2))
        return self.alpha * h + (1 - self.alpha) * x


class PirateNet(nn.Module):
    """PirateNet: Physics-informed Residual Adaptive Network.

    Args:
        layers: [input_dim, h, h, ..., h, output_dim]. All hidden dims must
                be equal. At least 1 hidden layer required. The number of
                hidden layers is rounded to the nearest multiple of 3 (min 3).
        activation: Activation function name.
        config: Full project config dict (reads 'fourier_features', 'rwf',
                'piratenet' sub-keys).
        is_base: If True, validate input_dim == spatial_dim + 1.
    """

    def __init__(self, layers: List[int], activation: str, config: Dict,
                 is_base: bool = True):
        super().__init__()

        self.is_base = is_base
        self.activation_name = activation
        self.config = config

        problem = config['problem']
        problem_config = config[problem]
        spatial_dim = problem_config['spatial_dim']
        output_dim = problem_config['output_dim']

        if is_base:
            expected_input_dim = spatial_dim + 1
            assert layers[0] == expected_input_dim, (
                f"PirateNet input_dim {layers[0]} != expected {expected_input_dim}"
            )
        assert layers[-1] == output_dim, (
            f"PirateNet output_dim {layers[-1]} != expected {output_dim}"
        )

        hidden = layers[1:-1]
        if len(hidden) < 1:
            raise ValueError(
                f"PirateNet requires at least 1 hidden layer, got {len(hidden)}.")
        if len(set(hidden)) != 1:
            raise ValueError(
                f"PirateNet requires uniform hidden width, got {hidden}.")

        input_dim = layers[0]
        h = hidden[0]
        n_requested = len(hidden)

        n_blocks = max(1, round(n_requested / 3))
        n_actual = n_blocks * 3

        if n_actual != n_requested:
            print(f"  [PirateNet] Requested {n_requested} hidden layers, "
                  f"adjusted to {n_actual} (= {n_blocks} blocks × 3 layers/block)")

        self.layers = [layers[0]] + [h] * n_actual + [layers[-1]]
        self.hidden_dim = h
        self.n_blocks = n_blocks
        self.n_layers = n_actual

        # Fourier Feature embedding.
        # Disabled for non-base experts whose input is a parent activation.
        ff_cfg = config['fourier_features']
        use_ff = ff_cfg['enabled'] and is_base
        use_periodic = ff_cfg['periodic']
        if use_ff:
            ff_dim = ff_cfg['dim']
            ff_scale = ff_cfg['scale']
            if use_periodic:
                _lo, _hi = problem_config['spatial_domain'][0]
                L = _hi - _lo
                self.ff_emb = PeriodicSpatialFourierEmbedding(spatial_dim, ff_dim, ff_scale, L)
            else:
                self.ff_emb = FourierFeatureEmbedding(input_dim, ff_dim, ff_scale)
            ff_out = self.ff_emb.output_dim
        else:
            self.ff_emb = None
            ff_out = input_dim

        # RWF — config['rwf'] is a dict {enabled, mean, std}
        _rwf = config['rwf']
        use_rwf = _rwf['enabled']
        rwf_mean = _rwf.get('mean', 1.0)
        rwf_std = _rwf.get('std', 0.1)
        self.rwf_mean = rwf_mean
        self.rwf_std = rwf_std
        LinearCls = partial(RWFLinear, mean=rwf_mean, std=rwf_std) if use_rwf else nn.Linear

        self.activation = _get_activation(activation)

        # U, V projections from FF space → hidden space
        self.U_proj = LinearCls(ff_out, h)
        self.V_proj = LinearCls(ff_out, h)

        # Input projection: x⁰ = σ(W_in · FF(z))
        self.input_proj = LinearCls(ff_out, h)

        # Residual blocks (3 Dense layers each)
        self.blocks = nn.ModuleList(
            [PirateBlock(h, self.activation, LinearCls)
             for _ in range(n_blocks)]
        )

        # Output projection — always plain nn.Linear for output scale stability
        self.output_proj = nn.Linear(h, layers[-1])

    def forward(self, x: torch.Tensor, return_activation: bool = False):
        """Forward pass.

        Args:
            x: Input tensor (N, input_dim).
            return_activation: If True, also returns the last hidden state.

        Returns:
            Output tensor (N, output_dim), or (output, hidden) if return_activation.
        """
        if self.ff_emb is not None:
            z = self.ff_emb(x)
        else:
            z = x

        U = self.activation(self.U_proj(z))
        V = self.activation(self.V_proj(z))

        h = self.activation(self.input_proj(z))

        for block in self.blocks:
            h = block(h, U, V)

        if return_activation:
            return self.output_proj(h), h
        return self.output_proj(h)

    def get_activation_dim(self) -> int:
        """Hidden dim (used if return_activation=True is ever needed)."""
        return self.hidden_dim

    def get_layer_names(self) -> List[str]:
        """PirateNet has no named-layer dict — returns empty list."""
        return []

    def debug_state(self):
        """Return per-block alpha values and W1/W2/W3 weight L2-norms for diagnostics."""
        alphas, wnorms = [], []
        for blk in self.blocks:
            alphas.append(blk.alpha.item())
            wnorms.append([
                blk.W1.weight.norm().item(),
                blk.W2.weight.norm().item(),
                blk.W3.weight.norm().item(),
            ])
        return {'alphas': alphas, 'block_w_norms': wnorms}

    def __repr__(self) -> str:
        ff_info = f"ff_out={self.ff_emb.output_dim}" if self.ff_emb else "no_ff"
        return (
            f"PirateNet(\n"
            f"  architecture: {self.layers}\n"
            f"  activation: {self.activation_name}\n"
            f"  hidden_dim: {self.hidden_dim}, n_blocks: {self.n_blocks}, "
            f"n_layers: {self.n_layers}\n"
            f"  {ff_info}, rwf={isinstance(self.input_proj, RWFLinear)}, "
            f"rwf_mean={self.rwf_mean}, rwf_std={self.rwf_std}\n"
            f")"
        )
