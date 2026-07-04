"""Fully-connected neural network with named layers."""

import torch
import torch.nn as nn
from functools import partial
from typing import List, Dict, Optional
from models.rwf_layer import RWFLinear
from models.fourier_features import FourierFeatureEmbedding, PeriodicSpatialFourierEmbedding


class FCNet(nn.Module):
    """
    Fully-connected neural network for PINN problems.

    Features:
    - Named layers (layer_1, layer_2, ...)
    - Configurable architecture and activation function
    - Optional Fourier-feature input embedding and RWF hidden layers
    """

    def __init__(self, layers: List[int], activation: str, config: Dict,
                 is_base: bool = True):
        """
        Initialize FCNet.

        Args:
            layers: List of layer sizes [input_dim, hidden1, ..., output_dim]
            activation: Activation function name ('tanh', 'relu', 'sigmoid')
            config: Configuration dict for verification
            is_base: If True (default), assert input_dim == spatial_dim + 1.
                     Set to False for expert networks whose input_dim differs
                     (e.g., ANT experts that take parent activations as input).

        Example:
            Base:   layers = [2, 50, 100, 50, 2]  (input = [x, t])
            Expert: layers = [50, 30, 30, 2]       (input = parent activation)
        """
        super().__init__()

        self.is_base = is_base

        # Verify architecture matches problem configuration
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

        self.layers = layers
        self.activation_name = activation
        self.config = config

        # Get activation function
        self.activation = self._get_activation(activation)

        # Fourier Features: embed input before first linear layer.
        # Disabled for non-base experts (e.g. ANT children) whose input is a
        # parent activation, not raw (x, t) coordinates.
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
            effective_input_dim = self.ff_emb.output_dim  # 2*ff_dim or 4*ff_dim

        # RWF — config['rwf'] is a dict {enabled, mean, std}
        _rwf = config['rwf']
        use_rwf = _rwf['enabled']
        rwf_mean = _rwf.get('mean', 1.0)
        rwf_std = _rwf.get('std', 0.1)
        n_layers = len(layers) - 1  # total linear layers
        LinearCls_hidden = partial(RWFLinear, mean=rwf_mean, std=rwf_std) if use_rwf else nn.Linear

        # Build network with named layers
        # First layer may have expanded input_dim due to FF embedding
        self.network = nn.ModuleDict()

        for i in range(n_layers):
            layer_name = f"layer_{i + 1}"
            is_output_layer = (i == n_layers - 1)
            in_dim = effective_input_dim if i == 0 else layers[i]
            out_dim = layers[i + 1]
            # Output layer always plain nn.Linear for output scale stability
            LinearCls = nn.Linear if is_output_layer else LinearCls_hidden
            self.network[layer_name] = LinearCls(in_dim, out_dim)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
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
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (N, input_dim).
               For base models: input_dim = spatial_dim + 1 (concatenated [x, t]).
               For experts: input_dim = parent's activation dim.
            return_activation: If True, also return the last hidden layer
                activation (before the output layer). Used by ANT to feed
                child experts.

        Returns:
            If return_activation is False: (N, output_dim) tensor.
            If return_activation is True:  tuple of (output, activation)
                where activation is (N, last_hidden_dim).
        """
        out = x

        # Apply Fourier Feature embedding if enabled
        if self.ff_emb is not None:
            out = self.ff_emb(out)

        # Pass through all layers except the last
        layer_names = list(self.network.keys())
        for i, layer_name in enumerate(layer_names[:-1]):
            out = self.network[layer_name](out)
            out = self.activation(out)

        last_hidden = out

        # Last layer (no activation)
        out = self.network[layer_names[-1]](out)

        if return_activation:
            return out, last_hidden
        return out

    def get_activation_dim(self) -> int:
        """Size of the last hidden layer (used by ANT to determine child input dim)."""
        return self.layers[-2]

    def get_layer_names(self) -> List[str]:
        """Get list of all layer names in the network."""
        return list(self.network.keys())

    def __repr__(self) -> str:
        """String representation of the model."""
        layers_str = " -> ".join(map(str, self.layers))
        return (
            f"FCNet(\n"
            f"  architecture: {layers_str}\n"
            f"  activation: {self.activation_name}\n"
            f"  layers: {self.get_layer_names()}\n"
            f")"
        )

