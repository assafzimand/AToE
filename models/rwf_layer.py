"""Random Weight Factorization (RWF) linear layer.

Factorizes the weight matrix as W_eff = diag(g) @ v, where g is a trainable
per-neuron scale vector and v is the base weight. This matches JaxPI's
implementation exactly.

Initialization:
    g = exp(N(mean, std))  -- scale parameter stored directly (not log)
    v = W_init / g         -- base weight preserves effective init scale

Forward:
    W_eff = g * v          -- no exp() in forward pass

Reference: Wang et al. (2022) "Random Weight Factorization Improves the
Training of Continuous Neural Representations."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RWFLinear(nn.Linear):
    """Linear layer with Random Weight Factorization (JaxPI-compatible).

    W_eff = diag(scale) @ weight
    output = x @ W_eff.T + bias

    Matches JaxPI exactly: scale is stored directly (not as log_scale),
    and the forward pass uses direct multiplication without exp().

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias.
        mean: Mean of the log-normal init for scale (default 1.0, JaxPI).
        std: Std of the log-normal init for scale (default 0.1, JaxPI).
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, mean: float = 1.0, std: float = 0.1):
        super().__init__(in_features, out_features, bias=bias)
        # Initialize scale as exp(N(mean, std)) - stored directly, not as log
        self.scale = nn.Parameter(
            torch.exp(torch.empty(out_features).normal_(mean, std))
        )
        # Divide base weight by scale so effective weight equals original init
        with torch.no_grad():
            self.weight.data = self.weight.data / self.scale.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Direct multiplication - no exp() here (matches JaxPI)
        W = self.scale.unsqueeze(1) * self.weight
        return F.linear(x, W, self.bias)

    def extra_repr(self) -> str:
        scale_mean = self.scale.data.mean().item()
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, rwf=True, "
            f"scale_mean~{scale_mean:.3f}"
        )
