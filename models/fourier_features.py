"""Fourier Feature Embedding for PINN models.

Maps input z ∈ ℝ^{input_dim} to [cos(Bz), sin(Bz)] ∈ ℝ^{2*fourier_dim}
using a fixed random projection matrix B ~ N(0, scale²).

The projection matrix B is stored as a non-trainable buffer so it:
- Moves to the correct device automatically with .to(device)
- Is excluded from optimizer parameter groups
- Is saved/loaded with model state_dict

Also provides PeriodicSpatialFourierEmbedding for problems with periodic
spatial BCs (e.g., KS). Enabled via fourier_features.periodic: true in config.

Reference: Tancik et al. (2020) "Fourier Features Let Networks Learn High
Frequency Functions in Low Dimensional Domains."
"""

import math
import torch
import torch.nn as nn


class FourierFeatureEmbedding(nn.Module):
    """Random Fourier Feature embedding with frozen projection matrix.

    Args:
        input_dim: Dimension of the input (e.g., spatial_dim + 1).
        fourier_dim: Number of Fourier features. Output size is 2*fourier_dim.
        scale: Standard deviation of B initialization. Controls frequency band.
               Higher scale → captures higher frequencies.
    """

    def __init__(self, input_dim: int, fourier_dim: int, scale: float = 1.0):
        super().__init__()
        B = torch.randn(fourier_dim, input_dim) * scale
        self.register_buffer('B', B)

    @property
    def output_dim(self) -> int:
        """Output dimensionality: 2 * fourier_dim."""
        return 2 * self.B.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map input to Fourier feature space.

        Args:
            x: Input tensor of shape (N, input_dim).

        Returns:
            Fourier features of shape (N, 2 * fourier_dim).
        """
        x = x.to(self.B.dtype)  # Cast to match B's precision
        proj = x @ self.B.T  # (N, fourier_dim)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

    def extra_repr(self) -> str:
        fourier_dim, input_dim = self.B.shape
        return (
            f"input_dim={input_dim}, fourier_dim={fourier_dim}, "
            f"output_dim={self.output_dim}, scale=frozen"
        )


class PeriodicSpatialFourierEmbedding(nn.Module):
    """Periodic Fourier embedding matching jaxpi (Wang et al. 2024).

    Step 1 — Periodic transform: x → [cos(2π/L · x), sin(2π/L · x)].
             One harmonic per spatial axis, exactly periodic with period L.
             No BC penalty needed.

    Step 2 — Random Fourier features: apply standard random projection B to
             the combined input [t, cos(2πx/L), sin(2πx/L)], creating
             space-time cross-terms so the network can learn t-dependent dynamics.

    Output dim = 2 * fourier_dim.

    Args:
        spatial_dim: Number of spatial input dimensions (1 for KS).
        fourier_dim: Number of random Fourier features. Output = 2 * fourier_dim.
        scale: Std of random B matrix (controls frequency band).
        L: Spatial period (e.g., 2π for KS on [0, 2π]).
    """

    def __init__(self, spatial_dim: int, fourier_dim: int, scale: float = 1.0,
                 L: float = 2 * math.pi):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.L = L
        # After periodic transform: [t (1), cos(x) (spatial_dim), sin(x) (spatial_dim)]
        transformed_dim = 1 + 2 * spatial_dim
        B = torch.randn(fourier_dim, transformed_dim) * scale
        self.register_buffer('B', B)    # (fourier_dim, 1 + 2*spatial_dim)

    @property
    def output_dim(self) -> int:
        """Output dimensionality: 2 * fourier_dim."""
        return 2 * self.B.shape[0]

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        """Map (x, t) input to periodic Fourier features.

        Args:
            xt: Input tensor (N, spatial_dim + 1). Last column is t.

        Returns:
            Feature tensor (N, 2 * fourier_dim).
        """
        x = xt[:, :self.spatial_dim]   # (N, spatial_dim)
        t = xt[:, self.spatial_dim:]   # (N, 1)

        # Step 1: periodic transform — one harmonic per spatial axis
        freq = 2 * math.pi / self.L
        cos_x = torch.cos(freq * x)    # (N, spatial_dim)
        sin_x = torch.sin(freq * x)    # (N, spatial_dim)

        # Step 2: random Fourier features on [t, cos(x), sin(x)]
        z = torch.cat([t, cos_x, sin_x], dim=-1)   # (N, 1+2*spatial_dim)
        z = z.to(self.B.dtype)                       # Cast to match B's precision
        proj = z @ self.B.T                          # (N, fourier_dim)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

    def extra_repr(self) -> str:
        fourier_dim = self.B.shape[0]
        return (
            f"spatial_dim={self.spatial_dim}, fourier_dim={fourier_dim}, "
            f"output_dim={self.output_dim}, L={self.L:.4f}, periodic=True"
        )
