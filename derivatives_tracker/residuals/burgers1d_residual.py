"""
Residual computation for 1D viscous Burgers equation.

Computes: f = h_t + h*h_x - (nu/pi)*h_xx = 0
where h is real-valued.
"""

import torch
import numpy as np
from typing import Dict, List


def get_relevant_derivatives() -> List[str]:
    """
    Returns the list of derivative terms relevant to Burgers1D residual.
    These will be plotted in term_magnitudes and derivative_heatmaps.

    Burgers1D residual: h_t + h*h_x - (nu/pi)*h_xx = 0
    """
    return ['h_t', 'h_x', 'h_xx']


def get_term_metadata() -> Dict[str, Dict[str, str]]:
    """
    Returns metadata for problem-specific terms in the Burgers residual.

    Returns:
        Dict mapping term_key -> metadata dict with:
            'label': display label for plots
            'marker': matplotlib marker style
            'color': matplotlib color
    """
    return {
        'convection': {
            'label': 'h·h_x',
            'marker': 'd',
            'color': 'purple'
        }
    }


def compute_residual_terms(
    h: torch.Tensor,
    h_t: torch.Tensor,
    h_x: torch.Tensor,
    h_xx: torch.Tensor,
    nu: float = 0.01,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Compute 1D viscous Burgers equation residual.

    Residual: f = h_t + h*h_x - (nu/pi)*h_xx

    Args:
        h: Probe predictions (N, 1) - real-valued
        h_t: Temporal derivative (N, 1)
        h_x: Spatial derivative (N, 1)
        h_xx: Second spatial derivative (N, 1)
        nu: Viscosity coefficient (default 0.01)
        **kwargs: Ignored (for compatibility, e.g., h_tt)

    Returns:
        Dictionary with:
            'residual': Full residual f (N, 1)
            'convection': Nonlinear convection term h*h_x (N, 1)
    """
    # Compute viscosity coefficient
    visc = nu / np.pi

    # Convection term: h * h_x
    convection = h * h_x  # (N, 1)

    # Compute residual: h_t + h*h_x - (nu/pi)*h_xx
    residual = h_t + convection - visc * h_xx  # (N, 1)

    return {
        'residual': residual,
        'convection': convection,
    }
