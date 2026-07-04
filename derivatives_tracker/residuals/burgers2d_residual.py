"""
Residual computation for 2D viscous Burgers equation.

Computes: f = h_t + h*(h_x0 + h_x1) - 0.1*(h_x0x0 + h_x1x1) = 0
where h is real-valued.
"""

import torch
import numpy as np
from typing import Dict, List


def get_relevant_derivatives() -> List[str]:
    """
    Returns the list of derivative terms relevant to Burgers2D residual.
    These will be plotted in term_magnitudes and derivative_heatmaps.

    Burgers2D residual: h_t + h*(h_x0 + h_x1) - 0.1*(h_x0x0 + h_x1x1) = 0
    """
    return ['h_t', 'h_x0', 'h_x1', 'h_x0x0', 'h_x1x1']


def get_term_metadata() -> Dict[str, Dict[str, str]]:
    """
    Returns metadata for problem-specific terms in the Burgers2D residual.

    Returns:
        Dict mapping term_key -> metadata dict with:
            'label': display label for plots
            'marker': matplotlib marker style
            'color': matplotlib color
    """
    return {
        'convection': {
            'label': 'h·(h_x0 + h_x1)',
            'marker': 'd',
            'color': 'purple'
        }
    }


def compute_residual_terms(
    h: torch.Tensor,
    h_t: torch.Tensor,
    h_x0: torch.Tensor,
    h_x1: torch.Tensor,
    h_x0x0: torch.Tensor,
    h_x1x1: torch.Tensor,
    nu: float = 0.1,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Compute 2D viscous Burgers equation residual.

    Residual: f = h_t + h*(h_x0 + h_x1) - nu*(h_x0x0 + h_x1x1)

    Args:
        h: Probe predictions (N, 1) - real-valued
        h_t: Temporal derivative (N, 1)
        h_x0: Spatial derivative w.r.t. x0 (N, 1)
        h_x1: Spatial derivative w.r.t. x1 (N, 1)
        h_x0x0: Second spatial derivative w.r.t. x0 (N, 1)
        h_x1x1: Second spatial derivative w.r.t. x1 (N, 1)
        nu: Viscosity coefficient (default 0.1)
        **kwargs: Ignored (for compatibility)

    Returns:
        Dictionary with:
            'residual': Full residual f (N, 1)
            'convection': Nonlinear convection term h*(h_x0 + h_x1) (N, 1)
    """
    # Convection term: h * (h_x0 + h_x1)
    convection = h * (h_x0 + h_x1)  # (N, 1)

    # Compute residual: h_t + h*(h_x0 + h_x1) - nu*(h_x0x0 + h_x1x1)
    residual = h_t + convection - nu * (h_x0x0 + h_x1x1)  # (N, 1)

    return {
        'residual': residual,
        'convection': convection,
    }

