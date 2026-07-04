"""
Residual computation for 1D wave equation.

Computes: f = h_tt - h_xx = 0
where h is real-valued.
"""

import torch
from typing import Dict, List


def get_relevant_derivatives() -> List[str]:
    """
    Returns the list of derivative terms relevant to Wave1D residual.
    These will be plotted in term_magnitudes and derivative_heatmaps.
    
    Wave1D residual: h_tt - h_xx = 0
    """
    return ['h_tt', 'h_xx']


def get_term_metadata() -> Dict[str, Dict[str, str]]:
    """
    Returns metadata for problem-specific terms in the wave residual.
    
    Wave equation is linear, so there are no additional terms beyond
    the base derivatives.
    
    Returns:
        Empty dict (no problem-specific nonlinear terms)
    """
    return {}


def compute_residual_terms(
    h: torch.Tensor,
    h_tt: torch.Tensor,
    h_xx: torch.Tensor,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Compute 1D wave equation residual.
    
    Residual: f = h_tt - h_xx
    
    Args:
        h: Probe predictions (N, 1) - real-valued
        h_tt: Second temporal derivative (N, 1)
        h_xx: Second spatial derivative (N, 1)
        **kwargs: Ignored (for compatibility, e.g., h_t, h_x)
        
    Returns:
        Dictionary with:
            'residual': Full residual f (N, 1)
    """
    # Compute residual: h_tt - h_xx
    residual = h_tt - h_xx  # (N, 1)
    
    return {
        'residual': residual,
    }

