"""
Residual computation for Schrödinger equation.

Computes: f = i*h_t + 0.5*h_xx + |h|²*h = 0
where h is complex-valued (h = u + iv).
"""

import torch
from typing import Dict, List


def get_relevant_derivatives() -> List[str]:
    """
    Returns the list of derivative terms relevant to Schrödinger residual.
    These will be plotted in term_magnitudes and derivative_heatmaps.
    
    Schrödinger residual: i*h_t + 0.5*h_xx + |h|²*h = 0
    """
    return ['h', 'h_t', 'h_xx']


def get_term_metadata() -> Dict[str, Dict[str, str]]:
    """
    Returns metadata for problem-specific terms in the Schrödinger residual.
    
    Returns:
        Dict mapping term_key -> metadata dict with:
            'label': display label for plots
            'marker': matplotlib marker style
            'color': matplotlib color
    """
    return {
        'cubic_nonlinearity': {
            'label': '|h|²h',
            'marker': 'd',
            'color': 'purple'
        }
    }


def compute_residual_terms(
    h: torch.Tensor,
    h_t: torch.Tensor,
    h_xx: torch.Tensor,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Compute Schrödinger residual terms using complex arithmetic.
    
    Given h = u + iv (stored as [u, v] in columns):
    - |h|² = u² + v²
    - |h|²h = (|h|²u, |h|²v)
    - Residual f = i*h_t + 0.5*h_xx + |h|²*h
      - f_u (real part) = -v_t + 0.5*u_xx + (u²+v²)*u
      - f_v (imag part) = u_t + 0.5*v_xx + (u²+v²)*v
    
    Args:
        h: Probe predictions (N, 2) where [:,0]=u, [:,1]=v
        h_t: Temporal derivatives (N, 2)
        h_xx: Second spatial derivatives (N, 2)
        **kwargs: Ignored (for compatibility with other residual functions)
        
    Returns:
        Dictionary with:
            'cubic_nonlinearity': |h|²*h term (N, 2)
            'residual': Full residual f (N, 2)
            'h_magnitude_sq': |h|² (N,) for reference
    """
    # Extract u and v
    u = h[:, 0:1]  # (N, 1)
    v = h[:, 1:2]  # (N, 1)
    
    u_t = h_t[:, 0:1]
    v_t = h_t[:, 1:2]
    
    u_xx = h_xx[:, 0:1]
    v_xx = h_xx[:, 1:2]
    
    # Compute |h|² = u² + v²
    h_mag_sq = u**2 + v**2  # (N, 1)
    
    # Compute cubic nonlinear term: |h|²*h = (|h|²*u, |h|²*v)
    cubic_u = h_mag_sq * u
    cubic_v = h_mag_sq * v
    cubic_nonlinearity = torch.cat([cubic_u, cubic_v], dim=1)  # (N, 2)
    
    # Compute residual using complex arithmetic
    # f = i*h_t + 0.5*h_xx + |h|²*h
    # Real part: f_u = -v_t + 0.5*u_xx + |h|²*u
    # Imag part: f_v = u_t + 0.5*v_xx + |h|²*v
    
    residual_u = -v_t + 0.5 * u_xx + cubic_u
    residual_v = u_t + 0.5 * v_xx + cubic_v
    residual = torch.cat([residual_u, residual_v], dim=1)  # (N, 2)
    
    return {
        'cubic_nonlinearity': cubic_nonlinearity,
        'residual': residual,
        'h_magnitude_sq': h_mag_sq.squeeze()  # (N,)
    }

