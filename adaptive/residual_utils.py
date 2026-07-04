"""Utility functions for computing point-wise PDE residuals.

Used by the adaptive region detector to weight wavelet norms by error.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Callable


def compute_loss_components(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    target: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    loss_fn: Callable,
    weights: Dict[str, float]
) -> Dict[str, np.ndarray]:
    """
    Compute per-sample loss components for tree spawning region weighting.
    
    This function calls the loss function in for_tree_spawning mode to get
    per-sample loss tensors for residual, IC, and BC components. These are
    then used by the region detector to compute total_loss(ω_i) for each
    RF tree node region.
    
    Args:
        model: The PINN model
        x: (N, spatial_dim) spatial coordinates
        t: (N, 1) temporal coordinates
        target: (N, output_dim) ground truth solution (h_gt or u_gt)
        masks: Dict with 'residual', 'IC', 'BC' boolean masks
        loss_fn: Loss function that accepts for_tree_spawning parameter
        weights: Dict with 'residual', 'ic', 'bc' weights
        
    Returns:
        Dict with keys:
            - 'residual': (N,) numpy array of per-sample residual losses
            - 'ic': (N,) numpy array of per-sample IC losses
            - 'bc': (N,) numpy array of per-sample BC losses
            - 'weights': Dict of loss weights
    """
    # Enable gradients for inputs (needed for PDE derivative computation)
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    
    batch = {
        'x': x,
        't': t,
        'h_gt': target,  # Works for both h_gt and u_gt naming
        'mask': masks
    }
    
    # No torch.no_grad() here - we need the computation graph to compute
    # derivatives w.r.t. inputs. Model parameters won't be updated since
    # we never call .backward() on the loss.
    loss_components = loss_fn(model, batch, for_tree_spawning=True)
    
    return {
        'residual': loss_components['residual'].detach().cpu().numpy(),
        'ic': loss_components['ic'].detach().cpu().numpy(),
        'bc': loss_components['bc'].detach().cpu().numpy(),
        'weights': weights
    }
