"""
Physics-Informed Loss Function for the 1D Viscous Burgers Equation.

Implements the three-component loss:
    L = w_res*MSE_f + w_ic*MSE_0 + w_bc*MSE_b

where:
- MSE_f: PDE residual loss (h_t + h*h_x - (nu/pi)*h_xx = 0)
- MSE_0: Initial condition loss (h(0,x) = -sin(pi*x))
- MSE_b: Boundary condition loss (Dirichlet: h(t,-1) = h(t,1) = 0)

LEGACY PATHS (DISABLED):
    - Decomposed derivative computation: DISABLED. With compact smoothstep windows,
      all derivatives are computed via autograd on composed output.
    - Analytical indicator derivatives: DISABLED. The compute_analytical_indicator_derivatives
      function was specific to the legacy sigmoid window and is preserved for reference only.
"""

import torch
import torch.nn as nn
from typing import Dict, Callable, Tuple
import numpy as np
from losses.causal_weighting import create_causal_state, compute_causal_residual


def compute_derivatives(
    h: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute derivatives of scalar field h using vectorized autograd.
    
    Computes dh/dt, dh/dx, and d²h/dx² using PyTorch autograd.
    All operations stay on the same device (GPU if available).
    
    Args:
        h: Scalar field, shape (batch_size,)
        x: Spatial coordinates, shape (batch_size, 1), requires_grad=True
        t: Temporal coordinates, shape (batch_size, 1), requires_grad=True
        
    Returns:
        Tuple of (h_t, h_x, h_xx):
        - h_t: dh/dt
        - h_x: dh/dx
        - h_xx: d²h/dx²
    """
    # Create grad_outputs once
    ones_h = torch.ones_like(h)
    
    # Compute h derivatives w.r.t. BOTH x and t in one call
    h_grads = torch.autograd.grad(
        outputs=h,
        inputs=[x, t],
        grad_outputs=ones_h,
        create_graph=True,
        retain_graph=True,
    )
    h_x = h_grads[0]
    h_t = h_grads[1]
    
    # Second derivative w.r.t. x
    ones_hx = torch.ones_like(h_x)
    
    h_xx = torch.autograd.grad(
        outputs=h_x,
        inputs=x,
        grad_outputs=ones_hx,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Squeeze to remove trailing dimensions
    h_t = h_t.squeeze(-1)
    h_x = h_x.squeeze(-1)
    h_xx = h_xx.squeeze(-1)
    
    return h_t, h_x, h_xx


def pde_residual(
    h: torch.Tensor,
    h_t: torch.Tensor,
    h_x: torch.Tensor,
    h_xx: torch.Tensor,
    nu: float = 0.01,
) -> torch.Tensor:
    """
    Compute the PDE residual: h_t + h*h_x - (nu/pi)*h_xx.
    
    For the viscous Burgers equation: h_t + h*h_x - (nu/pi)*h_xx = 0
    
    Args:
        h: Solution field h
        h_t: Time derivative dh/dt
        h_x: Spatial derivative dh/dx
        h_xx: Second spatial derivative d²h/dx²
        nu: Viscosity coefficient (default 0.01)
        
    Returns:
        Residual tensor
    """
    visc = nu / np.pi
    residual = h_t + h * h_x - visc * h_xx
    return residual


def build_loss(**cfg) -> Callable:
    """
    Build physics-informed loss function for the 1D viscous Burgers equation.
    
    Args:
        **cfg: Configuration dictionary containing:
            - problem: problem name (e.g., 'burgers1d')
            - burgers1d: dict with 'loss_weights' (residual, ic, bc) and 'nu'
            
    Returns:
        Callable loss function that takes (model, batch) and returns
        scalar tensor
    """
    # Extract loss weights and parameters
    problem = cfg['problem']
    problem_config = cfg[problem]
    loss_weights = problem_config['loss_weights']
    
    weight_residual = loss_weights['residual']
    weight_ic = loss_weights['ic']
    weight_bc = loss_weights['bc']
    
    # Get viscosity parameter
    nu = problem_config['nu']

    # Disable soft BC penalty when periodic Fourier embedding is used — BC is
    # enforced exactly by the embedding so the MSE term is redundant noise.
    use_bc = not cfg['fourier_features']['periodic']

    causal_state = create_causal_state(problem_config)
    
    def loss_fn(model: nn.Module, batch: Dict[str, torch.Tensor],
                for_tree_spawning: bool = False,
                return_components: bool = False,
                update_causal_state: bool = True):
        """
        Compute physics-informed loss for 1D viscous Burgers equation.
        
        Args:
            model: Neural network model (output_dim=1 for real-valued h)
            batch: Dictionary with keys:
                - 'x': (N, spatial_dim) spatial coordinates
                - 't': (N, 1) temporal coordinates
                - 'h_gt': (N, 1) ground truth (for IC)
                - 'mask': dict with 'residual', 'IC', 'BC' boolean masks
            for_tree_spawning: If True, return per-sample loss components dict
                
        Returns:
            - If for_tree_spawning=False: Scalar total loss
            - If for_tree_spawning=True: Dict with keys 'residual', 'ic', 'bc'
              containing per-sample loss tensors (N,)
        """
        x = batch['x']  # (N, spatial_dim)
        t = batch['t']  # (N, 1)
        h_gt = batch.get('h_gt', batch.get('u_gt'))  # (N, 1) - handle both names
        masks = batch['mask']  # dict with boolean masks
        
        N = x.shape[0]
        device = x.device
        
        # Timer (attached to model by trainer)
        _t = getattr(model, '_timer', None)
        
        # Initialize per-sample arrays if needed
        if for_tree_spawning:
            residual_per_sample = torch.zeros(N, device=device)
            ic_per_sample = torch.zeros(N, device=device)
            bc_per_sample = torch.zeros(N, device=device)
        
        # ============================================================
        # MSE_f: PDE Residual Loss
        # ============================================================
        if masks['residual'].sum() > 0:
            # Boolean indexing + .contiguous() for GPU efficiency
            x_f = x[masks['residual']].contiguous()
            t_f = t[masks['residual']].contiguous()
            
            # Enable gradients for autograd
            x_f = x_f.clone().detach().requires_grad_(True)
            t_f = t_f.clone().detach().requires_grad_(True)
            
            # Model prediction: concatenate x,t -> predict h
            xt_f = torch.cat([x_f, t_f], dim=1)
            
            if _t: _t.start('loss.residual.forward')
            h_pred = model(xt_f)  # (N_f, 1)
            if _t: _t.stop('loss.residual.forward')
            
            h_f = h_pred[:, 0]
            
            if _t: _t.start('loss.residual.derivatives')
            h_t, h_x, h_xx = compute_derivatives(h_f, x_f, t_f)
            if _t: _t.stop('loss.residual.derivatives')
            
            # Compute PDE residual: h_t + h*h_x - (nu/pi)*h_xx
            residual = pde_residual(h_f, h_t, h_x, h_xx, nu=nu)
            
            # Per-sample squared residual
            residual_squared = residual ** 2
            
            if for_tree_spawning:
                residual_per_sample[masks['residual']] = residual_squared
            else:
                # Cache residuals for adaptive sampling (no-op when disabled)
                if getattr(model, '_residual_cache_enabled', False):
                    model._residual_cache.append((
                        x_f.detach().clone(),
                        t_f.detach().clone(),
                        residual_squared.detach().clone(),
                    ))
                mse_residual = compute_causal_residual(residual_squared, t_f, causal_state, update_state=update_causal_state)
        else:
            if not for_tree_spawning:
                mse_residual = torch.tensor(0.0, device=device)
        
        # ============================================================
        # MSE_0: Initial Condition Loss
        # h(0, x) = -sin(pi*x)
        # ============================================================
        if masks['IC'].sum() > 0:
            x_0 = x[masks['IC']].contiguous()
            t_0 = t[masks['IC']].contiguous()
            h_gt_0 = h_gt[masks['IC']].contiguous()  # (N_0, 1)
            
            # Model prediction
            xt_0 = torch.cat([x_0, t_0], dim=1)
            if _t: _t.start('loss.ic.forward')
            h_pred_0 = model(xt_0)  # (N_0, 1)
            if _t: _t.stop('loss.ic.forward')
            
            # IC: h(0, x) = -sin(pi*x)
            ic_squared = (h_pred_0 - h_gt_0) ** 2
            
            if for_tree_spawning:
                ic_per_sample[masks['IC']] = ic_squared.squeeze(-1)
            else:
                mse_ic = torch.mean(ic_squared)
        else:
            if not for_tree_spawning:
                mse_ic = torch.tensor(0.0, device=device)
        
        # ============================================================
        # MSE_b: Boundary Condition Loss
        # h(t, -1) = h(t, 1) = 0 (Dirichlet)
        # Skipped when periodic Fourier embedding is active (use_bc=False).
        # ============================================================
        if use_bc and masks['BC'].sum() > 0:
            x_b = x[masks['BC']].contiguous()
            t_b = t[masks['BC']].contiguous()
            
            # Model prediction at boundary points
            xt_b = torch.cat([x_b, t_b], dim=1)
            if _t: _t.start('loss.bc.forward')
            h_pred_b = model(xt_b)  # (N_b, 1)
            if _t: _t.stop('loss.bc.forward')
            
            # BC: h should be 0 at boundaries (Dirichlet)
            bc_squared = h_pred_b ** 2
            
            if for_tree_spawning:
                bc_per_sample[masks['BC']] = bc_squared.squeeze(-1)
            else:
                mse_bc = torch.mean(bc_squared)
        else:
            if not for_tree_spawning:
                mse_bc = torch.tensor(0.0, device=device)
        
        # ============================================================
        # Return
        # ============================================================
        if for_tree_spawning:
            return {
                'residual': residual_per_sample,  # (N,)
                'ic': ic_per_sample,              # (N,)
                'bc': bc_per_sample               # (N,)
            }
        elif return_components:
            comps = {'residual': mse_residual, 'ic': mse_ic}
            if use_bc:
                comps['bc'] = mse_bc
            return comps
        else:
            total_loss = weight_residual * mse_residual + weight_ic * mse_ic
            if use_bc:
                total_loss = total_loss + weight_bc * mse_bc
            return total_loss

    loss_fn.causal_state = causal_state
    return loss_fn
