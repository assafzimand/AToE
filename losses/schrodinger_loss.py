"""
Physics-Informed Loss Function for the Schrödinger Equation.

Implements the three-component loss:
    L = w_res*MSE_f + w_ic*MSE_0 + w_bc*MSE_b

where:
- MSE_f: PDE residual loss (i*h_t + 0.5*h_xx + |h|²*h = 0)
- MSE_0: Initial condition loss (h(x,0) = 2*sech(x))
- MSE_b: Boundary condition loss (periodic BC)
"""

import torch
import torch.nn as nn
from typing import Dict, Callable, Tuple
import numpy as np
from losses.causal_weighting import create_causal_state, compute_causal_residual


def compute_derivatives(
    u: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute derivatives of complex field h = u + iv using vectorized autograd.
    
    Optimized to use 4 autograd calls (batched) for efficiency.
    Computes ∂h/∂t, ∂h/∂x, and ∂²h/∂x² using PyTorch autograd.
    All operations stay on the same device (GPU if available).
    
    Args:
        u: Real part of field, shape (batch_size,)
        v: Imaginary part of field, shape (batch_size,)
        x: Spatial coordinates, shape (batch_size, 1), requires_grad=True
        t: Temporal coordinates, shape (batch_size, 1), requires_grad=True
        
    Returns:
        Tuple of (h_t, h_x, h_xx):
        - h_t: ∂h/∂t, complex tensor
        - h_x: ∂h/∂x, complex tensor
        - h_xx: ∂²h/∂x², complex tensor
    """
    
    # Create grad_outputs once
    ones_u = torch.ones_like(u)
    ones_v = torch.ones_like(v)
    
    # Call 1: Compute u derivatives w.r.t. BOTH x and t in one call
    u_grads = torch.autograd.grad(
        outputs=u,
        inputs=[x, t],
        grad_outputs=ones_u,
        create_graph=True,
        retain_graph=True,
    )
    u_x = u_grads[0]
    u_t = u_grads[1]
    
    # Call 2: Compute v derivatives w.r.t. BOTH x and t in one call
    v_grads = torch.autograd.grad(
        outputs=v,
        inputs=[x, t],
        grad_outputs=ones_v,
        create_graph=True,
        retain_graph=True,
    )
    v_x = v_grads[0]
    v_t = v_grads[1]
    
    # Second derivatives w.r.t space
    ones_ux = torch.ones_like(u_x)
    ones_vx = torch.ones_like(v_x)
    
    # Call 3: u_xx
    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=x,
        grad_outputs=ones_ux,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Call 4: v_xx
    v_xx = torch.autograd.grad(
        outputs=v_x,
        inputs=x,
        grad_outputs=ones_vx,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Pack as complex (stays on device)
    # Use squeeze(-1) to only remove last dimension, preserve batch dimension
    h_t = torch.complex(u_t, v_t).squeeze(-1)
    h_x = torch.complex(u_x, v_x).squeeze(-1)
    h_xx = torch.complex(u_xx, v_xx).squeeze(-1)
    
    return h_t, h_x, h_xx


def pde_residual(
    h: torch.Tensor,
    h_t: torch.Tensor,
    h_xx: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the PDE residual: i*h_t + 0.5*h_xx + |h|²*h.
    
    For the Schrödinger equation: i*h_t + 0.5*h_xx + |h|²*h = 0
    
    Args:
        h: Complex field
        h_t: Time derivative ∂h/∂t
        h_xx: Second spatial derivative ∂²h/∂x²
        
    Returns:
        Complex residual tensor
    """
    # r = i*h_t + 0.5*h_xx + |h|²*h
    residual = 1j * h_t + 0.5 * h_xx + (h.abs() ** 2) * h
    return residual


def build_loss(**cfg) -> Callable:
    """
    Build physics-informed loss function for the Schrödinger equation.
    
    Args:
        **cfg: Configuration dictionary containing:
            - problem: problem name (e.g., 'problem1')
            - problem1: dict with 'loss_weights' (residual, ic, bc)
            
    Returns:
        Callable loss function that takes (model, batch) and returns
        scalar CUDA tensor
    """
    # Extract loss weights
    problem = cfg['problem']
    problem_config = cfg.get(problem, {})
    loss_weights = problem_config['loss_weights']
    
    weight_residual = loss_weights['residual']
    weight_ic = loss_weights['ic']
    weight_bc = loss_weights['bc']

    # Disable soft BC penalty when periodic Fourier embedding is used — BC is
    # enforced exactly by the embedding so the MSE term is redundant noise.
    use_bc = not cfg['fourier_features']['periodic']

    causal_state = create_causal_state(problem_config)
    
    def loss_fn(model: nn.Module, batch: Dict[str, torch.Tensor],
                for_tree_spawning: bool = False,
                return_components: bool = False,
                update_causal_state: bool = True):
        """
        Compute physics-informed loss for Schrödinger equation.
        
        Args:
            model: Neural network model (output_dim=2 for real, imag)
            batch: Dictionary with keys:
                - 'x': (N, spatial_dim) spatial coordinates
                - 't': (N, 1) temporal coordinates
                - 'h_gt': (N, 2) ground truth h = u + iv as (real, imag)
                - 'mask': dict with 'residual', 'IC', 'BC' boolean masks
            for_tree_spawning: If True, return per-sample loss components dict
                
        Returns:
            - If for_tree_spawning=False: Scalar total loss
            - If for_tree_spawning=True: Dict with keys 'residual', 'ic', 'bc'
              containing per-sample loss tensors (N,)
        """
        x = batch['x']  # (N, spatial_dim)
        t = batch['t']  # (N, 1)
        h_gt = batch['h_gt']  # (N, 2) as (real, imag)
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
            x_f = x[masks['residual']].contiguous()  # (N_f, spatial_dim)
            t_f = t[masks['residual']].contiguous()  # (N_f, 1)
            
            # Enable gradients for autograd
            x_f = x_f.clone().detach().requires_grad_(True)
            t_f = t_f.clone().detach().requires_grad_(True)
            
            # Model prediction: concatenate x,t -> predict (u,v)
            xt_f = torch.cat([x_f, t_f], dim=1)
            
            # === Standard approach: differentiate composed output directly ===
            if _t: _t.start('loss.residual.forward')
            uv_f = model(xt_f)  # (N_f, 2)
            if _t: _t.stop('loss.residual.forward')
            
            u_f = uv_f[:, 0]
            v_f = uv_f[:, 1]
            
            if _t: _t.start('loss.residual.derivatives')
            h_t, h_x, h_xx = compute_derivatives(u_f, v_f, x_f, t_f)
            if _t: _t.stop('loss.residual.derivatives')
            
            # Compute PDE residual: i*h_t + 0.5*h_xx + |h|²*h
            # Need h for |h|² term
            h_f = torch.complex(u_f, v_f)
            residual = pde_residual(h_f, h_t, h_xx)

            # Per-sample squared residual
            residual_squared = residual.real ** 2 + residual.imag ** 2

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
        # ============================================================
        if masks['IC'].sum() > 0:
            # Boolean indexing + .contiguous() for GPU efficiency
            x_0 = x[masks['IC']].contiguous()  # (N_0, spatial_dim)
            t_0 = t[masks['IC']].contiguous()  # (N_0, 1)
            h_gt_0 = h_gt[masks['IC']].contiguous()  # (N_0, 2)
            
            # Model prediction
            xt_0 = torch.cat([x_0, t_0], dim=1)
            if _t: _t.start('loss.ic.forward')
            uv_0 = model(xt_0)  # (N_0, 2)
            if _t: _t.stop('loss.ic.forward')
            
            # Convert to complex
            h_pred = torch.complex(uv_0[:, 0], uv_0[:, 1])
            h_true = torch.complex(h_gt_0[:, 0], h_gt_0[:, 1])
            
            # MSE: |h_pred - h_true|²
            diff = h_pred - h_true
            ic_squared = diff.real ** 2 + diff.imag ** 2
            
            if for_tree_spawning:
                ic_per_sample[masks['IC']] = ic_squared
            else:
                mse_ic = torch.mean(ic_squared)
        else:
            if not for_tree_spawning:
                mse_ic = torch.tensor(0.0, device=device)
        
        # ============================================================
        # MSE_b: Boundary Condition Loss (Periodic)
        # Skipped when periodic Fourier embedding is active (use_bc=False).
        # ============================================================
        if use_bc and masks['BC'].sum() > 0:
            # Boolean indexing + .contiguous() for GPU efficiency
            x_b = x[masks['BC']].contiguous()  # (N_b, spatial_dim)
            t_b = t[masks['BC']].contiguous()  # (N_b, 1)
            
            # Separate BC points by x-coordinate (works correctly after shuffle)
            # Left boundary: x close to x_min (-5)
            # Right boundary: x close to x_max (+5)
            x_min_val = -5.0
            x_max_val = 5.0
            x_mid = (x_min_val + x_max_val) / 2.0
            
            left_mask = x_b[:, 0] < x_mid
            right_mask = ~left_mask
            
            x_b_left = x_b[left_mask]
            t_b_left = t_b[left_mask]
            x_b_right = x_b[right_mask]
            t_b_right = t_b[right_mask]
            
            # Enable gradients for derivative computation
            x_b_left = x_b_left.clone().detach().requires_grad_(True)
            t_b_left = t_b_left.clone().detach().requires_grad_(True)
            x_b_right = x_b_right.clone().detach().requires_grad_(True)
            t_b_right = t_b_right.clone().detach().requires_grad_(True)
            
            # Vectorized: stack left and right, then split back
            x_stacked = torch.cat([x_b_left, x_b_right], dim=0)
            t_stacked = torch.cat([t_b_left, t_b_right], dim=0)
            n_b_left = len(x_b_left)
            n_b_right = len(x_b_right)
            
            # Single forward pass for both boundaries
            xt_stacked = torch.cat([x_stacked, t_stacked], dim=1)
            
            # === Standard approach ===
            if _t: _t.start('loss.bc.forward')
            uv_stacked = model(xt_stacked)
            if _t: _t.stop('loss.bc.forward')
            u_stacked = uv_stacked[:, 0]
            v_stacked = uv_stacked[:, 1]
            
            if _t: _t.start('loss.bc.derivatives')
            _, h_x_stacked, _ = compute_derivatives(u_stacked, v_stacked, x_stacked, t_stacked)
            if _t: _t.stop('loss.bc.derivatives')
            
            # Split predictions and derivatives
            u_left = u_stacked[:n_b_left]
            u_right = u_stacked[n_b_left:]
            v_left = v_stacked[:n_b_left]
            v_right = v_stacked[n_b_left:]
            h_x_left = h_x_stacked[:n_b_left]
            h_x_right = h_x_stacked[n_b_left:]
            
            # Convert to complex for comparison
            h_left = torch.complex(u_left, v_left)
            h_right = torch.complex(u_right, v_right)
            
            # Periodic BC: h(-5,t) = h(5,t) and h_x(-5,t) = h_x(5,t)
            # Match pairs by t-value (handles shuffle correctly)
            n_left_pts = len(h_left)
            n_right_pts = len(h_right)
            
            if n_left_pts == 0 or n_right_pts == 0:
                if not for_tree_spawning:
                    mse_bc = torch.tensor(0.0, device=device)
            else:
                # Sort both sides by t for pairing
                t_left_vals = t_b_left[:, 0]
                t_right_vals = t_b_right[:, 0]
                sort_left = torch.argsort(t_left_vals)
                sort_right = torch.argsort(t_right_vals)
                
                # Take min number of pairs available
                n_pairs = min(n_left_pts, n_right_pts)
                
                # Apply sorting for pairing
                h_left_sorted = h_left[sort_left[:n_pairs]]
                h_right_sorted = h_right[sort_right[:n_pairs]]
                h_x_left_sorted = h_x_left[sort_left[:n_pairs]]
                h_x_right_sorted = h_x_right[sort_right[:n_pairs]]
                
                # Compute differences
                diff_value = h_left_sorted - h_right_sorted
                bc_value_squared = diff_value.real ** 2 + diff_value.imag ** 2
                
                diff_derivative = h_x_left_sorted - h_x_right_sorted
                bc_deriv_squared = diff_derivative.real ** 2 + diff_derivative.imag ** 2
                
                bc_paired_loss = bc_value_squared + bc_deriv_squared
                
                if for_tree_spawning:
                    # Map sorted indices back to original BC mask indices
                    bc_mask_indices = torch.where(masks['BC'])[0]
                    left_bc_indices = bc_mask_indices[left_mask]
                    right_bc_indices = bc_mask_indices[right_mask]
                    
                    left_indices = left_bc_indices[sort_left[:n_pairs]]
                    right_indices = right_bc_indices[sort_right[:n_pairs]]
                    
                    bc_per_sample[left_indices] = bc_paired_loss / 2.0
                    bc_per_sample[right_indices] = bc_paired_loss / 2.0
                else:
                    mse_value = torch.mean(bc_value_squared)
                    mse_derivative = torch.mean(bc_deriv_squared)
                    mse_bc = mse_value + mse_derivative
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
            total_loss = weight_residual * mse_residual + weight_ic * mse_ic
            if use_bc:
                comps['bc'] = mse_bc
                total_loss = total_loss + weight_bc * mse_bc
            comps['total'] = total_loss
            return comps
        else:
            total_loss = weight_residual * mse_residual + weight_ic * mse_ic
            if use_bc:
                total_loss = total_loss + weight_bc * mse_bc
            return total_loss

    loss_fn.causal_state = causal_state
    return loss_fn
