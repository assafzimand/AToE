"""
Physics-Informed Loss Function for the Korteweg-de Vries (KdV) Equation.

Implements the three-component loss:
    L = w_res*MSE_f + w_ic*MSE_0 + w_bc*MSE_b

where:
- MSE_f: PDE residual loss (h_t + h*h_x + mu^2*h_xxx = 0, mu = 0.022)
- MSE_0: Initial condition loss (h(x,0) = cos(pi*x))
- MSE_b: Boundary condition loss (periodic: h(-1,t)=h(1,t), h_x(-1,t)=h_x(1,t))

LEGACY PATHS (DISABLED):
    - Decomposed derivative computation: DISABLED. The analytical 3rd derivative
      of sigmoid indicators (d³ψ/dx³) involves 1/σ³ terms causing catastrophic
      cancellation. Standard autograd on composed output is numerically stable.
    - Analytical indicator derivatives: DISABLED. The compute_analytical_indicator_derivatives
      function was specific to the legacy sigmoid window. With the new compact smoothstep
      windows, all derivatives are computed via autograd on the composed forward output.
      The function is preserved for reference but will not be called.
"""

import torch
import torch.nn as nn
from typing import Dict, Callable, Tuple
from losses.causal_weighting import create_causal_state, compute_causal_residual


def compute_derivatives(
    h: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute derivatives of scalar field h for the KdV equation.
    
    Requires dh/dt, dh/dx, and d³h/dx³ via three autograd calls.
    
    Args:
        h: Scalar field, shape (batch_size,)
        x: Spatial coordinates, shape (batch_size, 1), requires_grad=True
        t: Temporal coordinates, shape (batch_size, 1), requires_grad=True
        
    Returns:
        Tuple of (h_t, h_x, h_xxx)
    """
    ones_h = torch.ones_like(h)

    # Call 1: first derivatives w.r.t. x and t
    h_grads = torch.autograd.grad(
        outputs=h,
        inputs=[x, t],
        grad_outputs=ones_h,
        create_graph=True,
        retain_graph=True,
    )
    h_x = h_grads[0]
    h_t = h_grads[1]

    # Call 2: second spatial derivative
    h_xx = torch.autograd.grad(
        outputs=h_x,
        inputs=x,
        grad_outputs=torch.ones_like(h_x),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Call 3: third spatial derivative
    h_xxx = torch.autograd.grad(
        outputs=h_xx,
        inputs=x,
        grad_outputs=torch.ones_like(h_xx),
        create_graph=True,
        retain_graph=True,
    )[0]

    h_t = h_t.squeeze(-1)
    h_x = h_x.squeeze(-1)
    h_xxx = h_xxx.squeeze(-1)

    return h_t, h_x, h_xxx


def pde_residual(
    h: torch.Tensor,
    h_t: torch.Tensor,
    h_x: torch.Tensor,
    h_xxx: torch.Tensor,
    mu: float = 0.022,
) -> torch.Tensor:
    """
    Compute the PDE residual: h_t + h*h_x + mu^2*h_xxx.

    For the KdV equation: h_t + h*h_x + mu^2*h_xxx = 0
    where mu = 0.022 (Zabusky & Kruskal, 1965 dispersion coefficient).

    Args:
        h: Solution field h
        h_t: Time derivative dh/dt
        h_x: Spatial derivative dh/dx
        h_xxx: Third spatial derivative d³h/dx³
        mu: Dispersion coefficient (default 0.022; code uses mu^2 = 0.000484)

    Returns:
        Residual tensor
    """
    residual = h_t + h * h_x + mu**2 * h_xxx
    return residual


def build_loss(**cfg) -> Callable:
    """
    Build physics-informed loss function for the KdV equation.
    
    Args:
        **cfg: Configuration dictionary containing:
            - problem: problem name (e.g., 'kdv')
            - kdv: dict with 'loss_weights' (residual, ic, bc) and 'mu'
            
    Returns:
        Callable loss function that takes (model, batch) and returns scalar tensor
    """
    problem = cfg['problem']
    problem_config = cfg[problem]
    loss_weights = problem_config['loss_weights']

    weight_residual = loss_weights['residual']
    weight_ic = loss_weights['ic']
    weight_bc = loss_weights['bc']

    mu = problem_config['mu']

    # Disable soft BC penalty when periodic Fourier embedding is used — BC is
    # enforced exactly by the embedding so the MSE term is redundant noise.
    use_bc = not cfg['fourier_features']['periodic']

    causal_state = create_causal_state(problem_config)

    def loss_fn(model: nn.Module, batch: Dict[str, torch.Tensor],
                for_tree_spawning: bool = False,
                return_components: bool = False,
                update_causal_state: bool = True):
        """
        Compute physics-informed loss for the KdV equation.
        
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
        h_gt = batch.get('h_gt', batch.get('u_gt'))  # (N, 1)
        masks = batch['mask']

        N = x.shape[0]
        device = x.device

        _t = getattr(model, '_timer', None)

        if for_tree_spawning:
            residual_per_sample = torch.zeros(N, device=device)
            ic_per_sample = torch.zeros(N, device=device)
            bc_per_sample = torch.zeros(N, device=device)

        # ============================================================
        # MSE_f: PDE Residual Loss
        # ============================================================
        if masks['residual'].sum() > 0:
            x_f = x[masks['residual']].contiguous()
            t_f = t[masks['residual']].contiguous()

            x_f = x_f.clone().detach().requires_grad_(True)
            t_f = t_f.clone().detach().requires_grad_(True)

            xt_f = torch.cat([x_f, t_f], dim=1)

            if _t: _t.start('loss.residual.forward')
            h_pred = model(xt_f)  # (N_f, 1)
            if _t: _t.stop('loss.residual.forward')

            h_f = h_pred[:, 0]

            if _t: _t.start('loss.residual.derivatives')
            h_t_val, h_x_val, h_xxx_val = compute_derivatives(h_f, x_f, t_f)
            if _t: _t.stop('loss.residual.derivatives')

            residual = pde_residual(h_f, h_t_val, h_x_val, h_xxx_val, mu=mu)

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
        # h(x, 0) = cos(pi*x) -- from dataset
        # ============================================================
        if masks['IC'].sum() > 0:
            x_0 = x[masks['IC']].contiguous()
            t_0 = t[masks['IC']].contiguous()
            h_gt_0 = h_gt[masks['IC']].contiguous()  # (N_0, 1)

            xt_0 = torch.cat([x_0, t_0], dim=1)
            if _t: _t.start('loss.ic.forward')
            h_pred_0 = model(xt_0)  # (N_0, 1)
            if _t: _t.stop('loss.ic.forward')

            ic_squared = (h_pred_0 - h_gt_0) ** 2

            if for_tree_spawning:
                ic_per_sample[masks['IC']] = ic_squared.squeeze(-1)
            else:
                mse_ic = torch.mean(ic_squared)
        else:
            if not for_tree_spawning:
                mse_ic = torch.tensor(0.0, device=device)

        # ============================================================
        # MSE_b: Boundary Condition Loss (Periodic)
        # h(-1,t) = h(1,t) and h_x(-1,t) = h_x(1,t)
        # Skipped when periodic Fourier embedding is active (use_bc=False).
        # ============================================================
        if use_bc and masks['BC'].sum() > 0:
            x_b = x[masks['BC']].contiguous()  # (N_b, spatial_dim)
            t_b = t[masks['BC']].contiguous()  # (N_b, 1)

            # Separate BC points by x-coordinate (works correctly after shuffle)
            x_min_val = -1.0
            x_max_val = 1.0
            x_mid = (x_min_val + x_max_val) / 2.0
            
            left_mask = x_b[:, 0] < x_mid
            right_mask = ~left_mask

            x_b_left = x_b[left_mask]
            t_b_left = t_b[left_mask]
            x_b_right = x_b[right_mask]
            t_b_right = t_b[right_mask]

            x_b_left = x_b_left.clone().detach().requires_grad_(True)
            t_b_left = t_b_left.clone().detach().requires_grad_(True)
            x_b_right = x_b_right.clone().detach().requires_grad_(True)
            t_b_right = t_b_right.clone().detach().requires_grad_(True)

            x_stacked = torch.cat([x_b_left, x_b_right], dim=0)
            t_stacked = torch.cat([t_b_left, t_b_right], dim=0)
            n_left = len(x_b_left)
            n_right = len(x_b_right)

            xt_stacked = torch.cat([x_stacked, t_stacked], dim=1)

            if _t: _t.start('loss.bc.forward')
            h_pred_stacked = model(xt_stacked)  # (N_b, 1)
            if _t: _t.stop('loss.bc.forward')

            h_stacked = h_pred_stacked[:, 0]

            if _t: _t.start('loss.bc.derivatives')
            h_b_scalar = h_stacked
            h_x_stacked = torch.autograd.grad(
                h_b_scalar.sum(), x_stacked, create_graph=True, retain_graph=True
            )[0].squeeze(-1)
            if _t: _t.stop('loss.bc.derivatives')

            h_left = h_stacked[:n_left]
            h_right = h_stacked[n_left:]
            h_x_left = h_x_stacked[:n_left]
            h_x_right = h_x_stacked[n_left:]

            if n_left == 0 or n_right == 0:
                if not for_tree_spawning:
                    mse_bc = torch.tensor(0.0, device=device)
            else:
                # Sort both sides by t for pairing
                t_left_vals = t_b_left[:, 0]
                t_right_vals = t_b_right[:, 0]
                sort_left = torch.argsort(t_left_vals)
                sort_right = torch.argsort(t_right_vals)
                
                n_pairs = min(n_left, n_right)
                
                h_left_sorted = h_left[sort_left[:n_pairs]]
                h_right_sorted = h_right[sort_right[:n_pairs]]
                h_x_left_sorted = h_x_left[sort_left[:n_pairs]]
                h_x_right_sorted = h_x_right[sort_right[:n_pairs]]

                bc_value_diff = (h_left_sorted - h_right_sorted) ** 2
                bc_deriv_diff = (h_x_left_sorted - h_x_right_sorted) ** 2

                bc_paired_loss = bc_value_diff + bc_deriv_diff

                if for_tree_spawning:
                    bc_mask_indices = torch.where(masks['BC'])[0]
                    left_bc_indices = bc_mask_indices[left_mask]
                    right_bc_indices = bc_mask_indices[right_mask]
                    
                    left_indices = left_bc_indices[sort_left[:n_pairs]]
                    right_indices = right_bc_indices[sort_right[:n_pairs]]

                    bc_per_sample[left_indices] = bc_paired_loss / 2.0
                    bc_per_sample[right_indices] = bc_paired_loss / 2.0
                else:
                    mse_value = torch.mean(bc_value_diff)
                    mse_derivative = torch.mean(bc_deriv_diff)
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
