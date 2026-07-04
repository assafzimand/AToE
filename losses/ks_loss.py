"""
Physics-Informed Loss Function for the Kuramoto-Sivashinsky (KS) Equation.

Implements the three-component loss:
    L = w_res*MSE_f + w_ic*MSE_0 + w_bc*MSE_b

where:
- MSE_f: PDE residual loss (h_t + alpha*h*h_x + beta*h_xx + gamma*h_xxxx = 0)
- MSE_0: Initial condition loss (h(x,0) = cos(x)*(1+sin(x)))
- MSE_b: Boundary condition loss (periodic: h(0,t)=h(2*pi,t), h_x(0,t)=h_x(2*pi,t))

Parameters: alpha = 100/16, beta = 100/16^2, gamma = 100/16^4
(PirateNet benchmark; Wang et al., JMLR 2024)

LEGACY PATHS (DISABLED):
    - Decomposed derivative computation: DISABLED. The analytical 4th derivative
      of sigmoid indicators (d⁴ψ/dx⁴) involves 1/σ⁴ terms causing catastrophic
      cancellation. Standard autograd on composed output is numerically stable.
    - Analytical indicator derivatives: DISABLED. The compute_analytical_indicator_derivatives
      function was specific to the legacy sigmoid window. With the new compact smoothstep
      windows, all derivatives are computed via autograd on the composed forward output.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Callable, Tuple
from losses.causal_weighting import create_causal_state, compute_causal_residual

# Mutable context set by trainer (e.g. f"epoch {epoch} batch {b}") so prints are traceable.
_nan_ctx: list = ['']


def _chk(tensor: torch.Tensor, name: str) -> bool:
    """Check tensor for NaN/Inf every call; print detailed stats on first bad occurrence.

    Cost in the clean case: one GPU reduction (.any()) — negligible.
    """
    with torch.no_grad():
        t = tensor.detach().float().reshape(-1)
        bad = ~torch.isfinite(t)
        if not bad.any().item():
            return False
        n_bad = bad.sum().item()
        frac = n_bad / max(t.numel(), 1)
        max_abs = t.abs().max().item()
        finite = t[~bad]
        fin_mean = finite.mean().item() if finite.numel() > 0 else float('nan')
        fin_std = finite.std().item() if finite.numel() > 1 else 0.0
        has_nan = torch.isnan(t[bad]).any().item()
        label = 'NaN' if has_nan else 'Inf'
    ctx = _nan_ctx[0]
    print(f"  [NaN-Debug{' ' + ctx if ctx else ''}] {label} in '{name}': "
          f"n_bad={int(n_bad)}/{t.numel()} ({frac:.0%}), "
          f"max_abs={max_abs:.3e}, finite_mean={fin_mean:.3e}, finite_std={fin_std:.3e}")
    return True


def compute_derivatives(
    h: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute derivatives of scalar field h for the KS equation.

    Requires dh/dt, dh/dx, d²h/dx², and d⁴h/dx⁴ via four autograd calls.

    Returns:
        Tuple of (h_t, h_x, h_xx, h_xxxx)
    """
    ones_h = torch.ones_like(h)

    h_grads = torch.autograd.grad(
        outputs=h, inputs=[x, t], grad_outputs=ones_h,
        create_graph=True, retain_graph=True,
    )
    h_x = h_grads[0]
    h_t = h_grads[1]
    _chk(h_x, 'h_x')
    _chk(h_t, 'h_t')

    h_xx = torch.autograd.grad(
        outputs=h_x, inputs=x, grad_outputs=torch.ones_like(h_x),
        create_graph=True, retain_graph=True,
    )[0]
    _chk(h_xx, 'h_xx')

    h_xxx = torch.autograd.grad(
        outputs=h_xx, inputs=x, grad_outputs=torch.ones_like(h_xx),
        create_graph=True, retain_graph=True,
    )[0]
    _chk(h_xxx, 'h_xxx')

    h_xxxx = torch.autograd.grad(
        outputs=h_xxx, inputs=x, grad_outputs=torch.ones_like(h_xxx),
        create_graph=True, retain_graph=True,
    )[0]
    _chk(h_xxxx, 'h_xxxx')

    h_t = h_t.squeeze(-1)
    h_x = h_x.squeeze(-1)
    h_xx = h_xx.squeeze(-1)
    h_xxxx = h_xxxx.squeeze(-1)

    return h_t, h_x, h_xx, h_xxxx


def pde_residual(
    h: torch.Tensor,
    h_t: torch.Tensor,
    h_x: torch.Tensor,
    h_xx: torch.Tensor,
    h_xxxx: torch.Tensor,
    alpha: float = 100.0 / 16.0,
    beta: float = 100.0 / 16.0 ** 2,
    gamma: float = 100.0 / 16.0 ** 4,
) -> torch.Tensor:
    """
    Compute the PDE residual: h_t + alpha*h*h_x + beta*h_xx + gamma*h_xxxx.

    For the KS equation: h_t + alpha*h*h_x + beta*h_xx + gamma*h_xxxx = 0
    """
    return h_t + alpha * h * h_x + beta * h_xx + gamma * h_xxxx


def build_loss(**cfg) -> Callable:
    """
    Build physics-informed loss function for the KS equation.

    Args:
        **cfg: Configuration dictionary containing:
            - problem: problem name (e.g., 'ks')
            - ks: dict with 'loss_weights', 'alpha', 'beta', 'gamma'
    """
    problem = cfg['problem']
    problem_config = cfg.get(problem, {})
    loss_weights = problem_config['loss_weights']

    weight_residual = loss_weights['residual']
    weight_ic = loss_weights['ic']
    weight_bc = loss_weights['bc']

    alpha = problem_config['alpha']
    beta = problem_config['beta']
    gamma_val = problem_config['gamma']

    # Disable soft BC penalty when periodic Fourier embedding is used — BC is
    # enforced exactly by the embedding so the MSE term is redundant noise.
    use_bc = not cfg['fourier_features']['periodic']

    causal_state = create_causal_state(problem_config)

    def loss_fn(model: nn.Module, batch: Dict[str, torch.Tensor],
                for_tree_spawning: bool = False,
                return_components: bool = False,
                update_causal_state: bool = True):
        """
        Compute physics-informed loss for the KS equation.

        Args:
            model: Neural network model (output_dim=1)
            batch: Dictionary with keys 'x', 't', 'h_gt', 'mask'
            for_tree_spawning: If True, return per-sample loss components dict
            return_components: If True, return dict of unweighted loss components
            update_causal_state: If False, don't update causal state (use during eval)
        """
        x = batch['x']
        t = batch['t']
        h_gt = batch.get('h_gt', batch.get('u_gt'))
        masks = batch['mask']

        N = x.shape[0]
        device = x.device

        _t = getattr(model, '_timer', None)

        if for_tree_spawning:
            residual_per_sample = torch.zeros(N, device=device)
            ic_per_sample = torch.zeros(N, device=device)
            bc_per_sample = torch.zeros(N, device=device)

        # MSE_f: PDE Residual Loss
        if masks['residual'].sum() > 0:
            x_f = x[masks['residual']].contiguous()
            t_f = t[masks['residual']].contiguous()

            x_f = x_f.clone().detach().requires_grad_(True)
            t_f = t_f.clone().detach().requires_grad_(True)

            xt_f = torch.cat([x_f, t_f], dim=1)

            if _t: _t.start('loss.residual.forward')
            h_pred = model(xt_f)
            if _t: _t.stop('loss.residual.forward')

            h_f = h_pred[:, 0]
            _chk(h_f, 'h_f (model output)')

            if _t: _t.start('loss.residual.derivatives')
            h_t_val, h_x_val, h_xx_val, h_xxxx_val = compute_derivatives(h_f, x_f, t_f)
            if _t: _t.stop('loss.residual.derivatives')

            # Check individual PDE terms before combining
            _chk(h_t_val,                       'term: h_t')
            _chk(alpha * h_f * h_x_val,         'term: alpha*h*h_x')
            _chk(beta  * h_xx_val,              'term: beta*h_xx')
            _chk(gamma_val * h_xxxx_val,        'term: gamma*h_xxxx')

            residual = pde_residual(
                h_f, h_t_val, h_x_val, h_xx_val, h_xxxx_val,
                alpha=alpha, beta=beta, gamma=gamma_val)
            _chk(residual, 'residual')

            residual_squared = residual ** 2
            _chk(residual_squared, 'residual_squared')

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
                mse_residual = compute_causal_residual(
                    residual_squared, t_f, causal_state, update_state=update_causal_state)
                _chk(mse_residual, 'mse_residual (after causal weighting)')
        else:
            if not for_tree_spawning:
                mse_residual = torch.tensor(0.0, device=device)

        # MSE_0: Initial Condition Loss  (h(x,0) = cos(x)*(1+sin(x)))
        if masks['IC'].sum() > 0:
            x_0 = x[masks['IC']].contiguous()
            t_0 = t[masks['IC']].contiguous()
            h_gt_0 = h_gt[masks['IC']].contiguous()

            xt_0 = torch.cat([x_0, t_0], dim=1)
            if _t: _t.start('loss.ic.forward')
            h_pred_0 = model(xt_0)
            if _t: _t.stop('loss.ic.forward')

            ic_squared = (h_pred_0 - h_gt_0) ** 2
            _chk(ic_squared, 'ic_squared')

            if for_tree_spawning:
                ic_per_sample[masks['IC']] = ic_squared.squeeze(-1)
            else:
                mse_ic = torch.mean(ic_squared)
        else:
            if not for_tree_spawning:
                mse_ic = torch.tensor(0.0, device=device)

        # MSE_b: Boundary Condition Loss (Periodic)
        # h(0,t) = h(2*pi,t) and h_x(0,t) = h_x(2*pi,t)
        # Skipped when periodic Fourier embedding is active (use_bc=False).
        if use_bc and masks['BC'].sum() > 0:
            x_b = x[masks['BC']].contiguous()
            t_b = t[masks['BC']].contiguous()

            # Separate BC points by x-coordinate (works correctly after shuffle)
            x_min_val = 0.0
            x_max_val = 2.0 * math.pi
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
            h_pred_stacked = model(xt_stacked)
            if _t: _t.stop('loss.bc.forward')

            h_stacked = h_pred_stacked[:, 0]

            if _t: _t.start('loss.bc.derivatives')
            h_x_stacked = torch.autograd.grad(
                h_stacked.sum(), x_stacked,
                create_graph=True, retain_graph=True
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

        if for_tree_spawning:
            return {
                'residual': residual_per_sample,
                'ic': ic_per_sample,
                'bc': bc_per_sample,
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
