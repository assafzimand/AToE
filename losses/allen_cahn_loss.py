"""
Physics-Informed Loss Function for the Allen-Cahn Equation.

Implements the three-component loss:
    L = w_res*MSE_f + w_ic*MSE_0 + w_bc*MSE_b

where:
- MSE_f: PDE residual loss (h_t - D*h_xx - 5*(h - h^3) = 0)
- MSE_0: Initial condition loss (h(x,0) = x^2*cos(pi*x))
- MSE_b: Periodic boundary condition loss (h(-1,t) = h(1,t) and h_x(-1,t) = h_x(1,t))

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
    ones_h = torch.ones_like(h)
    h_grads = torch.autograd.grad(
        outputs=h, inputs=[x, t], grad_outputs=ones_h,
        create_graph=True, retain_graph=True,
    )
    h_x = h_grads[0]
    h_t = h_grads[1]
    ones_hx = torch.ones_like(h_x)
    h_xx = torch.autograd.grad(
        outputs=h_x, inputs=x, grad_outputs=ones_hx,
        create_graph=True, retain_graph=True,
    )[0]
    h_t = h_t.squeeze(-1)
    h_x = h_x.squeeze(-1)
    h_xx = h_xx.squeeze(-1)
    return h_t, h_x, h_xx


def pde_residual(h, h_t, h_x, h_xx, D=0.001):
    residual = h_t - D * h_xx - 5.0 * (h - h**3)
    return residual


def compute_analytical_indicator_derivatives(
    inputs, indicator_data, active_expert_indices,
    need_ht=True, need_hxx=True,
):
    D = inputs.shape[1]
    all_lower = indicator_data['all_lower']
    all_upper = indicator_data['all_upper']
    all_sigma = indicator_data['all_sigma']
    psi_base = indicator_data['psi_base']
    psi_experts_filtered = indicator_data['psi_experts_filtered']
    x_dim = 0
    t_dim = D - 1
    x_inp = inputs.unsqueeze(1)
    lower = all_lower.unsqueeze(0)
    upper = all_upper.unsqueeze(0)
    sigma = all_sigma.unsqueeze(0)
    dist_lower = (x_inp - lower) / sigma
    dist_upper = (upper - x_inp) / sigma
    sig_L = torch.sigmoid(dist_lower)
    sig_U = torch.sigmoid(dist_upper)
    psi_raw = (sig_L * sig_U).prod(dim=2)
    f_x = ((1 - sig_L[:, :, x_dim]) - (1 - sig_U[:, :, x_dim])) / all_sigma[:, x_dim].unsqueeze(0)
    dpsi_raw_dx = psi_raw * f_x
    if need_ht:
        f_t = ((1 - sig_L[:, :, t_dim]) - (1 - sig_U[:, :, t_dim])) / all_sigma[:, t_dim].unsqueeze(0)
        dpsi_raw_dt = psi_raw * f_t
    if need_hxx:
        sig_L_x = sig_L[:, :, x_dim]
        sig_U_x = sig_U[:, :, x_dim]
        sigma_x = all_sigma[:, x_dim].unsqueeze(0)
        d2psi_raw_dx2 = psi_raw * (
            f_x ** 2 - (sig_L_x * (1 - sig_L_x) + sig_U_x * (1 - sig_U_x)) / sigma_x ** 2
        )
    Z = psi_base + psi_experts_filtered.sum(dim=1, keepdim=True)
    Z = Z.clamp(min=1e-8)
    filt_mask = (psi_experts_filtered > 0).to(psi_experts_filtered.dtype)
    Z_x = (dpsi_raw_dx * filt_mask).sum(dim=1, keepdim=True)
    if need_ht:
        Z_t = (dpsi_raw_dt * filt_mask).sum(dim=1, keepdim=True)
    if need_hxx:
        Z_xx = (d2psi_raw_dx2 * filt_mask).sum(dim=1, keepdim=True)
    results = {}
    psi_norm_base = (psi_base / Z).detach()
    psi_norm_base_x = (-psi_norm_base * Z_x / Z).detach()
    results[0] = {'psi_d': psi_norm_base, 'dpsi_dx': psi_norm_base_x}
    if need_ht:
        results[0]['dpsi_dt'] = (-psi_norm_base * Z_t / Z).detach()
    if need_hxx:
        psi_norm_base_xx = (-psi_norm_base * Z_xx - 2 * psi_norm_base_x * Z_x) / Z
        results[0]['d2psi_dx2'] = psi_norm_base_xx.detach()
    for comp_idx, expert_idx in enumerate(active_expert_indices):
        eidx = expert_idx.item() if torch.is_tensor(expert_idx) else expert_idx
        pt_mask = filt_mask[:, eidx:eidx+1]
        dpsi_k_dx = dpsi_raw_dx[:, eidx:eidx+1] * pt_mask
        psi_norm_k = (psi_experts_filtered[:, eidx:eidx+1] / Z).detach()
        psi_norm_k_x = ((dpsi_k_dx - psi_norm_k * Z_x) / Z).detach()
        result_k = {'psi_d': psi_norm_k, 'dpsi_dx': psi_norm_k_x}
        if need_ht:
            dpsi_k_dt = dpsi_raw_dt[:, eidx:eidx+1] * pt_mask
            psi_norm_k_t = ((dpsi_k_dt - psi_norm_k * Z_t) / Z).detach()
            result_k['dpsi_dt'] = psi_norm_k_t
        if need_hxx:
            d2psi_k_dx2 = d2psi_raw_dx2[:, eidx:eidx+1] * pt_mask
            psi_norm_k_xx = ((d2psi_k_dx2 - psi_norm_k * Z_xx - 2 * psi_norm_k_x * Z_x) / Z).detach()
            result_k['d2psi_dx2'] = psi_norm_k_xx
        results[comp_idx + 1] = result_k
    return results


def compute_derivatives_decomposed(
    components, x, t, need_ht=True, need_hxx=True, indicator_data=None,
):
    N = x.shape[0]
    device = x.device
    num_components = len(components)
    spatial_dim = x.shape[1]
    t_col = spatial_dim
    all_inputs = [c['inputs'] for c in components]
    total_h = sum(c['u'][:, 0].sum() for c in components)
    h_grads = torch.autograd.grad(total_h, all_inputs, create_graph=True, retain_graph=True)
    d2_h = None
    if need_hxx:
        total_dh_dx = sum(h_grads[k][:, 0:spatial_dim].sum() for k in range(num_components))
        d2_h = torch.autograd.grad(total_dh_dx, all_inputs, create_graph=True, retain_graph=True)
    # LEGACY PATH DISABLED: analytical indicator derivatives not used with smoothstep windows
    use_analytical = False  # Original: (indicator_data is not None and indicator_data.get('all_sigma') is not None)
    if use_analytical:
        inputs_orig = torch.cat([x, t], dim=1)
        active_indices = indicator_data['active_expert_indices']
        psi_derivs = compute_analytical_indicator_derivatives(
            inputs_orig, indicator_data, active_indices, need_ht=need_ht, need_hxx=need_hxx)
    asm_x = torch.zeros(N, 1, device=device)
    asm_t = torch.zeros(N, 1, device=device) if need_ht else None
    asm_xx = torch.zeros(N, 1, device=device) if need_hxx else None
    for k, c in enumerate(components):
        h_k = c['u'][:, 0:1]
        psi_k = c['psi_norm']
        is_constant = c.get('constant_psi', False)
        dh_k_dx = h_grads[k][:, 0:spatial_dim]
        if need_ht:
            dh_k_dt = h_grads[k][:, t_col:t_col+1]
        if need_hxx:
            d2h_k_dx2 = d2_h[k][:, 0:spatial_dim]
        if use_analytical:
            psi_info = psi_derivs[k]
            psi_k_d = psi_info['psi_d']
            dpsi_dx = psi_info['dpsi_dx']
            if need_ht:
                dpsi_dt = psi_info['dpsi_dt']
            if need_hxx:
                d2psi_dx2 = psi_info['d2psi_dx2']
        elif is_constant:
            psi_k_d = psi_k.detach()
            dpsi_dx = torch.zeros(N, 1, device=device)
            if need_hxx:
                d2psi_dx2 = torch.zeros(N, 1, device=device)
            if need_ht:
                dpsi_dt = torch.zeros(N, 1, device=device)
        else:
            if need_hxx:
                dpsi_dx = torch.autograd.grad(psi_k.sum(), x, create_graph=True, retain_graph=True)[0]
                d2psi_dx2 = torch.autograd.grad(dpsi_dx.sum(), x, retain_graph=True)[0]
                dpsi_dx = dpsi_dx.detach()
                d2psi_dx2 = d2psi_dx2.detach()
            else:
                dpsi_dx = torch.autograd.grad(psi_k.sum(), x, retain_graph=True)[0]
                dpsi_dx = dpsi_dx.detach()
            if need_ht:
                dpsi_dt = torch.autograd.grad(psi_k.sum(), t, retain_graph=True)[0]
                dpsi_dt = dpsi_dt.detach()
            psi_k_d = psi_k.detach()
        asm_x = asm_x + dpsi_dx * h_k + psi_k_d * dh_k_dx
        if need_ht:
            asm_t = asm_t + dpsi_dt * h_k + psi_k_d * dh_k_dt
        if need_hxx:
            asm_xx = asm_xx + d2psi_dx2 * h_k + 2 * dpsi_dx * dh_k_dx + psi_k_d * d2h_k_dx2
    h_x_out = asm_x.squeeze(-1)
    h_t_out = asm_t.squeeze(-1) if need_ht else None
    h_xx_out = asm_xx.squeeze(-1) if need_hxx else None
    return h_t_out, h_x_out, h_xx_out


def build_loss(**cfg):
    problem = cfg['problem']
    problem_config = cfg[problem]
    loss_weights = problem_config['loss_weights']
    weight_residual = loss_weights['residual']
    weight_ic = loss_weights['ic']
    weight_bc = loss_weights['bc']
    D = problem_config['D']

    # Disable soft BC penalty when periodic Fourier embedding is used — BC is
    # enforced exactly by the embedding so the MSE term is redundant noise.
    use_bc = not cfg['fourier_features']['periodic']

    causal_state = create_causal_state(problem_config)

    def loss_fn(model, batch, for_tree_spawning=False, return_components=False, update_causal_state=True):
        x = batch['x']; t = batch['t']
        h_gt = batch.get('h_gt', batch.get('u_gt'))
        masks = batch['mask']
        N = x.shape[0]; device = x.device
        _t = getattr(model, '_timer', None)
        if for_tree_spawning:
            residual_per_sample = torch.zeros(N, device=device)
            ic_per_sample = torch.zeros(N, device=device)
            bc_per_sample = torch.zeros(N, device=device)
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
            if _t: _t.start('loss.residual.derivatives')
            h_t_val, h_x_val, h_xx_val = compute_derivatives(h_f, x_f, t_f)
            if _t: _t.stop('loss.residual.derivatives')
            residual = pde_residual(h_f, h_t_val, h_x_val, h_xx_val, D=D)
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
        if masks['IC'].sum() > 0:
            x_0 = x[masks['IC']].contiguous()
            t_0 = t[masks['IC']].contiguous()
            h_gt_0 = h_gt[masks['IC']].contiguous()
            xt_0 = torch.cat([x_0, t_0], dim=1)
            if _t: _t.start('loss.ic.forward')
            h_pred_0 = model(xt_0)
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
            x_b = x[masks['BC']].contiguous()
            t_b = t[masks['BC']].contiguous()

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
            h_pred_stacked = model(xt_stacked)
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
        if for_tree_spawning:
            return {'residual': residual_per_sample, 'ic': ic_per_sample, 'bc': bc_per_sample}
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
