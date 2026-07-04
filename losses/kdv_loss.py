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


def compute_analytical_indicator_derivatives(
    inputs: torch.Tensor,
    indicator_data: dict,
    active_expert_indices,
    need_ht: bool = True,
    need_hxxx: bool = True,
) -> dict:
    """
    [LEGACY - NOT USED WITH SMOOTHSTEP WINDOWS]
    
    Compute normalized indicator derivatives ANALYTICALLY for all components.
    
    This function was designed for the legacy sigmoid window where analytical
    derivatives could be computed. With the new compact smoothstep windows,
    all derivatives are computed via autograd on the composed forward output.
    This function is preserved for reference but will not be called.
    
    Extended from Burgers 1D to include the third spatial derivative needed by KdV.
    
    For the raw indicator psi_k = prod_d sigma_L(d) * sigma_U(d):
    
    First derivative:
        f_x = [(1 - sig_L_x) - (1 - sig_U_x)] / sigma_x
        dpsi/dx = psi * f_x
    
    Second derivative:
        f'_x = -[sig_L_x*(1-sig_L_x) + sig_U_x*(1-sig_U_x)] / sigma_x^2
        d2psi/dx2 = psi * (f_x^2 + f'_x)
    
    Third derivative:
        f''_x = -[sig_L_x*(1-sig_L_x)*(1-2*sig_L_x) - sig_U_x*(1-sig_U_x)*(1-2*sig_U_x)] / sigma_x^3
        d3psi/dx3 = psi * (f_x^3 + 3*f_x*f'_x + f''_x)
    
    Normalized (quotient rule for psi_tilde_k = psi_k / Z):
        psi_tilde_x   = (dpsi_dx - psi_tilde * Z_x) / Z
        psi_tilde_xx  = (d2psi_dx2 - psi_tilde * Z_xx - 2*psi_tilde_x * Z_x) / Z
        psi_tilde_xxx = (d3psi_dx3 - psi_tilde * Z_xxx - 3*psi_tilde_xx * Z_x - 3*psi_tilde_x * Z_xx) / Z
    
    Args:
        inputs: (N, D) original input coordinates
        indicator_data: dict with all_lower, all_upper, all_sigma, psi_base, psi_experts_filtered
        active_expert_indices: tensor of active expert indices
        need_ht: whether to compute time derivatives
        need_hxxx: whether to compute third spatial derivatives
    
    Returns:
        dict indexed by component (0=base, 1..K=experts), each with
        'psi_d', 'dpsi_dx', optionally 'dpsi_dt', 'd2psi_dx2', 'd3psi_dx3'
    """
    D = inputs.shape[1]

    all_lower = indicator_data['all_lower']       # (K, D)
    all_upper = indicator_data['all_upper']       # (K, D)
    all_sigma = indicator_data['all_sigma']       # (K, D)
    psi_base = indicator_data['psi_base']         # (N, 1)
    psi_experts_filtered = indicator_data['psi_experts_filtered']  # (N, K)

    x_dim = 0
    t_dim = D - 1

    # Recompute sigmoid intermediates for all K experts (vectorized)
    x_inp = inputs.unsqueeze(1)               # (N, 1, D)
    lower = all_lower.unsqueeze(0)            # (1, K, D)
    upper = all_upper.unsqueeze(0)            # (1, K, D)
    sigma = all_sigma.unsqueeze(0)            # (1, K, D)

    dist_lower = (x_inp - lower) / sigma      # (N, K, D)
    dist_upper = (upper - x_inp) / sigma      # (N, K, D)

    sig_L = torch.sigmoid(dist_lower)         # (N, K, D)
    sig_U = torch.sigmoid(dist_upper)         # (N, K, D)

    psi_raw = (sig_L * sig_U).prod(dim=2)     # (N, K)

    # Spatial dimension intermediates
    sig_L_x = sig_L[:, :, x_dim]  # (N, K)
    sig_U_x = sig_U[:, :, x_dim]  # (N, K)
    sigma_x = all_sigma[:, x_dim].unsqueeze(0)  # (1, K)

    # Raw first spatial derivative: f_x and dpsi/dx
    f_x = ((1 - sig_L_x) - (1 - sig_U_x)) / sigma_x
    dpsi_raw_dx = psi_raw * f_x  # (N, K)

    if need_ht:
        f_t = ((1 - sig_L[:, :, t_dim]) - (1 - sig_U[:, :, t_dim])) / all_sigma[:, t_dim].unsqueeze(0)
        dpsi_raw_dt = psi_raw * f_t  # (N, K)

    # Second derivative is always needed (used in h_xxx decomposition's product rule)
    f_prime_x = -(sig_L_x * (1 - sig_L_x) + sig_U_x * (1 - sig_U_x)) / sigma_x ** 2
    d2psi_raw_dx2 = psi_raw * (f_x ** 2 + f_prime_x)  # (N, K)

    # Third spatial derivative
    if need_hxxx:
        f_double_prime_x = -(
            sig_L_x * (1 - sig_L_x) * (1 - 2 * sig_L_x)
            - sig_U_x * (1 - sig_U_x) * (1 - 2 * sig_U_x)
        ) / sigma_x ** 3
        d3psi_raw_dx3 = psi_raw * (f_x ** 3 + 3 * f_x * f_prime_x + f_double_prime_x)  # (N, K)

    # Normalization: Z and its derivatives
    Z = psi_base + psi_experts_filtered.sum(dim=1, keepdim=True)  # (N, 1)
    Z = Z.clamp(min=1e-8)

    filt_mask = (psi_experts_filtered > 0).to(psi_experts_filtered.dtype)  # (N, K)

    Z_x = (dpsi_raw_dx * filt_mask).sum(dim=1, keepdim=True)    # (N, 1)
    Z_xx = (d2psi_raw_dx2 * filt_mask).sum(dim=1, keepdim=True)  # (N, 1)

    if need_ht:
        Z_t = (dpsi_raw_dt * filt_mask).sum(dim=1, keepdim=True)
    if need_hxxx:
        Z_xxx = (d3psi_raw_dx3 * filt_mask).sum(dim=1, keepdim=True)  # (N, 1)

    results = {}

    # Component 0: Base model (constant unnormalized psi -> zero raw derivatives)
    psi_norm_base = (psi_base / Z).detach()
    psi_norm_base_x = (-psi_norm_base * Z_x / Z).detach()
    results[0] = {
        'psi_d': psi_norm_base,
        'dpsi_dx': psi_norm_base_x,
    }
    if need_ht:
        results[0]['dpsi_dt'] = (-psi_norm_base * Z_t / Z).detach()

    # Base second derivative (always computed)
    psi_norm_base_xx = (
        -psi_norm_base * Z_xx - 2 * psi_norm_base_x * Z_x
    ) / Z
    results[0]['d2psi_dx2'] = psi_norm_base_xx.detach()

    if need_hxxx:
        psi_norm_base_xxx = (
            -psi_norm_base * Z_xxx - 3 * psi_norm_base_xx * Z_x - 3 * psi_norm_base_x * Z_xx
        ) / Z
        results[0]['d3psi_dx3'] = psi_norm_base_xxx.detach()

    # Components 1..num_active: active experts
    for comp_idx, expert_idx in enumerate(active_expert_indices):
        eidx = expert_idx.item() if torch.is_tensor(expert_idx) else expert_idx

        pt_mask = filt_mask[:, eidx:eidx+1]  # (N, 1)

        dpsi_k_dx = dpsi_raw_dx[:, eidx:eidx+1] * pt_mask
        psi_norm_k = (psi_experts_filtered[:, eidx:eidx+1] / Z).detach()

        psi_norm_k_x = ((dpsi_k_dx - psi_norm_k * Z_x) / Z).detach()

        result_k = {
            'psi_d': psi_norm_k,
            'dpsi_dx': psi_norm_k_x,
        }

        if need_ht:
            dpsi_k_dt = dpsi_raw_dt[:, eidx:eidx+1] * pt_mask
            psi_norm_k_t = ((dpsi_k_dt - psi_norm_k * Z_t) / Z).detach()
            result_k['dpsi_dt'] = psi_norm_k_t

        # Second derivative (always computed for h_xxx product rule)
        d2psi_k_dx2 = d2psi_raw_dx2[:, eidx:eidx+1] * pt_mask
        psi_norm_k_xx = (
            (d2psi_k_dx2 - psi_norm_k * Z_xx - 2 * psi_norm_k_x * Z_x) / Z
        ).detach()
        result_k['d2psi_dx2'] = psi_norm_k_xx

        if need_hxxx:
            d3psi_k_dx3 = d3psi_raw_dx3[:, eidx:eidx+1] * pt_mask
            psi_norm_k_xxx = (
                (d3psi_k_dx3 - psi_norm_k * Z_xxx - 3 * psi_norm_k_xx * Z_x - 3 * psi_norm_k_x * Z_xx) / Z
            ).detach()
            result_k['d3psi_dx3'] = psi_norm_k_xxx

        results[comp_idx + 1] = result_k

    return results


def compute_derivatives_decomposed(
    components: list,
    x: torch.Tensor,
    t: torch.Tensor,
    need_ht: bool = True,
    need_hxxx: bool = True,
    indicator_data: dict = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute derivatives of scalar field h via product rule on decomposed expert outputs,
    using BATCHED autograd for all experts simultaneously.
    
    For composed output h(x,t) = sum_k psi_tilde_k * h_k, the product rule gives:
        h_x   = sum_k (psi_tilde_k_x * h_k + psi_tilde_k * h_k_x)
        h_t   = sum_k (psi_tilde_k_t * h_k + psi_tilde_k * h_k_t)
        h_xxx = sum_k (psi_tilde_k_xxx * h_k + 3*psi_tilde_k_xx * h_k_x
                        + 3*psi_tilde_k_x * h_k_xx + psi_tilde_k * h_k_xxx)
    
    Three autograd calls total for the expert outputs (first, second, third spatial derivatives).
    
    Args:
        components: list of dicts from model.forward_for_pde_derivatives(), each with:
            - 'u': (N, 1) model output, on autograd graph
            - 'inputs': (N, D) per-expert input copy (leaf, requires_grad=True)
            - 'psi_norm': (N, 1) normalized weight, on autograd graph
            - 'constant_psi': bool, True if weight is constant
        x: spatial coordinates (N, spatial_dim), requires_grad=True
        t: temporal coordinates (N, 1), requires_grad=True
        need_ht: compute dh/dt
        need_hxxx: compute d³h/dx³
        indicator_data: dict from model with bounds/sigma for analytical derivatives (or None)
        
    Returns:
        Tuple of (h_t, h_x, h_xxx) as (N,) tensors.
        h_t is None if need_ht=False, h_xxx is None if need_hxxx=False.
    """
    N = x.shape[0]
    device = x.device
    num_components = len(components)
    spatial_dim = x.shape[1]
    t_col = spatial_dim

    # ====================================================================
    # BATCHED expert first derivatives (1 autograd call for ALL experts)
    # ====================================================================
    all_inputs = [c['inputs'] for c in components]

    total_h = sum(c['u'][:, 0].sum() for c in components)

    # Call 1: all first derivatives in one shot
    h_grads = torch.autograd.grad(
        total_h, all_inputs, create_graph=True, retain_graph=True)

    # ====================================================================
    # BATCHED expert second spatial derivatives (always needed for h_xxx product rule)
    # ====================================================================
    total_dh_dx = sum(
        h_grads[k][:, 0:spatial_dim].sum() for k in range(num_components))

    # Call 2: all second spatial derivatives
    d2_h = torch.autograd.grad(
        total_dh_dx, all_inputs, create_graph=True, retain_graph=True)

    # ====================================================================
    # BATCHED expert third spatial derivatives (if needed)
    # ====================================================================
    d3_h = None
    if need_hxxx:
        total_d2h_dx2 = sum(
            d2_h[k][:, 0:spatial_dim].sum() for k in range(num_components))

        # Call 3: all third spatial derivatives
        d3_h = torch.autograd.grad(
            total_d2h_dx2, all_inputs, create_graph=True, retain_graph=True)

    # ====================================================================
    # Indicator derivatives: autograd only (analytical path is LEGACY)
    # ====================================================================
    # LEGACY: The analytical path checked for 'all_sigma' which was specific to
    # sigmoid windows. With smoothstep windows, this key doesn't exist (renamed
    # to all_delta with different semantics), so use_analytical is always False.
    # All indicator derivatives are now computed via autograd on composed output.
    use_analytical = False  # LEGACY PATH DISABLED
    # Original check was: (indicator_data is not None and indicator_data.get('all_sigma') is not None)

    if use_analytical:
        # LEGACY: This branch is never taken with smoothstep windows
        inputs_orig = torch.cat([x, t], dim=1)  # (N, D)
        active_indices = indicator_data['active_expert_indices']
        psi_derivs = compute_analytical_indicator_derivatives(
            inputs_orig, indicator_data, active_indices,
            need_ht=need_ht, need_hxxx=need_hxxx)

    # ====================================================================
    # Per-component: product rule assembly
    # ====================================================================
    asm_x = torch.zeros(N, 1, device=device)
    asm_t = torch.zeros(N, 1, device=device) if need_ht else None
    asm_xxx = torch.zeros(N, 1, device=device) if need_hxxx else None

    for k, c in enumerate(components):
        h_k = c['u'][:, 0:1]     # (N, 1)
        psi_k = c['psi_norm']
        is_constant = c.get('constant_psi', False)

        dh_k_dx = h_grads[k][:, 0:spatial_dim]   # (N, spatial_dim)
        d2h_k_dx2 = d2_h[k][:, 0:spatial_dim]    # (N, spatial_dim)

        if need_ht:
            dh_k_dt = h_grads[k][:, t_col:t_col+1]  # (N, 1)

        if need_hxxx:
            d3h_k_dx3 = d3_h[k][:, 0:spatial_dim]  # (N, spatial_dim)

        # Indicator derivatives
        if use_analytical:
            psi_info = psi_derivs[k]
            psi_k_d = psi_info['psi_d']
            dpsi_dx = psi_info['dpsi_dx']
            d2psi_dx2 = psi_info['d2psi_dx2']
            if need_ht:
                dpsi_dt = psi_info['dpsi_dt']
            if need_hxxx:
                d3psi_dx3 = psi_info['d3psi_dx3']
        elif is_constant:
            psi_k_d = psi_k.detach()
            dpsi_dx = torch.zeros(N, 1, device=device)
            d2psi_dx2 = torch.zeros(N, 1, device=device)
            if need_hxxx:
                d3psi_dx3 = torch.zeros(N, 1, device=device)
            if need_ht:
                dpsi_dt = torch.zeros(N, 1, device=device)
        else:
            # Autograd fallback for indicator derivatives
            dpsi_dx = torch.autograd.grad(
                psi_k.sum(), x, create_graph=True, retain_graph=True)[0]
            d2psi_dx2 = torch.autograd.grad(
                dpsi_dx.sum(), x, create_graph=True, retain_graph=True)[0]

            if need_hxxx:
                d3psi_dx3 = torch.autograd.grad(
                    d2psi_dx2.sum(), x, retain_graph=True)[0]
                d3psi_dx3 = d3psi_dx3.detach()

            dpsi_dx = dpsi_dx.detach()
            d2psi_dx2 = d2psi_dx2.detach()

            if need_ht:
                dpsi_dt = torch.autograd.grad(
                    psi_k.sum(), t, retain_graph=True)[0]
                dpsi_dt = dpsi_dt.detach()

            psi_k_d = psi_k.detach()

        # Product rule assembly: h_x
        asm_x = asm_x + dpsi_dx * h_k + psi_k_d * dh_k_dx

        # Product rule assembly: h_t
        if need_ht:
            asm_t = asm_t + dpsi_dt * h_k + psi_k_d * dh_k_dt

        # Product rule assembly: h_xxx (general Leibniz rule, 4th order binomial)
        if need_hxxx:
            asm_xxx = (asm_xxx
                       + d3psi_dx3 * h_k
                       + 3 * d2psi_dx2 * dh_k_dx
                       + 3 * dpsi_dx * d2h_k_dx2
                       + psi_k_d * d3h_k_dx3)

    h_x_out = asm_x.squeeze(-1)

    h_t_out = None
    if need_ht:
        h_t_out = asm_t.squeeze(-1)

    h_xxx_out = None
    if need_hxxx:
        h_xxx_out = asm_xxx.squeeze(-1)

    return h_t_out, h_x_out, h_xxx_out


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
