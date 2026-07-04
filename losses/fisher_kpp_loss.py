"""
Physics-Informed Loss Function for the Fisher-KPP Equation.

Implements the three-component loss:
    L = w_res*MSE_f + w_ic*MSE_0 + w_bc*MSE_b

where:
- MSE_f: PDE residual loss (h_t - D*h_xx - kappa*h*(1 - h) = 0)
- MSE_0: Initial condition loss (h(x, 0) from dataset)
- MSE_b: Boundary condition loss (h(0, t) = 1, h(1, t) = 0)
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
    D: float = 1.0,
    kappa: float = 25.0,
) -> torch.Tensor:
    """
    Compute the PDE residual: h_t - D*h_xx - kappa*h*(1 - h).
    
    For the Fisher-KPP equation: h_t = D*h_xx + kappa*h*(1 - h)
    
    Args:
        h: Solution field h
        h_t: Time derivative dh/dt
        h_x: Spatial derivative dh/dx (unused, kept for interface consistency)
        h_xx: Second spatial derivative d²h/dx²
        D: Diffusion coefficient (default 1.0)
        kappa: Reaction rate (default 25.0)
        
    Returns:
        Residual tensor
    """
    residual = h_t - D * h_xx - kappa * h * (1.0 - h)
    return residual


def compute_analytical_indicator_derivatives(
    inputs: torch.Tensor,
    indicator_data: dict,
    active_expert_indices,
    need_ht: bool = True,
    need_hxx: bool = True,
) -> dict:
    """
    Compute normalized indicator derivatives ANALYTICALLY for all components.
    
    Uses closed-form sigmoid derivative formulas for soft indicators
    ψ_k = Π_d σ_L(d) · σ_U(d).
    
    For Fisher-KPP (spatial_dim=1): dpsi/dx, dpsi/dt, d2psi/dx2.
    
    First derivative:  ∂ψ_k/∂x_d = ψ_k · f_d
        where f_d = [(1 - σ_L(d)) - (1 - σ_U(d))] / σ_d
    Second derivative: ∂²ψ_k/∂x_d² = ψ_k · {f_d² - [σ_L(1-σ_L) + σ_U(1-σ_U)] / σ_d²}
    
    Normalized (quotient rule for ψ̃_k = ψ_k / Z):
        ψ̃_k_x  = (∂ψ_k/∂x - ψ̃_k · Z_x) / Z
        ψ̃_k_xx = (∂²ψ_k/∂x² - ψ̃_k · Z_xx - 2·ψ̃_k_x · Z_x) / Z
        ψ̃_k_t  = (∂ψ_k/∂t - ψ̃_k · Z_t) / Z
    
    Args:
        inputs: (N, D) original input coordinates
        indicator_data: dict with all_lower, all_upper, all_sigma, psi_base, psi_experts_filtered
        active_expert_indices: tensor of active expert indices
        need_ht: whether to compute time derivatives
        need_hxx: whether to compute second spatial derivatives
    
    Returns:
        dict indexed by component (0=base, 1..K=experts), each with
        'psi_d', 'dpsi_dx', optionally 'dpsi_dt', 'd2psi_dx2'
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

    # Raw first spatial derivative
    f_x = ((1 - sig_L[:, :, x_dim]) - (1 - sig_U[:, :, x_dim])) / all_sigma[:, x_dim].unsqueeze(0)
    dpsi_raw_dx = psi_raw * f_x  # (N, K)

    if need_ht:
        f_t = ((1 - sig_L[:, :, t_dim]) - (1 - sig_U[:, :, t_dim])) / all_sigma[:, t_dim].unsqueeze(0)
        dpsi_raw_dt = psi_raw * f_t  # (N, K)

    if need_hxx:
        sig_L_x = sig_L[:, :, x_dim]
        sig_U_x = sig_U[:, :, x_dim]
        sigma_x = all_sigma[:, x_dim].unsqueeze(0)

        d2psi_raw_dx2 = psi_raw * (
            f_x ** 2 - (sig_L_x * (1 - sig_L_x) + sig_U_x * (1 - sig_U_x)) / sigma_x ** 2
        )  # (N, K)

    # Normalization
    Z = psi_base + psi_experts_filtered.sum(dim=1, keepdim=True)  # (N, 1)
    Z = Z.clamp(min=1e-8)

    filt_mask = (psi_experts_filtered > 0).to(psi_experts_filtered.dtype)  # (N, K)

    Z_x = (dpsi_raw_dx * filt_mask).sum(dim=1, keepdim=True)  # (N, 1)
    if need_ht:
        Z_t = (dpsi_raw_dt * filt_mask).sum(dim=1, keepdim=True)
    if need_hxx:
        Z_xx = (d2psi_raw_dx2 * filt_mask).sum(dim=1, keepdim=True)

    results = {}

    # Component 0: Base model (constant unnormalized psi → zero raw derivatives)
    psi_norm_base = (psi_base / Z).detach()
    psi_norm_base_x = (-psi_norm_base * Z_x / Z).detach()
    results[0] = {
        'psi_d': psi_norm_base,
        'dpsi_dx': psi_norm_base_x,
    }
    if need_ht:
        results[0]['dpsi_dt'] = (-psi_norm_base * Z_t / Z).detach()
    if need_hxx:
        psi_norm_base_xx = (
            -psi_norm_base * Z_xx - 2 * psi_norm_base_x * Z_x
        ) / Z
        results[0]['d2psi_dx2'] = psi_norm_base_xx.detach()

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

        if need_hxx:
            d2psi_k_dx2 = d2psi_raw_dx2[:, eidx:eidx+1] * pt_mask
            psi_norm_k_xx = (
                (d2psi_k_dx2 - psi_norm_k * Z_xx - 2 * psi_norm_k_x * Z_x) / Z
            ).detach()
            result_k['d2psi_dx2'] = psi_norm_k_xx

        results[comp_idx + 1] = result_k

    return results


def compute_derivatives_decomposed(
    components: list,
    x: torch.Tensor,
    t: torch.Tensor,
    need_ht: bool = True,
    need_hxx: bool = True,
    indicator_data: dict = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute derivatives of scalar field h via product rule on decomposed expert outputs,
    using BATCHED autograd for all experts simultaneously.
    
    For composed output h(x,t) = Σ_k ψ̃_k · h_k, the product rule gives:
        h_x  = Σ_k (ψ̃_k_x · h_k  +  ψ̃_k · h_k_x)
        h_t  = Σ_k (ψ̃_k_t · h_k  +  ψ̃_k · h_k_t)
        h_xx = Σ_k (ψ̃_k_xx · h_k  +  2·ψ̃_k_x · h_k_x  +  ψ̃_k · h_k_xx)
    
    Only 2 autograd calls total (scalar field, vs 4 for complex field).
    
    Args:
        components: list of dicts from model.forward_for_pde_derivatives(), each with:
            - 'u': (N, 1) model output, on autograd graph
            - 'inputs': (N, D) per-expert input copy (leaf, requires_grad=True)
            - 'psi_norm': (N, 1) normalized weight, on autograd graph
            - 'constant_psi': bool, True if weight is constant
        x: spatial coordinates (N, spatial_dim), requires_grad=True
        t: temporal coordinates (N, 1), requires_grad=True
        need_ht: compute ∂h/∂t
        need_hxx: compute ∂²h/∂x²
        indicator_data: dict from model with bounds/sigma for analytical derivatives (or None)
        
    Returns:
        Tuple of (h_t, h_x, h_xx) as (N,) tensors.
        h_t is None if need_ht=False, h_xx is None if need_hxx=False.
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
    # BATCHED expert second spatial derivatives (1 more call if needed)
    # ====================================================================
    d2_h = None
    if need_hxx:
        total_dh_dx = sum(
            h_grads[k][:, 0:spatial_dim].sum() for k in range(num_components))

        # Call 2: all second spatial derivatives
        d2_h = torch.autograd.grad(
            total_dh_dx, all_inputs, create_graph=True, retain_graph=True)

    # ====================================================================
    # Indicator derivatives: analytical or autograd fallback
    # ====================================================================
    use_analytical = (indicator_data is not None
                      and indicator_data.get('all_sigma') is not None)

    if use_analytical:
        inputs_orig = torch.cat([x, t], dim=1)  # (N, D)
        active_indices = indicator_data['active_expert_indices']
        psi_derivs = compute_analytical_indicator_derivatives(
            inputs_orig, indicator_data, active_indices,
            need_ht=need_ht, need_hxx=need_hxx)

    # ====================================================================
    # Per-component: product rule assembly
    # ====================================================================
    asm_x = torch.zeros(N, 1, device=device)
    asm_t = torch.zeros(N, 1, device=device) if need_ht else None
    asm_xx = torch.zeros(N, 1, device=device) if need_hxx else None

    for k, c in enumerate(components):
        h_k = c['u'][:, 0:1]     # (N, 1)
        psi_k = c['psi_norm']
        is_constant = c.get('constant_psi', False)

        dh_k_dx = h_grads[k][:, 0:spatial_dim]  # (N, spatial_dim)

        if need_ht:
            dh_k_dt = h_grads[k][:, t_col:t_col+1]  # (N, 1)

        if need_hxx:
            d2h_k_dx2 = d2_h[k][:, 0:spatial_dim]  # (N, spatial_dim)

        # Indicator derivatives
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
                dpsi_dx = torch.autograd.grad(
                    psi_k.sum(), x, create_graph=True, retain_graph=True)[0]
                d2psi_dx2 = torch.autograd.grad(
                    dpsi_dx.sum(), x, retain_graph=True)[0]
                dpsi_dx = dpsi_dx.detach()
                d2psi_dx2 = d2psi_dx2.detach()
            else:
                dpsi_dx = torch.autograd.grad(
                    psi_k.sum(), x, retain_graph=True)[0]
                dpsi_dx = dpsi_dx.detach()

            if need_ht:
                dpsi_dt = torch.autograd.grad(
                    psi_k.sum(), t, retain_graph=True)[0]
                dpsi_dt = dpsi_dt.detach()

            psi_k_d = psi_k.detach()

        # Product rule assembly
        asm_x = asm_x + dpsi_dx * h_k + psi_k_d * dh_k_dx

        if need_ht:
            asm_t = asm_t + dpsi_dt * h_k + psi_k_d * dh_k_dt

        if need_hxx:
            asm_xx = (asm_xx
                      + d2psi_dx2 * h_k
                      + 2 * dpsi_dx * dh_k_dx
                      + psi_k_d * d2h_k_dx2)

    h_x_out = asm_x.squeeze(-1)

    h_t_out = None
    if need_ht:
        h_t_out = asm_t.squeeze(-1)

    h_xx_out = None
    if need_hxx:
        h_xx_out = asm_xx.squeeze(-1)

    return h_t_out, h_x_out, h_xx_out


def build_loss(**cfg) -> Callable:
    """
    Build physics-informed loss function for the Fisher-KPP equation.
    
    Args:
        **cfg: Configuration dictionary containing:
            - problem: problem name (e.g., 'fisher_kpp')
            - fisher_kpp: dict with 'loss_weights' (residual, ic, bc), 'D', and 'kappa'
            
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
    
    # Get Fisher-KPP parameters
    D = problem_config['D']
    kappa = problem_config['kappa']

    # Disable soft BC penalty when periodic Fourier embedding is used — BC is
    # enforced exactly by the embedding so the MSE term is redundant noise.
    use_bc = not cfg['fourier_features']['periodic']

    causal_state = create_causal_state(problem_config)
    
    def loss_fn(model: nn.Module, batch: Dict[str, torch.Tensor],
                for_tree_spawning: bool = False,
                return_components: bool = False,
                update_causal_state: bool = True):
        """
        Compute physics-informed loss for the Fisher-KPP equation.
        
        Args:
            model: Neural network model (output_dim=1 for real-valued h)
            batch: Dictionary with keys:
                - 'x': (N, spatial_dim) spatial coordinates
                - 't': (N, 1) temporal coordinates
                - 'h_gt': (N, 1) ground truth (for IC and BC)
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
            h_t_val, h_x_val, h_xx_val = compute_derivatives(h_f, x_f, t_f)
            if _t: _t.stop('loss.residual.derivatives')
            
            residual = pde_residual(h_f, h_t_val, h_x_val, h_xx_val, D=D, kappa=kappa)
            
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
        # h(x, 0) from dataset
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
        # MSE_b: Boundary Condition Loss
        # h(0, t) = 1, h(1, t) = 0 (Dirichlet)
        # BC values come from h_gt in the dataset
        # Skipped when periodic Fourier embedding is active (use_bc=False).
        # ============================================================
        if use_bc and masks['BC'].sum() > 0:
            x_b = x[masks['BC']].contiguous()
            t_b = t[masks['BC']].contiguous()
            h_gt_b = h_gt[masks['BC']].contiguous()  # (N_b, 1)
            
            xt_b = torch.cat([x_b, t_b], dim=1)
            if _t: _t.start('loss.bc.forward')
            h_pred_b = model(xt_b)  # (N_b, 1)
            if _t: _t.stop('loss.bc.forward')
            
            bc_squared = (h_pred_b - h_gt_b) ** 2
            
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
