"""
Physics-Informed Loss Function for the 1D Wave Equation.

Implements the three-component loss:
    L = w_res*MSE_f + w_ic*MSE_0 + w_bc*MSE_b

where:
- MSE_f: PDE residual loss (h_tt - h_xx = 0)
- MSE_0: Initial condition loss (h(x,0) = sin(x), h_t(x,0) = 0)
- MSE_b:  MSE_b: Boundary condition loss using the analytical boundary values from the
         dataset (not forced to zero). This keeps the loss consistent with the
         ground-truth solution and avoids injecting an inconsistent zero BC
         when the analytical solution is non-zero at the domain edges.
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
    
    Computes ∂h/∂t, ∂²h/∂t², ∂h/∂x, and ∂²h/∂x² using PyTorch autograd.
    All operations stay on the same device (GPU if available).
    
    Args:
        h: Scalar field, shape (batch_size,)
        x: Spatial coordinates, shape (batch_size, 1), requires_grad=True
        t: Temporal coordinates, shape (batch_size, 1), requires_grad=True
        
    Returns:
        Tuple of (h_t, h_tt, h_x, h_xx):
        - h_t: ∂h/∂t
        - h_tt: ∂²h/∂t²
        - h_x: ∂h/∂x
        - h_xx: ∂²h/∂x²
    """
    
    # Create grad_outputs once
    ones_h = torch.ones_like(h)
    
    # Call 1: Compute h derivatives w.r.t. BOTH x and t in one call
    h_grads = torch.autograd.grad(
        outputs=h,
        inputs=[x, t],
        grad_outputs=ones_h,
        create_graph=True,
        retain_graph=True,
    )
    h_x = h_grads[0]
    h_t = h_grads[1]
    
    # Second derivatives
    ones_hx = torch.ones_like(h_x)
    ones_ht = torch.ones_like(h_t)
    
    # Call 2: h_xx
    h_xx = torch.autograd.grad(
        outputs=h_x,
        inputs=x,
        grad_outputs=ones_hx,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Call 3: h_tt
    h_tt = torch.autograd.grad(
        outputs=h_t,
        inputs=t,
        grad_outputs=ones_ht,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Squeeze to remove trailing dimensions
    h_t = h_t.squeeze(-1)
    h_tt = h_tt.squeeze(-1)
    h_x = h_x.squeeze(-1)
    h_xx = h_xx.squeeze(-1)
    
    return h_t, h_tt, h_x, h_xx


def compute_analytical_indicator_derivatives(
    inputs: torch.Tensor,
    indicator_data: dict,
    active_expert_indices,
    need_htt: bool = True,
    need_hxx: bool = True,
) -> dict:
    """
    Compute normalized indicator derivatives ANALYTICALLY for all components.
    
    Same sigmoid math as Schrodinger/Burgers, but with the ADDITION of d²ψ/dt²
    needed for the wave equation's second time derivative term.
    
    Raw indicator: ψ_k = Π_d σ_L(d) · σ_U(d)
        f_d = [(1-σ_L(d)) - (1-σ_U(d))] / σ_d
        ∂ψ_k/∂x_d = ψ_k · f_d
        ∂²ψ_k/∂x_d² = ψ_k · [f_d² - (σ_L(1-σ_L) + σ_U(1-σ_U)) / σ_d²]
    
    Normalized (quotient rule for ψ̃_k = ψ_k / Z):
        ψ̃_k_x  = (∂ψ_k/∂x - ψ̃_k · Z_x) / Z
        ψ̃_k_xx = (∂²ψ_k/∂x² - ψ̃_k · Z_xx - 2·ψ̃_k_x · Z_x) / Z
        ψ̃_k_t  = (∂ψ_k/∂t - ψ̃_k · Z_t) / Z
        ψ̃_k_tt = (∂²ψ_k/∂t² - ψ̃_k · Z_tt - 2·ψ̃_k_t · Z_t) / Z
    
    Args:
        inputs: (N, D) original input coordinates
        indicator_data: dict from model with all_lower, all_upper, all_sigma, etc.
        active_expert_indices: tensor of active expert indices
        need_htt: whether to compute second time derivatives (d2psi_dt2)
        need_hxx: whether to compute second spatial derivatives (d2psi_dx2)
    
    Returns:
        dict with keys for each component index (0=base, 1..K=experts):
            'psi_d': (N, 1), 'dpsi_dx': (N, 1), 'dpsi_dt': (N, 1),
            'd2psi_dx2': (N, 1) if need_hxx, 'd2psi_dt2': (N, 1) if need_htt
            All detached (no grad).
    """
    D = inputs.shape[1]

    all_lower = indicator_data['all_lower']   # (K, D)
    all_upper = indicator_data['all_upper']   # (K, D)
    all_sigma = indicator_data['all_sigma']   # (K, D)
    psi_base = indicator_data['psi_base']     # (N, 1)
    psi_experts_filtered = indicator_data['psi_experts_filtered']  # (N, K)

    x_dim = 0       # spatial derivative dimension
    t_dim = D - 1   # time dimension (last column)

    # ---- Recompute sigmoid intermediates for ALL K experts (vectorized) ----
    x_inp = inputs.unsqueeze(1)               # (N, 1, D)
    lower = all_lower.unsqueeze(0)            # (1, K, D)
    upper = all_upper.unsqueeze(0)            # (1, K, D)
    sigma = all_sigma.unsqueeze(0)            # (1, K, D)

    dist_lower = (x_inp - lower) / sigma      # (N, K, D)
    dist_upper = (upper - x_inp) / sigma      # (N, K, D)

    sig_L = torch.sigmoid(dist_lower)         # (N, K, D)
    sig_U = torch.sigmoid(dist_upper)         # (N, K, D)

    psi_raw = (sig_L * sig_U).prod(dim=2)     # (N, K)

    # ---- Raw indicator first derivatives (always needed) ----
    # Spatial: f_x
    f_x = ((1 - sig_L[:, :, x_dim]) - (1 - sig_U[:, :, x_dim])) / all_sigma[:, x_dim].unsqueeze(0)  # (N, K)
    dpsi_raw_dx = psi_raw * f_x  # (N, K)

    # Temporal: f_t
    f_t = ((1 - sig_L[:, :, t_dim]) - (1 - sig_U[:, :, t_dim])) / all_sigma[:, t_dim].unsqueeze(0)  # (N, K)
    dpsi_raw_dt = psi_raw * f_t  # (N, K)

    # ---- Raw indicator second spatial derivatives ----
    if need_hxx:
        sig_L_x = sig_L[:, :, x_dim]  # (N, K)
        sig_U_x = sig_U[:, :, x_dim]  # (N, K)
        sigma_x = all_sigma[:, x_dim].unsqueeze(0)  # (1, K)

        d2psi_raw_dx2 = psi_raw * (
            f_x ** 2 - (sig_L_x * (1 - sig_L_x) + sig_U_x * (1 - sig_U_x)) / sigma_x ** 2
        )  # (N, K)

    # ---- Raw indicator second time derivatives (NEW for wave equation) ----
    if need_htt:
        sig_L_t = sig_L[:, :, t_dim]  # (N, K)
        sig_U_t = sig_U[:, :, t_dim]  # (N, K)
        sigma_t = all_sigma[:, t_dim].unsqueeze(0)  # (1, K)

        d2psi_raw_dt2 = psi_raw * (
            f_t ** 2 - (sig_L_t * (1 - sig_L_t) + sig_U_t * (1 - sig_U_t)) / sigma_t ** 2
        )  # (N, K)

    # ---- Normalization and quotient rule ----
    Z = psi_base + psi_experts_filtered.sum(dim=1, keepdim=True)  # (N, 1)
    Z = Z.clamp(min=1e-8)

    filt_mask = (psi_experts_filtered > 0).to(psi_experts_filtered.dtype)  # (N, K)

    Z_x = (dpsi_raw_dx * filt_mask).sum(dim=1, keepdim=True)   # (N, 1)
    Z_t = (dpsi_raw_dt * filt_mask).sum(dim=1, keepdim=True)   # (N, 1)
    if need_hxx:
        Z_xx = (d2psi_raw_dx2 * filt_mask).sum(dim=1, keepdim=True)  # (N, 1)
    if need_htt:
        Z_tt = (d2psi_raw_dt2 * filt_mask).sum(dim=1, keepdim=True)  # (N, 1)

    # ---- Build per-component results ----
    results = {}

    # Component 0: Base model (psi_base is constant → raw derivatives are 0)
    psi_norm_base = (psi_base / Z).detach()  # (N, 1)
    psi_norm_base_x = (-psi_norm_base * Z_x / Z).detach()
    psi_norm_base_t = (-psi_norm_base * Z_t / Z).detach()
    results[0] = {
        'psi_d': psi_norm_base,
        'dpsi_dx': psi_norm_base_x,
        'dpsi_dt': psi_norm_base_t,
    }
    if need_hxx:
        psi_norm_base_xx = (
            -psi_norm_base * Z_xx - 2 * psi_norm_base_x * Z_x
        ) / Z
        results[0]['d2psi_dx2'] = psi_norm_base_xx.detach()
    if need_htt:
        psi_norm_base_tt = (
            -psi_norm_base * Z_tt - 2 * psi_norm_base_t * Z_t
        ) / Z
        results[0]['d2psi_dt2'] = psi_norm_base_tt.detach()

    # Components 1..num_active: active experts
    for comp_idx, expert_idx in enumerate(active_expert_indices):
        eidx = expert_idx.item() if torch.is_tensor(expert_idx) else expert_idx

        pt_mask = filt_mask[:, eidx:eidx+1]  # (N, 1)

        dpsi_k_dx = dpsi_raw_dx[:, eidx:eidx+1] * pt_mask  # (N, 1)
        dpsi_k_dt = dpsi_raw_dt[:, eidx:eidx+1] * pt_mask  # (N, 1)

        psi_norm_k = (psi_experts_filtered[:, eidx:eidx+1] / Z).detach()

        psi_norm_k_x = ((dpsi_k_dx - psi_norm_k * Z_x) / Z).detach()
        psi_norm_k_t = ((dpsi_k_dt - psi_norm_k * Z_t) / Z).detach()

        result_k = {
            'psi_d': psi_norm_k,
            'dpsi_dx': psi_norm_k_x,
            'dpsi_dt': psi_norm_k_t,
        }

        if need_hxx:
            d2psi_k_dx2 = d2psi_raw_dx2[:, eidx:eidx+1] * pt_mask
            psi_norm_k_xx = (
                (d2psi_k_dx2 - psi_norm_k * Z_xx - 2 * psi_norm_k_x * Z_x) / Z
            ).detach()
            result_k['d2psi_dx2'] = psi_norm_k_xx

        if need_htt:
            d2psi_k_dt2 = d2psi_raw_dt2[:, eidx:eidx+1] * pt_mask
            psi_norm_k_tt = (
                (d2psi_k_dt2 - psi_norm_k * Z_tt - 2 * psi_norm_k_t * Z_t) / Z
            ).detach()
            result_k['d2psi_dt2'] = psi_norm_k_tt

        results[comp_idx + 1] = result_k

    return results


def compute_derivatives_decomposed(
    components: list,
    x: torch.Tensor,
    t: torch.Tensor,
    need_htt: bool = True,
    need_hxx: bool = True,
    indicator_data: dict = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute derivatives of scalar field h via product rule on decomposed expert
    outputs, using BATCHED autograd for all experts simultaneously.
    
    For composed output h(x,t) = Σ_k ψ̃_k · h_k, the product rule gives:
        h_t  = Σ_k (ψ̃_k_t  · h_k + ψ̃_k · h_k_t)
        h_tt = Σ_k (ψ̃_k_tt · h_k + 2·ψ̃_k_t · h_k_t + ψ̃_k · h_k_tt)
        h_x  = Σ_k (ψ̃_k_x  · h_k + ψ̃_k · h_k_x)
        h_xx = Σ_k (ψ̃_k_xx · h_k + 2·ψ̃_k_x · h_k_x + ψ̃_k · h_k_xx)
    
    Batched autograd (3 calls for scalar field):
        Call 1: grad(total_h_sum, all_inputs) → dh_k/dx, dh_k/dt
        Call 2 (if need_hxx): grad(sum_dh_dx, all_inputs) → d2h_k/dx2
        Call 3 (if need_htt): grad(sum_dh_dt, all_inputs) → d2h_k/dt2
    
    Args:
        components: list of dicts from model.forward_for_pde_derivatives(), each with:
            - 'u': (N, 1) model output, on autograd graph
            - 'inputs': (N, D) per-expert input copy (leaf, requires_grad=True)
            - 'psi_norm': (N, 1) normalized weight, on autograd graph
            - 'constant_psi': bool, True if weight is constant
        x: spatial coordinates (N, spatial_dim), requires_grad=True
        t: temporal coordinates (N, 1), requires_grad=True
        need_htt: compute ∂²h/∂t² (needed for wave PDE residual)
        need_hxx: compute ∂²h/∂x² (needed for wave PDE residual)
        indicator_data: dict from model with bounds/sigma for analytical derivatives
        
    Returns:
        Tuple of (h_t, h_tt, h_x, h_xx) as (N,) tensors.
        h_tt is None if need_htt=False, h_xx is None if need_hxx=False.
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

    total_h_sum = sum(c['u'][:, 0].sum() for c in components)

    # Call 1: all first derivatives in one shot
    h_grads = torch.autograd.grad(
        total_h_sum, all_inputs, create_graph=True, retain_graph=True)

    # ====================================================================
    # BATCHED expert second derivatives (up to 2 more calls)
    # ====================================================================
    d2h_x = None
    if need_hxx:
        total_dh_dx = sum(
            h_grads[k][:, 0:spatial_dim].sum() for k in range(num_components))
        # Call 2: all second spatial derivatives
        d2h_x = torch.autograd.grad(
            total_dh_dx, all_inputs, create_graph=True, retain_graph=True)

    d2h_t = None
    if need_htt:
        total_dh_dt = sum(
            h_grads[k][:, t_col:t_col+1].sum() for k in range(num_components))
        # Call 3: all second time derivatives
        d2h_t = torch.autograd.grad(
            total_dh_dt, all_inputs, create_graph=True, retain_graph=True)

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
            need_htt=need_htt, need_hxx=need_hxx)

    # ====================================================================
    # Per-component: product rule assembly
    # ====================================================================
    asm_x = torch.zeros(N, 1, device=device)
    asm_t = torch.zeros(N, 1, device=device)
    asm_xx = torch.zeros(N, 1, device=device) if need_hxx else None
    asm_tt = torch.zeros(N, 1, device=device) if need_htt else None

    for k, c in enumerate(components):
        h_k = c['u'][:, 0:1]   # (N, 1)
        psi_k = c['psi_norm']
        is_constant = c.get('constant_psi', False)

        dh_k_dx = h_grads[k][:, 0:spatial_dim]      # (N, spatial_dim)
        dh_k_dt = h_grads[k][:, t_col:t_col+1]      # (N, 1)

        if need_hxx:
            d2h_k_dx2 = d2h_x[k][:, 0:spatial_dim]  # (N, spatial_dim)
        if need_htt:
            d2h_k_dt2 = d2h_t[k][:, t_col:t_col+1]  # (N, 1)

        # ============================================================
        # Indicator derivatives
        # ============================================================
        if use_analytical:
            psi_info = psi_derivs[k]
            psi_k_d = psi_info['psi_d']
            dpsi_dx = psi_info['dpsi_dx']
            dpsi_dt = psi_info['dpsi_dt']
            if need_hxx:
                d2psi_dx2 = psi_info['d2psi_dx2']
            if need_htt:
                d2psi_dt2 = psi_info['d2psi_dt2']
        elif is_constant:
            psi_k_d = psi_k.detach()
            dpsi_dx = torch.zeros(N, 1, device=device)
            dpsi_dt = torch.zeros(N, 1, device=device)
            if need_hxx:
                d2psi_dx2 = torch.zeros(N, 1, device=device)
            if need_htt:
                d2psi_dt2 = torch.zeros(N, 1, device=device)
        else:
            # Fallback: autograd-based indicator derivatives
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

            if need_htt:
                dpsi_dt = torch.autograd.grad(
                    psi_k.sum(), t, create_graph=True, retain_graph=True)[0]
                d2psi_dt2 = torch.autograd.grad(
                    dpsi_dt.sum(), t, retain_graph=True)[0]
                dpsi_dt = dpsi_dt.detach()
                d2psi_dt2 = d2psi_dt2.detach()
            else:
                dpsi_dt = torch.autograd.grad(
                    psi_k.sum(), t, retain_graph=True)[0]
                dpsi_dt = dpsi_dt.detach()

            psi_k_d = psi_k.detach()

        # ============================================================
        # Product rule assembly
        # ============================================================
        asm_x = asm_x + dpsi_dx * h_k + psi_k_d * dh_k_dx
        asm_t = asm_t + dpsi_dt * h_k + psi_k_d * dh_k_dt

        if need_hxx:
            asm_xx = (asm_xx
                      + d2psi_dx2 * h_k
                      + 2 * dpsi_dx * dh_k_dx
                      + psi_k_d * d2h_k_dx2)

        if need_htt:
            asm_tt = (asm_tt
                      + d2psi_dt2 * h_k
                      + 2 * dpsi_dt * dh_k_dt
                      + psi_k_d * d2h_k_dt2)

    h_x_out = asm_x.squeeze(-1)
    h_t_out = asm_t.squeeze(-1)
    h_xx_out = asm_xx.squeeze(-1) if need_hxx else None
    h_tt_out = asm_tt.squeeze(-1) if need_htt else None

    return h_t_out, h_tt_out, h_x_out, h_xx_out


def pde_residual(
    h_tt: torch.Tensor,
    h_xx: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the PDE residual: h_tt - h_xx.
    
    For the wave equation: h_tt - h_xx = 0
    
    Args:
        h_tt: Second time derivative ∂²h/∂t²
        h_xx: Second spatial derivative ∂²h/∂x²
        
    Returns:
        Residual tensor
    """
    residual = h_tt - h_xx
    return residual


def build_loss(**cfg) -> Callable:
    """
    Build physics-informed loss function for the 1D wave equation.
    
    Args:
        **cfg: Configuration dictionary containing:
            - problem: problem name (e.g., 'wave1d')
            - wave1d: dict with 'loss_weights' (residual, ic, bc)
            
    Returns:
        Callable loss function that takes (model, batch) and returns
        scalar tensor
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
        Compute physics-informed loss for 1D wave equation.
        
        Args:
            model: Neural network model (output_dim=1 for real-valued h)
            batch: Dictionary with keys:
                - 'x': (N, spatial_dim) spatial coordinates
                - 't': (N, 1) temporal coordinates
                - 'h_gt': (N, 1) ground truth (for IC/BC)
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
            x_f = x[masks['residual']].contiguous()  # (N_f, spatial_dim)
            t_f = t[masks['residual']].contiguous()  # (N_f, 1)
            
            # Enable gradients for autograd
            x_f = x_f.clone().detach().requires_grad_(True)
            t_f = t_f.clone().detach().requires_grad_(True)
            
            # Model prediction: concatenate x,t -> predict h
            xt_f = torch.cat([x_f, t_f], dim=1)
            
            # === Standard approach: differentiate composed output directly ===
            if _t: _t.start('loss.residual.forward')
            h_pred = model(xt_f)  # (N_f, 1)
            if _t: _t.stop('loss.residual.forward')
            
            # Extract h (squeeze output dimension for derivative computation)
            h_f = h_pred[:, 0]
            
            # Compute derivatives
            if _t: _t.start('loss.residual.derivatives')
            h_t, h_tt, h_x, h_xx = compute_derivatives(h_f, x_f, t_f)
            if _t: _t.stop('loss.residual.derivatives')
            
            # Compute PDE residual: h_tt - h_xx
            residual = pde_residual(h_tt, h_xx)
            
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
        # ============================================================
        if masks['IC'].sum() > 0:
            # Boolean indexing + .contiguous() for GPU efficiency
            x_0 = x[masks['IC']].contiguous()  # (N_0, spatial_dim)
            t_0 = t[masks['IC']].contiguous()  # (N_0, 1)
            h_gt_0 = h_gt[masks['IC']].contiguous()  # (N_0, 1)
            
            # Enable gradients for h_t computation
            x_0 = x_0.clone().detach().requires_grad_(True)
            t_0 = t_0.clone().detach().requires_grad_(True)
            
            # Model prediction
            xt_0 = torch.cat([x_0, t_0], dim=1)
            if _t: _t.start('loss.ic.forward')
            h_pred = model(xt_0)  # (N_0, 1)
            if _t: _t.stop('loss.ic.forward')
            h_0 = h_pred[:, 0]
            
            # Compute h_t for velocity IC
            if _t: _t.start('loss.ic.derivatives')
            h_t, _, _, _ = compute_derivatives(h_0, x_0, t_0)
            if _t: _t.stop('loss.ic.derivatives')
            
            # IC: h(x,0) = sin(x) and h_t(x,0) = 0
            # MSE for position: |h(x,0) - sin(x)|²
            ic_position_squared = (h_0 - h_gt_0[:, 0]) ** 2
            
            # MSE for velocity: |h_t(x,0) - 0|²
            ic_velocity_squared = h_t ** 2
            
            if for_tree_spawning:
                # Sum both components per sample
                ic_per_sample[masks['IC']] = ic_position_squared + ic_velocity_squared
            else:
                mse_position = torch.mean(ic_position_squared)
                mse_velocity = torch.mean(ic_velocity_squared)
                mse_ic = mse_position + mse_velocity
        else:
            if not for_tree_spawning:
                mse_ic = torch.tensor(0.0, device=device)
        
        # ============================================================
        # MSE_b: Boundary Condition Loss (use analytical boundary values)
        # Skipped when periodic Fourier embedding is active (use_bc=False).
        # ============================================================
        if use_bc and masks['BC'].sum() > 0:
            # Boolean indexing + .contiguous() for GPU efficiency
            x_b = x[masks['BC']].contiguous()  # (N_b, spatial_dim)
            t_b = t[masks['BC']].contiguous()  # (N_b, 1)
            h_gt_b = h_gt[masks['BC']].contiguous()  # (N_b, 1)
            
            # Model prediction
            xt_b = torch.cat([x_b, t_b], dim=1)
            if _t: _t.start('loss.bc.forward')
            h_pred = model(xt_b)  # (N_b, 1)
            if _t: _t.stop('loss.bc.forward')
            
            # Match the analytical boundary values instead of forcing zero
            bc_squared = (h_pred - h_gt_b) ** 2
            
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
