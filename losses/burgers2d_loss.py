"""
Physics-Informed Loss Function for the 2D Viscous Burgers Equation.

Implements the three-component loss:
    L = w_res*MSE_f + w_ic*MSE_0 + w_bc*MSE_b

where:
- MSE_f: PDE residual loss (h_t + h*(h_x0 + h_x1) - 0.1*(h_x0x0 + h_x1x1) = 0)
- MSE_0: Initial condition loss (h(0, x0, x1) = 1/(1 + exp((x0 + x1)/0.2)))
- MSE_b: Boundary condition loss (h(t, x0_b, x1_b) = 1/(1 + exp((x0_b + x1_b - t)/0.2)))
"""

import torch
import torch.nn as nn
from typing import Dict, Callable, Tuple
from losses.causal_weighting import create_causal_state, compute_causal_residual


def compute_derivatives(
    h: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute derivatives of scalar field h using vectorized autograd.
    
    Computes dh/dt, dh/dx0, dh/dx1, d²h/dx0², and d²h/dx1² using PyTorch autograd.
    All operations stay on the same device (GPU if available).
    
    Args:
        h: Scalar field, shape (batch_size,)
        x: Spatial coordinates, shape (batch_size, 2), requires_grad=True
        t: Temporal coordinates, shape (batch_size, 1), requires_grad=True
        
    Returns:
        Tuple of (h_t, h_x0, h_x1, h_x0x0, h_x1x1):
        - h_t: dh/dt
        - h_x0: dh/dx0
        - h_x1: dh/dx1
        - h_x0x0: d²h/dx0²
        - h_x1x1: d²h/dx1²
    """
    ones_h = torch.ones_like(h)
    
    h_grads = torch.autograd.grad(
        outputs=h,
        inputs=[x, t],
        grad_outputs=ones_h,
        create_graph=True,
        retain_graph=True,
    )
    h_x = h_grads[0]  # (batch_size, 2) - gradients w.r.t. x0 and x1
    h_t = h_grads[1]  # (batch_size, 1) - gradient w.r.t. t
    
    h_x0 = h_x[:, 0]  # (batch_size,)
    h_x1 = h_x[:, 1]  # (batch_size,)
    
    ones_hx0 = torch.ones_like(h_x0)
    h_x0x0 = torch.autograd.grad(
        outputs=h_x0,
        inputs=x,
        grad_outputs=ones_hx0,
        create_graph=True,
        retain_graph=True,
    )[0][:, 0]  # (batch_size,)
    
    ones_hx1 = torch.ones_like(h_x1)
    h_x1x1 = torch.autograd.grad(
        outputs=h_x1,
        inputs=x,
        grad_outputs=ones_hx1,
        create_graph=True,
        retain_graph=True,
    )[0][:, 1]  # (batch_size,)
    
    h_t = h_t.squeeze(-1)  # (batch_size,)
    
    return h_t, h_x0, h_x1, h_x0x0, h_x1x1


def compute_analytical_indicator_derivatives(
    inputs: torch.Tensor,
    indicator_data: dict,
    active_expert_indices,
) -> dict:
    """
    Compute normalized indicator derivatives analytically for 2D Burgers (3D input).

    Input dimensions: x0 (dim 0), x1 (dim 1), t (dim 2).

    For soft indicator: psi_k = prod_d sigma_L(d) * sigma_U(d)
    where sigma_L(d) = sigmoid((x_d - lower_d) / sigma_d)
          sigma_U(d) = sigmoid((upper_d - x_d) / sigma_d)

    First derivative:  dpsi_k/dx_d = psi_k * f_d
        where f_d = [(1 - sigma_L(d)) - (1 - sigma_U(d))] / sigma_d

    Second derivative: d2psi_k/dx_d^2 = psi_k * [f_d^2 - (sigma_L(1-sigma_L) + sigma_U(1-sigma_U)) / sigma_d^2]

    Normalized (quotient rule for psi_tilde_k = psi_k / Z):
        psi_tilde_k_xd  = (dpsi_k/dx_d - psi_tilde_k * Z_xd) / Z
        psi_tilde_k_xdxd = (d2psi_k/dx_d^2 - psi_tilde_k * Z_xdxd - 2 * psi_tilde_k_xd * Z_xd) / Z

    Args:
        inputs: (N, 3) original input coordinates [x0, x1, t]
        indicator_data: dict from model with all_lower, all_upper, all_sigma, etc.
        active_expert_indices: tensor of active expert indices

    Returns:
        dict[component_idx] with keys:
            'psi_d': (N, 1), 'dpsi_dx0': (N, 1), 'dpsi_dx1': (N, 1),
            'dpsi_dt': (N, 1), 'd2psi_dx0x0': (N, 1), 'd2psi_dx1x1': (N, 1)
    """
    all_lower = indicator_data['all_lower']   # (K, D)
    all_upper = indicator_data['all_upper']   # (K, D)
    all_sigma = indicator_data['all_sigma']   # (K, D)
    psi_base = indicator_data['psi_base']     # (N, 1)
    psi_experts_filtered = indicator_data['psi_experts_filtered']

    x0_dim = 0
    x1_dim = 1
    t_dim = 2

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

    # ---- Raw first derivatives for x0, x1, t ----
    def _f_and_raw_derivs(dim_idx):
        f = ((1 - sig_L[:, :, dim_idx]) - (1 - sig_U[:, :, dim_idx])) / all_sigma[:, dim_idx].unsqueeze(0)
        dpsi = psi_raw * f
        return f, dpsi

    f_x0, dpsi_raw_dx0 = _f_and_raw_derivs(x0_dim)
    f_x1, dpsi_raw_dx1 = _f_and_raw_derivs(x1_dim)
    f_t, dpsi_raw_dt = _f_and_raw_derivs(t_dim)

    # ---- Raw second spatial derivatives for x0, x1 ----
    def _raw_second_deriv(dim_idx, f_d):
        sL = sig_L[:, :, dim_idx]
        sU = sig_U[:, :, dim_idx]
        s = all_sigma[:, dim_idx].unsqueeze(0)
        return psi_raw * (f_d ** 2 - (sL * (1 - sL) + sU * (1 - sU)) / s ** 2)

    d2psi_raw_dx0x0 = _raw_second_deriv(x0_dim, f_x0)
    d2psi_raw_dx1x1 = _raw_second_deriv(x1_dim, f_x1)

    # ---- Normalization ----
    Z = psi_base + psi_experts_filtered.sum(dim=1, keepdim=True)  # (N, 1)
    Z = Z.clamp(min=1e-8)

    filt_mask = (psi_experts_filtered > 0).to(psi_experts_filtered.dtype)  # (N, K)

    Z_x0 = (dpsi_raw_dx0 * filt_mask).sum(dim=1, keepdim=True)
    Z_x1 = (dpsi_raw_dx1 * filt_mask).sum(dim=1, keepdim=True)
    Z_t = (dpsi_raw_dt * filt_mask).sum(dim=1, keepdim=True)
    Z_x0x0 = (d2psi_raw_dx0x0 * filt_mask).sum(dim=1, keepdim=True)
    Z_x1x1 = (d2psi_raw_dx1x1 * filt_mask).sum(dim=1, keepdim=True)

    # ---- Build per-component results ----
    results = {}

    # Component 0: Base model (constant psi_base -> zero raw derivatives)
    psi_norm_base = (psi_base / Z).detach()
    psi_norm_base_x0 = (-psi_norm_base * Z_x0 / Z).detach()
    psi_norm_base_x1 = (-psi_norm_base * Z_x1 / Z).detach()
    psi_norm_base_t = (-psi_norm_base * Z_t / Z).detach()
    psi_norm_base_x0x0 = (
        (-psi_norm_base * Z_x0x0 - 2 * psi_norm_base_x0 * Z_x0) / Z
    ).detach()
    psi_norm_base_x1x1 = (
        (-psi_norm_base * Z_x1x1 - 2 * psi_norm_base_x1 * Z_x1) / Z
    ).detach()

    results[0] = {
        'psi_d': psi_norm_base,
        'dpsi_dx0': psi_norm_base_x0,
        'dpsi_dx1': psi_norm_base_x1,
        'dpsi_dt': psi_norm_base_t,
        'd2psi_dx0x0': psi_norm_base_x0x0,
        'd2psi_dx1x1': psi_norm_base_x1x1,
    }

    # Components 1..num_active: active experts
    for comp_idx, expert_idx in enumerate(active_expert_indices):
        eidx = expert_idx.item() if torch.is_tensor(expert_idx) else expert_idx

        pt_mask = filt_mask[:, eidx:eidx+1]  # (N, 1)

        dpsi_k_dx0 = dpsi_raw_dx0[:, eidx:eidx+1] * pt_mask
        dpsi_k_dx1 = dpsi_raw_dx1[:, eidx:eidx+1] * pt_mask
        dpsi_k_dt = dpsi_raw_dt[:, eidx:eidx+1] * pt_mask
        d2psi_k_dx0x0 = d2psi_raw_dx0x0[:, eidx:eidx+1] * pt_mask
        d2psi_k_dx1x1 = d2psi_raw_dx1x1[:, eidx:eidx+1] * pt_mask

        psi_norm_k = (psi_experts_filtered[:, eidx:eidx+1] / Z).detach()

        psi_norm_k_x0 = ((dpsi_k_dx0 - psi_norm_k * Z_x0) / Z).detach()
        psi_norm_k_x1 = ((dpsi_k_dx1 - psi_norm_k * Z_x1) / Z).detach()
        psi_norm_k_t = ((dpsi_k_dt - psi_norm_k * Z_t) / Z).detach()
        psi_norm_k_x0x0 = (
            (d2psi_k_dx0x0 - psi_norm_k * Z_x0x0 - 2 * psi_norm_k_x0 * Z_x0) / Z
        ).detach()
        psi_norm_k_x1x1 = (
            (d2psi_k_dx1x1 - psi_norm_k * Z_x1x1 - 2 * psi_norm_k_x1 * Z_x1) / Z
        ).detach()

        results[comp_idx + 1] = {
            'psi_d': psi_norm_k,
            'dpsi_dx0': psi_norm_k_x0,
            'dpsi_dx1': psi_norm_k_x1,
            'dpsi_dt': psi_norm_k_t,
            'd2psi_dx0x0': psi_norm_k_x0x0,
            'd2psi_dx1x1': psi_norm_k_x1x1,
        }

    return results


def compute_derivatives_decomposed(
    components: list,
    x: torch.Tensor,
    t: torch.Tensor,
    indicator_data: dict = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute derivatives of scalar field h via product rule on decomposed expert outputs,
    using batched autograd for all experts simultaneously.

    For composed output u(x,t) = sum_k psi_tilde_k * h_k, the product rule gives:
        u_t    = sum_k (psi_tilde_k_t    * h_k + psi_tilde_k * h_k_t)
        u_x0   = sum_k (psi_tilde_k_x0   * h_k + psi_tilde_k * h_k_x0)
        u_x1   = sum_k (psi_tilde_k_x1   * h_k + psi_tilde_k * h_k_x1)
        u_x0x0 = sum_k (psi_tilde_k_x0x0 * h_k + 2*psi_tilde_k_x0 * h_k_x0 + psi_tilde_k * h_k_x0x0)
        u_x1x1 = sum_k (psi_tilde_k_x1x1 * h_k + 2*psi_tilde_k_x1 * h_k_x1 + psi_tilde_k * h_k_x1x1)

    Batched autograd (3 calls):
        Call 1: grad(total_h_sum, all_inputs) -> per-expert (N, 3) grads
                col 0 = dh_k/dx0, col 1 = dh_k/dx1, col 2 = dh_k/dt
        Call 2: grad(sum_dh_dx0, all_inputs) -> d2h_k/dx0^2 (col 0)
        Call 3: grad(sum_dh_dx1, all_inputs) -> d2h_k/dx1^2 (col 1)

    Args:
        components: list of dicts from model.forward_for_pde_derivatives(), each with:
            - 'u': (N, 1) model output, on autograd graph
            - 'inputs': (N, 3) per-expert input copy (leaf, requires_grad=True)
            - 'psi_norm': (N, 1) normalized weight, on autograd graph
            - 'constant_psi': bool, True if weight is constant
        x: spatial coordinates (N, 2), requires_grad=True
        t: temporal coordinates (N, 1), requires_grad=True
        indicator_data: dict from model with bounds/sigma for analytical derivatives (or None)

    Returns:
        Tuple of (h_t, h_x0, h_x1, h_x0x0, h_x1x1) - each (N,) tensor
    """
    N = x.shape[0]
    device = x.device
    num_components = len(components)
    spatial_dim = x.shape[1]  # 2
    t_col = spatial_dim       # 2

    # ====================================================================
    # BATCHED expert first derivatives (1 autograd call for ALL experts)
    # ====================================================================
    all_inputs = [c['inputs'] for c in components]

    total_h_sum = sum(c['u'][:, 0].sum() for c in components)

    # Call 1: all first derivatives in one shot
    # grads[k] is (N, 3) with cols [dh_k/dx0, dh_k/dx1, dh_k/dt]
    first_grads = torch.autograd.grad(
        total_h_sum, all_inputs, create_graph=True, retain_graph=True)

    # ====================================================================
    # BATCHED expert second derivatives (2 more calls)
    # ====================================================================
    # Sum dh/dx0 across all experts for second derivative w.r.t. x0
    total_dh_dx0 = sum(
        first_grads[k][:, 0].sum() for k in range(num_components))
    # Sum dh/dx1 across all experts for second derivative w.r.t. x1
    total_dh_dx1 = sum(
        first_grads[k][:, 1].sum() for k in range(num_components))

    # Call 2: d2h_k/dx0^2 for all experts
    d2_dx0 = torch.autograd.grad(
        total_dh_dx0, all_inputs, create_graph=True, retain_graph=True)

    # Call 3: d2h_k/dx1^2 for all experts
    d2_dx1 = torch.autograd.grad(
        total_dh_dx1, all_inputs, create_graph=True, retain_graph=True)

    # ====================================================================
    # Indicator derivatives: analytical or autograd fallback
    # ====================================================================
    use_analytical = (indicator_data is not None
                      and indicator_data.get('all_sigma') is not None)

    if use_analytical:
        inputs_orig = torch.cat([x, t], dim=1)  # (N, 3)
        active_indices = indicator_data['active_expert_indices']
        psi_derivs = compute_analytical_indicator_derivatives(
            inputs_orig, indicator_data, active_indices)

    # ====================================================================
    # Per-component: product rule assembly
    # ====================================================================
    asm_x0 = torch.zeros(N, 1, device=device)
    asm_x1 = torch.zeros(N, 1, device=device)
    asm_t = torch.zeros(N, 1, device=device)
    asm_x0x0 = torch.zeros(N, 1, device=device)
    asm_x1x1 = torch.zeros(N, 1, device=device)

    for k, c in enumerate(components):
        h_k = c['u'][:, 0:1]  # (N, 1)
        psi_k = c['psi_norm']
        is_constant = c.get('constant_psi', False)

        # Extract per-expert derivatives from batched results
        h_k_x0 = first_grads[k][:, 0:1]   # (N, 1)
        h_k_x1 = first_grads[k][:, 1:2]   # (N, 1)
        h_k_t = first_grads[k][:, t_col:t_col+1]  # (N, 1)

        h_k_x0x0 = d2_dx0[k][:, 0:1]  # (N, 1)
        h_k_x1x1 = d2_dx1[k][:, 1:2]  # (N, 1)

        # ============================================================
        # Indicator derivatives
        # ============================================================
        if use_analytical:
            psi_info = psi_derivs[k]
            psi_k_d = psi_info['psi_d']
            dpsi_dx0 = psi_info['dpsi_dx0']
            dpsi_dx1 = psi_info['dpsi_dx1']
            dpsi_dt = psi_info['dpsi_dt']
            d2psi_dx0x0 = psi_info['d2psi_dx0x0']
            d2psi_dx1x1 = psi_info['d2psi_dx1x1']
        elif is_constant:
            psi_k_d = psi_k.detach()
            dpsi_dx0 = torch.zeros(N, 1, device=device)
            dpsi_dx1 = torch.zeros(N, 1, device=device)
            dpsi_dt = torch.zeros(N, 1, device=device)
            d2psi_dx0x0 = torch.zeros(N, 1, device=device)
            d2psi_dx1x1 = torch.zeros(N, 1, device=device)
        else:
            # Autograd fallback for indicator derivatives
            dpsi_dx_all = torch.autograd.grad(
                psi_k.sum(), x, create_graph=True, retain_graph=True)[0]  # (N, 2)
            d2psi_dx0x0_raw = torch.autograd.grad(
                dpsi_dx_all[:, 0].sum(), x, retain_graph=True)[0][:, 0:1]
            d2psi_dx1x1_raw = torch.autograd.grad(
                dpsi_dx_all[:, 1].sum(), x, retain_graph=True)[0][:, 1:2]

            dpsi_dx0 = dpsi_dx_all[:, 0:1].detach()
            dpsi_dx1 = dpsi_dx_all[:, 1:2].detach()
            d2psi_dx0x0 = d2psi_dx0x0_raw.detach()
            d2psi_dx1x1 = d2psi_dx1x1_raw.detach()

            dpsi_dt = torch.autograd.grad(
                psi_k.sum(), t, retain_graph=True)[0].detach()  # (N, 1)

            psi_k_d = psi_k.detach()

        # ============================================================
        # Product rule assembly
        # ============================================================
        asm_x0 = asm_x0 + dpsi_dx0 * h_k + psi_k_d * h_k_x0
        asm_x1 = asm_x1 + dpsi_dx1 * h_k + psi_k_d * h_k_x1
        asm_t = asm_t + dpsi_dt * h_k + psi_k_d * h_k_t

        asm_x0x0 = (asm_x0x0
                     + d2psi_dx0x0 * h_k
                     + 2 * dpsi_dx0 * h_k_x0
                     + psi_k_d * h_k_x0x0)
        asm_x1x1 = (asm_x1x1
                     + d2psi_dx1x1 * h_k
                     + 2 * dpsi_dx1 * h_k_x1
                     + psi_k_d * h_k_x1x1)

    # Squeeze to (N,) for consistency with pde_residual signature
    return (asm_t.squeeze(-1),
            asm_x0.squeeze(-1),
            asm_x1.squeeze(-1),
            asm_x0x0.squeeze(-1),
            asm_x1x1.squeeze(-1))


def pde_residual(
    h: torch.Tensor,
    h_t: torch.Tensor,
    h_x0: torch.Tensor,
    h_x1: torch.Tensor,
    h_x0x0: torch.Tensor,
    h_x1x1: torch.Tensor,
    nu: float = 0.1,
) -> torch.Tensor:
    """
    Compute the PDE residual: h_t + h*(h_x0 + h_x1) - nu*(h_x0x0 + h_x1x1).
    
    For the 2D viscous Burgers equation:
        h_t + h*(h_x0 + h_x1) - 0.1*(h_x0x0 + h_x1x1) = 0
    
    Args:
        h: Solution field h
        h_t: Time derivative dh/dt
        h_x0: Spatial derivative dh/dx0
        h_x1: Spatial derivative dh/dx1
        h_x0x0: Second spatial derivative d²h/dx0²
        h_x1x1: Second spatial derivative d²h/dx1²
        nu: Viscosity coefficient (default 0.1)
        
    Returns:
        Residual tensor
    """
    residual = h_t + h * (h_x0 + h_x1) - nu * (h_x0x0 + h_x1x1)
    return residual


def build_loss(**cfg) -> Callable:
    """
    Build physics-informed loss function for the 2D viscous Burgers equation.
    
    Args:
        **cfg: Configuration dictionary containing:
            - problem: problem name (e.g., 'burgers2d')
            - burgers2d: dict with 'loss_weights' (residual, ic, bc) and 'nu'
            
    Returns:
        Callable loss function that takes (model, batch) and returns
        a scalar loss tensor.
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
        Compute physics-informed loss for 2D viscous Burgers equation.
        
        Args:
            model: Neural network model (output_dim=1 for real-valued h)
            batch: Dictionary with keys:
                - 'x': (N, spatial_dim) spatial coordinates [x0, x1]
                - 't': (N, 1) temporal coordinates
                - 'h_gt': (N, 1) ground truth (for IC and BC)
                - 'mask': dict with 'residual', 'IC', 'BC' boolean masks
            for_tree_spawning: If True, return per-sample loss components dict
                
        Returns:
            - If for_tree_spawning=False: Scalar total loss
            - If for_tree_spawning=True: Dict with keys 'residual', 'ic', 'bc'
              containing per-sample loss tensors (N,)
        """
        x = batch['x']  # (N, 2)
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
            h_t_val, h_x0, h_x1, h_x0x0, h_x1x1 = compute_derivatives(h_f, x_f, t_f)
            if _t: _t.stop('loss.residual.derivatives')
            
            residual = pde_residual(h_f, h_t_val, h_x0, h_x1, h_x0x0, h_x1x1, nu=nu)
            
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
        # h(0, x0, x1) = 1 / (1 + exp((x0 + x1) / 0.2))
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
            
            # IC loss: MSE between predicted and ground truth
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
        # h(t, x0_b, x1_b) = 1 / (1 + exp((x0_b + x1_b - t) / 0.2))
        # Skipped when periodic Fourier embedding is active (use_bc=False).
        # ============================================================
        if use_bc and masks['BC'].sum() > 0:
            x_b = x[masks['BC']].contiguous()
            t_b = t[masks['BC']].contiguous()
            h_gt_b = h_gt[masks['BC']].contiguous()  # (N_b, 1)
            
            # Model prediction
            xt_b = torch.cat([x_b, t_b], dim=1)
            if _t: _t.start('loss.bc.forward')
            h_pred_b = model(xt_b)  # (N_b, 1)
            if _t: _t.stop('loss.bc.forward')
            
            # BC loss: MSE between predicted and analytical boundary values
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
