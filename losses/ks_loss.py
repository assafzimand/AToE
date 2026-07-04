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
from losses.causal_weighting import (create_causal_state, compute_causal_residual,
                                      compute_per_leaf_causal_residual)

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


def compute_analytical_indicator_derivatives(
    inputs: torch.Tensor,
    indicator_data: dict,
    active_expert_indices,
    need_ht: bool = True,
    need_hxxxx: bool = True,
) -> dict:
    """
    Compute normalized indicator derivatives ANALYTICALLY up to 4th order.

    Extended from KdV (3rd order) to include the 4th spatial derivative needed by KS.

    Raw indicator psi_k = prod_d sigma_L(d) * sigma_U(d).
    With f_x = d(log psi)/dx:

        f_x   = [(1-sig_L_x) - (1-sig_U_x)] / sigma_x
        f'_x  = -[sig_L_x*(1-sig_L_x) + sig_U_x*(1-sig_U_x)] / sigma_x^2
        f''_x = -[sig_L_x*(1-sig_L_x)*(1-2*sig_L_x)
                 - sig_U_x*(1-sig_U_x)*(1-2*sig_U_x)] / sigma_x^3
        f'''_x = -[sig_L_x*(1-sig_L_x)*(1 - 6*sig_L_x*(1-sig_L_x))
                  + sig_U_x*(1-sig_U_x)*(1 - 6*sig_U_x*(1-sig_U_x))] / sigma_x^4

    Raw derivatives via Bell polynomials (Faà di Bruno):
        d1 psi = psi * f_x
        d2 psi = psi * (f_x^2 + f'_x)
        d3 psi = psi * (f_x^3 + 3*f_x*f'_x + f''_x)
        d4 psi = psi * (f_x^4 + 6*f_x^2*f'_x + 3*f'_x^2 + 4*f_x*f''_x + f'''_x)

    Normalized (quotient rule for psi_tilde_k = psi_k / Z):
        psi_tilde^(n) = (d^n psi - sum_{j=0}^{n-1} C(n,j)*psi_tilde^(j)*Z^(n-j)) / Z
    """
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

    sig_L_x = sig_L[:, :, x_dim]
    sig_U_x = sig_U[:, :, x_dim]
    sigma_x = all_sigma[:, x_dim].unsqueeze(0)

    sL_sLbar = sig_L_x * (1 - sig_L_x)
    sU_sUbar = sig_U_x * (1 - sig_U_x)

    # f_x and raw 1st derivative
    f_x = ((1 - sig_L_x) - (1 - sig_U_x)) / sigma_x
    dpsi_raw_dx = psi_raw * f_x

    if need_ht:
        f_t = ((1 - sig_L[:, :, t_dim]) - (1 - sig_U[:, :, t_dim])) / all_sigma[:, t_dim].unsqueeze(0)
        dpsi_raw_dt = psi_raw * f_t

    # f'_x and raw 2nd derivative
    f_prime_x = -(sL_sLbar + sU_sUbar) / sigma_x ** 2
    d2psi_raw_dx2 = psi_raw * (f_x ** 2 + f_prime_x)

    # f''_x and raw 3rd derivative
    f_double_prime_x = -(
        sL_sLbar * (1 - 2 * sig_L_x) - sU_sUbar * (1 - 2 * sig_U_x)
    ) / sigma_x ** 3
    d3psi_raw_dx3 = psi_raw * (f_x ** 3 + 3 * f_x * f_prime_x + f_double_prime_x)

    # f'''_x and raw 4th derivative
    if need_hxxxx:
        f_triple_prime_x = -(
            sL_sLbar * (1 - 6 * sL_sLbar)
            + sU_sUbar * (1 - 6 * sU_sUbar)
        ) / sigma_x ** 4
        d4psi_raw_dx4 = psi_raw * (
            f_x ** 4
            + 6 * f_x ** 2 * f_prime_x
            + 3 * f_prime_x ** 2
            + 4 * f_x * f_double_prime_x
            + f_triple_prime_x
        )

    # Normalization
    Z = psi_base + psi_experts_filtered.sum(dim=1, keepdim=True)
    Z = Z.clamp(min=1e-8)

    filt_mask = (psi_experts_filtered > 0).to(psi_experts_filtered.dtype)

    Z_x = (dpsi_raw_dx * filt_mask).sum(dim=1, keepdim=True)
    Z_xx = (d2psi_raw_dx2 * filt_mask).sum(dim=1, keepdim=True)
    Z_xxx = (d3psi_raw_dx3 * filt_mask).sum(dim=1, keepdim=True)

    if need_ht:
        Z_t = (dpsi_raw_dt * filt_mask).sum(dim=1, keepdim=True)
    if need_hxxxx:
        Z_xxxx = (d4psi_raw_dx4 * filt_mask).sum(dim=1, keepdim=True)

    results = {}

    # Component 0: Base model (constant raw psi -> zero raw derivatives)
    psi_norm_base = (psi_base / Z).detach()
    psi_norm_base_x = (-psi_norm_base * Z_x / Z).detach()

    results[0] = {
        'psi_d': psi_norm_base,
        'dpsi_dx': psi_norm_base_x,
    }
    if need_ht:
        results[0]['dpsi_dt'] = (-psi_norm_base * Z_t / Z).detach()

    psi_norm_base_xx = (
        -psi_norm_base * Z_xx - 2 * psi_norm_base_x * Z_x
    ) / Z
    results[0]['d2psi_dx2'] = psi_norm_base_xx.detach()

    psi_norm_base_xxx = (
        -psi_norm_base * Z_xxx - 3 * psi_norm_base_xx * Z_x - 3 * psi_norm_base_x * Z_xx
    ) / Z
    results[0]['d3psi_dx3'] = psi_norm_base_xxx.detach()

    if need_hxxxx:
        psi_norm_base_xxxx = (
            -psi_norm_base * Z_xxxx
            - 4 * psi_norm_base_xxx * Z_x
            - 6 * psi_norm_base_xx * Z_xx
            - 4 * psi_norm_base_x * Z_xxx
        ) / Z
        results[0]['d4psi_dx4'] = psi_norm_base_xxxx.detach()

    # Expert components
    for comp_idx, expert_idx in enumerate(active_expert_indices):
        eidx = expert_idx.item() if torch.is_tensor(expert_idx) else expert_idx

        pt_mask = filt_mask[:, eidx:eidx+1]

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

        d2psi_k_dx2 = d2psi_raw_dx2[:, eidx:eidx+1] * pt_mask
        psi_norm_k_xx = (
            (d2psi_k_dx2 - psi_norm_k * Z_xx - 2 * psi_norm_k_x * Z_x) / Z
        ).detach()
        result_k['d2psi_dx2'] = psi_norm_k_xx

        d3psi_k_dx3 = d3psi_raw_dx3[:, eidx:eidx+1] * pt_mask
        psi_norm_k_xxx = (
            (d3psi_k_dx3 - psi_norm_k * Z_xxx - 3 * psi_norm_k_xx * Z_x - 3 * psi_norm_k_x * Z_xx) / Z
        ).detach()
        result_k['d3psi_dx3'] = psi_norm_k_xxx

        if need_hxxxx:
            d4psi_k_dx4 = d4psi_raw_dx4[:, eidx:eidx+1] * pt_mask
            psi_norm_k_xxxx = (
                (d4psi_k_dx4
                 - psi_norm_k * Z_xxxx
                 - 4 * psi_norm_k_xxx * Z_x
                 - 6 * psi_norm_k_xx * Z_xx
                 - 4 * psi_norm_k_x * Z_xxx) / Z
            ).detach()
            result_k['d4psi_dx4'] = psi_norm_k_xxxx

        results[comp_idx + 1] = result_k

    return results


def compute_derivatives_decomposed(
    components: list,
    x: torch.Tensor,
    t: torch.Tensor,
    need_ht: bool = True,
    need_hxxxx: bool = True,
    indicator_data: dict = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute derivatives of scalar field h via product rule on decomposed expert outputs,
    using BATCHED autograd for all experts simultaneously.

    For composed output h(x,t) = sum_k psi_tilde_k * h_k, the general Leibniz rule gives:
        h_x    = sum_k (psi'_k * h_k + psi_k * h_k_x)
        h_t    = sum_k (psi_k_t * h_k + psi_k * h_k_t)
        h_xx   = sum_k (psi''_k * h_k + 2*psi'_k * h_k_x + psi_k * h_k_xx)
        h_xxxx = sum_k (psi''''_k*h_k + 4*psi'''_k*h_k_x + 6*psi''_k*h_k_xx
                        + 4*psi'_k*h_k_xxx + psi_k*h_k_xxxx)

    Returns:
        Tuple of (h_t, h_x, h_xx, h_xxxx) as (N,) tensors.
    """
    N = x.shape[0]
    device = x.device
    num_components = len(components)
    spatial_dim = x.shape[1]
    t_col = spatial_dim

    all_inputs = [c['inputs'] for c in components]
    total_h = sum(c['u'][:, 0].sum() for c in components)

    # Call 1: first derivatives
    h_grads = torch.autograd.grad(
        total_h, all_inputs, create_graph=True, retain_graph=True)

    # Call 2: second spatial derivatives
    total_dh_dx = sum(
        h_grads[k][:, 0:spatial_dim].sum() for k in range(num_components))
    d2_h = torch.autograd.grad(
        total_dh_dx, all_inputs, create_graph=True, retain_graph=True)

    # Call 3: third spatial derivatives
    total_d2h_dx2 = sum(
        d2_h[k][:, 0:spatial_dim].sum() for k in range(num_components))
    d3_h = torch.autograd.grad(
        total_d2h_dx2, all_inputs, create_graph=True, retain_graph=True)

    # Call 4: fourth spatial derivatives (if needed)
    d4_h = None
    if need_hxxxx:
        total_d3h_dx3 = sum(
            d3_h[k][:, 0:spatial_dim].sum() for k in range(num_components))
        d4_h = torch.autograd.grad(
            total_d3h_dx3, all_inputs, create_graph=True, retain_graph=True)

    # Indicator derivatives: autograd only (analytical path is LEGACY)
    # LEGACY: The analytical path checked for 'all_sigma' which was specific to sigmoid windows.
    use_analytical = False  # LEGACY PATH DISABLED
    # Original: (indicator_data is not None and indicator_data.get('all_sigma') is not None)

    if use_analytical:
        inputs_orig = torch.cat([x, t], dim=1)
        active_indices = indicator_data['active_expert_indices']
        psi_derivs = compute_analytical_indicator_derivatives(
            inputs_orig, indicator_data, active_indices,
            need_ht=need_ht, need_hxxxx=need_hxxxx)

    # Per-component product rule assembly
    asm_x = torch.zeros(N, 1, device=device)
    asm_xx = torch.zeros(N, 1, device=device)
    asm_t = torch.zeros(N, 1, device=device) if need_ht else None
    asm_xxxx = torch.zeros(N, 1, device=device) if need_hxxxx else None

    for k, c in enumerate(components):
        h_k = c['u'][:, 0:1]
        psi_k = c['psi_norm']
        is_constant = c.get('constant_psi', False)

        dh_k_dx = h_grads[k][:, 0:spatial_dim]
        d2h_k_dx2 = d2_h[k][:, 0:spatial_dim]
        d3h_k_dx3 = d3_h[k][:, 0:spatial_dim]

        if need_ht:
            dh_k_dt = h_grads[k][:, t_col:t_col+1]

        if need_hxxxx:
            d4h_k_dx4 = d4_h[k][:, 0:spatial_dim]

        if use_analytical:
            psi_info = psi_derivs[k]
            psi_k_d = psi_info['psi_d']
            dpsi_dx = psi_info['dpsi_dx']
            d2psi_dx2 = psi_info['d2psi_dx2']
            d3psi_dx3 = psi_info['d3psi_dx3']
            if need_ht:
                dpsi_dt = psi_info['dpsi_dt']
            if need_hxxxx:
                d4psi_dx4 = psi_info['d4psi_dx4']
        elif is_constant:
            psi_k_d = psi_k.detach()
            dpsi_dx = torch.zeros(N, 1, device=device)
            d2psi_dx2 = torch.zeros(N, 1, device=device)
            d3psi_dx3 = torch.zeros(N, 1, device=device)
            if need_hxxxx:
                d4psi_dx4 = torch.zeros(N, 1, device=device)
            if need_ht:
                dpsi_dt = torch.zeros(N, 1, device=device)
        else:
            dpsi_dx = torch.autograd.grad(
                psi_k.sum(), x, create_graph=True, retain_graph=True)[0]
            d2psi_dx2 = torch.autograd.grad(
                dpsi_dx.sum(), x, create_graph=True, retain_graph=True)[0]
            d3psi_dx3 = torch.autograd.grad(
                d2psi_dx2.sum(), x, create_graph=True, retain_graph=True)[0]

            if need_hxxxx:
                d4psi_dx4 = torch.autograd.grad(
                    d3psi_dx3.sum(), x, retain_graph=True)[0]
                d4psi_dx4 = d4psi_dx4.detach()

            dpsi_dx = dpsi_dx.detach()
            d2psi_dx2 = d2psi_dx2.detach()
            d3psi_dx3 = d3psi_dx3.detach()

            if need_ht:
                dpsi_dt = torch.autograd.grad(
                    psi_k.sum(), t, retain_graph=True)[0]
                dpsi_dt = dpsi_dt.detach()

            psi_k_d = psi_k.detach()

        # h_x: Leibniz order 1
        asm_x = asm_x + dpsi_dx * h_k + psi_k_d * dh_k_dx

        # h_xx: Leibniz order 2
        asm_xx = asm_xx + d2psi_dx2 * h_k + 2 * dpsi_dx * dh_k_dx + psi_k_d * d2h_k_dx2

        # h_t
        if need_ht:
            asm_t = asm_t + dpsi_dt * h_k + psi_k_d * dh_k_dt

        # h_xxxx: Leibniz order 4  (C(4,0)=1, C(4,1)=4, C(4,2)=6, C(4,3)=4, C(4,4)=1)
        if need_hxxxx:
            asm_xxxx = (asm_xxxx
                        + d4psi_dx4 * h_k
                        + 4 * d3psi_dx3 * dh_k_dx
                        + 6 * d2psi_dx2 * d2h_k_dx2
                        + 4 * dpsi_dx * d3h_k_dx3
                        + psi_k_d * d4h_k_dx4)

    h_x_out = asm_x.squeeze(-1)
    h_xx_out = asm_xx.squeeze(-1)

    h_t_out = None
    if need_ht:
        h_t_out = asm_t.squeeze(-1)

    h_xxxx_out = None
    if need_hxxxx:
        h_xxxx_out = asm_xxxx.squeeze(-1)

    return h_t_out, h_x_out, h_xx_out, h_xxxx_out


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
    _leaf_state = {}  # mutable container; trainer populates when per_leaf_causal=True

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
                _leaf_causal_states = _leaf_state.get('causal_states')
                _leaf_info_perlf = _leaf_state.get('leaf_info')
                if _leaf_causal_states is not None and _leaf_info_perlf is not None:
                    mse_residual = compute_per_leaf_causal_residual(
                        residual_squared, x_f, t_f, _leaf_info_perlf,
                        _leaf_causal_states, update_state=update_causal_state)
                else:
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
            if use_bc:
                comps['bc'] = mse_bc
            return comps
        else:
            total_loss = weight_residual * mse_residual + weight_ic * mse_ic
            if use_bc:
                total_loss = total_loss + weight_bc * mse_bc
            return total_loss

    loss_fn.causal_state = causal_state
    loss_fn._leaf_state = _leaf_state
    return loss_fn
