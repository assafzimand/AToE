"""Per-expert split loss for PDD-style subdomain training.

Each leaf expert is trained on its OWN output (no PoU), with:
  - PDE residual inside its subdomain
  - Dirichlet matching to the frozen root on interior faces (interface)
  - True IC/BC on faces coinciding with global domain bounds
  - Neighbor-to-neighbor continuity on shared interior faces

For Allen-Cahn with periodic BC, global boundary points are
paired across experts (left/right at same t), penalizing both
value and spatial derivative mismatches.

Total loss = SUM over experts of:
    w_res*L_res + w_ic*L_ic + w_bc*L_bc + w_cont*L_cont
where interface faces inherit the IC or BC weight by face type.
"""

import torch
import importlib
from typing import Dict, Callable
from adaptive.subdomain_data import (
    KIND_RESIDUAL, KIND_IC_TRUE, KIND_INTERFACE, KIND_INTERFACE_BC, KIND_BC_TRUE,
    KIND_CONTINUITY,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


def build_split_loss(
    model,
    cfg: Dict,
    *,
    orig_loss_fn: Callable = None,
) -> Callable:
    """Build a split loss for per-expert subdomain training.

    Returns a callable ``loss_fn(model, batch)`` compatible
    with ``_train_segment``.

    If ``orig_loss_fn`` is provided, batches missing split-specific
    keys (expert_id, kind) will fall back to the original loss
    (used for eval batches).
    """
    problem = cfg['problem']
    pc = cfg[problem]
    loss_weights = pc['loss_weights']
    w_res = loss_weights['residual']
    w_ic = loss_weights['ic']
    w_bc = loss_weights['bc']
    w_cont = loss_weights.get('continuity', 1.0)

    # Allen-Cahn uses periodic BC pairing, so skip Dirichlet-to-zero
    is_allen_cahn = (problem == 'allen_cahn')

    pde_res_fn, deriv_fn = _import_pde_helpers(problem)
    pde_params = _get_pde_params(problem, pc)

    per_expert_history: Dict[int, Dict[str, list]] = {}
    # Per-epoch residual cache: list of (x, t, r²) tuples (detached CPU tensors).
    # Populated when split_loss_fn._cache_residuals is True; drained by the trainer
    # to produce diagnostic heatmap plots (same as the non-split residual-cache path).
    residual_cache: list = []

    def split_loss_fn(
        model, batch, return_components=False, **kw
    ):
        # Fix 1: Fall back to original loss if batch lacks split keys (eval batches)
        if 'expert_id' not in batch or 'kind' not in batch:
            if orig_loss_fn is not None:
                return orig_loss_fn(model, batch, return_components=return_components, **kw)
            else:
                # No fallback available, compute simple MSE
                x = batch['x']
                t = batch['t']
                h_gt = batch['h_gt']
                device = x.device
                xt = torch.cat([x, t], dim=1)
                h_pred = model(xt)
                return torch.mean((h_pred - h_gt) ** 2)
        
        x = batch['x']
        t = batch['t']
        h_gt = batch['h_gt']
        expert_ids = batch['expert_id']
        kinds = batch['kind']
        bc_face_ids = batch.get('bc_face_id', None)
        cont_neighbors = batch.get('cont_neighbor', None)
        cont_dims = batch.get('cont_dim', None)
        device = x.device

        unique_experts = expert_ids.unique().tolist()
        total_loss = torch.tensor(0.0, device=device)
        all_comps = {}

        # ── Residual: all experts share ONE autograd graph ──
        # Each expert's points go through its own network, but derivatives are
        # taken w.r.t. a single (xf, tf) leaf pair, so the (expensive, 2nd-4th
        # order) autograd calls run once instead of once per expert.
        residual_losses = _compute_all_residual_losses(
            model, x, t, expert_ids, kinds,
            pde_res_fn, deriv_fn, pde_params,
            residual_cache=(
                residual_cache
                if split_loss_fn._cache_residuals else None
            ),
        )

        for eidx in unique_experts:
            emask = (expert_ids == eidx)
            comps = _compute_expert_loss(
                model, eidx,
                x[emask], t[emask], h_gt[emask],
                kinds[emask],
                w_res, w_ic, w_bc, is_allen_cahn,
                device,
                residual_loss=residual_losses.get(eidx),
            )
            total_loss = total_loss + comps['total']
            _record(per_expert_history, eidx, comps)
            if return_components:
                all_comps[eidx] = comps

        # ── Periodic BC (Allen-Cahn): cross-expert pairing ──
        if is_allen_cahn and bc_face_ids is not None:
            bc_loss_contrib = _compute_periodic_bc_loss(
                model, x, t, expert_ids, kinds,
                bc_face_ids, deriv_fn, device,
            )
            if bc_loss_contrib.item() > 0:
                logger.debug(
                    f"[SplitLoss] Periodic BC contrib: "
                    f"{bc_loss_contrib.item():.6e}"
                )
            total_loss = total_loss + w_bc * bc_loss_contrib

        # ── Continuity loss: neighbor-to-neighbor on shared interior faces ──
        if cont_neighbors is not None and cont_dims is not None:
            cont_loss, cont_per_expert = _compute_continuity_loss(
                model, x, t, expert_ids, kinds,
                cont_neighbors, cont_dims, deriv_fn,
                pde_params, device, problem,
            )
            if cont_loss.item() > 0:
                logger.debug(
                    f"[SplitLoss] Continuity contrib: "
                    f"{cont_loss.item():.6e}"
                )
            total_loss = total_loss + w_cont * cont_loss
            # Record continuity per expert
            for eidx, cont_val in cont_per_expert.items():
                _record_continuity(per_expert_history, eidx, cont_val)

        if return_components:
            return all_comps
        return total_loss

    split_loss_fn._per_expert_history = per_expert_history
    split_loss_fn._residual_cache = residual_cache
    split_loss_fn._cache_residuals = False  # trainer sets True when a plot is due
    return split_loss_fn


def _compute_all_residual_losses(
    model, x, t, expert_ids, kinds,
    pde_res_fn, deriv_fn, pde_params,
    residual_cache=None,
):
    """Per-expert residual losses computed in a single autograd graph.

    All experts' residual points are stacked (grouped by expert) onto one
    ``(xf, tf)`` leaf pair; each block is forwarded through its own expert
    network, and the PDE derivatives are computed once over the concatenated
    output. The per-expert mean of r² over its own points is unchanged
    relative to computing each expert separately.

    Returns:
        Dict mapping expert_idx -> residual loss tensor (mean r² in region).
    """
    rmask = (kinds == KIND_RESIDUAL)
    if rmask.sum() == 0:
        return {}

    x_r = x[rmask]
    t_r = t[rmask]
    eid_r = expert_ids[rmask]

    # Group points into contiguous per-expert blocks
    order = torch.argsort(eid_r, stable=True)
    x_r = x_r[order]
    t_r = t_r[order]
    eid_r = eid_r[order]

    xf = x_r.clone().detach().requires_grad_(True)
    tf = t_r.clone().detach().requires_grad_(True)
    xt = torch.cat([xf, tf], dim=1)

    u_parts = []
    bounds = []  # (expert_idx, start, end) into the stacked tensors
    start = 0
    for eidx in eid_r.unique(sorted=True).tolist():
        n = int((eid_r == eidx).sum().item())
        u_parts.append(model.forward_single_expert(int(eidx), xt[start:start + n]))
        bounds.append((int(eidx), start, start + n))
        start += n
    u_all = torch.cat(u_parts, dim=0)

    hf = u_all[:, 0]
    derivs = deriv_fn(hf, xf, tf)
    res = pde_res_fn(hf, *derivs, **pde_params)
    r2 = res ** 2

    if residual_cache is not None:
        residual_cache.append((
            xf.detach().cpu(),
            tf.detach().cpu(),
            r2.detach().cpu(),
        ))

    return {eidx: r2[s:e].mean() for eidx, s, e in bounds}


def _compute_expert_loss(
    model, expert_idx, x, t, h_gt, kinds,
    w_res, w_ic, w_bc, is_allen_cahn, device,
    residual_loss=None,
):
    """Per-expert local loss (no PoU). Residual is supplied precomputed."""
    z = torch.tensor(0.0, device=device)
    comps = {
        'residual': residual_loss if residual_loss is not None else z.clone(),
        'ic': z.clone(),
        'interface_ic': z.clone(),
        'interface_bc': z.clone(),
        'bc': z.clone(),
    }

    # ── IC true (real t=0) ──
    ic_mask = (kinds == KIND_IC_TRUE)
    if ic_mask.sum() > 0:
        xt_ic = torch.cat(
            [x[ic_mask], t[ic_mask]], dim=1
        )
        u_ic = model.forward_single_expert(
            expert_idx, xt_ic
        )
        comps['ic'] = torch.mean(
            (u_ic - h_gt[ic_mask]) ** 2
        )

    # ── Interface IC (t-face interior boundary → w_ic) ──
    ifm_ic = (kinds == KIND_INTERFACE)
    if ifm_ic.sum() > 0:
        xt_if = torch.cat(
            [x[ifm_ic], t[ifm_ic]], dim=1
        )
        u_if = model.forward_single_expert(
            expert_idx, xt_if
        )
        comps['interface_ic'] = torch.mean(
            (u_if - h_gt[ifm_ic]) ** 2
        )

    # ── Interface BC (x-face interior boundary → w_bc) ──
    ifm_bc = (kinds == KIND_INTERFACE_BC)
    if ifm_bc.sum() > 0:
        xt_if_bc = torch.cat(
            [x[ifm_bc], t[ifm_bc]], dim=1
        )
        u_if_bc = model.forward_single_expert(
            expert_idx, xt_if_bc
        )
        comps['interface_bc'] = torch.mean(
            (u_if_bc - h_gt[ifm_bc]) ** 2
        )

    # ── BC true: Dirichlet (for non-periodic problems only) ──
    # Fix 2: Allen-Cahn uses periodic BC pairing at batch level, skip here
    bc_mask = (kinds == KIND_BC_TRUE)
    if (not is_allen_cahn) and bc_mask.sum() > 0:
        xt_bc = torch.cat(
            [x[bc_mask], t[bc_mask]], dim=1
        )
        u_bc = model.forward_single_expert(
            expert_idx, xt_bc
        )
        comps['bc'] = torch.mean(
            (u_bc - h_gt[bc_mask]) ** 2
        )

    comps['total'] = (
        w_res * comps['residual']
        + w_ic * (comps['ic'] + comps['interface_ic'])
        + w_bc * (comps['interface_bc'] + comps['bc'])
    )
    return comps


def _compute_periodic_bc_loss(
    model, x, t, expert_ids, kinds, bc_face_ids, deriv_fn, device,
):
    """Compute periodic BC loss for Allen-Cahn via cross-expert pairing.

    Pairs left/right boundary points by sorting on t-value and penalizes
    (u_left - u_right)² + (∂u/∂x_left - ∂u/∂x_right)².
    """
    bc_mask = (kinds == KIND_BC_TRUE)
    if bc_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    x_bc = x[bc_mask]
    t_bc = t[bc_mask]
    eid_bc = expert_ids[bc_mask]
    fid_bc = bc_face_ids[bc_mask]
    
    # Group by dimension (face_id // 2)
    dims = fid_bc // 2
    sides = fid_bc % 2
    
    unique_dims = dims.unique().tolist()
    total_bc_loss = torch.tensor(0.0, device=device)
    n_pairs = 0
    
    for d in unique_dims:
        d_mask = (dims == d)
        x_d = x_bc[d_mask]
        t_d = t_bc[d_mask]
        eid_d = eid_bc[d_mask]
        side_d = sides[d_mask]
        
        # Separate left (side=0) and right (side=1)
        left_mask = (side_d == 0)
        right_mask = (side_d == 1)
        
        n_left = left_mask.sum().item()
        n_right = right_mask.sum().item()
        
        if n_left == 0 or n_right == 0:
            continue
        
        # Extract left and right data
        x_left_all = x_d[left_mask]
        t_left_all = t_d[left_mask]
        eid_left_all = eid_d[left_mask]
        
        x_right_all = x_d[right_mask]
        t_right_all = t_d[right_mask]
        eid_right_all = eid_d[right_mask]
        
        # Sort both sides by t-value for matching
        t_left_vals = t_left_all[:, 0]
        t_right_vals = t_right_all[:, 0]
        sort_left = torch.argsort(t_left_vals)
        sort_right = torch.argsort(t_right_vals)
        
        n_match = min(n_left, n_right)
        
        x_left = x_left_all[sort_left[:n_match]]
        t_left = t_left_all[sort_left[:n_match]]
        eid_left = eid_left_all[sort_left[:n_match]]
        
        x_right = x_right_all[sort_right[:n_match]]
        t_right = t_right_all[sort_right[:n_match]]
        eid_right = eid_right_all[sort_right[:n_match]]
        
        # Vectorized evaluation: group by (expert_left, expert_right) pairs
        # Create pair keys for grouping
        pair_keys = eid_left * 10000 + eid_right  # assumes < 10000 experts
        unique_pairs = pair_keys.unique().tolist()
        
        for pair_key in unique_pairs:
            pair_mask = (pair_keys == pair_key)
            eid_l = pair_key // 10000
            eid_r = pair_key % 10000
            
            # Batch all points with this expert pair
            x_l_batch = x_left[pair_mask].clone().detach()
            x_l_batch.requires_grad_(True)
            t_l_batch = t_left[pair_mask].clone().detach()
            x_r_batch = x_right[pair_mask].clone().detach()
            x_r_batch.requires_grad_(True)
            t_r_batch = t_right[pair_mask].clone().detach()
            
            xt_l = torch.cat([x_l_batch, t_l_batch], dim=1)
            xt_r = torch.cat([x_r_batch, t_r_batch], dim=1)
            
            # Single batched forward pass per expert
            u_l = model.forward_single_expert(eid_l, xt_l)[:, 0]
            u_r = model.forward_single_expert(eid_r, xt_r)[:, 0]
            
            # Batched spatial derivative computation
            ux_l = torch.autograd.grad(
                u_l, x_l_batch,
                grad_outputs=torch.ones_like(u_l),
                create_graph=True, retain_graph=True,
            )[0][:, d]
            
            ux_r = torch.autograd.grad(
                u_r, x_r_batch,
                grad_outputs=torch.ones_like(u_r),
                create_graph=True, retain_graph=True,
            )[0][:, d]
            
            # Periodic penalty (vectorized sum)
            total_bc_loss = (
                total_bc_loss
                + torch.sum((u_l - u_r) ** 2)
                + torch.sum((ux_l - ux_r) ** 2)
            )
            n_pairs += pair_mask.sum().item()
    
    if n_pairs > 0:
        return total_bc_loss / n_pairs
    return torch.tensor(0.0, device=device)


def _record(history, expert_idx, comps):
    if expert_idx not in history:
        history[expert_idx] = {
            k: [] for k in [
                'residual', 'ic', 'interface_ic',
                'interface_bc', 'bc', 'total', 'continuity',
            ]
        }
    for k in history[expert_idx]:
        if k in comps:
            val = comps[k]
            history[expert_idx][k].append(
                val.item() if torch.is_tensor(val) else val
            )


def _record_continuity(history, expert_idx, cont_val):
    """Record continuity loss for an expert (separate from main _record)."""
    if expert_idx not in history:
        history[expert_idx] = {
            k: [] for k in [
                'residual', 'ic', 'interface_ic',
                'interface_bc', 'bc', 'total', 'continuity',
            ]
        }
    # Append to continuity; if list is shorter, pad with 0
    cont_list = history[expert_idx]['continuity']
    while len(cont_list) < len(history[expert_idx]['total']) - 1:
        cont_list.append(0.0)
    cont_list.append(cont_val if not torch.is_tensor(cont_val) else cont_val.item())


def _compute_continuity_loss(
    model, x, t, expert_ids, kinds, cont_neighbors, cont_dims,
    deriv_fn, pde_params, device, problem,
):
    """Compute continuity loss on shared interior faces between neighbors.
    
    For each pair (a, b) of face-neighbor experts, enforces agreement of:
    - Value: u_a = u_b
    - First derivative: ∂u_a/∂d = ∂u_b/∂d (where d is face-normal dim)
    - Second derivative: ∂²u_a/∂d² = ∂²u_b/∂d² (for PDE order >= 2)

    Returns:
        cont_loss: total continuity loss (scalar)
        cont_per_expert: dict mapping expert_idx -> continuity loss contribution
    """
    cont_mask = (kinds == KIND_CONTINUITY)
    if cont_mask.sum() == 0:
        return torch.tensor(0.0, device=device), {}
    
    x_cont = x[cont_mask]
    t_cont = t[cont_mask]
    eid_cont = expert_ids[cont_mask]
    neighbor_cont = cont_neighbors[cont_mask]
    dim_cont = cont_dims[cont_mask]
    
    # PDE order determines how many derivatives to match
    # Allen-Cahn has second-order spatial derivatives
    pde_order = 2 if problem in ('allen_cahn', 'burgers1d', 'kdv') else 1
    if problem == 'ks':
        pde_order = 4  # KS has 4th order
    
    total_loss = torch.tensor(0.0, device=device)
    cont_per_expert = {}
    n_pairs = 0
    
    # Group by (expert_a, expert_b, face_dim) for batched evaluation
    # Create composite key: a * 1e8 + b * 1e4 + d
    pair_keys = eid_cont * 100000000 + neighbor_cont * 10000 + dim_cont
    unique_keys = pair_keys.unique().tolist()
    
    for key in unique_keys:
        key_mask = (pair_keys == key)
        eidx_a = int(key // 100000000)
        eidx_b = int((key % 100000000) // 10000)
        face_dim = int(key % 10000)
        
        # Get points for this pair
        x_pair = x_cont[key_mask].clone().detach().requires_grad_(True)
        t_pair = t_cont[key_mask].clone().detach().requires_grad_(True)
        xt_pair = torch.cat([x_pair, t_pair], dim=1)
        
        n_pts = x_pair.shape[0]
        if n_pts == 0:
            continue
        
        # Evaluate both experts at same coordinates
        u_a = model.forward_single_expert(eidx_a, xt_pair)[:, 0]
        u_b = model.forward_single_expert(eidx_b, xt_pair)[:, 0]
        
        # Value mismatch
        pair_loss = torch.sum((u_a - u_b) ** 2)
        
        # First derivative mismatch (along face-normal dimension)
        if pde_order >= 1:
            if face_dim < x_pair.shape[1]:  # spatial dimension
                # Derivative w.r.t. x (spatial)
                du_a_dx = torch.autograd.grad(
                    u_a, x_pair,
                    grad_outputs=torch.ones_like(u_a),
                    create_graph=True, retain_graph=True,
                )[0][:, face_dim]
                du_b_dx = torch.autograd.grad(
                    u_b, x_pair,
                    grad_outputs=torch.ones_like(u_b),
                    create_graph=True, retain_graph=True,
                )[0][:, face_dim]
            else:
                # Derivative w.r.t. t (temporal dimension)
                du_a_dx = torch.autograd.grad(
                    u_a, t_pair,
                    grad_outputs=torch.ones_like(u_a),
                    create_graph=True, retain_graph=True,
                )[0][:, 0]
                du_b_dx = torch.autograd.grad(
                    u_b, t_pair,
                    grad_outputs=torch.ones_like(u_b),
                    create_graph=True, retain_graph=True,
                )[0][:, 0]
            pair_loss = pair_loss + torch.sum((du_a_dx - du_b_dx) ** 2)
        
        # Second derivative mismatch
        if pde_order >= 2:
            if face_dim < x_pair.shape[1]:
                d2u_a_dx2 = torch.autograd.grad(
                    du_a_dx, x_pair,
                    grad_outputs=torch.ones_like(du_a_dx),
                    create_graph=True, retain_graph=True,
                )[0][:, face_dim]
                d2u_b_dx2 = torch.autograd.grad(
                    du_b_dx, x_pair,
                    grad_outputs=torch.ones_like(du_b_dx),
                    create_graph=True, retain_graph=True,
                )[0][:, face_dim]
            else:
                d2u_a_dx2 = torch.autograd.grad(
                    du_a_dx, t_pair,
                    grad_outputs=torch.ones_like(du_a_dx),
                    create_graph=True, retain_graph=True,
                )[0][:, 0]
                d2u_b_dx2 = torch.autograd.grad(
                    du_b_dx, t_pair,
                    grad_outputs=torch.ones_like(du_b_dx),
                    create_graph=True, retain_graph=True,
                )[0][:, 0]
            pair_loss = pair_loss + torch.sum((d2u_a_dx2 - d2u_b_dx2) ** 2)
        
        total_loss = total_loss + pair_loss
        n_pairs += n_pts
        
        # Track per-expert contribution (split equally between a and b)
        pair_loss_val = pair_loss.item() / 2 if n_pts > 0 else 0.0
        cont_per_expert[eidx_a] = cont_per_expert.get(eidx_a, 0.0) + pair_loss_val
        cont_per_expert[eidx_b] = cont_per_expert.get(eidx_b, 0.0) + pair_loss_val
    
    if n_pairs > 0:
        total_loss = total_loss / n_pairs
        # Normalize per-expert values
        for eidx in cont_per_expert:
            cont_per_expert[eidx] /= n_pairs
    
    return total_loss, cont_per_expert


def _import_pde_helpers(problem: str):
    """Import problem-specific PDE residual + derivatives."""
    mod = importlib.import_module(f'losses.{problem}_loss')
    return mod.pde_residual, mod.compute_derivatives


def _get_pde_params(problem: str, pc: Dict) -> Dict:
    """PDE-specific kwargs for the residual function."""
    if problem == 'allen_cahn':
        return {'D': pc['D']}
    if problem == 'burgers1d':
        return {'nu': pc['nu']}
    if problem == 'kdv':
        return {'mu': pc['mu']}
    if problem == 'ks':
        return {
            'alpha': pc.get('alpha', 1.0),
            'beta': pc.get('beta', 1.0),
            'gamma': pc.get('gamma', 1.0),
        }
    return {}
