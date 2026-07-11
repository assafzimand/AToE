"""Split loss: composed PDE residual + per-expert u0 face guides.

The exact-object scheme — every physics term is evaluated on the blended
PoU composition u_theta (the reported object):
  - PDE residual of u_theta on ONE uniform collocation set over the whole
    domain. On each expert's exclusive zone (where only its window is
    active) this reduces to the expert's own local residual — same value,
    same gradients — while in the collars the gradients reach every
    active expert through the window weights.
  - The global loss's exact IC and BC terms, evaluated on the plain
    training set's IC/BC rows through the composed model, at full,
    un-annealed weight.

Per-expert scaffolding, on the expert's OWN output:
  - u0-guide matching on the interior faces of its EXCLUSIVE box
    Omega_hat (value on the lower-t face; value + d/dx on x-faces for
    problems whose global BC pairs u and u_x), scaled by the annealable
    interface weight. Faces on t_min / the physical boundary carry no
    guides (skipped at data build), and swallowed leaves (empty exclusive
    box) have no guide rows at all — they train purely through the
    composed residual, FBPINN-style.
  - Neighbor-to-neighbor continuity on shared interior faces (optional).

Total loss:
    L = w_res*L_res(u_theta) + w_ic*L_IC(u_theta) + w_bc*L_BC(u_theta)
        + sum_j [ s(e)*( w_ic*L_iface_t_j
                         + w_bc*(L_iface_x_j + L_iface_x_deriv_j) )
                  + w_cont*L_cont_j ]
with s(e) the linear interface anneal (interface_decrease_weight).
"""

import torch
import importlib
from typing import Dict, Callable
from adaptive.subdomain_data import (
    KIND_RESIDUAL, KIND_INTERFACE_T, KIND_INTERFACE_X,
    KIND_CONTINUITY, PERIODIC_PROBLEMS,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


def build_split_loss(
    model,
    cfg: Dict,
    *,
    orig_loss_fn: Callable = None,
    ic_bc_batch: Dict = None,
) -> Callable:
    """Build a split loss for per-expert subdomain training.

    Returns a callable ``loss_fn(model, batch)`` compatible
    with ``_train_segment``.

    ``orig_loss_fn`` serves two roles: fallback for eval batches missing
    the split keys (expert_id, kind), and — via ``ic_bc_batch`` — the
    source of the exact global IC/BC terms evaluated on the blended PoU
    composition. ``ic_bc_batch`` is a plain-format batch holding ONLY the
    training set's IC/BC rows (its residual mask is all-false, so the
    global loss skips the residual term).
    """
    problem = cfg['problem']
    pc = cfg[problem]
    loss_weights = pc['loss_weights']
    w_res = loss_weights['residual']
    w_ic = loss_weights['ic']
    w_bc = loss_weights['bc']
    w_cont = loss_weights.get('continuity', 1.0)

    # Problems whose global BC pairs u AND u_x (periodic set): their x-face
    # guides carry d/dx targets too, matched in the grad-enabled interface
    # pass. Value-only-BC problems (burgers1d) match values only.
    match_x_derivs = problem in PERIODIC_PROBLEMS

    pde_res_fn, deriv_fn = _import_pde_helpers(problem)
    pde_params = _get_pde_params(problem, pc)

    per_expert_history: Dict[int, Dict[str, list]] = {}
    # Composition IC/BC history (recorded once per epoch like the per-
    # expert history; read by the [SplitTerms] eval log).
    global_history: Dict[str, list] = {'ic_comp': [], 'bc_comp': []}
    # Per-epoch residual cache: list of (x, t, r²) tuples (detached CPU tensors).
    # Populated when split_loss_fn._cache_residuals is True; drained by the trainer
    # to produce diagnostic heatmap plots (same as the non-split residual-cache path).
    residual_cache: list = []

    def split_loss_fn(
        model, batch, return_components=False, **kw
    ):
        # Eval batches carry no split keys: fall back to the original loss
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
        h_x_gt = batch.get('h_x_gt', None)  # interface d/dx guide targets
        expert_ids = batch['expert_id']
        kinds = batch['kind']
        cont_neighbors = batch.get('cont_neighbor', None)
        cont_dims = batch.get('cont_dim', None)
        device = x.device
        # D7: interface-weight anneal scale (updated per epoch by the
        # trainer when interface_decrease_weight > 0; 1.0 otherwise).
        interface_scale = split_loss_fn._interface_scale

        # Per-expert history is recorded once per epoch (the trainer arms
        # _record_next at each epoch start). Recording every closure call
        # cost ~7*K GPU syncs per evaluation and polluted the curves with
        # line-search evaluations.
        record_now = split_loss_fn._record_next
        if record_now:
            split_loss_fn._record_next = False

        unique_experts = [e for e in expert_ids.unique().tolist() if e >= 0]
        total_loss = torch.tensor(0.0, device=device)
        all_comps = {}

        # ── PDE residual of the COMPOSITION on the uniform collocation set ──
        # One forward through the blended PoU model; on exclusive zones this
        # equals the owning expert's local residual (neighbor windows are
        # identically zero there), in collars the gradients reach every
        # active expert. Per-owner restricted means are returned detached,
        # for diagnostics only.
        residual_global, residual_by_owner = _compute_composed_residual(
            model, x, t, expert_ids, kinds,
            pde_res_fn, deriv_fn, pde_params, problem,
            residual_cache=(
                residual_cache
                if split_loss_fn._cache_residuals else None
            ),
        )
        if residual_global is not None:
            total_loss = total_loss + w_res * residual_global
        if return_components:
            all_comps['residual_global'] = residual_global

        for eidx in unique_experts:
            emask = (expert_ids == eidx)
            comps = _compute_expert_loss(
                model, eidx,
                x[emask], t[emask], h_gt[emask],
                kinds[emask],
                w_res, w_ic, w_bc, match_x_derivs,
                device,
                residual_diag=residual_by_owner.get(eidx),
                h_x_gt=h_x_gt[emask] if h_x_gt is not None else None,
                interface_scale=interface_scale,
            )
            # Only the guide terms enter the optimized loss per expert; the
            # residual entered once above, through the composition.
            total_loss = total_loss + comps['guides_scaled']
            if record_now:
                _record(per_expert_history, eidx, comps)
            if return_components:
                all_comps[eidx] = comps

        # ── Exact physics on the composition: global IC + BC on u_theta ──
        # The global loss's own IC/BC terms (periodic pairing included),
        # evaluated through the blended PoU composition on the plain
        # training set's IC/BC rows, at FULL weight — the u0 face guides
        # above are the annealable part, this is the ground truth.
        if ic_bc_batch is not None and orig_loss_fn is not None:
            comps_g = orig_loss_fn(model, ic_bc_batch,
                                   return_components=True,
                                   update_causal_state=False)
            ic_comp = comps_g.get('ic', torch.tensor(0.0, device=device))
            bc_comp = comps_g.get('bc', torch.tensor(0.0, device=device))
            total_loss = total_loss + w_ic * ic_comp + w_bc * bc_comp
            if record_now:
                global_history['ic_comp'].append(
                    float(ic_comp.detach()) if torch.is_tensor(ic_comp)
                    else float(ic_comp))
                global_history['bc_comp'].append(
                    float(bc_comp.detach()) if torch.is_tensor(bc_comp)
                    else float(bc_comp))
            if return_components:
                all_comps['composition'] = {'ic_comp': ic_comp,
                                            'bc_comp': bc_comp}

        # ── Continuity loss: neighbor-to-neighbor on shared interior faces ──
        # Skipped entirely (not computed, not recorded/plotted) when its
        # weight is 0 — the derivative matching is the expensive part.
        if cont_neighbors is not None and cont_dims is not None and w_cont != 0:
            cont_loss, cont_per_expert = _compute_continuity_loss(
                model, x, t, expert_ids, kinds,
                cont_neighbors, cont_dims, deriv_fn,
                pde_params, device, problem,
            )
            if logger.isEnabledFor(10):  # DEBUG
                logger.debug(
                    f"[SplitLoss] Continuity contrib: "
                    f"{cont_loss.item():.6e}"
                )
            total_loss = total_loss + w_cont * cont_loss
            # Record continuity per expert (same per-epoch cadence)
            if record_now:
                for eidx, cont_val in cont_per_expert.items():
                    _record_continuity(per_expert_history, eidx, cont_val)

        if return_components:
            return all_comps
        return total_loss

    split_loss_fn._per_expert_history = per_expert_history
    split_loss_fn._global_history = global_history
    split_loss_fn._residual_cache = residual_cache
    split_loss_fn._cache_residuals = False  # trainer sets True when a plot is due
    split_loss_fn._record_next = True  # trainer re-arms once per epoch
    # Interface-weight anneal — the trainer sets _interface_anneal_w
    # from config and updates _interface_scale each epoch.
    split_loss_fn._interface_scale = 1.0
    split_loss_fn._interface_anneal_w = 0.0
    return split_loss_fn


def _compute_composed_residual(
    model, x, t, expert_ids, kinds,
    pde_res_fn, deriv_fn, pde_params, problem,
    residual_cache=None,
):
    """PDE residual of the blended PoU composition on the residual rows.

    One forward through the composed model over all residual points; the
    PDE derivatives (window derivatives included) are computed w.r.t. a
    single ``(xf, tf)`` leaf pair. The returned global mean is the
    optimized term; the per-owner restricted means (owner = the leaf
    whose hard region contains the point, from the ``expert_id`` tag)
    are DETACHED diagnostics for the per-expert curves.

    Returns:
        (residual_global, residual_by_owner) where residual_global is the
        mean r² tensor (None when there are no residual rows) and
        residual_by_owner maps expert_idx -> detached mean r² over its
        owned points.
    """
    rmask = (kinds == KIND_RESIDUAL)
    if rmask.sum() == 0:
        return None, {}

    xf = x[rmask].clone().detach().requires_grad_(True)
    tf = t[rmask].clone().detach().requires_grad_(True)
    xt = torch.cat([xf, tf], dim=1)
    u_all = model(xt)

    if problem == 'schrodinger':
        # Complex field h = u + iv: deriv_fn(u, v, x, t) -> complex derivatives,
        # residual is complex — r² is the squared complex magnitude.
        u_c = u_all[:, 0]
        v_c = u_all[:, 1]
        h_t, h_x, h_xx = deriv_fn(u_c, v_c, xf, tf)
        h = torch.complex(u_c, v_c)
        res = pde_res_fn(h, h_t, h_xx)
        r2 = res.abs() ** 2
    else:
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

    residual_global = r2.mean()

    residual_by_owner = {}
    r2_det = r2.detach()
    owners = expert_ids[rmask]
    for eidx in owners.unique().tolist():
        if eidx < 0:
            continue
        residual_by_owner[int(eidx)] = r2_det[owners == eidx].mean()

    return residual_global, residual_by_owner


def _compute_expert_loss(
    model, expert_idx, x, t, h_gt, kinds,
    w_res, w_ic, w_bc, match_x_derivs, device,
    residual_diag=None,
    h_x_gt=None,
    interface_scale=1.0,
):
    """Per-expert u0-guide terms, on the expert's own output (no PoU).

    The guide faces sit on the interior boundary of the expert's
    exclusive box (data build skips t_min / physical-boundary faces and
    swallowed leaves entirely):
      * ``interface_t`` — lower-t face, value matching, one plain forward;
      * ``interface_x`` — x-faces, value matching, plus d(u_j)/dx matched
        to the minted d(u_0)/dx targets (``interface_x_deriv``) when the
        problem's global BC pairs u and u_x (``match_x_derivs``) — that
        pass is grad-enabled; value-only problems keep it in the plain
        forward.

    ``interface_scale`` multiplies ALL guide terms (the exact IC/BC live
    on the composition, at full weight, outside this function).

    ``residual_diag`` is the DETACHED composed-residual mean over this
    expert's owned points — recorded in ``comps['residual']`` and folded
    into ``comps['total']`` for the per-expert curves, but never added to
    the optimized loss here (the residual enters once, globally, through
    the composition). Only ``comps['guides_scaled']`` carries gradient
    out of this function.
    """
    z = torch.tensor(0.0, device=device)
    comps = {
        'residual': residual_diag if residual_diag is not None else z.clone(),
        'interface_t': z.clone(),
        'interface_x': z.clone(),
        'interface_x_deriv': z.clone(),
    }

    deriv_pass = match_x_derivs and h_x_gt is not None

    # Kinds handled by the single plain (no input-grad) face forward.
    face_kinds = [('interface_t', KIND_INTERFACE_T)]
    if not deriv_pass:
        face_kinds.append(('interface_x', KIND_INTERFACE_X))

    face_mask = torch.zeros_like(kinds, dtype=torch.bool)
    for _, kval in face_kinds:
        face_mask |= (kinds == kval)

    if face_mask.any():
        xt_face = torch.cat([x[face_mask], t[face_mask]], dim=1)
        u_face = model.forward_single_expert(expert_idx, xt_face)
        se_face = (u_face - h_gt[face_mask]) ** 2
        kinds_face = kinds[face_mask]
        for key, kval in face_kinds:
            kmask = (kinds_face == kval)
            if kmask.any():
                comps[key] = torch.mean(se_face[kmask])

    # ── x-faces with value + d/dx guide matching (grad-enabled pass) ──
    if deriv_pass:
        ifm_x = (kinds == KIND_INTERFACE_X)
        if ifm_x.any():
            x_if = x[ifm_x].clone().detach().requires_grad_(True)
            t_if = t[ifm_x].clone().detach()
            u_if = model.forward_single_expert(
                expert_idx, torch.cat([x_if, t_if], dim=1))
            comps['interface_x'] = torch.mean((u_if - h_gt[ifm_x]) ** 2)
            n_out = u_if.shape[1]
            d_loss = torch.tensor(0.0, device=device)
            for c in range(n_out):
                g = torch.autograd.grad(
                    u_if[:, c].sum(), x_if,
                    create_graph=True, retain_graph=True,
                )[0][:, 0]
                d_loss = d_loss + torch.mean(
                    (g - h_x_gt[ifm_x][:, c]) ** 2)
            # Mean over output components (matches the value term's
            # mean-over-all-elements convention).
            comps['interface_x_deriv'] = d_loss / n_out

    comps['guides_scaled'] = interface_scale * (
        w_ic * comps['interface_t']
        + w_bc * (comps['interface_x'] + comps['interface_x_deriv'])
    )
    # Recording-only local total (residual part is detached diagnostics).
    comps['total'] = w_res * comps['residual'] + comps['guides_scaled']
    return comps


def _record(history, expert_idx, comps):
    if expert_idx not in history:
        history[expert_idx] = {
            k: [] for k in [
                'residual', 'interface_t', 'interface_x',
                'interface_x_deriv', 'total', 'continuity',
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
                'residual', 'interface_t', 'interface_x',
                'interface_x_deriv', 'total', 'continuity',
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

    pde_order = _pde_spatial_order(problem)

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

        # Evaluate both experts at the same coordinates (all output components)
        u_a_full = model.forward_single_expert(eidx_a, xt_pair)
        u_b_full = model.forward_single_expert(eidx_b, xt_pair)

        # Face-normal differentiation target: spatial dim or time
        is_spatial = face_dim < x_pair.shape[1]
        wrt = x_pair if is_spatial else t_pair
        col = face_dim if is_spatial else 0

        pair_loss = torch.tensor(0.0, device=device)
        for c in range(u_a_full.shape[1]):
            cur_a = u_a_full[:, c]
            cur_b = u_b_full[:, c]
            # Value mismatch
            pair_loss = pair_loss + torch.sum((cur_a - cur_b) ** 2)
            # Derivative mismatches up to the PDE's spatial order along the
            # face-normal dimension
            for _order in range(pde_order):
                cur_a = torch.autograd.grad(
                    cur_a, wrt,
                    grad_outputs=torch.ones_like(cur_a),
                    create_graph=True, retain_graph=True,
                )[0][:, col]
                cur_b = torch.autograd.grad(
                    cur_b, wrt,
                    grad_outputs=torch.ones_like(cur_b),
                    create_graph=True, retain_graph=True,
                )[0][:, col]
                pair_loss = pair_loss + torch.sum((cur_a - cur_b) ** 2)

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


# Highest spatial derivative order per PDE (drives continuity matching depth).
_PDE_SPATIAL_ORDER = {
    'allen_cahn': 2,
    'burgers1d': 2,
    'kdv': 3,
    'ks': 4,
    'schrodinger': 2,
}


def _pde_spatial_order(problem: str) -> int:
    if problem not in _PDE_SPATIAL_ORDER:
        raise ValueError(
            f"split loss has no PDE order mapping for '{problem}' "
            f"(known: {sorted(_PDE_SPATIAL_ORDER)})")
    return _PDE_SPATIAL_ORDER[problem]


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
            'alpha': pc['alpha'],
            'beta': pc['beta'],
            'gamma': pc['gamma'],
        }
    if problem == 'schrodinger':
        return {}  # fixed coefficients (1/2 and 1)
    raise ValueError(f"split loss has no PDE params mapping for '{problem}'")
