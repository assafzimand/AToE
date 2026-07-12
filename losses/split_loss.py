"""Owner-imitator loss: every term on the expert's OWN raw output u_j.

The PoU composition u_theta is READOUT ONLY — no loss term is ever
evaluated on it. Every expert has exactly one role at every point:
OWNER (physics) on its hard tile, IMITATOR (distillation to the minted
target u*) on its collar. Per expert j:

  L_j = w_res * L_res_j            PDE residual of u_j on X_res_j ⊂ Omega_j
      + w_ic  * L_ic_j             ||u_j - g_ic||^2 on the tile's t=0 rows
      + w_bc  * (L_bc_j + L_per_j) true Dirichlet data (non-periodic) OR
                                   mirror-minted value+d/dx matching
                                   (periodic), on the BC dataset rows
      + L_imit_j                   sum_alpha w_alpha ||D^alpha u_j -
                                   D^alpha u*||^2 on the collar rows
      + w_cont * L_cont_j          optional neighbor continuity (kept,
                                   weight 0 by default)

Total = sum_j L_j; the summands are mutually independent, so all experts
train in parallel under one optimizer. The minted targets u* (values +
axis derivatives, baked constants — see adaptive/subdomain_data.py
mint_targets) carry the owner exchange; lambda lives only in the mint.
"""

import torch
import importlib
from typing import Dict, Callable
from adaptive.subdomain_data import (
    KIND_RESIDUAL, KIND_IC, KIND_BC, KIND_PER, KIND_IMIT,
    KIND_CONTINUITY, PERIODIC_PROBLEMS,  # noqa: F401
    _axis_derivative_stack, imit_order,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


def build_owner_imitator_loss(
    model,
    cfg: Dict,
    *,
    orig_loss_fn: Callable = None,
) -> Callable:
    """Build the owner-imitator loss for per-expert subdomain training.

    Returns a callable ``loss_fn(model, batch)`` compatible with
    ``_train_segment``. ``orig_loss_fn`` is the fallback for batches
    missing the split keys (``expert_id``/``kind``) — the rel-L2 eval
    probe on the plain training set goes through it, so eval stays on
    the composed readout's classic loss.
    """
    problem = cfg['problem']
    pc = cfg[problem]
    loss_weights = pc['loss_weights']
    w_res = loss_weights['residual']
    w_ic = loss_weights['ic']
    w_bc = loss_weights['bc']
    w_cont = loss_weights.get('continuity', 1.0)

    q = imit_order(cfg)
    n_slots = 2 * q + 1
    oi_cfg = (cfg.get('adaptive_pinn', {}).get('owner_imitator', {}) or {})
    imit_weights = oi_cfg.get('imit_weights', None)
    if imit_weights is None:
        imit_weights = [1.0] * n_slots
    imit_weights = [float(w) for w in imit_weights]
    if len(imit_weights) != n_slots:
        raise ValueError(
            f"owner_imitator.imit_weights must have length 2q+1={n_slots} "
            f"(q={q}); got {len(imit_weights)}")

    pde_res_fn, deriv_fn = _import_pde_helpers(problem)
    pde_params = _get_pde_params(problem, pc)

    per_expert_history: Dict[int, Dict[str, list]] = {}
    # Per-epoch residual cache: list of (x, t, r²) tuples (detached CPU
    # tensors), one per expert. Populated when _cache_residuals is True;
    # drained by the trainer for the diagnostic residual heatmap. The
    # per-expert tiles partition the domain, so the union of the cached
    # sets is exactly the full residual draw.
    residual_cache: list = []

    def oi_loss_fn(model, batch, return_components=False, **kw):
        # Eval batches carry no split keys: fall back to the original loss
        if 'expert_id' not in batch or 'kind' not in batch:
            if orig_loss_fn is not None:
                return orig_loss_fn(model, batch,
                                    return_components=return_components,
                                    **kw)
            x = batch['x']
            t = batch['t']
            h_gt = batch['h_gt']
            h_pred = model(torch.cat([x, t], dim=1))
            return torch.mean((h_pred - h_gt) ** 2)

        x = batch['x']
        t = batch['t']
        h_gt = batch['h_gt']
        mint = batch['mint']
        expert_ids = batch['expert_id']
        kinds = batch['kind']
        cont_neighbors = batch.get('cont_neighbor', None)
        cont_dims = batch.get('cont_dim', None)
        device = x.device

        # Per-expert history is recorded once per epoch (the trainer arms
        # _record_next at each epoch start). Recording every closure call
        # cost ~7*K GPU syncs per evaluation and polluted the curves with
        # line-search evaluations.
        record_now = oi_loss_fn._record_next
        if record_now:
            oi_loss_fn._record_next = False

        unique_experts = [e for e in expert_ids.unique().tolist() if e >= 0]
        total_loss = torch.tensor(0.0, device=device)
        all_comps = {}

        for eidx in unique_experts:
            emask = (expert_ids == eidx)
            z = torch.tensor(0.0, device=device)
            comps = {'residual': z.clone(), 'ic': z.clone(),
                     'bc': z.clone(), 'per': z.clone(), 'imit': z.clone()}

            # ── 1. PDE residual of u_j on its OWN tile rows ──
            rmask = emask & (kinds == KIND_RESIDUAL)
            if rmask.any():
                xf = x[rmask].clone().detach().requires_grad_(True)
                tf = t[rmask].clone().detach().requires_grad_(True)
                u_e = model.forward_single_expert(
                    eidx, torch.cat([xf, tf], dim=1))
                if problem == 'schrodinger':
                    # Complex field h = u + iv: residual is complex — r²
                    # is the squared complex magnitude.
                    u_c = u_e[:, 0]
                    v_c = u_e[:, 1]
                    h_t, h_x, h_xx = deriv_fn(u_c, v_c, xf, tf)
                    h = torch.complex(u_c, v_c)
                    res = pde_res_fn(h, h_t, h_xx)
                    r2 = res.abs() ** 2
                else:
                    hf = u_e[:, 0]
                    derivs = deriv_fn(hf, xf, tf)
                    res = pde_res_fn(hf, *derivs, **pde_params)
                    r2 = res ** 2
                comps['residual'] = r2.mean()
                if oi_loss_fn._cache_residuals:
                    residual_cache.append((
                        xf.detach().cpu(),
                        tf.detach().cpu(),
                        r2.detach().cpu(),
                    ))

            # ── 2 + 3. True IC / BC data, one plain forward per expert ──
            imask = emask & (kinds == KIND_IC)
            bmask = emask & (kinds == KIND_BC)
            face_mask = imask | bmask
            if face_mask.any():
                xt_face = torch.cat([x[face_mask], t[face_mask]], dim=1)
                u_face = model.forward_single_expert(eidx, xt_face)
                se_face = (u_face - h_gt[face_mask]) ** 2
                kinds_face = kinds[face_mask]
                if imask.any():
                    comps['ic'] = torch.mean(se_face[kinds_face == KIND_IC])
                if bmask.any():
                    comps['bc'] = torch.mean(se_face[kinds_face == KIND_BC])

            # ── 4. Periodic BC via mirror-minted targets (value + d/dx) ──
            pmask = emask & (kinds == KIND_PER)
            if pmask.any():
                x_p = x[pmask].clone().detach().requires_grad_(True)
                t_p = t[pmask].clone().detach()
                u_p = model.forward_single_expert(
                    eidx, torch.cat([x_p, t_p], dim=1))
                mint_p = mint[pmask]
                n_out = u_p.shape[1]
                val_mse = torch.mean((u_p - mint_p[:, 0, :]) ** 2)
                d_loss = torch.tensor(0.0, device=device)
                for c in range(n_out):
                    g = torch.autograd.grad(
                        u_p[:, c].sum(), x_p,
                        create_graph=True, retain_graph=True,
                    )[0][:, 0]
                    d_loss = d_loss + torch.mean(
                        (g - mint_p[:, 1, c]) ** 2)
                comps['per'] = val_mse + d_loss / n_out

            # ── 5. Imitation on the collar (Sobolev matching to u*) ──
            mmask = emask & (kinds == KIND_IMIT)
            if mmask.any():
                x_m = x[mmask].clone().detach().requires_grad_(True)
                t_m = t[mmask].clone().detach().requires_grad_(True)
                u_m = model.forward_single_expert(
                    eidx, torch.cat([x_m, t_m], dim=1))
                stack = _axis_derivative_stack(u_m, x_m, t_m, q)
                se = (stack - mint[mmask]) ** 2   # (n, 2q+1, C)
                imit_loss = torch.tensor(0.0, device=device)
                for a in range(n_slots):
                    if imit_weights[a] != 0.0:
                        imit_loss = imit_loss + (
                            imit_weights[a] * se[:, a, :].mean())
                comps['imit'] = imit_loss

            comps['total'] = (w_res * comps['residual']
                              + w_ic * comps['ic']
                              + w_bc * (comps['bc'] + comps['per'])
                              + comps['imit'])
            total_loss = total_loss + comps['total']
            if record_now:
                _record(per_expert_history, eidx, comps)
            if return_components:
                for k, v in comps.items():
                    all_comps[f'e{eidx}_{k}'] = v

        # ── Continuity loss: neighbor-to-neighbor on shared interior faces ──
        # Skipped entirely (not computed, not recorded/plotted) when its
        # weight is 0 — the derivative matching is the expensive part.
        if cont_neighbors is not None and cont_dims is not None and w_cont != 0:
            cont_loss, cont_per_expert = _compute_continuity_loss(
                model, x, t, expert_ids, kinds,
                cont_neighbors, cont_dims, deriv_fn,
                pde_params, device, problem,
            )
            total_loss = total_loss + w_cont * cont_loss
            if record_now:
                for eidx, cont_val in cont_per_expert.items():
                    _record_continuity(per_expert_history, eidx, cont_val)
            if return_components:
                all_comps['continuity'] = cont_loss

        if return_components:
            return all_comps
        return total_loss

    oi_loss_fn._per_expert_history = per_expert_history
    oi_loss_fn._residual_cache = residual_cache
    oi_loss_fn._cache_residuals = False  # trainer sets True when a plot is due
    oi_loss_fn._record_next = True  # trainer re-arms once per epoch
    oi_loss_fn._mint_lambda = 1.0   # updated at each refresh (logging only)
    return oi_loss_fn


_TERM_KEYS = ['residual', 'ic', 'bc', 'per', 'imit', 'total', 'continuity']


def _record(history, expert_idx, comps):
    if expert_idx not in history:
        history[expert_idx] = {k: [] for k in _TERM_KEYS}
    for k in history[expert_idx]:
        if k in comps:
            val = comps[k]
            history[expert_idx][k].append(
                val.item() if torch.is_tensor(val) else val
            )


def _record_continuity(history, expert_idx, cont_val):
    """Record continuity loss for an expert (separate from main _record)."""
    if expert_idx not in history:
        history[expert_idx] = {k: [] for k in _TERM_KEYS}
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


# Highest spatial derivative order per PDE (drives continuity matching depth
# and is the reference point m for the imitation order q <= m-1).
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
            f"owner-imitator loss has no PDE order mapping for '{problem}' "
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
    raise ValueError(f"owner-imitator loss has no PDE params mapping for "
                     f"'{problem}'")
