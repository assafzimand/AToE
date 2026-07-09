"""Per-expert subdomain data builder for split expert training.

Builds a combined training dataset where each point is tagged with the
owning expert and its kind. Every outer face of an expert's inflated
training box is a u0-GUIDED INTERFACE — including faces that land on the
physical boundary or on t_min. The exact IC/BC physics is NOT enforced
per expert; the split loss applies the true global IC/BC terms once, on
the blended PoU composition (the reported object). The u0 face targets
act as an annealable guide (interface_decrease_weight).

Used by the split-loss training path for AToE-Leaves.
"""

import torch
from typing import Dict, List
from adaptive.indicators import RegionDescriptor, inflated_bounds  # noqa: F401
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Problems whose GLOBAL BC pairs value AND d/dx (periodic). Their x-face
# interfaces mimic the same structure: u0 value AND d(u0)/dx targets.
# Value-only-BC problems (burgers1d Dirichlet) get value-only interfaces.
PERIODIC_PROBLEMS = frozenset({'allen_cahn', 'kdv', 'ks', 'schrodinger'})

# Integer codes stored in the ``kind`` tensor.
# (Codes 1 and 4 were the removed per-expert true-IC/true-BC kinds; the
# numbering of the survivors is kept stable.)
KIND_RESIDUAL = 0
KIND_INTERFACE_T = 2    # lower-t face of the training box (weighted by w_ic)
KIND_INTERFACE_X = 3    # x-faces of the training box (weighted by w_bc)
KIND_CONTINUITY = 5     # continuity points on shared interior faces (neighbor-to-neighbor)

KIND_NAMES = {
    KIND_RESIDUAL: 'residual',
    KIND_INTERFACE_T: 'interface_t',
    KIND_INTERFACE_X: 'interface_x',
    KIND_CONTINUITY: 'continuity',
}

# Tolerance for face-neighbor adjacency checks
ADJACENCY_TOL = 1e-8


def _face_counts(cfg: Dict) -> tuple:
    """Resolve (n_ic_per_face, n_bc_per_face) from the sampling config."""
    sampling = cfg.get('sampling', {})
    n_res_total = sampling.get('n_residual_train', 10000)
    n_ic_per_face = sampling.get('n_initial_train')
    if n_ic_per_face is None:
        ic_ratio = sampling.get('initial_train_ratio', 0.026)
        n_ic_per_face = round(n_res_total * ic_ratio)
    n_bc_per_face = sampling.get('n_boundary_train')
    if n_bc_per_face is None:
        bc_ratio = sampling.get('boundary_train_ratio', 0.026)
        n_bc_per_face = round(n_res_total * bc_ratio)
    return max(1, int(n_ic_per_face)), max(1, int(n_bc_per_face))


def _domain_box(pc: Dict) -> tuple:
    """(lower, upper) lists of the domain box over spatial dims + time."""
    spatial_dim = pc['spatial_dim']
    lo = ([pc['spatial_domain'][d][0] for d in range(spatial_dim)]
          + [pc['temporal_domain'][0]])
    hi = ([pc['spatial_domain'][d][1] for d in range(spatial_dim)]
          + [pc['temporal_domain'][1]])
    return lo, hi


def sample_subdomain_residuals(
    new_expert_indices: List[int],
    regions,
    cfg: Dict,
    device: torch.device,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Draw fresh residual collocation points tagged per owning expert.

    A global uniform draw over the domain is filtered into each expert's
    INFLATED training box (region + window collar, clipped to the domain;
    D2), so every expert trains on the full support of its blending window.
    Collar points shared by several windows are assigned to every covering
    expert. This is the only part of the split dataset that changes on
    resample.
    """
    torch.manual_seed(seed)

    problem = cfg['problem']
    pc = cfg[problem]
    spatial_dim = pc['spatial_dim']
    spatial_domain = pc['spatial_domain']
    t_min_global, t_max_global = pc['temporal_domain']
    output_dim = pc['output_dim']
    n_res_total = cfg.get('sampling', {}).get('n_residual_train', 10000)
    sigma_fraction = cfg['adaptive_pinn']['sigma_fraction']
    g_lo, g_hi = _domain_box(pc)

    x_g = torch.zeros(n_res_total, spatial_dim, device=device)
    t_g = torch.zeros(n_res_total, 1, device=device)
    for d in range(spatial_dim):
        lo, hi = spatial_domain[d]
        x_g[:, d] = torch.rand(n_res_total, device=device) * (hi - lo) + lo
    t_g[:, 0] = (torch.rand(n_res_total, device=device)
                 * (t_max_global - t_min_global) + t_min_global)

    xs, ts, gs, eids, ks = [], [], [], [], []
    for eidx in new_expert_indices:
        region = regions[eidx]
        bl, bu = inflated_bounds(region, sigma_fraction, g_lo, g_hi)

        mask = torch.ones(n_res_total, dtype=torch.bool, device=device)
        for d in range(spatial_dim):
            mask &= (x_g[:, d] >= bl[d]) & (x_g[:, d] <= bu[d])
        mask &= (t_g[:, 0] >= bl[spatial_dim]) & (t_g[:, 0] <= bu[spatial_dim])

        n = mask.sum().item()
        if n > 0:
            xs.append(x_g[mask])
            ts.append(t_g[mask])
            gs.append(torch.zeros(n, output_dim, device=device))
            eids.append(torch.full((n,), eidx, dtype=torch.long, device=device))
            ks.append(torch.full((n,), KIND_RESIDUAL, dtype=torch.long, device=device))

    if not xs:
        return _empty(spatial_dim, output_dim, device)

    n_pts = sum(x.shape[0] for x in xs)
    return {
        'x': torch.cat(xs, dim=0),
        't': torch.cat(ts, dim=0),
        'h_gt': torch.cat(gs, dim=0),
        'h_x_gt': torch.zeros(n_pts, output_dim, device=device),
        'expert_id': torch.cat(eids, dim=0),
        'kind': torch.cat(ks, dim=0),
        'cont_neighbor': torch.full((n_pts,), -1, dtype=torch.long, device=device),
        'cont_dim': torch.full((n_pts,), -1, dtype=torch.long, device=device),
    }


def build_subdomain_static(
    model: torch.nn.Module,
    new_expert_indices: List[int],
    regions,
    cfg: Dict,
    device: torch.device,
    seed: int = 0,
    interface_model: torch.nn.Module = None,
) -> Dict[str, torch.Tensor]:
    """Build the static (non-residual) part of the split dataset.

    Every outer face of each expert's inflated training box becomes a
    u0-guided interface (KIND_INTERFACE_T on the lower-t face,
    KIND_INTERFACE_X on both x-faces) — the physical boundary and t_min get
    no special treatment here, because the split loss enforces the true
    global IC/BC once, on the blended composition. Faces, minted targets,
    and the O(K²) continuity-neighbor pairs depend only on the regions and
    the frozen snapshot — both constant within a training segment — so this
    is built ONCE per segment and reused across resamples.
    """
    torch.manual_seed(seed)

    problem = cfg['problem']
    pc = cfg[problem]
    spatial_dim = pc['spatial_dim']
    spatial_domain = pc['spatial_domain']
    temporal_domain = pc['temporal_domain']
    output_dim = pc['output_dim']

    n_t_face, n_x_face = _face_counts(cfg)

    if len(new_expert_indices) == 0:
        return _empty(spatial_dim, output_dim, device)

    xs, ts, gs, eids, ks = [], [], [], [], []

    # ── Interface faces per expert, on the INFLATED training box ──
    # (D2: the expert trains on region + window collar; its faces sit on
    # the outer edge of that box, clipped to the domain.)
    sigma_fraction = cfg['adaptive_pinn']['sigma_fraction']
    g_lo, g_hi = _domain_box(pc)
    for eidx in new_expert_indices:
        region = regions[eidx]
        infl_bl, infl_bu = inflated_bounds(region, sigma_fraction, g_lo, g_hi)
        _add_t_interface_face(
            eidx, region, infl_bl, infl_bu, spatial_dim, temporal_domain[0],
            n_t_face, output_dim, device, xs, ts, gs, eids, ks,
        )
        _add_x_interface_faces(
            eidx, region, infl_bl, infl_bu, spatial_dim, spatial_domain,
            n_x_face, output_dim, device, xs, ts, gs, eids, ks,
        )

    # ── Continuity faces: neighbor-to-neighbor on shared interior faces ──
    cont_xs, cont_ts, cont_gs, cont_eids, cont_ks = [], [], [], [], []
    cont_neighbors, cont_dims = [], []
    _add_continuity_faces(
        new_expert_indices, regions, spatial_dim, spatial_domain,
        temporal_domain, n_x_face, output_dim, device,
        cont_xs, cont_ts, cont_gs, cont_eids, cont_ks,
        cont_neighbors, cont_dims,
    )

    x_cat = torch.cat(xs, dim=0)
    t_cat = torch.cat(ts, dim=0)
    h_gt_cat = torch.cat(gs, dim=0)
    eid_cat = torch.cat(eids, dim=0)
    kind_cat = torch.cat(ks, dim=0)

    n_main = x_cat.shape[0]
    cont_neighbor_main = torch.full((n_main,), -1, dtype=torch.long, device=device)
    cont_dim_main = torch.full((n_main,), -1, dtype=torch.long, device=device)

    # ── Mint u0-guide targets from the frozen field ──
    # interface_model overrides which frozen field defines the face targets.
    # For AToE-Leaves it is the base (root), so targets are good root
    # predictions even when experts differ in shape from the base; None
    # falls back to `model` (composed snapshot).
    iface_src = interface_model if interface_model is not None else model
    iface_t_mask = (kind_cat == KIND_INTERFACE_T)
    if iface_t_mask.sum() > 0:
        with torch.no_grad():
            xt_if = torch.cat([x_cat[iface_t_mask], t_cat[iface_t_mask]], dim=1)
            h_gt_cat[iface_t_mask] = iface_src(xt_if)

    iface_x_mask = (kind_cat == KIND_INTERFACE_X)
    if iface_x_mask.sum() > 0:
        with torch.no_grad():
            xt_if_x = torch.cat([x_cat[iface_x_mask], t_cat[iface_x_mask]], dim=1)
            h_gt_cat[iface_x_mask] = iface_src(xt_if_x)
    _src = 'base(root)' if interface_model is not None else 'composed'
    logger.info(f"[SplitData] interface guide targets minted from {_src} model "
                f"(n_t_face={int(iface_t_mask.sum())}, "
                f"n_x_face={int(iface_x_mask.sum())})")

    # ── Derivative targets on x-interfaces ──
    # For problems whose global BC pairs u AND u_x (periodic set), x-face
    # guides mimic the same structure: d(u0)/dx minted at the face points
    # (face normal = spatial dim 0; the pipeline is 1D-spatial). Value-only
    # BC problems (burgers1d Dirichlet) keep value-only guides.
    h_x_gt_cat = torch.zeros_like(h_gt_cat)
    if problem in PERIODIC_PROBLEMS and iface_x_mask.sum() > 0:
        x_if = x_cat[iface_x_mask].clone().detach().requires_grad_(True)
        t_if = t_cat[iface_x_mask].clone().detach()
        u_if = iface_src(torch.cat([x_if, t_if], dim=1))
        n_out = u_if.shape[1]
        deriv_cols = []
        for c in range(n_out):
            g = torch.autograd.grad(
                u_if[:, c].sum(), x_if,
                retain_graph=(c < n_out - 1),
            )[0][:, 0]
            deriv_cols.append(g)
        h_x_gt_cat[iface_x_mask] = torch.stack(deriv_cols, dim=1).detach()
        _norms = ', '.join(
            f'comp{c}={h_x_gt_cat[iface_x_mask][:, c].norm().item():.4e}'
            for c in range(n_out))
        logger.info(f"[SplitData] interface u_x guide targets minted from "
                    f"{_src}: n={int(iface_x_mask.sum())} points, norms "
                    f"[{_norms}] (u/u_x-BC problem '{problem}')")
    elif problem not in PERIODIC_PROBLEMS:
        logger.info(f"[SplitData] value-only interface guides "
                    f"(value-only-BC problem '{problem}')")

    # ── Append continuity data (if any) ──
    if cont_xs:
        cont_x_cat = torch.cat(cont_xs, dim=0)
        n_cont = cont_x_cat.shape[0]
        x_cat = torch.cat([x_cat, cont_x_cat], dim=0)
        t_cat = torch.cat([t_cat, torch.cat(cont_ts, dim=0)], dim=0)
        h_gt_cat = torch.cat([h_gt_cat, torch.cat(cont_gs, dim=0)], dim=0)
        h_x_gt_cat = torch.cat([
            h_x_gt_cat,
            torch.zeros(n_cont, h_x_gt_cat.shape[1], device=device),
        ], dim=0)
        eid_cat = torch.cat([eid_cat, torch.cat(cont_eids, dim=0)], dim=0)
        kind_cat = torch.cat([kind_cat, torch.cat(cont_ks, dim=0)], dim=0)
        cont_neighbor_main = torch.cat(
            [cont_neighbor_main, torch.cat(cont_neighbors, dim=0)], dim=0)
        cont_dim_main = torch.cat(
            [cont_dim_main, torch.cat(cont_dims, dim=0)], dim=0)
        logger.info(f"[SplitData] continuity points: {n_cont}")

    return {
        'x': x_cat,
        't': t_cat,
        'h_gt': h_gt_cat,
        'h_x_gt': h_x_gt_cat,
        'expert_id': eid_cat,
        'kind': kind_cat,
        'cont_neighbor': cont_neighbor_main,
        'cont_dim': cont_dim_main,
    }


def build_subdomain_data(
    model: torch.nn.Module,
    new_expert_indices: List[int],
    regions,
    cfg: Dict,
    device: torch.device,
    seed: int = 0,
    interface_model: torch.nn.Module = None,
    static: Dict[str, torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Build per-expert dataset for split-loss training.

    Returns dict with keys ``x``, ``t``, ``h_gt``, ``h_x_gt``,
    ``expert_id``, ``kind``, ``cont_neighbor``, ``cont_dim``.

    Args:
        static: Optional pre-built static part (from
            :func:`build_subdomain_static`). Pass the cached value on
            resample so only residual interiors are redrawn; when None the
            static part is built here.
    """
    problem = cfg['problem']
    pc = cfg[problem]
    if len(new_expert_indices) == 0:
        return _empty(pc['spatial_dim'], pc['output_dim'], device)

    residuals = sample_subdomain_residuals(
        new_expert_indices, regions, cfg, device, seed=seed)

    if static is None:
        static = build_subdomain_static(
            model, new_expert_indices, regions, cfg, device,
            seed=seed, interface_model=interface_model)

    return {k: torch.cat([residuals[k], static[k]], dim=0) for k in residuals}


# ── Helpers ─────────────────────────────────────────────


def _empty(spatial_dim, output_dim, device):
    return {
        'x': torch.zeros(0, spatial_dim, device=device),
        't': torch.zeros(0, 1, device=device),
        'h_gt': torch.zeros(0, output_dim, device=device),
        'h_x_gt': torch.zeros(0, output_dim, device=device),
        'expert_id': torch.zeros(
            0, dtype=torch.long, device=device
        ),
        'kind': torch.zeros(
            0, dtype=torch.long, device=device
        ),
        'cont_neighbor': torch.zeros(
            0, dtype=torch.long, device=device
        ),
        'cont_dim': torch.zeros(
            0, dtype=torch.long, device=device
        ),
    }


def _is_global_boundary(val, global_lo, global_hi, tol=1e-8):
    return (abs(val - global_lo) < tol
            or abs(val - global_hi) < tol)


def _add_t_interface_face(
    eidx, region, infl_bl, infl_bu, spatial_dim, t_min_global,
    n_pts, output_dim, device, xs, ts, gs, eids, ks,
):
    """Add the lower-t interface face of the INFLATED training box (D2).

    Always a u0-guided interface (KIND_INTERFACE_T) — a face that lands on
    the domain's t_min gets no special treatment: u0 was trained on the
    true IC there, so the guide is already near-exact, and the exact IC is
    enforced by the split loss's composition term. Targets are placeholder
    zeros here; minted from the frozen root by the caller.
    """
    t_face = infl_bl[spatial_dim]
    t_face_hard = region.bounds_lower[spatial_dim]
    on_t_min = abs(t_face - t_min_global) < 1e-8

    x_face = torch.zeros(n_pts, spatial_dim, device=device)
    for d in range(spatial_dim):
        lo, hi = infl_bl[d], infl_bu[d]
        x_face[:, d] = (torch.rand(n_pts, device=device)
                        * (hi - lo) + lo)
    t_face_pts = torch.full((n_pts, 1), t_face, device=device)

    logger.info(
        f"[SplitData] expert={eidx} face=t-lower kind=interface_t "
        f"at t={t_face:.6f} (hard {t_face_hard:.6f}, "
        f"on_t_min={on_t_min}) "
        f"n={n_pts} x_range=[{infl_bl[0]:.6f}, {infl_bu[0]:.6f}]")

    xs.append(x_face)
    ts.append(t_face_pts)
    gs.append(torch.zeros(n_pts, output_dim, device=device))
    eids.append(torch.full(
        (n_pts,), eidx, dtype=torch.long, device=device
    ))
    ks.append(torch.full(
        (n_pts,), KIND_INTERFACE_T, dtype=torch.long, device=device
    ))


def _add_x_interface_faces(
    eidx, region, infl_bl, infl_bu, spatial_dim, spatial_domain,
    n_pts, output_dim, device, xs, ts, gs, eids, ks,
):
    """Add both x-interface faces of the INFLATED training box (D2).

    Always u0-guided interfaces (KIND_INTERFACE_X) — faces clipped onto the
    physical boundary get no special treatment: the exact BC (periodic
    pairing / Dirichlet) is enforced by the split loss's composition term,
    and the u0 guide there anneals away with interface_decrease_weight.
    Value targets are placeholder zeros here; minted from the frozen root
    (plus d/dx targets for u/u_x-BC problems) by the caller.
    """
    t_lo_infl = infl_bl[spatial_dim]
    t_hi_infl = infl_bu[spatial_dim]

    for d in range(spatial_dim):
        g_lo_d, g_hi_d = spatial_domain[d]
        for side, (face_val, hard_val) in enumerate(
                [(infl_bl[d], region.bounds_lower[d]),
                 (infl_bu[d], region.bounds_upper[d])]):
            on_boundary = _is_global_boundary(face_val, g_lo_d, g_hi_d)
            logger.info(
                f"[SplitData] expert={eidx} "
                f"face=x{'-lower' if side == 0 else '-upper'} "
                f"kind=interface_x at x={face_val:.6f} "
                f"(hard {hard_val:.6f}, on_domain_boundary={on_boundary}) "
                f"n={n_pts} t_range=[{t_lo_infl:.6f}, {t_hi_infl:.6f}]")

            t_face_pts = (
                torch.rand(n_pts, 1, device=device)
                * (t_hi_infl - t_lo_infl) + t_lo_infl
            )
            x_face = torch.zeros(n_pts, spatial_dim, device=device)
            x_face[:, d] = face_val
            for d2 in range(spatial_dim):
                if d2 != d:
                    lo2, hi2 = infl_bl[d2], infl_bu[d2]
                    x_face[:, d2] = (
                        torch.rand(n_pts, device=device)
                        * (hi2 - lo2) + lo2
                    )

            xs.append(x_face)
            ts.append(t_face_pts)
            gs.append(torch.zeros(n_pts, output_dim, device=device))
            eids.append(torch.full(
                (n_pts,), eidx, dtype=torch.long, device=device
            ))
            ks.append(torch.full(
                (n_pts,), KIND_INTERFACE_X, dtype=torch.long, device=device
            ))


def _are_face_neighbors(region_a, region_b, n_dims, tol=ADJACENCY_TOL):
    """Check if two regions are face-neighbors along some dimension.
    
    Two regions are face-neighbors along dimension d if:
    1. They touch in d: A.upper[d] ~= B.lower[d] or B.upper[d] ~= A.lower[d]
    2. They overlap in all other dimensions
    
    Returns (is_neighbor, face_dim, face_val, overlap_lo, overlap_hi) where:
    - is_neighbor: bool
    - face_dim: the dimension along which they touch (-1 if not neighbors)
    - face_val: the coordinate value of the shared face
    - overlap_lo: list of lower bounds for the overlap region (other dims)
    - overlap_hi: list of upper bounds for the overlap region (other dims)
    """
    a_lo, a_hi = region_a.bounds_lower, region_a.bounds_upper
    b_lo, b_hi = region_b.bounds_lower, region_b.bounds_upper
    
    for d in range(n_dims):
        # Check if A's upper face touches B's lower face
        if abs(a_hi[d] - b_lo[d]) < tol:
            face_val = a_hi[d]
        # Check if B's upper face touches A's lower face
        elif abs(b_hi[d] - a_lo[d]) < tol:
            face_val = b_hi[d]
        else:
            continue
        
        # Check overlap in all other dimensions
        overlap_lo = []
        overlap_hi = []
        has_overlap = True
        
        for d2 in range(n_dims):
            if d2 == d:
                continue
            # Compute overlap interval
            lo = max(a_lo[d2], b_lo[d2])
            hi = min(a_hi[d2], b_hi[d2])
            if hi <= lo + tol:  # No positive overlap
                has_overlap = False
                break
            overlap_lo.append(lo)
            overlap_hi.append(hi)
        
        if has_overlap:
            return True, d, face_val, overlap_lo, overlap_hi
    
    return False, -1, 0.0, [], []


def _add_continuity_faces(
    new_expert_indices: List[int],
    regions,
    spatial_dim: int,
    spatial_domain,
    temporal_domain,
    n_pts_per_face: int,
    output_dim: int,
    device: torch.device,
    xs: list, ts: list, gs: list,
    eids: list, ks: list,
    cont_neighbors: list, cont_dims: list,
):
    """Add continuity points on shared interior faces between leaf neighbors.
    
    For each pair of face-neighbor leaves (a, b), sample points on their shared
    interior face. Points are tagged with:
    - expert_id = a
    - cont_neighbor = b
    - cont_dim = face-normal dimension
    - kind = KIND_CONTINUITY
    
    Both experts a and b are evaluated at the SAME coordinates in the loss,
    so no left/right pairing is needed.
    
    We only add points where A.upper[d] touches B.lower[d] (not the reverse),
    to avoid duplicating pairs. The loss function handles both directions.
    """
    n_dims = spatial_dim + 1  # spatial dims + time
    t_min_global, t_max_global = temporal_domain
    
    # Get global spatial bounds for checking interior faces
    global_lo = [spatial_domain[d][0] for d in range(spatial_dim)] + [t_min_global]
    global_hi = [spatial_domain[d][1] for d in range(spatial_dim)] + [t_max_global]
    
    n_pairs = 0
    
    # Check all pairs of new experts
    for i, eidx_a in enumerate(new_expert_indices):
        region_a = regions[eidx_a]
        
        for eidx_b in new_expert_indices[i+1:]:
            region_b = regions[eidx_b]
            
            is_neighbor, face_dim, face_val, overlap_lo, overlap_hi = \
                _are_face_neighbors(region_a, region_b, n_dims)
            
            if not is_neighbor:
                continue
            
            # Skip if the shared face is on the global boundary (not interior)
            if abs(face_val - global_lo[face_dim]) < ADJACENCY_TOL or \
               abs(face_val - global_hi[face_dim]) < ADJACENCY_TOL:
                continue
            
            # Sample points on the shared face
            n_pts = n_pts_per_face
            
            # Build coordinates: face_dim is fixed at face_val
            # Other dims are sampled from overlap region
            x_cont = torch.zeros(n_pts, spatial_dim, device=device)
            t_cont = torch.zeros(n_pts, 1, device=device)
            
            overlap_idx = 0
            for d in range(n_dims):
                if d == face_dim:
                    # Fixed face coordinate
                    if d < spatial_dim:
                        x_cont[:, d] = face_val
                    else:
                        t_cont[:, 0] = face_val
                else:
                    # Sample from overlap region
                    lo, hi = overlap_lo[overlap_idx], overlap_hi[overlap_idx]
                    vals = torch.rand(n_pts, device=device) * (hi - lo) + lo
                    if d < spatial_dim:
                        x_cont[:, d] = vals
                    else:
                        t_cont[:, 0] = vals
                    overlap_idx += 1
            
            # Add points with expert_id = a, cont_neighbor = b
            xs.append(x_cont)
            ts.append(t_cont)
            gs.append(torch.zeros(n_pts, output_dim, device=device))
            eids.append(torch.full((n_pts,), eidx_a, dtype=torch.long, device=device))
            ks.append(torch.full((n_pts,), KIND_CONTINUITY, dtype=torch.long, device=device))
            cont_neighbors.append(torch.full((n_pts,), eidx_b, dtype=torch.long, device=device))
            cont_dims.append(torch.full((n_pts,), face_dim, dtype=torch.long, device=device))
            
            n_pairs += 1
    
    if n_pairs > 0:
        logger.info(f"[SplitData] Found {n_pairs} face-neighbor pairs for continuity")
