"""Per-expert subdomain data builder for split IC/BC training.

Builds a combined training dataset where each point is tagged with the
owning expert, its kind (residual / ic_true / interface / bc_true),
and bc_face_id for periodic BC pairing.

For bc_true points on global spatial boundaries, bc_face_id encodes
dim*2 + side (side=0 lower, side=1 upper) to enable cross-expert
pairing in periodic BC loss (Allen-Cahn).

Used by the split-loss training path for AToE-Leaves and ANT.
"""

import torch
from typing import Dict, List
from adaptive.indicators import RegionDescriptor  # noqa: F401
from utils.dataset_gen import _analytic_ic
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Integer codes stored in the ``kind`` tensor
KIND_RESIDUAL = 0
KIND_IC_TRUE = 1
KIND_INTERFACE = 2      # t-face interface (weighted by w_ic)
KIND_INTERFACE_BC = 3   # x-face interface (weighted by w_bc)
KIND_BC_TRUE = 4
KIND_CONTINUITY = 5     # continuity points on shared interior faces (neighbor-to-neighbor)

KIND_NAMES = {
    KIND_RESIDUAL: 'residual',
    KIND_IC_TRUE: 'ic_true',
    KIND_INTERFACE: 'interface_ic',
    KIND_INTERFACE_BC: 'interface_bc',
    KIND_BC_TRUE: 'bc_true',
    KIND_CONTINUITY: 'continuity',
}

# Tolerance for face-neighbor adjacency checks
ADJACENCY_TOL = 1e-8


def build_subdomain_data(
    model: torch.nn.Module,
    new_expert_indices: List[int],
    regions,
    cfg: Dict,
    device: torch.device,
    seed: int = 0,
    additive: bool = False,
    interface_model: torch.nn.Module = None,
) -> Dict[str, torch.Tensor]:
    """Build per-expert dataset for split-loss training.

    Returns dict with keys ``x``, ``t``, ``h_gt``,
    ``expert_id``, ``kind``, ``bc_face_id``, ``cont_neighbor``, ``cont_dim``.
    
    ``bc_face_id`` encodes which spatial boundary face
    for periodic pairing: ``dim * 2 + side`` where
    side=0 for lower, side=1 for upper.
    
    For periodic BC, left (side=0) and right (side=1) points
    on the same dimension share identical t-values to enable
    cross-expert pairing.
    
    When ``additive=True``:
      - Interface targets, ic_true targets, and bc_true targets are set to 0
        (leaves must output 0 on their bounds as they are corrections to the root)
      - Continuity points are still generated for neighbor-to-neighbor agreement
    """
    torch.manual_seed(seed)

    problem = cfg['problem']
    pc = cfg[problem]
    spatial_dim = pc['spatial_dim']
    spatial_domain = pc['spatial_domain']
    temporal_domain = pc['temporal_domain']
    t_min_global = temporal_domain[0]
    t_max_global = temporal_domain[1]
    output_dim = pc['output_dim']

    sampling = cfg.get('sampling', {})
    n_res_total = sampling.get('n_residual_train', 10000)
    # Allow explicit override via sampling.n_initial_train / n_boundary_train
    # (absolute counts); falls back to the ratio-based calculation otherwise.
    n_ic_per_face = sampling.get('n_initial_train')
    if n_ic_per_face is None:
        ic_ratio = sampling.get('initial_train_ratio', 0.026)
        n_ic_per_face = round(n_res_total * ic_ratio)
    n_bc_per_face = sampling.get('n_boundary_train')
    if n_bc_per_face is None:
        bc_ratio = sampling.get('boundary_train_ratio', 0.026)
        n_bc_per_face = round(n_res_total * bc_ratio)
    n_ic_per_face = max(1, int(n_ic_per_face))
    n_bc_per_face = max(1, int(n_bc_per_face))

    num_experts = len(new_expert_indices)
    if num_experts == 0:
        return _empty(spatial_dim, output_dim, device)

    # Fix 3+4: Generate shared t-values per dimension (full global range)
    # Each expert will filter to its own temporal range
    bc_t_global = {}
    for d in range(spatial_dim):
        bc_t_global[d] = (
            torch.rand(n_bc_per_face, 1, device=device)
            * (t_max_global - t_min_global) + t_min_global
        )

    # ── Residual: global uniform, filter into regions ──
    x_g = torch.zeros(n_res_total, spatial_dim, device=device)
    t_g = torch.zeros(n_res_total, 1, device=device)
    for d in range(spatial_dim):
        lo, hi = spatial_domain[d]
        x_g[:, d] = (torch.rand(n_res_total, device=device)
                      * (hi - lo) + lo)
    t_g[:, 0] = (torch.rand(n_res_total, device=device)
                  * (t_max_global - t_min_global) + t_min_global)

    xs, ts, gs, eids, ks, bc_fids = [], [], [], [], [], []

    for eidx in new_expert_indices:
        region = regions[eidx]
        bl, bu = region.bounds_lower, region.bounds_upper

        mask = torch.ones(
            n_res_total, dtype=torch.bool, device=device
        )
        for d in range(spatial_dim):
            mask &= ((x_g[:, d] >= bl[d])
                      & (x_g[:, d] <= bu[d]))
        mask &= ((t_g[:, 0] >= bl[spatial_dim])
                  & (t_g[:, 0] <= bu[spatial_dim]))

        n = mask.sum().item()
        if n > 0:
            xs.append(x_g[mask])
            ts.append(t_g[mask])
            gs.append(torch.zeros(n, output_dim, device=device))
            eids.append(torch.full(
                (n,), eidx, dtype=torch.long, device=device
            ))
            ks.append(torch.full(
                (n,), KIND_RESIDUAL, dtype=torch.long,
                device=device
            ))
            bc_fids.append(torch.full(
                (n,), -1, dtype=torch.long, device=device
            ))

    # ── IC / BC faces per expert ──
    for eidx in new_expert_indices:
        region = regions[eidx]
        _add_ic_face(
            eidx, region, spatial_dim, spatial_domain,
            t_min_global, n_ic_per_face, output_dim,
            problem, pc, device, xs, ts, gs, eids, ks,
            bc_fids, additive=additive,
        )
        _add_bc_faces_periodic(
            eidx, region, spatial_dim, spatial_domain,
            n_bc_per_face, output_dim, device,
            xs, ts, gs, eids, ks, bc_fids,
            bc_t_global, additive=additive,
        )

    # ── Continuity faces: neighbor-to-neighbor on shared interior faces ──
    cont_xs, cont_ts, cont_gs, cont_eids, cont_ks = [], [], [], [], []
    cont_neighbors, cont_dims = [], []
    _add_continuity_faces(
        new_expert_indices, regions, spatial_dim, spatial_domain,
        temporal_domain, n_bc_per_face, output_dim, device,
        cont_xs, cont_ts, cont_gs, cont_eids, cont_ks,
        cont_neighbors, cont_dims,
    )

    # ── Concatenate main data ──
    x_cat = torch.cat(xs, dim=0)
    t_cat = torch.cat(ts, dim=0)
    h_gt_cat = torch.cat(gs, dim=0)
    eid_cat = torch.cat(eids, dim=0)
    kind_cat = torch.cat(ks, dim=0)
    bc_fid_cat = torch.cat(bc_fids, dim=0)
    
    # Initialize cont_neighbor and cont_dim for main data (all -1)
    n_main = x_cat.shape[0]
    cont_neighbor_main = torch.full((n_main,), -1, dtype=torch.long, device=device)
    cont_dim_main = torch.full((n_main,), -1, dtype=torch.long, device=device)

    # ── Mint interface targets (skip if additive) ──
    # interface_model overrides which frozen field defines the interface targets.
    # For non-additive AToELeaves it is the base (root), so targets are good root
    # predictions even when experts cannot inherit the root; None falls back to
    # `model` (composed snapshot) for legacy behaviour.
    if not additive:
        iface_src = interface_model if interface_model is not None else model
        # t-face interfaces (KIND_INTERFACE, weighted by w_ic)
        iface_mask = (kind_cat == KIND_INTERFACE)
        if iface_mask.sum() > 0:
            with torch.no_grad():
                xt_if = torch.cat(
                    [x_cat[iface_mask], t_cat[iface_mask]], dim=1
                )
                h_gt_cat[iface_mask] = iface_src(xt_if)

        # x-face interfaces (KIND_INTERFACE_BC, weighted by w_bc)
        iface_bc_mask = (kind_cat == KIND_INTERFACE_BC)
        if iface_bc_mask.sum() > 0:
            with torch.no_grad():
                xt_if_bc = torch.cat(
                    [x_cat[iface_bc_mask], t_cat[iface_bc_mask]], dim=1
                )
                h_gt_cat[iface_bc_mask] = iface_src(xt_if_bc)
        _src = 'base(root)' if interface_model is not None else 'composed'
        logger.info(f"[SplitData] interface targets minted from {_src} model")
    else:
        logger.info("[SplitData] additive=True: interface/ic/bc targets are 0")

    # Log BC statistics for periodic pairing
    bc_true_mask = (kind_cat == KIND_BC_TRUE)
    n_bc_true = bc_true_mask.sum().item()
    if n_bc_true > 0:
        unique_fids = bc_fid_cat[bc_true_mask].unique().tolist()
        logger.info(
            f"[SplitData] bc_true points: {n_bc_true}, "
            f"unique face_ids: {unique_fids}"
        )

    # ── Concatenate continuity data (if any) ──
    if cont_xs:
        cont_x_cat = torch.cat(cont_xs, dim=0)
        cont_t_cat = torch.cat(cont_ts, dim=0)
        cont_g_cat = torch.cat(cont_gs, dim=0)
        cont_eid_cat = torch.cat(cont_eids, dim=0)
        cont_kind_cat = torch.cat(cont_ks, dim=0)
        cont_neighbor_cat = torch.cat(cont_neighbors, dim=0)
        cont_dim_cat = torch.cat(cont_dims, dim=0)
        
        # BC face id is -1 for continuity points
        cont_bc_fid_cat = torch.full(
            (cont_x_cat.shape[0],), -1, dtype=torch.long, device=device
        )
        
        # Merge main + continuity
        x_cat = torch.cat([x_cat, cont_x_cat], dim=0)
        t_cat = torch.cat([t_cat, cont_t_cat], dim=0)
        h_gt_cat = torch.cat([h_gt_cat, cont_g_cat], dim=0)
        eid_cat = torch.cat([eid_cat, cont_eid_cat], dim=0)
        kind_cat = torch.cat([kind_cat, cont_kind_cat], dim=0)
        bc_fid_cat = torch.cat([bc_fid_cat, cont_bc_fid_cat], dim=0)
        cont_neighbor_main = torch.cat([cont_neighbor_main, cont_neighbor_cat], dim=0)
        cont_dim_main = torch.cat([cont_dim_main, cont_dim_cat], dim=0)
        
        logger.info(
            f"[SplitData] continuity points: {cont_x_cat.shape[0]}"
        )

    return {
        'x': x_cat,
        't': t_cat,
        'h_gt': h_gt_cat,
        'expert_id': eid_cat,
        'kind': kind_cat,
        'bc_face_id': bc_fid_cat,
        'cont_neighbor': cont_neighbor_main,
        'cont_dim': cont_dim_main,
    }


# ── Helpers ─────────────────────────────────────────────


def _empty(spatial_dim, output_dim, device):
    return {
        'x': torch.zeros(0, spatial_dim, device=device),
        't': torch.zeros(0, 1, device=device),
        'h_gt': torch.zeros(0, output_dim, device=device),
        'expert_id': torch.zeros(
            0, dtype=torch.long, device=device
        ),
        'kind': torch.zeros(
            0, dtype=torch.long, device=device
        ),
        'bc_face_id': torch.zeros(
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


def _add_ic_face(
    eidx, region, spatial_dim, spatial_domain,
    t_min_global, n_pts, output_dim, problem, pc,
    device, xs, ts, gs, eids, ks, bc_fids,
    additive: bool = False,
):
    """Add IC face points (t = region lower-t boundary).
    
    When additive=True, ic_true target is set to 0 (leaf should output 0 on boundary).
    """
    bl, bu = region.bounds_lower, region.bounds_upper
    t_face = bl[spatial_dim]
    is_true = abs(t_face - t_min_global) < 1e-8

    x_ic = torch.zeros(n_pts, spatial_dim, device=device)
    for d in range(spatial_dim):
        lo, hi = bl[d], bu[d]
        x_ic[:, d] = (torch.rand(n_pts, device=device)
                       * (hi - lo) + lo)
    t_ic = torch.full((n_pts, 1), t_face, device=device)

    if is_true:
        if additive:
            # In additive mode, leaves should output 0 on true boundaries
            h_gt = torch.zeros(n_pts, output_dim, device=device)
        else:
            h_gt = _analytic_ic(problem, x_ic, pc)
        kind_val = KIND_IC_TRUE
    else:
        # Interface target: 0 if additive (model mints later if not additive)
        h_gt = torch.zeros(n_pts, output_dim, device=device)
        kind_val = KIND_INTERFACE

    xs.append(x_ic)
    ts.append(t_ic)
    gs.append(h_gt)
    eids.append(torch.full(
        (n_pts,), eidx, dtype=torch.long, device=device
    ))
    ks.append(torch.full(
        (n_pts,), kind_val, dtype=torch.long, device=device
    ))
    bc_fids.append(torch.full(
        (n_pts,), -1, dtype=torch.long, device=device
    ))


def _add_bc_faces_periodic(
    eidx, region, spatial_dim, spatial_domain,
    n_pts, output_dim, device,
    xs, ts, gs, eids, ks, bc_fids,
    bc_t_global,
    additive: bool = False,
):
    """Add BC face points with periodic pairing support.
    
    For bc_true faces on global boundaries:
    - Uses shared t-values per dimension (both left and right sides)
    - Filters to t-values within this expert's temporal range
    - Assigns bc_face_id = dim*2 + side (side=0 lower, 1 upper)
    
    For interior x-face interfaces (non-global boundaries):
    - Uses KIND_INTERFACE_BC (weighted by w_bc)
    
    When additive=True, all targets are set to 0.
    """
    bl, bu = region.bounds_lower, region.bounds_upper
    t_lo = bl[spatial_dim]
    t_hi = bu[spatial_dim]

    for d in range(spatial_dim):
        g_lo, g_hi = spatial_domain[d]
        for side_idx, face_val in enumerate([bl[d], bu[d]]):
            is_true = _is_global_boundary(
                face_val, g_lo, g_hi
            )

            if is_true:
                # Fix 3+4: Use shared t per dimension, filter to expert's t-range
                t_global = bc_t_global[d]
                # Filter t-values that fall within this expert's temporal range
                t_mask = (t_global[:, 0] >= t_lo) & (t_global[:, 0] <= t_hi)
                t_bc = t_global[t_mask]
                n_actual = t_bc.shape[0]
                
                if n_actual == 0:
                    continue
                
                kind_val = KIND_BC_TRUE
                face_id = d * 2 + side_idx
            else:
                # Fix 6: x-face interface uses KIND_INTERFACE_BC (weighted by w_bc)
                t_bc = (
                    torch.rand(n_pts, 1, device=device)
                    * (t_hi - t_lo) + t_lo
                )
                n_actual = n_pts
                kind_val = KIND_INTERFACE_BC
                face_id = -1

            x_bc = torch.zeros(
                n_actual, spatial_dim, device=device
            )
            x_bc[:, d] = face_val
            for d2 in range(spatial_dim):
                if d2 != d:
                    lo2, hi2 = bl[d2], bu[d2]
                    x_bc[:, d2] = (
                        torch.rand(n_actual, device=device)
                        * (hi2 - lo2) + lo2
                    )

            h_gt = torch.zeros(
                n_actual, output_dim, device=device
            )

            xs.append(x_bc)
            ts.append(t_bc)
            gs.append(h_gt)
            eids.append(torch.full(
                (n_actual,), eidx,
                dtype=torch.long, device=device
            ))
            ks.append(torch.full(
                (n_actual,), kind_val,
                dtype=torch.long, device=device
            ))
            bc_fids.append(torch.full(
                (n_actual,), face_id,
                dtype=torch.long, device=device
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
