"""Subdomain data builder for owner-imitator expert training.

Builds the tagged point set consumed by the owner-imitator loss
(losses/split_loss.py). The hard leaf regions tile the domain, so every
point has a unique OWNER — the leaf whose tile contains it. Each expert
has exactly one role at every point: owner (physics) on its tile,
imitator (distillation to the minted target u*) on its collar.

Row kinds:

* RESIDUAL rows — one uniform draw over the WHOLE domain; ``expert_id``
  is the tile owner (unique). The loss evaluates the PDE residual of the
  OWNER's raw output u_j on its own rows (one mean per expert).
* IC rows — the plain training set's true-IC rows (t = 0), partitioned
  by tile owner. Value matching of u_j against g_ic.
* BC rows — the plain training set's true-BC rows for NON-periodic
  problems (burgers1d Dirichlet): value matching of u_j against g_bc.
* PER rows — the plain training set's BC rows for PERIODIC problems:
  each row carries the mirror coordinate sigma(X) (``mint_x``) and the
  mirror tile owner (``mint_owner``); u_j (value + d/dx) is matched to
  the minted target u* at the mirror.
* IMIT rows — collar points, filtered from the SAME residual draw: a
  point owned by k that falls inside expert j's inflated box but outside
  j's hard tile becomes an IMIT row (expert_id=j, mint_owner=k). Points
  covered by several collars are duplicated once per covering expert.
  u_j (value + axis derivatives up to the configured order) is matched
  to the minted target u* at the point.
* CONTINUITY rows — optional neighbor-to-neighbor pairs on shared hard
  faces (weight 0 by default; kept for experimentation).

The minted target u*(X) = lambda*u0(X) + (1-lambda)*sg[u_k(X)] (values
and axis derivatives) is baked into the ``mint`` column by
:func:`mint_targets` — constant data between refresh events.
"""

import torch
from typing import Dict, List
from adaptive.indicators import RegionDescriptor, inflated_bounds  # noqa: F401
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Problems whose GLOBAL BC pairs value AND d/dx across the periodic
# boundary. Their BC rows become PER rows (mirror-minted targets);
# value-only-BC problems (burgers1d Dirichlet) keep true-BC rows.
PERIODIC_PROBLEMS = frozenset({'allen_cahn', 'kdv', 'ks', 'schrodinger'})

# Integer codes stored in the ``kind`` tensor.
KIND_RESIDUAL = 0    # hard-tile interior physics (owner role)
KIND_IC = 1          # true IC rows (t = 0), owner-partitioned
KIND_BC = 2          # true non-periodic BC rows (Dirichlet)
KIND_PER = 3         # periodic-boundary rows, mirror-minted targets
KIND_IMIT = 4        # collar rows, owner-minted targets
KIND_CONTINUITY = 5  # continuity points on shared interior faces

KIND_NAMES = {
    KIND_RESIDUAL: 'residual',
    KIND_IC: 'ic',
    KIND_BC: 'bc',
    KIND_PER: 'per',
    KIND_IMIT: 'imit',
    KIND_CONTINUITY: 'continuity',
}

# Tolerance for face-neighbor adjacency checks
ADJACENCY_TOL = 1e-8
# Tolerance for hard-tile membership (closed intervals; shared-face
# points go deterministically to the first matching tile).
OWNER_TOL = 1e-12


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


def imit_order(cfg: Dict) -> int:
    """Imitation derivative order q (Sobolev matching order, default 1)."""
    oi_cfg = (cfg.get('adaptive_pinn', {}).get('owner_imitator', {}) or {})
    return int(oi_cfg.get('imit_derivative_order', 1))


def n_mint_slots(cfg: Dict) -> int:
    """Mint slot count A = 2*max(q,1)+1: [value, dx^1..dx^q, dt^1..dt^q].

    The stack is minted with at least order 1 even when q=0, because the
    periodic-BC term structurally needs the mirror d/dx target (slot 1)
    regardless of the imitation order; the imitation term only reads its
    own first 2q+1 slots.
    """
    return 2 * max(imit_order(cfg), 1) + 1


def _in_box_mask(x: torch.Tensor, t: torch.Tensor, lo, hi,
                 tol: float = OWNER_TOL) -> torch.Tensor:
    """Closed-interval membership mask of (x, t) points in box (lo, hi)."""
    spatial_dim = x.shape[1]
    mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    for d in range(spatial_dim):
        mask &= (x[:, d] >= lo[d] - tol) & (x[:, d] <= hi[d] + tol)
    mask &= ((t[:, 0] >= lo[spatial_dim] - tol)
             & (t[:, 0] <= hi[spatial_dim] + tol))
    return mask


def owner_of_points(
    x: torch.Tensor,
    t: torch.Tensor,
    expert_indices: List[int],
    regions,
    tol: float = OWNER_TOL,
) -> torch.Tensor:
    """Unique tile owner of every (x, t) point.

    The hard leaf regions tile the domain, so first-match membership over
    closed intervals (expanded by ``tol``) assigns every point exactly one
    owner; shared-face points go to the lowest-index matching tile.
    Raises if any point is unowned — the tiles are a partition, an
    unowned point is a bug.
    """
    n = x.shape[0]
    owner = torch.full((n,), -1, dtype=torch.long, device=x.device)
    for eidx in expert_indices:
        r = regions[eidx]
        mask = (owner == -1) & _in_box_mask(
            x, t, r.bounds_lower, r.bounds_upper, tol=tol)
        owner[mask] = eidx
    if (owner < 0).any():
        bad = (owner < 0).nonzero(as_tuple=True)[0][:5]
        coords = [(float(x[i, 0]), float(t[i, 0])) for i in bad]
        raise RuntimeError(
            f"[OIData] owner_of_points: {int((owner < 0).sum())} points "
            f"have no owning tile (tiles must partition the domain). "
            f"First offenders (x, t): {coords}")
    return owner


def sample_owner_imitator_draw(
    expert_indices: List[int],
    regions,
    cfg: Dict,
    device: torch.device,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """One uniform draw -> RESIDUAL rows (owner-tagged) + IMIT collar rows.

    A single uniform draw of ``n_residual_train`` points over the whole
    domain. Every point gets its unique tile owner:

    * The draw itself becomes the RESIDUAL rows (``expert_id`` = owner).
    * Per expert j, points inside j's inflated box but outside j's hard
      tile are DUPLICATED as IMIT rows (``expert_id`` = j,
      ``mint_owner`` = the point's owner, ``mint_x`` = x). A point in
      several collars yields one IMIT row per covering expert — each row
      feeds only its expert's imitation mean, all sharing the owner's
      minted target.

    Collar point density therefore matches the residual density; the
    ``mint`` column is allocated here (zeros) and filled by
    :func:`mint_targets`.
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
    n_slots = n_mint_slots(cfg)

    x_g = torch.zeros(n_res_total, spatial_dim, device=device)
    t_g = torch.zeros(n_res_total, 1, device=device)
    for d in range(spatial_dim):
        lo, hi = spatial_domain[d]
        x_g[:, d] = torch.rand(n_res_total, device=device) * (hi - lo) + lo
    t_g[:, 0] = (torch.rand(n_res_total, device=device)
                 * (t_max_global - t_min_global) + t_min_global)

    owner = owner_of_points(x_g, t_g, expert_indices, regions)

    # ── IMIT rows: per-expert collar filtering of the same draw ──
    imit_xs, imit_ts, imit_eids, imit_owners = [], [], [], []
    imit_counts = {}
    for eidx in expert_indices:
        r = regions[eidx]
        infl_lo, infl_hi = inflated_bounds(r, sigma_fraction, g_lo, g_hi)
        in_infl = _in_box_mask(x_g, t_g, infl_lo, infl_hi)
        in_hard = _in_box_mask(x_g, t_g, r.bounds_lower, r.bounds_upper)
        cmask = in_infl & ~in_hard
        n_c = int(cmask.sum())
        imit_counts[eidx] = n_c
        if n_c == 0:
            logger.warning(f"[OIData] expert={eidx} imitation set EMPTY "
                           f"this draw (no collar points caught)")
            continue
        imit_xs.append(x_g[cmask])
        imit_ts.append(t_g[cmask])
        imit_eids.append(torch.full((n_c,), eidx, dtype=torch.long,
                                    device=device))
        imit_owners.append(owner[cmask])

    res_counts = {int(e): int((owner == e).sum())
                  for e in owner.unique().tolist()}
    for eidx in expert_indices:
        if res_counts.get(eidx, 0) == 0:
            logger.warning(f"[OIData] expert={eidx} residual set EMPTY "
                           f"this draw (tile caught no points)")

    n_imit = sum(imit_counts.values())
    logger.info(f"[OIData] draw: {n_res_total} residual points "
                f"(per-owner {res_counts}); {n_imit} duplicated IMIT "
                f"collar rows (per-expert {imit_counts})")

    # Assemble: residual rows first, then IMIT rows.
    x_parts = [x_g] + imit_xs
    t_parts = [t_g] + imit_ts
    eid_parts = [owner] + imit_eids
    kind_parts = [torch.full((n_res_total,), KIND_RESIDUAL,
                             dtype=torch.long, device=device)]
    mint_owner_parts = [torch.full((n_res_total,), -1,
                                   dtype=torch.long, device=device)]
    mint_x_parts = [torch.zeros(n_res_total, spatial_dim, device=device)]
    if n_imit > 0:
        kind_parts.append(torch.full((n_imit,), KIND_IMIT,
                                     dtype=torch.long, device=device))
        mint_owner_parts.append(torch.cat(imit_owners, dim=0))
        mint_x_parts.append(torch.cat(imit_xs, dim=0))

    x_cat = torch.cat(x_parts, dim=0)
    n_all = x_cat.shape[0]
    return {
        'x': x_cat,
        't': torch.cat(t_parts, dim=0),
        'h_gt': torch.zeros(n_all, output_dim, device=device),
        'mint': torch.zeros(n_all, n_slots, output_dim, device=device),
        'expert_id': torch.cat(eid_parts, dim=0),
        'kind': torch.cat(kind_parts, dim=0),
        'mint_owner': torch.cat(mint_owner_parts, dim=0),
        'mint_x': torch.cat(mint_x_parts, dim=0),
        'cont_neighbor': torch.full((n_all,), -1, dtype=torch.long,
                                    device=device),
        'cont_dim': torch.full((n_all,), -1, dtype=torch.long,
                               device=device),
    }


def build_owner_imitator_static(
    plain_train_data: Dict[str, torch.Tensor],
    expert_indices: List[int],
    regions,
    cfg: Dict,
    device: torch.device,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Static rows: true IC, true BC or PER (from the plain training set),
    plus optional continuity faces. Points are fixed for the whole
    segment; only the PER rows' ``mint`` column is refreshed by
    :func:`mint_targets`.

    The global BC type routes the BC rows: non-periodic problems
    (burgers1d) keep them as true-data KIND_BC rows; periodic problems
    turn them into KIND_PER rows carrying the mirror coordinate
    ``mint_x`` = sigma(x) and the mirror tile owner ``mint_owner``.
    """
    torch.manual_seed(seed)

    problem = cfg['problem']
    pc = cfg[problem]
    spatial_dim = pc['spatial_dim']
    spatial_domain = pc['spatial_domain']
    temporal_domain = pc['temporal_domain']
    output_dim = pc['output_dim']
    n_slots = n_mint_slots(cfg)
    is_periodic = problem in PERIODIC_PROBLEMS

    if len(expert_indices) == 0:
        return _empty(spatial_dim, output_dim, n_slots, device)

    xs, ts, gs, eids, ks = [], [], [], [], []
    mint_owners, mint_xs = [], []

    # ── True IC rows (t = 0), owner-partitioned ──
    ic_sel = plain_train_data['mask']['IC']
    n_ic = int(ic_sel.sum())
    if n_ic > 0:
        x_ic = plain_train_data['x'][ic_sel].to(device)
        t_ic = plain_train_data['t'][ic_sel].to(device)
        g_ic = plain_train_data['h_gt'][ic_sel].to(device)
        ic_owner = owner_of_points(x_ic, t_ic, expert_indices, regions)
        xs.append(x_ic)
        ts.append(t_ic)
        gs.append(g_ic)
        eids.append(ic_owner)
        ks.append(torch.full((n_ic,), KIND_IC, dtype=torch.long,
                             device=device))
        mint_owners.append(torch.full((n_ic,), -1, dtype=torch.long,
                                      device=device))
        mint_xs.append(torch.zeros(n_ic, spatial_dim, device=device))
        _ic_counts = {int(e): int((ic_owner == e).sum())
                      for e in ic_owner.unique().tolist()}
        logger.info(f"[OIData] IC rows (true data, t=0): {n_ic} points, "
                    f"per-owner {_ic_counts}")

    # ── BC rows: KIND_BC (true data) or KIND_PER (mirror-minted) ──
    bc_sel = plain_train_data['mask']['BC']
    n_bc = int(bc_sel.sum())
    if n_bc > 0:
        x_bc = plain_train_data['x'][bc_sel].to(device)
        t_bc = plain_train_data['t'][bc_sel].to(device)
        g_bc = plain_train_data['h_gt'][bc_sel].to(device)
        bc_owner = owner_of_points(x_bc, t_bc, expert_indices, regions)
        xs.append(x_bc)
        ts.append(t_bc)
        gs.append(g_bc)
        eids.append(bc_owner)
        if is_periodic:
            # PER: mirror map sigma on the (1D) spatial boundary.
            assert spatial_dim == 1, (
                "PER rows (periodic mirror map) support 1D-spatial "
                "problems only")
            x_lo, x_hi = spatial_domain[0]
            mirror_x = (x_lo + x_hi) - x_bc
            per_owner = owner_of_points(
                mirror_x, t_bc, expert_indices, regions)
            ks.append(torch.full((n_bc,), KIND_PER, dtype=torch.long,
                                 device=device))
            mint_owners.append(per_owner)
            mint_xs.append(mirror_x)
            _per_counts = {int(e): int((bc_owner == e).sum())
                           for e in bc_owner.unique().tolist()}
            logger.info(f"[OIData] PER rows (periodic BC dataset, "
                        f"mirror-minted): {n_bc} points, per-owner "
                        f"{_per_counts}")
        else:
            ks.append(torch.full((n_bc,), KIND_BC, dtype=torch.long,
                                 device=device))
            mint_owners.append(torch.full((n_bc,), -1, dtype=torch.long,
                                          device=device))
            mint_xs.append(torch.zeros(n_bc, spatial_dim, device=device))
            _bc_counts = {int(e): int((bc_owner == e).sum())
                          for e in bc_owner.unique().tolist()}
            logger.info(f"[OIData] BC rows (true Dirichlet data): {n_bc} "
                        f"points, per-owner {_bc_counts}")

    # ── Continuity faces: neighbor-to-neighbor on shared interior faces ──
    _, n_x_face = _face_counts(cfg)
    cont_xs, cont_ts, cont_gs, cont_eids, cont_ks = [], [], [], [], []
    cont_neighbors, cont_dims = [], []
    _add_continuity_faces(
        expert_indices, regions, spatial_dim, spatial_domain,
        temporal_domain, n_x_face, output_dim, device,
        cont_xs, cont_ts, cont_gs, cont_eids, cont_ks,
        cont_neighbors, cont_dims,
    )

    if not xs and not cont_xs:
        return _empty(spatial_dim, output_dim, n_slots, device)

    if xs:
        x_cat = torch.cat(xs, dim=0)
        t_cat = torch.cat(ts, dim=0)
        h_gt_cat = torch.cat(gs, dim=0)
        eid_cat = torch.cat(eids, dim=0)
        kind_cat = torch.cat(ks, dim=0)
        mint_owner_cat = torch.cat(mint_owners, dim=0)
        mint_x_cat = torch.cat(mint_xs, dim=0)
    else:
        x_cat = torch.zeros(0, spatial_dim, device=device)
        t_cat = torch.zeros(0, 1, device=device)
        h_gt_cat = torch.zeros(0, output_dim, device=device)
        eid_cat = torch.zeros(0, dtype=torch.long, device=device)
        kind_cat = torch.zeros(0, dtype=torch.long, device=device)
        mint_owner_cat = torch.zeros(0, dtype=torch.long, device=device)
        mint_x_cat = torch.zeros(0, spatial_dim, device=device)

    n_main = x_cat.shape[0]
    cont_neighbor_main = torch.full((n_main,), -1, dtype=torch.long,
                                    device=device)
    cont_dim_main = torch.full((n_main,), -1, dtype=torch.long,
                               device=device)

    if cont_xs:
        cont_x_cat = torch.cat(cont_xs, dim=0)
        n_cont = cont_x_cat.shape[0]
        x_cat = torch.cat([x_cat, cont_x_cat], dim=0)
        t_cat = torch.cat([t_cat, torch.cat(cont_ts, dim=0)], dim=0)
        h_gt_cat = torch.cat([h_gt_cat, torch.cat(cont_gs, dim=0)], dim=0)
        eid_cat = torch.cat([eid_cat, torch.cat(cont_eids, dim=0)], dim=0)
        kind_cat = torch.cat([kind_cat, torch.cat(cont_ks, dim=0)], dim=0)
        mint_owner_cat = torch.cat([
            mint_owner_cat,
            torch.full((n_cont,), -1, dtype=torch.long, device=device),
        ], dim=0)
        mint_x_cat = torch.cat([
            mint_x_cat,
            torch.zeros(n_cont, spatial_dim, device=device),
        ], dim=0)
        cont_neighbor_main = torch.cat(
            [cont_neighbor_main, torch.cat(cont_neighbors, dim=0)], dim=0)
        cont_dim_main = torch.cat(
            [cont_dim_main, torch.cat(cont_dims, dim=0)], dim=0)
        logger.info(f"[OIData] continuity points: {n_cont}")

    n_all = x_cat.shape[0]
    return {
        'x': x_cat,
        't': t_cat,
        'h_gt': h_gt_cat,
        'mint': torch.zeros(n_all, n_slots, output_dim, device=device),
        'expert_id': eid_cat,
        'kind': kind_cat,
        'mint_owner': mint_owner_cat,
        'mint_x': mint_x_cat,
        'cont_neighbor': cont_neighbor_main,
        'cont_dim': cont_dim_main,
    }


def build_owner_imitator_data(
    expert_indices: List[int],
    regions,
    cfg: Dict,
    device: torch.device,
    seed: int = 0,
    static: Dict[str, torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Full owner-imitator dataset: fresh dynamic draw + static rows.

    ``static`` (from :func:`build_owner_imitator_static`) holds the
    IC/BC-or-PER/continuity rows, fixed for the segment; the dynamic
    part (residual + IMIT collar rows) is redrawn every refresh. The
    returned dict's ``mint`` column is all-zeros — call
    :func:`mint_targets` afterwards. NOTE: the static ``mint`` rows are
    freshly allocated here too (concatenated zeros), so re-minting after
    every build is mandatory, PER rows included.
    """
    problem = cfg['problem']
    pc = cfg[problem]
    if len(expert_indices) == 0:
        return _empty(pc['spatial_dim'], pc['output_dim'],
                      n_mint_slots(cfg), device)

    if static is None:
        raise ValueError(
            "build_owner_imitator_data requires the pre-built static part "
            "(build_owner_imitator_static needs the plain training data)")

    dynamic = sample_owner_imitator_draw(
        expert_indices, regions, cfg, device, seed=seed)

    return {k: torch.cat([dynamic[k], static[k]], dim=0) for k in dynamic}


# ── Minting ─────────────────────────────────────────────


def _axis_derivative_stack(
    u: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    q: int,
) -> torch.Tensor:
    """Stack [value, dx^1..dx^q, dt^1..dt^q] of u w.r.t. (x, t).

    ``u`` is (n, C) computed from ``x``/``t`` leaf tensors with
    ``requires_grad``. Pure axis derivatives (face-normal directions of
    the axis-aligned seams), no mixed partials; componentwise over C.
    Keeps the graph (create_graph=True) so it is usable both inside the
    loss (gradients flow to the expert) and for minting (caller
    detaches). Returns (n, 2q+1, C).
    """
    n_out = u.shape[1]
    slots = [u]
    for wrt in (x, t):
        cur = u
        for _p in range(q):
            cols = []
            for c in range(n_out):
                g = torch.autograd.grad(
                    cur[:, c].sum(), wrt,
                    create_graph=True, retain_graph=True,
                )[0][:, 0]
                cols.append(g)
            cur = torch.stack(cols, dim=1)
            slots.append(cur)
    return torch.stack(slots, dim=1)


def mint_targets(
    data: Dict[str, torch.Tensor],
    model,
    root_model: torch.nn.Module,
    lam: float,
    q: int,
) -> Dict[str, float]:
    """Bake the minted targets u* into ``data['mint']`` (PER + IMIT rows).

        u*(X) = lam * u0(X) + (1 - lam) * sg[ u_k(X) ]

    evaluated at ``mint_x`` (the point itself for IMIT, the mirror for
    PER), with the full axis-derivative stack up to order ``q``. The
    stop-gradient sg[.] is realized by evaluating the LIVE model and
    detaching into constant tensors — targets cannot move with the
    weights until the next mint. At ``lam >= 1`` the owner forwards are
    skipped entirely and the stack is bit-for-bit the root's.

    Returns stats: {'n_minted', 'mean_abs_dev_from_root'} where the
    deviation measures the owner exchange magnitude (exactly 0 at
    lam=1).
    """
    kinds = data['kind']
    rows = (kinds == KIND_PER) | (kinds == KIND_IMIT)
    n_minted = int(rows.sum())
    if n_minted == 0:
        return {'n_minted': 0, 'mean_abs_dev_from_root': 0.0}

    # Stack order is at least 1 even when q=0: the PER term needs the
    # mirror d/dx target (slot 1) independent of the imitation order.
    q_stack = max(q, 1)
    mint = data['mint']
    assert mint.shape[1] == 2 * q_stack + 1, (
        f"mint slot mismatch: tensor has {mint.shape[1]} slots, "
        f"expected {2 * q_stack + 1} (q={q})")
    row_idx = rows.nonzero(as_tuple=True)[0]
    mx = data['mint_x'][row_idx]
    mt = data['t'][row_idx]
    owners = data['mint_owner'][row_idx]

    def _stack_of(forward_fn, x_pts, t_pts):
        x_l = x_pts.clone().detach().requires_grad_(True)
        t_l = t_pts.clone().detach().requires_grad_(True)
        u = forward_fn(torch.cat([x_l, t_l], dim=1))
        return _axis_derivative_stack(u, x_l, t_l, q_stack).detach()

    root_stack = _stack_of(root_model, mx, mt)

    if lam >= 1.0:
        mint[row_idx] = root_stack
        assert not data['mint'].requires_grad
        return {'n_minted': n_minted, 'mean_abs_dev_from_root': 0.0}

    own_stack = torch.zeros_like(root_stack)
    for k in owners.unique().tolist():
        assert k >= 0, "minted row without a mint_owner"
        kmask = (owners == k)
        own_stack[kmask] = _stack_of(
            lambda xt, _k=k: model.forward_single_expert(_k, xt),
            mx[kmask], mt[kmask])

    combined = lam * root_stack + (1.0 - lam) * own_stack
    mint[row_idx] = combined
    assert not data['mint'].requires_grad
    dev = float((combined - root_stack).abs().mean())
    return {'n_minted': n_minted, 'mean_abs_dev_from_root': dev}


# ── Helpers ─────────────────────────────────────────────


def _empty(spatial_dim, output_dim, n_slots, device):
    return {
        'x': torch.zeros(0, spatial_dim, device=device),
        't': torch.zeros(0, 1, device=device),
        'h_gt': torch.zeros(0, output_dim, device=device),
        'mint': torch.zeros(0, n_slots, output_dim, device=device),
        'expert_id': torch.zeros(0, dtype=torch.long, device=device),
        'kind': torch.zeros(0, dtype=torch.long, device=device),
        'mint_owner': torch.zeros(0, dtype=torch.long, device=device),
        'mint_x': torch.zeros(0, spatial_dim, device=device),
        'cont_neighbor': torch.zeros(0, dtype=torch.long, device=device),
        'cont_dim': torch.zeros(0, dtype=torch.long, device=device),
    }


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
        logger.info(f"[OIData] Found {n_pairs} face-neighbor pairs for continuity")
