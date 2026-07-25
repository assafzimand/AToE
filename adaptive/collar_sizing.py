"""Gradient-driven collar sizing for the AToE partition of unity.

The fixed collar of Section 3.6 gives every region the same relative
half-width in every direction: delta_j = sigma_fraction * (b_j - a_j).
Because the tree makes regions LARGE exactly where the solution is smooth,
that rule spends the most overlap where the least is needed.

This module sizes each collar from the solution instead. For every leaf
face it marches outward until the mean directional gradient on that face
drops below a global quantile of the same gradient field, so the blend
transition lands in a quiet part of the domain and the collar wraps
whatever feature the interface cuts through.

Sensor, per dimension z, over the whole domain:

    G_z(X) = || d u / d z ||_2      (L2 over output channels)
    ref_z  = quantile(G_z, q)

For leaf R, dimension z, upper side, with face at b_z:

    s(c) = mean of G_z over { X : X_z = c, X_other in R }
    delta_hi = (first c > b_z with s(c) < ref_z) - b_z

clamped to [sigma_min, sigma_max] * (b_z - a_z). Both clamps are in the
SAME unit as sigma_fraction, so sigma_min=0.02 means "never narrower than
sigma_fraction 0.02 would give". If no face in the scan window clears the
threshold the argmin of s over that window is used, so a non-monotone
sensor can never run an edge out to the cap by accident.

Faces lying on the physical domain boundary have no blending partner, so
their collar is set to the floor and never scanned.

The sensor is evaluated on the ROOT u_0 by default: it is the field the
method already trusts for interface data (Section 3.5), it is available
without composing anything, and using it keeps collar sizing independent
of the very windows it is about to define.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from adaptive.indicators import RegionDescriptor
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Total sensor-grid points to stay under, so 3D domains reduce the
# per-axis resolution instead of allocating tens of millions of points.
_MAX_GRID_POINTS = 4_000_000


def _grid_axes(domain_bounds: Dict, per_axis: int) -> List[np.ndarray]:
    """Regular grid coordinates per dimension, budget-capped in total size."""
    n_dims = len(domain_bounds['lower'])
    capped = int(max(9, min(per_axis, _MAX_GRID_POINTS ** (1.0 / n_dims))))
    if capped < per_axis:
        logger.info(f"  [CollarSizing] Sensor grid reduced "
                    f"{per_axis} -> {capped} per axis ({n_dims}D budget)")
    return [np.linspace(lo, hi, capped)
            for lo, hi in zip(domain_bounds['lower'], domain_bounds['upper'])]


def sensor_fields(
    sensor_fn: Callable[[torch.Tensor], torch.Tensor],
    axes: Sequence[np.ndarray],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 262_144,
) -> List[np.ndarray]:
    """Per-dimension gradient magnitude of ``sensor_fn`` on the grid.

    Derivatives come from autograd rather than finite differences, so the
    sensor carries no grid-spacing error and the result does not depend on
    the chosen resolution beyond where it is sampled.

    Returns one array per dimension, each shaped like the grid.
    """
    mesh = np.meshgrid(*axes, indexing='ij')
    shape = mesh[0].shape
    pts = np.stack([m.ravel() for m in mesh], axis=1)
    n_dims = pts.shape[1]

    out = [np.empty(pts.shape[0], dtype=np.float64) for _ in range(n_dims)]
    with torch.enable_grad():
        for start in range(0, pts.shape[0], batch_size):
            chunk = torch.tensor(pts[start:start + batch_size],
                                 device=device, dtype=dtype,
                                 requires_grad=True)
            u = sensor_fn(chunk)
            if u.ndim == 1:
                u = u.unsqueeze(1)
            # Sum of squared per-channel gradients, accumulated channel by
            # channel: || d u / d z ||_2 over the output channels.
            acc = torch.zeros(chunk.shape[0], n_dims,
                              device=device, dtype=dtype)
            for c in range(u.shape[1]):
                g, = torch.autograd.grad(
                    u[:, c].sum(), chunk,
                    retain_graph=(c < u.shape[1] - 1), create_graph=False)
                acc = acc + g ** 2
            mag = acc.sqrt().detach().cpu().numpy()
            for d in range(n_dims):
                out[d][start:start + chunk.shape[0]] = mag[:, d]

    return [o.reshape(shape) for o in out]


def _region_axis_indices(axes, region, exclude_dim) -> List[np.ndarray]:
    """Grid indices inside the region, per axis, with ``exclude_dim`` empty.

    A region thinner than the grid spacing along some axis would otherwise
    select nothing; those axes fall back to the single nearest node so the
    face statistic is always defined.
    """
    idx = []
    for d, coord in enumerate(axes):
        if d == exclude_dim:
            idx.append(None)
            continue
        lo, hi = region.bounds_lower[d], region.bounds_upper[d]
        sel = np.nonzero((coord >= lo) & (coord <= hi))[0]
        if sel.size == 0:
            sel = np.array([int(np.argmin(np.abs(coord - 0.5 * (lo + hi))))])
        idx.append(sel)
    return idx


def _face_stats(field, axes, region, dim, candidates) -> np.ndarray:
    """Mean of ``field`` on each candidate face, restricted to the region.

    One gather for all candidates at once, then a mean over every axis but
    ``dim`` — the face is a point in 1D-spatial problems' terms, a line in
    2D (x,t), a plane in 3D.
    """
    idx = _region_axis_indices(axes, region, exclude_dim=dim)
    idx[dim] = candidates
    block = field[np.ix_(*idx)]
    other = tuple(d for d in range(block.ndim) if d != dim)
    return block.mean(axis=other) if other else block


def compute_adaptive_collars(
    sensor_fn: Callable[[torch.Tensor], torch.Tensor],
    regions: List[RegionDescriptor],
    domain_bounds: Dict,
    quantile: float,
    sigma_min: float,
    sigma_max: float,
    grid_per_axis: int,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Size every leaf collar from the sensor field.

    Returns ``(delta_lo, delta_hi, diagnostics)`` with both delta arrays
    shaped (K, D) in absolute domain units.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    n_dims = len(domain_bounds['lower'])
    K = len(regions)

    axes = _grid_axes(domain_bounds, grid_per_axis)
    grads = sensor_fields(sensor_fn, axes, device, dtype)
    refs = [float(np.quantile(g, quantile)) for g in grads]

    logger.info(
        f"  [CollarSizing] grid={'x'.join(str(len(a)) for a in axes)}, "
        f"q={quantile}, clamp sigma in [{sigma_min}, {sigma_max}]")
    for d in range(n_dims):
        g = grads[d]
        logger.info(
            f"  [CollarSizing]   dim {d}: |du/dz| mean={g.mean():.4g} "
            f"median={np.median(g):.4g} max={g.max():.4g} "
            f"-> ref(q={quantile})={refs[d]:.4g}")

    delta_lo = np.zeros((K, n_dims))
    delta_hi = np.zeros((K, n_dims))
    outcomes = {'threshold': 0, 'argmin': 0, 'boundary': 0}
    clamped = {'floor': 0, 'cap': 0}
    per_face = []

    for k, region in enumerate(regions):
        for d in range(n_dims):
            a, b = region.bounds_lower[d], region.bounds_upper[d]
            extent = b - a
            floor, cap = sigma_min * extent, sigma_max * extent
            coord = axes[d]

            for side in ('lo', 'hi'):
                face = a if side == 'lo' else b
                on_boundary = (
                    abs(face - domain_bounds['lower'][d]) < 1e-12
                    or abs(face - domain_bounds['upper'][d]) < 1e-12)

                if on_boundary:
                    # No neighbour to blend with: the collar here only ever
                    # covers points outside the domain.
                    delta, outcome = floor, 'boundary'
                else:
                    if side == 'lo':
                        window = (coord <= face) & (coord >= face - cap)
                    else:
                        window = (coord >= face) & (coord <= face + cap)
                    cand = np.nonzero(window)[0]
                    if cand.size == 0:
                        delta, outcome = floor, 'boundary'
                    else:
                        cand = cand[np.argsort(np.abs(coord[cand] - face))]
                        stats = _face_stats(grads[d], axes, region, d, cand)
                        below = np.nonzero(stats < refs[d])[0]
                        if below.size:
                            pick, outcome = cand[below[0]], 'threshold'
                        else:
                            pick, outcome = cand[int(np.argmin(stats))], 'argmin'
                        delta = abs(float(coord[pick]) - face)

                raw = delta
                delta = float(np.clip(delta, floor, cap))
                if outcome != 'boundary':
                    if delta > raw + 1e-15:
                        clamped['floor'] += 1
                    elif delta < raw - 1e-15:
                        clamped['cap'] += 1
                outcomes[outcome] += 1

                if side == 'lo':
                    delta_lo[k, d] = delta
                else:
                    delta_hi[k, d] = delta
                per_face.append({
                    'region': k, 'dim': d, 'side': side,
                    'face': float(face), 'delta': delta,
                    'sigma_equiv': delta / extent if extent > 0 else 0.0,
                    'outcome': outcome,
                })

    _log_summary(regions, delta_lo, delta_hi, outcomes, clamped, n_dims)

    return delta_lo, delta_hi, {
        'quantile': quantile,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'refs': refs,
        'grid_per_axis': [len(a) for a in axes],
        'outcomes': outcomes,
        'clamped': clamped,
        'faces': per_face,
    }


def _log_summary(regions, delta_lo, delta_hi, outcomes, clamped, n_dims):
    """Per-region collar table plus aggregate outcome counts."""
    total = sum(outcomes.values())
    logger.info(
        f"  [CollarSizing] {total} faces: "
        f"{outcomes['threshold']} met threshold, "
        f"{outcomes['argmin']} fell back to argmin, "
        f"{outcomes['boundary']} on domain boundary "
        f"| clamped: {clamped['floor']} to floor, {clamped['cap']} to cap")

    header = "  [CollarSizing]  leaf " + "  ".join(
        f"| d{d}: lo/hi delta (sigma-equiv)" for d in range(n_dims))
    logger.info(header)
    for k, region in enumerate(regions):
        parts = []
        for d in range(n_dims):
            ext = region.bounds_upper[d] - region.bounds_lower[d]
            slo = delta_lo[k, d] / ext if ext > 0 else 0.0
            shi = delta_hi[k, d] / ext if ext > 0 else 0.0
            parts.append(f"| {delta_lo[k, d]:.4f}/{delta_hi[k, d]:.4f} "
                         f"({slo:.3f}/{shi:.3f})")
        logger.info(f"  [CollarSizing]  {k:4d} " + "  ".join(parts))

    for d in range(n_dims):
        ext = np.array([r.bounds_upper[d] - r.bounds_lower[d]
                        for r in regions])
        sig = np.concatenate([delta_lo[:, d] / ext, delta_hi[:, d] / ext])
        logger.info(
            f"  [CollarSizing]  dim {d} sigma-equiv: "
            f"min={sig.min():.4f} mean={sig.mean():.4f} max={sig.max():.4f}")


def size_collars_from_root(
    model,
    fine_tune_cfg: Dict,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, Dict]]:
    """Size the leaf collars from the frozen root u_0.

    Returns ``(delta_lo, delta_hi, diagnostics)`` as (K, D) tensors indexed
    by the model's FULL expert list (non-leaf slots keep the fixed-sigma
    value), or None when there is nothing to size.
    """
    leaf_list = sorted(i for i in model.leaf_indices if i >= 0)
    if len(leaf_list) < 2:
        logger.info("  [CollarSizing] Fewer than 2 leaf experts, skipped.")
        return None

    leaf_regions = [model.regions[i] for i in leaf_list]
    domain_bounds = model.get_domain_bounds()

    was_training = model.base_model.training
    model.base_model.eval()
    try:
        delta_lo_np, delta_hi_np, diag = compute_adaptive_collars(
            sensor_fn=model.base_model,
            regions=leaf_regions,
            domain_bounds=domain_bounds,
            quantile=float(fine_tune_cfg.get('adaptive_collar_quantile', 0.85)),
            sigma_min=float(fine_tune_cfg.get('adaptive_collar_sigma_min',
                                              0.02)),
            sigma_max=float(fine_tune_cfg.get('adaptive_collar_sigma_max',
                                              0.5)),
            grid_per_axis=int(fine_tune_cfg.get('adaptive_collar_grid', 257)),
            device=device,
        )
    finally:
        if was_training:
            model.base_model.train()

    # Scatter the leaf-indexed results back into full expert indexing, so
    # BatchedIndicators (which is built over all K experts) lines up.
    n_experts = len(model.regions)
    n_dims = delta_lo_np.shape[1]
    sigma = float(model.sigma_fraction)
    full_lo = np.zeros((n_experts, n_dims))
    full_hi = np.zeros((n_experts, n_dims))
    for j, region in enumerate(model.regions):
        ext = np.array(region.bounds_upper) - np.array(region.bounds_lower)
        full_lo[j] = sigma * ext
        full_hi[j] = sigma * ext
    for local, expert_idx in enumerate(leaf_list):
        full_lo[expert_idx] = delta_lo_np[local]
        full_hi[expert_idx] = delta_hi_np[local]

    dtype = torch.get_default_dtype()
    diag['leaf_indices'] = leaf_list
    return (torch.tensor(full_lo, dtype=dtype, device=device),
            torch.tensor(full_hi, dtype=dtype, device=device),
            diag)
