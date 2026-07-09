"""Training utility functions."""

import math
import torch
import numpy as np
from typing import Dict, Optional


def compute_relative_l2_error(
    h_pred: torch.Tensor,
    h_gt: torch.Tensor
) -> torch.Tensor:
    """
    Compute relative L2 error.

    Args:
        h_pred: Predicted values (N, output_dim)
        h_gt: Ground truth values (N, output_dim)

    Returns:
        Scalar relative L2 error: ||h_pred - h_gt||_2 / ||h_gt||_2
    """
    diff = h_pred - h_gt
    numerator = torch.norm(diff, p=2)
    denominator = torch.norm(h_gt, p=2) + 1e-10  # Avoid division by zero

    return numerator / denominator


def compute_infinity_norm_error(
    h_pred: torch.Tensor,
    h_gt: torch.Tensor
) -> torch.Tensor:
    """
    Compute infinity norm (max absolute) error.

    Args:
        h_pred: Predicted values (N, output_dim)
        h_gt: Ground truth values (N, output_dim)

    Returns:
        Scalar infinity norm error: max(|h_pred - h_gt|)
    """
    diff = h_pred - h_gt
    return torch.max(torch.abs(diff))


def compute_native_grid_metrics(
    model: torch.nn.Module,
    cfg: Dict,
    device: torch.device,
    chunk_size: int = 65536,
) -> Optional[Dict]:
    """Rel-L2 and inf-norm of the model on the solver's NATIVE solution grid.

    This is the paper-comparable metric: the literature reports rel-L2 on the
    reference solution's own grid, with no interpolation (off-node GT queries
    pick up large artificial errors across steep fronts). The grid is
    restricted to the config's temporal domain so time-marching windows are
    scored on their own window. The solver memoizes the solution in-process,
    so calling this every eval is cheap (one chunked no-grad forward).

    Returns:
        {'rel_l2', 'inf_norm', 'n_points', 'grid_shape'} or None if the
        solver grid is unavailable.
    """
    import importlib

    problem = cfg['problem']
    try:
        solver = importlib.import_module(f'solvers.{problem}_solver')
        x_grid, t_grid, h_sol = solver._get_solution_cached(cfg)
    except Exception:
        return None

    t0, t1 = cfg[problem]['temporal_domain']
    t_mask = (t_grid >= t0 - 1e-12) & (t_grid <= t1 + 1e-12)
    t_grid = np.asarray(t_grid)[t_mask]
    h_sol = np.asarray(h_sol)[t_mask]  # (nt, nx), complex for schrodinger

    # Flatten grid to (N, 2) inputs and (N, output_dim) ground truth
    T, X = np.meshgrid(t_grid, np.asarray(x_grid), indexing='ij')
    xt = np.column_stack([X.ravel(), T.ravel()])
    if np.iscomplexobj(h_sol):
        gt = np.column_stack([h_sol.real.ravel(), h_sol.imag.ravel()])
    else:
        gt = h_sol.reshape(-1, 1)

    dtype = next(model.parameters()).dtype
    was_training = model.training
    model.eval()
    total_diff_sq = 0.0
    total_gt_sq = 0.0
    inf_norm = 0.0
    with torch.no_grad():
        for start in range(0, xt.shape[0], chunk_size):
            xb = torch.tensor(xt[start:start + chunk_size], dtype=dtype, device=device)
            gb = torch.tensor(gt[start:start + chunk_size], dtype=dtype, device=device)
            pred = model(xb)
            diff = pred - gb
            total_diff_sq += (diff ** 2).sum().item()
            total_gt_sq += (gb ** 2).sum().item()
            inf_norm = max(inf_norm, diff.abs().max().item())
    if was_training:
        model.train()

    rel_l2 = math.sqrt(total_diff_sq) / (math.sqrt(total_gt_sq) + 1e-10)
    return {
        'rel_l2': rel_l2,
        'inf_norm': inf_norm,
        'n_points': xt.shape[0],
        'grid_shape': (len(t_grid), len(x_grid)),
    }


def native_ground_truth_grid(
    cfg: Dict,
    max_points_per_axis: int = 512,
) -> Optional[tuple]:
    """Ground-truth heatmap grid from the solver's NATIVE solution.

    Single GT source for plots and tree fitting — no random eval sample, no
    interpolation between solver grid nodes. The grid is restricted to the
    config's temporal domain (time-marching windows see their own window) and
    strided down to at most ``max_points_per_axis`` per axis so pcolormesh
    stays fast on large solver grids.

    Returns:
        (gt_grid, grid_x, grid_t) where ``gt_grid`` is (nx, nt) for scalar
        problems or (nx, nt, output_dim) for multi-output (complex) ones,
        or None if the solver grid is unavailable.
    """
    import importlib

    problem = cfg['problem']
    try:
        solver = importlib.import_module(f'solvers.{problem}_solver')
        x_grid, t_grid, h_sol = solver._get_solution_cached(cfg)
    except Exception:
        return None

    t0, t1 = cfg[problem]['temporal_domain']
    t_grid = np.asarray(t_grid)
    t_mask = (t_grid >= t0 - 1e-12) & (t_grid <= t1 + 1e-12)
    t_grid = t_grid[t_mask]
    h_sol = np.asarray(h_sol)[t_mask]        # (nt, nx)
    x_grid = np.asarray(x_grid)

    # Subsample for plotting speed (always keep the last grid point so the
    # domain edges stay covered).
    def _stride_idx(n):
        stride = max(1, int(math.ceil(n / max_points_per_axis)))
        idx = list(range(0, n, stride))
        if idx[-1] != n - 1:
            idx.append(n - 1)
        return np.asarray(idx)

    xi = _stride_idx(len(x_grid))
    ti = _stride_idx(len(t_grid))
    x_grid = x_grid[xi]
    t_grid = t_grid[ti]
    h_sol = h_sol[np.ix_(ti, xi)]

    h_xt = h_sol.T                            # (nx, nt)
    if np.iscomplexobj(h_xt):
        gt = np.stack([h_xt.real, h_xt.imag], axis=2)
    else:
        gt = np.asarray(h_xt, dtype=np.float64)
    return gt, x_grid, t_grid
