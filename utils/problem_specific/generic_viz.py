"""
Generic evaluation visualization for all 1D-spatial PDE problems.

Produces `predictions_and_error_maps.png`: one row per output dimension,
three columns — Ground Truth | Prediction | Error.
Works for any problem whose solver exposes `_get_interpolator(config)`.
"""

import importlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


# Maps problem name → solver module path and output-dimension labels.
# Labels: list of strings, one per output dimension.
_PROBLEM_META = {
    'allen_cahn':  ('solvers.allen_cahn_solver',   ['u']),
    'burgers1d':   ('solvers.burgers1d_solver',    ['u']),
    'kdv':         ('solvers.kdv_solver',           ['u']),
    'ks':          ('solvers.ks_solver',            ['u']),
    'wave1d':      ('solvers.wave1d_solver',        ['u']),
    'conv_diff':   ('solvers.conv_diff_solver',     ['u']),
    'fisher_kpp':  ('solvers.fisher_kpp_solver',    ['u']),
    'schrodinger': ('solvers.schrodinger_solver',   ['u (Real)', 'v (Imaginary)']),
}


def _get_gt_channels(interp, x_flat: np.ndarray, t_flat: np.ndarray,
                     output_dim: int) -> List[np.ndarray]:
    """
    Query the solver interpolator and return a list of (n_t, n_x) arrays,
    one per output channel.
    """
    raw = interp(x_flat, t_flat)          # shape (N,) real or complex
    n = len(x_flat)

    if output_dim == 1:
        return [np.asarray(raw, dtype=np.float64).reshape(-1)]
    else:
        # Schrödinger: complex → [real, imag]
        raw_c = np.asarray(raw, dtype=np.complex128)
        channels = [raw_c.real, raw_c.imag]
        return channels[:output_dim]


def plot_predictions_and_error_maps(
    model: torch.nn.Module,
    save_dir: Path,
    config: Dict,
    filename: str = "predictions_and_error_maps.png",
    n_x: int = 256,
    n_t: int = 200,
):
    """
    Create a heatmap figure with one row per output dimension, columns:
      [Ground Truth | Prediction | Error]

    Args:
        model:    Trained model.
        save_dir: Directory to save the figure.
        config:   Full config dict.
        filename: Output filename.
        n_x:      Number of spatial grid points for the dense evaluation grid.
        n_t:      Number of temporal grid points.
    """
    problem = config.get('problem', '')
    if problem not in _PROBLEM_META:
        print(f"  [generic_viz] No metadata for problem '{problem}'. Skipping.")
        return

    solver_module_path, dim_labels = _PROBLEM_META[problem]
    output_dim = config[problem].get('output_dim', 1)
    # Use labels only up to output_dim (in case labels list has extras)
    dim_labels = dim_labels[:output_dim]

    problem_config = config[problem]
    x_min, x_max = problem_config['spatial_domain'][0]
    t_min, t_max = problem_config['temporal_domain']

    # Dense evaluation grid
    x_grid = np.linspace(x_min, x_max, n_x)
    t_grid = np.linspace(t_min, t_max, n_t)
    X, T = np.meshgrid(x_grid, t_grid)          # (n_t, n_x)
    x_flat = X.flatten()
    t_flat = T.flatten()

    # Model predictions
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype  # Match model's precision (float32 or float64)
    model.eval()
    x_tensor = torch.tensor(x_flat, dtype=dtype, device=device).view(-1, 1)
    t_tensor = torch.tensor(t_flat, dtype=dtype, device=device).view(-1, 1)
    with torch.no_grad():
        pred = model(torch.cat([x_tensor, t_tensor], dim=1))   # (N, output_dim)
    pred_np = pred.cpu().numpy()                                # (N, output_dim)

    # Ground truth from solver interpolator
    solver_mod = importlib.import_module(solver_module_path)
    interp = solver_mod._get_interpolator(config)
    gt_channels = _get_gt_channels(interp, x_flat, t_flat, output_dim)

    # Build figure: output_dim rows × 3 cols
    fig, axes = plt.subplots(output_dim, 3,
                             figsize=(15, 4.5 * output_dim),
                             squeeze=False)

    for d, label in enumerate(dim_labels):
        gt  = gt_channels[d].reshape(n_t, n_x)
        pr  = pred_np[:, d].reshape(n_t, n_x)
        err = np.abs(pr - gt)

        vmax = max(np.abs(gt).max(), np.abs(pr).max())
        vmin = -vmax

        # Ground Truth
        im0 = axes[d, 0].contourf(X, T, gt, levels=50, cmap='RdBu_r',
                                   vmin=vmin, vmax=vmax)
        axes[d, 0].set_title(f'{label} — Ground Truth', fontsize=12, fontweight='bold')
        axes[d, 0].set_xlabel('x')
        axes[d, 0].set_ylabel('t')
        plt.colorbar(im0, ax=axes[d, 0])

        # Prediction
        im1 = axes[d, 1].contourf(X, T, pr, levels=50, cmap='RdBu_r',
                                   vmin=vmin, vmax=vmax)
        axes[d, 1].set_title(f'{label} — Prediction', fontsize=12, fontweight='bold')
        axes[d, 1].set_xlabel('x')
        axes[d, 1].set_ylabel('t')
        plt.colorbar(im1, ax=axes[d, 1])

        # Error
        im2 = axes[d, 2].contourf(X, T, err, levels=50, cmap='Reds')
        axes[d, 2].set_title(f'{label} — Absolute Error', fontsize=12, fontweight='bold')
        axes[d, 2].set_xlabel('x')
        axes[d, 2].set_ylabel('t')
        plt.colorbar(im2, ax=axes[d, 2])

    plt.suptitle(f'{problem} — Predictions & Error Maps', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = Path(save_dir) / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Predictions & error maps saved to {save_path}")
