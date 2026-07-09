"""
Generic evaluation visualization for all 1D-spatial PDE problems.

Produces `predictions_and_error_maps.png`: one row per output dimension,
three columns — Ground Truth | Prediction | Error.

Ground truth comes from the solver's NATIVE solution grid
(`_get_solution_cached`), evaluated with no interpolation: querying the
interpolator between grid nodes introduces large artificial errors across
steep fronts (e.g. ~5e-2 fake error needles along the Burgers shock where
the true model error is ~2e-4). Falls back to a dense grid + interpolator
only if the solver has no cached-grid API.
"""

import importlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from typing import Dict, List, Optional


# Maps problem name → solver module path and output-dimension labels.
# Labels: list of strings, one per output dimension.
_PROBLEM_META = {
    'allen_cahn':  ('solvers.allen_cahn_solver',   ['u']),
    'burgers1d':   ('solvers.burgers1d_solver',    ['u']),
    'kdv':         ('solvers.kdv_solver',           ['u']),
    'ks':          ('solvers.ks_solver',            ['u']),
    'schrodinger': ('solvers.schrodinger_solver',   ['u (Real)', 'v (Imaginary)']),
}


def _gt_channels_from_grid(h_sol: np.ndarray, output_dim: int) -> List[np.ndarray]:
    """Split an (n_t, n_x) solution grid into per-output-channel arrays."""
    if np.iscomplexobj(h_sol):
        return [h_sol.real, h_sol.imag][:output_dim]
    return [np.asarray(h_sol, dtype=np.float64)]


def _get_gt_channels(interp, x_flat: np.ndarray, t_flat: np.ndarray,
                     output_dim: int) -> List[np.ndarray]:
    """
    Query the solver interpolator and return a list of flat arrays,
    one per output channel. Fallback path only — interpolating between
    solver grid nodes is inaccurate across steep fronts.
    """
    raw = interp(x_flat, t_flat)          # shape (N,) real or complex

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
    title: Optional[str] = None,
):
    """
    Create a heatmap figure with one row per output dimension, columns:
      [Ground Truth | Prediction | Error] — GT and Prediction share one
    colorbar (identical scale); the error panel has its own.

    Paper-ready: no suptitle; run metadata belongs in the filename/caption.
    If ``filename`` contains the placeholder ``{relL2}``, it is substituted
    with the computed rel-L2 (e.g. ``pred_final_{relL2}.png`` →
    ``pred_final_2.41e-03.png``). Saved as PNG.

    The model is evaluated on the solver's native solution grid (restricted
    to the config's temporal domain, so time-marching windows are scored on
    their own window). n_x / n_t are only used by the interpolator fallback.

    Args:
        model:    Trained model.
        save_dir: Directory to save the figure.
        config:   Full config dict.
        filename: Output filename (may contain the ``{relL2}`` placeholder).
        n_x:      Spatial grid size for the fallback dense grid.
        n_t:      Temporal grid size for the fallback dense grid.
        title:    Deprecated — ignored (kept for caller compatibility).

    Returns:
        The overall rel-L2 (all output dims), or None if skipped.
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
    t_min, t_max = problem_config['temporal_domain']

    solver_mod = importlib.import_module(solver_module_path)

    # ── Ground truth: prefer the solver's native grid (no interpolation) ──
    gt_channels = None
    try:
        x_grid, t_grid, h_sol = solver_mod._get_solution_cached(config)
        t_mask = (t_grid >= t_min - 1e-12) & (t_grid <= t_max + 1e-12)
        t_grid = np.asarray(t_grid)[t_mask]
        h_sol = np.asarray(h_sol)[t_mask]          # (n_t, n_x)
        x_grid = np.asarray(x_grid)
        gt_channels = _gt_channels_from_grid(h_sol, output_dim)
        n_t_eval, n_x_eval = len(t_grid), len(x_grid)
    except Exception as e:
        print(f"  [generic_viz] Native solver grid unavailable ({e}); "
              f"falling back to interpolator on a {n_x}x{n_t} grid.")

    if gt_channels is None:
        x_min, x_max = problem_config['spatial_domain'][0]
        x_grid = np.linspace(x_min, x_max, n_x)
        t_grid = np.linspace(t_min, t_max, n_t)
        interp = solver_mod._get_interpolator(config)
        X_f, T_f = np.meshgrid(x_grid, t_grid)      # (n_t, n_x)
        flat = _get_gt_channels(interp, X_f.flatten(), T_f.flatten(), output_dim)
        gt_channels = [c.reshape(n_t, n_x) for c in flat]
        n_t_eval, n_x_eval = n_t, n_x

    X, T = np.meshgrid(x_grid, t_grid)              # (n_t_eval, n_x_eval)
    x_flat = X.flatten()
    t_flat = T.flatten()

    # ── Model predictions (chunked; native grids can be large) ──
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype  # Match model's precision (float32 or float64)
    model.eval()
    preds = []
    chunk = 65536
    with torch.no_grad():
        for start in range(0, len(x_flat), chunk):
            xb = torch.tensor(x_flat[start:start + chunk], dtype=dtype, device=device).view(-1, 1)
            tb = torch.tensor(t_flat[start:start + chunk], dtype=dtype, device=device).view(-1, 1)
            preds.append(model(torch.cat([xb, tb], dim=1)).cpu())
    pred_np = torch.cat(preds).numpy()              # (N, output_dim)

    # Overall rel-L2 across all output dims (goes into filename/caption)
    gt_all = np.stack([gt_channels[d].reshape(n_t_eval, n_x_eval)
                       for d in range(output_dim)])
    pr_all = np.stack([pred_np[:, d].reshape(n_t_eval, n_x_eval)
                       for d in range(output_dim)])
    rel_l2 = (np.linalg.norm(pr_all - gt_all)
              / (np.linalg.norm(gt_all) + 1e-300))

    # Build figure: output_dim rows × 3 cols (constrained layout places the
    # shared colorbars without overlap)
    fig, axes = plt.subplots(output_dim, 3,
                             figsize=(14, 4.5 * output_dim),
                             squeeze=False, layout='constrained')

    for d, label in enumerate(dim_labels):
        gt, pr = gt_all[d], pr_all[d]
        err = np.abs(pr - gt)

        vmax = max(np.abs(gt).max(), np.abs(pr).max())
        vmin = -vmax
        # Prefix panel titles with the channel label only when multi-output
        prefix = f'{label} — ' if output_dim > 1 else ''

        # pcolormesh (not contourf): continuous paper-style rendering with
        # the exact grid values — contour levels band smooth fields and
        # make near-noise-floor error maps look artificially grainy.
        # Ground Truth + Prediction share one colorbar (same scale)
        im0 = axes[d, 0].pcolormesh(X, T, gt, shading='auto', cmap='RdBu_r',
                                    vmin=vmin, vmax=vmax)
        axes[d, 0].set_title(f'{prefix}Ground truth', fontsize=14)
        im1 = axes[d, 1].pcolormesh(X, T, pr, shading='auto', cmap='RdBu_r',
                                    vmin=vmin, vmax=vmax)
        axes[d, 1].set_title(f'{prefix}Prediction', fontsize=14)
        fig.colorbar(im1, ax=axes[d, :2], pad=0.01)

        # Error panel: LOG color scale. A linear autoscaled map is dominated
        # by localized spikes (thin shock/interface lines, boundary strips):
        # with max/typical error ratios of 50-1000x, the bulk renders blank
        # white. Log scale shows the full error structure; the reported
        # rel-L2 numbers are untouched (this is display-only). The window is
        # capped at 5 decades below the max so noise-floor pixels don't
        # stretch the scale.
        err_max = float(err.max())
        if err_max > 0:
            err_pos = err[err > 0]
            err_vmin = max(float(err_pos.min()), err_max * 1e-5)
            im2 = axes[d, 2].pcolormesh(
                X, T, np.maximum(err, err_vmin), shading='auto', cmap='Reds',
                norm=LogNorm(vmin=err_vmin, vmax=err_max))
        else:
            im2 = axes[d, 2].pcolormesh(X, T, err, shading='auto', cmap='Reds')
        axes[d, 2].set_title(f'{prefix}Absolute error', fontsize=14)
        fig.colorbar(im2, ax=axes[d, 2], pad=0.01)

        for c in range(3):
            axes[d, c].set_xlabel('x', fontsize=13)
            axes[d, c].set_ylabel('t', fontsize=13)
            axes[d, c].tick_params(labelsize=11)

    if '{relL2}' in filename:
        filename = filename.replace('{relL2}', f'{rel_l2:.2e}')
    from utils.plot_io import save_png
    save_path = save_png(Path(save_dir) / filename, fig=fig)
    plt.close(fig)
    print(f"  Predictions & error maps saved to {save_path} "
          f"(rel-L2 = {rel_l2:.3e})")
    return rel_l2
