"""Build a parameter-capacity heatmap for each run in an experiment batch.

For every point (x, t) in the domain the capacity is:

    capacity(x, t) = base_params
                     + Σ_{k : point ∈ Ω_k} expert_k_params

Which experts are counted depends on the model type:
  - AToE  : all experts (each has a hard region Ω_k)
  - AToELeaves : leaf experts only (non-leaf parents are frozen)
  - ANT   : all experts (same as AToE; output layers of non-leaves
            are excluded from the count, matching trainer accounting)

The output per run is a side-by-side figure:
  Left  – expert_regions_final.png (spatial region layout)
  Right – capacity heatmap (continuous, same domain axes)

Usage:
    python make_capacity_map.py <batch_or_run_dir>
"""

import sys
import json
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


_TS_RE = re.compile(r'\d{8}_\d{6}$')

# ──────────────────────────────────────────────────────────────────────────────
#  Directory discovery (mirrors plot_experts_predictions.py)
# ──────────────────────────────────────────────────────────────────────────────

def _find_run_dirs(batch_path: Path):
    """Return list of (label, ts_dir) pairs."""
    child_dirs = sorted(
        d for d in batch_path.iterdir()
        if d.is_dir() and d.name != 'checkpoints'
    )
    if not child_dirs:
        return []

    # If the path itself is a time-marching run (window_* dirs present), return it
    # directly instead of treating window_* as model-name dirs, which would cause
    # the nested-layout fallback to enumerate adaptive_plots/, training_plots/, etc.
    # as fake timestamp dirs.
    if _is_time_marching_run(batch_path):
        return [(batch_path.name, batch_path)]

    # Flat layout: batch_path contains timestamp dirs directly
    # Include both regular runs (have metrics.json) and time-marching runs (have window_0/)
    def _is_valid_run(d):
        if not _TS_RE.match(d.name):
            return False
        # Regular run
        if (d / 'metrics.json').exists():
            return True
        # Time-marching run
        window_0 = d / 'window_0'
        if window_0.exists() and (window_0 / 'metrics.json').exists():
            return True
        return False
    
    flat_ts = [d for d in child_dirs if _is_valid_run(d)]
    if flat_ts:
        runs = []
        for ts_dir in flat_ts:
            cfg_file = ts_dir / 'config_used.yaml'
            label = ts_dir.name
            if cfg_file.exists():
                try:
                    with open(cfg_file) as f:
                        cfg = yaml.safe_load(f)
                    label = cfg.get('problem', ts_dir.name)
                except Exception:
                    pass
            runs.append((label, ts_dir))
        return runs

    # Nested layout: batch_path / model_name / timestamp(s)
    # Expand ALL timestamps per architecture (not just latest)
    runs = []
    for model_dir in child_dirs:
        ts_dirs = sorted(
            d for d in model_dir.iterdir()
            if d.is_dir() and d.name != 'checkpoints'
        )
        for ts_dir in ts_dirs:
            runs.append((f"{model_dir.name}/{ts_dir.name}", ts_dir))
    return runs


# ──────────────────────────────────────────────────────────────────────────────
#  Hard indicator (step function) — no torch dependency
# ──────────────────────────────────────────────────────────────────────────────

def _hard_mask(points: np.ndarray, lower: list, upper: list) -> np.ndarray:
    """Return boolean (N,) – True where point is inside the box [lower, upper]."""
    lo = np.array(lower)   # (D,)
    hi = np.array(upper)   # (D,)
    return np.all((points >= lo) & (points <= hi), axis=1)


# ──────────────────────────────────────────────────────────────────────────────
#  Capacity computation
# ──────────────────────────────────────────────────────────────────────────────

def _compute_capacity(points: np.ndarray, metrics: dict,
                      model_type: str) -> np.ndarray:
    """
    Compute per-point capacity (parameter count) for the model.

    Parameters
    ----------
    points     : (N, D) array of [x, t] (or [x, y, t]) coordinates
    metrics    : loaded metrics.json dict
    model_type : 'AToE' | 'AToELeaves' | 'ANT' (from config_used.yaml)

    Returns
    -------
    capacity : (N,) float array
    """
    adaptive = metrics.get('adaptive_pinn')
    if adaptive is None:
        total = metrics.get('total_params', 0)
        return np.full(len(points), float(total))

    base_params   = adaptive['base_params']
    expert_params = adaptive['expert_params']   # list[int], one per expert
    regions       = adaptive['regions']
    leaf_indices  = set(adaptive.get('leaf_expert_indices', []))

    # AToELeaves and ANT: only leaf experts contribute capacity at a point.
    # AToE: ALL experts whose region contains the point contribute.
    leaves_only = (model_type == 'AToELeaves')

    N = len(points)
    capacity = np.full(N, float(base_params))

    for i, region in enumerate(regions):
        if i >= len(expert_params):
            continue
        if leaves_only and i not in leaf_indices:
            continue
        lower = region['bounds_lower']
        upper = region['bounds_upper']
        mask  = _hard_mask(points, lower, upper)
        capacity[mask] += expert_params[i]

    return capacity


# ──────────────────────────────────────────────────────────────────────────────
#  Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_grid(x: np.ndarray, t: np.ndarray,
               values: np.ndarray, n_x=300, n_t=300):
    """Interpolate values onto a regular grid for smooth continuous heatmap."""
    from scipy.interpolate import griddata
    
    x_min, x_max = x.min(), x.max()
    t_min, t_max = t.min(), t.max()

    # Create regular grid
    x_grid = np.linspace(x_min, x_max, n_x)
    t_grid = np.linspace(t_min, t_max, n_t)
    X_grid, T_grid = np.meshgrid(x_grid, t_grid, indexing='ij')
    
    # Interpolate using linear method for smooth result
    points = np.column_stack([x, t])
    grid_values = griddata(points, values, (X_grid, T_grid), method='linear')
    
    # Fill any remaining NaN at edges with nearest neighbor
    nan_mask = np.isnan(grid_values)
    if nan_mask.any():
        grid_values_nn = griddata(points, values, (X_grid, T_grid), method='nearest')
        grid_values[nan_mask] = grid_values_nn[nan_mask]

    return x_grid, t_grid, grid_values


def _draw_regions(ax, regions, leaf_indices, leaves_only_model):
    """Overlay region boxes on an axis."""
    for i, region in enumerate(regions):
        lo = region['bounds_lower']
        hi = region['bounds_upper']
        is_leaf = (i in leaf_indices)
        if leaves_only_model and not is_leaf:
            color, lw, ls = 'grey', 1.0, '--'
        else:
            color, lw, ls = 'red', 1.5, '-'
        rect = mpatches.Rectangle(
            (lo[0], lo[1]),
            hi[0] - lo[0], hi[1] - lo[1],
            linewidth=lw, edgecolor=color,
            facecolor='none', linestyle=ls)
        ax.add_patch(rect)


# ──────────────────────────────────────────────────────────────────────────────
#  Time-marching support
# ──────────────────────────────────────────────────────────────────────────────

def _is_time_marching_run(ts_dir: Path) -> bool:
    """Check if this is a time-marching run (has window_N subdirectories)."""
    window_dirs = sorted(d for d in ts_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('window_'))
    if not window_dirs:
        return False
    # Check at least window_0 has metrics
    return (window_dirs[0] / 'metrics.json').exists()


def _load_time_marching_metrics(ts_dir: Path):
    """Load and stitch metrics from all windows in a time-marching run.
    
    Returns:
        Combined metrics dict with stitched regions, expert_params, etc.
    """
    window_dirs = sorted(
        d for d in ts_dir.iterdir() 
        if d.is_dir() and d.name.startswith('window_')
    )
    
    all_regions = []
    all_expert_params = []
    all_leaf_indices = []
    base_params = None
    
    expert_offset = 0  # Track expert index offset for leaf_indices
    
    for window_dir in window_dirs:
        metrics_path = window_dir / 'metrics.json'
        if not metrics_path.exists():
            continue
            
        with open(metrics_path) as f:
            win_metrics = json.load(f)
        
        adaptive = win_metrics.get('adaptive_pinn')
        if adaptive is None:
            continue
        
        # Use base_params from first window (should be same across all)
        if base_params is None:
            base_params = adaptive.get('base_params', 0)
        
        # Stitch regions and expert_params
        win_regions = adaptive.get('regions', [])
        win_expert_params = adaptive.get('expert_params', [])
        win_leaf_indices = adaptive.get('leaf_expert_indices', [])
        
        all_regions.extend(win_regions)
        all_expert_params.extend(win_expert_params)
        
        # Offset leaf indices to account for experts from previous windows
        all_leaf_indices.extend([idx + expert_offset for idx in win_leaf_indices])
        
        expert_offset += len(win_regions)
    
    # Build combined metrics structure
    combined_metrics = {
        'adaptive_pinn': {
            'base_params': base_params or 0,
            'expert_params': all_expert_params,
            'regions': all_regions,
            'leaf_expert_indices': all_leaf_indices,
        }
    }
    
    return combined_metrics, len(window_dirs)


# ──────────────────────────────────────────────────────────────────────────────
#  Per-run processing
# ──────────────────────────────────────────────────────────────────────────────

def process_run(label: str, ts_dir: Path):
    # Check for time-marching run first
    if _is_time_marching_run(ts_dir):
        return process_time_marching_run(label, ts_dir)
    
    metrics_path = ts_dir / 'metrics.json'
    if not metrics_path.exists():
        print(f"  [{label}] No metrics.json, skipping")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    # ── Load config first (need model_type before adaptive checks) ──────────────
    cfg_path = ts_dir / 'config_used.yaml'
    if not cfg_path.exists():
        print(f"  [{label}] No config_used.yaml, skipping")
        return
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    problem    = cfg['problem']
    model_type = cfg.get('model', 'AToE')   # 'AToE' | 'AToELeaves' | 'ANT'
    eval_path  = Path('datasets') / problem / 'eval_data.pt'

    adaptive = metrics.get('adaptive_pinn')
    if adaptive is None:
        print(f"  [{label}] No adaptive_pinn section in metrics, skipping")
        return

    regions           = adaptive['regions']
    leaf_indices      = set(adaptive.get('leaf_expert_indices', []))
    leaves_only_model = (model_type == 'AToELeaves')
    if not eval_path.exists():
        print(f"  [{label}] No eval data at {eval_path}, skipping")
        return

    import torch
    eval_data = torch.load(eval_path, map_location='cpu', weights_only=False)
    x_np = eval_data['x'].numpy()   # (N, spatial_dim)
    t_np = eval_data['t'].numpy()   # (N, 1)

    spatial_dim = x_np.shape[1]
    if spatial_dim != 1:
        print(f"  [{label}] Only 1D spatial problems supported for now, skipping")
        return

    x_flat = x_np[:, 0]   # (N,)
    t_flat = t_np[:, 0]   # (N,)
    
    # ── Create dense grid for continuous heatmap ────────────────────────────────
    # Compute capacity directly on grid (not interpolated from sparse eval data)
    n_grid = 300  # Resolution for smooth heatmap
    x_grid = np.linspace(x_flat.min(), x_flat.max(), n_grid)
    t_grid = np.linspace(t_flat.min(), t_flat.max(), n_grid)
    X_grid, T_grid = np.meshgrid(x_grid, t_grid, indexing='ij')
    grid_points = np.column_stack([X_grid.ravel(), T_grid.ravel()])  # (n_grid^2, 2)
    
    # ── Compute capacity at each grid point ────────────────────────────────────
    capacity_grid = _compute_capacity(grid_points, metrics, model_type)  # (n_grid^2,)
    cap_grid = capacity_grid.reshape(n_grid, n_grid)  # (n_x, n_t)

    # ── Find expert_regions_final.png ────────────────────────────────────────
    regions_img_path = ts_dir / 'adaptive_plots' / 'expert_regions_final.png'

    # ── Build figure ─────────────────────────────────────────────────────────
    has_regions_img = regions_img_path.exists()
    n_cols = 2 if has_regions_img else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    n_exp  = len(regions)
    n_leaf = len(leaf_indices)
    title  = f"{problem}  |  {n_exp} experts ({n_leaf} leaves)"
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Left panel: expert regions image (if present)
    if has_regions_img:
        img = plt.imread(str(regions_img_path))
        axes[0].imshow(img)
        axes[0].set_axis_off()
        axes[0].set_title('Expert Regions (final)', fontsize=11)

    # Right panel: capacity heatmap (computed on dense grid for continuous display)
    ax = axes[-1]
    vmin = cap_grid.min()
    vmax = cap_grid.max()
    im = ax.imshow(
        cap_grid.T,
        extent=[x_grid[0], x_grid[-1], t_grid[0], t_grid[-1]],
        origin='lower',
        aspect='auto',
        cmap='YlOrRd',
        vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Parameters')

    _draw_regions(ax, regions, leaf_indices, leaves_only_model)

    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('t', fontsize=11)
    ax.set_title('Capacity Heatmap (params per point)', fontsize=11)

    # Legend
    solid = mpatches.Patch(edgecolor='red', facecolor='none',
                           linestyle='-', label='Leaf expert region')
    handles = [solid]
    if leaves_only_model:
        grey = mpatches.Patch(edgecolor='grey', facecolor='none',
                              linestyle='--', label='Non-leaf (frozen)')
        handles.append(grey)
    ax.legend(handles=handles, fontsize=8, loc='upper right')

    plt.tight_layout()

    out_path = ts_dir / 'capacity_map.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [{label}] Saved {out_path}")


def process_time_marching_run(label: str, ts_dir: Path):
    """Process a time-marching run by stitching regions from all windows."""
    print(f"  [{label}] Time-marching run detected")
    
    # Load config
    cfg_path = ts_dir / 'config_used.yaml'
    if not cfg_path.exists():
        print(f"  [{label}] No config_used.yaml, skipping")
        return
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    problem = cfg['problem']
    model_type = cfg.get('model', 'AToE')
    problem_cfg = cfg.get(problem, {})
    
    # Get full domain from config
    spatial_domain = problem_cfg.get('spatial_domain', [[0, 1]])
    temporal_domain = problem_cfg.get('temporal_domain', [0, 1])
    x_min, x_max = spatial_domain[0]
    t_min, t_max = temporal_domain
    
    # Load and stitch metrics from all windows
    combined_metrics, num_windows = _load_time_marching_metrics(ts_dir)
    
    adaptive = combined_metrics.get('adaptive_pinn')
    if adaptive is None or not adaptive.get('regions'):
        print(f"  [{label}] No regions found in any window, skipping")
        return
    
    regions = adaptive['regions']
    leaf_indices = set(adaptive.get('leaf_expert_indices', []))
    leaves_only_model = (model_type == 'AToELeaves')
    
    print(f"  [{label}] Stitched {len(regions)} regions from {num_windows} windows")
    
    # Create dense grid over FULL domain
    n_grid = 300
    x_grid = np.linspace(x_min, x_max, n_grid)
    t_grid = np.linspace(t_min, t_max, n_grid)
    X_grid, T_grid = np.meshgrid(x_grid, t_grid, indexing='ij')
    grid_points = np.column_stack([X_grid.ravel(), T_grid.ravel()])
    
    # Compute capacity at each grid point
    capacity_grid = _compute_capacity(grid_points, combined_metrics, model_type)
    cap_grid = capacity_grid.reshape(n_grid, n_grid)
    
    # Build figure (single panel - no pre-existing regions image for combined)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    n_exp = len(regions)
    n_leaf = len(leaf_indices)
    title = f"{problem}  |  {n_exp} experts ({n_leaf} leaves)  |  {num_windows} windows"
    fig.suptitle(title, fontsize=13, fontweight='bold')
    
    # Capacity heatmap
    vmin = cap_grid.min()
    vmax = cap_grid.max()
    im = ax.imshow(
        cap_grid.T,
        extent=[x_grid[0], x_grid[-1], t_grid[0], t_grid[-1]],
        origin='lower',
        aspect='auto',
        cmap='YlOrRd',
        vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Parameters')
    
    _draw_regions(ax, regions, leaf_indices, leaves_only_model)
    
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('t', fontsize=11)
    ax.set_title('Capacity Heatmap (params per point)', fontsize=11)
    
    # Legend
    solid = mpatches.Patch(edgecolor='red', facecolor='none',
                           linestyle='-', label='Leaf expert region')
    handles = [solid]
    if leaves_only_model:
        grey = mpatches.Patch(edgecolor='grey', facecolor='none',
                              linestyle='--', label='Non-leaf (frozen)')
        handles.append(grey)
    ax.legend(handles=handles, fontsize=8, loc='upper right')
    
    plt.tight_layout()
    
    out_path = ts_dir / 'capacity_map.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [{label}] Saved {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(batch_dir: str):
    batch_path = Path(batch_dir)
    if not batch_path.exists():
        print(f"Error: {batch_path} not found")
        return

    print("\n" + "=" * 70)
    print("Capacity Map Generator")
    print(f"Batch: {batch_path.name}")
    print("=" * 70)

    runs = _find_run_dirs(batch_path)
    if not runs:
        # Maybe the path IS a run dir (has metrics.json directly or is time-marching)
        if (batch_path / 'metrics.json').exists():
            label = batch_path.name
            process_run(label, batch_path)
        elif (batch_path / 'window_0' / 'metrics.json').exists():
            label = batch_path.name
            process_run(label, batch_path)  # Will detect time-marching
        else:
            print("No runs found")
        return

    print(f"Found {len(runs)} run(s)\n")
    for label, ts_dir in runs:
        process_run(label, ts_dir)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python make_capacity_map.py <batch_or_run_dir>")
