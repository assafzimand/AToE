"""Plot model predictions vs ground truth for all experiments in a folder.

Produces per run:
  1. composed_pred_vs_gt.png — full model output | GT | error
  2. base_pred_vs_gt.png    — base network only | GT | error
  3. experts_pred_vs_gt.png — grid: GT, composed, error, base, E1 … EN

Produces per batch:
  - composed_predictions_summary.png
  - base_predictions_summary.png

Checkpoint priority: last segment's best_model_<segment>.pt (reconciled best);
legacy runs fall back to final_model.pt / best_model.pt.

Ground truth source (1D spatial):
  - Uses solver interpolator (high accuracy, full spectral solution)
  - Falls back to eval_data.pt if solver unavailable
  - Evaluates on 200×200 dense grid (matches in-run plots)

Usage:
    python plot_predictions.py <experiment_batch_dir>
"""

import sys
import math
import re
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


_TS_RE = re.compile(r'\d{8}_\d{6}$')


# -----------------------------------------------------------------
#  Directory discovery
# -----------------------------------------------------------------

def _find_run_dirs(batch_path):
    """Find all (label, timestamp_dir) pairs in a batch."""
    child_dirs = sorted(
        d for d in batch_path.iterdir()
        if d.is_dir() and d.name != 'checkpoints'
    )
    if not child_dirs:
        return []

    flat_ts = [
        d for d in child_dirs
        if _TS_RE.match(d.name)
        and (d / 'metrics.json').exists()
    ]
    if flat_ts:
        runs = []
        for ts_dir in flat_ts:
            cfg_file = ts_dir / 'config_used.yaml'
            if cfg_file.exists():
                try:
                    with open(cfg_file) as f:
                        cfg = yaml.safe_load(f)
                    label = cfg.get('problem', ts_dir.name)
                except Exception:
                    label = ts_dir.name
            else:
                label = ts_dir.name
            runs.append((label, ts_dir))
        return runs

    runs = []
    for model_dir in child_dirs:
        ts_dirs = sorted(
            d for d in model_dir.iterdir()
            if d.is_dir() and d.name != 'checkpoints'
        )
        if not ts_dirs:
            continue
        runs.append((model_dir.name, ts_dirs[-1]))
    return runs


# -----------------------------------------------------------------
#  Model construction & checkpoint loading
# -----------------------------------------------------------------

def _build_model(cfg):
    from models.network_factory import create_network
    from models.atoe_leaves import AToELeaves

    # Match the run's precision (like the trainer does at setup): evaluating
    # a float64 model as a float32 cast adds ~1e-7 relative rounding noise,
    # which pollutes error maps and rel-L2 for well-converged models. Set the
    # DEFAULT dtype (not .double()) so non-parameter tensors created at build
    # and checkpoint-load time (indicator windows, FF matrices) match too.
    # Intentionally not restored — checkpoint loading right after this call
    # must run under the same default.
    if cfg.get('precision', 'float32') == 'float64':
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    architecture = cfg['base_architecture']
    activation = cfg.get('activation', 'tanh')
    adaptive_cfg = cfg.get('adaptive_pinn', {})
    is_adaptive = adaptive_cfg.get('enabled', False)

    if is_adaptive:
        return AToELeaves(architecture, activation, cfg, adaptive_cfg)
    expert_type = adaptive_cfg.get('expert_type', 'mlp')
    return create_network(architecture, activation, cfg,
                          is_base=True, expert_type=expert_type)


def _load_checkpoint(model, ckpt_path, is_adaptive):
    checkpoint = torch.load(
        ckpt_path, map_location='cpu', weights_only=False)

    if is_adaptive and checkpoint.get('is_adaptive', False):
        model.load_state_dict_extended(
            checkpoint['adaptive_state'])
    else:
        sd = checkpoint.get(
            'model_state_dict',
            checkpoint.get('model', checkpoint))
        try:
            model.load_state_dict(sd, strict=False)
        except RuntimeError:
            remapped = {}
            arch = getattr(model, 'base_architecture', None)
            for k, v in sd.items():
                if k.startswith('output.') and arch:
                    n = len(arch) - 1
                    remapped[k.replace(
                        'output.',
                        f'network.layer_{n}.')] = v
                elif k.startswith('layer_'):
                    remapped[f'network.{k}'] = v
                else:
                    remapped[k] = v
            model.load_state_dict(remapped, strict=False)

    return checkpoint.get('epoch', '?')


# The run's result checkpoint: the LAST pipeline segment's reconciled best
# (best == end-of-segment by construction). Older runs fall back to
# final_model.pt / best_model.pt.
_CKPT_PRIORITY = [
    'best_model_fine_tune.pt', 'best_model_phase3.pt', 'best_model_root.pt',
    'best_model_main.pt', 'final_model.pt', 'best_model.pt',
]


def _find_checkpoint(ts_dir, cfg):
    ckpt_dir = ts_dir / 'checkpoints'

    spawn_epoch = cfg.get('adaptive_pinn', {}).get('spawn_every_epochs')
    if spawn_epoch is not None:
        spawn_ckpt = ckpt_dir / f'checkpoint_epoch_{spawn_epoch}.pt'
        if spawn_ckpt.exists():
            return spawn_ckpt

    for name in _CKPT_PRIORITY:
        candidate = ckpt_dir / name
        if candidate.exists():
            return candidate

    # Legacy fallback: old structure with checkpoints at parent level
    problem = cfg['problem']
    arch = cfg['base_architecture']
    act = cfg.get('activation', 'tanh')
    arch_str = f"{problem}-{'-'.join(map(str, arch))}-{act}"
    legacy_dir = ts_dir.parent / 'checkpoints' / arch_str
    for name in _CKPT_PRIORITY:
        candidate = legacy_dir / name
        if candidate.exists():
            return candidate

    return None


# -----------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------

def _to_magnitude(h):
    """(N, d) -> (N,) scalar magnitude."""
    if h.ndim == 1:
        return h
    if h.shape[1] == 2:
        return np.sqrt(h[:, 0] ** 2 + h[:, 1] ** 2)
    return h[:, 0]


def _get_gt_from_solver(cfg, x_flat, t_flat, output_dim):
    """Get ground truth from solver interpolator (high accuracy).
    
    Returns list of (N,) arrays, one per output dimension.
    Mimics the in-run plot approach for accurate GT.
    """
    import importlib
    
    problem = cfg['problem']
    try:
        solver_mod = importlib.import_module(f"solvers.{problem}_solver")
        interp = solver_mod._get_interpolator(cfg)
        gt_raw = interp(x_flat, t_flat)
        
        # Handle multi-dimensional output (e.g., Schrödinger: complex -> [real, imag])
        if output_dim == 1:
            return [np.asarray(gt_raw, dtype=np.float64)]
        else:
            gt_c = np.asarray(gt_raw, dtype=np.complex128)
            channels = [gt_c.real, gt_c.imag]
            return channels[:output_dim]
    except Exception as e:
        print(f"  Warning: Could not load solver interpolator: {e}")
        print(f"  Falling back to eval_data.pt (less accurate)")
        return None


def _grid_1d(x, t, values, n_x=200, n_t=200):
    from scipy.interpolate import griddata
    x_lin = np.linspace(x.min(), x.max(), n_x)
    t_lin = np.linspace(t.min(), t.max(), n_t)
    X, T = np.meshgrid(x_lin, t_lin, indexing='ij')
    Z = griddata(
        np.column_stack([x, t]), values, (X, T),
        method='linear')
    mask = np.isnan(Z)
    if mask.any():
        Z_nn = griddata(
            np.column_stack([x, t]), values, (X, T),
            method='nearest')
        Z[mask] = Z_nn[mask]
    return x_lin, t_lin, Z


# -----------------------------------------------------------------
#  3-panel plot: prediction | GT | error  (reused for both base
#  and composed — caller decides which prediction to pass)
# -----------------------------------------------------------------

def _plot_triplet_1d(x, t, h_pred, h_gt, pred_label,
                     title, save_path):
    mag_pred = _to_magnitude(h_pred)
    mag_gt = _to_magnitude(h_gt)

    x_lin, t_lin, Z_pred = _grid_1d(x, t, mag_pred)
    _, _, Z_gt = _grid_1d(x, t, mag_gt)
    Z_err = np.abs(Z_pred - Z_gt)

    vmin = np.nanmin([np.nanmin(Z_pred), np.nanmin(Z_gt)])
    vmax = np.nanmax([np.nanmax(Z_pred), np.nanmax(Z_gt)])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    kw = dict(shading='auto', vmin=vmin, vmax=vmax)
    im = axes[0].pcolormesh(
        x_lin, t_lin, Z_pred.T, cmap='viridis', **kw)
    axes[0].set_title(pred_label)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    plt.colorbar(im, ax=axes[0])

    im = axes[1].pcolormesh(
        x_lin, t_lin, Z_gt.T, cmap='viridis', **kw)
    axes[1].set_title('Ground Truth')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')
    plt.colorbar(im, ax=axes[1])

    im = axes[2].pcolormesh(
        x_lin, t_lin, Z_err.T, shading='auto', cmap='hot')
    axes[2].set_title('Pointwise Error')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('t')
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_triplet_2d(x, t, h_pred, h_gt, pred_label,
                     title, save_path):
    mag_pred = _to_magnitude(h_pred)
    mag_gt = _to_magnitude(h_gt)

    vmin = np.nanmin([np.nanmin(mag_pred), np.nanmin(mag_gt)])
    vmax = np.nanmax([np.nanmax(mag_pred), np.nanmax(mag_gt)])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    skw = dict(s=2, cmap='viridis', vmin=vmin, vmax=vmax)
    sc = axes[0].scatter(
        x[:, 0], x[:, 1], c=mag_pred, **skw)
    axes[0].set_title(pred_label)
    axes[0].set_xlabel('x0')
    axes[0].set_ylabel('x1')
    plt.colorbar(sc, ax=axes[0])

    sc = axes[1].scatter(
        x[:, 0], x[:, 1], c=mag_gt, **skw)
    axes[1].set_title('Ground Truth')
    axes[1].set_xlabel('x0')
    axes[1].set_ylabel('x1')
    plt.colorbar(sc, ax=axes[1])

    err = np.abs(mag_pred - mag_gt)
    sc = axes[2].scatter(
        x[:, 0], x[:, 1], c=err, s=2, cmap='hot')
    axes[2].set_title('Pointwise Error')
    axes[2].set_xlabel('x0')
    axes[2].set_ylabel('x1')
    plt.colorbar(sc, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------
#  Expert grid: GT | composed | error | base | E1 … EN
# -----------------------------------------------------------------

def _plot_experts_1d(x, t, decomp, n_experts, regions,
                     title, save_path):
    has_base = 'base' in decomp
    n_panels = 3 + (1 if has_base else 0) + n_experts
    n_cols = 4
    n_rows = max(math.ceil(n_panels / n_cols), 1)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).flatten()
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

    gt_mag = _to_magnitude(decomp['gt'])
    composed_mag = _to_magnitude(decomp['composed'])

    x_lin, t_lin, Z_gt = _grid_1d(x, t, gt_mag)
    _, _, Z_comp = _grid_1d(x, t, composed_mag)
    Z_err = np.abs(Z_comp - Z_gt)

    vmin = np.nanmin(Z_gt)
    vmax = np.nanmax(Z_gt)
    kw = dict(shading='auto', vmin=vmin, vmax=vmax,
              cmap='viridis')

    im = axes[0].pcolormesh(x_lin, t_lin, Z_gt.T, **kw)
    axes[0].set_title('Ground Truth', fontsize=10)
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    im = axes[1].pcolormesh(x_lin, t_lin, Z_comp.T, **kw)
    axes[1].set_title('Composed (full model)', fontsize=10)
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    im = axes[2].pcolormesh(
        x_lin, t_lin, Z_err.T, shading='auto', cmap='hot')
    axes[2].set_title('|Error|', fontsize=10)
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    panel_idx = 3
    if has_base:
        base_mag = _to_magnitude(decomp['base'])
        _, _, Z_base = _grid_1d(x, t, base_mag)
        im = axes[panel_idx].pcolormesh(x_lin, t_lin, Z_base.T, **kw)
        axes[panel_idx].set_title('Base (backbone only)', fontsize=10)
        plt.colorbar(im, ax=axes[panel_idx], fraction=0.046)
        panel_idx += 1

    for i in range(n_experts):
        ax = axes[panel_idx + i]
        key = f'expert_{i}'
        raw = decomp[key]
        mask = decomp['masks'].get(key)
        weighted = raw * mask if mask is not None else raw
        mag = _to_magnitude(weighted)
        _, _, Z_e = _grid_1d(x, t, mag)
        im = ax.pcolormesh(
            x_lin, t_lin, Z_e.T,
            shading='auto', cmap='viridis')
        etitle = f'E{i + 1}'
        if i < len(regions):
            r = regions[i]
            bl = [f'{v:.2f}' for v in r.bounds_lower]
            bu = [f'{v:.2f}' for v in r.bounds_upper]
            etitle += (
                f' [{",".join(bl)}]'
                f'\u2192[{",".join(bu)}]')
        ax.set_title(etitle, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)

        if i < len(regions):
            r = regions[i]
            rect = Rectangle(
                (r.bounds_lower[0], r.bounds_lower[1]),
                r.bounds_upper[0] - r.bounds_lower[0],
                r.bounds_upper[1] - r.bounds_lower[1],
                linewidth=2, edgecolor='red',
                facecolor='none', linestyle='--')
            ax.add_patch(rect)

    for ax in axes[:n_panels]:
        ax.set_xlabel('x', fontsize=8)
        ax.set_ylabel('t', fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()


def _plot_experts_2d(x, t, decomp, n_experts, regions,
                     title, save_path):
    has_base = 'base' in decomp
    n_panels = 3 + (1 if has_base else 0) + n_experts
    n_cols = 4
    n_rows = max(math.ceil(n_panels / n_cols), 1)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).flatten()
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

    gt_mag = _to_magnitude(decomp['gt'])
    comp_mag = _to_magnitude(decomp['composed'])
    err = np.abs(comp_mag - gt_mag)

    vmin, vmax = gt_mag.min(), gt_mag.max()
    skw = dict(s=2, cmap='viridis', vmin=vmin, vmax=vmax)

    axes[0].scatter(x[:, 0], x[:, 1], c=gt_mag, **skw)
    axes[0].set_title('Ground Truth', fontsize=10)
    axes[1].scatter(x[:, 0], x[:, 1], c=comp_mag, **skw)
    axes[1].set_title('Composed (full model)', fontsize=10)
    axes[2].scatter(x[:, 0], x[:, 1], c=err, s=2, cmap='hot')
    axes[2].set_title('|Error|', fontsize=10)

    panel_idx = 3
    if has_base:
        base_mag = _to_magnitude(decomp['base'])
        axes[panel_idx].scatter(x[:, 0], x[:, 1], c=base_mag, **skw)
        axes[panel_idx].set_title('Base (backbone only)', fontsize=10)
        panel_idx += 1

    for i in range(n_experts):
        ax = axes[panel_idx + i]
        key = f'expert_{i}'
        raw = decomp[key]
        mask = decomp['masks'].get(key)
        weighted = raw * mask if mask is not None else raw
        mag = _to_magnitude(weighted)
        ax.scatter(x[:, 0], x[:, 1], c=mag, s=2,
                   cmap='viridis')
        ax.set_title(f'E{i + 1}', fontsize=9)

    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------
#  Per-run processing
# -----------------------------------------------------------------

def process_run(label, ts_dir):
    """Return (composed_path, base_path) or (None, None)."""
    cfg_path = ts_dir / 'config_used.yaml'
    if not cfg_path.exists():
        print(f"  [{label}] No config_used.yaml, skipping")
        return None, None

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    problem = cfg['problem']
    problem_cfg = cfg[problem]
    is_adaptive = cfg.get(
        'adaptive_pinn', {}).get('enabled', False)

    ckpt_path = _find_checkpoint(ts_dir, cfg)
    if ckpt_path is None:
        print(f"  [{label}] No checkpoint found, skipping")
        return None, None

    ckpt_name = ckpt_path.name
    print(f"  [{label}] Loading {ckpt_name}...")
    model = _build_model(cfg)
    epoch = _load_checkpoint(model, ckpt_path, is_adaptive)
    model.eval()

    # Get domain bounds and create dense evaluation grid (mimics in-run plots)
    spatial_dim = problem_cfg.get('spatial_dim', 1)
    output_dim = problem_cfg.get('output_dim', 1)
    
    if spatial_dim == 1:
        x_min, x_max = problem_cfg['spatial_domain'][0]
        t_min, t_max = problem_cfg['temporal_domain']
        
        # Dense grid (200x200 like in-run plots)
        resolution = 200
        x_grid = np.linspace(x_min, x_max, resolution)
        t_grid = np.linspace(t_min, t_max, resolution)
        X, T = np.meshgrid(x_grid, t_grid, indexing='ij')
        x_flat = X.ravel()
        t_flat = T.ravel()
        
        # Get ground truth from solver interpolator (high accuracy)
        gt_channels = _get_gt_from_solver(cfg, x_flat, t_flat, output_dim)
        
        # Fallback to eval_data if solver interpolator unavailable
        if gt_channels is None:
            eval_path = Path('datasets') / problem / 'eval_data.pt'
            if not eval_path.exists():
                print(f"  [{label}] No eval data and no solver, skipping")
                return None, None
            eval_data = torch.load(eval_path, map_location='cpu', weights_only=False)
            x_np = eval_data['x'].numpy()
            t_np = eval_data['t'].numpy()
            h_gt_np = eval_data['h_gt'].numpy()
        else:
            # Convert solver GT to numpy array
            h_gt_np = np.column_stack(gt_channels) if output_dim > 1 else gt_channels[0].reshape(-1, 1)
            x_np = x_flat.reshape(-1, 1)
            t_np = t_flat.reshape(-1, 1)
        
        # Evaluate model on dense grid
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        x_tensor = torch.from_numpy(x_np.astype(np.float32)).to(device=device, dtype=model_dtype)
        t_tensor = torch.from_numpy(t_np.astype(np.float32)).to(device=device, dtype=model_dtype)
        inputs = torch.cat([x_tensor, t_tensor], dim=1)
        
        with torch.no_grad():
            h_composed = model(inputs).cpu().numpy()
    else:
        # 2D spatial: fall back to eval_data approach
        eval_path = Path('datasets') / problem / 'eval_data.pt'
        if not eval_path.exists():
            print(f"  [{label}] No eval data (2D not supported with solver yet), skipping")
            return None, None
        eval_data = torch.load(eval_path, map_location='cpu', weights_only=False)
        x_np = eval_data['x'].numpy()
        t_np = eval_data['t'].numpy()
        h_gt_np = eval_data['h_gt'].numpy()
        with torch.no_grad():
            inputs = torch.cat([eval_data['x'], eval_data['t']], dim=1)
            h_composed = model(inputs).numpy()

    tag = f"{problem}  ({ckpt_name} @ epoch {epoch})"
    plot_fn = (_plot_triplet_1d if spatial_dim == 1
               else _plot_triplet_2d)
    x_arg = x_np[:, 0] if spatial_dim == 1 else x_np
    t_arg = t_np[:, 0] if spatial_dim == 1 else t_np

    # --- 1. Composed (full model) vs GT ---
    composed_path = ts_dir / "composed_pred_vs_gt.png"
    plot_fn(x_arg, t_arg, h_composed, h_gt_np,
            'Composed (full model)', tag, composed_path)
    print(f"    Saved {composed_path.name}")

    # --- 2 & 3. Decomposed: base-only vs GT + expert grid ---
    has_experts = (
        is_adaptive
        and hasattr(model, 'forward_decomposed')
        and hasattr(model, 'num_experts')
        and model.num_experts > 0
    )

    base_path = None
    if has_experts:
        with torch.no_grad():
            decomp_t = model.forward_decomposed(inputs)

        decomp = {}
        for k, v in decomp_t.items():
            if k == 'masks':
                decomp['masks'] = {
                    mk: mv.cpu().numpy()
                    for mk, mv in v.items()
                }
            elif k == 'weights_normalized':
                continue
            elif isinstance(v, torch.Tensor):
                decomp[k] = v.cpu().numpy()
        decomp['gt'] = h_gt_np

        # Base-only prediction vs GT (only if base exists in decomposition)
        if 'base' in decomp:
            h_base = decomp['base']
            base_path = ts_dir / "base_pred_vs_gt.png"
            plot_fn(x_arg, t_arg, h_base, h_gt_np,
                    'Base (backbone only)', tag, base_path)
            print(f"    Saved {base_path.name}")
        else:
            base_path = None
            print(f"    Skipped base plot (base not in composition for this model)")

        # Expert grid
        n_exp = model.num_experts
        regions = getattr(model, 'regions', [])
        exp_title = (
            f"{problem} \u2014 {n_exp} experts "
            f"({ckpt_name} @ epoch {epoch})")
        exp_path = ts_dir / "experts_pred_vs_gt.png"

        if spatial_dim == 1:
            _plot_experts_1d(
                x_np[:, 0], t_np[:, 0], decomp,
                n_exp, regions, exp_title, exp_path)
        else:
            _plot_experts_2d(
                x_np, t_np, decomp,
                n_exp, regions, exp_title, exp_path)
        print(f"    Saved {exp_path.name} ({n_exp} experts)")
    else:
        base_path = ts_dir / "base_pred_vs_gt.png"
        plot_fn(x_arg, t_arg, h_composed, h_gt_np,
                'Base (backbone only)', tag, base_path)
        print(f"    Saved {base_path.name} (no experts, base = composed)")

    return composed_path, base_path


# -----------------------------------------------------------------
#  Summary grids
# -----------------------------------------------------------------

def _make_summary(batch_path, items, title_prefix, filename):
    valid = [(lbl, p) for lbl, p in items if p is not None]
    if not valid:
        return

    n = len(valid)
    fig, axes = plt.subplots(n, 1, figsize=(18, 5.5 * n))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f'{title_prefix} \u2014 {batch_path.name}',
        fontsize=16, fontweight='bold', y=1.0)

    for ax, (lbl, img_path) in zip(axes, valid):
        img = plt.imread(str(img_path))
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(lbl, fontsize=12, fontweight='bold')

    plt.tight_layout()
    out = batch_path / filename
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Summary saved: {out}")


# -----------------------------------------------------------------
#  Entry point
# -----------------------------------------------------------------

def main(batch_dir):
    batch_path = Path(batch_dir)
    if not batch_path.exists():
        print(f"Error: {batch_path} not found")
        return

    print("\n" + "=" * 70)
    print("Model Predictions vs Ground Truth")
    print(f"Batch: {batch_path.name}")
    print("=" * 70)

    runs = _find_run_dirs(batch_path)
    if not runs:
        print("No runs found")
        return

    print(f"Found {len(runs)} run(s)\n")

    composed_results = []
    base_results = []
    for label, ts_dir in runs:
        comp_p, base_p = process_run(label, ts_dir)
        composed_results.append((label, comp_p))
        base_results.append((label, base_p))

    print()
    _make_summary(batch_path, composed_results,
                  'Composed (full model) vs GT',
                  'composed_predictions_summary.png')
    _make_summary(batch_path, base_results,
                  'Base (backbone only) vs GT',
                  'base_predictions_summary.png')

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python plot_predictions.py "
              "<experiment_batch_dir>")
