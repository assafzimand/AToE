"""Re-evaluate phase-3 best checkpoints with HARD vs SOFT indicators.

Diagnostic for the owner-imitator runs: during phase 3 every loss term is on
an expert's own raw output, but eval uses the SOFT PoU readout, which mixes
imitators into the collars. This script quantifies how much of the eval error
is the "blend tax" vs genuine owner error.

Per run (each timestamp dir with checkpoints/best_model_phase3.pt):
  * native-grid rel-L2 / inf-norm (identical metric to the training logs) for
      - soft PoU readout (must reproduce the logged phase-3 best)
      - hard owner-only readout
      - the frozen base (root) alone
  * error breakdown: blend-zone (union of collars) vs tile-interior points —
    share of points vs share of squared error, RMS per zone
  * per-expert table: RMS of the expert's RAW output inside its own tile
    (owner quality) and inside its collar ring (imitator quality)
  * heatmaps: log|err| soft / hard / root + log|u_soft - u_hard|, with tile
    boundaries overlaid

Run from the repo root on branch AToE (the branch these checkpoints were
trained with):
    python scripts/eval_phase3_hard_indicators.py <experiment_batch_dir>

Outputs go to <ts_dir>/hard_eval/ plus a batch-level hard_eval_summary.csv.
"""

import sys
import csv
import math
import importlib
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent.parent))

CKPT_NAME = 'best_model_phase3.pt'
TOL = 1e-12


# -----------------------------------------------------------------
#  Model loading (mirrors scripts/plot_experts_predictions.py)
# -----------------------------------------------------------------

def _build_and_load(cfg, ckpt_path):
    from models.atoe_leaves import AToELeaves

    if cfg.get('precision', 'float32') == 'float64':
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    model = AToELeaves(cfg['base_architecture'], cfg.get('activation', 'tanh'),
                       cfg, cfg['adaptive_pinn'])
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if not checkpoint.get('is_adaptive', False):
        raise RuntimeError(f"{ckpt_path} is not an adaptive checkpoint")
    model.load_state_dict_extended(checkpoint['adaptive_state'])
    model.eval()
    return model, checkpoint.get('epoch', '?')


# -----------------------------------------------------------------
#  Native solver grid (same source as compute_native_grid_metrics)
# -----------------------------------------------------------------

def _native_grid(cfg):
    problem = cfg['problem']
    solver = importlib.import_module(f'solvers.{problem}_solver')
    x_grid, t_grid, h_sol = solver._get_solution_cached(cfg)

    t0, t1 = cfg[problem]['temporal_domain']
    t_mask = (np.asarray(t_grid) >= t0 - TOL) & (np.asarray(t_grid) <= t1 + TOL)
    t_grid = np.asarray(t_grid)[t_mask]
    h_sol = np.asarray(h_sol)[t_mask]          # (nt, nx)
    x_grid = np.asarray(x_grid)

    T, X = np.meshgrid(t_grid, x_grid, indexing='ij')
    xt = np.column_stack([X.ravel(), T.ravel()])   # (N, 2), same order as trainer
    if np.iscomplexobj(h_sol):
        gt = np.column_stack([h_sol.real.ravel(), h_sol.imag.ravel()])
    else:
        gt = h_sol.reshape(-1, 1)
    return x_grid, t_grid, xt, gt


def _forward_chunked(fn, xt, out_dim, dtype, chunk=65536):
    out = np.empty((xt.shape[0], out_dim))
    with torch.no_grad():
        for s in range(0, xt.shape[0], chunk):
            xb = torch.tensor(xt[s:s + chunk], dtype=dtype)
            out[s:s + chunk] = fn(xb).cpu().numpy()
    return out


def _rel_l2(pred, gt):
    return math.sqrt(((pred - gt) ** 2).sum()) / (math.sqrt((gt ** 2).sum()) + 1e-10)


def _rms(pred, gt):
    return math.sqrt(((pred - gt) ** 2).mean())


# -----------------------------------------------------------------
#  Geometry masks
# -----------------------------------------------------------------

def _in_box(xt, lo, hi):
    m = np.ones(xt.shape[0], dtype=bool)
    for d in range(xt.shape[1]):
        m &= (xt[:, d] >= lo[d] - TOL) & (xt[:, d] <= hi[d] + TOL)
    return m


def _leaf_masks(model, cfg, xt):
    """Per-leaf hard-tile and collar-ring masks + first-match owner id."""
    from adaptive.indicators import inflated_bounds

    problem_cfg = cfg[cfg['problem']]
    g_lo = [problem_cfg['spatial_domain'][0][0], problem_cfg['temporal_domain'][0]]
    g_hi = [problem_cfg['spatial_domain'][0][1], problem_cfg['temporal_domain'][1]]
    sigma = cfg['adaptive_pinn']['sigma_fraction']

    leaf_ids = sorted(model.leaf_indices)
    tile, collar = {}, {}
    owner = np.full(xt.shape[0], -1, dtype=int)
    for j in leaf_ids:
        r = model.regions[j]
        tile[j] = _in_box(xt, r.bounds_lower, r.bounds_upper)
        lo, hi = inflated_bounds(r, sigma, g_lo, g_hi)
        collar[j] = _in_box(xt, lo, hi) & ~tile[j]
        unclaimed = (owner == -1) & tile[j]
        owner[unclaimed] = j
    if (owner == -1).any():
        raise RuntimeError(f"{(owner == -1).sum()} grid points have no owner tile")
    return leaf_ids, tile, collar, owner


# -----------------------------------------------------------------
#  Per-run evaluation
# -----------------------------------------------------------------

def process_run(ts_dir):
    cfg_path = ts_dir / 'config_used.yaml'
    ckpt_path = ts_dir / 'checkpoints' / CKPT_NAME
    if not cfg_path.exists() or not ckpt_path.exists():
        print(f"  [{ts_dir.name}] missing config or {CKPT_NAME}, skipping")
        return None
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    tag = cfg.get('experiment_tag', ts_dir.name)
    print(f"\n=== {tag} ({ts_dir.name}) ===")

    model, epoch = _build_and_load(cfg, ckpt_path)
    dtype = next(model.parameters()).dtype
    print(f"  loaded {CKPT_NAME} @ epoch {epoch} | {model.num_experts} experts | "
          f"configured blending: {model.blending_mode}")

    x_grid, t_grid, xt, gt = _native_grid(cfg)
    out_dim = gt.shape[1]
    nt, nx = len(t_grid), len(x_grid)
    print(f"  native grid {nt}x{nx} = {xt.shape[0]:,} points")

    # --- headline metrics: soft / hard / root, same grid & formula as logs ---
    orig_mode = model.blending_mode
    model.blending_mode = 'soft'
    pred_soft = _forward_chunked(model, xt, out_dim, dtype)
    model.blending_mode = 'hard'
    pred_hard = _forward_chunked(model, xt, out_dim, dtype)
    model.blending_mode = orig_mode
    pred_root = _forward_chunked(model.base_model, xt, out_dim, dtype)

    res = {'run': ts_dir.name, 'tag': tag, 'epoch': epoch}
    for name, pred in (('soft', pred_soft), ('hard', pred_hard), ('root', pred_root)):
        res[f'rel_l2_{name}'] = _rel_l2(pred, gt)
        res[f'inf_{name}'] = float(np.abs(pred - gt).max())
        print(f"  rel-L2 {name:5s} = {res[f'rel_l2_{name}']:.6e}   "
              f"inf = {res[f'inf_{name}']:.3e}")

    # --- blend-zone vs interior breakdown ---
    leaf_ids, tile, collar, owner = _leaf_masks(model, cfg, xt)
    blend_zone = np.zeros(xt.shape[0], dtype=bool)
    for j in leaf_ids:
        blend_zone |= collar[j]
    interior = ~blend_zone

    print(f"\n  blend zone: {blend_zone.mean() * 100:.1f}% of points")
    for name, pred in (('soft', pred_soft), ('hard', pred_hard)):
        sq = ((pred - gt) ** 2).sum(axis=1)
        share = sq[blend_zone].sum() / sq.sum() * 100
        res[f'blend_err_share_{name}'] = share
        print(f"  {name:5s}: {share:5.1f}% of squared error in blend zone | "
              f"RMS blend={math.sqrt(sq[blend_zone].mean()):.3e} "
              f"interior={math.sqrt(sq[interior].mean()):.3e}")

    # --- per-expert owner vs imitator quality (raw expert outputs) ---
    print(f"\n  {'expert':>6} {'tile pts':>9} {'own-tile RMS':>13} "
          f"{'collar pts':>10} {'collar RMS':>12}  (raw expert vs GT)")
    for j in leaf_ids:
        row = {'expert': j}
        for zone_name, mask in (('tile', tile[j] & (owner == j)), ('collar', collar[j])):
            if mask.any():
                pred_j = _forward_chunked(
                    lambda xb, j=j: model.forward_single_expert(j, xb),
                    xt[mask], out_dim, dtype)
                row[zone_name] = (int(mask.sum()), _rms(pred_j, gt[mask]))
            else:
                row[zone_name] = (0, float('nan'))
        print(f"  {j:>6} {row['tile'][0]:>9} {row['tile'][1]:>13.3e} "
              f"{row['collar'][0]:>10} {row['collar'][1]:>12.3e}")

    # --- heatmaps ---
    out_dir = ts_dir / 'hard_eval'
    out_dir.mkdir(exist_ok=True)

    def _mag(a):
        return np.linalg.norm(a, axis=1) if a.shape[1] > 1 else np.abs(a[:, 0])

    panels = [
        (f'|err| SOFT readout  relL2={res["rel_l2_soft"]:.2e}', _mag(pred_soft - gt)),
        (f'|err| HARD readout  relL2={res["rel_l2_hard"]:.2e}', _mag(pred_hard - gt)),
        (f'|err| ROOT (base)   relL2={res["rel_l2_root"]:.2e}', _mag(pred_root - gt)),
        ('|u_soft - u_hard|  (blend perturbation)', _mag(pred_soft - pred_hard)),
    ]
    all_vals = np.concatenate([p[1] for p in panels])
    vmin = max(all_vals[all_vals > 0].min(), 1e-12)
    vmax = all_vals.max()

    fig, axes = plt.subplots(1, 4, figsize=(26, 5.5))
    fig.suptitle(f'{tag} — {CKPT_NAME} @ epoch {epoch} (native grid {nt}x{nx})',
                 fontsize=14, fontweight='bold')
    for ax, (title, vals) in zip(axes, panels):
        Z = vals.reshape(nt, nx)
        im = ax.pcolormesh(x_grid, t_grid, np.maximum(Z, vmin), shading='auto',
                           cmap='hot', norm=LogNorm(vmin=vmin, vmax=vmax))
        for j in leaf_ids:
            r = model.regions[j]
            ax.add_patch(Rectangle(
                (r.bounds_lower[0], r.bounds_lower[1]),
                r.bounds_upper[0] - r.bounds_lower[0],
                r.bounds_upper[1] - r.bounds_lower[1],
                linewidth=0.8, edgecolor='cyan', facecolor='none', linestyle='--'))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig_path = out_dir / 'phase3_hard_vs_soft.png'
    plt.savefig(fig_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"\n  saved {fig_path}")

    with open(out_dir / 'summary.txt', 'w') as f:
        for k, v in res.items():
            f.write(f"{k}: {v}\n")
    return res


def main(batch_dir):
    batch = Path(batch_dir)
    ts_dirs = sorted(d for d in batch.rglob('checkpoints')
                     if (d / CKPT_NAME).exists())
    if not ts_dirs:
        print(f"No {CKPT_NAME} found anywhere under {batch}")
        return
    results = []
    for ckpt_dir in ts_dirs:
        r = process_run(ckpt_dir.parent)
        if r:
            results.append(r)

    if results:
        csv_path = batch / 'hard_eval_summary.csv'
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"\nBatch summary -> {csv_path}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python scripts/eval_phase3_hard_indicators.py "
              "<experiment_batch_dir>")
