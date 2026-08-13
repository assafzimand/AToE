"""Standalone: capacity-density maps, paper-styled (bigger/bold axis + colorbar text).

Two source kinds, auto-detected per input path:
  - A perfect_tree_examples/sweep/<problem>/<combo>/ dir (contains
    perfect_trees.json): each accepted node becomes a leaf expert with a
    synthetic MLP architecture [in_dim, 20, 20, 20, out_dim].
  - A live training run dir (contains metrics.json + config_used.yaml, e.g.
    .../allen_cahn-.../20260724_142510): uses the REAL trained expert
    architectures/param counts and the run's own (top-level, pre-fine_tune-
    migration) adaptive_pinn.sigma_fraction -- same fields
    scripts/make_capacity_map.py reads, just with paper styling and both
    overlap variants.

For each source, produces two maps:
  - capacity_map_<tag>_no_overlap.png
        Parameter density per exclusive region (hard tiling): each leaf
        counts only over its own box, matching a plain weights/volume ratio.
  - capacity_map_<tag>_overlap_sigma<S>.png
        Same, but each leaf's ACTIVE support is widened by the fixed PoU
        collar delta = sigma_fraction * region_extent (matching
        utils/dataset_gen.py's fixed-collar formula) on every side, so
        overlapping collars sum their capacity -- what the code actually
        does at sigma_fraction=<S>.

Rendering is a local paper-styled reimplementation of
adaptive.visualization.plot_capacity_map (same density computation, bigger/
bold tick and colorbar text) -- the shared function itself is untouched.

Usage:
    python scripts/plot_capacity_maps_from_trees.py <source_dir> [<source_dir> ...] \\
        [--sigma-fraction 0.05] [--out-dir outputs/paper_figures/capacity_maps]

    --sigma-fraction only applies to tree combo_dirs (no sigma_fraction of
    their own); run dirs always use their own config's value.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from perfect_tree_examples import create_prefect_trees as cpt
from utils.plot_io import save_png

TICK_SIZE = 15
LABEL_SIZE = 18
CBAR_TICK_SIZE = 13
CBAR_LABEL_SIZE = 15


def mlp_param_count(layers):
    return sum(a * b + b for a, b in zip(layers[:-1], layers[1:]))


def _style_axes(ax):
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.xaxis.label.set_fontsize(LABEL_SIZE)
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontsize(LABEL_SIZE)
    ax.yaxis.label.set_fontweight('bold')


def _style_colorbar(cbar):
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight('bold')
    cbar.ax.yaxis.label.set_fontsize(CBAR_LABEL_SIZE)
    cbar.ax.yaxis.label.set_fontweight('bold')


def plot_capacity_map_paper(regions, expert_params, leaf_indices, domain_bounds,
                            output_path, n_grid=300, delta_lo=None, delta_hi=None):
    """Paper-styled reimplementation of adaptive.visualization.plot_capacity_map."""
    lower, upper = domain_bounds['lower'], domain_bounds['upper']

    def _bounds(r):
        return (r['bounds_lower'], r['bounds_upper']) if isinstance(r, dict) \
            else (r.bounds_lower, r.bounds_upper)

    x_grid = np.linspace(lower[0], upper[0], n_grid)
    t_grid = np.linspace(lower[1], upper[1], n_grid)
    X, T = np.meshgrid(x_grid, t_grid, indexing='ij')
    pts = np.column_stack([X.ravel(), T.ravel()])

    if delta_lo is not None:
        delta_lo = np.asarray(delta_lo, dtype=float)
        delta_hi = np.asarray(delta_hi, dtype=float)

    leaf_set = set(int(i) for i in leaf_indices)
    density = np.zeros(pts.shape[0])
    supports = {}
    for i in sorted(leaf_set):
        if i >= len(regions) or i >= len(expert_params):
            continue
        lo, hi = _bounds(regions[i])
        vol = 1.0
        for a, b in zip(lo, hi):
            vol *= max(b - a, 1e-12)
        if delta_lo is not None and i < delta_lo.shape[0]:
            s_lo = np.asarray(lo, dtype=float) - delta_lo[i]
            s_hi = np.asarray(hi, dtype=float) + delta_hi[i]
        else:
            s_lo, s_hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        supports[i] = (s_lo, s_hi)
        mask = np.all((pts >= s_lo) & (pts <= s_hi), axis=1)
        density[mask] += expert_params[i] / vol

    density = density.reshape(n_grid, n_grid)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(density.T, extent=[x_grid[0], x_grid[-1], t_grid[0], t_grid[-1]],
                   origin='lower', aspect='auto', cmap='YlOrRd')
    label = ('Active params / unit volume (overlap-summed)'
             if delta_lo is not None else 'Parameters / unit volume')
    cbar = plt.colorbar(im, ax=ax, label=label)
    _style_colorbar(cbar)

    for i, region in enumerate(regions):
        lo, hi = _bounds(region)
        is_leaf = i in leaf_set
        color, lw, ls = ('red', 1.5, '-') if is_leaf else ('grey', 1.0, '--')
        ax.add_patch(patches.Rectangle(
            (lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
            linewidth=lw, edgecolor=color, facecolor='none', linestyle=ls))
    for i, (s_lo, s_hi) in supports.items():
        ax.add_patch(patches.Rectangle(
            (s_lo[0], s_lo[1]), s_hi[0] - s_lo[0], s_hi[1] - s_lo[1],
            linewidth=0.7, edgecolor='#1f77b4', facecolor='none', linestyle=':'))

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    _style_axes(ax)

    plt.tight_layout()
    save_png(output_path, fig=fig)
    plt.close(fig)
    print(f"  saved {Path(output_path).name} "
          f"(peak {density.max():.4g} params/vol)")


def _overlap_deltas(regions, leaf_indices, in_dim, sigma_fraction):
    n = len(regions)
    delta_lo = np.zeros((n, in_dim))
    delta_hi = np.zeros((n, in_dim))
    for i in leaf_indices:
        r = regions[i]
        lo = np.asarray(r['bounds_lower'] if isinstance(r, dict) else r.bounds_lower, dtype=float)
        hi = np.asarray(r['bounds_upper'] if isinstance(r, dict) else r.bounds_upper, dtype=float)
        d = np.maximum(sigma_fraction * (hi - lo), 1e-6)
        delta_lo[i] = d
        delta_hi[i] = d
    return delta_lo, delta_hi


def process_combo(combo_dir: Path, base_cfg, sigma_fraction: float, out_dir: Path):
    json_path = combo_dir / 'perfect_trees.json'
    data = json.loads(json_path.read_text(encoding='utf-8'))
    problem = next(iter(data))
    tree = data[problem]
    domain_bounds = tree['domain_bounds']
    regions = [n for n in tree['all_nodes'] if n.get('accepted')]
    params = tree['tree_params']
    n_leaves = len(regions)

    in_dim = len(domain_bounds['lower'])
    out_dim = base_cfg[problem].get('output_dim', 1)
    expert_arch = [in_dim, 20, 20, 20, out_dim]
    params_per_leaf = mlp_param_count(expert_arch)
    expert_params = [params_per_leaf] * n_leaves
    leaf_indices = list(range(n_leaves))
    total_leaf_params = params_per_leaf * n_leaves

    W = params.get('num_windows')
    M = params['global_M']
    dist = params.get('m_distribution')
    tag = (f"{problem}_" + (f"W{W}_" if W is not None else '')
          + (f"{dist}_" if dist else '') + f"M{M}_L{n_leaves}_leafp{total_leaf_params}")

    print(f"{combo_dir}  ({n_leaves} leaves, arch={expert_arch}, "
          f"{params_per_leaf} params/leaf)")

    plot_capacity_map_paper(
        regions, expert_params, leaf_indices, domain_bounds,
        out_dir / f'capacity_map_{tag}_no_overlap.png')

    sigma_tag = f"{sigma_fraction:g}".replace('.', 'p')
    delta_lo, delta_hi = _overlap_deltas(regions, leaf_indices, in_dim, sigma_fraction)
    plot_capacity_map_paper(
        regions, expert_params, leaf_indices, domain_bounds,
        out_dir / f'capacity_map_{tag}_overlap_sigma{sigma_tag}.png',
        delta_lo=delta_lo, delta_hi=delta_hi)


def process_run(run_dir: Path, out_dir: Path):
    cfg = yaml.safe_load((run_dir / 'config_used.yaml').read_text(encoding='utf-8'))
    metrics = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
    problem = cfg['problem']
    pc = cfg[problem]
    domain_bounds = {
        'lower': [pc['spatial_domain'][0][0], pc['temporal_domain'][0]],
        'upper': [pc['spatial_domain'][0][1], pc['temporal_domain'][1]],
    }
    sigma_fraction = cfg['adaptive_pinn'].get('sigma_fraction')
    if sigma_fraction is None:
        sigma_fraction = (cfg['adaptive_pinn'].get('fine_tune') or {}).get('sigma_fraction')
    if sigma_fraction is None:
        raise SystemExit(f"No sigma_fraction found in {run_dir}/config_used.yaml")

    ap = metrics['adaptive_pinn']
    regions = ap['regions']
    expert_params = ap['expert_params']
    leaf_indices = ap.get('leaf_expert_indices', list(range(len(regions))))
    base_params = ap.get('base_params', 0)
    n_leaves = len(leaf_indices)
    total_leaf_params = sum(expert_params[i] for i in leaf_indices)
    epoch = max((regions[i].get('spawn_epoch', 0) for i in leaf_indices), default=0)

    print(f"{run_dir}  ({problem}, {n_leaves} leaves, sigma_fraction={sigma_fraction}, "
          f"base_params={base_params}, leaf_params={total_leaf_params})")

    tag = f"{problem}_epoch{epoch}_L{n_leaves}_leafp{total_leaf_params}_rootp{base_params}"

    plot_capacity_map_paper(
        regions, expert_params, leaf_indices, domain_bounds,
        out_dir / f'capacity_map_{tag}_no_overlap.png')

    sigma_tag = f"{sigma_fraction:g}".replace('.', 'p')
    in_dim = len(domain_bounds['lower'])
    delta_lo, delta_hi = _overlap_deltas(regions, leaf_indices, in_dim, sigma_fraction)
    plot_capacity_map_paper(
        regions, expert_params, leaf_indices, domain_bounds,
        out_dir / f'capacity_map_{tag}_overlap_sigma{sigma_tag}.png',
        delta_lo=delta_lo, delta_hi=delta_hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source_dirs', nargs='+', type=Path)
    ap.add_argument('--sigma-fraction', type=float, default=0.05,
                    help='Only used for tree combo_dirs (no sigma_fraction of their own)')
    ap.add_argument('--out-dir', type=Path,
                     default=Path('outputs/paper_figures/capacity_maps'))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = None
    for source_dir in args.source_dirs:
        if (source_dir / 'perfect_trees.json').exists():
            if base_cfg is None:
                base_cfg = cpt.load_config(REPO_ROOT / 'experiments_plan.yaml')
            process_combo(source_dir, base_cfg, args.sigma_fraction, args.out_dir)
        elif (source_dir / 'metrics.json').exists() and (source_dir / 'config_used.yaml').exists():
            process_run(source_dir, args.out_dir)
        else:
            print(f"Skipping {source_dir}: no perfect_trees.json or metrics.json+config_used.yaml found")


if __name__ == '__main__':
    main()
