"""Standalone: paper-styled re-render of a windowed perfect-tree sweep figure.

Each perfect_tree_examples/sweep/<problem>_*/<combo>/ folder holds the
2-panel comparison PNG (all candidate tree nodes | accepted after global
top-M pruning) plus the perfect_trees.json that produced it. This re-renders
that same 2-panel figure straight from the JSON -- no titles, bigger/bold
tick, axis-label and colorbar text -- using the solver's native ground-truth
grid as the backdrop, exactly like the original.

The temporal domain for the GT grid is taken from the JSON's domain_bounds,
so half-domain sweeps (e.g. ks_t05, t in [0, 0.5]) render on the domain they
were actually fit on regardless of what experiments_plan.yaml currently says.

Usage:
    python scripts/plot_window_trees_paper.py <combo_dir> [<combo_dir> ...] \\
        [--out-dir outputs/paper_figures/trees]
"""

import sys
import json
import copy
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from perfect_tree_examples import create_prefect_trees as cpt
from utils.plot_io import save_png

TICK_SIZE = 14
LABEL_SIZE = 17
CBAR_TICK_SIZE = 13


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


def plot_regions_panel_paper(ax, regions, domain_bounds, gt_grid, grid_x, grid_t,
                             linewidth=1.0):
    """Same drawing as cpt._plot_regions_panel (GT heatmap + region boxes),
    paper styling, no title."""
    x_min, t_min = domain_bounds['lower'][:2]
    x_max, t_max = domain_bounds['upper'][:2]

    display = np.linalg.norm(gt_grid, axis=2) if gt_grid.ndim == 3 else gt_grid
    T, X = np.meshgrid(grid_t, grid_x)
    im = ax.pcolormesh(X, T, display, shading='auto', cmap='viridis', alpha=0.7, zorder=0)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    _style_colorbar(cbar)

    for nd in regions:
        bl, bu = nd['bounds_lower'], nd['bounds_upper']
        ax.add_patch(patches.Rectangle(
            (bl[0], bl[-1]), bu[0] - bl[0], bu[-1] - bl[-1],
            linewidth=linewidth, edgecolor='black', facecolor='none', zorder=10))

    pad = 0.03
    ax.set_xlim(x_min - pad * (x_max - x_min), x_max + pad * (x_max - x_min))
    ax.set_ylim(t_min - pad * (t_max - t_min), t_max + pad * (t_max - t_min))
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_aspect('auto')
    _style_axes(ax)


def process_combo(combo_dir: Path, base_cfg, out_dir: Path):
    data = json.loads((combo_dir / 'perfect_trees.json').read_text(encoding='utf-8'))
    problem = next(iter(data))
    tree = data[problem]
    domain_bounds = tree['domain_bounds']
    all_nodes = tree['all_nodes']
    accepted = [n for n in all_nodes if n.get('accepted')]
    params = tree['tree_params']
    n_acc = tree['summary']['accepted_nodes']

    cfg = copy.deepcopy(base_cfg)
    cfg[problem]['temporal_domain'] = [float(domain_bounds['lower'][-1]),
                                       float(domain_bounds['upper'][-1])]
    gt = cpt.build_native_grid_data(problem, cfg)
    if gt is None:
        raise SystemExit(f"No native solver grid available for '{problem}'")
    _, _, (gt_grid, grid_x, grid_t) = gt

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    plot_regions_panel_paper(axes[0], all_nodes, domain_bounds, gt_grid, grid_x, grid_t,
                             linewidth=1.0)
    plot_regions_panel_paper(axes[1], accepted, domain_bounds, gt_grid, grid_x, grid_t,
                             linewidth=1.5)
    plt.tight_layout()

    W = params.get('num_windows')
    M = params['global_M']
    tag = f"W{W}_" if W is not None else ''
    out_path = out_dir / f"tree_{problem}_{tag}M{M}_acc{n_acc}_2panel.png"
    save_png(out_path, fig=fig)
    plt.close(fig)
    print(f"  saved {out_path.name}  ({len(all_nodes)} candidate nodes, {n_acc} accepted)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('combo_dirs', nargs='+', type=Path)
    ap.add_argument('--out-dir', type=Path, default=Path('outputs/paper_figures/trees'))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = cpt.load_config(REPO_ROOT / 'experiments_plan.yaml')
    for combo_dir in args.combo_dirs:
        print(combo_dir)
        process_combo(combo_dir, base_cfg, args.out_dir)


if __name__ == '__main__':
    main()
