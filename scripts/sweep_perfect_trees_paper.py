"""Paper variant of sweep_perfect_trees.py: no headers, bigger/bold axis and
colorbar text. Same data pipeline (perfect_tree_examples.create_prefect_trees),
just monkeypatches its two panel-drawing functions with paper-styled versions
before calling process_problem() for each combo, so the tree-fitting logic
is untouched -- only the rendering changes.

Fixed combos (no time-marching windows for kdv/ks this time):
    allen_cahn, burgers1d, schrodinger : M = 20
    kdv, ks                            : M = 30

Dataset: also monkeypatches build_native_grid_data() to always report "no
native grid", forcing process_problem()'s fallback path -- fit on a
STANDALONE symmetric grid (build_symmetric_grid_data, resolution=200 -> 200x200
= 40,000 points) interpolated from eval_data.pt, instead of the solver's full
native grid (e.g. 512K points for Allen-Cahn). This matches the dataset size
that produced perfect_tree_examples/first_sweep_run, before the pipeline
switched to native-grid metrics; min_samples_leaf stays literal (50), not
rescaled the way the pre-"Literal tree min_samples_leaf" code did.

Usage (from the repo root):
    python scripts/sweep_perfect_trees_paper.py
"""

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from perfect_tree_examples import create_prefect_trees as cpt  # noqa: E402
from utils.plot_io import save_png  # noqa: E402

SWEEPS = {
    'allen_cahn': {'M': 20},
    'burgers1d': {'M': 20},
    'schrodinger': {'M': 20},
    'kdv': {'M': 30},
    'ks': {'M': 30},
}

OUTPUT_ROOT = REPO_ROOT / 'outputs' / 'paper_figures' / 'perfect_trees_sweep'

# first_sweep_run's pre-"literal" pipeline rescaled min_samples_leaf by
# (fit-grid size / eval-sample count), landing at 381 on this 40,000-point
# grid. That rescaling was removed from create_prefect_trees.py itself (now
# always literal), so it's fixed here instead at a round 400, ONLY for this
# standalone script -- the live training pipeline keeps today's literal 50.
MIN_SAMPLES_LEAF = 400

TICK_SIZE = 15
LABEL_SIZE = 18
CBAR_TICK_SIZE = 13
CBAR_LABEL_SIZE = 15


def _bold_ticks(ax):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')


def _style_axes(ax):
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    _bold_ticks(ax)
    if ax.xaxis.label.get_text():
        ax.xaxis.label.set_fontsize(LABEL_SIZE)
        ax.xaxis.label.set_fontweight('bold')
    if ax.yaxis.label.get_text():
        ax.yaxis.label.set_fontsize(LABEL_SIZE)
        ax.yaxis.label.set_fontweight('bold')


def _style_colorbar(cbar):
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight('bold')
    if cbar.ax.yaxis.label.get_text():
        cbar.ax.yaxis.label.set_fontsize(CBAR_LABEL_SIZE)
        cbar.ax.yaxis.label.set_fontweight('bold')


def _plot_regions_panel_paper(ax, regions_dicts, domain_bounds, gt_grid, grid_x, grid_t, title):
    """Same as cpt._plot_regions_panel, minus the title, plus bigger/bold text."""
    import matplotlib.patches as patches

    x_min, t_min = domain_bounds['lower'][:2]
    x_max, t_max = domain_bounds['upper'][:2]

    if gt_grid is not None and grid_x is not None and grid_t is not None:
        display = np.linalg.norm(gt_grid, axis=2) if gt_grid.ndim == 3 else gt_grid
        T, X = np.meshgrid(grid_t, grid_x)
        im = ax.pcolormesh(X, T, display, shading='auto', cmap='viridis', alpha=0.7, zorder=0)
        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        _style_colorbar(cbar)

    for nd in regions_dicts:
        bl, bu = nd['bounds_lower'], nd['bounds_upper']
        rx_min, rt_min = bl[0], bl[-1]
        rx_max, rt_max = bu[0], bu[-1]
        ax.add_patch(patches.Rectangle(
            (rx_min, rt_min), rx_max - rx_min, rt_max - rt_min,
            linewidth=1.0, edgecolor='black', facecolor='none', zorder=10))

    pad = 0.03
    xr, tr = x_max - x_min, t_max - t_min
    ax.set_xlim(x_min - pad * xr, x_max + pad * xr)
    ax.set_ylim(t_min - pad * tr, t_max + pad * tr)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_aspect('auto')
    _style_axes(ax)


def _plot_hierarchy_panel_paper(ax, all_nodes, variable_for_node_accept):
    """Same as cpt._plot_hierarchy_panel, minus the title, plus bigger/bold text."""
    if not all_nodes:
        return

    children_map, node_map = {}, {}
    for n in all_nodes:
        nid = n['node_id']
        pid = n.get('parent_node_id', -1)
        node_map[nid] = n
        children_map.setdefault(pid, []).append(n)

    root_children = children_map.get(0, [])
    if not root_children:
        root_children = [n for n in all_nodes if n.get('parent_node_id', -1) == -1]
    if not root_children:
        return

    leaf_counter = [0]
    positions = {}

    def _layout(nid):
        kids = children_map.get(nid, [])
        depth = node_map[nid]['tree_depth']
        if not kids:
            x = leaf_counter[0]
            leaf_counter[0] += 1
            positions[nid] = (x, depth)
            return x
        child_xs = [_layout(c['node_id']) for c in sorted(kids, key=lambda c: c['node_id'])]
        x = np.mean(child_xs)
        positions[nid] = (x, depth)
        return x

    rc_xs = [_layout(c['node_id']) for c in sorted(root_children, key=lambda c: c['node_id'])]
    root_x = np.mean(rc_xs)
    positions[0] = (root_x, 0)

    max_depth = max(n['tree_depth'] for n in all_nodes) if all_nodes else 1

    if variable_for_node_accept == 'smoothness':
        metric_values = [n['smoothness_alpha'] for n in all_nodes if n.get('smoothness_alpha') is not None]
        cmap = plt.get_cmap('RdYlGn')
        metric_label = 'Smoothness α'
        def get_metric(node):
            return node.get('smoothness_alpha')
    elif variable_for_node_accept == 'norm':
        metric_values = [n['wavelet_norm_squared'] for n in all_nodes if n.get('wavelet_norm_squared') is not None]
        cmap = plt.get_cmap('coolwarm')
        metric_label = 'Wavelet Norm²'
        def get_metric(node):
            return node.get('wavelet_norm_squared')
    elif variable_for_node_accept == 'new_norm':
        metric_values = [n['new_wavelet_norm_squared'] for n in all_nodes if n.get('new_wavelet_norm_squared') is not None]
        cmap = plt.get_cmap('coolwarm')
        metric_label = 'New Norm²'
        def get_metric(node):
            return node.get('new_wavelet_norm_squared')
    else:
        metric_values = []
        cmap = plt.get_cmap('viridis')
        metric_label = 'Value'
        def get_metric(node):
            return 0.0

    norm = (plt.Normalize(vmin=min(metric_values), vmax=max(metric_values))
           if metric_values else plt.Normalize(vmin=0.0, vmax=1.0))

    for n in all_nodes:
        nid = n['node_id']
        pid = n.get('parent_node_id', -1)
        if pid >= 0 and pid in positions and nid in positions:
            px, py = positions[pid]
            cx, cy = positions[nid]
            edge_alpha = 0.7 if n['accepted'] else 0.15
            ax.plot([px, cx], [-py, -cy], 'k-', alpha=edge_alpha, lw=0.6, zorder=1)

    ax.scatter(root_x, 0, c='white', s=60, edgecolors='black', linewidths=1.5, zorder=6, marker='s')

    for n in all_nodes:
        nid = n['node_id']
        if nid not in positions:
            continue
        x, y = positions[nid]
        metric_val = get_metric(n)
        if metric_val is None:
            color = ['#bbbbbb'] if n['accepted'] else ['#dddddd']
            ec = 'black' if n['accepted'] else 'gray'
            node_alpha = 1.0 if n['accepted'] else 0.3
        else:
            color = [cmap(norm(metric_val))]
            ec = 'black' if n['accepted'] else 'gray'
            node_alpha = 1.0 if n['accepted'] else 0.25
        size = 35 if n['accepted'] else 15
        lw = 0.8 if n['accepted'] else 0.3
        ax.scatter(x, -y, c=color, s=size, edgecolors=ec, linewidths=lw,
                  alpha=node_alpha, zorder=5 if n['accepted'] else 4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label=metric_label, shrink=0.7)
    _style_colorbar(cbar)

    ax.set_ylabel('Depth')
    ax.set_yticks([-d for d in range(max_depth + 1)])
    ax.set_yticklabels([str(d) for d in range(max_depth + 1)])
    ax.set_xticks([])
    ax.grid(True, alpha=0.15, axis='y')
    _style_axes(ax)


def main():
    cpt._plot_regions_panel = _plot_regions_panel_paper
    cpt._plot_hierarchy_panel = _plot_hierarchy_panel_paper
    # Force the eval-grid fallback (200x200 = 40,000-point standalone grid)
    # instead of the solver's native grid, matching first_sweep_run's dataset.
    cpt.build_native_grid_data = lambda problem, base_cfg: None

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base_cfg = cpt.load_config(REPO_ROOT / 'experiments_plan.yaml')

    summary = {}
    for problem, spec in SWEEPS.items():
        if problem not in base_cfg:
            print(f"Skipping {problem}: not in experiments_plan.yaml")
            continue
        M = spec['M']
        cfg = copy.deepcopy(base_cfg)
        cfg['adaptive_pinn']['M_experts_num'] = M
        # No time-marching windows for this sweep, regardless of the base yaml.
        cfg[problem].setdefault('time_marching', {})['enabled'] = False

        cfg['adaptive_pinn']['tree_min_samples_leaf'] = MIN_SAMPLES_LEAF

        print(f"\n{'#' * 60}\n#  {problem}  (M={M}, no time marching, "
              f"min_samples_leaf={MIN_SAMPLES_LEAF})\n{'#' * 60}")
        try:
            tree_data = cpt.process_problem(problem, cfg, OUTPUT_ROOT)
        except Exception as e:
            print(f"  ERROR processing {problem}: {e}")
            import traceback
            traceback.print_exc()
            continue
        if tree_data is None:
            continue
        summary[problem] = {
            'M': M,
            'accepted': tree_data['summary']['accepted_nodes'],
            'leaves': tree_data['summary']['pruned_tree_leaves'],
        }

    with open(OUTPUT_ROOT / 'sweep_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}\n{'problem':<14}{'M':>5}{'accepted':>10}{'leaves':>10}")
    for problem, res in summary.items():
        print(f"{problem:<14}{res['M']:>5}{res['accepted']:>10}{res['leaves']:>10}")
    print(f"\nResults in {OUTPUT_ROOT}")


if __name__ == '__main__':
    main()
