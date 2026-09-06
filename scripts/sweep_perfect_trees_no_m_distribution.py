"""Perfect trees for time-marching windows WITHOUT a per-window M split.

Standard windowed trees (create_prefect_trees / sweep_perfect_trees) give
each window its own budget m_k = compute_m_per_window(M, W, distribution)
and run an independent top-m_k selection per window. This script instead
lets the windows compete for ONE global budget:

  A. Per window: fit the full decision tree on the window's slice of the
     ground truth (full-domain box, exactly like the pipeline) and compute
     the wavelet norms of every node.
  B. Pool the non-root nodes of ALL windows, sort by the acceptance metric
     (adaptive_pinn.variable_for_node_accept, 'norm' = wavelet_norm_squared)
     and take the top M globally (+ epsilon ties at the cutoff, same rule
     as RegionDetector.fit_full_tree_and_prune).
  C. Close each window's selection inside its own tree: all ancestors,
     then siblings iteratively (retain_siblings=True). A window that
     received no top-M node keeps its ROOT: a single accepted leaf spanning
     the whole slice, the same convention the pipeline uses for an M=0
     slice (trainer.orchestrator, *_zero distributions).

The realized per-window allocation is recorded in tree_params so it can be
compared with the fixed distributions.

Output: perfect_tree_examples/sweep/{problem}_no_m_distribution/M_{M}_W_{W}/
  perfect_trees.json  (same schema as the standard windowed trees)
  perfect_tree_{problem}_W{W}_global_M{M}_acc{n}.png

Usage (from the repo root):
    python scripts/sweep_perfect_trees_no_m_distribution.py
"""

import copy
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from perfect_tree_examples import create_prefect_trees as cpt  # noqa: E402
from utils.plot_io import save_png  # noqa: E402

PROBLEM = 'kdv'
NUM_WINDOWS = 3
M_LIST = [10, 12, 15, 18, 20, 25, 30, 35]

OUTPUT_ROOT = (REPO_ROOT / 'perfect_tree_examples' / 'sweep'
               / f'{PROBLEM}_no_m_distribution')


# A/B/C live in adaptive.window_tree_selection — the SAME code the training
# pipeline runs (orchestrator structure-only windowed tree and
# trainer.time_marching.preselect_window_trees), so these reference trees
# are exactly what a run with the same M / windows would build on GT.
from adaptive.window_tree_selection import (  # noqa: E402
    fit_window_tree, select_global_top_m, close_window, build_window_result,
)


def window_outputs(w, res, win_idx):
    """JSON rows (all nodes + BFS accepted) from one window's result dict."""
    accepted_ids = {n.node_id for n, _ in res['accepted_nodes']}
    node_dicts = []
    for d in res['diag_nodes']:
        nd = w['node_lookup'][d['node_id']]
        node_dicts.append({
            **d,
            'smoothness_r2': nd.smoothness_r2,
            'smoothness_n_levels': nd.smoothness_n_levels,
            'accepted': bool(d['node_id'] in accepted_ids),
        })
    cl = res['children_left']
    bfs_accepted = []
    for n, parent in res['accepted_nodes']:
        child = cl.get(n.node_id, -1)
        bfs_accepted.append({
            'node_id': n.node_id,
            'parent_tree_node_id': parent,
            'bounds_lower': list(n.bounds_lower),
            'bounds_upper': list(n.bounds_upper),
            'wavelet_norm_squared': n.wavelet_norm_squared,
            'new_wavelet_norm_squared': n.new_wavelet_norm_squared,
            'smoothness_alpha': n.smoothness_alpha,
            'n_samples': n.n_samples,
            'tree_depth': res['node_tree_depth'].get(n.node_id, -1),
            'is_leaf_in_pruned_tree': bool(child == -1 or child not in accepted_ids),
            'window_idx': win_idx,
            'window_root_kept': bool(res['root_kept']),
        })
    return node_dicts, bfs_accepted


# ─────────────────────────────── driver ───────────────────────────────────

def main():
    base_cfg = cpt.load_config(REPO_ROOT / 'experiments_plan.yaml')
    problem_cfg = base_cfg[PROBLEM]
    adaptive_cfg = base_cfg.get('adaptive_pinn', {})

    max_depth = adaptive_cfg.get('tree_max_depth', 30)
    min_samples_leaf = adaptive_cfg.get('tree_min_samples_leaf', 10)
    variable = adaptive_cfg.get('variable_for_node_accept', 'norm')
    eps = float(adaptive_cfg.get('epsilon_node_acceptance', 0.0))
    domain_bounds = cpt.build_domain_bounds(problem_cfg)

    print(f"Problem {PROBLEM}: W={NUM_WINDOWS}, M in {M_LIST}, "
          f"max_depth={max_depth}, min_samples_leaf={min_samples_leaf}, "
          f"metric={variable}, eps={eps}")

    native = cpt.build_native_grid_data(PROBLEM, base_cfg)
    if native is not None:
        X_full, y_full, native_heatmap = native
        print("GT source: native solver grid")
    else:
        eval_data = cpt.ensure_eval_data(PROBLEM, base_cfg)
        X_full, y_full = cpt.build_symmetric_grid_data(eval_data, domain_bounds)
        native_heatmap = None
        print("GT source: interpolated eval sample")

    t_min, t_max = problem_cfg['temporal_domain']
    edges = np.linspace(float(t_min), float(t_max), NUM_WINDOWS + 1)

    # A — the fits do not depend on M: do them once.
    t0 = time.time()
    windows = []
    for k in range(NUM_WINDOWS):
        lo, hi = float(edges[k]), float(edges[k + 1])
        mask = (X_full[:, -1] >= lo - 1e-12) & (X_full[:, -1] <= hi + 1e-12)
        print(f"\n[A] window {k}: t in [{lo:.4f}, {hi:.4f}], "
              f"{int(mask.sum())} samples -> fitting full tree")
        w = fit_window_tree(X_full[mask], y_full[mask], max_depth,
                            min_samples_leaf, domain_bounds)
        w['lo'], w['hi'] = lo, hi
        print(f"    {w['node_count']} nodes")
        windows.append(w)
    print(f"\n[A] fits done in {time.time() - t0:.0f}s")

    if native_heatmap is not None:
        gt_grid, grid_x, grid_t = native_heatmap
    else:
        gt_grid, grid_x, grid_t = cpt.prepare_ground_truth_grid(
            eval_data, domain_bounds, resolution=150)

    summary = {}
    for M in M_LIST:
        combo = f"M_{M}_W_{NUM_WINDOWS}"
        out_dir = OUTPUT_ROOT / combo
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'#' * 60}\n#  {PROBLEM} / {combo} (global top-M)\n{'#' * 60}")

        # B
        top, M_actual, n_ties, cutoff, pool_size = select_global_top_m(
            windows, M, variable, eps)
        print(f"[B] pooled {pool_size} candidate nodes; top {M_actual} "
              f"(+{n_ties} ties) -> per window "
              f"{[len(top[k]) for k in range(NUM_WINDOWS)]}; "
              f"cutoff {variable}={cutoff:.4g}")

        # C
        all_nodes, all_bfs = [], []
        per_window = []
        for k, w in enumerate(windows):
            accepted = close_window(w, top[k], retain_siblings=True)
            res = build_window_result(
                w, accepted, top[k], k, w['lo'], w['hi'], domain_bounds,
                t_dim=len(domain_bounds['lower']) - 1, id_offset=0,
                variable=variable)
            nd, bfs = window_outputs(w, res, k)
            kept_root = bool(res['root_kept'])
            per_window.append({
                'top_M': len(top[k]),
                'accepted': len(bfs),
                'leaves': sum(1 for b in bfs if b['is_leaf_in_pruned_tree']),
                'root_kept': kept_root,
            })
            print(f"[C] window {k}: {len(top[k])} top-M -> {len(bfs)} "
                  f"accepted after closure"
                  f"{' (ROOT KEPT: whole-slice expert)' if kept_root else ''}")
            all_nodes.extend(nd)
            all_bfs.extend(bfs)

        n_acc = len(all_bfs)
        n_leaves = sum(1 for b in all_bfs if b['is_leaf_in_pruned_tree'])
        tree_data = {
            'domain_bounds': domain_bounds,
            'tree_params': {
                'max_depth': max_depth,
                'min_samples_leaf': min_samples_leaf,
                'global_M': M,
                'epsilon_node_acceptance': eps,
                'num_windows': NUM_WINDOWS,
                'm_distribution': 'global_top_M',
                'm_per_window': [pw['top_M'] for pw in per_window],
                'selection_cutoff_metric': cutoff,
                'per_window': per_window,
            },
            'summary': {
                'total_nodes': len(all_nodes),
                'accepted_nodes': n_acc,
                'pruned_tree_leaves': n_leaves,
            },
            'accepted_nodes_bfs': all_bfs,
            'all_nodes': all_nodes,
        }
        with open(out_dir / 'perfect_trees.json', 'w') as f:
            json.dump({PROBLEM: tree_data}, f, indent=2,
                      cls=cpt._NumpySafeEncoder)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        cpt._plot_regions_panel(
            axes[0], all_nodes, domain_bounds, gt_grid, grid_x, grid_t,
            'Original trees')
        # Draw the accepted set incl. kept window roots (all_nodes only
        # carries real tree nodes, so use the BFS list here).
        cpt._plot_regions_panel(
            axes[1], all_bfs, domain_bounds, gt_grid, grid_x, grid_t,
            f'After pruning (global top-{M}: '
            f'{[pw["top_M"] for pw in per_window]})')
        plt.tight_layout()
        out_path = save_png(
            out_dir / (f'perfect_tree_{PROBLEM}_W{NUM_WINDOWS}_global'
                       f'_M{M}_acc{n_acc}.png'),
            fig=fig)
        plt.close()
        print(f"Saved {out_path}")

        summary[combo] = {
            'accepted': n_acc,
            'leaves': n_leaves,
            'm_per_window': [pw['top_M'] for pw in per_window],
            'root_kept': [pw['root_kept'] for pw in per_window],
        }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / 'sweep_summary.json'
    merged = {}
    if summary_path.exists():
        with open(summary_path) as f:
            merged = json.load(f)
    merged.setdefault(PROBLEM, {}).update(summary)
    with open(summary_path, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"\n{'=' * 74}")
    print(f"{'combo':<14}{'accepted':>10}{'leaves':>8}   m_per_window   root_kept")
    for combo, r in summary.items():
        print(f"{combo:<14}{r['accepted']:>10}{r['leaves']:>8}   "
              f"{r['m_per_window']}   {r['root_kept']}")
    print(f"\nDone in {time.time() - t0:.0f}s. Results in {OUTPUT_ROOT}")


if __name__ == '__main__':
    main()
