"""Perfect-tree sweep for the PROPOSED zero-based M distributions.

Like scripts/sweep_perfect_trees.py, but the per-window M allocation uses
ZERO-BASED weights (i = 0..W-1) with Hamilton rounding and NO per-window
floor of 1 — window 0 always gets M=0, and an M=0 window is rendered as ONE
whole-window region (the "single expert spans the full window" semantic).

    quadratic_zero (weights i^2):
        W=3: M=8 -> [0, 2, 6]   M=10 -> [0, 2, 8]        M=20 -> [0, 4, 16]
        W=5: M=8 -> [0, 0, 1, 3, 4]  (two whole-window experts!)
    linear_zero (weights i):
        W=3: M=8 -> [0, 3, 5]   M=10 -> [0, 3, 7]        M=20 -> [0, 7, 13]
        W=5: M=8 -> [0, 1, 2, 2, 3]  M=10 -> [0, 1, 2, 3, 4]  M=20 -> [0, 2, 4, 6, 8]

Everything is self-contained here: neither name exists in
trainer.time_marching.compute_m_per_window — this script wraps the creator's
allocation and tree-fit hooks instead of changing repo code.

Output: perfect_tree_examples/sweep/{problem}/M_{M}_W_{W}_{dist}_e_0.0/

Usage (from the repo root):
    python scripts/sweep_perfect_trees_quadratic_zero.py
"""

import copy
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # ensure_eval_data uses paths relative to the repo root

from perfect_tree_examples import create_prefect_trees as cpt  # noqa: E402

SWEEPS = {
    'kdv': {
        'M': [12, 15, 18],
        'num_windows': [3],
        'm_distribution': ['quadratic_zero', 'linear_zero'],
        'eps': [0.0],
    },
    'ks': {
        'M': [8, 10, 20],
        'num_windows': [5],
        'm_distribution': ['quadratic_zero', 'linear_zero'],
        'eps': [0.0],
    },
}

OUTPUT_ROOT = REPO_ROOT / 'perfect_tree_examples' / 'sweep'


ZERO_POWERS = {'linear_zero': 1, 'quadratic_zero': 2}


def compute_m_per_window_zero(global_M: int, num_windows: int, power: int):
    """Zero-based allocation: weights i^power (i=0..W-1), Hamilton, no floor.

    Window 0's weight is 0, so it always receives M=0 (one whole-window
    expert under the proposed semantic); later windows split the full M
    by i^power with largest-remainder rounding.
    """
    weights = [i ** power for i in range(num_windows)]
    total_weight = sum(weights)
    raw = [global_M * w / total_weight for w in weights]
    floors = [int(r) for r in raw]
    leftover = global_M - sum(floors)
    order = sorted(range(num_windows),
                   key=lambda i: raw[i] - floors[i], reverse=True)
    for i in order[:leftover]:
        floors[i] += 1
    assert sum(floors) == global_M
    return floors


def _install_quadratic_zero():
    """Route 'quadratic_zero' through the local allocator and make M=0
    windows produce one synthetic whole-window accepted leaf."""
    orig_compute = cpt.compute_m_per_window

    def compute_patched(global_M, num_windows, distribution):
        if distribution in ZERO_POWERS:
            return compute_m_per_window_zero(global_M, num_windows,
                                             ZERO_POWERS[distribution])
        return orig_compute(global_M, num_windows, distribution)

    cpt.compute_m_per_window = compute_patched

    orig_fit = cpt.fit_and_get_all_nodes

    def fit_patched(X, y, max_depth, min_samples_leaf, M,
                    variable_for_node_accept, domain_bounds,
                    epsilon_node_acceptance=0.0):
        if M > 0:
            return orig_fit(X, y, max_depth, min_samples_leaf, M,
                            variable_for_node_accept, domain_bounds,
                            epsilon_node_acceptance=epsilon_node_acceptance)
        # M=0 window: ONE region = the exact window box, no tree fit.
        print(f"    [zero-dist] M=0 -> single whole-window region "
              f"{domain_bounds['lower']} .. {domain_bounds['upper']}")
        node = {
            'node_id': 1,
            'parent_node_id': -1,
            'wavelet_norm_squared': 0.0,
            'new_wavelet_norm_squared': 0.0,
            'smoothness_alpha': None,
            'smoothness_r2': None,
            'smoothness_n_levels': 0,
            'n_samples': int(len(X)),
            'is_leaf': True,
            'bounds_lower': list(domain_bounds['lower']),
            'bounds_upper': list(domain_bounds['upper']),
            'accepted': True,
            'tree_depth': 1,
        }
        bfs_accepted = [{
            'node_id': 1,
            'parent_tree_node_id': -1,
            'bounds_lower': list(domain_bounds['lower']),
            'bounds_upper': list(domain_bounds['upper']),
            'wavelet_norm_squared': 0.0,
            'new_wavelet_norm_squared': 0.0,
            'smoothness_alpha': None,
            'n_samples': int(len(X)),
            'tree_depth': 1,
            'is_leaf_in_pruned_tree': True,
        }]
        children_left = np.array([-1, -1])  # ids 0 (unused root), 1 (leaf)
        return [node], {1}, bfs_accepted, children_left

    cpt.fit_and_get_all_nodes = fit_patched


def _install_caches():
    """Memoize the per-problem data loading/interpolation inside cpt."""
    eval_cache = {}
    orig_ensure = cpt.ensure_eval_data

    def ensure_cached(problem, base_cfg):
        if problem not in eval_cache:
            eval_cache[problem] = orig_ensure(problem, base_cfg)
        return eval_cache[problem]

    cpt.ensure_eval_data = ensure_cached

    grid_cache = {}
    orig_grid = cpt.build_symmetric_grid_data

    def grid_cached(eval_data, domain_bounds, resolution=200):
        key = (id(eval_data), tuple(domain_bounds['lower']),
               tuple(domain_bounds['upper']), resolution)
        if key not in grid_cache:
            grid_cache[key] = orig_grid(eval_data, domain_bounds, resolution)
        return grid_cache[key]

    cpt.build_symmetric_grid_data = grid_cached

    gt_cache = {}
    orig_gt = cpt.prepare_ground_truth_grid

    def gt_cached(eval_data, domain_bounds, resolution=100):
        key = (id(eval_data), tuple(domain_bounds['lower']),
               tuple(domain_bounds['upper']), resolution)
        if key not in gt_cache:
            gt_cache[key] = orig_gt(eval_data, domain_bounds, resolution)
        return gt_cache[key]

    cpt.prepare_ground_truth_grid = gt_cached


def _combo_name(M, num_windows, m_distribution, eps):
    parts = [f"M_{M}"]
    if num_windows is not None:
        parts.append(f"W_{num_windows}")
    if m_distribution is not None:
        parts.append(m_distribution)
    if eps is not None:
        parts.append(f"e_{eps}")
    return '_'.join(parts)


def _build_cfg(base_cfg, problem, M, num_windows, m_distribution, eps):
    cfg = copy.deepcopy(base_cfg)
    cfg['adaptive_pinn']['M_experts_num'] = M
    if eps is not None:
        cfg['adaptive_pinn']['epsilon_node_acceptance'] = eps
    tm_cfg = cfg[problem].setdefault('time_marching', {})
    tm_cfg['enabled'] = True
    tm_cfg['num_windows'] = num_windows
    tm_cfg['m_distribution'] = m_distribution
    # This script builds trees only — make sure tree-only/debug leftovers in
    # the plan's base_config don't leak into the creator.
    tm_cfg.pop('only_for_tree_structure', None)
    tm_cfg['debug_windows'] = None
    return cfg


def main():
    _install_caches()
    _install_quadratic_zero()

    base_cfg = cpt.load_config(REPO_ROOT / 'experiments_plan.yaml')

    print("Allocations under the zero-based distributions:")
    for problem, spec in SWEEPS.items():
        for M, W, dist in itertools.product(spec['M'], spec['num_windows'],
                                            spec['m_distribution']):
            print(f"  {problem}: {dist} M={M}, W={W} -> "
                  f"{compute_m_per_window_zero(M, W, ZERO_POWERS[dist])}")

    summary = {}
    t0 = time.time()
    for problem, spec in SWEEPS.items():
        if problem not in base_cfg:
            print(f"Skipping {problem}: not in experiments_plan.yaml")
            continue
        summary[problem] = {}
        for M, W, dist, eps in itertools.product(
                spec['M'], spec['num_windows'],
                spec['m_distribution'], spec['eps']):
            combo = _combo_name(M, W, dist, eps)
            out_dir = OUTPUT_ROOT / problem / combo
            if (out_dir / 'perfect_trees.json').exists():
                print(f"\n#  {problem} / {combo}: already built — skipping "
                      f"(delete the folder to refit)")
                with open(out_dir / 'perfect_trees.json') as f:
                    prev = json.load(f)[problem]
                summary[problem][combo] = {
                    'accepted': prev['summary']['accepted_nodes'],
                    'leaves': prev['summary']['pruned_tree_leaves'],
                }
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n{'#' * 60}\n#  {problem} / {combo}\n{'#' * 60}")

            cfg = _build_cfg(base_cfg, problem, M, W, dist, eps)
            try:
                tree_data = cpt.process_problem(problem, cfg, out_dir)
            except Exception as e:
                print(f"  ERROR processing {problem} / {combo}: {e}")
                import traceback
                traceback.print_exc()
                continue
            if tree_data is None:
                continue

            with open(out_dir / 'perfect_trees.json', 'w') as f:
                json.dump({problem: tree_data}, f, indent=2,
                          cls=cpt._NumpySafeEncoder)

            summary[problem][combo] = {
                'accepted': tree_data['summary']['accepted_nodes'],
                'leaves': tree_data['summary']['pruned_tree_leaves'],
            }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ROOT / 'sweep_summary_zero_dists.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 74}")
    print(f"{'problem':<10}{'combo':<34}{'accepted':>10}{'leaves':>10}")
    for problem, combos in summary.items():
        for combo, res in combos.items():
            print(f"{problem:<10}{combo:<34}"
                  f"{res['accepted']:>10}{res['leaves']:>10}")
    print(f"\nDone in {time.time() - t0:.0f}s. "
          f"Results in {OUTPUT_ROOT}, summary in sweep_summary_zero_dists.json")


if __name__ == '__main__':
    main()
