"""Sweep perfect-tree generation over per-problem parameter grids.

Runs perfect_tree_examples/create_prefect_trees.py for every combination in
each problem's sweep spec below, writing each combo's plots and
perfect_trees.json into perfect_tree_examples/sweep/{problem}/{combo}/.

Sweepable parameters per problem:
  - M               : global expert budget (M_experts_num)
  - num_windows     : time-marching window counts (time-marching problems
                      only, e.g. kdv/ks). Listing this forces
                      time_marching.enabled = True for the combo.
  - m_distribution  : per-window M allocation ('equal'|'linear'|'quadratic')
  - eps             : epsilon_node_acceptance (optional; defaults to the
                      value in experiments_plan.yaml)

Eval data and the symmetric-grid interpolation are cached across combos
(they depend only on the problem, not on the swept params), so each extra
combo costs only the tree fits + plots.

Usage (from the repo root):
    python scripts/sweep_perfect_trees.py
"""

import copy
import itertools
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # ensure_eval_data uses paths relative to the repo root

from perfect_tree_examples import create_prefect_trees as cpt  # noqa: E402

# Per-problem sweep specs. Omit 'num_windows'/'m_distribution' for
# non-time-marching problems ([None] placeholder keeps the product loop
# uniform and leaves the yaml config untouched).
SWEEPS = {
    'kdv': {
        'M': [10, 15, 20],
        'num_windows': [3, 4, 5],
        'm_distribution': ['quadratic'],
    },
    'ks': {
        'M': [10, 20, 30],
        'num_windows': [5, 7, 10],
        'm_distribution': ['quadratic'],
    },
}

OUTPUT_ROOT = REPO_ROOT / 'perfect_tree_examples' / 'sweep'


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
    if num_windows is not None or m_distribution is not None:
        tm_cfg = cfg[problem].setdefault('time_marching', {})
        tm_cfg['enabled'] = True
        if num_windows is not None:
            tm_cfg['num_windows'] = num_windows
        if m_distribution is not None:
            tm_cfg['m_distribution'] = m_distribution
    return cfg


def main():
    _install_caches()

    base_cfg = cpt.load_config(REPO_ROOT / 'experiments_plan.yaml')

    n_combos = sum(
        len(spec['M'])
        * len(spec.get('num_windows', [None]))
        * len(spec.get('m_distribution', [None]))
        * len(spec.get('eps', [None]))
        for spec in SWEEPS.values()
    )
    print(f"Sweep problems: {list(SWEEPS)} ({n_combos} combos total)")

    summary = {}
    t0 = time.time()
    for problem, spec in SWEEPS.items():
        if problem not in base_cfg:
            print(f"Skipping {problem}: not in experiments_plan.yaml")
            continue
        summary[problem] = {}
        for M, W, dist, eps in itertools.product(
                spec['M'],
                spec.get('num_windows', [None]),
                spec.get('m_distribution', [None]),
                spec.get('eps', [None])):
            combo = _combo_name(M, W, dist, eps)
            out_dir = OUTPUT_ROOT / problem / combo
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
    with open(OUTPUT_ROOT / 'sweep_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 74}")
    print(f"{'problem':<10}{'combo':<26}{'accepted':>10}{'leaves':>10}")
    for problem, combos in summary.items():
        for combo, res in combos.items():
            print(f"{problem:<10}{combo:<26}"
                  f"{res['accepted']:>10}{res['leaves']:>10}")
    print(f"\nDone in {time.time() - t0:.0f}s. "
          f"Results in {OUTPUT_ROOT}, summary in sweep_summary.json")


if __name__ == '__main__':
    main()
