"""Sweep perfect-tree generation over (M, epsilon_node_acceptance) combos.

Runs perfect_tree_examples/create_prefect_trees.py for every combination of
M_experts_num and epsilon_node_acceptance below, writing each combo's plots
and perfect_trees.json into perfect_tree_examples/M_{M}_e_{eps}/.

Eval data and the symmetric-grid interpolation are cached across combos
(they depend only on the problem, not on M/eps), so each extra combo costs
only the tree fits + plots.

Usage (from the repo root):
    python scripts/sweep_perfect_trees.py
"""

import copy
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # ensure_eval_data uses paths relative to the repo root

from perfect_tree_examples import create_prefect_trees as cpt  # noqa: E402

M_VALUES = [15]
# 0.0 = exact top-M baseline; 0.02/0.05 cover typical near-tie gaps at the
# cutoff; 0.1 is aggressive (observed mirror-pair gaps ran up to ~13%).
EPS_VALUES = [0.0]

OUTPUT_ROOT = REPO_ROOT / 'perfect_tree_examples'


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


def main():
    _install_caches()

    base_cfg = cpt.load_config(REPO_ROOT / 'experiments_plan.yaml')
    problems = cpt.get_problem_list(base_cfg)
    print(f"Problems: {problems}")
    print(f"Sweep: M in {M_VALUES}, eps in {EPS_VALUES} "
          f"({len(M_VALUES) * len(EPS_VALUES)} combos)")

    summary = {}
    t0 = time.time()
    for M in M_VALUES:
        for eps in EPS_VALUES:
            combo = f"M_{M}_e_{eps}"
            out_dir = OUTPUT_ROOT / combo
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n{'#' * 60}\n#  {combo}\n{'#' * 60}")

            cfg = copy.deepcopy(base_cfg)
            cfg['adaptive_pinn']['M_experts_num'] = M
            cfg['adaptive_pinn']['epsilon_node_acceptance'] = eps

            all_trees = {}
            for problem in problems:
                try:
                    tree_data = cpt.process_problem(problem, cfg, out_dir)
                    if tree_data is not None:
                        all_trees[problem] = tree_data
                except Exception as e:
                    print(f"  ERROR processing {problem}: {e}")
                    import traceback
                    traceback.print_exc()

            with open(out_dir / 'perfect_trees.json', 'w') as f:
                json.dump(all_trees, f, indent=2, cls=cpt._NumpySafeEncoder)

            summary[combo] = {
                p: {'accepted': d['summary']['accepted_nodes'],
                    'leaves': d['summary']['pruned_tree_leaves']}
                for p, d in all_trees.items()
            }

    with open(OUTPUT_ROOT / 'sweep_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 74}")
    print(f"{'combo':<14}" + ''.join(f"{p[:12]:>12}" for p in problems)
          + "   (leaves)")
    for combo, per_p in summary.items():
        row = ''.join(f"{per_p[p]['leaves'] if p in per_p else '-':>12}"
                      for p in problems)
        print(f"{combo:<14}{row}")
    print(f"\nDone in {time.time() - t0:.0f}s. "
          f"Results in {OUTPUT_ROOT}\\M_*_e_*, summary in sweep_summary.json")


if __name__ == '__main__':
    main()
