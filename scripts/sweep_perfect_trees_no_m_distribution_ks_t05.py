"""Global top-M reference trees for KS on t in [0, 0.5], W=5.

Runs scripts/sweep_perfect_trees_no_m_distribution.py for KS with the time
domain cut to [0, 0.5] (experiments_plan.yaml has [0, 1.0]; every tree under
perfect_tree_examples/ks_t05 is on the half domain) and eval data generated
in memory (never cached under datasets/ks).

Output: perfect_tree_examples/sweep/ks_t05_no_m_distribution/M_{M}_W_5/

Usage (from the repo root):
    python scripts/sweep_perfect_trees_no_m_distribution_ks_t05.py
"""

import copy
import importlib
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    'sweep_no_m', REPO_ROOT / 'scripts' / 'sweep_perfect_trees_no_m_distribution.py')
sw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sw)
cpt = sw.cpt

T_MAX = 0.5
sw.PROBLEM = 'ks'
sw.NUM_WINDOWS = 5
sw.M_LIST = [10, 12, 15, 18, 20, 25, 30, 35]
sw.OUTPUT_ROOT = (REPO_ROOT / 'perfect_tree_examples' / 'sweep'
                  / 'ks_t05_no_m_distribution')

_orig_load_config = cpt.load_config


def _load_config_t05(plan_path):
    base_cfg = copy.deepcopy(_orig_load_config(plan_path))
    base_cfg['ks']['temporal_domain'] = [0.0, T_MAX]
    return base_cfg


def _ensure_eval_data_no_cache(problem, base_cfg):
    import torch
    from utils.dataset_gen import calculate_dataset_sizes
    cfg = dict(base_cfg)
    cfg['problem'] = problem
    cfg['cuda'] = False
    sizes = calculate_dataset_sizes(cfg)
    solver = importlib.import_module(f'solvers.{problem}_solver')
    return solver.generate_dataset(
        n_residual=sizes['n_residual_eval'], n_ic=sizes['n_initial_eval'],
        n_bc=sizes['n_boundary_eval'], device=torch.device('cpu'), config=cfg)


if __name__ == '__main__':
    cpt.load_config = _load_config_t05
    cpt.ensure_eval_data = _ensure_eval_data_no_cache
    sw.main()
