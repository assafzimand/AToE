"""Perfect-tree sweep for KS on the truncated time domain t in [0, 0.5].

Reuses scripts/sweep_perfect_trees.py (caches, combo naming, config
building, process loop) but:
  - overrides ks.temporal_domain to [0, 0.5] (experiments_plan.yaml has
    [0, 1.0]; the ks_t05 trees were all fit on the half domain),
  - writes into perfect_tree_examples/ks_t05/ks/{combo}/,
  - generates the eval data in memory instead of caching it under
    datasets/ks/eval_data.pt, so a half-domain eval set can never be
    silently picked up by a full-domain run later.

Usage (from the repo root):
    python scripts/sweep_perfect_trees_ks_t05.py
"""

import copy
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    'sweep_perfect_trees', REPO_ROOT / 'scripts' / 'sweep_perfect_trees.py')
sw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sw)
cpt = sw.cpt

T_MAX = 0.5
_orig_load_config = cpt.load_config

sw.SWEEPS = {
    'ks': [
        {
            'M': [35, 40],
            'num_windows': [5],
            'm_distribution': ['linear', 'quadratic'],
        },
    ],
}
sw.OUTPUT_ROOT = REPO_ROOT / 'perfect_tree_examples' / 'ks_t05'


def _load_config_t05(plan_path):
    base_cfg = copy.deepcopy(_orig_load_config(plan_path))
    base_cfg['ks']['temporal_domain'] = [0.0, T_MAX]
    return base_cfg


def _ensure_eval_data_no_cache(problem, base_cfg):
    """Same as cpt.ensure_eval_data but never reads/writes datasets/."""
    import importlib
    import torch
    from utils.dataset_gen import calculate_dataset_sizes
    cfg = dict(base_cfg)
    cfg['problem'] = problem
    cfg['cuda'] = False
    sizes = calculate_dataset_sizes(cfg)
    solver = importlib.import_module(f'solvers.{problem}_solver')
    return solver.generate_dataset(
        n_residual=sizes['n_residual_eval'],
        n_ic=sizes['n_initial_eval'],
        n_bc=sizes['n_boundary_eval'],
        device=torch.device('cpu'),
        config=cfg,
    )


if __name__ == '__main__':
    cpt.load_config = _load_config_t05
    cpt.ensure_eval_data = _ensure_eval_data_no_cache
    sw.main()
