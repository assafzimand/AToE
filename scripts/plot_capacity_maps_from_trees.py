"""Standalone: capacity-density maps for the trees built by plot_final_trees.py.

Reuses adaptive.visualization.plot_capacity_map (the same function
run_experiments calls after expert spawning, producing
``capacity_map_epoch_<N>_L<leaves>_leafp<P>_rootp<P>.png``) but driven from a
perfect_tree_examples/sweep/<problem>/<combo>/perfect_trees.json instead of a
live training run's metrics.json. Each accepted node in the JSON becomes a
leaf expert with a synthetic MLP architecture [in_dim, 20, 20, 20, out_dim]
(in_dim/out_dim read from the problem's own domain/output dimensionality).

For each tree, produces two maps:
  - capacity_map_<tag>_no_overlap.png
        Parameter density per exclusive region (hard tiling): each leaf
        counts only over its own box, matching a plain weights/volume ratio.
  - capacity_map_<tag>_overlap_sigma<S>.png
        Same, but each leaf's ACTIVE support is widened by the fixed PoU
        collar delta = sigma_fraction * region_extent (matching
        utils/dataset_gen.py's fixed-collar formula) on every side, so
        overlapping collars sum their capacity -- what the code actually
        does at sigma_fraction=<S>.

Usage:
    python scripts/plot_capacity_maps_from_trees.py <combo_dir> [<combo_dir> ...] \\
        [--sigma-fraction 0.05] [--out-dir outputs/paper_figures/capacity_maps]

    combo_dir: e.g. perfect_tree_examples/sweep/kdv/M_20_W_3_quadratic
               (must contain perfect_trees.json)
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from perfect_tree_examples import create_prefect_trees as cpt
from adaptive.visualization import plot_capacity_map


def mlp_param_count(layers):
    return sum(a * b + b for a, b in zip(layers[:-1], layers[1:]))


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

    plot_capacity_map(
        regions=regions, expert_params=expert_params, leaf_indices=leaf_indices,
        base_params=0, domain_bounds=domain_bounds,
        output_path=out_dir / f'capacity_map_{tag}_no_overlap.png',
    )

    sigma_tag = f"{sigma_fraction:g}".replace('.', 'p')
    delta_lo = np.zeros((n_leaves, in_dim))
    delta_hi = np.zeros((n_leaves, in_dim))
    for i, r in enumerate(regions):
        lo = np.asarray(r['bounds_lower'], dtype=float)
        hi = np.asarray(r['bounds_upper'], dtype=float)
        d = np.maximum(sigma_fraction * (hi - lo), 1e-6)
        delta_lo[i] = d
        delta_hi[i] = d

    plot_capacity_map(
        regions=regions, expert_params=expert_params, leaf_indices=leaf_indices,
        base_params=0, domain_bounds=domain_bounds,
        output_path=out_dir / f'capacity_map_{tag}_overlap_sigma{sigma_tag}.png',
        delta_lo=delta_lo, delta_hi=delta_hi,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('combo_dirs', nargs='+', type=Path)
    ap.add_argument('--sigma-fraction', type=float, default=0.05)
    ap.add_argument('--out-dir', type=Path,
                     default=Path('outputs/paper_figures/capacity_maps'))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = cpt.load_config(REPO_ROOT / 'experiments_plan.yaml')
    for combo_dir in args.combo_dirs:
        process_combo(combo_dir, base_cfg, args.sigma_fraction, args.out_dir)


if __name__ == '__main__':
    main()
