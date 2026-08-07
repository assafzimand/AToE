"""Standalone: re-render just the final ("after pruning") tree panel, no header.

Each perfect_tree_examples/sweep/<problem>/<combo>/ folder holds a 2-panel
comparison PNG (raw candidate trees | after pruning) plus the perfect_trees.json
that produced it. This re-renders ONLY the "after pruning" panel -- the tree
actually used -- as its own clean, title-free image, straight from the JSON
data and the same _plot_regions_panel() the original figure used (so it's a
fresh render, not a crop of the existing PNG).

Usage:
    python scripts/plot_final_trees.py <combo_dir> [<combo_dir> ...] [--out-dir DIR]

    combo_dir: e.g. perfect_tree_examples/sweep/kdv/M_20_W_3_quadratic
               (must contain perfect_trees.json)
"""

import sys
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from perfect_tree_examples import create_prefect_trees as cpt
from utils.plot_io import save_png


def process_combo(combo_dir: Path, base_cfg, out_dir: Path):
    json_path = combo_dir / 'perfect_trees.json'
    data = json.loads(json_path.read_text(encoding='utf-8'))
    problem = next(iter(data))
    tree = data[problem]
    domain_bounds = tree['domain_bounds']
    accepted = [n for n in tree['all_nodes'] if n.get('accepted')]
    params = tree['tree_params']
    n_acc = tree['summary']['accepted_nodes']

    gt = cpt.build_native_grid_data(problem, base_cfg)
    if gt is None:
        raise SystemExit(f"No native solver grid available for '{problem}'")
    _, _, (gt_grid, grid_x, grid_t) = gt

    fig, ax = plt.subplots(figsize=(9, 7))
    cpt._plot_regions_panel(ax, accepted, domain_bounds, gt_grid, grid_x, grid_t, title='')
    plt.tight_layout()

    W = params.get('num_windows')
    M = params['global_M']
    dist = params.get('m_distribution')
    tag = (f"W{W}_" if W is not None else '') + (f"{dist}_" if dist else '')
    out_path = out_dir / f"tree_{problem}_{tag}M{M}_acc{n_acc}.png"
    save_png(out_path, fig=fig)
    plt.close(fig)
    print(f"  saved {out_path.name}  ({n_acc} accepted regions)")
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
