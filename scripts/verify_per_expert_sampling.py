"""Compare the two ways of drawing per-expert residual points.

OLD  draw ``n_residual_train`` points uniformly over the whole domain, then
     filter them into each leaf by its bounding box (topping any starved leaf
     up to ``min_points_per_expert``).
NEW  give each leaf its share of the budget by volume,
     ``n_e = max(floor, round(N * vol_e / vol_domain))``, and draw those points
     directly inside its box.

NEW was adopted because filtering has a data-dependent output shape, which
forces a device->host sync per leaf. This script checks the claim that the two
are otherwise statistically the same, on the REAL leaf geometries in
perfect_tree_examples/M_8_e_0.0/perfect_trees.json.

Three things are compared, per PDE and per leaf:
  * count      -- NEW's deterministic count vs OLD's Monte-Carlo mean
  * density    -- points per unit volume (the invariant that actually matters:
                  filtering gives uniform DENSITY, not equal counts)
  * shape      -- two-sample KS test on the x and t marginals inside each leaf,
                  i.e. are the points distributed the same way within the box

Usage:
    python scripts/verify_per_expert_sampling.py
    python scripts/verify_per_expert_sampling.py --repeats 400 --budget 8192
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TREES = (REPO / 'perfect_tree_examples' / 'M_8_e_0.0'
                 / 'perfect_trees.json')


def leaf_boxes(tree):
    """(lower, upper) arrays for the pruned tree's leaves."""
    leaves = [n for n in tree['accepted_nodes_bfs']
              if n['is_leaf_in_pruned_tree']]
    lo = np.array([n['bounds_lower'] for n in leaves], dtype=float)
    hi = np.array([n['bounds_upper'] for n in leaves], dtype=float)
    return lo, hi


def old_draw(rng, lo, hi, dom_lo, dom_hi, budget, floor):
    """Global uniform draw, filtered per leaf, then floored."""
    pts = rng.uniform(dom_lo, dom_hi, size=(budget, len(dom_lo)))
    out = []
    for k in range(lo.shape[0]):
        m = np.all((pts >= lo[k]) & (pts <= hi[k]), axis=1)
        sel = pts[m]
        if sel.shape[0] < floor:                    # top-up inside the box
            extra = rng.uniform(lo[k], hi[k],
                                size=(floor - sel.shape[0], len(dom_lo)))
            sel = np.vstack([sel, extra]) if sel.size else extra
        out.append(sel)
    return out


def new_draw(rng, lo, hi, dom_lo, dom_hi, budget, floor):
    """Volume-proportional budget, drawn directly inside each leaf."""
    dom_vol = float(np.prod(dom_hi - dom_lo))
    out = []
    for k in range(lo.shape[0]):
        vol = float(np.prod(hi[k] - lo[k]))
        n = max(floor, int(round(budget * vol / dom_vol)))
        out.append(rng.uniform(lo[k], hi[k], size=(n, len(dom_lo))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trees', type=Path, default=DEFAULT_TREES)
    ap.add_argument('--budget', type=int, default=8192,
                    help='n_residual_train')
    ap.add_argument('--floor', type=int, default=1024,
                    help='min_points_per_expert')
    ap.add_argument('--repeats', type=int, default=200,
                    help='Monte-Carlo draws for the OLD method')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    try:
        from scipy.stats import ks_2samp
        have_scipy = True
    except ImportError:
        have_scipy = False
        print("  (scipy unavailable -- skipping KS tests)\n")

    trees = json.load(open(args.trees, encoding='utf-8'))
    print(f"trees  : {args.trees}")
    print(f"budget : {args.budget}   floor : {args.floor}   "
          f"repeats : {args.repeats}\n")

    worst_count_dev = 0.0
    worst_density_dev = 0.0
    min_ks_p = 1.0

    for problem in sorted(trees):
        tree = trees[problem]
        lo, hi = leaf_boxes(tree)
        dom_lo = np.array(tree['domain_bounds']['lower'], dtype=float)
        dom_hi = np.array(tree['domain_bounds']['upper'], dtype=float)
        vols = np.prod(hi - lo, axis=1)
        dom_vol = float(np.prod(dom_hi - dom_lo))
        K = lo.shape[0]

        rng = np.random.default_rng(args.seed)
        old_counts = np.zeros((args.repeats, K), dtype=int)
        old_pool = [[] for _ in range(K)]
        for r in range(args.repeats):
            blocks = old_draw(rng, lo, hi, dom_lo, dom_hi,
                              args.budget, args.floor)
            for k, b in enumerate(blocks):
                old_counts[r, k] = b.shape[0]
                if r < 20:                      # pool a few draws for the KS test
                    old_pool[k].append(b)

        new_blocks = new_draw(rng, lo, hi, dom_lo, dom_hi,
                              args.budget, args.floor)
        new_counts = np.array([b.shape[0] for b in new_blocks])

        print(f"=== {problem}   ({K} leaves, domain volume {dom_vol:.4f})")
        print(f"{'leaf':>5} {'vol share':>10} {'OLD mean':>12} {'OLD std':>8} "
              f"{'NEW':>7} {'dev':>7} {'OLD dens':>10} {'NEW dens':>10} "
              f"{'KS p (x)':>9} {'KS p (t)':>9}")
        for k in range(K):
            om, os_ = old_counts[:, k].mean(), old_counts[:, k].std()
            nc = new_counts[k]
            dev = abs(nc - om) / max(om, 1.0)
            od, nd = om / vols[k], nc / vols[k]
            ddev = abs(nd - od) / max(od, 1e-12)
            worst_count_dev = max(worst_count_dev, dev)
            worst_density_dev = max(worst_density_dev, ddev)

            ps = []
            if have_scipy:
                op = np.vstack(old_pool[k])
                npt = new_blocks[k]
                for d in range(lo.shape[1]):
                    ps.append(ks_2samp(op[:, d], npt[:, d]).pvalue)
                min_ks_p = min(min_ks_p, min(ps))
            ks_s = (f"{ps[0]:>9.3f} {ps[1]:>9.3f}" if ps
                    else f"{'-':>9} {'-':>9}")
            print(f"{k:>5} {vols[k] / dom_vol:>9.1%} {om:>12.1f} {os_:>8.1f} "
                  f"{nc:>7} {dev:>6.2%} {od:>10.0f} {nd:>10.0f} {ks_s}")
        print(f"      totals: OLD {old_counts.sum(axis=1).mean():.1f}  "
              f"NEW {new_counts.sum()}\n")

    print("=" * 72)
    print(f"worst per-leaf COUNT deviation   : {worst_count_dev:.3%}")
    print(f"worst per-leaf DENSITY deviation : {worst_density_dev:.3%}")
    if have_scipy:
        print(f"smallest KS p-value across all leaves/dims : {min_ks_p:.4f}")
        print("  (p > 0.05 => cannot distinguish the within-leaf distributions)")
    print("=" * 72)


if __name__ == '__main__':
    main()
