"""Standalone: example normalized blending weights for the AAToE (Additive
AToE) composition -- Fig. "soft_weights_atoe" in the paper's method section.

Unlike the PoU blending in plot_soft_weights_from_regions.py (leaves-only,
normalized to sum to 1 everywhere), AAToE treats the root as an always-on
additive background (coefficient exactly 1 everywhere) plus per-LEVEL
normalized corrections:

    u_theta = u_0 + sum_{level=1}^{D} sum_{i: level(i)=level} w_i * u_i
    w_i(x,t) = Psi_i(x,t) / (1 + sum_{k: level(k)=level(i)} Psi_k(x,t))

The "+1" is the root's constant background: it keeps w_i well-defined even
where no same-level expert covers (x,t), which is exactly what lets the
tree be TRIMMED to a subset of nodes (any level can leave part of the domain
uncovered -- the root just carries it).

To illustrate that trimming explicitly, this script:
  1. Loads the trained tree's leaf regions from expert_regions.json (level =
     each region's own recorded tree depth).
  2. Finds a leaf that has a same-level sibling (level(i) shared by >=2
     leaves) and drops it -- demonstrating that siblings need not both be
     kept.
  3. Renders: one constant panel for the root (Psi_0 = 1 everywhere), and
     one panel per surviving leaf showing its per-level-normalized weight
     w_i, with the trimmed leaf's former territory visibly uncovered (all
     surviving weights -> 0 there, root fully carries it).

Same Psi_i geometry (tensor-product smoothstep window) and paper styling
(no headers, bigger/bold axis + colorbar text) as plot_soft_weights_from_regions.py.

Usage:
    python scripts/plot_aatoe_weights.py <run_dir> [--sigma-fraction 0.2] \\
        [--drop-index N] [--resolution 200] [--out-dir outputs/paper_figures/soft_weights]

    run_dir: leaf run dir containing adaptive_plots/expert_regions.json and
             config_used.yaml.
    --drop-index: force which leaf index to trim (default: auto-pick a leaf
             that shares its level with another leaf).
"""

import sys
import json
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.plot_io import save_png

TICK_SIZE = 14
LABEL_SIZE = 17
CBAR_TICK_SIZE = 13
CBAR_LABEL_SIZE = 15


def smoothstep_N(t, N):
    if N == 1:
        return 3 * t**2 - 2 * t**3
    if N == 2:
        return 6 * t**5 - 15 * t**4 + 10 * t**3
    if N == 3:
        return 35 * t**4 - 84 * t**5 + 70 * t**6 - 20 * t**7
    if N == 4:
        return 126 * t**5 - 420 * t**6 + 540 * t**7 - 315 * t**8 + 70 * t**9
    raise ValueError(f"Unsupported smoothstep order N={N}")


def rho_N(s, N):
    return smoothstep_N(np.clip(s, 0.0, 1.0), N)


def omega_1d(X_j, a, b, sigma_fraction, N):
    delta = max(sigma_fraction * (b - a), 1e-12)
    s_lo = (X_j - (a - delta)) / delta
    s_hi = ((b + delta) - X_j) / delta
    return rho_N(s_lo, N) * rho_N(s_hi, N)


def psi_region(X, T, bounds_lower, bounds_upper, sigma_fraction, N):
    return (omega_1d(X, bounds_lower[0], bounds_upper[0], sigma_fraction, N)
           * omega_1d(T, bounds_lower[1], bounds_upper[1], sigma_fraction, N))


def _style_axes(ax):
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.xaxis.label.set_fontsize(LABEL_SIZE)
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontsize(LABEL_SIZE)
    ax.yaxis.label.set_fontweight('bold')


def _style_colorbar(cbar):
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight('bold')
    cbar.ax.yaxis.label.set_fontsize(CBAR_LABEL_SIZE)
    cbar.ax.yaxis.label.set_fontweight('bold')


def pick_leaf_to_drop(regions, forced_index=None):
    if forced_index is not None:
        return forced_index
    by_level = defaultdict(list)
    for i, r in enumerate(regions):
        by_level[r['depth']].append(i)
    siblings = [ids for ids in by_level.values() if len(ids) >= 2]
    if not siblings:
        raise SystemExit(
            "No two leaves share a level (no siblings to demonstrate "
            "trimming with) -- pass --drop-index to force one.")
    # Deepest sibling group, highest index within it (arbitrary but deterministic).
    group = max(siblings, key=lambda ids: regions[ids[0]]['depth'])
    return max(group)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--sigma-fraction', type=float, default=None,
                    help='Overrides config; falls back to 0.2 if the config has none')
    ap.add_argument('--drop-index', type=int, default=None,
                    help='Force which leaf (by index in expert_regions.json) to trim')
    ap.add_argument('--resolution', type=int, default=200)
    ap.add_argument('--out-dir', type=Path,
                     default=Path('outputs/paper_figures/soft_weights'))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load((args.run_dir / 'config_used.yaml').read_text(encoding='utf-8'))
    er = json.loads((args.run_dir / 'adaptive_plots' / 'expert_regions.json').read_text(encoding='utf-8'))
    problem = cfg['problem']
    pc = cfg[problem]

    sigma_fraction = args.sigma_fraction
    if sigma_fraction is None:
        ap_cfg = cfg['adaptive_pinn']
        sigma_fraction = ap_cfg.get('sigma_fraction')
        if sigma_fraction is None:
            sigma_fraction = (ap_cfg.get('fine_tune') or {}).get('sigma_fraction')
        if sigma_fraction is None:
            sigma_fraction = 0.2
            print("No sigma_fraction in config -- defaulting to 0.2")
    N = pc.get('window_smoothness_order', 2)

    all_regions = er['regions']
    drop_idx = pick_leaf_to_drop(all_regions, args.drop_index)
    dropped = all_regions[drop_idx]
    print(f"Trimming leaf {drop_idx} (level {dropped['depth']}, "
          f"bounds {dropped['bounds_lower']}-{dropped['bounds_upper']}) to "
          f"demonstrate partial coverage -- its territory now falls back to "
          f"the root alone.")
    regions = [r for i, r in enumerate(all_regions) if i != drop_idx]
    kept_indices = [i for i in range(len(all_regions)) if i != drop_idx]

    print(f"{problem}: {len(regions)} experts kept (of {len(all_regions)}), "
          f"sigma_fraction={sigma_fraction}, N={N}")

    x_min, x_max = pc['spatial_domain'][0]
    t_min, t_max = pc['temporal_domain']
    x_grid = np.linspace(x_min, x_max, args.resolution)
    t_grid = np.linspace(t_min, t_max, args.resolution)
    X, T = np.meshgrid(x_grid, t_grid, indexing='ij')

    psis = [psi_region(X, T, r['bounds_lower'], r['bounds_upper'], sigma_fraction, N)
           for r in regions]
    levels = [r['depth'] for r in regions]
    level_sum = defaultdict(lambda: np.zeros_like(X))
    for psi, lvl in zip(psis, levels):
        level_sum[lvl] += psi
    weights = [psi / (1.0 + level_sum[lvl]) for psi, lvl in zip(psis, levels)]

    n_panels = len(regions) + 1  # +1 for the root
    n_cols = min(4, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows),
                             squeeze=False, layout='constrained')
    axes = axes.flatten()
    cmap = plt.cm.Reds

    # Every panel shares the SAME [0, 1] scale, one colorbar per panel (not
    # a merged one) -- root is the strongest at a constant 1, every expert
    # is visibly smaller AND fades within its own collar, directly
    # comparable at a glance because the color scale never changes.

    # ── Root: constant coefficient 1 everywhere ──
    ax0 = axes[0]
    im_root = ax0.pcolormesh(x_grid, t_grid, np.ones_like(X).T, cmap=cmap,
                             vmin=0, vmax=1, shading='auto')
    ax0.text(0.04, 0.94, 'Root', transform=ax0.transAxes, fontsize=LABEL_SIZE,
             fontweight='bold', color='black', va='top', ha='left',
             path_effects=[path_effects.withStroke(linewidth=2.5, foreground='white')])
    ax0.set_xlabel('x')
    ax0.set_ylabel('t')
    _style_axes(ax0)
    cbar0 = fig.colorbar(im_root, ax=ax0, label='Weight', shrink=0.85)
    _style_colorbar(cbar0)

    # ── Leaves: per-level normalized weight ──
    for panel_idx, (orig_idx, r, w) in enumerate(zip(kept_indices, regions, weights), start=1):
        ax = axes[panel_idx]
        im = ax.pcolormesh(x_grid, t_grid, w.T, cmap=cmap, vmin=0, vmax=1, shading='auto')
        lo, hi = r['bounds_lower'], r['bounds_upper']
        ax.add_patch(patches.Rectangle(
            (lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
            linewidth=2, edgecolor='black', facecolor='none', linestyle='--'))
        ax.text(0.04, 0.94, f'E{orig_idx}', transform=ax.transAxes, fontsize=LABEL_SIZE,
                fontweight='bold', color='black', va='top', ha='left',
                path_effects=[path_effects.withStroke(linewidth=2.5, foreground='white')])
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        _style_axes(ax)
        cbar = fig.colorbar(im, ax=ax, label='Weight', shrink=0.85)
        _style_colorbar(cbar)

    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    out_path = args.out_dir / 'soft_weights_atoe.png'
    save_png(out_path, fig=fig)
    plt.close(fig)
    print(f"saved {out_path.name}")


if __name__ == '__main__':
    main()
