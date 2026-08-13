"""Standalone: normalized PoU (partition-of-unity) soft-weight heatmaps,
reconstructed geometrically from expert_regions.json -- no model checkpoint
needed. Mirrors adaptive.visualization.plot_expert_soft_weights's per-expert
grid layout, paper-clean: no headers (a small "E<i>" label in-panel instead
of a title, since a multi-panel grid needs SOME way to tell panels apart),
bigger/bold axis + colorbar text.

Psi_i(x,t) is the same tensor-product smoothstep window used everywhere else
in the codebase (adaptive/indicators.py, scripts/plot_window_demo.py):
    delta = sigma_fraction * (region_extent)
    Psi_i = rho_N((x-(a-delta))/delta) * rho_N(((b+delta)-x)/delta)  [x rho_N(...) for t]
normalized as psi_tilde_i = Psi_i / sum_k Psi_k. sigma_fraction and the
smoothness order N are read from the run's config_used.yaml
(adaptive_pinn.sigma_fraction / .fine_tune.sigma_fraction and
<problem>.window_smoothness_order); --sigma-fraction overrides, and 0.2 is
the fallback when the config has neither.

Only leaf experts are reconstructed (no "Root" panel): expert_regions.json
doesn't carry the base network's own window contribution, and M-term-tree
spawning tiles the full domain with the leaves alone, so this omission
doesn't leave any point uncovered.

Usage:
    python scripts/plot_soft_weights_from_regions.py <run_dir> [--sigma-fraction 0.2] \\
        [--resolution 200] [--out-dir outputs/paper_figures/soft_weights]

    run_dir: leaf run dir containing adaptive_plots/expert_regions.json and
             config_used.yaml, e.g.:
             outputs/experiments/CORRECTOR_TESTS/.../20260723_030120
"""

import sys
import json
import argparse
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--sigma-fraction', type=float, default=None,
                    help='Overrides config; falls back to 0.2 if the config has none')
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
    regions = er['regions']
    n_experts = len(regions)
    print(f"{problem}: {n_experts} experts, sigma_fraction={sigma_fraction}, N={N}")

    x_min, x_max = pc['spatial_domain'][0]
    t_min, t_max = pc['temporal_domain']
    x_grid = np.linspace(x_min, x_max, args.resolution)
    t_grid = np.linspace(t_min, t_max, args.resolution)
    X, T = np.meshgrid(x_grid, t_grid, indexing='ij')

    psis = np.stack([
        psi_region(X, T, r['bounds_lower'], r['bounds_upper'], sigma_fraction, N)
        for r in regions
    ], axis=0)
    denom = np.maximum(psis.sum(axis=0), 1e-12)
    psi_norm = psis / denom

    n_cols = min(4, n_experts)
    n_rows = (n_experts + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows),
                             squeeze=False, layout='constrained')
    axes = axes.flatten()
    cmap = plt.cm.Reds

    im = None
    for i, r in enumerate(regions):
        ax = axes[i]
        im = ax.pcolormesh(x_grid, t_grid, psi_norm[i].T, cmap=cmap,
                           vmin=0, vmax=1, shading='auto')
        lo, hi = r['bounds_lower'], r['bounds_upper']
        ax.add_patch(patches.Rectangle(
            (lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
            linewidth=2, edgecolor='black', facecolor='none', linestyle='--'))
        ax.text(0.04, 0.94, f'E{i}', transform=ax.transAxes, fontsize=LABEL_SIZE,
                fontweight='bold', color='black', va='top', ha='left',
                path_effects=[path_effects.withStroke(linewidth=2.5, foreground='white')])
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        _style_axes(ax)

    for j in range(n_experts, len(axes)):
        axes[j].set_visible(False)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes[:n_experts].tolist(),
                            label='Normalized weight', shrink=0.85, pad=0.01)
        _style_colorbar(cbar)

    sigma_tag = f"{sigma_fraction:g}".replace('.', 'p')
    out_path = args.out_dir / f'soft_weights_{problem}_E{n_experts}_sigma{sigma_tag}.png'
    save_png(out_path, fig=fig)
    plt.close(fig)
    print(f"saved {out_path.name}")


if __name__ == '__main__':
    main()
