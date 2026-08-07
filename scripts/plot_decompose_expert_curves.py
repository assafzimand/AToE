"""Standalone: one clean image per expert (region + interface samples + loss terms).

During phase-3 (local-experts) training, ``training_plots/expert_curves_after_
<segment>_E<n>.png`` packs every expert into one row each: a term-wise loss
curve, and its region highlighted on the ground truth with its IC/BC/
interface sample points scattered on top. This script decomposes that into
one standalone file PER EXPERT, same two panels, paper-clean (no titles).

Data sources:
  - Region bounds, leaf indices: metrics.json['per_expert_rel_l2'][segment]
  - Loss-term curves (dense, per training step): metrics.json['split_expert_losses'][segment]
  - Ground truth heatmap: solved fresh from config_used.yaml (no checkpoint needed)
  - IC/BC/interface sample points: NOT persisted anywhere (they were an
    ephemeral training-time draw) -- re-drawn here with the training
    codebase's own sampler (adaptive.subdomain_data.build_subdomain_static)
    using the run's actual region boxes and config, so face classification
    (true IC/BC vs. interior interface) and per-face point counts match
    training exactly. Only the *locations* are geometric/config-driven; the
    interface *target values* it also computes are irrelevant here (a
    freshly-initialized throwaway model stands in) and are discarded. The
    exact points drawn during training are not reproduced (no checkpoint,
    no epoch-specific seed) -- this is the same sampling scheme, not a
    replay.

Usage:
    python scripts/plot_decompose_expert_curves.py <run_dir> [--segment phase3] [--out-dir DIR] [--seed 0]
"""

import sys
import json
import importlib
import argparse
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.plot_io import save_png
from adaptive.indicators import RegionDescriptor
from adaptive.subdomain_data import build_subdomain_static


_TERM_COLORS = {
    'residual': '#e74c3c', 'ic': '#3498db', 'interface_ic': '#9b59b6',
    'interface_bc': '#f39c12', 'interface_bc_dx': '#d68910',
    'interface_bc_dxx': '#b9770e', 'interface_bc_dxxx': '#7e5109',
    'bc': '#2ecc71', 'bc_dx': '#27ae60', 'bc_dxx': '#1e8449', 'bc_dxxx': '#145a32',
    'continuity': '#e67e22', 'total': '#2c3e50',
}

# Sample-point kind codes, matching adaptive/subdomain_data.py exactly.
_KIND_IC_TRUE, _KIND_IFACE_IC, _KIND_IFACE_BC, _KIND_BC_TRUE = 1, 2, 3, 4
_ICBC_STYLE = {
    _KIND_IC_TRUE:  ('IC true',      '#3498db', 'o', 16),
    _KIND_IFACE_IC: ('Interface IC', '#9b59b6', 's', 16),
    _KIND_IFACE_BC: ('Interface BC', '#f39c12', '^', 16),
    _KIND_BC_TRUE:  ('BC true',      '#2ecc71', 'D', 16),
}


def load_run(run_dir: Path):
    metrics = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
    cfg = yaml.safe_load((run_dir / 'config_used.yaml').read_text(encoding='utf-8'))
    return metrics, cfg


def native_grid(cfg):
    """Ground-truth solution on the solver's native grid (no checkpoint needed)."""
    problem = cfg['problem']
    solver = importlib.import_module(f'solvers.{problem}_solver')
    x_grid, t_grid, h_sol = solver._get_solution_cached(cfg)
    t0, t1 = cfg[problem]['temporal_domain']
    t_grid = np.asarray(t_grid)
    t_mask = (t_grid >= t0 - 1e-12) & (t_grid <= t1 + 1e-12)
    return np.asarray(x_grid), t_grid[t_mask], np.asarray(h_sol)[t_mask]


def build_icbc_samples(cfg, regions, leaf_indices, seed=0):
    """Re-draw IC/BC/interface sample locations with the training sampler.

    Returns {leaf_idx: {kind_code: (x_arr, t_arr)}} for the 4 scatter kinds
    plot_per_expert_curves shows (residual/continuity points excluded, same
    as the original).
    """
    import torch

    if cfg.get('precision', 'float32') == 'float64':
        torch.set_default_dtype(torch.float64)
    device = torch.device('cpu')

    problem = cfg['problem']
    pc = cfg[problem]
    spatial_dim = pc['spatial_dim']
    output_dim = pc['output_dim']

    # Interface target VALUES require a model forward pass; we only want the
    # sample LOCATIONS, so a throwaway randomly-initialized model stands in
    # and its output is discarded below.
    dummy_model = torch.nn.Linear(spatial_dim + 1, output_dim).to(device)
    dummy_model.eval()

    static = build_subdomain_static(
        dummy_model, leaf_indices, regions, cfg, device,
        seed=seed, interface_model=None)

    x = static['x'].detach().cpu().numpy()
    t = static['t'].detach().cpu().numpy()
    eid = static['expert_id'].detach().cpu().numpy()
    kind = static['kind'].detach().cpu().numpy()

    by_expert = {}
    for idx in leaf_indices:
        emask = (eid == idx)
        by_expert[idx] = {}
        for kcode in _ICBC_STYLE:
            kmask = emask & (kind == kcode)
            if kmask.any():
                by_expert[idx][kcode] = (x[kmask, 0], t[kmask, 0])
    return by_expert


def plot_expert_figure(pde, seg, idx, leaf_indices, bounds_lower, bounds_upper,
                       x_grid, t_grid, gt_img, icbc_for_expert, loss_terms,
                       final_rel_l2, out_dir):
    extent = [x_grid.min(), x_grid.max(), t_grid.min(), t_grid.max()]
    my_lo = bounds_lower[leaf_indices.index(idx)]
    my_hi = bounds_upper[leaf_indices.index(idx)]

    fig, (ax_reg, ax_loss) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Left: region on GT + interface samples ──
    im = ax_reg.imshow(gt_img, origin='lower', aspect='auto', extent=extent,
                       cmap='viridis', alpha=0.85, zorder=0)
    plt.colorbar(im, ax=ax_reg, fraction=0.046, pad=0.04)
    for other_idx, lo, hi in zip(leaf_indices, bounds_lower, bounds_upper):
        is_mine = (other_idx == idx)
        ax_reg.add_patch(patches.Rectangle(
            (lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
            linewidth=2.5 if is_mine else 0.8,
            edgecolor='red' if is_mine else 'black',
            facecolor='none', alpha=1.0 if is_mine else 0.35,
            zorder=10 if is_mine else 5))
    for kcode, (label, color, marker, ms) in _ICBC_STYLE.items():
        if kcode in icbc_for_expert:
            xs, ts = icbc_for_expert[kcode]
            ax_reg.scatter(xs, ts, s=ms, c=color, marker=marker, label=label,
                          zorder=20, alpha=0.85, linewidths=0)
    pad = 0.05
    xr, tr = my_hi[0] - my_lo[0], my_hi[1] - my_lo[1]
    ax_reg.set_xlim(my_lo[0] - pad * xr - 0.15 * (extent[1] - extent[0]),
                    my_hi[0] + pad * xr + 0.15 * (extent[1] - extent[0]))
    ax_reg.set_ylim(max(extent[2], my_lo[1] - pad * tr - 0.15 * (extent[3] - extent[2])),
                    min(extent[3], my_hi[1] + pad * tr + 0.15 * (extent[3] - extent[2])))
    ax_reg.set_xlabel('x', fontsize=12)
    ax_reg.set_ylabel('t', fontsize=12)
    ax_reg.tick_params(labelsize=10)
    if icbc_for_expert:
        ax_reg.legend(fontsize=9, loc='upper right', framealpha=0.9)

    # ── Right: term-wise loss curves ──
    any_plotted = False
    for term, values in loss_terms.items():
        if not values:
            continue
        v = np.asarray(values, dtype=float)
        v = np.where(v > 0, v, np.nan)
        if not np.isfinite(v).any():
            continue
        ax_loss.plot(np.arange(1, len(v) + 1), v,
                    color=_TERM_COLORS.get(term, 'gray'), label=term,
                    linewidth=1.3, alpha=0.85)
        any_plotted = True
    if any_plotted:
        ax_loss.set_yscale('log')
        ax_loss.legend(fontsize=9)
    ax_loss.set_xlabel('Training step (segment-local)', fontsize=12)
    ax_loss.set_ylabel('Loss term (log)', fontsize=12)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.tick_params(labelsize=10)

    plt.tight_layout()
    stat = f'expert_E{idx}_{pde}_{seg}'
    if final_rel_l2 is not None:
        stat += f'_relL2_{final_rel_l2:.2e}'
    out_path = out_dir / f'{stat}.png'
    save_png(out_path, fig=fig)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--segment', default='phase3')
    ap.add_argument('--out-dir', type=Path,
                     default=Path('outputs/paper_figures/expert_curves'))
    ap.add_argument('--seed', type=int, default=0,
                     help='Seed for the re-drawn IC/BC/interface sample locations '
                          '(illustrative -- see module docstring)')
    args = ap.parse_args()

    metrics, cfg = load_run(args.run_dir)
    pde = cfg['problem']
    seg = args.segment

    pe = metrics.get('per_expert_rel_l2', {}).get(seg)
    if pe is None:
        raise SystemExit(
            f"No per_expert_rel_l2['{seg}'] in {args.run_dir}/metrics.json "
            f"(available segments: {list(metrics.get('per_expert_rel_l2', {}).keys())})")
    leaf_indices = pe['leaf_indices']
    bounds_lower = pe['bounds_lower']
    bounds_upper = pe['bounds_upper']

    sel = metrics.get('split_expert_losses', {}).get(seg, {})
    if not sel:
        print(f"Warning: no split_expert_losses['{seg}'] -- loss-term panels will be empty.")

    print(f"PDE: {pde}  segment: {seg}  experts: {len(leaf_indices)}")
    x_grid, t_grid, h_sol = native_grid(cfg)
    gt_img = np.abs(h_sol) if np.iscomplexobj(h_sol) else h_sol

    regions = {idx: RegionDescriptor(bounds_lower=lo, bounds_upper=hi)
               for idx, lo, hi in zip(leaf_indices, bounds_lower, bounds_upper)}
    print(f"Drawing illustrative IC/BC/interface sample locations (seed={args.seed})...")
    icbc = build_icbc_samples(cfg, regions, leaf_indices, seed=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for idx, lo, hi in zip(leaf_indices, bounds_lower, bounds_upper):
        vals = pe['experts'].get(str(idx))
        final_rel_l2 = vals[-1] if vals else None
        out_path = plot_expert_figure(
            pde, seg, idx, leaf_indices, bounds_lower, bounds_upper,
            x_grid, t_grid, gt_img, icbc.get(idx, {}),
            sel.get(str(idx), {}), final_rel_l2, args.out_dir)
        print(f"  saved {out_path.name}")


if __name__ == '__main__':
    main()
