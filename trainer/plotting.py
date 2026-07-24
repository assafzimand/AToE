"""Plotting utilities for training metrics."""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import numpy as np


def _safe_log_scale(ax, values_list):
    """Set log scale on y-axis only if all data has positive values.
    
    Returns:
        bool: True if log scale was applied, False if linear scale is used.
    """
    all_values = []
    for v in values_list:
        if isinstance(v, (list, np.ndarray)):
            all_values.extend(np.array(v).flatten())
        else:
            all_values.append(v)
    all_values = np.array(all_values)
    # Filter out NaN values for the check
    valid_values = all_values[~np.isnan(all_values)]
    if len(valid_values) > 0 and np.all(valid_values > 0):
        ax.set_yscale('log')
        return True
    return False


# Per-segment marker style for the training-curve vertical lines:
# name -> (color, legend label). Strongly contrasting colors, avoiding green
# (optimizer switch) and red/blue solid (loss curves).
_SEGMENT_STYLES = {
    'root':      ('#1f77b4', 'Root start'),
    'phase3':    ('#8e24aa', 'Local Experts start'),   # vivid purple
    'fine_tune': ('#e67e22', 'Fine-Tune start'),       # orange
}


def _draw_segment_markers(ax, segment_markers):
    """Vertical dashed lines at segment starts, one color+label per segment.

    Bolder than the optimizer-switch markers (these are the primary phase
    boundaries). ``segment_markers`` is a list of (start_epoch, segment_name)
    tuples. Epoch <= 1 (start of the first segment) is skipped. Each segment
    name is labeled once per axis.
    """
    labeled = set()
    for epoch, name in segment_markers:
        if epoch <= 1:
            continue
        color, label = _SEGMENT_STYLES.get(
            name, ('#7f7f7f', f'{name} start'))
        ax.axvline(x=epoch, color=color, linestyle='--',
                   linewidth=2.5, alpha=0.9,
                   label=label if name not in labeled else None)
        labeled.add(name)


def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_dir: Path,
    optimizer_switch_epochs: List[int] = None,
    segment_markers: List = None,
    name_suffix: str = ''
) -> None:
    """
    Plot training curves (paper-ready: no panel titles; the log/linear scale
    is noted in the y-label; run metadata goes into the filename via
    ``name_suffix``). Saves the combined figure AND each panel as its own
    figure, as PNG.

    Args:
        metrics: Dictionary with keys:
                - 'train_loss_epochs', 'train_loss' (all epochs)
                - 'epochs', 'rel_l2' (solver-grid rel-L2, eval epochs only)
                - Optional: 'loss_components' dict with 'epochs', 'residual', 'ic', 'bc' lists
        save_dir: Directory to save plots
        optimizer_switch_epochs: List of epochs where optimizer switched.
                                Green dashed vertical lines drawn at each.
        segment_markers: List of (start_epoch, segment_name) tuples for
                        training-segment boundaries. Dashed vertical lines,
                        one color + legend label per segment name.
        name_suffix: Appended to filenames, e.g. 'burgers1d_ep39300_E7' →
                    training_curves_burgers1d_ep39300_E7.png
    """
    from utils.plot_io import save_png

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loss_epochs = metrics['train_loss_epochs']
    eval_epochs = metrics['epochs']

    optimizer_switch_epochs = optimizer_switch_epochs or []
    segment_markers = segment_markers or []

    # Check if we have loss components for term-wise plot
    loss_comps = metrics.get('loss_components', {})
    has_components = (loss_comps.get('epochs') and
                      len(loss_comps.get('epochs', [])) > 0 and
                      any(loss_comps.get(k) for k in ['residual', 'ic', 'bc']))

    def _draw_markers(ax):
        for i, epoch in enumerate(optimizer_switch_epochs):
            label = 'Optimizer switch' if i == 0 else None
            ax.axvline(x=epoch, color='green', linestyle='--',
                       linewidth=1.5, alpha=0.7, label=label)
        _draw_segment_markers(ax, segment_markers)

    def _finish(ax, ylabel, log_series):
        ax.set_xlabel('Epoch', fontsize=14)
        is_log = _safe_log_scale(ax, log_series) if log_series else False
        ax.set_ylabel(f'{ylabel}{" (log)" if is_log else ""}', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)

    # Panel drawers (each renders into a given ax; reused for the combined
    # figure and the standalone per-panel figures)
    def _panel_loss(ax):
        # Training loss only — patience and this curve both track train loss;
        # eval loss is intentionally not shown.
        ax.plot(train_loss_epochs, metrics['train_loss'], 'b-',
                label='Train loss', linewidth=2, alpha=0.8)
        _draw_markers(ax)
        _finish(ax, 'Loss', [metrics['train_loss']])

    def _panel_rel_l2(ax):
        # 'rel_l2' is the solver-grid metric ('eval_rel_l2' in older runs)
        _rl2 = metrics.get('rel_l2', metrics.get('eval_rel_l2', []))
        ax.plot(eval_epochs, _rl2, 'r-',
                label='Rel. $L^2$ error', linewidth=2, alpha=0.8)
        # Baseline (horizontal black line): the loaded phase-3 expert
        # checkpoint's rel-L2 when the run started from
        # pretrained_local_expert_checkpoint; otherwise the root's rel-L2,
        # shown only when the root was LOADED from a checkpoint (if it was
        # trained in this session, the curve already contains that phase).
        experts_rel_l2 = metrics.get('pretrained_experts_rel_l2')
        root_rel_l2 = metrics.get('root_rel_l2')
        if experts_rel_l2 is not None and experts_rel_l2 > 0:
            ax.axhline(y=experts_rel_l2, color='black', linestyle='-',
                       linewidth=1.5, alpha=0.8,
                       label=f'Phase-3 ckpt ({experts_rel_l2:.2e})')
        elif (root_rel_l2 is not None and root_rel_l2 > 0
              and metrics.get('root_loaded_from_checkpoint', False)):
            ax.axhline(y=root_rel_l2, color='black', linestyle='-',
                       linewidth=1.5, alpha=0.8,
                       label=f'Root ({root_rel_l2:.2e})')
        _draw_markers(ax)
        # Safe log scale (falls back to linear if any value is <= 0). The
        # log scale is plot-only: reported/tabulated rel-L2 stays raw.
        _finish(ax, 'Relative $L^2$ error', [_rl2])

    def _panel_components(ax):
        comp_epochs = loss_comps['epochs']
        term_colors = {
            'residual': '#e74c3c',  # red
            'ic': '#3498db',         # blue
            'bc': '#2ecc71',         # green
            'bc_dx': '#27ae60',      # darker green — 1st-deriv periodicity
            'bc_dxx': '#1e8449',     # 2nd-deriv (KdV, KS)
            'bc_dxxx': '#145a32',    # 3rd-deriv (KS)
            'continuity': '#e67e22', # orange
            'l2sp': '#9b59b6',       # purple
        }
        term_labels = {
            'residual': 'PDE residual',
            'ic': 'Initial condition',
            'bc': 'Boundary condition',
            'bc_dx': 'BC ∂ₓ periodicity',
            'bc_dxx': 'BC ∂ₓₓ periodicity',
            'bc_dxxx': 'BC ∂ₓₓₓ periodicity',
            'continuity': 'Continuity',
            'l2sp': 'L2-SP anchor',
        }
        values_for_log = []
        for term in ['residual', 'ic', 'bc', 'bc_dx', 'bc_dxx', 'bc_dxxx',
                     'continuity', 'l2sp']:
            if loss_comps.get(term) and len(loss_comps[term]) > 0:
                values = loss_comps[term]
                # Skip terms that are all-zero for this run (l2sp when anchoring
                # is off; bc_d* for Dirichlet problems / PDEs without that order)
                if (term in ('l2sp', 'bc_dx', 'bc_dxx', 'bc_dxxx')
                        and not any(v > 0 for v in values)):
                    continue
                ax.plot(comp_epochs, values, '-',
                        color=term_colors.get(term, 'gray'),
                        label=term_labels.get(term, term),
                        linewidth=1.5, alpha=0.8)
                values_for_log.append(values)
        _draw_markers(ax)
        _finish(ax, 'Loss component', values_for_log)

    def _panel_drift(ax):
        # Anchor drift ||θ-θ₀|| in raw weight-space units — the λ-independent
        # view of what the L2-SP penalty means (the penalty line in the
        # components panel is λ/2·drift², whose log-scale 'jump' from an
        # exactly-zero start is visually misleading).
        drift = loss_comps.get('l2sp_drift', [])
        comp_epochs = loss_comps['epochs']
        anchor_norm = metrics.get('l2sp_anchor_norm')
        label = r'$\|\theta-\theta_0\|$'
        if anchor_norm and drift:
            label += (f'  (final: {drift[-1]:.2e} = '
                      f'{drift[-1] / anchor_norm:.1e} of anchor norm '
                      f'{anchor_norm:.1f})')
        ax.plot(comp_epochs, drift, '-', color='#9b59b6',
                label=label, linewidth=1.8, alpha=0.9)
        _draw_markers(ax)
        _finish(ax, 'Weight drift from anchor', [drift])

    panels = [('loss', _panel_loss), ('rel_l2', _panel_rel_l2)]
    if has_components:
        panels.append(('components', _panel_components))
    if any(v > 0 for v in loss_comps.get('l2sp_drift', [])):
        panels.append(('anchor_drift', _panel_drift))

    suffix = f'_{name_suffix}' if name_suffix else ''

    # Combined figure
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (_, draw) in zip(axes, panels):
        draw(ax)
    plt.tight_layout()
    save_path = save_png(save_dir / f'training_curves{suffix}.png', fig=fig)
    plt.close(fig)

    # Standalone per-panel figures (papers rarely place all panels together)
    for key, draw in panels:
        fig_s, ax_s = plt.subplots(figsize=(7, 5))
        draw(ax_s)
        plt.tight_layout()
        save_png(save_dir / f'training_curves_{key}{suffix}.png', fig=fig_s)
        plt.close(fig_s)

    print(f"  Training curves saved to {save_path} (+ per-panel files)")


def plot_per_expert_region_report(
    epochs: List,
    series: Dict,
    leaf_indices: List[int],
    bounds_lower: List,
    bounds_upper: List,
    grids: Dict,
    gt_sq_grid,
    x_grid,
    t_grid,
    out_path,
    segment_name: str,
    gt_grid=None,
    gt_x=None,
    gt_t=None,
    loss_series: Dict = None,
    loss_epochs: List = None,
) -> None:
    """Per-expert region report for one segment.

    One row per leaf expert: a ground-truth heatmap with all leaf boundaries
    and this row's region highlighted (like plot_per_expert_curves), its
    region rel-L2 curve (eval cadence, same metric grid as the global curve),
    per-term loss curves measured on the composed u_θ restricted to Ω_j
    (when ``loss_series`` is given), then log10 |err| heatmaps of its tile at
    segment start / segment best / segment final (final only when it differs
    from best).

    Args:
        epochs: eval epochs of the segment (shared x-axis for all curves).
        series: expert_idx (str or int) -> list of region rel-L2 values.
        grids: state name ('start' / 'best' / 'final') ->
            (err_grid (nt, nx), epoch_label). 'final' key optional.
        gt_sq_grid, x_grid, t_grid: from
            compute_native_grid_metrics(..., return_grids=True).
        gt_grid, gt_x, gt_t: optional plot-ready ground truth (ctx.gt_grid
            orientation: (nx, nt)) for the region-locator panel; the panel is
            omitted when not given.
    """
    import matplotlib.patches as patches

    states = [s for s in ('start', 'best', 'final') if s in grids]
    n = len(leaf_indices)
    if n == 0 or not states:
        return
    has_gt_panel = (gt_grid is not None and gt_x is not None
                    and gt_t is not None)
    has_loss_panel = bool(loss_series) and loss_epochs is not None
    ncols = ((2 if has_gt_panel else 1) + (1 if has_loss_panel else 0)
             + len(states))
    fig, axes = plt.subplots(n, ncols, figsize=(4.2 * ncols, 3.0 * n),
                             squeeze=False)
    _loss_term_style = {
        'residual': '#e74c3c', 'ic': '#3498db',
        'bc': '#2ecc71', 'total': '#2c3e50',
    }

    if has_gt_panel:
        gt_disp = (np.linalg.norm(gt_grid, axis=2)
                   if getattr(gt_grid, 'ndim', 2) == 3 else gt_grid)
        GT_T, GT_X = np.meshgrid(gt_t, gt_x)

    for row, (eidx, lo, hi) in enumerate(
            zip(leaf_indices, bounds_lower, bounds_upper)):
        col0 = 0
        if has_gt_panel:
            axg = axes[row][0]
            axg.pcolormesh(GT_X, GT_T, gt_disp, shading='auto',
                           cmap='viridis', alpha=0.7, zorder=0)
            for ri, (rlo, rhi) in enumerate(zip(bounds_lower, bounds_upper)):
                is_this = (ri == row)
                axg.add_patch(patches.Rectangle(
                    (rlo[0], rlo[1]), rhi[0] - rlo[0], rhi[1] - rlo[1],
                    linewidth=2.5 if is_this else 0.8,
                    edgecolor='red' if is_this else 'black',
                    facecolor='none', zorder=10 if is_this else 5))
            axg.set_title(f"Expert {eidx}", fontsize=10)
            axg.set_ylabel('t', fontsize=8)
            if row == n - 1:
                axg.set_xlabel('x')
            col0 = 1

        vals = series.get(str(eidx), series.get(eidx, []))
        ax = axes[row][col0]
        if vals:
            ax.semilogy(epochs[:len(vals)], vals, 'r-', linewidth=1.5)
        ax.set_title(f"E{eidx}  x[{lo[0]:.2f},{hi[0]:.2f}] "
                     f"t[{lo[1]:.2f},{hi[1]:.2f}]", fontsize=9)
        ax.set_ylabel('region rel-$L^2$', fontsize=8)
        ax.grid(alpha=0.3)
        if row == n - 1:
            ax.set_xlabel('epoch')

        if has_loss_panel:
            col0 += 1
            axl = axes[row][col0]
            _es = loss_series.get(str(eidx), loss_series.get(eidx, {}))
            _plotted = False
            for _term, _tvals in _es.items():
                if not _tvals:
                    continue
                _v = np.asarray(_tvals, dtype=float)
                _v = np.where(_v > 0, _v, np.nan)  # keep the axis log-friendly
                if np.isfinite(_v).any():
                    axl.plot(loss_epochs[:len(_v)], _v,
                             color=_loss_term_style.get(_term, 'gray'),
                             label=_term, linewidth=1.2, alpha=0.85)
                    _plotted = True
            if _plotted:
                axl.set_yscale('log')
                axl.legend(fontsize=6)
            axl.set_title(r'loss terms on $u_\theta$ in $\Omega_j$',
                          fontsize=9)
            axl.grid(alpha=0.3)
            if row == n - 1:
                axl.set_xlabel('epoch')

        ix = (x_grid >= lo[0]) & (x_grid <= hi[0])
        it = (t_grid >= lo[1]) & (t_grid <= hi[1])
        if not ix.any() or not it.any():
            continue
        subs = {s: grids[s][0][np.ix_(it, ix)] for s in states}
        log_subs = {s: np.log10(subs[s] + 1e-16) for s in states}
        vmin = min(ls.min() for ls in log_subs.values())
        vmax = max(ls.max() for ls in log_subs.values())
        gt_sub_sum = gt_sq_grid[np.ix_(it, ix)].sum()
        for col, s in enumerate(states, start=col0 + 1):
            axh = axes[row][col]
            rel = float(np.sqrt((subs[s] ** 2).sum())
                        / (np.sqrt(gt_sub_sum) + 1e-10))
            im = axh.imshow(log_subs[s], origin='lower', aspect='auto',
                            extent=[lo[0], hi[0], lo[1], hi[1]],
                            cmap='hot', vmin=vmin, vmax=vmax)
            axh.set_title(f"{s} (ep {grids[s][1]}) rel={rel:.2e}", fontsize=9)
            plt.colorbar(im, ax=axh, fraction=0.046, pad=0.04)

    fig.suptitle(f"Per-expert region rel-L2 — segment '{segment_name}'",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_per_expert_curves(
    per_expert_history: dict,
    regions: list,
    save_path,
    domain_bounds: dict = None,
    gt_grid=None,
    grid_x=None,
    grid_t=None,
    segment_name: str = '',
    split_data: dict = None,
) -> None:
    """Per-expert term-wise loss + region-on-GT panel.

    Args:
        per_expert_history: ``{expert_idx: {term: [values]}}``
        regions: model.regions (list of RegionDescriptor)
        save_path: output file path
        domain_bounds: ``{'lower': [...], 'upper': [...]}``
        gt_grid: optional ground-truth 2-D array
        grid_x, grid_t: 1-D coordinate arrays for gt_grid
        segment_name: label for the figure title
        split_data: optional subdomain dataset dict with keys
            ``x``, ``t``, ``expert_id``, ``kind``.  When provided, the
            non-residual points (IC/interface/BC) are overlaid as a scatter
            on the bottom region panel, colour-coded by kind.
    """
    import matplotlib.patches as patches

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    expert_ids = sorted(per_expert_history.keys())
    n_experts = len(expert_ids)
    if n_experts == 0:
        return

    # Pre-process split_data into per-expert numpy arrays keyed by kind
    # KIND codes match adaptive/subdomain_data.py
    _KIND_IC_TRUE    = 1
    _KIND_IFACE_IC   = 2
    _KIND_IFACE_BC   = 3
    _KIND_BC_TRUE    = 4
    _icbc_kinds = {
        _KIND_IC_TRUE:  ('IC true',      '#3498db', 'o',  18),
        _KIND_IFACE_IC: ('Interface IC', '#9b59b6', 's',  18),
        _KIND_IFACE_BC: ('Interface BC', '#f39c12', '^',  18),
        _KIND_BC_TRUE:  ('BC true',      '#2ecc71', 'D',  18),
    }
    _sd_by_expert: dict = {}   # {eidx: {kind_code: (x_arr, t_arr)}}
    if split_data is not None:
        try:
            import torch
            sd_x   = split_data['x']
            sd_t   = split_data['t']
            sd_eid = split_data['expert_id']
            sd_k   = split_data['kind']
            if isinstance(sd_x, torch.Tensor):
                sd_x   = sd_x.cpu().numpy()
                sd_t   = sd_t.cpu().numpy()
                sd_eid = sd_eid.cpu().numpy()
                sd_k   = sd_k.cpu().numpy()
            for eidx in expert_ids:
                emask = (sd_eid == eidx)
                _sd_by_expert[eidx] = {}
                for kcode in _icbc_kinds:
                    kmask = emask & (sd_k == kcode)
                    if kmask.any():
                        _sd_by_expert[eidx][kcode] = (
                            sd_x[kmask, 0],
                            sd_t[kmask, 0],
                        )
        except Exception:
            _sd_by_expert = {}

    fig, axes = plt.subplots(
        2, n_experts,
        figsize=(5 * n_experts, 8),
        squeeze=False,
    )

    term_colors = {
        'residual': '#e74c3c',
        'ic': '#3498db',
        'interface_ic': '#9b59b6',
        'interface_bc': '#f39c12',        # amber — interface value match
        'interface_bc_dx': '#d68910',     # interface ∂ₓ match to root
        'interface_bc_dxx': '#b9770e',    # interface ∂ₓₓ match (KdV, KS)
        'interface_bc_dxxx': '#7e5109',   # interface ∂ₓₓₓ match (KS)
        'bc': '#2ecc71',                  # green — periodic value match
        'bc_dx': '#27ae60',               # periodic ∂ₓ match
        'bc_dxx': '#1e8449',              # periodic ∂ₓₓ match (KdV, KS)
        'bc_dxxx': '#145a32',             # periodic ∂ₓₓₓ match (KS)
        'continuity': '#e67e22',  # orange for continuity term
        'total': '#2c3e50',
    }

    for col, eidx in enumerate(expert_ids):
        eh = per_expert_history[eidx]

        # ── Top: term-wise loss curves ──
        ax = axes[0, col]
        vals_for_log = []
        for term, values in eh.items():
            if not values:
                continue
            # Zeros (e.g. exactly-satisfied terms) would force a linear axis
            # where the initial spike flattens the entire history — mask them
            # to NaN (plotted as gaps) so the axis can stay logarithmic.
            v = np.asarray(values, dtype=float)
            v_masked = np.where(v > 0, v, np.nan)
            ax.plot(
                v_masked,
                color=term_colors.get(term, 'gray'),
                label=term,
                linewidth=1.2,
                alpha=0.85,
            )
            if np.isfinite(v_masked).any():
                vals_for_log.append(v_masked[np.isfinite(v_masked)])
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss (log)', fontsize=11)
        ax.grid(True, alpha=0.3)
        if vals_for_log:
            ax.set_yscale('log')
        ax.set_title(f'Expert {eidx}', fontsize=12)

        # ── Bottom: region on GT heatmap ──
        ax2 = axes[1, col]
        if (gt_grid is not None
                and grid_x is not None
                and grid_t is not None):
            if gt_grid.ndim == 3:
                gt_disp = np.linalg.norm(gt_grid, axis=2)
            else:
                gt_disp = gt_grid
            T, X = np.meshgrid(grid_t, grid_x)
            ax2.pcolormesh(
                X, T, gt_disp,
                shading='auto', cmap='viridis',
                alpha=0.7, zorder=0,
            )

        if eidx < len(regions):
            r = regions[eidx]
            bl, bu = r.bounds_lower, r.bounds_upper
            rect = patches.Rectangle(
                (bl[0], bl[-1]),
                bu[0] - bl[0],
                bu[-1] - bl[-1],
                linewidth=2.5,
                edgecolor='red',
                facecolor='none',
                zorder=10,
            )
            ax2.add_patch(rect)

        # Draw all regions faintly
        for ri, r in enumerate(regions):
            if ri == eidx:
                continue
            bl, bu = r.bounds_lower, r.bounds_upper
            rect_f = patches.Rectangle(
                (bl[0], bl[-1]),
                bu[0] - bl[0],
                bu[-1] - bl[-1],
                linewidth=0.8,
                edgecolor='black',
                facecolor='none',
                alpha=0.3,
                zorder=9,
            )
            ax2.add_patch(rect_f)

        if domain_bounds:
            lo = domain_bounds['lower']
            hi = domain_bounds['upper']
            pad = 0.05
            xr = hi[0] - lo[0]
            tr = hi[-1] - lo[-1]
            ax2.set_xlim(lo[0] - pad * xr, hi[0] + pad * xr)
            ax2.set_ylim(lo[-1] - pad * tr, hi[-1] + pad * tr)

        # Scatter IC/BC interface samples for this expert (legend is shared
        # at figure level, so labels are attached only on the first column)
        if eidx in _sd_by_expert:
            for kcode, (label, color, marker, ms) in _icbc_kinds.items():
                if kcode in _sd_by_expert[eidx]:
                    xs, ts = _sd_by_expert[eidx][kcode]
                    ax2.scatter(
                        xs, ts,
                        s=ms, c=color, marker=marker,
                        label=label, zorder=20, alpha=0.8,
                        linewidths=0,
                    )

        ax2.set_xlabel('x', fontsize=11)
        ax2.set_ylabel('t', fontsize=11)
        ax2.set_title(f'Region E{eidx}', fontsize=12)

    # Shared figure-level legends (deduplicated across panels) instead of
    # one repeated legend per panel; no suptitle — segment/expert counts
    # belong in the filename and the paper caption.
    def _fig_legend(row_axes, loc_y):
        seen = {}
        for ax_ in row_axes:
            h_, l_ = ax_.get_legend_handles_labels()
            for h, l in zip(h_, l_):
                seen.setdefault(l, h)
        if seen:
            fig.legend(seen.values(), seen.keys(), ncol=len(seen),
                       fontsize=10, loc='upper center',
                       bbox_to_anchor=(0.5, loc_y), frameon=True)

    _fig_legend(axes[0, :], 1.06)   # loss terms
    _fig_legend(axes[1, :], 0.52)   # IC/BC scatter kinds

    plt.tight_layout()
    from utils.plot_io import save_png
    save_png(save_path, fig=fig)
    plt.close()


