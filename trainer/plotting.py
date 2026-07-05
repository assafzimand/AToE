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


def plot_training_curves(
    metrics: Dict[str, List[float]], 
    save_dir: Path,
    optimizer_switch_epochs: List[int] = None,
    segment_start_epochs: List[int] = None
) -> None:
    """
    Plot training and evaluation curves.

    Args:
        metrics: Dictionary with keys:
                - 'train_loss_epochs', 'train_loss' (all epochs)
                - 'epochs', 'eval_loss', 'eval_rel_l2' (eval epochs only)
                - Optional: 'loss_components' dict with 'epochs', 'residual', 'ic', 'bc' lists
        save_dir: Directory to save plots
        optimizer_switch_epochs: List of epochs where optimizer switched.
                                Green dashed vertical lines drawn at each.
        segment_start_epochs: List of epochs where new training segments started.
                             Blue dotted vertical lines drawn at each.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loss_epochs = metrics['train_loss_epochs']
    eval_epochs = metrics['epochs']

    optimizer_switch_epochs = optimizer_switch_epochs or []
    segment_start_epochs = segment_start_epochs or []
    
    # Check if we have loss components for term-wise plot
    loss_comps = metrics.get('loss_components', {})
    has_components = (loss_comps.get('epochs') and 
                      len(loss_comps.get('epochs', [])) > 0 and
                      any(loss_comps.get(k) for k in ['residual', 'ic', 'bc']))

    # Create figure with 2 or 3 subplots depending on whether we have components
    n_plots = 3 if has_components else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))

    # Plot 1: Loss curves
    ax = axes[0]
    ax.plot(train_loss_epochs, metrics['train_loss'], 'b-', label='Train Loss',
            linewidth=2, alpha=0.8)
    ax.plot(eval_epochs, metrics['eval_loss'], 'r-', label='Eval Loss',
            linewidth=2, alpha=0.8)
    
    # Add optimizer switch markers (green dashed)
    for i, epoch in enumerate(optimizer_switch_epochs):
        label = 'Optimizer Switch' if i == 0 else None
        ax.axvline(x=epoch, color='green', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label=label)
    
    # Add segment boundary markers (blue dotted), skip epoch 1 (start of first segment)
    _seg_labeled = False
    for epoch in segment_start_epochs:
        if epoch <= 1:
            continue
        ax.axvline(x=epoch, color='blue', linestyle=':',
                   linewidth=1.5, alpha=0.6,
                   label='New Level Start' if not _seg_labeled else None)
        _seg_labeled = True
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    is_log_loss = _safe_log_scale(ax, [metrics['train_loss'], metrics['eval_loss']])
    scale_str_loss = "[log]" if is_log_loss else "[linear]"
    ax.set_title(f'Training and Evaluation Loss {scale_str_loss}', fontsize=14, fontweight='bold')

    # Plot 2: Relative L2 error
    ax = axes[1]
    ax.plot(eval_epochs, metrics['eval_rel_l2'], 'r-', label='Eval Rel. L2',
            linewidth=2, alpha=0.8)

    # Root rel-L2 baseline (horizontal black line), if available
    root_rel_l2 = metrics.get('root_rel_l2')
    if root_rel_l2 is not None and root_rel_l2 > 0:
        ax.axhline(y=root_rel_l2, color='black', linestyle='-',
                   linewidth=1.5, alpha=0.8,
                   label=f'Root rel-L2 ({root_rel_l2:.2e})')

    # Add optimizer switch markers (green dashed)
    for i, epoch in enumerate(optimizer_switch_epochs):
        label = 'Optimizer Switch' if i == 0 else None
        ax.axvline(x=epoch, color='green', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label=label)
    
    # Add segment boundary markers (blue dotted), skip epoch 1
    _seg_labeled = False
    for epoch in segment_start_epochs:
        if epoch <= 1:
            continue
        ax.axvline(x=epoch, color='blue', linestyle=':',
                   linewidth=1.5, alpha=0.6,
                   label='New Level Start' if not _seg_labeled else None)
        _seg_labeled = True
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    _l2_series = [metrics['eval_rel_l2']]
    if root_rel_l2 is not None and root_rel_l2 > 0:
        _l2_series.append([root_rel_l2])
    is_log_l2 = _safe_log_scale(ax, _l2_series)
    scale_str_l2 = "[log]" if is_log_l2 else "[linear]"
    ax.set_title(f'Relative L2 Error {scale_str_l2}', fontsize=14, fontweight='bold')

    # Plot 3: Term-wise loss components (if available)
    if has_components:
        ax = axes[2]
        comp_epochs = loss_comps['epochs']
        
        # Color scheme for different loss terms
        term_colors = {
            'residual': '#e74c3c',  # red
            'ic': '#3498db',         # blue
            'bc': '#2ecc71',         # green
            'continuity': '#e67e22', # orange
        }
        term_labels = {
            'residual': 'PDE Residual',
            'ic': 'Initial Condition',
            'bc': 'Boundary Condition',
            'continuity': 'Continuity',
        }
        
        values_for_log = []
        for term in ['residual', 'ic', 'bc', 'continuity']:
            if loss_comps.get(term) and len(loss_comps[term]) > 0:
                values = loss_comps[term]
                ax.plot(comp_epochs, values, '-', 
                       color=term_colors.get(term, 'gray'),
                       label=term_labels.get(term, term),
                       linewidth=1.5, alpha=0.8)
                values_for_log.append(values)
        
        # Add optimizer switch markers
        for i, epoch in enumerate(optimizer_switch_epochs):
            label = 'Optimizer Switch' if i == 0 else None
            ax.axvline(x=epoch, color='green', linestyle='--', 
                       linewidth=1.5, alpha=0.7, label=label)
        
        # Add segment boundary markers
        _seg_labeled = False
        for epoch in segment_start_epochs:
            if epoch <= 1:
                continue
            ax.axvline(x=epoch, color='blue', linestyle=':',
                       linewidth=1.5, alpha=0.6,
                       label='New Level Start' if not _seg_labeled else None)
            _seg_labeled = True
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss Component', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        is_log_comp = _safe_log_scale(ax, values_for_log) if values_for_log else False
        scale_str_comp = "[log]" if is_log_comp else "[linear]"
        ax.set_title(f'Loss Components {scale_str_comp}', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure
    save_path = save_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Training curves saved to {save_path}")


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
        'interface_bc': '#f39c12',
        'bc': '#2ecc71',
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
            ax.plot(
                values,
                color=term_colors.get(term, 'gray'),
                label=term,
                linewidth=1.2,
                alpha=0.85,
            )
            vals_for_log.append(values)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        _safe_log_scale(ax, vals_for_log)
        ax.set_title(f'Expert {eidx}', fontsize=11)

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

        # Scatter IC/BC interface samples for this expert
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
            ax2.legend(fontsize=6, loc='upper right',
                       markerscale=1.2, framealpha=0.7)

        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
        ax2.set_title(f'Region (expert {eidx})', fontsize=10)

    fig.suptitle(
        f'Per-Expert Curves — {segment_name}',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


