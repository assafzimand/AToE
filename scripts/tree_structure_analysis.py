"""Tree structure analysis: spawning diagnostics visualization.

Dispatches to method-specific analysis pipelines based on the spawning
method recorded in expert_regions.json.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import numpy as np
from pathlib import Path
import sys
import torch
from scipy.interpolate import griddata


# =============================================================================
# Shared helpers
# =============================================================================

def _detect_problem_from_label(label: str) -> str:
    known = [
        'schrodinger', 'burgers2d', 'burgers1d', 'wave1d',
        'allen_cahn', 'kdv', 'ks', 'fisher_kpp', 'conv_diff',
    ]
    for p in known:
        if p in label.lower():
            return p
    return 'unknown'


def load_ground_truth(output_dir, problem=None):
    """Load ground truth for 2D problems (1 spatial + time).

    Returns (x_data, t_data, h_magnitude, eval_data) or (None,)*4.
    """
    if problem is None:
        problem = _detect_problem_from_label(str(output_dir))
    try:
        eval_data_path = Path("datasets") / problem / "eval_data.pt"
        if eval_data_path.exists():
            eval_data = torch.load(eval_data_path, map_location='cpu')
            x_data = eval_data['x'].numpy()
            if x_data.ndim == 2 and x_data.shape[1] > 1:
                return None, None, None, None
            t_data = eval_data['t'].numpy()
            for key in ('h_gt', 'h', 'u_gt'):
                if key in eval_data:
                    h_data = eval_data[key].numpy()
                    break
            else:
                return None, None, None, None
            if h_data.shape[1] == 2:
                h_magnitude = np.sqrt(h_data[:, 0]**2 + h_data[:, 1]**2)
            else:
                h_magnitude = h_data[:, 0]
            return x_data, t_data, h_magnitude, eval_data
    except Exception as e:
        print(f"    Could not load ground truth: {e}")
    return None, None, None, None


def _render_gt_bg(ax, x_data, t_data, h_magnitude):
    """Render ground truth as gray background."""
    if x_data is None:
        return
    res = 100
    gx = np.linspace(x_data.min(), x_data.max(), res)
    gt = np.linspace(t_data.min(), t_data.max(), res)
    Xg, Tg = np.meshgrid(gx, gt, indexing='ij')
    pts = np.column_stack([x_data, t_data])
    Hg = griddata(pts, h_magnitude, (Xg, Tg), method='cubic')
    im = ax.pcolormesh(gx, gt, Hg.T, shading='auto', cmap='gray', alpha=0.6, zorder=0)
    plt.colorbar(im, ax=ax, label='|h|')


def _load_loss_curves(metrics_path):
    train, evl = {}, {}
    if metrics_path and Path(metrics_path).exists():
        with open(metrics_path) as f:
            m = json.load(f)
        for ep, v in zip(m.get('train_loss_epochs', []), m.get('train_loss', [])):
            train[ep] = v
        for ep, v in zip(m.get('epochs', []), m.get('eval_loss', [])):
            evl[ep] = v
    return train, evl


def _dim_from_items(items, key='bounds_lower'):
    for it in items:
        bl = it.get(key)
        if bl:
            return len(bl)
    return 2


def _set_ax_limits(ax, items, key_lo='bounds_lower', key_hi='bounds_upper'):
    """Set 2D axis limits with margin from items that have bounds."""
    all_lo = [it[key_lo] for it in items if it.get(key_lo)]
    all_hi = [it[key_hi] for it in items if it.get(key_hi)]
    if not all_lo:
        return
    x0 = min(b[0] for b in all_lo)
    x1 = max(b[0] for b in all_hi)
    t0 = min(b[1] for b in all_lo)
    t1 = max(b[1] for b in all_hi)
    mx, mt = (x1 - x0) * 0.05, (t1 - t0) * 0.05
    ax.set_xlim(x0 - mx, x1 + mx)
    ax.set_ylim(t0 - mt, t1 + mt)
    ax.set_xlabel('Space (x)', fontsize=11)
    ax.set_ylabel('Time (t)', fontsize=11)


def _draw_rects(ax, items, value_key, cmap_name, cbar_label, alpha_face=0.45, lw=2.0):
    """Draw 2D rectangles colored by a value field. Returns (vmin, vmax)."""
    vals = [it[value_key] for it in items]
    if not vals:
        return 0, 1
    vmin, vmax = min(vals), max(vals)
    vrange = vmax - vmin if vmax > vmin else 1.0
    cmap = plt.get_cmap(cmap_name)

    for it in items:
        bl, bu = it['bounds_lower'], it['bounds_upper']
        v = it[value_key]
        nv = (v - vmin) / vrange if vrange > 0 else 0.5
        c = cmap(nv)
        face = list(c[:3]) + [alpha_face]
        rect = patches.Rectangle(
            (bl[0], bl[1]), bu[0] - bl[0], bu[1] - bl[1],
            linewidth=lw, edgecolor=c, facecolor=face, zorder=10)
        ax.add_patch(rect)

    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=cbar_label)
    return vmin, vmax


def _draw_rects_binary(ax, accepted, rejected, acc_color='#2ecc71', rej_color='#e74c3c'):
    """Draw 2D rectangles with fixed colors for accepted/rejected."""
    for it in accepted:
        bl, bu = it['bounds_lower'], it['bounds_upper']
        rect = patches.Rectangle(
            (bl[0], bl[1]), bu[0] - bl[0], bu[1] - bl[1],
            linewidth=2, edgecolor=acc_color, facecolor=list(plt.cm.colors.to_rgba(acc_color)[:3]) + [0.35],
            zorder=10)
        ax.add_patch(rect)
    for it in rejected:
        bl, bu = it['bounds_lower'], it['bounds_upper']
        rect = patches.Rectangle(
            (bl[0], bl[1]), bu[0] - bl[0], bu[1] - bl[1],
            linewidth=1.5, edgecolor=rej_color, facecolor=list(plt.cm.colors.to_rgba(rej_color)[:3]) + [0.2],
            linestyle='--', zorder=9)
        ax.add_patch(rect)

    legend_els = [
        Patch(facecolor=acc_color, alpha=0.35, edgecolor=acc_color, label=f'Accepted ({len(accepted)})'),
        Patch(facecolor=rej_color, alpha=0.2, edgecolor=rej_color, linestyle='--', label=f'Rejected ({len(rejected)})'),
    ]
    ax.legend(handles=legend_els, loc='upper right', fontsize=9, framealpha=0.9)


def _draw_rects_3d(ax, regions, cmap_name='RdYlGn', value_key='wavelet_norm_squared'):
    """Draw 3D wireframe boxes colored by a value."""
    vals = [r[value_key] for r in regions]
    if not vals:
        return
    vmin, vmax = min(vals), max(vals)
    vrange = vmax - vmin if vmax > vmin else 1.0
    cmap = plt.get_cmap(cmap_name)
    edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]

    for r in regions:
        lo, hi = r['bounds_lower'], r['bounds_upper']
        corners = np.array([
            [lo[0],lo[1],lo[2]], [hi[0],lo[1],lo[2]],
            [lo[0],hi[1],lo[2]], [hi[0],hi[1],lo[2]],
            [lo[0],lo[1],hi[2]], [hi[0],lo[1],hi[2]],
            [lo[0],hi[1],hi[2]], [hi[0],hi[1],hi[2]],
        ])
        nv = (r[value_key] - vmin) / vrange if vrange > 0 else 0.5
        c = cmap(nv)
        for i, j in edges:
            ax.plot3D(*zip(corners[i], corners[j]), color=c, linewidth=1.2, alpha=0.7)
    ax.set_xlabel('x0'); ax.set_ylabel('x1'); ax.set_zlabel('t')
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1, label=value_key.replace('_', ' ').title())


# =============================================================================
# by_mean_loss analysis
# =============================================================================

def _plot_loss_heatmaps(loss_entries, output_dir, x_data, t_data, h_magnitude):
    """Flat-colored rectangles (green=low loss, red=high loss) on GT background."""
    n = len(loss_entries)
    n_cols = min(n, 4)
    n_rows = max(1, (n + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    fig.suptitle('Loss Heatmap by Region', fontsize=16, fontweight='bold')

    for idx, entry in enumerate(loss_entries):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        epoch = entry['epoch']
        leaves = entry['leaves']
        _render_gt_bg(ax, x_data, t_data, h_magnitude)

        if not leaves:
            ax.set_title(f'Epoch {epoch} (no leaves)')
            continue

        losses = [lf['mean_loss'] for lf in leaves]
        vmin, vmax = min(losses), max(losses)
        vrange = vmax - vmin if vmax > vmin else 1.0

        for lf in leaves:
            bl, bu = lf['bounds_lower'], lf['bounds_upper']
            nv = (lf['mean_loss'] - vmin) / vrange if vrange > 0 else 0.5
            color = plt.cm.RdYlGn_r(nv)
            face = list(color[:3]) + [0.5]
            if len(bl) >= 2:
                rect = patches.Rectangle(
                    (bl[0], bl[1]), bu[0] - bl[0], bu[1] - bl[1],
                    linewidth=2, edgecolor=color, facecolor=face, zorder=10)
                ax.add_patch(rect)
                cx, cy = (bl[0] + bu[0]) / 2, (bl[1] + bu[1]) / 2
                ax.text(cx, cy, f'{lf["mean_loss"]:.4f}', ha='center', va='center',
                        fontsize=7, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6),
                        zorder=15)

        _set_ax_limits(ax, leaves)
        ax.set_title(f'Epoch {epoch}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, zorder=0)

        sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Mean Loss')

    for idx in range(n, n_rows * n_cols):
        ri, ci = divmod(idx, n_cols)
        axes[ri, ci].set_visible(False)

    plt.tight_layout()
    out = output_dir / 'loss_heatmap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Loss heatmap saved: {out.name}")


def _plot_loss_distributions(loss_entries, output_dir):
    """Per-epoch bar chart of leaf mean losses with mean line."""
    n = len(loss_entries)
    n_cols = min(n, 6)
    n_rows = max(1, (n + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle('Leaf Loss Distribution per Spawn Epoch', fontsize=16, fontweight='bold')

    for idx, entry in enumerate(loss_entries):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        leaves = entry['leaves']
        if not leaves:
            ax.set_title(f'Epoch {entry["epoch"]} (no data)')
            continue
        losses = [lf['mean_loss'] for lf in leaves]
        labels = [f"E{lf['leaf_idx']+1}" if lf['leaf_idx'] >= 0 else 'Base' for lf in leaves]
        worst = int(np.argmax(losses))
        colors = ['steelblue'] * len(losses)
        colors[worst] = 'crimson'
        ax.bar(labels, losses, color=colors, edgecolor='black', linewidth=0.8)
        ax.axhline(np.mean(losses), color='orange', ls=':', lw=1.5,
                    label=f'Mean: {np.mean(losses):.4f}')
        ax.set_title(f'Epoch {entry["epoch"]}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    for idx in range(n, n_rows * n_cols):
        ri, ci = divmod(idx, n_cols)
        axes[ri, ci].set_visible(False)
    plt.tight_layout()
    out = output_dir / 'loss_distributions.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Loss distributions saved: {out.name}")


def _plot_leaf_loss_bars(leaf_loss_history, output_dir):
    """Per-epoch grouped bar charts + summary overlay."""
    if not leaf_loss_history:
        return
    n_epochs = len(leaf_loss_history)
    colors_ep = plt.cm.viridis(np.linspace(0.2, 0.9, n_epochs))

    max_per_page = 6
    pages = [leaf_loss_history[i:i + max_per_page]
             for i in range(0, n_epochs, max_per_page)]

    for page_idx, page in enumerate(pages):
        nc = len(page)
        fig, axes = plt.subplots(1, nc, figsize=(5 * nc, 5), squeeze=False)
        fig.suptitle('Leaf Mean Loss at Each Spawn Step', fontsize=15, fontweight='bold')
        for col, entry in enumerate(page):
            ax = axes[0, col]
            leaves = entry['leaves']
            if not leaves:
                ax.set_title(f'Epoch {entry["epoch"]} (no leaves)')
                continue
            labels = [f"E{lf['leaf_idx']+1}" if lf['leaf_idx'] >= 0 else 'Base' for lf in leaves]
            losses = [lf['mean_loss'] for lf in leaves]
            worst = int(np.argmax(losses))
            bar_c = [colors_ep[page_idx * max_per_page + col]] * len(losses)
            bar_c[worst] = 'crimson'
            bars = ax.bar(labels, losses, color=bar_c, edgecolor='black', linewidth=0.8)
            for ib, bar in enumerate(bars):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{losses[ib]:.4f}', ha='center', va='bottom', fontsize=7, rotation=45)
            ax.set_title(f'Epoch {entry["epoch"]}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Mean Loss'); ax.set_xlabel('Leaf')
            ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        suffix = f'_p{page_idx + 1}' if len(pages) > 1 else ''
        out = output_dir / f'leaf_loss_bars{suffix}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Leaf loss bars saved: {out.name}")

    # Summary overlay
    fig, ax = plt.subplots(figsize=(max(10, 2 * n_epochs), 6))
    fig.suptitle('Leaf Mean Loss Across Spawn Steps', fontsize=15, fontweight='bold')
    all_ids = sorted({lf['leaf_idx'] for e in leaf_loss_history for lf in e['leaves']})
    id2lbl = {lid: (f'E{lid+1}' if lid >= 0 else 'Base') for lid in all_ids}
    x = np.arange(n_epochs)
    n_max = max(len(e['leaves']) for e in leaf_loss_history)
    bw = 0.8 / max(n_max, 1)
    handles = {}
    for ei, entry in enumerate(leaf_loss_history):
        leaves = entry['leaves']
        worst = int(np.argmax([lf['mean_loss'] for lf in leaves])) if leaves else -1
        for li, lf in enumerate(leaves):
            off = (li - len(leaves) / 2 + 0.5) * bw
            is_w = (li == worst)
            c = 'crimson' if is_w else colors_ep[ei]
            bar = ax.bar(x[ei] + off, lf['mean_loss'], bw, color=c, edgecolor='black', lw=0.5)
            lbl = id2lbl[lf['leaf_idx']]
            if lbl not in handles and not is_w:
                handles[lbl] = bar[0]
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ep {e['epoch']}" for e in leaf_loss_history], fontsize=9)
    ax.set_ylabel('Mean Loss'); ax.set_xlabel('Spawn Step')
    ax.grid(True, alpha=0.3, axis='y')
    leg_h = list(handles.values()) + [
        Patch(facecolor='crimson', edgecolor='black',
              label='Worst (split)')]
    leg_l = list(handles.keys()) + ['Worst (split)']
    ax.legend(leg_h, leg_l, fontsize=8, loc='upper right')
    plt.tight_layout()
    out = output_dir / 'leaf_loss_summary.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Leaf loss summary saved: {out.name}")


def _analyze_by_mean_loss(data, output_dir, metrics_path, x_data, t_data, h_magnitude):
    """Full analysis pipeline for by_mean_loss spawning."""
    leaf_loss_history = data.get('leaf_loss_history', [])
    diags = [d for d in data.get('spawning_diagnostics', []) if d.get('method') == 'by_mean_loss']

    source = leaf_loss_history if leaf_loss_history else [
        {'epoch': d['epoch'], 'leaves': d['evaluated_leaves']} for d in diags
    ]
    if not source:
        print("    No by_mean_loss data found")
        return

    print("    [by_mean_loss] Generating bar charts...")
    _plot_leaf_loss_bars(source, output_dir)

    if _dim_from_items(source[0].get('leaves', [])) == 2:
        print("    [by_mean_loss] Generating loss heatmaps...")
        _plot_loss_heatmaps(source, output_dir, x_data, t_data, h_magnitude)

    print("    [by_mean_loss] Generating loss distributions...")
    _plot_loss_distributions(source, output_dir)


# =============================================================================
# accept_split_by_norm analysis
# =============================================================================

def _plot_norm_per_epoch_accept(diags, output_dir, x_data, t_data, h_magnitude):
    """Per spawn-epoch: norm histogram (accepted vs rejected) + spatial visualization."""
    is_2d = x_data is not None
    for idx, diag in enumerate(diags):
        epoch = diag['epoch']
        threshold = diag.get('wavelet_threshold', 0)
        evaluated = diag.get('evaluated_leaves', [])

        all_children_acc, all_children_rej = [], []
        for leaf in evaluated:
            children = leaf.get('children', [])
            if leaf.get('accepted', False):
                all_children_acc.extend(children)
            else:
                all_children_rej.extend(children)

        norms_acc = [c['wavelet_norm_squared'] for c in all_children_acc]
        norms_rej = [c['wavelet_norm_squared'] for c in all_children_rej]
        all_norms = norms_acc + norms_rej

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Epoch {epoch} — accept_split_by_norm (threshold={threshold})',
                     fontsize=14, fontweight='bold')

        # Left: histogram
        ax = axes[0]
        bins = np.histogram_bin_edges(all_norms, bins=20) if all_norms else 10
        if norms_acc:
            ax.hist(norms_acc, bins=bins, alpha=0.8, color='#2ecc71', edgecolor='black',
                    lw=0.8, label=f'Accepted ({len(norms_acc)})')
        if norms_rej:
            ax.hist(norms_rej, bins=bins, alpha=0.5, color='#e74c3c', edgecolor='black',
                    lw=0.8, hatch='///', label=f'Rejected ({len(norms_rej)})')
        ax.axvline(threshold, color='blue', ls='--', lw=2, label=f'Threshold={threshold}')
        ax.set_xlabel('Wavelet Norm'); ax.set_ylabel('Count')
        ax.set_title('Children Norm Distribution')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

        # Right: spatial
        ax = axes[1]
        if is_2d:
            _render_gt_bg(ax, x_data, t_data, h_magnitude)
            _draw_rects_binary(ax, all_children_acc, all_children_rej)
            _set_ax_limits(ax, all_children_acc + all_children_rej)
            ax.set_title('Regions (green=accepted, red=rejected)')
            ax.grid(True, alpha=0.3, zorder=0)
        else:
            ax.text(0.5, 0.5, '3D — spatial plot skipped', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='gray')

        plt.tight_layout()
        out = output_dir / f'accept_norm_epoch_{epoch}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      Epoch {epoch} analysis saved: {out.name}")


def _plot_norm_evolution_accept(diags, output_dir):
    """Scatter plot: norms of accepted/rejected children across spawn epochs."""
    epochs_acc, norms_acc = [], []
    epochs_rej, norms_rej = [], []

    for diag in diags:
        ep = diag['epoch']
        for leaf in diag.get('evaluated_leaves', []):
            for child in leaf.get('children', []):
                if leaf.get('accepted', False):
                    epochs_acc.append(ep)
                    norms_acc.append(child['wavelet_norm_squared'])
                else:
                    epochs_rej.append(ep)
                    norms_rej.append(child['wavelet_norm_squared'])

    if not epochs_acc and not epochs_rej:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    if epochs_rej:
        ax.scatter(epochs_rej, norms_rej, c='#e74c3c', alpha=0.5, s=25, label='Rejected', marker='x')
    if epochs_acc:
        ax.scatter(epochs_acc, norms_acc, c='#2ecc71', alpha=0.7, s=35, label='Accepted', marker='o')

    threshold = diags[0].get('wavelet_threshold', 0)
    ax.axhline(threshold, color='blue', ls='--', lw=1.5, label=f'Threshold={threshold}')

    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Wavelet Norm', fontsize=12)
    ax.set_title('Norm Evolution Across Spawn Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / 'norm_evolution.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Norm evolution saved: {out.name}")


def _analyze_accept_split_by_norm(data, output_dir, metrics_path, x_data, t_data, h_magnitude):
    """Full analysis pipeline for accept_split_by_norm spawning."""
    diags = [d for d in data.get('spawning_diagnostics', [])
             if d.get('method') == 'accept_split_by_norm']
    regions = data.get('regions', [])

    if not diags:
        print("    No accept_split_by_norm diagnostics found")
        return

    print("    [accept_split_by_norm] Per-epoch norm analysis...")
    _plot_norm_per_epoch_accept(diags, output_dir, x_data, t_data, h_magnitude)

    if len(diags) > 1:
        print("    [accept_split_by_norm] Norm evolution...")
        _plot_norm_evolution_accept(diags, output_dir)

    spawned = [r for r in regions if r.get('spawned', True)]
    if spawned and _dim_from_items(spawned) == 2:
        print("    [accept_split_by_norm] Final regions summary...")
        fig, ax = plt.subplots(figsize=(10, 7))
        _render_gt_bg(ax, x_data, t_data, h_magnitude)
        _draw_rects(ax, spawned, 'wavelet_norm_squared', 'RdYlGn', 'Wavelet Norm')
        _set_ax_limits(ax, spawned)
        ax.set_title(f'Final Expert Regions (n={len(spawned)})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, zorder=0)
        plt.tight_layout()
        out = output_dir / 'final_regions_summary.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Final regions summary saved: {out.name}")


# =============================================================================
# full_tree_by_norm analysis
# =============================================================================

def _plot_prune_comparison(all_nodes, accepted, rejected, threshold, output_dir):
    """Side-by-side histograms: all nodes (before) vs accepted only (after)."""
    all_norms = [n['wavelet_norm_squared'] for n in all_nodes]
    acc_norms = [n['wavelet_norm_squared'] for n in accepted]
    rej_norms = [n['wavelet_norm_squared'] for n in rejected]
    bins = np.histogram_bin_edges(all_norms, bins=40) if all_norms else 20

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Norm Distribution: Before vs After Pruning (threshold={threshold})',
                 fontsize=15, fontweight='bold')

    # Before pruning
    ax = axes[0]
    ax.hist(all_norms, bins=bins, alpha=0.8, color='steelblue', edgecolor='black', lw=0.8)
    ax.axvline(threshold, color='red', ls='--', lw=2, label=f'Threshold={threshold}')
    if all_norms:
        ax.axvline(np.median(all_norms), color='orange', ls=':', lw=1.5,
                    label=f'Median={np.median(all_norms):.4f}')
    ax.set_xlabel('Wavelet Norm'); ax.set_ylabel('Count')
    ax.set_title(f'Before Pruning (n={len(all_nodes)})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

    # After pruning
    ax = axes[1]
    if acc_norms:
        ax.hist(acc_norms, bins=bins, alpha=0.8, color='#2ecc71', edgecolor='black', lw=0.8,
                label=f'Accepted ({len(accepted)})')
    if rej_norms:
        ax.hist(rej_norms, bins=bins, alpha=0.4, color='#e74c3c', edgecolor='black', lw=0.6,
                hatch='///', label=f'Rejected ({len(rejected)})')
    ax.axvline(threshold, color='red', ls='--', lw=2, label=f'Threshold={threshold}')
    ax.set_xlabel('Wavelet Norm'); ax.set_ylabel('Count')
    ax.set_title(f'After Pruning ({len(accepted)} accepted)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = output_dir / 'prune_comparison_histograms.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Prune comparison saved: {out.name}")

    # By-depth breakdown
    depths = sorted({n['tree_depth'] for n in all_nodes})
    n_d = len(depths)
    if n_d == 0:
        return
    fig, axes = plt.subplots(2, min(n_d, 6), figsize=(5 * min(n_d, 6), 8), squeeze=False)
    fig.suptitle('Norm Distribution by Depth: Before (top) / After (bottom) Pruning',
                 fontsize=14, fontweight='bold')
    depth_colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_d))

    for di, depth in enumerate(depths[:6]):
        d_all = [n['wavelet_norm_squared'] for n in all_nodes if n['tree_depth'] == depth]
        d_acc = [n['wavelet_norm_squared'] for n in accepted if n['tree_depth'] == depth]
        d_rej = [n['wavelet_norm_squared'] for n in rejected if n['tree_depth'] == depth]
        d_bins = np.histogram_bin_edges(d_all, bins=15) if d_all else 10

        ax = axes[0, di]
        ax.hist(d_all, bins=d_bins, alpha=0.8, color=depth_colors[di], edgecolor='black', lw=0.7)
        ax.axvline(threshold, color='red', ls='--', lw=1.5)
        ax.set_title(f'Depth {depth} (n={len(d_all)})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        if di == 0:
            ax.set_ylabel('Count (before)')

        ax = axes[1, di]
        if d_acc:
            ax.hist(d_acc, bins=d_bins, alpha=0.8, color='#2ecc71', edgecolor='black', lw=0.7,
                    label=f'Acc ({len(d_acc)})')
        if d_rej:
            ax.hist(d_rej, bins=d_bins, alpha=0.4, color='#e74c3c', edgecolor='black', lw=0.5,
                    hatch='///', label=f'Rej ({len(d_rej)})')
        ax.axvline(threshold, color='red', ls='--', lw=1.5)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=7)
        if di == 0:
            ax.set_ylabel('Count (after)')

    n_cols_shown = min(n_d, 6)
    for di in range(n_d, n_cols_shown):
        axes[0, di].set_visible(False)
        axes[1, di].set_visible(False)

    plt.tight_layout()
    out = output_dir / 'prune_comparison_by_depth.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Prune comparison by depth saved: {out.name}")


def _plot_prune_spatial(all_nodes, accepted, output_dir, x_data, t_data, h_magnitude):
    """Before vs after pruning spatial visualization on GT background."""
    is_2d = _dim_from_items(all_nodes) == 2

    if is_2d:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle('Spatial Regions: Before vs After Pruning', fontsize=15, fontweight='bold')

        ax = axes[0]
        _render_gt_bg(ax, x_data, t_data, h_magnitude)
        _draw_rects(ax, all_nodes, 'wavelet_norm_squared', 'hot', 'Wavelet Norm')
        _set_ax_limits(ax, all_nodes)
        ax.set_title(f'Before Pruning (all {len(all_nodes)} nodes)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, zorder=0)

        ax = axes[1]
        _render_gt_bg(ax, x_data, t_data, h_magnitude)
        if accepted:
            _draw_rects(ax, accepted, 'wavelet_norm_squared', 'hot', 'Wavelet Norm')
        _set_ax_limits(ax, all_nodes)
        ax.set_title(f'After Pruning ({len(accepted)} accepted)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, zorder=0)

        plt.tight_layout()
        out = output_dir / 'prune_spatial_comparison.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Prune spatial comparison saved: {out.name}")
    elif _dim_from_items(all_nodes) == 3:
        fig = plt.figure(figsize=(18, 7))
        fig.suptitle('Spatial Regions: Before vs After Pruning (3D)', fontsize=15, fontweight='bold')
        ax1 = fig.add_subplot(121, projection='3d')
        _draw_rects_3d(ax1, all_nodes, 'hot', 'wavelet_norm_squared')
        ax1.set_title(f'Before ({len(all_nodes)} nodes)')
        ax2 = fig.add_subplot(122, projection='3d')
        if accepted:
            _draw_rects_3d(ax2, accepted, 'hot', 'wavelet_norm_squared')
        ax2.set_title(f'After ({len(accepted)} accepted)')
        plt.tight_layout()
        out = output_dir / 'prune_spatial_comparison_3d.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Prune spatial comparison (3D) saved: {out.name}")


def _plot_depth_analysis_full_tree(all_nodes, threshold, output_dir, x_data, t_data, h_magnitude):
    """Per-depth 2x2 figure: histograms + spatial for accepted/rejected."""
    is_2d = _dim_from_items(all_nodes) == 2
    depths = sorted({n['tree_depth'] for n in all_nodes})

    for di, depth in enumerate(depths):
        d_acc = [n for n in all_nodes if n['tree_depth'] == depth and n['accepted']]
        d_rej = [n for n in all_nodes if n['tree_depth'] == depth and not n['accepted']]
        n_acc = [n['wavelet_norm_squared'] for n in d_acc]
        n_rej = [n['wavelet_norm_squared'] for n in d_rej]
        all_d_norms = n_acc + n_rej
        bins = np.histogram_bin_edges(all_d_norms, bins=20) if all_d_norms else 10

        if is_2d:
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        else:
            fig = plt.figure(figsize=(18, 12))
            axes = np.empty((2, 2), dtype=object)
            axes[0, 0] = fig.add_subplot(2, 2, 1)
            axes[0, 1] = fig.add_subplot(2, 2, 2)
            axes[1, 0] = fig.add_subplot(2, 2, 3, projection='3d')
            axes[1, 1] = fig.add_subplot(2, 2, 4, projection='3d')
        fig.suptitle(f'Depth {depth} Analysis (threshold={threshold})',
                     fontsize=15, fontweight='bold')

        # Top-left: accepted histogram
        ax = axes[0, 0]
        if n_acc:
            ax.hist(n_acc, bins=bins, alpha=0.8, color='#2ecc71', edgecolor='black', lw=0.8)
            ax.axvline(np.median(n_acc), color='orange', ls=':', lw=1.5,
                        label=f'Median={np.median(n_acc):.4f}')
        ax.axvline(threshold, color='red', ls='--', lw=1.5, label=f'Threshold={threshold}')
        ax.set_title(f'Accepted (n={len(d_acc)})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Wavelet Norm'); ax.set_ylabel('Count')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

        # Top-right: rejected histogram
        ax = axes[0, 1]
        if n_rej:
            ax.hist(n_rej, bins=bins, alpha=0.7, color='#e74c3c', edgecolor='black', lw=0.8)
            ax.axvline(np.median(n_rej), color='orange', ls=':', lw=1.5,
                        label=f'Median={np.median(n_rej):.4f}')
        ax.axvline(threshold, color='red', ls='--', lw=1.5, label=f'Threshold={threshold}')
        ax.set_title(f'Rejected (n={len(d_rej)})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Wavelet Norm'); ax.set_ylabel('Count')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

        # Bottom: spatial
        if is_2d:
            ax = axes[1, 0]
            _render_gt_bg(ax, x_data, t_data, h_magnitude)
            if d_acc:
                _draw_rects(ax, d_acc, 'wavelet_norm_squared', 'hot', 'Wavelet Norm', lw=1.5)
            _set_ax_limits(ax, d_acc + d_rej)
            ax.set_title('Accepted Regions')
            ax.grid(True, alpha=0.3, zorder=0)

            ax = axes[1, 1]
            _render_gt_bg(ax, x_data, t_data, h_magnitude)
            if d_rej:
                _draw_rects(ax, d_rej, 'wavelet_norm_squared',
                            'hot', 'Wavelet Norm',
                            lw=1.0, alpha_face=0.2)
            _set_ax_limits(ax, d_acc + d_rej)
            ax.set_title('Rejected Regions')
            ax.grid(True, alpha=0.3, zorder=0)
        else:
            ax = axes[1, 0]
            if d_acc:
                _draw_rects_3d(ax, d_acc, 'hot', 'wavelet_norm_squared')
            ax.set_title('Accepted')
            ax = axes[1, 1]
            if d_rej:
                _draw_rects_3d(ax, d_rej, 'hot', 'wavelet_norm_squared')
            ax.set_title('Rejected')

        plt.tight_layout()
        out = output_dir / f'depth_{depth}_analysis.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      Depth {depth}: {len(d_acc)} accepted, {len(d_rej)} rejected -> {out.name}")


def _plot_acceptance_rate(all_nodes, output_dir):
    """Bar chart showing accept/reject counts and acceptance rate per depth."""
    depths = sorted({n['tree_depth'] for n in all_nodes})
    if not depths:
        return

    acc_counts, rej_counts, rates = [], [], []
    for d in depths:
        a = sum(1 for n in all_nodes if n['tree_depth'] == d and n['accepted'])
        r = sum(1 for n in all_nodes if n['tree_depth'] == d and not n['accepted'])
        acc_counts.append(a)
        rej_counts.append(r)
        rates.append(a / (a + r) * 100 if (a + r) > 0 else 0)

    x = np.arange(len(depths))
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(max(10, len(depths) * 1.5), 6))
    bars_a = ax1.bar(x - w/2, acc_counts, w, color='#2ecc71', edgecolor='black', lw=0.8, label='Accepted')
    bars_r = ax1.bar(x + w/2, rej_counts, w, color='#e74c3c', edgecolor='black', lw=0.8, label='Rejected')

    for i, (ba, br) in enumerate(zip(bars_a, bars_r)):
        ax1.text(ba.get_x() + ba.get_width()/2, ba.get_height(), str(acc_counts[i]),
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(br.get_x() + br.get_width()/2, br.get_height(), str(rej_counts[i]),
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xlabel('Tree Depth', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(d) for d in depths])
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = ax1.twinx()
    ax2.plot(x, rates, 'ko-', lw=2, markersize=8, label='Accept %')
    for i, rate in enumerate(rates):
        ax2.annotate(f'{rate:.0f}%', (x[i], rate), textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
    ax2.set_ylabel('Acceptance Rate (%)', fontsize=12)
    ax2.set_ylim(0, 105)
    ax2.legend(loc='upper right', fontsize=10)

    fig.suptitle('Acceptance Rate by Tree Depth', fontsize=15, fontweight='bold')
    plt.tight_layout()
    out = output_dir / 'acceptance_rate_by_depth.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Acceptance rate by depth saved: {out.name}")


def _plot_threshold_cdf(all_nodes, threshold, output_dir):
    """CDF of wavelet norms with threshold line — shows sensitivity to threshold choice."""
    norms = sorted([n['wavelet_norm_squared'] for n in all_nodes])
    if not norms:
        return
    cdf = np.arange(1, len(norms) + 1) / len(norms)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(norms, cdf, 'b-', lw=2)
    ax.axvline(threshold, color='red', ls='--', lw=2, label=f'Threshold={threshold}')

    n_above = sum(1 for n in norms if n >= threshold)
    frac = n_above / len(norms)
    ax.axhline(1 - frac, color='red', ls=':', lw=1, alpha=0.5)
    ax.plot(threshold, 1 - frac, 'ro', markersize=10, zorder=5)
    ax.annotate(f'{n_above}/{len(norms)} above\n({frac*100:.1f}%)',
                (threshold, 1 - frac), textcoords='offset points', xytext=(15, -15),
                fontsize=10, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red'))

    ax.fill_betweenx(cdf, norms, alpha=0.1, color='blue')
    ax.set_xlabel('Wavelet Norm', fontsize=12)
    ax.set_ylabel('CDF (fraction of nodes ≤ norm)', fontsize=12)
    ax.set_title('Threshold Sensitivity: Wavelet Norm CDF', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / 'threshold_sensitivity_cdf.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Threshold sensitivity CDF saved: {out.name}")


def _plot_tree_hierarchy(all_nodes, threshold, output_dir):
    """Dendrogram-style tree diagram with nodes colored by norm, accepted=solid, rejected=faded."""
    if not all_nodes:
        return

    children_map = {}
    node_map = {}
    for n in all_nodes:
        nid = n['node_id']
        pid = n.get('parent_node_id', -1)
        node_map[nid] = n
        children_map.setdefault(pid, []).append(n)

    root_children = children_map.get(0, [])
    if not root_children:
        root_children = [n for n in all_nodes if n.get('parent_node_id', -1) == -1]
    if not root_children:
        return

    leaf_counter = [0]
    positions = {}

    def _layout(nid):
        kids = children_map.get(nid, [])
        depth = node_map[nid]['tree_depth']
        if not kids:
            x = leaf_counter[0]
            leaf_counter[0] += 1
            positions[nid] = (x, depth)
            return x
        child_xs = [_layout(c['node_id']) for c in sorted(kids, key=lambda c: c['node_id'])]
        x = np.mean(child_xs)
        positions[nid] = (x, depth)
        return x

    rc_xs = [_layout(c['node_id']) for c in sorted(root_children, key=lambda c: c['node_id'])]
    root_x = np.mean(rc_xs)
    positions[0] = (root_x, 0)

    max_depth = max(n['tree_depth'] for n in all_nodes) if all_nodes else 1
    n_leaves = leaf_counter[0]
    fig_w = max(14, n_leaves * 0.25)
    fig_h = max(8, max_depth * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    norms = [n['wavelet_norm_squared'] for n in all_nodes]
    vmin, vmax = min(norms), max(norms)
    vrange = vmax - vmin if vmax > vmin else 1.0
    cmap = plt.get_cmap('hot')

    # Draw edges
    for n in all_nodes:
        nid = n['node_id']
        pid = n.get('parent_node_id', -1)
        if pid >= 0 and pid in positions and nid in positions:
            px, py = positions[pid]
            cx, cy = positions[nid]
            edge_alpha = 0.7 if n['accepted'] else 0.15
            ax.plot([px, cx], [-py, -cy], 'k-', alpha=edge_alpha, lw=0.8, zorder=1)

    # Draw virtual root
    ax.scatter(root_x, 0, c='white', s=80, edgecolors='black', linewidths=2, zorder=6, marker='s')

    # Draw nodes
    for n in all_nodes:
        nid = n['node_id']
        if nid not in positions:
            continue
        x, y = positions[nid]
        nv = (n['wavelet_norm_squared'] - vmin) / vrange if vrange > 0 else 0.5
        c = cmap(nv)
        if n['accepted']:
            ax.scatter(x, -y, c=[c], s=45, edgecolors='black', linewidths=1.0, zorder=5)
        else:
            ax.scatter(x, -y, c=[c], s=20, edgecolors='gray', linewidths=0.4,
                       alpha=0.25, zorder=4)

    sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Wavelet Norm', shrink=0.7)

    ax.axhline(-0.5, color='red', ls=':', lw=0.5, alpha=0.3)
    ax.set_ylabel('Depth', fontsize=12)
    ax.set_yticks([-d for d in range(max_depth + 1)])
    ax.set_yticklabels([str(d) for d in range(max_depth + 1)])
    ax.set_xticks([])
    n_acc = sum(1 for n in all_nodes if n['accepted'])
    ax.set_title(f'Tree Hierarchy (threshold={threshold})\n'
                 f'{n_acc} accepted (solid) / {len(all_nodes) - n_acc} rejected (faded) '
                 f'out of {len(all_nodes)} nodes',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.15, axis='y')
    plt.tight_layout()
    out = output_dir / 'tree_hierarchy.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Tree hierarchy saved: {out.name}")


def _print_full_tree_stats(all_nodes, accepted, rejected, threshold):
    """Print summary statistics for full_tree_by_norm."""
    print(f"\n{'='*60}")
    print(f"Full Tree Statistics (threshold={threshold})")
    print(f"{'='*60}")
    print(f"Total nodes: {len(all_nodes)}, Accepted: {len(accepted)}, Rejected: {len(rejected)}")

    for label, subset in [('Accepted', accepted), ('Rejected', rejected)]:
        if subset:
            norms = [n['wavelet_norm_squared'] for n in subset]
            print(f"\n  {label} (n={len(subset)}):")
            print(f"    Norm: mean={np.mean(norms):.4f}, median={np.median(norms):.4f}, "
                  f"std={np.std(norms):.4f}")
            print(f"    Range: [{np.min(norms):.4f}, {np.max(norms):.4f}]")

    depths = sorted({n['tree_depth'] for n in all_nodes})
    print(f"\n  By Depth:")
    for d in depths:
        a = sum(1 for n in all_nodes if n['tree_depth'] == d and n['accepted'])
        r = sum(1 for n in all_nodes if n['tree_depth'] == d and not n['accepted'])
        total = a + r
        pct = a / total * 100 if total > 0 else 0
        print(f"    Depth {d:2d}: {a:4d} accepted, {r:4d} rejected ({pct:.0f}% accept rate)")
    print(f"{'='*60}\n")


def _analyze_full_tree_by_norm(data, output_dir, metrics_path, x_data, t_data, h_magnitude):
    """Full analysis pipeline for full_tree_by_norm spawning."""
    diags = [d for d in data.get('spawning_diagnostics', [])
             if d.get('method') == 'full_tree_by_norm']
    if not diags:
        print("    No full_tree_by_norm diagnostics found")
        return

    diag = diags[0]
    nodes = diag.get('nodes', [])
    threshold = diag.get('wavelet_threshold', 0)
    accepted = [n for n in nodes if n.get('accepted', False)]
    rejected = [n for n in nodes if not n.get('accepted', False)]

    if not nodes:
        print("    No tree nodes in diagnostics")
        return

    print("    [full_tree] Prune comparison histograms...")
    _plot_prune_comparison(nodes, accepted, rejected, threshold, output_dir)

    print("    [full_tree] Spatial before/after pruning...")
    _plot_prune_spatial(nodes, accepted, output_dir, x_data, t_data, h_magnitude)

    print("    [full_tree] Per-depth analysis...")
    _plot_depth_analysis_full_tree(nodes, threshold, output_dir, x_data, t_data, h_magnitude)

    print("    [full_tree] Acceptance rate by depth...")
    _plot_acceptance_rate(nodes, output_dir)

    print("    [full_tree] Threshold sensitivity CDF...")
    _plot_threshold_cdf(nodes, threshold, output_dir)

    print("    [full_tree] Tree hierarchy diagram...")
    _plot_tree_hierarchy(nodes, threshold, output_dir)

    _print_full_tree_stats(nodes, accepted, rejected, threshold)


# =============================================================================
# Legacy fallback (for old data without spawning_diagnostics)
# =============================================================================

def _analyze_legacy(data, output_dir, metrics_path, x_data, t_data, h_magnitude):
    """Fallback analysis for old-format data without spawning_diagnostics."""
    regions = data.get('regions', [])
    leaf_loss_history = data.get('leaf_loss_history', None)

    if leaf_loss_history:
        print("    [legacy] Generating leaf loss plots...")
        _plot_leaf_loss_bars(leaf_loss_history, output_dir)

    all_norms = [r.get('wavelet_norm_squared', 0) for r in regions]
    if not any(n > 0 for n in all_norms):
        print("    [legacy] No non-zero wavelet norms, skipping norm plots")
        return

    spawned = [r for r in regions if r.get('spawned', True)]
    rejected = [r for r in regions if not r.get('spawned', True)]

    if _dim_from_items(regions) == 2 and spawned:
        fig, ax = plt.subplots(figsize=(10, 7))
        _render_gt_bg(ax, x_data, t_data, h_magnitude)
        _draw_rects(ax, spawned, 'wavelet_norm_squared', 'RdYlGn', 'Wavelet Norm')
        _set_ax_limits(ax, spawned + rejected)
        ax.set_title(f'Expert Regions (n={len(spawned)})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, zorder=0)
        plt.tight_layout()
        out = output_dir / 'expert_norm_distributions.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    [legacy] Expert regions saved: {out.name}")


# =============================================================================
# Main dispatch and batch processing
# =============================================================================

def plot_expert_norms_for_model(json_path, output_dir, metrics_path=None):
    """Generate analysis plots for a single model run.

    Detects the spawning method and dispatches to the appropriate pipeline.
    """
    with open(json_path) as f:
        data = json.load(f)

    regions = data.get('regions', [])
    if not regions and not data.get('leaf_loss_history') and not data.get('spawning_diagnostics'):
        print("    No data found, skipping")
        return None

    x_data, t_data, h_magnitude, _ = load_ground_truth(output_dir)

    method = data.get('spawning_method')
    if not method:
        diags = data.get('spawning_diagnostics', [])
        if diags:
            method = diags[0].get('method')

    if not method and metrics_path and Path(metrics_path).exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        m_diags = metrics.get('spawning_diagnostics', [])
        if m_diags:
            method = m_diags[0].get('method')
            if 'spawning_diagnostics' not in data:
                data['spawning_diagnostics'] = m_diags

    print(f"    Detected spawning method: {method or 'unknown (legacy)'}")

    if method == 'by_mean_loss':
        _analyze_by_mean_loss(data, output_dir, metrics_path, x_data, t_data, h_magnitude)
    elif method == 'accept_split_by_norm':
        _analyze_accept_split_by_norm(data, output_dir, metrics_path, x_data, t_data, h_magnitude)
    elif method == 'full_tree_by_norm':
        _analyze_full_tree_by_norm(data, output_dir, metrics_path, x_data, t_data, h_magnitude)
    else:
        _analyze_legacy(data, output_dir, metrics_path, x_data, t_data, h_magnitude)

    return output_dir


def _find_run_dirs(batch_path):
    """Find all (label, timestamp_dir) pairs in a batch directory.

    Handles three layouts:
      1. Flat / multi-PDE: batch/YYYYMMDD_HHMMSS/ dirs with metrics.json
      2. Multiple architectures: batch/arch_a/timestamp/, batch/arch_b/timestamp/
      3. Single architecture with many runs: batch/arch/ts1/, batch/arch/ts2/
    """
    import re
    import yaml
    _TS_RE = re.compile(r'\d{8}_\d{6}$')

    child_dirs = sorted(
        d for d in batch_path.iterdir()
        if d.is_dir() and d.name != 'checkpoints'
    )
    if not child_dirs:
        return []

    flat_ts = [d for d in child_dirs
               if _TS_RE.match(d.name) and (d / 'metrics.json').exists()]
    if flat_ts:
        runs = []
        for ts_dir in flat_ts:
            cfg_file = ts_dir / 'config_used.yaml'
            if cfg_file.exists():
                try:
                    with open(cfg_file) as f:
                        cfg = yaml.safe_load(f)
                    label = cfg.get('model', ts_dir.name)
                except Exception:
                    label = ts_dir.name
            else:
                label = ts_dir.name
            runs.append((label, ts_dir))
        return runs

    runs = []
    for model_dir in child_dirs:
        ts_dirs = sorted(
            d for d in model_dir.iterdir()
            if d.is_dir() and d.name != 'checkpoints'
        )
        if not ts_dirs:
            continue
        if len(child_dirs) == 1 and len(ts_dirs) > 1:
            for ts_dir in ts_dirs:
                runs.append((ts_dir.name, ts_dir))
        else:
            runs.append((model_dir.name, ts_dirs[-1]))

    return runs


def process_batch(batch_dir):
    """Process all models in a batch directory."""
    batch_path = Path(batch_dir)
    if not batch_path.exists():
        print(f"Error: Directory not found: {batch_path}")
        return

    print(f"\n{'='*70}")
    print(f"Processing batch: {batch_path.name}")
    print(f"{'='*70}")

    runs = _find_run_dirs(batch_path)
    if not runs:
        print(f"  No model runs found in {batch_path}")
        return

    print(f"  Found {len(runs)} run(s)")

    for label, ts_dir in runs:
        print(f"\n  Processing: {label}")

        json_path = ts_dir / "adaptive_plots" / "expert_regions.json"
        if not json_path.exists():
            print("    No expert_regions.json found, skipping")
            continue

        metrics_path = ts_dir / "metrics.json"
        if not metrics_path.exists():
            metrics_path = None

        output_dir = json_path.parent
        plot_expert_norms_for_model(json_path, output_dir, metrics_path)

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        batch_dir = sys.argv[1]
    else:
        batch_dir = "outputs/experiments/AToE-New/schrodinger_tests_20260210_055733-non-pretrained-10k-epochs"
    process_batch(batch_dir)
