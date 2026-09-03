"""Standalone: concatenate training curves across run dirs into one figure.

Takes several run dirs for the *same PDE*, given in chronological order (e.g.
a root+phase3 "experts creation" run, followed by a fine-tune run loaded from
its checkpoint) and stitches their metrics.json into one continuous
timeline: epochs of run N are shifted by the final epoch of run N-1. The
result is rendered with trainer.plotting.plot_training_curves, so it looks
exactly like a normal training-curve figure (clean, no titles/suptitles —
run stats live in the filename) except the x-axis now spans every stage,
with segment-boundary markers (root / phase3 / fine_tune) at each stitch
point.

Usage:
    python scripts/plot_concat_training_curves.py <run_dir1> <run_dir2> [...] [--out-dir DIR]

Each run_dir is a leaf run directory (contains metrics.json,
config_used.yaml), e.g.:
    outputs/experiments/<experiment>/allen_cahn-.../20260724_142510
"""

import os
import sys
import json
import argparse
from pathlib import Path

import yaml
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.plotting import _safe_log_scale, _draw_segment_markers
from utils.plot_io import save_png

TICK_SIZE = 14
LABEL_SIZE = 17
LEGEND_SIZE = 13


def _style_axes(ax):
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.xaxis.label.set_fontsize(LABEL_SIZE)
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontsize(LABEL_SIZE)
    ax.yaxis.label.set_fontweight('bold')


def plot_training_curves_paper(metrics, save_dir, optimizer_switch_epochs=None,
                               segment_markers=None, name_suffix=''):
    """Paper-styled reimplementation of trainer.plotting.plot_training_curves:
    same panels/data/markers (reuses its _safe_log_scale, _draw_segment_markers
    helpers), just bigger/bold tick, axis-label and legend text. Kept local to
    this script rather than editing the shared function, since that one is
    also used by the live training pipeline.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loss_epochs = metrics['train_loss_epochs']
    eval_epochs = metrics['epochs']
    optimizer_switch_epochs = optimizer_switch_epochs or []
    segment_markers = segment_markers or []

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
        ax.set_xlabel('Epoch')
        is_log = _safe_log_scale(ax, log_series) if log_series else False
        ax.set_ylabel(f'{ylabel}{" (log)" if is_log else ""}')
        ax.legend(fontsize=LEGEND_SIZE)
        ax.grid(True, alpha=0.3)
        _style_axes(ax)

    def _panel_loss(ax):
        ax.plot(train_loss_epochs, metrics['train_loss'], 'b-',
                label='Train loss', linewidth=2, alpha=0.8)
        _draw_markers(ax)
        _finish(ax, 'Loss', [metrics['train_loss']])

    def _panel_rel_l2(ax):
        _rl2 = metrics.get('rel_l2', metrics.get('eval_rel_l2', []))
        ax.plot(eval_epochs, _rl2, 'r-',
                label='Rel. $L^2$ error', linewidth=2, alpha=0.8)
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
        _finish(ax, 'Relative $L^2$ error', [_rl2])

    def _panel_components(ax):
        comp_epochs = loss_comps['epochs']
        term_colors = {
            'residual': '#e74c3c', 'ic': '#3498db', 'bc': '#2ecc71',
            'bc_dx': '#27ae60', 'bc_dxx': '#1e8449', 'bc_dxxx': '#145a32',
            'continuity': '#e67e22', 'l2sp': '#9b59b6',
        }
        term_labels = {
            'residual': 'PDE residual', 'ic': 'Initial condition',
            'bc': 'Boundary condition', 'bc_dx': 'BC ∂ₓ periodicity',
            'bc_dxx': 'BC ∂ₓₓ periodicity',
            'bc_dxxx': 'BC ∂ₓₓₓ periodicity',
            'continuity': 'Continuity', 'l2sp': 'L2-SP anchor',
        }
        values_for_log = []
        for term in ['residual', 'ic', 'bc', 'bc_dx', 'bc_dxx', 'bc_dxxx',
                     'continuity', 'l2sp']:
            if loss_comps.get(term) and len(loss_comps[term]) > 0:
                values = loss_comps[term]
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

    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 5.5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (_, draw) in zip(axes, panels):
        draw(ax)
    plt.tight_layout()
    save_path = save_png(save_dir / f'training_curves{suffix}.png', fig=fig)
    plt.close(fig)

    for key, draw in panels:
        fig_s, ax_s = plt.subplots(figsize=(7.5, 5.5))
        draw(ax_s)
        plt.tight_layout()
        save_png(save_dir / f'training_curves_{key}{suffix}.png', fig=fig_s)
        plt.close(fig_s)

    print(f"  Training curves saved to {save_path} (+ per-panel files)")


def _winlong(path: Path) -> str:
    """Windows extended-length path (bypasses the 260-char MAX_PATH limit)."""
    s = str(Path(path).resolve())
    if os.name == 'nt' and not s.startswith('\\\\?\\'):
        return '\\\\?\\' + s
    return s


def load_run(run_dir: Path):
    with open(_winlong(run_dir / 'metrics.json'), encoding='utf-8') as f:
        metrics = json.load(f)
    with open(_winlong(run_dir / 'config_used.yaml'), encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return metrics, cfg


def parse_run_arg(arg: str):
    """'<run_dir>' or '<run_dir>@<segment>' or '<run_dir>@<=N' or
    '<run_dir>@continue' -- @segment truncates that run to just one of its
    named segments (e.g. a run's metrics.json may hold both 'root' and
    'phase3'; @root keeps only the root portion, dropping the rest, so it
    can be spliced against a DIFFERENT run's phase3/fine_tune). @<=N
    truncates to epochs <= N instead, for splicing mid-segment -- e.g.
    dropping a trailing optimizer excursion that diverged from the
    checkpoint actually used downstream. @continue marks this run as a
    continuation of the PREVIOUS run's segment (e.g. an LBFGS resume
    seeded from a mid-segment checkpoint of the run before it): its own
    segment-start marker is suppressed and drawn as an optimizer-switch
    line instead, so the two runs read as one unbroken segment."""
    if '@' in arg:
        path_str, spec = arg.rsplit('@', 1)
        if spec.startswith('<=') and spec[2:].isdigit():
            return Path(path_str), ('max_epoch', int(spec[2:]))
        if spec == 'continue':
            return Path(path_str), ('continue',)
        return Path(path_str), spec
    return Path(arg), None


def truncate_to_segment(metrics, segment):
    """Keep only one named segment's epoch range from a multi-segment run's
    metrics (train_loss, eval rel_l2/inf_norm, loss_components, events)."""
    seg_event = next((e for e in metrics.get('segment_events', []) if e['segment'] == segment), None)
    rec_event = next((e for e in metrics.get('segment_reconcile_events', []) if e['segment'] == segment), None)
    if seg_event is None or rec_event is None:
        available = [e['segment'] for e in metrics.get('segment_events', [])]
        raise ValueError(f"No segment '{segment}' in this run (available: {available})")
    lo, hi = seg_event['start_epoch'], rec_event['end_epoch']

    def _filter(epochs, *value_lists):
        idx = [i for i, e in enumerate(epochs) if lo <= e <= hi]
        return ([epochs[i] for i in idx],
                *([vl[i] for i in idx] for vl in value_lists))

    out = dict(metrics)
    out['train_loss_epochs'], out['train_loss'] = _filter(
        metrics['train_loss_epochs'], metrics['train_loss'])

    inf_norm = metrics.get('inf_norm') or []
    if inf_norm and len(inf_norm) == len(metrics['epochs']):
        out['epochs'], out['rel_l2'], out['inf_norm'] = _filter(
            metrics['epochs'], metrics['rel_l2'], inf_norm)
    else:
        out['epochs'], out['rel_l2'] = _filter(metrics['epochs'], metrics['rel_l2'])
        out['inf_norm'] = []

    lc = metrics.get('loss_components', {})
    if lc.get('epochs'):
        term_keys = [k for k in lc if k != 'epochs']
        lc_ep, *term_vals = _filter(lc['epochs'], *[lc[k] for k in term_keys])
        out['loss_components'] = {'epochs': lc_ep, **dict(zip(term_keys, term_vals))}

    out['segment_events'] = [e for e in metrics.get('segment_events', []) if e['segment'] == segment]
    out['segment_reconcile_events'] = [
        e for e in metrics.get('segment_reconcile_events', []) if e['segment'] == segment]
    out['optimizer_events'] = [
        e for e in metrics.get('optimizer_events', []) if lo <= e['epoch'] <= hi]
    return out


def truncate_to_max_epoch(metrics, max_epoch):
    """Keep only epochs <= max_epoch from a run's metrics (splicing mid-
    segment, e.g. dropping a trailing optimizer excursion past the epoch
    whose checkpoint was actually used to seed a downstream run)."""
    lo, hi = 1, max_epoch

    def _filter(epochs, *value_lists):
        idx = [i for i, e in enumerate(epochs) if lo <= e <= hi]
        return ([epochs[i] for i in idx],
                *([vl[i] for i in idx] for vl in value_lists))

    out = dict(metrics)
    out['train_loss_epochs'], out['train_loss'] = _filter(
        metrics['train_loss_epochs'], metrics['train_loss'])

    inf_norm = metrics.get('inf_norm') or []
    if inf_norm and len(inf_norm) == len(metrics['epochs']):
        out['epochs'], out['rel_l2'], out['inf_norm'] = _filter(
            metrics['epochs'], metrics['rel_l2'], inf_norm)
    else:
        out['epochs'], out['rel_l2'] = _filter(metrics['epochs'], metrics['rel_l2'])
        out['inf_norm'] = []

    lc = metrics.get('loss_components', {})
    if lc.get('epochs'):
        term_keys = [k for k in lc if k != 'epochs']
        lc_ep, *term_vals = _filter(lc['epochs'], *[lc[k] for k in term_keys])
        out['loss_components'] = {'epochs': lc_ep, **dict(zip(term_keys, term_vals))}

    out['segment_events'] = [e for e in metrics.get('segment_events', []) if e['start_epoch'] <= hi]
    out['segment_reconcile_events'] = [
        e for e in metrics.get('segment_reconcile_events', []) if e.get('end_epoch', hi) <= hi]
    out['optimizer_events'] = [
        e for e in metrics.get('optimizer_events', []) if lo <= e['epoch'] <= hi]
    return out


def concat_runs(run_specs):
    """Stitch metrics.json from several runs into one continuous timeline.

    run_specs: list of (run_dir, segment_or_None) pairs, as from parse_run_arg.

    Returns (combined_metrics, segment_markers, optimizer_switch_epochs, meta)
    ready to hand to trainer.plotting.plot_training_curves.
    """
    runs_data = []
    offset = 0
    pde = None
    for run_dir, segment in run_specs:
        metrics, cfg = load_run(run_dir)
        is_continuation = False
        if segment is not None:
            if isinstance(segment, tuple) and segment[0] == 'max_epoch':
                metrics = truncate_to_max_epoch(metrics, segment[1])
            elif isinstance(segment, tuple) and segment[0] == 'continue':
                is_continuation = True
            else:
                metrics = truncate_to_segment(metrics, segment)
        if pde is None:
            pde = cfg['problem']
        elif cfg['problem'] != pde:
            raise ValueError(
                f"PDE mismatch: {run_dir} is '{cfg['problem']}', expected '{pde}'")
        local_final = max(metrics['train_loss_epochs']) if metrics['train_loss_epochs'] else 0
        runs_data.append((run_dir, metrics, offset, local_final, is_continuation))
        offset += local_final

    all_terms = set()
    for _, metrics, _, _, _ in runs_data:
        lc = metrics.get('loss_components', {})
        all_terms.update(k for k in lc if k != 'epochs')

    train_loss_epochs, train_loss = [], []
    eval_epochs, rel_l2, inf_norm = [], [], []
    lc_epochs = []
    lc_terms = {t: [] for t in all_terms}
    segment_markers = []
    optimizer_switch_epochs = []
    stage_tags = []

    for run_dir, metrics, off, local_final, is_continuation in runs_data:
        train_loss_epochs += [e + off for e in metrics['train_loss_epochs']]
        train_loss += metrics['train_loss']
        eval_epochs += [e + off for e in metrics['epochs']]
        rel_l2 += metrics['rel_l2']
        inf_norm += metrics.get('inf_norm', [])

        lc = metrics.get('loss_components', {})
        lc_ep = lc.get('epochs', [])
        lc_epochs += [e + off for e in lc_ep]
        for term in all_terms:
            vals = lc.get(term)
            if vals is None or len(vals) != len(lc_ep):
                vals = [float('nan')] * len(lc_ep)
            lc_terms[term] += vals

        segs = metrics.get('segment_events', [])
        if is_continuation and segs:
            optimizer_switch_epochs.append(segs[0]['start_epoch'] + off)
            for ev in segs[1:]:
                segment_markers.append((ev['start_epoch'] + off, ev['segment']))
        else:
            for ev in segs:
                segment_markers.append((ev['start_epoch'] + off, ev['segment']))
        for ev in metrics.get('optimizer_events', []):
            optimizer_switch_epochs.append(ev['epoch'] + off)

        stage_name = metrics.get('segment_events', [{}])[-1].get('segment', run_dir.name)
        stage_tags.append(f"{stage_name}{local_final}")

    combined = {
        'train_loss_epochs': train_loss_epochs,
        'train_loss': train_loss,
        'epochs': eval_epochs,
        'rel_l2': rel_l2,
        'inf_norm': inf_norm,
        'loss_components': {'epochs': lc_epochs, **lc_terms},
    }
    last_metrics = runs_data[-1][1]
    meta = {
        'pde': pde,
        'total_epochs': offset,
        'num_experts': last_metrics.get('adaptive_pinn', {}).get('num_experts'),
        'final_rel_l2': last_metrics.get('final_dense_rel_l2', rel_l2[-1] if rel_l2 else None),
        'stage_tags': stage_tags,
    }
    return combined, segment_markers, optimizer_switch_epochs, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dirs', nargs='+', type=str,
                     help="Run dirs in chronological order; '<dir>@<segment>' "
                          "truncates a multi-segment run to just that segment")
    ap.add_argument('--out-dir', type=Path,
                     default=Path('outputs/paper_figures/training_curves'),
                     help='Output directory for the figures')
    args = ap.parse_args()

    run_specs = [parse_run_arg(a) for a in args.run_dirs]
    combined, segment_markers, optimizer_switch_epochs, meta = concat_runs(run_specs)

    stage_str = '_'.join(meta['stage_tags'])
    relL2 = meta['final_rel_l2']
    name_suffix = (f"{meta['pde']}_{stage_str}_ep{meta['total_epochs']}"
                   f"_E{meta['num_experts']}"
                   + (f"_relL2_{relL2:.2e}" if relL2 is not None else ''))

    print(f"PDE: {meta['pde']}")
    print(f"Concatenated {len(run_specs)} run(s), total epochs = {meta['total_epochs']}")
    for epoch, name in segment_markers:
        print(f"  segment '{name}' starts at epoch {epoch}")
    relL2_str = f"{relL2:.3e}" if relL2 is not None else "n/a"
    print(f"Final experts: {meta['num_experts']}, final rel-L2 (dense grid): {relL2_str}")

    plot_training_curves_paper(
        combined,
        save_dir=args.out_dir,
        optimizer_switch_epochs=optimizer_switch_epochs,
        segment_markers=segment_markers,
        name_suffix=name_suffix,
    )


if __name__ == '__main__':
    main()
