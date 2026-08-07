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

import sys
import json
import argparse
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.plotting import plot_training_curves


def load_run(run_dir: Path):
    metrics = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
    cfg = yaml.safe_load((run_dir / 'config_used.yaml').read_text(encoding='utf-8'))
    return metrics, cfg


def concat_runs(run_dirs):
    """Stitch metrics.json from several runs into one continuous timeline.

    Returns (combined_metrics, segment_markers, optimizer_switch_epochs, meta)
    ready to hand to trainer.plotting.plot_training_curves.
    """
    runs_data = []
    offset = 0
    pde = None
    for run_dir in run_dirs:
        metrics, cfg = load_run(run_dir)
        if pde is None:
            pde = cfg['problem']
        elif cfg['problem'] != pde:
            raise ValueError(
                f"PDE mismatch: {run_dir} is '{cfg['problem']}', expected '{pde}'")
        local_final = max(metrics['train_loss_epochs']) if metrics['train_loss_epochs'] else 0
        runs_data.append((run_dir, metrics, offset, local_final))
        offset += local_final

    all_terms = set()
    for _, metrics, _, _ in runs_data:
        lc = metrics.get('loss_components', {})
        all_terms.update(k for k in lc if k != 'epochs')

    train_loss_epochs, train_loss = [], []
    eval_epochs, rel_l2, inf_norm = [], [], []
    lc_epochs = []
    lc_terms = {t: [] for t in all_terms}
    segment_markers = []
    optimizer_switch_epochs = []
    stage_tags = []

    for run_dir, metrics, off, local_final in runs_data:
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

        for ev in metrics.get('segment_events', []):
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
    ap.add_argument('run_dirs', nargs='+', type=Path,
                     help='Run dirs in chronological order')
    ap.add_argument('--out-dir', type=Path,
                     default=Path('outputs/paper_figures/training_curves'),
                     help='Output directory for the figures')
    args = ap.parse_args()

    combined, segment_markers, optimizer_switch_epochs, meta = concat_runs(args.run_dirs)

    stage_str = '_'.join(meta['stage_tags'])
    relL2 = meta['final_rel_l2']
    name_suffix = (f"{meta['pde']}_{stage_str}_ep{meta['total_epochs']}"
                   f"_E{meta['num_experts']}"
                   + (f"_relL2_{relL2:.2e}" if relL2 is not None else ''))

    print(f"PDE: {meta['pde']}")
    print(f"Concatenated {len(args.run_dirs)} run(s), total epochs = {meta['total_epochs']}")
    for epoch, name in segment_markers:
        print(f"  segment '{name}' starts at epoch {epoch}")
    relL2_str = f"{relL2:.3e}" if relL2 is not None else "n/a"
    print(f"Final experts: {meta['num_experts']}, final rel-L2 (dense grid): {relL2_str}")

    plot_training_curves(
        combined,
        save_dir=args.out_dir,
        optimizer_switch_epochs=optimizer_switch_epochs,
        segment_markers=segment_markers,
        name_suffix=name_suffix,
    )


if __name__ == '__main__':
    main()
