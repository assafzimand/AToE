"""Regenerate a time-marching run's combined loss-curve plot with the
native-grid full-domain rel-L2.

Older runs annotated time_marching_combined_loss_curves.png with a full-domain
rel-L2 computed against LINEARLY INTERPOLATED ground truth on a coarse uniform
grid; across steep fronts the interpolation alone contributes ~1e-3, hiding
true model errors of ~1e-7. This script rebuilds the run's TimeMarchingModel
from its collected window checkpoints, rescores it on the solver's native grid
(the same metric as the per-window curves), overwrites the plot, and updates
full_domain_rel_l2 in time_marching_final_metrics.json.

Usage:
    python scripts/regenerate_tm_combined_curves.py <run_dir>

<run_dir> is the timestamped run folder containing window_*/ and
checkpoints/<segment>/ (e.g. outputs/experiments/<exp>/<name>/<timestamp>).
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.atoe_leaves import AToELeaves
from models.time_marching_model import TimeMarchingModel
from trainer.time_marching import (
    TimeWindow,
    _compute_full_domain_rel_l2,
    _load_prev_window_model,
    _plot_combined_loss_curves,
)

# Latest training stage wins — same priority as the combined checkpoint
SEGMENT_PRIORITY = ('fine_tune', 'phase3', 'root', 'main')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    seg_dir = next((run_dir / 'checkpoints' / s for s in SEGMENT_PRIORITY
                    if (run_dir / 'checkpoints' / s / 'manifest.json').exists()),
                   None)
    if seg_dir is None:
        sys.exit(f"No checkpoints/<segment>/manifest.json under {run_dir}")
    with open(seg_dir / 'manifest.json', 'r') as f:
        manifest = json.load(f)
    entries = sorted(manifest['windows'], key=lambda e: e['idx'])
    if len(entries) != manifest['num_windows']:
        sys.exit(f"{seg_dir.name} holds {len(entries)} of "
                 f"{manifest['num_windows']} windows — partial run, the "
                 f"full-domain metric is undefined.")
    print(f"Using segment '{manifest['segment']}' "
          f"({len(entries)} windows) from {seg_dir}")

    windows = [TimeWindow(idx=e['idx'], t_start=e['t_start'],
                          t_end=e['t_end'], is_first=(e['idx'] == 0),
                          M=e['M'])
               for e in entries]

    # Full-run config: the per-window checkpoints store their NARROWED config;
    # widen it back via the saved original temporal domain.
    ckpt0 = torch.load(seg_dir / entries[0]['file'], map_location='cpu',
                       weights_only=False)
    cfg = copy.deepcopy(ckpt0.get('config') or {})
    if not cfg:
        sys.exit(f"{entries[0]['file']} carries no stored config — cannot "
                 f"rebuild the run's model.")
    tm_flag = cfg.pop('_time_marching_window', None)
    full_domain = (tm_flag or {}).get('original_temporal_domain',
                                      manifest['temporal_domain'])
    cfg[cfg['problem']]['temporal_domain'] = list(full_domain)

    device = torch.device('cuda' if (cfg.get('cuda', False)
                                     and torch.cuda.is_available()) else 'cpu')
    print(f"Rebuilding TimeMarchingModel on {device} "
          f"(temporal domain {full_domain})")
    pairs = [(w, _load_prev_window_model(
                 str(seg_dir / e['file']), AToELeaves,
                 cfg['base_architecture'], cfg['activation'], cfg, w, device))
             for w, e in zip(windows, entries)]
    combined_model = TimeMarchingModel(pairs)

    final_rel_l2 = _compute_full_domain_rel_l2(combined_model, cfg, device)
    print(f"Full-domain rel-L2 (native solver grid): {final_rel_l2:.6e}")

    metrics_path = run_dir / 'time_marching_final_metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            final_metrics = json.load(f)
        old = final_metrics.get('full_domain_rel_l2')
        if old is not None:
            final_metrics['full_domain_rel_l2_interpolated_legacy'] = old
        final_metrics['full_domain_rel_l2'] = final_rel_l2
        final_metrics['full_domain_rel_l2_segment'] = manifest['segment']
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        print(f"Updated {metrics_path.name} "
              f"(legacy interpolated value kept: {old})")

    _plot_combined_loss_curves(windows, run_dir, final_rel_l2=final_rel_l2)


if __name__ == '__main__':
    main()
