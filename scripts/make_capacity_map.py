"""Regenerate capacity-density maps for finished runs from their metrics.json.

The heatmap shows parameter DENSITY (leaf params / region volume) using the
shared adaptive.visualization.plot_capacity_map — the same plot the trainer
emits after expert spawning. Use this script to (re)build the map for old runs
or time-marching runs (windows are stitched over the full domain).

Usage:
    python scripts/make_capacity_map.py <batch_or_run_dir>
"""

import sys
import json
import re
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptive.visualization import plot_capacity_map

_TS_RE = re.compile(r'\d{8}_\d{6}$')


def _is_time_marching_run(ts_dir: Path) -> bool:
    window_dirs = sorted(d for d in ts_dir.iterdir()
                         if d.is_dir() and d.name.startswith('window_'))
    return bool(window_dirs) and (window_dirs[0] / 'metrics.json').exists()


def _find_run_dirs(batch_path: Path):
    """Return list of (label, ts_dir) pairs for flat or nested batch layouts."""
    child_dirs = sorted(
        d for d in batch_path.iterdir()
        if d.is_dir() and d.name != 'checkpoints'
    )
    if not child_dirs:
        return []

    if _is_time_marching_run(batch_path):
        return [(batch_path.name, batch_path)]

    def _is_valid_run(d):
        if not _TS_RE.match(d.name):
            return False
        if (d / 'metrics.json').exists():
            return True
        return (d / 'window_0' / 'metrics.json').exists()

    flat_ts = [d for d in child_dirs if _is_valid_run(d)]
    if flat_ts:
        return [(d.name, d) for d in flat_ts]

    runs = []
    for model_dir in child_dirs:
        for ts_dir in sorted(d for d in model_dir.iterdir()
                             if d.is_dir() and d.name != 'checkpoints'):
            runs.append((f"{model_dir.name}/{ts_dir.name}", ts_dir))
    return runs


def _load_adaptive(ts_dir: Path):
    """Load adaptive metrics; stitches windows for time-marching runs."""
    if _is_time_marching_run(ts_dir):
        window_dirs = sorted(d for d in ts_dir.iterdir()
                             if d.is_dir() and d.name.startswith('window_'))
        regions, expert_params, leaf_indices = [], [], []
        base_params = 0
        offset = 0
        for wd in window_dirs:
            mp = wd / 'metrics.json'
            if not mp.exists():
                continue
            with open(mp) as f:
                adaptive = json.load(f).get('adaptive_pinn')
            if not adaptive:
                continue
            base_params = base_params or adaptive.get('base_params', 0)
            regions.extend(adaptive.get('regions', []))
            expert_params.extend(adaptive.get('expert_params', []))
            leaf_indices.extend(i + offset
                                for i in adaptive.get('leaf_expert_indices', []))
            offset = len(regions)
        return regions, expert_params, leaf_indices, base_params

    with open(ts_dir / 'metrics.json') as f:
        adaptive = json.load(f).get('adaptive_pinn')
    if not adaptive:
        return None
    return (adaptive['regions'], adaptive['expert_params'],
            adaptive.get('leaf_expert_indices', []),
            adaptive.get('base_params', 0))


def process_run(label: str, ts_dir: Path):
    cfg_path = ts_dir / 'config_used.yaml'
    if not cfg_path.exists():
        print(f"  [{label}] No config_used.yaml, skipping")
        return
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    loaded = _load_adaptive(ts_dir)
    if not loaded or not loaded[0]:
        print(f"  [{label}] No adaptive regions in metrics, skipping")
        return
    regions, expert_params, leaf_indices, base_params = loaded

    problem = cfg['problem']
    pc = cfg.get(problem, {})
    spatial_domain = pc.get('spatial_domain', [[0, 1]])
    temporal_domain = pc.get('temporal_domain', [0, 1])
    if len(spatial_domain) != 1:
        print(f"  [{label}] Only 1D-spatial problems supported, skipping")
        return
    domain_bounds = {
        'lower': [spatial_domain[0][0], temporal_domain[0]],
        'upper': [spatial_domain[0][1], temporal_domain[1]],
    }

    out_path = ts_dir / 'capacity_map.png'
    plot_capacity_map(
        regions=regions,
        expert_params=expert_params,
        leaf_indices=leaf_indices,
        base_params=base_params,
        domain_bounds=domain_bounds,
        output_path=out_path,
        title_suffix=f" — {problem}",
    )
    print(f"  [{label}] Saved {out_path}")


def main(batch_dir: str):
    batch_path = Path(batch_dir)
    if not batch_path.exists():
        print(f"Error: {batch_path} not found")
        return

    runs = _find_run_dirs(batch_path)
    if not runs:
        if (batch_path / 'metrics.json').exists() or _is_time_marching_run(batch_path):
            runs = [(batch_path.name, batch_path)]
        else:
            print("No runs found")
            return

    print(f"Found {len(runs)} run(s)")
    for label, ts_dir in runs:
        process_run(label, ts_dir)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python scripts/make_capacity_map.py <batch_or_run_dir>")
