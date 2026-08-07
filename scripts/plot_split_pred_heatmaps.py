"""Standalone: split pred_after_<segment>_best_*.png into GT + error images.

``training_plots``/``adaptive_plots`` save ``pred_after_<segment>_best_ep<N>
_relL2_<val>.png``: one row per output channel, three columns [Ground Truth |
Prediction | Absolute error], GT+Prediction sharing one colorbar. This
script produces two clean, separate, paper-ready images per (PDE, segment):

  - ``ground_truth_<pde>[_<channel>].png``
        Regenerated fresh from the solver (no checkpoint needed) with its
        own colorbar. Segment-independent, so it's written once per PDE and
        reused across segments (root / phase3 / fine_tune all share the same
        ground truth).
  - ``error_<pde>_<segment>[_<channel>]_ep<N>_relL2_<val>.png``
        The "Absolute error" panel + its own colorbar, CROPPED out of the
        existing pred_after_..._best.png. This can't be regenerated (no
        model checkpoint is persisted for these runs), so it's pixel-cropped
        instead. Panel boundaries are found per-image by detecting
        whitespace gutters (matplotlib always renders this figure with the
        same structure: per-channel row band containing title/heatmap/
        ticks/x-label, and within it 3 roughly-equal-width heatmap bodies
        plus 1-2 colorbars) -- not hardcoded pixel offsets, so it adapts to
        each image's own content/DPI.

``[_<channel>]`` is added only for multi-output problems (Schrodinger: u,v).

Windows note: these run directories nest deep enough that the full path to
a pred_after_*.png can exceed the 260-char MAX_PATH limit, which breaks
plain ``open()``/PIL even though directory *listing* (glob/iterdir) still
works. Reads of existing files go through the ``\\\\?\\`` extended-length
prefix to work around this.

Usage:
    python scripts/plot_split_pred_heatmaps.py <experiment_root> [--out-dir DIR] [--tag best]

    experiment_root: searched recursively for pred_after_*_<tag>_ep*_relL2_*.png
                      (so pass the top-level experiment folder to also pick up
                      a nested INITIAL_FINE_TUNE_RESULTS/ subfolder).
"""

import os
import re
import sys
import importlib
import argparse
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.plot_io import save_png


_PRED_RE = re.compile(
    r'^pred_after_(?P<segment>.+)_(?P<tag>best|final)_ep(?P<epoch>\d+)'
    r'_relL2_(?P<relL2>[^_]+)\.png$')

_DIM_LABELS = {
    'allen_cahn': ['u'], 'burgers1d': ['u'], 'kdv': ['u'], 'ks': ['u'],
    'schrodinger': ['u', 'v'],
}

_WHITE_THRESH = 0.98     # RGB > this counts as background
_GAP_THRESH = 15         # px gap between content runs treated as a real boundary
_PAD = 8                 # px breathing room added around a detected crop


def _winlong(path: Path) -> str:
    """Windows extended-length path (bypasses the 260-char MAX_PATH limit)."""
    s = str(Path(path).resolve())
    if os.name == 'nt' and not s.startswith('\\\\?\\'):
        return '\\\\?\\' + s
    return s


def find_runs(mask_1d):
    """Contiguous True runs of a 1-D boolean array, as inclusive (start, end)."""
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(mask_1d):
        if v and not in_run:
            start, in_run = i, True
        if not v and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(mask_1d) - 1))
    return runs


def split_row_groups(row_runs, n_groups):
    """Split row-content runs into n_groups by cutting at the largest gaps."""
    if n_groups <= 1 or len(row_runs) <= 1:
        return [row_runs]
    gaps = [(row_runs[i + 1][0] - row_runs[i][1] - 1, i)
            for i in range(len(row_runs) - 1)]
    gaps.sort(key=lambda g: -g[0])
    cut_after = sorted(idx for _, idx in gaps[:n_groups - 1])
    groups, prev = [], 0
    for idx in cut_after:
        groups.append(row_runs[prev:idx + 1])
        prev = idx + 1
    groups.append(row_runs[prev:])
    return groups


def crop_error_panel(img, row_band, gap_thresh=_GAP_THRESH, pad=_PAD):
    """Crop the [error heatmap + its colorbar] out of one channel's row band.

    Returns the cropped RGBA array, or None if the expected 3-body structure
    isn't found (caller should fall back to skipping/warning).
    """
    r_lo = max(0, row_band[0][0] - pad)
    r_hi = min(img.shape[0] - 1, row_band[-1][1] + pad)
    band = img[row_band[0][0]:row_band[-1][1] + 1, :, :3]

    is_white = (band > _WHITE_THRESH).all(axis=2)
    col_content = ~is_white.all(axis=0)
    runs = find_runs(col_content)
    if len(runs) < 3:
        return None

    bodies = sorted(runs, key=lambda r: -(r[1] - r[0]))[:3]
    bodies.sort(key=lambda r: r[0])
    gt_body, pred_body, err_body = bodies

    # Self-calibrated margin: reuse how much left-side space GT's own
    # y-axis furniture (label + tick numbers) needed as a template for how
    # much Error's own furniture needs -- same font/style, rendered once
    # per channel by the same call, so this transfers directly instead of
    # guessing an absolute pixel constant.
    margin = max(0, gt_body[0] - _PAD)
    err_left = max(pred_body[1] + 1, err_body[0] - margin)
    err_right = min(band.shape[1] - 1, max(r[1] for r in runs) + pad)

    return img[r_lo:r_hi + 1, err_left:err_right + 1, :]


def native_grid(cfg):
    problem = cfg['problem']
    solver = importlib.import_module(f'solvers.{problem}_solver')
    x_grid, t_grid, h_sol = solver._get_solution_cached(cfg)
    t0, t1 = cfg[problem]['temporal_domain']
    t_grid = np.asarray(t_grid)
    t_mask = (t_grid >= t0 - 1e-12) & (t_grid <= t1 + 1e-12)
    return np.asarray(x_grid), t_grid[t_mask], np.asarray(h_sol)[t_mask]


def save_ground_truth(pde, cfg, out_dir):
    """One clean heatmap per output channel, regenerated from the solver."""
    x_grid, t_grid, h_sol = native_grid(cfg)
    labels = _DIM_LABELS.get(pde, ['u'])
    channels = ([h_sol.real, h_sol.imag][:len(labels)]
               if np.iscomplexobj(h_sol) else [h_sol])
    extent = [x_grid.min(), x_grid.max(), t_grid.min(), t_grid.max()]

    saved = []
    for label, gt in zip(labels, channels):
        vmax = float(np.abs(gt).max())
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.pcolormesh(
            np.meshgrid(x_grid, t_grid)[0], np.meshgrid(x_grid, t_grid)[1],
            gt, shading='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, pad=0.02)
        ax.set_xlabel('x', fontsize=13)
        ax.set_ylabel('t', fontsize=13)
        ax.tick_params(labelsize=11)
        plt.tight_layout()
        suffix = f'_{label}' if len(labels) > 1 else ''
        out_path = out_dir / f'ground_truth_{pde}{suffix}.png'
        save_png(out_path, fig=fig)
        plt.close(fig)
        saved.append(out_path)
    return saved


def process_pred_file(png_path: Path, out_dir: Path):
    m = _PRED_RE.match(png_path.name)
    if not m:
        return None
    segment, epoch, relL2 = m.group('segment'), m.group('epoch'), m.group('relL2')

    run_dir = png_path.parent.parent   # adaptive_plots/ -> run dir
    cfg = yaml.safe_load((run_dir / 'config_used.yaml').read_text(encoding='utf-8'))
    pde = cfg['problem']
    labels = _DIM_LABELS.get(pde, ['u'])
    output_dim = len(labels)

    img = mpimg.imread(_winlong(png_path))
    is_white = (img[..., :3] > _WHITE_THRESH).all(axis=2)
    row_content = ~is_white.all(axis=1)
    row_runs = find_runs(row_content)
    row_groups = split_row_groups(row_runs, output_dim)

    saved = []
    for label, row_band in zip(labels, row_groups):
        crop = crop_error_panel(img, row_band)
        if crop is None:
            print(f"  WARNING: couldn't find 3-panel structure for "
                  f"{png_path.name} (channel {label}); skipping")
            continue
        suffix = f'_{label}' if output_dim > 1 else ''
        out_path = out_dir / f'error_{pde}_{segment}{suffix}_ep{epoch}_relL2_{relL2}.png'
        mpimg.imsave(out_path, crop)
        saved.append(out_path)
    return pde, cfg, saved


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('experiment_root', type=Path)
    ap.add_argument('--out-dir', type=Path,
                     default=Path('outputs/paper_figures/heatmaps'))
    ap.add_argument('--tag', default='best', choices=['best', 'final'])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(args.experiment_root.rglob(f'pred_after_*_{args.tag}_ep*_relL2_*.png'))
    if not candidates:
        raise SystemExit(f"No pred_after_*_{args.tag}_*.png found under {args.experiment_root}")
    print(f"Found {len(candidates)} pred_after_*_{args.tag}_*.png file(s)")

    gt_done = set()
    for png_path in candidates:
        result = process_pred_file(png_path, args.out_dir)
        if result is None:
            print(f"  skip (name doesn't match expected pattern): {png_path}")
            continue
        pde, cfg, saved = result
        print(f"{png_path.relative_to(args.experiment_root)}")
        for p in saved:
            print(f"  saved {p.name}")
        if pde not in gt_done:
            gt_paths = save_ground_truth(pde, cfg, args.out_dir)
            for p in gt_paths:
                print(f"  saved {p.name} (ground truth, shared across segments)")
            gt_done.add(pde)


if __name__ == '__main__':
    main()
