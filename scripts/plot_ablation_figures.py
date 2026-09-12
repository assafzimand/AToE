"""Standalone: paper/thesis figures for the ablation section (Sec. 5.4).

Drives the existing paper plotters over a FIXED table of the runs chosen as
ablation evidence (see docs/ablations_evidence.md, the starred entries) and
writes everything under outputs/paper_figures/ablations/<group>/:

  time_tiling/       KdV f32 best chain: 3 time tiles vs one full-domain tree
                     (same root, same 3x45 experts, 24 leaves each), phase-3.
  overlap_width/     sigma sweeps at fixed experts/fine-tune: Allen-Cahn f64
                     (older M8 experts), KdV f32 50k-Adam chain, KdV f32 paper
                     chain (30k+30k), KdV f64 best chain.
  overlap_sampling/  fine-tune with uniform vs overlap-concentrated (0.3)
                     collocation on the paper experts (Burgers primary; the
                     neutral Allen-Cahn / Schrodinger pairs too).

Per cell it produces
  - training_curves_<group>_<pde>_<label>_<stage><N>_ep<N>_E<k>_relL2_<v>.png
    (+ per-panel _loss/_rel_l2/_components files), rendered with
    scripts/plot_concat_training_curves.plot_training_curves_paper; the
    rel-L2 panel y-range is shared across the cells of one (group, pde) so
    the curves are comparable side by side.
  - error_<pde>_<label>_<segment>_ep<N>_relL2_<v>_sharedscale.png
    when the run has checkpoints/best_model_<segment>.pt: |error| on the
    solver's native grid, log colour scale SHARED across the (group, pde)
    cells (same machinery as scripts/plot_root_vs_finetune_error.py).
  - error_<pde>_<label>_<segment>_ep<N>_relL2_<v>_cropped.png
    fallback for runs without a checkpoint: the error panel cropped out of
    the run's own pred_after_<segment>_*.png (its own colour scale, as in
    scripts/plot_split_pred_heatmaps.py).
  - ground_truth_<pde>.png once per (group, pde).

Labels in the file names spell out what varies (sigma value, sampling ratio,
tiling) so the figures can be picked by name.

Usage (from the repo root, NCC-PINN venv):
    python scripts/plot_ablation_figures.py [--groups time_tiling overlap_width overlap_sampling]
                                             [--out-dir outputs/paper_figures/ablations]
                                             [--no-heatmaps] [--no-curves]
"""

import os
import re
import sys
import math
import argparse
from pathlib import Path

import numpy as np
import yaml
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LogNorm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

from utils.plot_io import save_png                                   # noqa: E402
import plot_concat_training_curves as pc                             # noqa: E402
from plot_root_vs_finetune_error import (                            # noqa: E402
    native_grid, build_model, load_checkpoint, predict,
    save_ground_truth, _style_axes, _style_colorbar, _DIM_LABELS)
from plot_split_pred_heatmaps import (                               # noqa: E402
    find_runs, split_row_groups, crop_error_panel, _winlong, _WHITE_THRESH)

EXP = REPO / 'outputs' / 'experiments'
DESK = Path(r'C:\Users\assaf\Desktop')

_TANH_KDV128 = 'kdv-base-2-128-128-128-128-128-128-1-experts-2-45-45-45-1-tanh'
_TANH_KDV30 = 'kdv-base-2-128-128-128-128-128-128-1-experts-2-30-30-30-1-tanh'
_TANH_KDV64 = 'kdv-base-2-60-60-60-60-60-1-experts-2-24-24-24-1-tanh'
_TANH_AC = 'allen_cahn-base-2-60-60-60-60-1-experts-2-20-20-20-1-tanh'
_TANH_BG = 'burgers1d-base-2-60-60-60-60-1-experts-2-20-20-20-1-tanh'
_TANH_SC = 'schrodinger-base-2-60-60-60-60-2-experts-2-20-20-20-2-tanh'

# (label, run_dir, segment).  Labels name what varies in that group.
GROUPS = {
    'time_tiling': [
        ('tiled3_M25_24leaves',
         EXP / 'kdv_f32_ft_M25global_rootm_lrgrid_then_p3_globalbc_20260907_160942-kdv-best'
             / _TANH_KDV128 / '20260907_161222', 'phase3'),
        ('fulldomain_tree_M24_24leaves',
         EXP / 'kdv_f32_p3_fulldomtree_M24_e3x45_h100_globalbc_20260909_171732-time-tile-ablation'
             / _TANH_KDV128 / '20260909_171738', 'phase3'),
    ],
    'overlap_width': [
        # Allen-Cahn f64, older M8 experts, 30k SSBroyden, collar 0.3 (no checkpoints -> crops)
        ('allen_cahn_sigma0.03_col0.3',
         DESK / 'allen_cahn_ft_sigma_collar_sweep_20260713_195256' / _TANH_AC / '20260714_102634', 'fine_tune'),
        ('allen_cahn_sigma0.05_col0.3',
         DESK / 'allen_cahn_ft_sigma_collar_sweep_20260713_195256' / _TANH_AC / '20260714_025846', 'fine_tune'),
        ('allen_cahn_sigma0.10_col0.3',
         DESK / 'allen_cahn_ft_sigma_collar_sweep_20260713_195256' / _TANH_AC / '20260713_195259', 'fine_tune'),
        # KdV f32, 50k-Adam chain, M18 W3 3x30, SW L-BFGS lr 0.1, collar 0.3 (ft freezes at ep 1 => blend)
        ('kdv_f32_50a40l_sigma0.03_col0.3',
         DESK / 'kdv_f32_ft_ablation_lbfgslr_sigma_collar_20260827_145457' / _TANH_KDV30 / '20260827_145502', 'fine_tune'),
        ('kdv_f32_50a40l_sigma0.05_col0.3',
         DESK / 'kdv_f32_ft_ablation_lbfgslr_sigma_collar_20260827_145457' / _TANH_KDV30 / '20260827_151938', 'fine_tune'),
        ('kdv_f32_50a40l_sigma0.10_col0.3',
         DESK / 'kdv_f32_ft_ablation_lbfgslr_sigma_collar_20260827_145457' / _TANH_KDV30 / '20260827_154348', 'fine_tune'),
        # KdV f32 paper chain (30k+30k), same A1 experts, SW L-BFGS lr 1e-3
        ('kdv_f32_30a30l_sigma0.03_col0.3',
         EXP / 'kdv_lbfgs_fine_tunes_20260823_050609' / _TANH_KDV30 / '20260823_050614', 'fine_tune'),
        ('kdv_f32_30a30l_sigma0.05_col0.3',
         EXP / 'kdv_lbfgs_fine_tunes_20260823_050609' / _TANH_KDV30 / '20260823_055355', 'fine_tune'),
        ('kdv_f32_30a30l_sigma0.07_col0.3',
         EXP / 'kdv_lbfgs_fine_tunes_20260823_050609' / _TANH_KDV30 / '20260823_062137', 'fine_tune'),
        ('kdv_f32_30a30l_sigma0.20_col0',
         DESK / 'kdv_lbfgs_fine_tunes_20260822_171003' / _TANH_KDV30 / '20260822_174015', 'fine_tune'),
        ('kdv_f32_30a30l_sigma0.30_col0',
         DESK / 'kdv_lbfgs_fine_tunes_20260822_171003' / _TANH_KDV30 / '20260822_184227', 'fine_tune'),
        # KdV f64 best chain (M12 W2 3x24), SSBroyden lr 1.0, collar 0.3
        ('kdv_f64_sigma0.03_col0.3',
         EXP / 'kdv_f64_ft_ssb_grid_sigma_collar_lr_20260828_123908' / _TANH_KDV64 / '20260828_125609', 'fine_tune'),
        ('kdv_f64_sigma0.05_col0.3',
         EXP / 'kdv_f64_ft_ssb_grid_sigma_collar_lr_20260828_123908' / _TANH_KDV64 / '20260828_151100', 'fine_tune'),
        ('kdv_f64_sigma0.10_col0.3',
         EXP / 'kdv_f64_ft_ssb_grid_sigma_collar_lr_20260828_123908' / _TANH_KDV64 / '20260828_202054', 'fine_tune'),
    ],
    'overlap_sampling': [
        # paper experts (sha256-identical checkpoints), sigma 0.05, SSBroyden lr 1.0
        ('burgers1d_uniform_col0',
         EXP / 'finetune_corrector_ablation_20260725_053553' / _TANH_BG / '20260725_091154', 'fine_tune'),
        ('burgers1d_overlap_col0.3',
         EXP / 'ft_noresample_factorial_2x2x2_20260805_063430' / _TANH_BG / '20260805_063435', 'fine_tune'),
        ('allen_cahn_uniform_col0',
         EXP / 'finetune_nocorr_40k_20260725_170953' / _TANH_AC / '20260725_204151', 'fine_tune'),
        ('allen_cahn_overlap_col0.3',
         EXP / 'ft_noresample_factorial_2x2x2_20260805_063430' / _TANH_AC / '20260805_081437', 'fine_tune'),
        ('schrodinger_uniform_col0',
         EXP / 'finetune_nocorr_40k_20260725_170953' / _TANH_SC / '20260725_170956', 'fine_tune'),
        ('schrodinger_overlap_col0.3',
         EXP / 'ft_noresample_factorial_2x2x2_20260805_063430' / _TANH_SC / '20260805_095246', 'fine_tune'),
    ],
}

# Upper end of the shared log colour scale, per (group, scale key), for
# groups where one arm has no checkpoint and is shown as a crop of its own
# pred png: the rendered arm's colourbar is extended up to the crop's max so
# the two bars are comparable in their upper decades (the crop's own scale
# is whatever the original run plotted).
SCALE_VMAX_OVERRIDE = {
    ('overlap_sampling', 'burgers1d'): 3e-2,   # uniform-arm crop tops out ~3e-2
}

_PRED_RE = re.compile(
    r'^pred_after_(?P<segment>.+?)_(?:(?P<tag>best|final)_)?ep(?P<epoch>\d+)'
    r'_relL2_(?P<relL2>[^_]+)\.png$')


# ----------------------------------------------------------------- helpers
def _pde_label(pde: str, label: str) -> str:
    """'<pde>_<label>' unless the label already starts with the PDE name."""
    return label if label.startswith(pde) else f'{pde}_{label}'


def _scale_key(group: str, pde: str, label: str) -> str:
    """Cells that share one colour scale / rel-L2 y-range. For the width
    sweep that is one chain (label up to '_sigma'), since the KdV f32 and
    f64 chains sit 3 decades apart; otherwise the PDE."""
    if group == 'overlap_width':
        return label.split('_sigma')[0]
    return pde


def _load_cfg(run_dir: Path):
    with open(_winlong(run_dir / 'config_used.yaml'), encoding='utf-8') as f:
        return yaml.safe_load(f)


_orig_load_run = pc.load_run


def _sanitized_load_run(run_dir):
    """Older metrics.json files store None for unused event lists; the
    concat helper iterates them, so normalise to []."""
    metrics, cfg = _orig_load_run(run_dir)
    for k in ('optimizer_events', 'segment_events', 'segment_reconcile_events',
              'inf_norm'):
        if metrics.get(k) is None:
            metrics[k] = []
    return metrics, cfg


pc.load_run = _sanitized_load_run


# ------------------------------------------------------------ training curves
def curves_for_group(group, cells, out_dir):
    """One training-curve figure per cell; rel-L2 y-range shared per PDE."""
    loaded = []
    for label, run_dir, segment in cells:
        combined, seg_markers, opt_switch, meta = pc.concat_runs([(run_dir, None)])
        # concat_runs keeps only the curve series; carry the reference-line
        # metadata (root / phase-3-checkpoint rel-L2) so the rel-L2 panel
        # draws the horizontal reference exactly as the live plots do.
        raw, _ = pc.load_run(run_dir)
        for k in ('root_rel_l2', 'root_loaded_from_checkpoint',
                  'pretrained_experts_rel_l2'):
            if raw.get(k) is not None:
                combined[k] = raw[k]
        loaded.append((label, combined, seg_markers, opt_switch, meta))

    ylim_by_key = {}
    for label, combined, _, _, meta in loaded:
        rl = [v for v in combined['rel_l2'] if v and v > 0]
        if not rl:
            continue
        key = _scale_key(group, meta['pde'], label)
        lo, hi = ylim_by_key.get(key, (math.inf, 0.0))
        ylim_by_key[key] = (min(lo, min(rl)), max(hi, max(rl)))

    for label, combined, seg_markers, opt_switch, meta in loaded:
        lo, hi = ylim_by_key[_scale_key(group, meta['pde'], label)]
        ylim = (lo / 2.0, hi * 2.0)
        relL2 = meta['final_rel_l2']
        suffix = (f"{group}_{_pde_label(meta['pde'], label)}_{'_'.join(meta['stage_tags'])}"
                  f"_ep{meta['total_epochs']}_E{meta['num_experts']}"
                  + (f"_relL2_{relL2:.2e}" if relL2 is not None else ''))
        print(f"[curves] {label}: final rel-L2 {relL2:.3e}" if relL2 else f"[curves] {label}")
        pc.plot_training_curves_paper(
            combined, save_dir=out_dir, optimizer_switch_epochs=opt_switch,
            segment_markers=seg_markers, name_suffix=suffix,
            show_root_ref=True, rel_l2_ylim=ylim)


# ------------------------------------------------------------------ heatmaps
def _error_from_checkpoint(run_dir, cfg, segment, xt, gt_flat, n_t, n_x):
    ckpt_path = run_dir / 'checkpoints' / f'best_model_{segment}.pt'
    if not ckpt_path.exists():
        return None
    if cfg.get('precision', 'float32') == 'float64':
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg, device)
    ckpt = load_checkpoint(model, ckpt_path)
    pred = predict(model, xt, device)
    err = np.abs(pred - gt_flat)
    rel = math.sqrt(((pred - gt_flat) ** 2).sum() / (gt_flat ** 2).sum())
    ckpt_rel = ckpt.get('rel_l2')
    print(f"    ckpt {ckpt_path.name}: epoch={ckpt['epoch']} ckpt rel_l2="
          f"{'n/a' if ckpt_rel is None else f'{ckpt_rel:.3e}'} recomputed={rel:.3e}")
    grids = [err[:, d].reshape(n_t, n_x) for d in range(err.shape[1])]
    return grids, ckpt['epoch'], (ckpt_rel if ckpt_rel is not None else rel)


def _save_shared_error_map(pde, label, segment, chan, n_chan, x_grid, t_grid,
                           err, vmin, vmax, epoch, rel_l2, out_dir):
    X, T = np.meshgrid(x_grid, t_grid)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(X, T, np.maximum(err, vmin), shading='auto', cmap='Reds',
                       norm=LogNorm(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(im, ax=ax, pad=0.02, label='|error| (log)')
    _style_colorbar(cbar)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    _style_axes(ax)
    plt.tight_layout()
    suffix = f'_{chan}' if n_chan > 1 else ''
    out = out_dir / (f'error_{_pde_label(pde, label)}_{segment}{suffix}_ep{epoch}'
                     f'_relL2_{rel_l2:.2e}_sharedscale.png')
    save_png(out, fig=fig)
    plt.close(fig)
    return out


def _crop_from_png(run_dir, cfg, label, segment, out_dir):
    """Fallback for runs without checkpoints: crop the error panel out of
    the run's pred_after_<segment>_*.png (prefer the *_best_* one)."""
    plots = run_dir / 'adaptive_plots'
    cands = [p for p in plots.glob(f'pred_after_{segment}_*relL2_*.png')
             if _PRED_RE.match(p.name)]
    if not cands:
        print(f"    no checkpoint and no pred_after_{segment} png -> skipped")
        return []
    cands.sort(key=lambda p: (0 if '_best_' in p.name else 1, p.name))
    png = cands[0]
    m = _PRED_RE.match(png.name)
    pde = cfg['problem']
    labels = _DIM_LABELS.get(pde, ['u'])
    img = mpimg.imread(_winlong(png))
    is_white = (img[..., :3] > _WHITE_THRESH).all(axis=2)
    row_runs = find_runs(~is_white.all(axis=1))
    saved = []
    for chan, band in zip(labels, split_row_groups(row_runs, len(labels))):
        crop = crop_error_panel(img, band)
        if crop is None:
            print(f"    WARNING: 3-panel structure not found in {png.name} ({chan})")
            continue
        suffix = f'_{chan}' if len(labels) > 1 else ''
        out = out_dir / (f"error_{_pde_label(pde, label)}_{segment}{suffix}_ep{m.group('epoch')}"
                         f"_relL2_{m.group('relL2')}_cropped.png")
        mpimg.imsave(out, crop)
        saved.append(out)
        print(f"    cropped {png.name} -> {out.name}")
    return saved


def heatmaps_for_group(group, cells, out_dir):
    # group cells by scale key (PDE, or chain for the width sweep) so the
    # shared colour scale only spans comparable cells
    by_key = {}
    for label, run_dir, segment in cells:
        cfg = _load_cfg(run_dir)
        by_key.setdefault(_scale_key(group, cfg['problem'], label), []).append(
            (label, run_dir, segment, cfg))

    for key, items in by_key.items():
        cfg0 = items[0][3]
        pde = cfg0['problem']
        labels = _DIM_LABELS.get(pde, ['u'])
        x_grid, t_grid, h_sol = native_grid(cfg0)
        gt_channels = ([h_sol.real, h_sol.imag][:len(labels)]
                       if np.iscomplexobj(h_sol) else [h_sol])
        X, T = np.meshgrid(x_grid, t_grid)
        xt = np.column_stack([X.ravel(), T.ravel()])
        gt_flat = np.stack([c.ravel() for c in gt_channels], axis=1)
        for p in save_ground_truth(pde, cfg0, x_grid, t_grid, gt_channels, labels, out_dir):
            print(f"  [{group}/{pde}] saved {p.name}")

        rendered = []   # (label, segment, grids, epoch, rel)
        for label, run_dir, segment, cfg in items:
            print(f"  [{group}/{pde}] {label}")
            res = _error_from_checkpoint(run_dir, cfg, segment, xt, gt_flat,
                                         len(t_grid), len(x_grid))
            if res is None:
                _crop_from_png(run_dir, cfg, label, segment, out_dir)
            else:
                rendered.append((label, segment, *res))

        if not rendered:
            continue
        for d, chan in enumerate(labels):
            grids = [r[2][d] for r in rendered]
            vmax = max(float(g.max()) for g in grids)
            pos = np.concatenate([g[g > 0] for g in grids])
            vmin = max(float(pos.min()), vmax * 1e-5) if pos.size else vmax
            if (group, key) in SCALE_VMAX_OVERRIDE:
                vmax = max(vmax, SCALE_VMAX_OVERRIDE[(group, key)])
            print(f"  [{group}/{key}] channel {chan}: shared scale [{vmin:.2e}, {vmax:.2e}]")
            for label, segment, gr, epoch, rel in rendered:
                out = _save_shared_error_map(pde, label, segment, chan, len(labels),
                                             x_grid, t_grid, gr[d], vmin, vmax,
                                             epoch, rel, out_dir)
                print(f"    saved {out.name}")


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--groups', nargs='+', choices=sorted(GROUPS), default=sorted(GROUPS))
    ap.add_argument('--out-dir', type=Path, default=REPO / 'outputs' / 'paper_figures' / 'ablations')
    ap.add_argument('--no-heatmaps', action='store_true')
    ap.add_argument('--no-curves', action='store_true')
    args = ap.parse_args()

    os.chdir(REPO)   # solver caches / relative checkpoint paths in configs
    for group in args.groups:
        cells = GROUPS[group]
        missing = [str(r) for _, r, _ in cells if not (r / 'config_used.yaml').exists()]
        if missing:
            raise SystemExit(f"[{group}] missing run dirs:\n  " + "\n  ".join(missing))
        out_dir = args.out_dir / group
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"===== {group} -> {out_dir}")
        if not args.no_curves:
            curves_for_group(group, cells, out_dir)
        if not args.no_heatmaps:
            heatmaps_for_group(group, cells, out_dir)


if __name__ == '__main__':
    main()
