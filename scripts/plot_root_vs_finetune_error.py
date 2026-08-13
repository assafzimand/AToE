"""Standalone: root vs. fine-tune error heatmaps, checkpoint-driven, shared color scale.

Unlike plot_split_pred_heatmaps.py (which crops pre-rendered PNGs because no
checkpoints existed for that experiment), this script loads real model
checkpoints and predicts on the solver's native grid -- so the two error
maps for a PDE (root, fine_tune) can be put on ONE shared log color scale,
making them directly comparable at a glance.

For each PDE this loads:
  - root:      roots_checkpoints/double_precision/no_resample-4-60-roots/<tag>_root.pt
               (num_experts=0, pure base network)
  - fine_tune: <fine_tune_run_dir>/checkpoints/best_model_fine_tune.pt
               (the fully split + fine-tuned AToELeaves model)

and renders, per output channel:
  - ground_truth_<pde>[_<channel>].png            (solver reference, own colorbar)
  - error_<pde>_root_ep<N>_relL2_<val>.png         (log-scale |error|)
  - error_<pde>_fine_tune_ep<N>_relL2_<val>.png    (SAME vmin/vmax as root)

Saved into the same outputs/paper_figures/heatmaps/ folder as
plot_split_pred_heatmaps.py, overwriting the root/fine_tune files there
(the phase3 crops from that script are untouched -- this script doesn't
touch phase3).

Usage:
    python scripts/plot_root_vs_finetune_error.py <fine_tune_run_dir> [<fine_tune_run_dir> ...] \\
        [--roots-dir roots_checkpoints/double_precision/no_resample-4-60-roots] \\
        [--out-dir outputs/paper_figures/heatmaps]

    Each fine_tune_run_dir is a leaf run dir containing config_used.yaml and
    checkpoints/best_model_fine_tune.pt, e.g.:
    outputs/experiments/ft_noresample_factorial_2x2x2_20260805_063430/
        burgers1d-base-2-60-60-60-60-1-experts-2-20-20-20-1-tanh/20260805_063435
"""

import sys
import math
import importlib
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.plot_io import save_png
from models.atoe_leaves import AToELeaves

TICK_SIZE = 14
LABEL_SIZE = 17
CBAR_TICK_SIZE = 13
CBAR_LABEL_SIZE = 15


def _style_axes(ax):
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.xaxis.label.set_fontsize(LABEL_SIZE)
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontsize(LABEL_SIZE)
    ax.yaxis.label.set_fontweight('bold')


def _style_colorbar(cbar):
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight('bold')
    if cbar.ax.yaxis.label.get_text():
        cbar.ax.yaxis.label.set_fontsize(CBAR_LABEL_SIZE)
        cbar.ax.yaxis.label.set_fontweight('bold')


_DIM_LABELS = {
    'allen_cahn': ['u'], 'burgers1d': ['u'], 'kdv': ['u'], 'ks': ['u'],
    'schrodinger': ['u', 'v'],
}
# roots_checkpoints file-name tag for each problem (burgers1d -> "burgers")
_ROOT_TAG = {'allen_cahn': 'allen_cahn', 'burgers1d': 'burgers', 'schrodinger': 'schrodinger'}


def native_grid(cfg):
    problem = cfg['problem']
    solver = importlib.import_module(f'solvers.{problem}_solver')
    x_grid, t_grid, h_sol = solver._get_solution_cached(cfg)
    t0, t1 = cfg[problem]['temporal_domain']
    t_grid = np.asarray(t_grid)
    t_mask = (t_grid >= t0 - 1e-12) & (t_grid <= t1 + 1e-12)
    return np.asarray(x_grid), t_grid[t_mask], np.asarray(h_sol)[t_mask]


def build_model(cfg, device):
    return AToELeaves(
        base_architecture=cfg['base_architecture'],
        activation=cfg['activation'],
        config=cfg,
        adaptive_config=cfg['adaptive_pinn'],
        experts_architecture=cfg.get('experts_architecture'),
    ).to(device)


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict_extended(ckpt['adaptive_state'])
    model.eval()
    return ckpt


def predict(model, xt, device, chunk=65536):
    dtype = next(model.parameters()).dtype
    out = []
    with torch.no_grad():
        for s in range(0, xt.shape[0], chunk):
            xb = torch.tensor(xt[s:s + chunk], dtype=dtype, device=device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


def save_ground_truth(pde, cfg, x_grid, t_grid, gt_channels, labels, out_dir):
    extent = [x_grid.min(), x_grid.max(), t_grid.min(), t_grid.max()]
    saved = []
    for label, gt in zip(labels, gt_channels):
        vmax = float(np.abs(gt).max())
        fig, ax = plt.subplots(figsize=(6, 5))
        X, T = np.meshgrid(x_grid, t_grid)
        im = ax.pcolormesh(X, T, gt, shading='auto', cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        _style_colorbar(cbar)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        _style_axes(ax)
        plt.tight_layout()
        suffix = f'_{label}' if len(labels) > 1 else ''
        out_path = out_dir / f'ground_truth_{pde}{suffix}.png'
        save_png(out_path, fig=fig)
        plt.close(fig)
        saved.append(out_path)
    return saved


def save_error_map(pde, segment, label, n_labels, x_grid, t_grid, err,
                    vmin, vmax, epoch, rel_l2, out_dir):
    extent = [x_grid.min(), x_grid.max(), t_grid.min(), t_grid.max()]
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
    suffix = f'_{label}' if n_labels > 1 else ''
    out_path = out_dir / f'error_{pde}_{segment}{suffix}_ep{epoch}_relL2_{rel_l2:.2e}.png'
    save_png(out_path, fig=fig)
    plt.close(fig)
    return out_path


def process_pde(ft_run_dir: Path, roots_dir: Path, out_dir: Path):
    cfg = yaml.safe_load((ft_run_dir / 'config_used.yaml').read_text(encoding='utf-8'))
    pde = cfg['problem']
    labels = _DIM_LABELS.get(pde, ['u'])

    if cfg.get('precision', 'float32') == 'float64':
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"=== {pde} ===")
    x_grid, t_grid, h_sol = native_grid(cfg)
    gt_channels = ([h_sol.real, h_sol.imag][:len(labels)]
                  if np.iscomplexobj(h_sol) else [h_sol])
    X, T = np.meshgrid(x_grid, t_grid)
    xt = np.column_stack([X.ravel(), T.ravel()])
    gt_flat = np.stack([c.ravel() for c in gt_channels], axis=1)

    root_ckpt_path = roots_dir / f'{_ROOT_TAG[pde]}_root.pt'
    ft_ckpt_path = ft_run_dir / 'checkpoints' / 'best_model_fine_tune.pt'

    segments = {}
    for seg_name, ckpt_path in [('root', root_ckpt_path), ('fine_tune', ft_ckpt_path)]:
        model = build_model(cfg, device)
        ckpt = load_checkpoint(model, ckpt_path)
        pred_flat = predict(model, xt, device)
        err = np.abs(pred_flat - gt_flat)                    # (N, n_channels)
        rel_l2 = math.sqrt(((pred_flat - gt_flat) ** 2).sum()
                           / (gt_flat ** 2).sum())
        print(f"  {seg_name}: epoch={ckpt['epoch']}  "
              f"ckpt rel_l2={ckpt.get('rel_l2'):.3e}  recomputed rel_l2={rel_l2:.3e}")
        err_grids = [err[:, d].reshape(len(t_grid), len(x_grid))
                    for d in range(len(labels))]
        segments[seg_name] = {'err_grids': err_grids, 'epoch': ckpt['epoch'],
                              'rel_l2': ckpt.get('rel_l2', rel_l2)}

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_paths = save_ground_truth(pde, cfg, x_grid, t_grid, gt_channels, labels, out_dir)
    for p in gt_paths:
        print(f"  saved {p.name}")

    for d, label in enumerate(labels):
        root_err = segments['root']['err_grids'][d]
        ft_err = segments['fine_tune']['err_grids'][d]
        joint_max = max(float(root_err.max()), float(ft_err.max()))
        pos = np.concatenate([root_err[root_err > 0], ft_err[ft_err > 0]])
        joint_min = max(float(pos.min()), joint_max * 1e-5) if pos.size else joint_max
        print(f"  channel {label}: shared scale [{joint_min:.2e}, {joint_max:.2e}]")

        for seg_name in ('root', 'fine_tune'):
            err = segments[seg_name]['err_grids'][d]
            out_path = save_error_map(
                pde, seg_name, label, len(labels), x_grid, t_grid, err,
                joint_min, joint_max, segments[seg_name]['epoch'],
                segments[seg_name]['rel_l2'], out_dir)
            print(f"    saved {out_path.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('fine_tune_run_dirs', nargs='+', type=Path)
    ap.add_argument('--roots-dir', type=Path,
                     default=Path('roots_checkpoints/double_precision/no_resample-4-60-roots'))
    ap.add_argument('--out-dir', type=Path,
                     default=Path('outputs/paper_figures/heatmaps'))
    args = ap.parse_args()

    for run_dir in args.fine_tune_run_dirs:
        process_pde(run_dir, args.roots_dir, args.out_dir)


if __name__ == '__main__':
    main()
