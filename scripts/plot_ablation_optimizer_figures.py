"""Thesis ablation figures: SOAP vs two-phase Adam->L-BFGS for the higher order roots.

Writes to outputs/paper_figures/ablations/optimizer_higher_order/:
  ablation_optimizer_ks_w0_curves.png   rel-L2 vs iteration on the first KS window
                                        (configs identical except the optimizer keys)
  ablation_optimizer_kdv_windows.png    per-window rel-L2 of the 3-window KdV roots
  README.md                             run provenance

Runs (see docs/ablations_evidence.md / memory): the window-0 pair comes from
Desktop/outputs/ks_f32_w0dbg_ext_grid_v3_20260910_175232 (configs differ ONLY
in optimizer_1/optimizer_2/optimizer_switch_epoch); the tuned two-phase cell is
causS_t90 of ks_f32_w0dbg_h1k_causal_grid_completion_20260908_171049. Chains:
KS Adam-LBFGS = ks_timewindow_root_w5_T05_f32f64_pair_20260903_133238 (20k+30k,
non-periodic RFF), KS SOAP = ks_f32_tw5_T05_soap_clean_ext_100k_20260912_161744,
KdV Adam-LBFGS = kdv_f32_timewindow_root_w3_50a40l_20260903_051541 (NOT config-comparable:
flat lr, no extension), KdV SOAP =
ks_kdv_f32_soap_clean_ext_100k_chains_20260912_162205/.../20260912_215651.

Usage (repo root, NCC-PINN venv):
    python scripts/plot_ablation_optimizer_figures.py [--out-dir ...]
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils.plot_io import save_png  # noqa: E402

TICK_SIZE, LABEL_SIZE, LEGEND_SIZE = 14, 17, 13
C_SOAP, C_AL, C_AL_TUNED = '#1f77b4', '#d62728', '#ff7f0e'
DESK = Path(r'C:/Users/assaf/Desktop/outputs')
KS_BASE = 'ks-base-2-128-128-128-128-128-128-1-experts-2-30-30-30-1-tanh'

W0_CELLS = {
    'SOAP': DESK / 'ks_f32_w0dbg_ext_grid_v3_20260910_175232' / KS_BASE / '20260911_131510' / 'window_0' / 'metrics.json',
    'Adam→L-BFGS': DESK / 'ks_f32_w0dbg_ext_grid_v3_20260910_175232' / KS_BASE / '20260910_204134' / 'window_0' / 'metrics.json',
}
SWITCH_EPOCH = 50000

# per-window rel-L2 (native grid, each window on its own slab), from the runs' window_i summaries
KS_WINDOWS = {
    'Adam→L-BFGS': [5.546e-4, 5.911e-3, 6.290e-2, 3.371e-1, 7.083e-1],
    'SOAP':        [5.781e-6, 2.607e-5, 5.662e-5, 2.022e-4, 4.440e-4],
}
KS_FULL = {'Adam→L-BFGS': 4.327e-1, 'SOAP': 2.710e-4}
KDV_WINDOWS = {  # NOT config-comparable pair; kept for reference, not plotted
    'Adam→L-BFGS': [9.851e-4, 3.725e-3, 7.974e-3],
    'SOAP':        [8.741e-5, 4.023e-4, 1.792e-3],
}
KDV_FULL = {'Adam→L-BFGS': 5.116e-3, 'SOAP': 1.062e-3}


def _style(ax):
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    for axis in (ax.xaxis, ax.yaxis):
        axis.label.set_fontsize(LABEL_SIZE)
        axis.label.set_fontweight('bold')
    ax.grid(True, which='major', alpha=0.3)


def plot_w0_curves(ax):
    styles = {
        'SOAP': dict(color=C_SOAP, lw=2.2, ls='-'),
        'Adam→L-BFGS': dict(color=C_AL, lw=2.2, ls='-'),
    }
    for name, path in W0_CELLS.items():
        d = json.loads(path.read_text(encoding='utf-8'))
        ep, rl = np.asarray(d['epochs']), np.asarray(d['rel_l2'])
        best = float(np.min(rl))
        ax.plot(ep, rl, label=f'{name}: {best:.1e}', **styles[name])
    ax.axvline(SWITCH_EPOCH, color='gray', ls=':', lw=1.8)
    ax.text(SWITCH_EPOCH, ax.get_ylim()[1] if False else 2.0, ' L-BFGS\n switch', fontsize=11,
            fontweight='bold', color='gray', va='top')
    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Rel. $L^2$ error')
    ax.set_ylim(2e-6, 5)
    ax.legend(fontsize=LEGEND_SIZE, loc='lower left')
    _style(ax)


def plot_windows(ax, data, full, t_edges, title):
    n = len(t_edges) - 1
    x = np.arange(n)
    width = 0.38
    for k, (name, vals) in enumerate(data.items()):
        col = C_AL if name.startswith('Adam') else C_SOAP
        bars = ax.bar(x + (k - 0.5) * width, vals, width, color=col, alpha=0.9,
                      label=f'{name}  (full domain {full[name]:.2e})')
    ax.set_yscale('log')
    lo = min(min(v) for v in data.values()); hi = max(max(v) for v in data.values())
    ax.set_ylim(lo / 3, hi * 8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'w{i}\n[{t_edges[i]:.2g}, {t_edges[i+1]:.2g}]' for i in range(n)])
    ax.set_xlabel('Time window')
    ax.set_ylabel('Rel. $L^2$ error (window)')
    ax.set_title(title, fontsize=LABEL_SIZE, fontweight='bold')
    ax.legend(fontsize=LEGEND_SIZE - 1, loc='upper left')
    _style(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path,
                    default=REPO / 'outputs/paper_figures/ablations/optimizer_higher_order')
    a = ap.parse_args()
    out = a.out_dir; out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.5)); plot_w0_curves(ax); plt.tight_layout()
    print('saved', save_png(out / 'ablation_optimizer_ks_w0_curves.png', fig=fig)); plt.close(fig)



if __name__ == '__main__':
    main()
