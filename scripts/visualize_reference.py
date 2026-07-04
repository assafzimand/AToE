"""
Visualize Raissi's reference .mat datasets as heatmaps.

Saves one PNG per dataset directly next to the .mat file in
datasets/reference/{pde}/reference_heatmap.png

Usage:
    python scripts/visualize_reference.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

REF_DIR = Path("datasets") / "reference"


def _save_heatmap(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def _heatmap_panel(ax, T, X, Z, title, cmap='RdBu_r', symmetric=False):
    if symmetric:
        vmax = np.abs(Z).max()
        im = ax.pcolormesh(T, X, Z, shading='auto', cmap=cmap,
                           vmin=-vmax, vmax=vmax)
    else:
        im = ax.pcolormesh(T, X, Z, shading='auto', cmap=cmap)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


# ── Burgers 1D ────────────────────────────────────────────────────────────────
def viz_burgers1d():
    mat_path = REF_DIR / 'burgers1d' / 'burgers_shock.mat'
    if not mat_path.exists():
        print("  [burgers1d] .mat not found, skipping.")
        return

    mat = sio.loadmat(str(mat_path))
    x = mat['x'].flatten()          # (256,)
    t = mat['t'].flatten()           # (100,)
    u = np.real(mat['usol'])         # (256, 100)
    T, X = np.meshgrid(t, x)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f'Burgers 1D — Raissi reference\n'
                 f'x∈[{x.min():.2f}, {x.max():.2f}], '
                 f't∈[{t.min():.4f}, {t.max():.4f}], '
                 f'grid {u.shape}', fontsize=11)

    _heatmap_panel(axes[0], T, X, u, 'u(x,t)', cmap='RdBu_r', symmetric=True)

    # Temporal slices
    axes[1].set_title('Temporal slices', fontsize=11)
    n_slices = min(5, len(t))
    idxs = np.linspace(0, len(t) - 1, n_slices, dtype=int)
    for j in idxs:
        axes[1].plot(x, u[:, j], label=f't={t[j]:.3f}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('u')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_heatmap(fig, mat_path.parent / 'reference_heatmap.png')


# ── Schrödinger ───────────────────────────────────────────────────────────────
def viz_schrodinger():
    mat_path = REF_DIR / 'schrodinger' / 'NLS.mat'
    if not mat_path.exists():
        print("  [schrodinger] .mat not found, skipping.")
        return

    mat = sio.loadmat(str(mat_path))
    x = mat['x'].flatten()           # (256,)
    t = mat['tt'].flatten()           # (201,)
    uu = mat['uu']                    # complex (256, 201)
    T, X = np.meshgrid(t, x)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(f'Schrödinger (NLS) — Raissi reference\n'
                 f'x∈[{x.min():.2f}, {x.max():.2f}], '
                 f't∈[{t.min():.4f}, {t.max():.4f}], '
                 f'grid {uu.shape}', fontsize=11)

    _heatmap_panel(axes[0], T, X, np.abs(uu),  '|h(x,t)|', cmap='viridis')
    _heatmap_panel(axes[1], T, X, uu.real,     'Re h(x,t)', cmap='RdBu_r', symmetric=True)
    _heatmap_panel(axes[2], T, X, uu.imag,     'Im h(x,t)', cmap='RdBu_r', symmetric=True)

    plt.tight_layout()
    _save_heatmap(fig, mat_path.parent / 'reference_heatmap.png')


# ── KdV ───────────────────────────────────────────────────────────────────────
def viz_kdv():
    mat_path = REF_DIR / 'kdv' / 'KdV.mat'
    if not mat_path.exists():
        print("  [kdv] .mat not found, skipping.")
        return

    mat = sio.loadmat(str(mat_path))
    x = mat['x'].flatten()           # (512,)
    t = mat['tt'].flatten()          # (201,)
    u = np.real(mat['uu'])           # (512, 201)
    T, X = np.meshgrid(t, x)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f'KdV — Raissi reference\n'
                 f'x∈[{x.min():.2f}, {x.max():.2f}], '
                 f't∈[{t.min():.4f}, {t.max():.4f}], '
                 f'grid {u.shape}', fontsize=11)

    _heatmap_panel(axes[0], T, X, u, 'u(x,t)', cmap='RdBu_r', symmetric=True)

    axes[1].set_title('Temporal slices', fontsize=11)
    n_slices = min(5, len(t))
    idxs = np.linspace(0, len(t) - 1, n_slices, dtype=int)
    for j in idxs:
        axes[1].plot(x, u[:, j], label=f't={t[j]:.3f}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('u')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_heatmap(fig, mat_path.parent / 'reference_heatmap.png')


# ── Allen-Cahn ────────────────────────────────────────────────────────────────
def viz_allen_cahn():
    mat_path = REF_DIR / 'allen_cahn' / 'AC.mat'
    if not mat_path.exists():
        print("  [allen_cahn] .mat not found, skipping.")
        return

    mat = sio.loadmat(str(mat_path))
    x = mat['x'].flatten()           # (512,)
    t = mat['tt'].flatten()          # (201,)
    u = np.real(mat['uu'])           # (512, 201)
    T, X = np.meshgrid(t, x)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f'Allen-Cahn — Raissi reference\n'
                 f'x∈[{x.min():.2f}, {x.max():.2f}], '
                 f't∈[{t.min():.4f}, {t.max():.4f}], '
                 f'grid {u.shape}', fontsize=11)

    _heatmap_panel(axes[0], T, X, u, 'u(x,t)', cmap='RdBu_r', symmetric=True)

    axes[1].set_title('Temporal slices', fontsize=11)
    n_slices = min(5, len(t))
    idxs = np.linspace(0, len(t) - 1, n_slices, dtype=int)
    for j in idxs:
        axes[1].plot(x, u[:, j], label=f't={t[j]:.3f}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('u')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_heatmap(fig, mat_path.parent / 'reference_heatmap.png')


VISUALIZERS = [
    ('burgers1d',  viz_burgers1d),
    ('schrodinger', viz_schrodinger),
    ('kdv',        viz_kdv),
    ('allen_cahn', viz_allen_cahn),
]


def main():
    print("=" * 60)
    print("VISUALIZE RAISSI REFERENCE DATASETS")
    print("=" * 60)
    for name, fn in VISUALIZERS:
        print(f"\n[{name}]")
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
    print("\nDone.")


if __name__ == "__main__":
    main()
