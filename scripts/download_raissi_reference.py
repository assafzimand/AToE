"""
Download Raissi's reference .mat datasets from the PINNs GitHub repo
and optionally compare them with our generated ground truth.

Reference: https://github.com/maziarraissi/PINNs

Datasets:
  burgers_shock.mat  - Burgers 1D (exact Cole-Hopf, nu=0.01/pi)
  NLS.mat            - Nonlinear Schrodinger
  KdV.mat            - Korteweg-de Vries
  AC.mat             - Allen-Cahn
"""

import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

MAIN_DATA = ("https://github.com/maziarraissi/PINNs/"
             "raw/master/main/Data")
APPENDIX_DATA = ("https://github.com/maziarraissi/PINNs/"
                 "raw/master/appendix/Data")

DATASETS = {
    'burgers1d': (APPENDIX_DATA, 'burgers_shock.mat'),
    'schrodinger': (MAIN_DATA, 'NLS.mat'),
    'kdv': (MAIN_DATA, 'KdV.mat'),
    'allen_cahn': (MAIN_DATA, 'AC.mat'),
}


def download_all(dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for pde, (base_url, fname) in DATASETS.items():
        pde_dir = dest_dir / pde
        pde_dir.mkdir(parents=True, exist_ok=True)
        target = pde_dir / fname
        if target.exists():
            print(f"  [{pde}] {fname} already exists, skipping.")
            continue
        url = f"{base_url}/{fname}"
        print(f"  [{pde}] Downloading {fname} ...")
        try:
            urllib.request.urlretrieve(url, str(target))
            size_mb = target.stat().st_size / 1024 / 1024
            print(f"         Saved ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"         FAILED: {e}")


def compare_burgers1d(ref_dir: Path, our_dir: Path):
    import scipy.io as sio
    mat = sio.loadmat(str(ref_dir / 'burgers1d' / 'burgers_shock.mat'))
    x_ref = mat['x'].flatten()
    t_ref = mat['t'].flatten()
    u_ref = np.real(mat['usol'])  # (nx, nt)

    import torch
    fg = torch.load(our_dir / 'burgers1d' / 'frequency_grid.pt',
                    weights_only=False)
    gs = fg['grid_shape']
    xg = fg['x_grid'].numpy()
    hg = fg['h_gt_grid'].numpy()
    n_x, n_t = gs[0], gs[1]
    x_our = xg[::n_t, 0]
    t_our = xg[:n_t, 1]
    u_our = hg[:, 0].reshape(n_x, n_t)  # (n_x, n_t)

    from scipy.interpolate import RegularGridInterpolator
    interp_ours = RegularGridInterpolator(
        (x_our, t_our), u_our, method='linear', bounds_error=False,
        fill_value=None)

    X, T = np.meshgrid(x_ref, t_ref, indexing='ij')
    pts = np.column_stack([X.ravel(), T.ravel()])
    u_ours_on_ref = interp_ours(pts).reshape(u_ref.shape)

    diff = u_ours_on_ref - u_ref
    norm_ref = np.linalg.norm(u_ref)
    rel_l2 = np.linalg.norm(diff) / norm_ref if norm_ref > 0 else float('inf')
    print(f"  Burgers1D: ||ours - Raissi|| / ||Raissi|| = {rel_l2:.6e}")
    print(f"             max |diff| = {np.abs(diff).max():.6e}")
    return rel_l2


def compare_schrodinger(ref_dir: Path, our_dir: Path):
    import scipy.io as sio
    mat = sio.loadmat(str(ref_dir / 'schrodinger' / 'NLS.mat'))
    x_ref = mat['x'].flatten()   # (256,)
    t_ref = mat['tt'].flatten()   # (201,)
    uu = mat['uu']                # complex (256, 201)

    import torch
    fg = torch.load(our_dir / 'schrodinger' / 'frequency_grid.pt',
                    weights_only=False)
    gs = fg['grid_shape']
    xg = fg['x_grid'].numpy()
    hg = fg['h_gt_grid'].numpy()
    n_x, n_t = gs[0], gs[1]
    x_our = xg[::n_t, 0]
    t_our = xg[:n_t, 1]
    u_our = hg[:, 0].reshape(n_x, n_t)
    v_our = hg[:, 1].reshape(n_x, n_t)

    from scipy.interpolate import RegularGridInterpolator
    interp_u = RegularGridInterpolator(
        (x_our, t_our), u_our, method='linear', bounds_error=False,
        fill_value=None)
    interp_v = RegularGridInterpolator(
        (x_our, t_our), v_our, method='linear', bounds_error=False,
        fill_value=None)

    X, T = np.meshgrid(x_ref, t_ref, indexing='ij')
    pts = np.column_stack([X.ravel(), T.ravel()])
    u_on_ref = interp_u(pts).reshape(uu.shape)
    v_on_ref = interp_v(pts).reshape(uu.shape)
    h_ours = u_on_ref + 1j * v_on_ref

    diff = h_ours - uu
    norm_ref = np.linalg.norm(uu)
    rel_l2 = np.linalg.norm(diff) / norm_ref if norm_ref > 0 else float('inf')
    print(f"  Schrodinger: ||ours - Raissi|| / ||Raissi|| = {rel_l2:.6e}")
    print(f"               max |diff| = {np.abs(diff).max():.6e}")
    return rel_l2


def compare_kdv(ref_dir: Path, our_dir: Path):
    import scipy.io as sio
    mat = sio.loadmat(str(ref_dir / 'kdv' / 'KdV.mat'))
    x_ref = mat['x'].flatten()
    t_ref = mat['tt'].flatten()
    u_ref = np.real(mat['uu'])  # (nx, nt)

    import torch
    fg = torch.load(our_dir / 'kdv' / 'frequency_grid.pt',
                    weights_only=False)
    gs = fg['grid_shape']
    xg = fg['x_grid'].numpy()
    hg = fg['h_gt_grid'].numpy()
    n_x, n_t = gs[0], gs[1]
    x_our = xg[::n_t, 0]
    t_our = xg[:n_t, 1]
    u_our = hg[:, 0].reshape(n_x, n_t)

    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (x_our, t_our), u_our, method='linear', bounds_error=False,
        fill_value=None)

    X, T = np.meshgrid(x_ref, t_ref, indexing='ij')
    pts = np.column_stack([X.ravel(), T.ravel()])
    u_ours_on_ref = interp(pts).reshape(u_ref.shape)

    diff = u_ours_on_ref - u_ref
    norm_ref = np.linalg.norm(u_ref)
    rel_l2 = np.linalg.norm(diff) / norm_ref if norm_ref > 0 else float('inf')
    print(f"  KdV: ||ours - Raissi|| / ||Raissi|| = {rel_l2:.6e}")
    print(f"       max |diff| = {np.abs(diff).max():.6e}")
    return rel_l2


def compare_allen_cahn(ref_dir: Path, our_dir: Path):
    import scipy.io as sio
    mat = sio.loadmat(str(ref_dir / 'allen_cahn' / 'AC.mat'))
    x_ref = mat['x'].flatten()
    t_ref = mat['tt'].flatten()
    u_ref = np.real(mat['uu'])  # (nx, nt)

    import torch
    fg = torch.load(our_dir / 'allen_cahn' / 'frequency_grid.pt',
                    weights_only=False)
    gs = fg['grid_shape']
    xg = fg['x_grid'].numpy()
    hg = fg['h_gt_grid'].numpy()
    n_x, n_t = gs[0], gs[1]
    x_our = xg[::n_t, 0]
    t_our = xg[:n_t, 1]
    u_our = hg[:, 0].reshape(n_x, n_t)

    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (x_our, t_our), u_our, method='linear', bounds_error=False,
        fill_value=None)

    X, T = np.meshgrid(x_ref, t_ref, indexing='ij')
    pts = np.column_stack([X.ravel(), T.ravel()])
    u_ours_on_ref = interp(pts).reshape(u_ref.shape)

    diff = u_ours_on_ref - u_ref
    norm_ref = np.linalg.norm(u_ref)
    rel_l2 = np.linalg.norm(diff) / norm_ref if norm_ref > 0 else float('inf')
    print(f"  Allen-Cahn: ||ours - Raissi|| / ||Raissi|| = {rel_l2:.6e}")
    print(f"              max |diff| = {np.abs(diff).max():.6e}")
    return rel_l2


COMPARATORS = {
    'burgers1d': compare_burgers1d,
    'schrodinger': compare_schrodinger,
    'kdv': compare_kdv,
    'allen_cahn': compare_allen_cahn,
}


def main():
    ref_dir = Path("datasets") / "reference"
    our_dir = Path("datasets")

    print("=" * 60)
    print("DOWNLOAD RAISSI REFERENCE DATASETS")
    print("=" * 60)
    download_all(ref_dir)

    print("\n" + "=" * 60)
    print("COMPARE OUR GROUND TRUTH vs RAISSI")
    print("=" * 60)

    for pde, cmp_fn in COMPARATORS.items():
        _, fname = DATASETS[pde]
        mat_file = ref_dir / pde / fname
        fg_file = our_dir / pde / 'frequency_grid.pt'
        if not mat_file.exists():
            print(f"  [{pde}] Reference .mat not found, skipping comparison.")
            continue
        if not fg_file.exists():
            print(f"  [{pde}] Our frequency_grid.pt not found, skipping.")
            continue
        try:
            cmp_fn(ref_dir, our_dir)
        except Exception as e:
            print(f"  [{pde}] Comparison failed: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
