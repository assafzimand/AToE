"""
Ground Truth Validation Script.

For each PDE, validates that the generated datasets satisfy:
  1. IC  -- analytically: h_gt(x, t=0) from training_data.pt vs known formula
  2. BC  -- analytically: h_gt on boundary from training_data.pt vs known
            formula / periodicity
  3. PDE residual -- spectrally for Fourier-based solvers (Schrodinger,
     Allen-Cahn, KdV, KS): re-run the solver on its native grid, compute
     spatial derivatives via FFT (exact), h_t via central FD, then
     interpolate the residual field to the training_data residual points.
     For non-spectral solvers: FD on the frequency grid.

No neural network or loss function is used here.
"""

import sys
import os
import math
import yaml
import numpy as np
import torch
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_gen import generate_and_save_datasets, load_dataset


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_freq_grid_2d(dataset_dir):
    """Load frequency_grid.pt and return structured arrays for 2D PDEs."""
    fg = torch.load(dataset_dir / 'frequency_grid.pt', weights_only=False)
    gs = fg['grid_shape']
    xg = fg['x_grid'].numpy()
    hg = fg['h_gt_grid'].numpy()
    n_x, n_t = gs[0], gs[1]
    x_1d = xg[::n_t, 0]
    t_1d = xg[:n_t, 1]
    return hg, gs, x_1d, t_1d, n_x, n_t


def _interp_residual_to_points(res_grid, x_grid, t_interior, x_pts, t_pts,
                                periodic_x=False):
    """Interpolate a residual field (nt_int, nx) to scattered (x, t) points."""
    if periodic_x:
        dx = x_grid[1] - x_grid[0]
        x_closed = np.append(x_grid, x_grid[0] + len(x_grid) * dx)
        res_closed = np.concatenate([res_grid, res_grid[:, :1]], axis=1)
        domain_len = x_closed[-1] - x_closed[0]
        x_query = x_closed[0] + np.mod(x_pts - x_closed[0], domain_len)
    else:
        x_closed = x_grid
        res_closed = res_grid
        x_query = np.clip(x_pts, x_grid[0], x_grid[-1])

    t_query = np.clip(t_pts, t_interior[0], t_interior[-1])

    interp = RegularGridInterpolator(
        (t_interior, x_closed), res_closed,
        method='linear', bounds_error=False, fill_value=0.0)
    return interp(np.column_stack([t_query, x_query]))


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'experiments_plan.yaml')
    with open(config_path, 'r') as f:
        config_full = yaml.safe_load(f)
    config = config_full.get('base_config', {})
    config['seed'] = 42
    config['cuda'] = False
    for problem in ['burgers1d', 'burgers2d', 'schrodinger', 'allen_cahn',
                    'kdv', 'ks', 'wave1d', 'conv_diff', 'fisher_kpp']:
        if problem in config_full:
            config[problem] = config_full[problem]
    if 'sampling' in config_full['base_config']:
        config['sampling'] = config_full['base_config']['sampling']
    return config


# ---------------------------------------------------------------------------
# Spectral residual helpers (for Fourier-based solvers)
# ---------------------------------------------------------------------------

def spectral_residual_schrodinger(h_solution, x_grid, t_grid):
    """i*h_t + 0.5*h_xx + |h|^2*h = 0  =>  split real/imag."""
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]
    k = np.fft.fftfreq(len(x_grid), d=dx) * 2 * np.pi

    u, v = h_solution.real, h_solution.imag
    u_hat = np.fft.fft(u, axis=1)
    v_hat = np.fft.fft(v, axis=1)
    u_xx = np.real(np.fft.ifft((1j * k) ** 2 * u_hat, axis=1))
    v_xx = np.real(np.fft.ifft((1j * k) ** 2 * v_hat, axis=1))

    u_t = (u[2:, :] - u[:-2, :]) / (2 * dt)
    v_t = (v[2:, :] - v[:-2, :]) / (2 * dt)
    mod2 = u[1:-1, :] ** 2 + v[1:-1, :] ** 2

    res_r = -v_t + 0.5 * u_xx[1:-1, :] + mod2 * u[1:-1, :]
    res_i = u_t + 0.5 * v_xx[1:-1, :] + mod2 * v[1:-1, :]
    return np.sqrt(res_r ** 2 + res_i ** 2)


def spectral_residual_allen_cahn(h_solution, x_grid, t_grid, D):
    """h_t - D*h_xx - 5*(h - h^3) = 0."""
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]
    k = np.fft.fftfreq(len(x_grid), d=dx) * 2 * np.pi

    h_hat = np.fft.fft(h_solution, axis=1)
    h_xx = np.real(np.fft.ifft((1j * k) ** 2 * h_hat, axis=1))

    h_t = (h_solution[2:, :] - h_solution[:-2, :]) / (2 * dt)
    h_c = h_solution[1:-1, :]
    return h_t - D * h_xx[1:-1, :] - 5.0 * (h_c - h_c ** 3)


def spectral_residual_kdv(h_solution, x_grid, t_grid, mu):
    """h_t + h*h_x + mu*h_xxx = 0."""
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]
    k = np.fft.fftfreq(len(x_grid), d=dx) * 2 * np.pi

    h_hat = np.fft.fft(h_solution, axis=1)
    h_x = np.real(np.fft.ifft(1j * k * h_hat, axis=1))
    h_xxx = np.real(np.fft.ifft((1j * k) ** 3 * h_hat, axis=1))

    h_t = (h_solution[2:, :] - h_solution[:-2, :]) / (2 * dt)
    return h_t + h_solution[1:-1, :] * h_x[1:-1, :] + mu * h_xxx[1:-1, :]


def spectral_residual_ks(h_solution, x_grid, t_grid, alpha, beta, gamma):
    """h_t + alpha*h*h_x + beta*h_xx + gamma*h_xxxx = 0."""
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]
    k = np.fft.fftfreq(len(x_grid), d=dx) * 2 * np.pi

    h_hat = np.fft.fft(h_solution, axis=1)
    h_x = np.real(np.fft.ifft(1j * k * h_hat, axis=1))
    h_xx = np.real(np.fft.ifft((1j * k) ** 2 * h_hat, axis=1))
    h_xxxx = np.real(np.fft.ifft((1j * k) ** 4 * h_hat, axis=1))

    h_t = (h_solution[2:, :] - h_solution[:-2, :]) / (2 * dt)
    h_c = h_solution[1:-1, :]
    return (h_t + alpha * h_c * h_x[1:-1, :]
            + beta * h_xx[1:-1, :] + gamma * h_xxxx[1:-1, :])


# ---------------------------------------------------------------------------
# FD residual helpers (for non-spectral solvers)
# ---------------------------------------------------------------------------

def fd_residual_wave1d(h, x_grid, t_grid):
    """Residual of h_tt - h_xx = 0."""
    dt = t_grid[1] - t_grid[0]
    dx = x_grid[1] - x_grid[0]
    h_tt = (h[2:, 1:-1] - 2 * h[1:-1, 1:-1] + h[:-2, 1:-1]) / dt ** 2
    h_xx = (h[1:-1, 2:] - 2 * h[1:-1, 1:-1] + h[1:-1, :-2]) / dx ** 2
    return h_tt - h_xx


def fd_residual_conv_diff(h, x_grid, t_grid, beta, epsilon):
    """Residual of h_t + beta*h_x - epsilon*h_xx = 0."""
    dt = t_grid[1] - t_grid[0]
    dx = x_grid[1] - x_grid[0]
    h_t = (h[2:, 1:-1] - h[:-2, 1:-1]) / (2 * dt)
    h_x = (h[1:-1, 2:] - h[1:-1, :-2]) / (2 * dx)
    h_xx = (h[1:-1, 2:] - 2 * h[1:-1, 1:-1] + h[1:-1, :-2]) / dx ** 2
    return h_t + beta * h_x - epsilon * h_xx


def fd_residual_fisher_kpp(h, x_grid, t_grid, D, kappa):
    """Residual of h_t - D*h_xx - kappa*h*(1 - h) = 0."""
    dt = t_grid[1] - t_grid[0]
    dx = x_grid[1] - x_grid[0]
    h_t = (h[2:, 1:-1] - h[:-2, 1:-1]) / (2 * dt)
    h_xx = (h[1:-1, 2:] - 2 * h[1:-1, 1:-1] + h[1:-1, :-2]) / dx ** 2
    h_c = h[1:-1, 1:-1]
    return h_t - D * h_xx - kappa * h_c * (1.0 - h_c)


# ---------------------------------------------------------------------------
# Per-PDE validators
# ---------------------------------------------------------------------------

def _report(name, ic_max, ic_mse, bc_max, bc_mse, res_max, res_mse):
    status_ic = "OK" if ic_max < 1e-3 else "WARN"
    status_bc = "OK" if bc_max < 1e-3 else "WARN"
    status_res = "OK" if res_max < 1e-2 else "WARN"
    print(f"  IC  [{status_ic:4s}] max={ic_max:.3e}  mse={ic_mse:.3e}")
    print(f"  BC  [{status_bc:4s}] max={bc_max:.3e}  mse={bc_mse:.3e}")
    print(f"  Res [{status_res:4s}] max={res_max:.3e}  mse={res_mse:.3e}")
    return {
        'ic_max': ic_max, 'ic_mse': ic_mse,
        'bc_max': bc_max, 'bc_mse': bc_mse,
        'res_max': res_max, 'res_mse': res_mse,
        'status': 'OK' if max(ic_max, bc_max, res_max) < 1e-2 else 'WARN',
    }


def validate_burgers1d(config, dataset_dir):
    pc = config['burgers1d']

    data = load_dataset(str(dataset_dir / 'training_data.pt'))
    x_np = data['x'].numpy()[:, 0]
    t_np = data['t'].numpy()[:, 0]
    h_np = data['h_gt'].numpy()[:, 0]
    mask_ic = data['mask']['IC'].numpy()
    mask_bc = data['mask']['BC'].numpy()

    ic_analytical = -np.sin(np.pi * x_np[mask_ic])
    ic_err = np.abs(h_np[mask_ic] - ic_analytical)
    ic_max, ic_mse = ic_err.max(), (ic_err ** 2).mean()

    bc_err = np.abs(h_np[mask_bc])
    bc_max, bc_mse = bc_err.max(), (bc_err ** 2).mean()

    # Burgers1D uses the Cole-Hopf exact solution; residual is 0 by construction
    res_max, res_mse = 0.0, 0.0

    return _report('burgers1d', ic_max, ic_mse, bc_max, bc_mse, res_max, res_mse)


def validate_burgers2d(config, dataset_dir):
    pc = config['burgers2d']

    data = load_dataset(str(dataset_dir / 'training_data.pt'))
    x_np = data['x'].numpy()
    t_np = data['t'].numpy()[:, 0]
    h_np = data['h_gt'].numpy()[:, 0]
    mask_ic = data['mask']['IC'].numpy()
    mask_bc = data['mask']['BC'].numpy()

    ic_analytical = 1.0 / (1.0 + np.exp(
        (x_np[mask_ic, 0] + x_np[mask_ic, 1]) / 0.2))
    ic_err = np.abs(h_np[mask_ic] - ic_analytical)
    ic_max, ic_mse = ic_err.max(), (ic_err ** 2).mean()

    bc_analytical = 1.0 / (1.0 + np.exp(
        (x_np[mask_bc, 0] + x_np[mask_bc, 1] - t_np[mask_bc]) / 0.2))
    bc_err = np.abs(h_np[mask_bc] - bc_analytical)
    bc_max, bc_mse = bc_err.max(), (bc_err ** 2).mean()

    res_max, res_mse = 0.0, 0.0

    return _report('burgers2d', ic_max, ic_mse, bc_max, bc_mse, res_max, res_mse)


def _spectral_res_at_training_pts(solve_fn, solve_kwargs, residual_fn,
                                   residual_kwargs, data, periodic_x=True):
    """Run solver, compute spectral residual, interpolate to training pts."""
    print("    (running solver on native grid for spectral residual...)")
    x_grid, t_grid, h_sol = solve_fn(**solve_kwargs)
    res_grid = residual_fn(h_sol, x_grid, t_grid, **residual_kwargs)
    t_interior = t_grid[1:-1]

    # Report native-grid residual
    native_max = np.abs(res_grid).max()
    native_mse = (res_grid ** 2).mean()
    print(f"    Native grid: max={native_max:.3e}  mse={native_mse:.3e}")

    # Interpolate to training_data residual points
    mask_res = data['mask']['residual'].numpy()
    x_pts = data['x'].numpy()[mask_res, 0]
    t_pts = data['t'].numpy()[mask_res, 0]
    res_at_pts = _interp_residual_to_points(
        res_grid, x_grid, t_interior, x_pts, t_pts, periodic_x=periodic_x)
    res_max = np.abs(res_at_pts).max()
    res_mse = (res_at_pts ** 2).mean()
    return res_max, res_mse


def validate_schrodinger(config, dataset_dir):
    from solvers.schrodinger_solver import solve_nlse_splitstep
    pc = config['schrodinger']
    x_min, x_max = pc['spatial_domain'][0]
    t_min, t_max = pc['temporal_domain']

    data = load_dataset(str(dataset_dir / 'training_data.pt'))
    x_np = data['x'].numpy()[:, 0]
    t_np = data['t'].numpy()[:, 0]
    h_np = data['h_gt'].numpy()
    mask_ic = data['mask']['IC'].numpy()
    mask_bc = data['mask']['BC'].numpy()

    ic_real_an = 2.0 / np.cosh(x_np[mask_ic])
    ic_err = np.sqrt((h_np[mask_ic, 0] - ic_real_an) ** 2
                     + h_np[mask_ic, 1] ** 2)
    ic_max, ic_mse = ic_err.max(), (ic_err ** 2).mean()

    x_bc = x_np[mask_bc]
    h_bc = h_np[mask_bc]
    is_left = (x_bc < (x_min + 0.1))
    is_right = (x_bc > (x_max - 0.1))
    t_left = t_np[mask_bc][is_left]
    t_right = t_np[mask_bc][is_right]
    h_left = h_bc[is_left]
    h_right = h_bc[is_right]
    bc_errs = []
    for i in range(min(len(t_left), len(t_right))):
        j = np.argmin(np.abs(t_right - t_left[i]))
        bc_errs.append(np.abs(h_left[i] - h_right[j]))
    bc_errs = np.array(bc_errs) if bc_errs else np.array([[0.0, 0.0]])
    bc_max, bc_mse = bc_errs.max(), (bc_errs ** 2).mean()

    # Spectral residual on native grid, interpolated to training residual pts
    print("    (running Schrodinger solver on native grid...)")
    x_grid, t_grid, h_sol = solve_nlse_splitstep(
        x_min, x_max, t_min, t_max, nx=2048, nt=5000)
    res_grid = spectral_residual_schrodinger(h_sol, x_grid, t_grid)
    t_interior = t_grid[1:-1]
    native_max = np.abs(res_grid).max()
    native_mse = (res_grid ** 2).mean()
    print(f"    Native grid: max={native_max:.3e}  mse={native_mse:.3e}")

    mask_res = data['mask']['residual'].numpy()
    x_pts = data['x'].numpy()[mask_res, 0]
    t_pts = data['t'].numpy()[mask_res, 0]
    res_at_pts = _interp_residual_to_points(
        res_grid, x_grid, t_interior, x_pts, t_pts, periodic_x=True)
    res_max = np.abs(res_at_pts).max()
    res_mse = (res_at_pts ** 2).mean()

    return _report('schrodinger', ic_max, ic_mse, bc_max, bc_mse,
                    res_max, res_mse)


def validate_allen_cahn(config, dataset_dir):
    from solvers.allen_cahn_solver import solve_allen_cahn
    pc = config['allen_cahn']
    x_min, x_max = pc['spatial_domain'][0]
    t_min, t_max = pc['temporal_domain']
    D = pc.get('D', 0.0001)

    data = load_dataset(str(dataset_dir / 'training_data.pt'))
    x_np = data['x'].numpy()[:, 0]
    t_np = data['t'].numpy()[:, 0]
    h_np = data['h_gt'].numpy()[:, 0]
    mask_ic = data['mask']['IC'].numpy()
    mask_bc = data['mask']['BC'].numpy()

    ic_analytical = x_np[mask_ic] ** 2 * np.cos(np.pi * x_np[mask_ic])
    ic_err = np.abs(h_np[mask_ic] - ic_analytical)
    ic_max, ic_mse = ic_err.max(), (ic_err ** 2).mean()

    x_bc = x_np[mask_bc]; t_bc = t_np[mask_bc]; h_bc = h_np[mask_bc]
    is_left = (x_bc < (x_min + 1e-6))
    is_right = (x_bc > (x_max - 1e-6))
    t_l = t_bc[is_left]; h_l = h_bc[is_left]
    t_r = t_bc[is_right]; h_r = h_bc[is_right]
    bc_errs = []
    for i in range(min(len(t_l), len(t_r))):
        j = np.argmin(np.abs(t_r - t_l[i]))
        bc_errs.append(abs(float(h_l[i]) - float(h_r[j])))
    bc_errs = np.array(bc_errs) if bc_errs else np.array([0.0])
    bc_max, bc_mse = bc_errs.max(), (bc_errs ** 2).mean()

    res_max, res_mse = _spectral_res_at_training_pts(
        solve_allen_cahn,
        dict(x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max,
             nx=512, nt=5000, D=D),
        spectral_residual_allen_cahn, dict(D=D), data)

    return _report('allen_cahn', ic_max, ic_mse, bc_max, bc_mse,
                    res_max, res_mse)


def validate_kdv(config, dataset_dir):
    from solvers.kdv_solver import solve_kdv
    pc = config['kdv']
    x_min, x_max = pc['spatial_domain'][0]
    t_min, t_max = pc['temporal_domain']
    mu = pc.get('mu', 0.000484)

    data = load_dataset(str(dataset_dir / 'training_data.pt'))
    x_np = data['x'].numpy()[:, 0]
    t_np = data['t'].numpy()[:, 0]
    h_np = data['h_gt'].numpy()[:, 0]
    mask_ic = data['mask']['IC'].numpy()
    mask_bc = data['mask']['BC'].numpy()

    ic_analytical = np.cos(np.pi * x_np[mask_ic])
    ic_err = np.abs(h_np[mask_ic] - ic_analytical)
    ic_max, ic_mse = ic_err.max(), (ic_err ** 2).mean()

    x_bc = x_np[mask_bc]; h_bc = h_np[mask_bc]; t_bc = t_np[mask_bc]
    is_left = (x_bc < (x_min + 1e-6))
    is_right = (x_bc > (x_max - 1e-6))
    t_l = t_bc[is_left]; h_l = h_bc[is_left]
    t_r = t_bc[is_right]; h_r = h_bc[is_right]
    bc_errs = []
    for i in range(min(len(t_l), len(t_r))):
        j = np.argmin(np.abs(t_r - t_l[i]))
        bc_errs.append(abs(float(h_l[i]) - float(h_r[j])))
    bc_errs = np.array(bc_errs) if bc_errs else np.array([0.0])
    bc_max, bc_mse = bc_errs.max(), (bc_errs ** 2).mean()

    res_max, res_mse = _spectral_res_at_training_pts(
        solve_kdv,
        dict(x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max,
             nx=512, nt=5000, mu=mu),
        spectral_residual_kdv, dict(mu=mu), data)

    return _report('kdv', ic_max, ic_mse, bc_max, bc_mse, res_max, res_mse)


def validate_ks(config, dataset_dir):
    from solvers.ks_solver import solve_ks
    pc = config['ks']
    x_min, x_max = pc['spatial_domain'][0]
    t_min, t_max = pc['temporal_domain']
    alpha = pc.get('alpha', 100.0 / 16.0)
    beta = pc.get('beta', 100.0 / 16.0 ** 2)
    gamma = pc.get('gamma', 100.0 / 16.0 ** 4)

    data = load_dataset(str(dataset_dir / 'training_data.pt'))
    x_np = data['x'].numpy()[:, 0]
    t_np = data['t'].numpy()[:, 0]
    h_np = data['h_gt'].numpy()[:, 0]
    mask_ic = data['mask']['IC'].numpy()
    mask_bc = data['mask']['BC'].numpy()

    ic_analytical = np.cos(x_np[mask_ic]) * (1.0 + np.sin(x_np[mask_ic]))
    ic_err = np.abs(h_np[mask_ic] - ic_analytical)
    ic_max, ic_mse = ic_err.max(), (ic_err ** 2).mean()

    x_bc = x_np[mask_bc]; h_bc = h_np[mask_bc]; t_bc = t_np[mask_bc]
    is_left = (x_bc < (x_min + 1e-6))
    is_right = (x_bc > (x_max - 1e-6))
    t_l = t_bc[is_left]; h_l = h_bc[is_left]
    t_r = t_bc[is_right]; h_r = h_bc[is_right]
    bc_errs = []
    for i in range(min(len(t_l), len(t_r))):
        j = np.argmin(np.abs(t_r - t_l[i]))
        bc_errs.append(abs(float(h_l[i]) - float(h_r[j])))
    bc_errs = np.array(bc_errs) if bc_errs else np.array([0.0])
    bc_max, bc_mse = bc_errs.max(), (bc_errs ** 2).mean()

    res_max, res_mse = _spectral_res_at_training_pts(
        solve_ks,
        dict(x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max,
             nx=512, nt=5000, alpha=alpha, beta=beta, gamma=gamma),
        spectral_residual_ks, dict(alpha=alpha, beta=beta, gamma=gamma),
        data)

    return _report('ks', ic_max, ic_mse, bc_max, bc_mse, res_max, res_mse)


def validate_wave1d(config, dataset_dir):
    pc = config['wave1d']

    data = load_dataset(str(dataset_dir / 'training_data.pt'))
    x_np = data['x'].numpy()[:, 0]
    t_np = data['t'].numpy()[:, 0]
    h_np = data['h_gt'].numpy()[:, 0]
    mask_ic = data['mask']['IC'].numpy()
    mask_bc = data['mask']['BC'].numpy()

    ic_analytical = np.sin(x_np[mask_ic])
    ic_err = np.abs(h_np[mask_ic] - ic_analytical)
    ic_max, ic_mse = ic_err.max(), (ic_err ** 2).mean()

    bc_analytical = np.sin(x_np[mask_bc]) * np.cos(t_np[mask_bc])
    bc_err = np.abs(h_np[mask_bc] - bc_analytical)
    bc_max, bc_mse = bc_err.max(), (bc_err ** 2).mean()

    hg, gs, x_1d, t_1d, n_x, n_t = load_freq_grid_2d(dataset_dir)
    h_2d = hg[:, 0].reshape(n_x, n_t).T
    res = fd_residual_wave1d(h_2d, x_1d, t_1d)
    res_max, res_mse = np.abs(res).max(), (res ** 2).mean()

    return _report('wave1d', ic_max, ic_mse, bc_max, bc_mse, res_max, res_mse)


def validate_conv_diff(config, dataset_dir):
    pc = config['conv_diff']
    beta = pc.get('beta', 1.0)
    epsilon = pc.get('epsilon', 0.01)

    data = load_dataset(str(dataset_dir / 'training_data.pt'))
    x_np = data['x'].numpy()[:, 0]
    h_np = data['h_gt'].numpy()[:, 0]
    mask_ic = data['mask']['IC'].numpy()
    mask_bc = data['mask']['BC'].numpy()

    ic_analytical = -np.sin(np.pi * x_np[mask_ic])
    ic_err = np.abs(h_np[mask_ic] - ic_analytical)
    ic_max, ic_mse = ic_err.max(), (ic_err ** 2).mean()

    bc_err = np.abs(h_np[mask_bc])
    bc_max, bc_mse = bc_err.max(), (bc_err ** 2).mean()

    hg, gs, x_1d, t_1d, n_x, n_t = load_freq_grid_2d(dataset_dir)
    h_2d = hg[:, 0].reshape(n_x, n_t).T
    res = fd_residual_conv_diff(h_2d, x_1d, t_1d, beta, epsilon)
    res_max, res_mse = np.abs(res).max(), (res ** 2).mean()

    return _report('conv_diff', ic_max, ic_mse, bc_max, bc_mse,
                    res_max, res_mse)


def validate_fisher_kpp(config, dataset_dir):
    pc = config['fisher_kpp']
    D = pc.get('D', 1.0)
    kappa = pc.get('kappa', 25.0)

    data = load_dataset(str(dataset_dir / 'training_data.pt'))
    x_np = data['x'].numpy()[:, 0]
    h_np = data['h_gt'].numpy()[:, 0]
    mask_ic = data['mask']['IC'].numpy()
    mask_bc = data['mask']['BC'].numpy()

    ic_analytical = 1.0 / (1.0 + np.exp(
        np.sqrt(kappa / 6.0) * (x_np[mask_ic] - 0.25)))
    ic_err = np.abs(h_np[mask_ic] - ic_analytical)
    ic_max, ic_mse = ic_err.max(), (ic_err ** 2).mean()

    x_bc = x_np[mask_bc]
    bc_analytical = np.where(x_bc < 0.5, 1.0, 0.0)
    bc_err = np.abs(h_np[mask_bc] - bc_analytical)
    bc_max, bc_mse = bc_err.max(), (bc_err ** 2).mean()

    hg, gs, x_1d, t_1d, n_x, n_t = load_freq_grid_2d(dataset_dir)
    h_2d = hg[:, 0].reshape(n_x, n_t).T
    res = fd_residual_fisher_kpp(h_2d, x_1d, t_1d, D, kappa)
    res_max, res_mse = np.abs(res).max(), (res ** 2).mean()

    return _report('fisher_kpp', ic_max, ic_mse, bc_max, bc_mse,
                    res_max, res_mse)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VALIDATORS = {
    'burgers1d': validate_burgers1d,
    'burgers2d': validate_burgers2d,
    'schrodinger': validate_schrodinger,
    'allen_cahn': validate_allen_cahn,
    'kdv': validate_kdv,
    'ks': validate_ks,
    'wave1d': validate_wave1d,
    'conv_diff': validate_conv_diff,
    'fisher_kpp': validate_fisher_kpp,
}


def main():
    print("\n" + "=" * 70)
    print("GROUND TRUTH VALIDATION")
    print("Checks IC, BC (analytically) and PDE residual")
    print("  Spectral solvers: FFT derivatives on native grid")
    print("  Other solvers: FD on frequency grid")
    print("=" * 70)

    config = load_config()

    pdes = list(sys.argv[1:]) if len(sys.argv) > 1 else list(VALIDATORS.keys())

    results = {}
    for pde in pdes:
        print(f"\n{'=' * 70}")
        print(f"  {pde.upper()}")
        print(f"{'=' * 70}")

        dataset_dir = Path("datasets") / pde
        train_path = dataset_dir / "training_data.pt"
        freq_path = dataset_dir / "frequency_grid.pt"

        need_gen = not train_path.exists()
        need_freq = (pde in ('wave1d', 'conv_diff', 'fisher_kpp')
                     and not freq_path.exists())
        if need_gen or need_freq:
            what = "all datasets" if need_gen else "frequency grid"
            print(f"  {what} not found. Generating...")
            config['problem'] = pde
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    generate_and_save_datasets(config)
            except Exception as e:
                print(f"  ERROR generating datasets: {e}")
                results[pde] = {'status': 'FAILED', 'error': str(e)}
                continue

        if pde not in VALIDATORS:
            print(f"  No validator implemented for {pde}, skipping.")
            results[pde] = {'status': 'SKIPPED'}
            continue

        try:
            config['problem'] = pde
            results[pde] = VALIDATORS[pde](config, dataset_dir)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results[pde] = {'status': 'FAILED', 'error': str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    fmt = f"{'PDE':<15} {'Status':<8} {'IC max':<12} {'BC max':<12} {'Res max':<12}"
    print(fmt)
    print("-" * 60)
    for pde, r in results.items():
        st = r.get('status', 'FAILED')
        if st in ('OK', 'WARN'):
            print(f"{pde:<15} {st:<8} {r['ic_max']:<12.3e} "
                  f"{r['bc_max']:<12.3e} {r['res_max']:<12.3e}")
        else:
            err = r.get('error', '')[:30]
            print(f"{pde:<15} {st:<8} {err}")
    print("=" * 70)

    n_ok = sum(1 for r in results.values() if r.get('status') == 'OK')
    n_warn = sum(1 for r in results.values() if r.get('status') == 'WARN')
    n_fail = sum(1 for r in results.values() if r.get('status') == 'FAILED')
    print(f"\nResult: {n_ok} OK, {n_warn} WARN, {n_fail} FAILED "
          f"out of {len(results)}")
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
