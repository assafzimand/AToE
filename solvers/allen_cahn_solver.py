"""
Allen-Cahn Equation Solver using Fourier Pseudo-Spectral + ETDRK4.

Solves: h_t = D * h_xx + 5*(h - h^3)
Domain: x in [-1, 1], t in [0, 1]
Initial Condition: h(x, 0) = x^2 * cos(pi * x)
Boundary Conditions: Periodic: h(-1,t) = h(1,t), h_x(-1,t) = h_x(1,t)
Parameters: D = 0.0001 (standard Raissi/PirateNet benchmark)

Uses Fourier pseudo-spectral method for spatial discretization and
ETDRK4 (Exponential Time Differencing RK4) for time integration.
ETDRK4 handles the stiff linear diffusion term exactly in Fourier space.
Reference: Kassam & Trefethen, SIAM J. Sci. Comput. 26(4), 2005.
"""

import numpy as np
import torch
from typing import Tuple, Dict


def solve_allen_cahn(
    x_min: float = -1.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    nx: int = 512,
    nt: int = 500,
    D: float = 0.0001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the Allen-Cahn equation using Fourier pseudo-spectral + ETDRK4.

    Equation: h_t = D * h_xx + 5*(h - h^3)
    Rewritten: h_t = D*h_xx + 5*h - 5*h^3
    In Fourier space: dv/dt = Lk*v + N_hat(v)
      where Lk = -D*k^2 (linear diffusion, treated exactly)
      and N_hat = FFT(5*h - 5*h^3) (nonlinear reaction, stepped explicitly)
    """
    domain_len = x_max - x_min
    dx = domain_len / nx
    x_grid = np.linspace(x_min, x_max - dx, nx, dtype=np.float64)
    t_grid = np.linspace(t_min, t_max, nt, dtype=np.float64)
    dt_save = t_grid[1] - t_grid[0] if nt > 1 else (t_max - t_min)

    k = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi

    # Initial condition: h(x, 0) = x^2 * cos(pi*x)
    h0 = x_grid ** 2 * np.cos(np.pi * x_grid)
    v = np.fft.fft(h0)

    h_solution = np.zeros((nt, nx), dtype=np.float64)
    h_solution[0, :] = h0.copy()

    # Linear operator in Fourier space: Lk = -D*k^2
    # From: h_t = D*h_xx + ... => F[h_xx] = -k^2 * v => D*F[h_xx] = -D*k^2 * v
    Lk = -D * k ** 2

    # Adaptive sub-stepping based on stiffness
    # 1. Linear diffusion stability: max|Lk| = D*k_max^2
    Lk_max = np.max(np.abs(Lk))
    dt_diffusion = 1.0 / (Lk_max + 1e-10)
    
    # 2. Nonlinear reaction stability: derivative of 5*(h - h^3) is 5*(1 - 3*h^2)
    # Maximum occurs at h=+/-1: |d/dh[5(h-h^3)]| = 10 at h=+/-1
    # Conservative estimate: dt < 1 / max_reaction_rate
    max_reaction_rate = 10.0  # From 5*(1 - 3*h^2) at h=+/-1
    dt_reaction = 1.0 / max_reaction_rate
    
    dt_safe = min(dt_diffusion, dt_reaction)
    n_sub = max(int(np.ceil(dt_save / dt_safe)), 1)
    dt = dt_save / n_sub

    # ETDRK4 coefficients via contour integrals (Kassam & Trefethen 2005)
    E = np.exp(Lk * dt)
    E2 = np.exp(Lk * dt / 2.0)

    M = 64
    r = np.exp(2j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * Lk[:, np.newaxis] + r[np.newaxis, :]

    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean(
        (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3, axis=1))
    f2 = dt * np.real(np.mean(
        (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3, axis=1))
    f3 = dt * np.real(np.mean(
        (-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3, axis=1))

    def N_hat(v_hat):
        """Nonlinear term in Fourier space: FFT(5*h - 5*h^3)."""
        h_phys = np.real(np.fft.ifft(v_hat))
        reaction = 5.0 * h_phys - 5.0 * h_phys ** 3
        return np.fft.fft(reaction)

    print(f"  Solving Allen-Cahn with ETDRK4 ({nx} modes, {nt} save points, {n_sub} sub-steps/save)...")
    for save_idx in range(1, nt):
        for _ in range(n_sub):
            Nv = N_hat(v)
            a = E2 * v + Q * Nv
            Na = N_hat(a)
            b = E2 * v + Q * Na
            Nb = N_hat(b)
            c = E2 * a + Q * (2.0 * Nb - Nv)
            Nc = N_hat(c)
            v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3

        h_solution[save_idx, :] = np.real(np.fft.ifft(v))

    return x_grid, t_grid, h_solution


_cached_solution = None
_cached_config_hash = None


def _get_solution_cached(config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get cached Allen-Cahn solution grid.
    
    Returns:
        (x_grid, t_grid, h_solution): Native grid arrays from the solver
    """
    global _cached_solution, _cached_config_hash
    problem = config.get('problem', 'allen_cahn')
    pc = config[problem]
    config_tuple = (
        tuple(pc['spatial_domain'][0]),
        tuple(pc['temporal_domain']),
        pc['D'],
    )
    if _cached_solution is None or _cached_config_hash != config_tuple:
        print("  Generating Allen-Cahn solution (512x500 grid, ETDRK4)...")
        x_min, x_max = pc['spatial_domain'][0]
        t_min, t_max = pc['temporal_domain']
        D = pc['D']
        
        x_grid, t_grid, h_sol = solve_allen_cahn(
            x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max,
            nx=512, nt=500, D=D,
        )
        _cached_solution = (x_grid, t_grid, h_sol)
        _cached_config_hash = config_tuple
        print("  Solution computed.")
    return _cached_solution


def _get_interpolator(config: Dict):
    """Return callable interp(x_flat, t_flat) -> values for ground truth."""
    from scipy.interpolate import RegularGridInterpolator
    x_grid, t_grid, h_solution = _get_solution_cached(config)
    rgi = RegularGridInterpolator(
        (t_grid, x_grid), h_solution, method='linear',
        bounds_error=False, fill_value=None,
    )

    def _interp(x_flat, t_flat):
        return rgi(np.column_stack([t_flat, x_flat]))

    return _interp


def generate_dataset(
    n_residual: int, n_ic: int, n_bc: int,
    device: torch.device, config: Dict,
) -> Dict[str, torch.Tensor]:
    """Generate dataset with Allen-Cahn ground truth sampled from solver grid."""
    seed = config['seed']
    problem = config.get('problem', 'allen_cahn')
    pc = config[problem]
    spatial_dim = pc['spatial_dim']
    x_min, x_max = pc['spatial_domain'][0]
    t_min, t_max = pc['temporal_domain']

    # Get solver grid
    x_grid, t_grid, h_solution = _get_solution_cached(config)
    nx, nt = len(x_grid), len(t_grid)
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = n_residual + n_ic + n_bc
    x = torch.zeros(N, spatial_dim, device=device)
    t = torch.zeros(N, 1, device=device)
    h_gt = torch.zeros(N, 1, device=device, dtype=torch.float32)
    idx = 0

    # Residual: sample random grid indices
    print(f"  Sampling {n_residual} residual points from grid...")
    i_t = np.random.choice(nt, size=n_residual, replace=True)
    i_x = np.random.choice(nx, size=n_residual, replace=True)
    x[idx:idx + n_residual, 0] = torch.from_numpy(x_grid[i_x].astype(np.float32)).to(device)
    t[idx:idx + n_residual, 0] = torch.from_numpy(t_grid[i_t].astype(np.float32)).to(device)
    h_gt[idx:idx + n_residual, 0] = torch.from_numpy(h_solution[i_t, i_x].astype(np.float32)).to(device)
    idx += n_residual

    # IC: sample random x from grid, t=t_min
    print(f"  Sampling {n_ic} initial condition points from grid...")
    i_x_ic = np.random.choice(nx, size=n_ic, replace=True)
    x[idx:idx + n_ic, 0] = torch.from_numpy(x_grid[i_x_ic].astype(np.float32)).to(device)
    t[idx:idx + n_ic, 0] = t_min
    idx += n_ic

    # BC: paired points at x=x_min and x=x_max, sample random t from grid
    print(f"  Sampling {n_bc} boundary condition points from grid...")
    n_bc_left = n_bc // 2
    n_bc_right = n_bc - n_bc_left
    i_t_bc = np.random.choice(nt, size=max(n_bc_left, n_bc_right), replace=True)
    
    x[idx:idx + n_bc_left, 0] = x_min
    t[idx:idx + n_bc_left, 0] = torch.from_numpy(t_grid[i_t_bc[:n_bc_left]].astype(np.float32)).to(device)
    h_gt[idx:idx + n_bc_left, 0] = torch.from_numpy(h_solution[i_t_bc[:n_bc_left], 0].astype(np.float32)).to(device)
    idx += n_bc_left

    x[idx:idx + n_bc_right, 0] = x_max
    t[idx:idx + n_bc_right, 0] = torch.from_numpy(t_grid[i_t_bc[:n_bc_right]].astype(np.float32)).to(device)
    # Periodic: h(x_max, t) = h(x_min, t); x_max not in half-open grid, use index 0
    h_gt[idx:idx + n_bc_right, 0] = torch.from_numpy(h_solution[i_t_bc[:n_bc_right], 0].astype(np.float32)).to(device)

    # Masks
    mask_res = torch.zeros(N, dtype=torch.bool, device=device)
    mask_res[:n_residual] = True
    mask_ic = torch.zeros(N, dtype=torch.bool, device=device)
    mask_ic[n_residual:n_residual + n_ic] = True
    mask_bc = torch.zeros(N, dtype=torch.bool, device=device)
    mask_bc[n_residual + n_ic:] = True

    # Overwrite IC with exact analytical values
    h_gt[mask_ic, 0] = (x[mask_ic, 0] ** 2 * torch.cos(np.pi * x[mask_ic, 0])).float()

    print("  Dataset generated successfully")
    return {
        "x": x, "t": t, "h_gt": h_gt,
        "mask": {"residual": mask_res, "IC": mask_ic, "BC": mask_bc},
    }


def evaluate_on_grid(x_grid: torch.Tensor, config: Dict) -> torch.Tensor:
    """Evaluate ground truth on a regular grid for frequency analysis.
    
    Note: This returns values from the solver's native grid. If x_grid points
    don't align with the native grid, this will return the nearest grid value.
    """
    x_grid_np, t_grid_np, h_solution = _get_solution_cached(config)
    
    # For each point in x_grid, find nearest grid point
    x_query = x_grid.cpu().numpy()[:, 0]
    t_query = x_grid.cpu().numpy()[:, 1]
    
    # Find nearest indices
    i_x = np.searchsorted(x_grid_np, x_query)
    i_x = np.clip(i_x, 0, len(x_grid_np) - 1)
    
    i_t = np.searchsorted(t_grid_np, t_query)
    i_t = np.clip(i_t, 0, len(t_grid_np) - 1)
    
    h = h_solution[i_t, i_x]
    return torch.from_numpy(h.reshape(-1, 1).astype(np.float32))
