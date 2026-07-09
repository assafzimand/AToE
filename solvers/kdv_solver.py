"""
Korteweg-de Vries (KdV) Equation Solver using Pseudo-Spectral + ETDRK4.

Solves: h_t + h * h_x + mu^2 * h_xxx = 0
Domain: x in [-1, 1], t in [0, 1]
Initial Condition: h(x, 0) = cos(pi*x)
Boundary Conditions: Periodic
Parameters: mu = 0.022 (Zabusky & Kruskal 1965 dispersion coefficient; code uses mu^2 = 0.000484)

Uses Fourier pseudo-spectral method for spatial discretization and
ETDRK4 (Exponential Time Differencing RK4) for time integration.
ETDRK4 handles the stiff linear dispersive term exactly in Fourier space,
allowing much larger timesteps than explicit RK4.
Reference: Kassam & Trefethen, SIAM J. Sci. Comput. 26(4), 2005.
"""

import numpy as np
import torch
from typing import Tuple, Dict


def initial_condition(x: torch.Tensor) -> torch.Tensor:
    """Exact IC h(x, 0) = cos(pi*x). Also the whole-domain target of the
    PirateNets physics-informed output init (u(x,t) ≈ u0(x) for all t)."""
    return torch.cos(np.pi * x[:, :1]).float()


def solve_kdv(
    x_min: float = -1.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    nx: int = 256,
    nt: int = 201,
    mu: float = 0.022,
    n_sub_factor: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the KdV equation using Fourier pseudo-spectral + ETDRK4.

    Equation: h_t + h * h_x + mu^2 * h_xxx = 0
    Rewritten in Fourier space: dv/dt = Lk*v + N_hat(v)
      where Lk = i*mu^2*k^3 (stiff dispersive part, handled exactly)
      and N_hat = FFT(-u*u_x) (nonlinear part, stepped explicitly, 2/3-dealiased)

    n_sub_factor multiplies the number of substeps per saved frame — used by
    the self-convergence check in _get_solution_cached (compare factor 1 vs 2).
    """
    domain_len = x_max - x_min
    dx = domain_len / nx
    x_grid = np.linspace(x_min, x_max - dx, nx, dtype=np.float64)
    t_grid = np.linspace(t_min, t_max, nt, dtype=np.float64)
    dt_save = t_grid[1] - t_grid[0] if nt > 1 else (t_max - t_min)

    k = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi

    u0 = np.cos(np.pi * x_grid)
    v = np.fft.fft(u0)

    u_solution = np.zeros((nt, nx), dtype=np.float64)
    u_solution[0, :] = u0.copy()

    # Linear operator in Fourier space: Lk = i*mu^2*k^3
    # Derived from: h_t = -h*h_x - mu^2*h_xxx
    # F[h_xxx] = (ik)^3 * v = -ik^3 * v, so -mu^2*F[h_xxx] = i*mu^2*k^3 * v
    Lk = 1j * mu**2 * k ** 3

    # Timestep: only limited by nonlinear CFL (ETDRK4 handles linear part exactly)
    k_max = np.max(np.abs(k))
    dt_nonlinear = 0.4 / (k_max + 1e-10)
    n_sub = max(int(np.ceil(dt_save / dt_nonlinear)), 1) * max(int(n_sub_factor), 1)
    dt = dt_save / n_sub

    # ETDRK4 coefficients via contour integrals (Kassam & Trefethen 2005).
    # KdV's Lk is purely IMAGINARY, so the coefficients are genuinely complex
    # (Trefethen p27.m keeps complex means). Taking real() here — correct only
    # for real operators like KS/kursiv.m — degrades the scheme to 1st order
    # and produced a reference with ~16% rel-L2 error at these settings.
    E = np.exp(Lk * dt)
    E2 = np.exp(Lk * dt / 2.0)

    M = 64
    r = np.exp(2j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * Lk[:, np.newaxis] + r[np.newaxis, :]

    Q = dt * np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1)
    f1 = dt * np.mean(
        (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3, axis=1)
    f2 = dt * np.mean(
        (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3, axis=1)
    f3 = dt * np.mean(
        (-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3, axis=1)

    # 2/3-rule dealiasing mask for the quadratic nonlinearity (cheap insurance;
    # at nx=512 the spectrum is well inside the mask so it changes ~nothing)
    dealias_mask = (np.abs(k) <= (2.0 / 3.0) * k_max).astype(np.float64)

    def N_hat(v_hat):
        """Nonlinear term in Fourier space: FFT(-u * u_x), 2/3-dealiased."""
        u_phys = np.real(np.fft.ifft(v_hat))
        u_x = np.real(np.fft.ifft(1j * k * v_hat))
        return dealias_mask * np.fft.fft(-u_phys * u_x)

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

        u_solution[save_idx, :] = np.real(np.fft.ifft(v))

    return x_grid, t_grid, u_solution


_cached_solution = None
_cached_config_hash = None


def _get_solution_cached(config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get cached KdV solution grid (in-memory memo + on-disk .npz cache).

    On first generation, runs a self-convergence check (same solve with 2x
    substeps) so a silently inaccurate reference can never ship again; the
    estimated error is stored in the cache file and logged.

    Returns:
        (x_grid, t_grid, h_solution): Native grid arrays from the solver
    """
    global _cached_solution, _cached_config_hash
    problem = config.get('problem', 'kdv')
    pc = config[problem]

    # Time marching narrows the config's temporal_domain per window, but the
    # numerical solution must always start from t=0 with the true IC (this
    # solver hardcodes u0 = cos(pi*x) at its t_min). Solve the ORIGINAL full
    # domain once and let callers slice their window's time range from it —
    # otherwise windows >= 1 would be scored against a solution restarted
    # from the t=0 IC at t_start. (Same handling as ks_solver.)
    tm_window = config.get('_time_marching_window', {})
    original_td = tm_window.get('original_temporal_domain')
    if original_td is not None:
        t_min, t_max = original_td
    else:
        t_min, t_max = pc['temporal_domain']

    config_tuple = (
        tuple(pc['spatial_domain'][0]),
        (t_min, t_max),
        pc['mu'],
    )
    if _cached_solution is None or _cached_config_hash != config_tuple:
        x_min, x_max = pc['spatial_domain'][0]
        mu = pc['mu']
        nx, nt = 512, 500

        from pathlib import Path
        cache_dir = Path(__file__).resolve().parent.parent / 'datasets' / 'gt_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        # v2: complex ETDRK4 coefficients + 2/3 dealiasing (v1 had the
        # real()-coefficient bug and ~16% error — never load such caches)
        cache_file = cache_dir / (
            f"kdv_gt_v2_{x_min}_{x_max}_{t_min}_{t_max}_mu{mu}_{nx}x{nt}.npz")

        if cache_file.exists():
            data = np.load(cache_file)
            _cached_solution = (data['x_grid'], data['t_grid'], data['h_sol'])
            _cached_config_hash = config_tuple
            print(f"  Loaded KdV solution from cache ({cache_file.name}, "
                  f"self-convergence rel-L2 = {float(data['conv_rel_l2']):.3e})")
            return _cached_solution

        print(f"  Generating KdV solution ({nx}x{nt} grid, ETDRK4)...")
        x_grid, t_grid, h_sol = solve_kdv(
            x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max,
            nx=nx, nt=nt, mu=mu,
        )
        # Self-convergence check: re-solve with 2x substeps; ETDRK4 is 4th
        # order, so this diff is a tight upper bound on the solution's error.
        _, _, h_fine = solve_kdv(
            x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max,
            nx=nx, nt=nt, mu=mu, n_sub_factor=2,
        )
        conv_rel_l2 = float(np.linalg.norm(h_sol - h_fine)
                            / (np.linalg.norm(h_fine) + 1e-300))
        print(f"  Solution computed (self-convergence rel-L2 = {conv_rel_l2:.3e})")
        if conv_rel_l2 > 1e-6:
            print(f"  WARNING: KdV reference self-convergence {conv_rel_l2:.3e} "
                  f"> 1e-6 — rel-L2 metrics below this level are unreliable.")

        np.savez_compressed(cache_file, x_grid=x_grid, t_grid=t_grid,
                            h_sol=h_sol, conv_rel_l2=conv_rel_l2)
        _cached_solution = (x_grid, t_grid, h_sol)
        _cached_config_hash = config_tuple
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
    """Generate dataset with KdV ground truth sampled from solver grid.

    BC points are sampled at x=x_min and x=x_max for periodic enforcement:
    h(x_min, t) = h(x_max, t) and h_x(x_min, t) = h_x(x_max, t).
    """
    seed = config['seed']
    problem = config.get('problem', 'kdv')
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

    mask_res = torch.zeros(N, dtype=torch.bool, device=device)
    mask_res[:n_residual] = True
    mask_ic = torch.zeros(N, dtype=torch.bool, device=device)
    mask_ic[n_residual:n_residual + n_ic] = True
    mask_bc = torch.zeros(N, dtype=torch.bool, device=device)
    mask_bc[n_residual + n_ic:] = True

    # Overwrite IC with exact analytical values
    h_gt[mask_ic, 0] = torch.cos(np.pi * x[mask_ic, 0]).float()

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
