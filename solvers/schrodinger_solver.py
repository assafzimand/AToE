"""
Schrodinger Equation Solver using Split-Step Fourier Method.

Solves: i*h_t + 0.5*h_xx + |h|^2*h = 0
Domain: x in [-5, 5], t in [0, pi/2]
Initial Condition: h(x, 0) = 2*sech(x)
Boundary Conditions: Periodic
"""

import numpy as np
import torch
from typing import Dict, Tuple


def initial_condition_analytical(x: np.ndarray) -> np.ndarray:
    """
    Analytical initial condition for NLSE: h(x, 0) = 2*sech(x).
    
    Args:
        x: Spatial coordinates (numpy array)
        
    Returns:
        h0: Complex initial condition (real-valued soliton)
    """
    h0 = 2.0 / np.cosh(x)
    return h0.astype(np.complex128)


def initial_condition(x: torch.Tensor) -> torch.Tensor:
    """Exact IC h(x, 0) = 2*sech(x) (real part; imaginary part 0). Also the
    whole-domain target of the PirateNets physics-informed output init
    (u(x,t) ≈ u0(x) for all t). Returns (N, 2) = [Re, Im]."""
    real = (2.0 / torch.cosh(x[:, :1])).float()
    return torch.cat([real, torch.zeros_like(real)], dim=1)


def solve_nlse_splitstep(
    x_min: float = -5.0,
    x_max: float = 5.0,
    t_min: float = 0.0,
    t_max: float = np.pi / 2,
    nx: int = 1024,
    nt: int = 800,
    n_sub: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the Nonlinear Schrodinger Equation using split-step Fourier method.

    Equation: i*h_t + 0.5*h_xx + |h|^2*h = 0

    The split-step (Strang) method alternates between:
    - Linear step (dispersion): i*h_t + 0.5*h_xx = 0 (solved in Fourier space)
    - Nonlinear step: i*h_t + |h|^2*h = 0 (solved in real space)

    Strang splitting is 2nd order in the step size, so the integrator takes
    n_sub substeps per saved frame. With dt tied to the save grid (the old
    n_sub=1 behavior) the 2048x1000 solution was off by 1.4e-4 vs an
    adaptive-integrator gold run; n_sub=16 puts it near 5e-7.

    Args:
        x_min: Minimum spatial coordinate
        x_max: Maximum spatial coordinate
        t_min: Initial time (typically 0)
        t_max: Final time (typically pi/2)
        nx: Number of spatial grid points (use power of 2 for FFT efficiency)
        nt: Number of temporal grid points (saved frames)
        n_sub: Strang substeps per saved frame

    Returns:
        x_grid: Spatial grid (nx,)
        t_grid: Temporal grid (nt,)
        h_solution: Complex solution field (nt, nx)
    """
    # Create spatial grid (periodic: half-open interval [x_min, x_max))
    domain_len = x_max - x_min
    dx = domain_len / nx
    x_grid = np.linspace(x_min, x_max - dx, nx, dtype=np.float64)

    # Create temporal grid
    t_grid = np.linspace(t_min, t_max, nt, dtype=np.float64)
    dt = (t_grid[1] - t_grid[0]) / max(int(n_sub), 1)

    # Create wavenumber grid for Fourier space (periodic BC)
    k = 2.0 * np.pi * np.fft.fftfreq(nx, dx)

    # Initialize solution array
    h_solution = np.zeros((nt, nx), dtype=np.complex128)

    # Set initial condition
    h_solution[0, :] = initial_condition_analytical(x_grid)

    # Precompute linear evolution operator in Fourier space
    # Linear step: exp(-i * 0.5 * k^2 * dt)
    # From: i*h_t + 0.5*h_xx = 0 => h_t = i*0.5*h_xx
    # In Fourier space: h_t = -i*0.5*k^2*h_k
    # Solution: h_k(t+dt) = h_k(t) * exp(-i*0.5*k^2*dt)
    linear_operator = np.exp(-0.5j * k**2 * dt)

    # Time integration using split-step (Strang) method
    h = h_solution[0, :].copy()

    for n in range(nt - 1):
        for _ in range(max(int(n_sub), 1)):
            # Half-step nonlinear evolution in real space
            # From: i*h_t + |h|^2*h = 0 => h_t = i*|h|^2*h
            # Solution: h(t+dt/2) = h(t) * exp(i*|h|^2*dt/2)
            h = h * np.exp(1j * np.abs(h)**2 * (dt / 2))

            # Full-step linear evolution in Fourier space
            h = np.fft.ifft(np.fft.fft(h) * linear_operator)

            # Half-step nonlinear evolution in real space
            h = h * np.exp(1j * np.abs(h)**2 * (dt / 2))

        # Store solution
        h_solution[n + 1, :] = h

    return x_grid, t_grid, h_solution


# Global solution cache (initialized on first dataset generation)
_cached_solution = None
_cached_config_hash = None


def _get_solution_cached(config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get cached NLSE solution grid (in-memory memo + on-disk .npz cache).

    The memo is keyed on the domain config (the old version cached
    unconditionally, so a changed domain silently reused the stale solution).
    On first generation runs a self-convergence check (2x substeps; Strang is
    2nd order so the diff bounds the temporal error), logs it, and stores it
    in the cache file.

    Returns:
        (x_grid, t_grid, h_solution): Native grid arrays from the solver.
        h_solution is complex-valued (nt, nx)
    """
    global _cached_solution, _cached_config_hash

    problem = config.get('problem', 'problem1')
    problem_config = config[problem]
    x_min, x_max = problem_config['spatial_domain'][0]
    t_min, t_max = problem_config['temporal_domain']
    config_tuple = ((x_min, x_max), (t_min, t_max))

    if _cached_solution is None or _cached_config_hash != config_tuple:
        nx, nt = 2048, 1000

        from pathlib import Path
        cache_dir = Path(__file__).resolve().parent.parent / 'datasets' / 'gt_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        # v2: n_sub=16 Strang substeps per frame (n_sub=1 had ~1.4e-4 error)
        cache_file = cache_dir / (
            f"schrodinger_gt_v2_{x_min}_{x_max}_{t_min}_{t_max}_{nx}x{nt}.npz")

        if cache_file.exists():
            data = np.load(cache_file)
            _cached_solution = (data['x_grid'], data['t_grid'], data['h_sol'])
            _cached_config_hash = config_tuple
            print(f"  Loaded NLSE solution from cache ({cache_file.name}, "
                  f"self-convergence rel-L2 = {float(data['conv_rel_l2']):.3e})")
            return _cached_solution

        print(f"  Generating NLSE ground truth solution ({nx}x{nt} grid, "
              f"split-step n_sub=16)...")
        x_grid, t_grid, h_solution = solve_nlse_splitstep(
            x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max,
            nx=nx, nt=nt, n_sub=16,
        )
        _, _, h_fine = solve_nlse_splitstep(
            x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max,
            nx=nx, nt=nt, n_sub=32,
        )
        conv_rel_l2 = float(np.linalg.norm(h_solution - h_fine)
                            / (np.linalg.norm(h_fine) + 1e-300))
        print(f"  Solution computed (self-convergence rel-L2 = {conv_rel_l2:.3e})")
        if conv_rel_l2 > 1e-6:
            print(f"  WARNING: NLSE reference self-convergence {conv_rel_l2:.3e} "
                  f"> 1e-6 — rel-L2 metrics below this level are unreliable.")

        np.savez_compressed(cache_file, x_grid=x_grid, t_grid=t_grid,
                            h_sol=h_solution, conv_rel_l2=conv_rel_l2)
        _cached_solution = (x_grid, t_grid, h_solution)
        _cached_config_hash = config_tuple

    return _cached_solution


def _get_interpolator(config: Dict):
    """Return callable interp(x_flat, t_flat) -> complex values for ground truth."""
    from scipy.interpolate import RegularGridInterpolator
    x_grid, t_grid, h_solution = _get_solution_cached(config)
    rgi_re = RegularGridInterpolator(
        (t_grid, x_grid), h_solution.real, method='linear',
        bounds_error=False, fill_value=None,
    )
    rgi_im = RegularGridInterpolator(
        (t_grid, x_grid), h_solution.imag, method='linear',
        bounds_error=False, fill_value=None,
    )

    def _interp(x_flat, t_flat):
        pts = np.column_stack([t_flat, x_flat])
        return rgi_re(pts) + 1j * rgi_im(pts)

    return _interp


def generate_dataset(
    n_residual: int,
    n_ic: int,
    n_bc: int,
    device: torch.device,
    config: Dict
) -> Dict[str, torch.Tensor]:
    """Generate dataset with Schrodinger ground truth sampled from solver grid.

    All (x,t) points are sampled from the solver's native grid nodes.
    Residual h_gt is looked up directly from h_solution (no interpolation).
    IC h_gt is set analytically. BC h_gt is kept from the solver grid lookup
    (periodic: value at x_min equals value at x_max).
    """
    seed = config['seed']
    problem = config.get('problem', 'problem1')
    problem_config = config[problem]
    spatial_dim = problem_config['spatial_dim']
    spatial_domain = problem_config['spatial_domain']
    temporal_domain = problem_config['temporal_domain']

    # Get solver grid
    x_grid, t_grid, h_solution = _get_solution_cached(config)
    nx, nt = len(x_grid), len(t_grid)

    torch.manual_seed(seed)
    np.random.seed(seed)

    N = n_residual + n_ic + n_bc
    x = torch.zeros(N, spatial_dim, device=device)
    t = torch.zeros(N, 1, device=device)
    h_gt = torch.zeros(N, 2, device=device, dtype=torch.float32)

    x_min, x_max = spatial_domain[0]
    t_min, t_max = temporal_domain

    idx = 0

    # Residual: sample random grid indices
    print(f"  Sampling {n_residual} residual points from grid...")
    i_t = np.random.choice(nt, size=n_residual, replace=True)
    i_x = np.random.choice(nx, size=n_residual, replace=True)
    x[idx:idx + n_residual, 0] = torch.from_numpy(x_grid[i_x].astype(np.float32)).to(device)
    t[idx:idx + n_residual, 0] = torch.from_numpy(t_grid[i_t].astype(np.float32)).to(device)
    h_gt[idx:idx + n_residual, 0] = torch.from_numpy(h_solution[i_t, i_x].real.astype(np.float32)).to(device)
    h_gt[idx:idx + n_residual, 1] = torch.from_numpy(h_solution[i_t, i_x].imag.astype(np.float32)).to(device)
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
    # BC h_gt from solver: h_solution at (i_t, x=x_min) which is index 0
    h_gt[idx:idx + n_bc_left, 0] = torch.from_numpy(h_solution[i_t_bc[:n_bc_left], 0].real.astype(np.float32)).to(device)
    h_gt[idx:idx + n_bc_left, 1] = torch.from_numpy(h_solution[i_t_bc[:n_bc_left], 0].imag.astype(np.float32)).to(device)
    idx += n_bc_left

    x[idx:idx + n_bc_right, 0] = x_max
    t[idx:idx + n_bc_right, 0] = torch.from_numpy(t_grid[i_t_bc[:n_bc_right]].astype(np.float32)).to(device)
    # For periodic grid x_max is not in the grid; use first index (periodicity)
    h_gt[idx:idx + n_bc_right, 0] = torch.from_numpy(h_solution[i_t_bc[:n_bc_right], 0].real.astype(np.float32)).to(device)
    h_gt[idx:idx + n_bc_right, 1] = torch.from_numpy(h_solution[i_t_bc[:n_bc_right], 0].imag.astype(np.float32)).to(device)

    mask_residual = torch.zeros(N, dtype=torch.bool, device=device)
    mask_residual[:n_residual] = True
    mask_ic = torch.zeros(N, dtype=torch.bool, device=device)
    mask_ic[n_residual:n_residual + n_ic] = True
    mask_bc = torch.zeros(N, dtype=torch.bool, device=device)
    mask_bc[n_residual + n_ic:] = True

    # Overwrite IC with exact analytical values
    h_gt[mask_ic, 0] = (2.0 / torch.cosh(x[mask_ic, 0])).float()
    h_gt[mask_ic, 1] = 0.0

    print("  Dataset generated successfully")

    return {
        "x": x, "t": t, "h_gt": h_gt,
        "mask": {"residual": mask_residual, "IC": mask_ic, "BC": mask_bc},
    }


def evaluate_on_grid(x_grid: torch.Tensor, config: Dict) -> torch.Tensor:
    """
    Evaluate ground truth solution on a regular grid for frequency analysis.
    
    Args:
        x_grid: Grid points (N, 2) with columns [x, t]
        config: Configuration dictionary
        
    Returns:
        h_gt: Ground truth values (N, 2) with columns [real, imag]
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
    
    # Extract complex values and split into real/imag
    h_complex = h_solution[i_t, i_x]
    h_gt = np.zeros((len(x_query), 2), dtype=np.float32)
    h_gt[:, 0] = h_complex.real.astype(np.float32)
    h_gt[:, 1] = h_complex.imag.astype(np.float32)
    
    return torch.from_numpy(h_gt)
