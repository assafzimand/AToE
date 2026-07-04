"""
Fisher-KPP Equation Solver using Method of Lines + BDF.

Solves: h_t = D * h_xx + kappa * h * (1 - h)
Domain: x in [0, 1], t in [0, 1]
Initial Condition: h(x, 0) = 1 / (1 + exp(sqrt(kappa/6) * (x - 0.25)))
Boundary Conditions: h(0, t) = 1, h(1, t) = 0 (Dirichlet)
Parameters: D = 1.0, kappa = 25.0
"""

import numpy as np
import torch
from typing import Tuple, Dict
from scipy.integrate import solve_ivp


def solve_fisher_kpp(
    x_min: float = 0.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    nx: int = 512,
    nt: int = 201,
    D: float = 1.0,
    kappa: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the Fisher-KPP equation using method of lines with BDF integrator.

    Equation: h_t = D * h_xx + kappa * h * (1 - h)
    """
    x_grid = np.linspace(x_min, x_max, nx + 2, dtype=np.float64)
    dx = x_grid[1] - x_grid[0]
    t_grid = np.linspace(t_min, t_max, nt, dtype=np.float64)

    x_int = x_grid[1:-1]

    width = np.sqrt(kappa / 6.0)
    u0 = 1.0 / (1.0 + np.exp(width * (x_int - 0.25)))

    bc_left = 1.0
    bc_right = 0.0

    coeff = D / dx ** 2

    def rhs(t_val, u):
        du = np.empty_like(u)
        du[0] = coeff * (bc_left - 2 * u[0] + u[1])
        du[1:-1] = coeff * (u[:-2] - 2 * u[1:-1] + u[2:])
        du[-1] = coeff * (u[-2] - 2 * u[-1] + bc_right)
        du += kappa * u * (1.0 - u)
        return du

    print("  Solving Fisher-KPP with BDF integrator...")
    sol = solve_ivp(
        rhs,
        (t_min, t_max),
        u0,
        method='BDF',
        t_eval=t_grid,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.01,
    )

    if not sol.success:
        print(f"  WARNING: solver message: {sol.message}")

    h_int = sol.y.T  # (nt, n_int)

    h_solution = np.zeros((nt, nx + 2), dtype=np.float64)
    h_solution[:, 0] = bc_left
    h_solution[:, -1] = bc_right
    h_solution[:, 1:-1] = h_int

    return x_grid, t_grid, h_solution


_cached_solution = None
_cached_config_hash = None


def _get_solution_cached(config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get cached Fisher-KPP solution grid.
    
    Returns:
        (x_grid, t_grid, h_solution): Native grid arrays from the solver
    """
    global _cached_solution, _cached_config_hash
    problem = config.get('problem', 'fisher_kpp')
    pc = config[problem]
    config_tuple = (
        tuple(pc['spatial_domain'][0]),
        tuple(pc['temporal_domain']),
        pc['D'],
        pc['kappa'],
    )
    if _cached_solution is None or _cached_config_hash != config_tuple:
        print("  Generating Fisher-KPP solution (514x201 grid)...")
        x_min, x_max = pc['spatial_domain'][0]
        t_min, t_max = pc['temporal_domain']
        D = pc['D']
        kappa = pc['kappa']

        x_grid, t_grid, h_sol = solve_fisher_kpp(
            x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max,
            nx=512, nt=201, D=D, kappa=kappa,
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
    """Generate dataset with Fisher-KPP ground truth sampled from solver grid."""
    seed = config['seed']
    problem = config.get('problem', 'fisher_kpp')
    pc = config[problem]
    spatial_dim = pc['spatial_dim']
    x_min, x_max = pc['spatial_domain'][0]
    t_min, t_max = pc['temporal_domain']
    kappa = pc['kappa']

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

    # BC: sample random t from grid
    print(f"  Sampling {n_bc} boundary condition points from grid...")
    n_bc_left = n_bc // 2
    n_bc_right = n_bc - n_bc_left
    i_t_bc = np.random.choice(nt, size=max(n_bc_left, n_bc_right), replace=True)

    x[idx:idx + n_bc_left, 0] = x_min
    t[idx:idx + n_bc_left, 0] = torch.from_numpy(t_grid[i_t_bc[:n_bc_left]].astype(np.float32)).to(device)
    idx += n_bc_left
    x[idx:idx + n_bc_right, 0] = x_max
    t[idx:idx + n_bc_right, 0] = torch.from_numpy(t_grid[i_t_bc[:n_bc_right]].astype(np.float32)).to(device)

    mask_res = torch.zeros(N, dtype=torch.bool, device=device)
    mask_res[:n_residual] = True
    mask_ic = torch.zeros(N, dtype=torch.bool, device=device)
    mask_ic[n_residual:n_residual + n_ic] = True
    mask_bc = torch.zeros(N, dtype=torch.bool, device=device)
    mask_bc[n_residual + n_ic:] = True

    # Overwrite IC/BC with exact analytical values (no interpolation error)
    width = np.sqrt(kappa / 6.0)
    h_gt[mask_ic, 0] = (1.0 / (1.0 + torch.exp(width * (x[mask_ic, 0] - 0.25)))).float()
    mid = (x_min + x_max) / 2.0
    x_bc = x[mask_bc, 0]
    h_gt[mask_bc, 0] = torch.where(
        x_bc < mid, torch.ones_like(x_bc),
        torch.zeros_like(x_bc)).float()

    print("  Dataset generated successfully")
    return {
        "x": x, "t": t, "h_gt": h_gt,
        "mask": {"residual": mask_res, "IC": mask_ic, "BC": mask_bc},
    }


def evaluate_on_grid(x_grid: torch.Tensor, config: Dict) -> torch.Tensor:
    """Evaluate ground truth on a regular grid for frequency analysis."""
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
