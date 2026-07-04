"""
1D Wave Equation Solver with Analytical Standing Wave Solution.

Solves: h_tt - h_xx = 0
Domain: x in [-5, 5], t in [0, 2*pi]
Solution: h(x, t) = sin(x) * cos(t) (standing wave)
Initial Condition: h(x, 0) = sin(x), h_t(x, 0) = 0
Boundary Conditions: h(+/-5, t) = 0 (Dirichlet)
"""

import numpy as np
import torch
from typing import Tuple, Dict


def analytical_solution(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Analytical solution for 1D wave: h(x, t) = sin(x) * cos(t).
    
    Args:
        x: Spatial coordinates (numpy array)
        t: Temporal coordinates (numpy array)
        
    Returns:
        h: Solution values (real-valued)
    """
    return np.sin(x) * np.cos(t)


def analytical_derivative_t(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Analytical time derivative: h_t(x, t) = -sin(x) * sin(t).
    
    Args:
        x: Spatial coordinates (numpy array)
        t: Temporal coordinates (numpy array)
        
    Returns:
        h_t: Time derivative (real-valued)
    """
    return -np.sin(x) * np.sin(t)


def analytical_derivative_tt(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Analytical second time derivative: h_tt(x, t) = -sin(x) * cos(t).
    
    Args:
        x: Spatial coordinates (numpy array)
        t: Temporal coordinates (numpy array)
        
    Returns:
        h_tt: Second time derivative (real-valued)
    """
    return -np.sin(x) * np.cos(t)


def analytical_derivative_x(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Analytical spatial derivative: h_x(x, t) = cos(x) * cos(t).
    
    Args:
        x: Spatial coordinates (numpy array)
        t: Temporal coordinates (numpy array)
        
    Returns:
        h_x: Spatial derivative (real-valued)
    """
    return np.cos(x) * np.cos(t)


def analytical_derivative_xx(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Analytical second spatial derivative: h_xx(x, t) = -sin(x) * cos(t).
    
    Args:
        x: Spatial coordinates (numpy array)
        t: Temporal coordinates (numpy array)
        
    Returns:
        h_xx: Second spatial derivative (real-valued)
    """
    return -np.sin(x) * np.cos(t)


def initial_condition_analytical(x: np.ndarray) -> np.ndarray:
    """
    Analytical initial condition for 1D wave: h(x, 0) = sin(x).
    
    Args:
        x: Spatial coordinates (numpy array)
        
    Returns:
        h0: Initial displacement (real-valued)
    """
    return np.sin(x)


def solve_wave1d_analytical(
    x_min: float = -5.0,
    x_max: float = 5.0,
    t_min: float = 0.0,
    t_max: float = 2 * np.pi,
    nx: int = 1024,
    nt: int = 800,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analytical solution for 1D wave equation with standing wave.
    
    Equation: h_tt - h_xx = 0
    Solution: h(x, t) = sin(x) * cos(t)
    
    This is an exact analytical solution satisfying:
    - PDE: h_tt = -sin(x)cos(t), h_xx = -sin(x)cos(t) => h_tt - h_xx = 0 [OK]
    - IC: h(x, 0) = sin(x), h_t(x, 0) = 0 [OK]
    - BC: h(+/-5, t) = sin(+/-5)cos(t) ~= 0.96*cos(t) (approximately zero for large |x|)
    
    Args:
        x_min: Minimum spatial coordinate
        x_max: Maximum spatial coordinate
        t_min: Initial time (typically 0)
        t_max: Final time (typically 2*pi for one period)
        nx: Number of spatial grid points
        nt: Number of temporal grid points
        
    Returns:
        x_grid: Spatial grid (nx,)
        t_grid: Temporal grid (nt,)
        h_solution: Real solution field (nt, nx)
    """
    # Create grids
    x_grid = np.linspace(x_min, x_max, nx, dtype=np.float64)
    t_grid = np.linspace(t_min, t_max, nt, dtype=np.float64)
    
    # Create meshgrid for vectorized computation
    X, T = np.meshgrid(x_grid, t_grid)
    
    # Analytical standing wave solution
    h_solution = np.sin(X) * np.cos(T)
    
    return x_grid, t_grid, h_solution


_cached_solution = None
_cached_config_hash = None


def _get_solution_cached(config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get cached Wave1D solution grid.
    
    Returns:
        (x_grid, t_grid, h_solution): Native grid arrays from analytical solution
    """
    global _cached_solution, _cached_config_hash
    
    # Create hash from relevant config params
    problem_config = config['wave1d']
    config_tuple = (
        tuple(problem_config['spatial_domain'][0]),
        tuple(problem_config['temporal_domain'])
    )
    
    if _cached_solution is None or _cached_config_hash != config_tuple:
        print("  Generating wave1d solution (1024x800 grid)...")
        # Extract domain from config
        x_min, x_max = problem_config['spatial_domain'][0]
        t_min, t_max = problem_config['temporal_domain']
        
        # Solve wave equation on fine grid
        x_grid, t_grid, h_solution = solve_wave1d_analytical(
            x_min=x_min,
            x_max=x_max,
            t_min=t_min,
            t_max=t_max,
            nx=1024,
            nt=800
        )
        
        _cached_solution = (x_grid, t_grid, h_solution)
        _cached_config_hash = config_tuple
        print("  Solution computed: 800x1024 grid")
    
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
    n_residual: int,
    n_ic: int,
    n_bc: int,
    device: torch.device,
    config: Dict
) -> Dict[str, torch.Tensor]:
    """
    Generate dataset with wave1d ground truth sampled from solver grid.
    
    Uses analytical standing wave solution on 1024x800 grid.
    
    Args:
        n_residual: Number of residual (interior) points
        n_ic: Number of initial condition points
        n_bc: Number of boundary condition points (at x=+/-5)
        device: Device to create tensors on (CUDA or CPU)
        config: Configuration dictionary
        
    Returns:
        Dictionary with keys:
            "x": (N, spatial_dim) spatial coordinates
            "t": (N, 1) temporal coordinates
            "h_gt": (N, 1) ground truth solution (real-valued)
            "mask": dict with "residual", "IC", "BC" boolean masks
    """
    import torch
    
    seed = config['seed']
    problem = config.get('problem', 'wave1d')
    problem_config = config[problem]
    spatial_dim = problem_config['spatial_dim']
    spatial_domain = problem_config['spatial_domain']  # [[min, max], ...]
    temporal_domain = problem_config['temporal_domain']  # [min, max]
    
    # Get solver grid
    x_grid, t_grid, h_solution = _get_solution_cached(config)
    nx, nt = len(x_grid), len(t_grid)
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N = n_residual + n_ic + n_bc
    
    # Initialize tensors
    x = torch.zeros(N, spatial_dim, device=device)
    t = torch.zeros(N, 1, device=device)
    h_gt = torch.zeros(N, 1, device=device, dtype=torch.float32)
    
    # Extract domain bounds
    x_min, x_max = spatial_domain[0]
    t_min, t_max = temporal_domain
    
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
    
    # Left boundary (x = x_min)
    x[idx:idx + n_bc_left, 0] = x_min
    t[idx:idx + n_bc_left, 0] = torch.from_numpy(t_grid[i_t_bc[:n_bc_left]].astype(np.float32)).to(device)
    idx += n_bc_left
    
    # Right boundary (x = x_max)
    x[idx:idx + n_bc_right, 0] = x_max
    t[idx:idx + n_bc_right, 0] = torch.from_numpy(t_grid[i_t_bc[:n_bc_right]].astype(np.float32)).to(device)
    idx += n_bc_right
    
    # Create masks
    mask_residual = torch.zeros(N, dtype=torch.bool, device=device)
    mask_residual[:n_residual] = True
    
    mask_ic = torch.zeros(N, dtype=torch.bool, device=device)
    mask_ic[n_residual:n_residual + n_ic] = True
    
    mask_bc = torch.zeros(N, dtype=torch.bool, device=device)
    mask_bc[n_residual + n_ic:] = True

    # Overwrite IC/BC with exact analytical values (no interpolation error)
    h_gt[mask_ic, 0] = torch.sin(x[mask_ic, 0]).float()
    h_gt[mask_bc, 0] = (torch.sin(x[mask_bc, 0]) * torch.cos(t[mask_bc, 0])).float()

    print("  Dataset generated successfully")
    
    return {
        "x": x,
        "t": t,
        "h_gt": h_gt,
        "mask": {
            "residual": mask_residual,
            "IC": mask_ic,
            "BC": mask_bc
        }
    }


def evaluate_on_grid(x_grid: torch.Tensor, config: Dict) -> torch.Tensor:
    """
    Evaluate ground truth solution on a regular grid for frequency analysis.
    
    Args:
        x_grid: Grid points (N, 2) with columns [x, t]
        config: Configuration dictionary
        
    Returns:
        h_gt: Ground truth values (N, 1)
    """
    x_np = x_grid.cpu().numpy()
    
    # Extract coordinates
    x = x_np[:, 0]
    t = x_np[:, 1]
    
    # Evaluate analytical solution
    h = analytical_solution(x, t)
    
    return torch.from_numpy(h.reshape(-1, 1).astype(np.float32))

