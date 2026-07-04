"""
2D Viscous Burgers Equation Solver with Analytical Solution.

Solves: h_t + h*(h_x0 + h_x1) - 0.1*(h_x0x0 + h_x1x1) = 0
Domain: (x0, x1) ∈ [0, 1] × [0, 1], t ∈ [0, 2]
Initial Condition: h(0, x0, x1) = 1 / (1 + exp((x0 + x1) / 0.2))
Boundary Conditions: h(t, x0_b, x1_b) = 1 / (1 + exp((x0_b + x1_b - t) / 0.2))

Analytical Solution: h(t, x0, x1) = 1 / (1 + exp((x0 + x1 - t) / 0.2))
"""

import numpy as np
import torch
from typing import Tuple, Dict


# =============================================================================
# ANALYTICAL SOLUTION AND DERIVATIVES
# =============================================================================

def analytical_solution(x0: np.ndarray, x1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute the analytical solution h(t, x0, x1).
    
    h(t, x0, x1) = 1 / (1 + exp((x0 + x1 - t) / 0.2))
    
    Args:
        x0: First spatial coordinate (N,)
        x1: Second spatial coordinate (N,)
        t: Time coordinate (N,)
        
    Returns:
        h: Solution values (N,)
    """
    z = (x0 + x1 - t) / 0.2
    # Clip z to avoid overflow in exp
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(z))


def analytical_h_t(x0: np.ndarray, x1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute dh/dt analytically.
    
    h_t = 5 * h * (1 - h)
    
    Args:
        x0, x1, t: Coordinates (N,)
        
    Returns:
        h_t: Time derivative (N,)
    """
    h = analytical_solution(x0, x1, t)
    return 5.0 * h * (1.0 - h)


def analytical_h_x0(x0: np.ndarray, x1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute dh/dx0 analytically.
    
    h_x0 = -5 * h * (1 - h)
    
    Args:
        x0, x1, t: Coordinates (N,)
        
    Returns:
        h_x0: Spatial derivative w.r.t. x0 (N,)
    """
    h = analytical_solution(x0, x1, t)
    return -5.0 * h * (1.0 - h)


def analytical_h_x1(x0: np.ndarray, x1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute dh/dx1 analytically.
    
    h_x1 = -5 * h * (1 - h)
    
    Args:
        x0, x1, t: Coordinates (N,)
        
    Returns:
        h_x1: Spatial derivative w.r.t. x1 (N,)
    """
    h = analytical_solution(x0, x1, t)
    return -5.0 * h * (1.0 - h)


def analytical_h_x0x0(x0: np.ndarray, x1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute d²h/dx0² analytically.
    
    h_x0x0 = 25 * h * (1 - h) * (1 - 2h)
    
    Args:
        x0, x1, t: Coordinates (N,)
        
    Returns:
        h_x0x0: Second spatial derivative w.r.t. x0 (N,)
    """
    h = analytical_solution(x0, x1, t)
    return 25.0 * h * (1.0 - h) * (1.0 - 2.0 * h)


def analytical_h_x1x1(x0: np.ndarray, x1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute d²h/dx1² analytically.
    
    h_x1x1 = 25 * h * (1 - h) * (1 - 2h)
    
    Args:
        x0, x1, t: Coordinates (N,)
        
    Returns:
        h_x1x1: Second spatial derivative w.r.t. x1 (N,)
    """
    h = analytical_solution(x0, x1, t)
    return 25.0 * h * (1.0 - h) * (1.0 - 2.0 * h)


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_dataset(
    n_residual: int,
    n_ic: int,
    n_bc: int,
    device: torch.device,
    config: Dict
) -> Dict[str, torch.Tensor]:
    """
    Generate dataset with burgers2d ground truth using analytical solution.
    
    Args:
        n_residual: Number of residual (interior) points
        n_ic: Number of initial condition points
        n_bc: Number of boundary condition points
        device: Device to create tensors on (CUDA or CPU)
        config: Configuration dictionary
        
    Returns:
        Dictionary with keys:
            "x": (N, spatial_dim) spatial coordinates [x0, x1]
            "t": (N, 1) temporal coordinates
            "h_gt": (N, 1) ground truth solution
            "mask": dict with "residual", "IC", "BC" boolean masks
    """
    seed = config['seed']
    problem = config.get('problem', 'burgers2d')
    problem_config = config[problem]
    spatial_dim = problem_config['spatial_dim']
    spatial_domain = problem_config['spatial_domain']
    temporal_domain = problem_config['temporal_domain']
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N = n_residual + n_ic + n_bc
    
    # Initialize tensors
    x = torch.zeros(N, spatial_dim, device=device)
    t = torch.zeros(N, 1, device=device)
    
    # Extract domain bounds
    x0_min, x0_max = spatial_domain[0]
    x1_min, x1_max = spatial_domain[1]
    t_min, t_max = temporal_domain
    
    idx = 0
    
    # 1. Residual points (uniform random in interior)
    print(f"  Sampling {n_residual} residual points...")
    x[idx:idx + n_residual, 0] = torch.rand(n_residual, device=device) * (x0_max - x0_min) + x0_min
    x[idx:idx + n_residual, 1] = torch.rand(n_residual, device=device) * (x1_max - x1_min) + x1_min
    t[idx:idx + n_residual, 0] = torch.rand(n_residual, device=device) * (t_max - t_min) + t_min
    idx += n_residual
    
    # 2. Initial condition points (uniform random at t=0)
    print(f"  Sampling {n_ic} initial condition points...")
    x[idx:idx + n_ic, 0] = torch.rand(n_ic, device=device) * (x0_max - x0_min) + x0_min
    x[idx:idx + n_ic, 1] = torch.rand(n_ic, device=device) * (x1_max - x1_min) + x1_min
    t[idx:idx + n_ic, 0] = t_min
    idx += n_ic
    
    # 3. Boundary condition points (on all 4 edges)
    print(f"  Sampling {n_bc} boundary condition points...")
    n_bc_per_edge = n_bc // 4
    n_bc_remaining = n_bc - 4 * n_bc_per_edge
    
    # Edge 1: x0 = x0_min (left edge)
    n_edge1 = n_bc_per_edge + (1 if n_bc_remaining > 0 else 0)
    x[idx:idx + n_edge1, 0] = x0_min
    x[idx:idx + n_edge1, 1] = torch.rand(n_edge1, device=device) * (x1_max - x1_min) + x1_min
    t[idx:idx + n_edge1, 0] = torch.rand(n_edge1, device=device) * (t_max - t_min) + t_min
    idx += n_edge1
    n_bc_remaining = max(0, n_bc_remaining - 1)
    
    # Edge 2: x0 = x0_max (right edge)
    n_edge2 = n_bc_per_edge + (1 if n_bc_remaining > 0 else 0)
    x[idx:idx + n_edge2, 0] = x0_max
    x[idx:idx + n_edge2, 1] = torch.rand(n_edge2, device=device) * (x1_max - x1_min) + x1_min
    t[idx:idx + n_edge2, 0] = torch.rand(n_edge2, device=device) * (t_max - t_min) + t_min
    idx += n_edge2
    n_bc_remaining = max(0, n_bc_remaining - 1)
    
    # Edge 3: x1 = x1_min (bottom edge)
    n_edge3 = n_bc_per_edge + (1 if n_bc_remaining > 0 else 0)
    x[idx:idx + n_edge3, 0] = torch.rand(n_edge3, device=device) * (x0_max - x0_min) + x0_min
    x[idx:idx + n_edge3, 1] = x1_min
    t[idx:idx + n_edge3, 0] = torch.rand(n_edge3, device=device) * (t_max - t_min) + t_min
    idx += n_edge3
    n_bc_remaining = max(0, n_bc_remaining - 1)
    
    # Edge 4: x1 = x1_max (top edge)
    n_edge4 = n_bc_per_edge + (1 if n_bc_remaining > 0 else 0)
    x[idx:idx + n_edge4, 0] = torch.rand(n_edge4, device=device) * (x0_max - x0_min) + x0_min
    x[idx:idx + n_edge4, 1] = x1_max
    t[idx:idx + n_edge4, 0] = torch.rand(n_edge4, device=device) * (t_max - t_min) + t_min
    idx += n_edge4
    
    # Create masks
    mask_residual = torch.zeros(N, dtype=torch.bool, device=device)
    mask_residual[:n_residual] = True
    
    mask_ic = torch.zeros(N, dtype=torch.bool, device=device)
    mask_ic[n_residual:n_residual + n_ic] = True
    
    mask_bc = torch.zeros(N, dtype=torch.bool, device=device)
    mask_bc[n_residual + n_ic:] = True
    
    # Compute ground truth using analytical solution
    print("  Computing ground truth values...")
    x_np = x.cpu().numpy()
    t_np = t.cpu().numpy()[:, 0]
    
    h_gt_np = analytical_solution(x_np[:, 0], x_np[:, 1], t_np)
    
    # Convert to (N, 1) format
    h_gt = torch.zeros(N, 1, device=device, dtype=torch.float32)
    h_gt[:, 0] = torch.from_numpy(h_gt_np.astype(np.float32)).to(device)
    
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
        x_grid: Grid points (N, 3) with columns [x0, x1, t]
        config: Configuration dictionary
        
    Returns:
        h_gt: Ground truth values (N, 1)
    """
    x_np = x_grid.cpu().numpy()
    
    # Extract coordinates
    x0 = x_np[:, 0]
    x1 = x_np[:, 1]
    t = x_np[:, 2]
    
    # Evaluate analytical solution
    h = analytical_solution(x0, x1, t)
    
    return torch.from_numpy(h.reshape(-1, 1).astype(np.float32))

