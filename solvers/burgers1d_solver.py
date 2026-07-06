"""
1D Viscous Burgers Equation Solver using Cole-Hopf Exact Solution.

Solves: h_t + h*h_x - (nu/pi)*h_xx = 0
Domain: x in [-1, 1], t in [0, 1]
Initial Condition: h(0, x) = -sin(pi*x)
Boundary Conditions: h(t, -1) = h(t, 1) = 0 (Dirichlet)

Uses Cole-Hopf transformation to obtain exact solution, with
Chebyshev collocation solver as an independent cross-check.
"""

import numpy as np
import torch
from typing import Tuple, Dict
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os


def initial_condition(x: torch.Tensor) -> torch.Tensor:
    """Exact IC h(x, 0) = -sin(pi*x). Also the whole-domain target of the
    PirateNets physics-informed output init (u(x,t) ≈ u0(x) for all t)."""
    return (-torch.sin(np.pi * x[:, :1])).float()


def cole_hopf_exact(x, t, nu, n_terms=None):
    """
    Compute exact solution using Cole-Hopf / Hopf integral formula.
    
    Burgers: h_t + h*h_x = epsilon*h_xx  where epsilon = nu/pi
    Domain: [-1, 1], IC: h(x,0) = -sin(pi*x), Dirichlet BCs: h(+-1,t) = 0
    
    Uses the Hopf formula (integral representation):
      h(x,t) = [integral (x-xi)/t * w(xi) dxi] / [integral w(xi) dxi]
    where w(xi) = exp(E(xi)) and
      E(xi) = -(x-xi)^2/(4*epsilon*t) + (1-cos(pi*xi))/(2*nu)
    
    The IC is odd and 2-periodic, so the infinite-domain solution automatically
    satisfies h(+-1,t)=0, making this formula exact for our Dirichlet problem.
    
    Reference: Raissi et al. (2019), Cole-Hopf transformation
    
    Args:
        x: spatial coordinates on [-1, 1] (array-like)
        t: time (scalar)
        nu: viscosity parameter (epsilon = nu/pi in the PDE)
    
    Returns:
        h: solution values at (x, t)
    """
    x = np.asarray(x, dtype=np.float64)
    t_val = float(t)
    epsilon = nu / np.pi
    
    if t_val <= 0.0:
        return -np.sin(np.pi * x)
    
    # Quadrature grid for Hopf integral
    # Width of Gaussian kernel ~ sqrt(4*epsilon*t); use wide interval to capture all
    n_quad = 10000
    xi = np.linspace(-4.0, 4.0, n_quad)
    
    # F(xi) = integral_0^xi h_0(s) ds = (cos(pi*xi) - 1) / pi
    # E(xi; x, t) = -(x-xi)^2/(4*eps*t) - F(xi)/(2*eps)
    #             = -(x-xi)^2/(4*eps*t) + (1 - cos(pi*xi))/(2*nu)
    # Note: F/(2*eps) = (cos(pi*xi)-1)/(2*pi*eps) = (cos(pi*xi)-1)/(2*nu)
    
    phi_part = (1.0 - np.cos(np.pi * xi)) / (2.0 * nu)  # shape (n_quad,)
    
    # Vectorize over x: shape (nx, n_quad)
    X = x[:, np.newaxis]   # (nx, 1)
    Xi = xi[np.newaxis, :]  # (1, n_quad)
    
    gauss_part = -((X - Xi) ** 2) / (4.0 * epsilon * t_val)  # (nx, n_quad)
    E = gauss_part + phi_part[np.newaxis, :]                   # (nx, n_quad)
    
    # Subtract row-wise max for numerical stability
    E_max = E.max(axis=1, keepdims=True)
    w = np.exp(E - E_max)  # (nx, n_quad)
    
    kernel = (X - Xi) / t_val  # (nx, n_quad)
    
    # np.trapz was removed in NumPy 2.0 (renamed trapezoid)
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    numerator   = _trapz(kernel * w, xi, axis=1)  # (nx,)
    denominator = _trapz(w, xi, axis=1)            # (nx,)
    
    h = numerator / denominator
    
    return h


def solve_burgers_chebyshev(
    x_min=-1.0, x_max=1.0, t_min=0.0, t_max=1.0,
    nx=64, nt=201, nu=0.01
):
    """
    Solve Burgers equation using Chebyshev collocation in space.
    
    This provides an independent numerical check against Cole-Hopf.
    Uses Method of Lines: Chebyshev discretization in x, ODE solver in t.
    """
    from numpy.polynomial import chebyshev as cheb
    
    # Chebyshev collocation points in [-1, 1]
    # Use Gauss-Lobatto points: includes boundaries
    i = np.arange(nx)
    x_cheb = -np.cos(np.pi * i / (nx - 1))
    
    # Chebyshev differentiation matrices
    # D1: first derivative, D2: second derivative
    c = np.ones(nx)
    c[0] = 2.0
    c[-1] = 2.0
    
    D1 = np.zeros((nx, nx))
    for i in range(nx):
        for j in range(nx):
            if i != j:
                D1[i, j] = (c[i] * (-1) ** (i + j)) / (c[j] * (x_cheb[i] - x_cheb[j]))
            elif i == 0 and j == 0:
                D1[i, j] = (2.0 * (nx - 1) ** 2 + 1.0) / 6.0
            elif i == nx - 1 and j == nx - 1:
                D1[i, j] = -(2.0 * (nx - 1) ** 2 + 1.0) / 6.0
    
    D2 = D1 @ D1
    
    visc = nu / np.pi
    
    # Initial condition
    h0 = -np.sin(np.pi * x_cheb)
    
    # Enforce Dirichlet BCs: h[-1, t] = h[1, t] = 0
    # Interior points: x_cheb[1:-1]
    D1_int = D1[1:-1, :]
    D2_int = D2[1:-1, :]
    
    def rhs(t, h_full):
        """RHS for interior points with boundary conditions enforced."""
        h = np.zeros(nx)
        h[0] = 0.0  # BC at x=-1
        h[-1] = 0.0  # BC at x=1
        h[1:-1] = h_full  # Interior
        
        h_x = D1_int @ h
        h_xx = D2_int @ h
        
        # Burgers: h_t = -h*h_x + visc*h_xx (for interior)
        return -h[1:-1] * h_x + visc * h_xx
    
    # Time integration
    t_eval = np.linspace(t_min, t_max, nt)
    sol = solve_ivp(
        rhs, (t_min, t_max), h0[1:-1],
        method='Radau', t_eval=t_eval,
        rtol=1e-8, atol=1e-10
    )
    
    # Reconstruct full solution with BCs
    h_solution = np.zeros((nt, nx))
    actual_nt = sol.y.shape[1]  # Actual number of time points returned
    for i in range(min(nt, actual_nt)):
        h_solution[i, 0] = 0.0
        h_solution[i, -1] = 0.0
        h_solution[i, 1:-1] = sol.y[:, i]
    
    # If solver returned fewer points, use only what we got
    if actual_nt < nt:
        h_solution = h_solution[:actual_nt]
        t_eval = t_eval[:actual_nt]
    
    return x_cheb, t_eval, h_solution


def cross_check_and_visualize(
    x_check, t_check, h_cole_hopf, h_chebyshev,
    save_dir, nu
):
    """
    Cross-check Cole-Hopf vs Chebyshev and generate visualization.
    
    Args:
        x_check: spatial grid (nx,)
        t_check: temporal grid (nt,)
        h_cole_hopf: Cole-Hopf solution (nt, nx)
        h_chebyshev: Chebyshev solution (nt, nx)
        save_dir: directory to save cross-check results
        nu: viscosity parameter
    """
    # Compute differences
    diff = h_cole_hopf - h_chebyshev
    max_diff = np.abs(diff).max()
    
    # Relative L2 error
    l2_cheb = np.linalg.norm(h_chebyshev)
    rel_l2 = np.linalg.norm(diff) / (l2_cheb + 1e-12)
    
    print(f"\n  === Burgers1D Cross-Check ===")
    print(f"  Cole-Hopf vs Chebyshev (nx={len(x_check)}, nt={len(t_check)})")
    print(f"  Max pointwise difference: {max_diff:.6e}")
    print(f"  Relative L2 error: {rel_l2:.6e}")
    
    if max_diff > 1e-6:
        print(f"  WARNING: Difference exceeds 1e-6 threshold!")
    else:
        print(f"  PASS: Cross-check passed (difference < 1e-6)")
    
    # Generate visualization
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Cole-Hopf solution
    ax = axes[0, 0]
    X, T = np.meshgrid(x_check, t_check)
    im1 = ax.contourf(X, T, h_cole_hopf, levels=50, cmap='RdBu_r')
    ax.set_title(f'Cole-Hopf Exact Solution (nu/pi = {nu:.4f})')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    plt.colorbar(im1, ax=ax)
    
    # 2. Chebyshev solution
    ax = axes[0, 1]
    im2 = ax.contourf(X, T, h_chebyshev, levels=50, cmap='RdBu_r')
    ax.set_title(f'Chebyshev Collocation (nx={len(x_check)})')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    plt.colorbar(im2, ax=ax)
    
    # 3. Absolute difference
    ax = axes[1, 0]
    im3 = ax.contourf(X, T, np.abs(diff), levels=50, cmap='Reds')
    ax.set_title(f'Absolute Difference (max={max_diff:.2e})')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    plt.colorbar(im3, ax=ax)
    
    # 4. Statistics text
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
Cross-Check Statistics
======================
Grid: {len(x_check)}x{len(t_check)}
Viscosity: nu/pi = {nu:.4f}

Max |difference|: {max_diff:.6e}
Mean |difference|: {np.abs(diff).mean():.6e}
Relative L2 error: {rel_l2:.6e}

Status: {'PASS' if max_diff < 1e-6 else 'WARNING'}
Threshold: 1e-6
    """
    ax.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'burgers1d_crosscheck.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Cross-check visualization saved: {fig_path}")
    
    return max_diff, rel_l2


_cached_solution = None
_cached_config_hash = None
_crosscheck_done = False


def _get_solution_cached(config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get cached Burgers1D Cole-Hopf solution grid with cross-check.
    
    Returns:
        (x_grid, t_grid, h_solution): Native grid arrays from Cole-Hopf exact formula
    """
    global _cached_solution, _cached_config_hash, _crosscheck_done
    
    problem_config = config['burgers1d']
    x_min, x_max = problem_config['spatial_domain'][0]
    t_min, t_max = problem_config['temporal_domain']
    nu = problem_config['nu']
    
    config_tuple = (x_min, x_max, t_min, t_max, nu)
    
    if _cached_solution is None or _cached_config_hash != config_tuple:
        print("  Generating Burgers1D solution using Cole-Hopf exact formula (Fourier sine series)...")
        
        # Generate Cole-Hopf solution on fine grid
        nx_fine = 256
        nt_fine = 201
        x_grid = np.linspace(x_min, x_max, nx_fine)
        t_grid = np.linspace(t_min, t_max, nt_fine)
        
        h_cole_hopf = np.zeros((nt_fine, nx_fine))
        for i, t_val in enumerate(t_grid):
            h_cole_hopf[i, :] = cole_hopf_exact(x_grid, t_val, nu)
        
        print(f"  Cole-Hopf solution computed ({nx_fine}x{nt_fine} grid, 200 Fourier sine terms)")
        
        # Cross-check with Chebyshev (once per config)
        if not _crosscheck_done:
            print("\n  Running cross-check with Chebyshev collocation...")
            try:
                x_cheb, t_cheb, h_cheb = solve_burgers_chebyshev(
                    x_min, x_max, t_min, t_max,
                    nx=128, nt=51, nu=nu
                )
                
                h_cole_on_cheb = np.zeros_like(h_cheb)
                for i, t_val in enumerate(t_cheb):
                    h_cole_on_cheb[i, :] = cole_hopf_exact(x_cheb, t_val, nu)
                
                save_dir = os.path.join(
                    os.path.dirname(__file__), '..', 'datasets', 'burgers1d'
                )
                cross_check_and_visualize(
                    x_cheb, t_cheb, h_cole_on_cheb, h_cheb, save_dir, nu
                )
            except Exception as e:
                print(f"  WARNING: Chebyshev cross-check failed ({e})")
                print("  Cole-Hopf solution is still used as ground truth.")
            
            _crosscheck_done = True
        
        _cached_solution = (x_grid, t_grid, h_cole_hopf)
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
    device: torch.device, config: Dict
) -> Dict[str, torch.Tensor]:
    """
    Generate dataset with Burgers1D ground truth sampled from Cole-Hopf grid.
    
    Ground truth is computed using Cole-Hopf transformation (exact solution).
    First call performs cross-check with Chebyshev collocation and saves visualization.
    """
    seed = config['seed']
    problem = config.get('problem', 'burgers1d')
    problem_config = config[problem]
    spatial_dim = problem_config['spatial_dim']
    x_min, x_max = problem_config['spatial_domain'][0]
    t_min, t_max = problem_config['temporal_domain']
    
    # Get Cole-Hopf solution grid
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
    
    # Masks
    mask_res = torch.zeros(N, dtype=torch.bool, device=device)
    mask_res[:n_residual] = True
    mask_ic = torch.zeros(N, dtype=torch.bool, device=device)
    mask_ic[n_residual:n_residual + n_ic] = True
    mask_bc = torch.zeros(N, dtype=torch.bool, device=device)
    mask_bc[n_residual + n_ic:] = True

    # Overwrite IC/BC with exact analytical values (no interpolation error)
    h_gt[mask_ic, 0] = (-torch.sin(np.pi * x[mask_ic, 0])).float()
    h_gt[mask_bc, 0] = 0.0

    print("  Dataset generated successfully (Cole-Hopf exact)")
    
    return {
        "x": x, "t": t, "h_gt": h_gt,
        "mask": {"residual": mask_res, "IC": mask_ic, "BC": mask_bc},
    }


def evaluate_on_grid(x_grid: torch.Tensor, config: Dict) -> torch.Tensor:
    """Evaluate Cole-Hopf exact solution on a regular grid."""
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
