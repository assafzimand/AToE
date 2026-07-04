#!/usr/bin/env python3
"""
Standalone visualization of compact smoothstep windows.

No repo imports — uses only numpy + matplotlib.
Sanity-checks the flat-top shape: =1 inside region, C^N ramps in collars, exact 0 outside.

Usage:
    python scripts/plot_window_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# -----------------------------------------------------------------------------
# Smoothstep polynomials S_N(t), t in [0,1], C^N continuity
# -----------------------------------------------------------------------------

def smoothstep_1(t):
    """C^1 smoothstep: S_1(t) = 3t^2 - 2t^3"""
    return 3 * t**2 - 2 * t**3


def smoothstep_2(t):
    """C^2 smoothstep: S_2(t) = 6t^5 - 15t^4 + 10t^3"""
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def smoothstep_3(t):
    """C^3 smoothstep: S_3(t) = 35t^4 - 84t^5 + 70t^6 - 20t^7"""
    return 35 * t**4 - 84 * t**5 + 70 * t**6 - 20 * t**7


def smoothstep_4(t):
    """C^4 smoothstep: S_4(t) = 126t^5 - 420t^6 + 540t^7 - 315t^8 + 70t^9"""
    return 126 * t**5 - 420 * t**6 + 540 * t**7 - 315 * t**8 + 70 * t**9


SMOOTHSTEP_FNS = {1: smoothstep_1, 2: smoothstep_2, 3: smoothstep_3, 4: smoothstep_4}


def smoothstep_N(t, N):
    """Dispatch to the appropriate smoothstep polynomial."""
    if N not in SMOOTHSTEP_FNS:
        raise ValueError(f"Unsupported smoothstep order N={N}. Supported: {list(SMOOTHSTEP_FNS.keys())}")
    return SMOOTHSTEP_FNS[N](t)


# -----------------------------------------------------------------------------
# One-sided compact ramp rho_N(s): 0 for s<=0, S_N(s) for 0<s<1, 1 for s>=1
# -----------------------------------------------------------------------------

def rho_N(s, N):
    """One-sided compact ramp with C^N continuity."""
    s_clamped = np.clip(s, 0.0, 1.0)
    return smoothstep_N(s_clamped, N)


# -----------------------------------------------------------------------------
# 1D window omega_ij(X_j) for region i, dim j
# -----------------------------------------------------------------------------

def omega_1d(X_j, a, b, alpha, N):
    """
    1D window for a region [a, b] with collar fraction alpha.
    
    delta = alpha * (b - a)  (collar half-width)
    s_lo = (X_j - (a - delta)) / delta   # 0 at a-delta, 1 at a
    s_hi = ((b + delta) - X_j) / delta   # 1 at b, 0 at b+delta
    omega = rho_N(s_lo) * rho_N(s_hi)
    """
    delta = alpha * (b - a)
    s_lo = (X_j - (a - delta)) / delta
    s_hi = ((b + delta) - X_j) / delta
    return rho_N(s_lo, N) * rho_N(s_hi, N)


# -----------------------------------------------------------------------------
# Region indicator Psi_i(X) = product over all dims
# -----------------------------------------------------------------------------

def psi_region(X, bounds_lower, bounds_upper, alpha, N):
    """
    Tensor-product region indicator.
    
    X: array of shape (..., d) where d is the number of dimensions
    bounds_lower: array of shape (d,) — lower bounds [a_1, ..., a_d]
    bounds_upper: array of shape (d,) — upper bounds [b_1, ..., b_d]
    alpha: collar fraction
    N: smoothness order
    
    Returns: array of shape (...) — the indicator value at each point
    """
    d = len(bounds_lower)
    result = np.ones(X.shape[:-1])
    for j in range(d):
        result *= omega_1d(X[..., j], bounds_lower[j], bounds_upper[j], alpha, N)
    return result


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def plot_window_demo(output_dir=None):
    """
    Plot the compact smoothstep window for a dummy region [0.2, 0.8]^2 on [0,1]^2.
    
    Generates:
    - 2D heatmaps for N = 1, 2, 4
    - 1D slices along x at y=0.5 for N = 1, 2, 4
    """
    # Region and collar parameters
    a, b = 0.2, 0.8  # region bounds (same in x and y)
    alpha = 0.2  # collar fraction (sigma_fraction)
    delta = alpha * (b - a)  # = 0.2 * 0.6 = 0.12
    
    # Grid
    resolution = 500
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X_grid, Y_grid = np.meshgrid(x, y)
    XY = np.stack([X_grid, Y_grid], axis=-1)  # shape (res, res, 2)
    
    bounds_lower = np.array([a, a])
    bounds_upper = np.array([b, b])
    
    orders = [1, 2, 4]
    
    # Expected zones for sanity check
    inner_start = a
    inner_end = b
    outer_start = a - delta  # 0.2 - 0.12 = 0.08
    outer_end = b + delta    # 0.8 + 0.12 = 0.92
    
    print(f"Region: [{a}, {b}] x [{a}, {b}]")
    print(f"Collar fraction (alpha): {alpha}")
    print(f"Collar width (delta): {delta:.4f}")
    print(f"Expected zones:")
    print(f"  - Flat-top (=1): [{inner_start}, {inner_end}]")
    print(f"  - Ramp collars: [{outer_start:.2f}, {inner_start}] and [{inner_end}, {outer_end:.2f}]")
    print(f"  - Exactly 0 outside: [0, {outer_start:.2f}) and ({outer_end:.2f}, 1]")
    print()
    
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs" / "window_demo"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 2D heatmaps ---
    fig, axes = plt.subplots(1, len(orders), figsize=(4 * len(orders), 4))
    for idx, N in enumerate(orders):
        psi = psi_region(XY, bounds_lower, bounds_upper, alpha, N)
        ax = axes[idx]
        im = ax.imshow(psi, origin='lower', extent=[0, 1, 0, 1], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'$\\Psi_i(X)$, N={N} (C$^{N}$)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # Mark region boundaries
        ax.axvline(a, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axvline(b, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(a, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(b, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
        # Mark collar outer boundaries
        ax.axvline(outer_start, color='red', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.axvline(outer_end, color='red', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.axhline(outer_start, color='red', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.axhline(outer_end, color='red', linestyle=':', linewidth=0.8, alpha=0.7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Compact Smoothstep Windows (region [{a},{b}]², alpha={alpha})', y=1.02)
    plt.tight_layout()
    heatmap_path = output_dir / "window_2d_heatmaps.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"Saved 2D heatmaps: {heatmap_path}")
    plt.close()
    
    # --- 1D slices along x at y=0.5 ---
    y_slice = 0.5
    x_1d = np.linspace(0, 1, 1000)
    XY_1d = np.stack([x_1d, np.full_like(x_1d, y_slice)], axis=-1)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    for N in orders:
        psi_1d = psi_region(XY_1d, bounds_lower, bounds_upper, alpha, N)
        ax.plot(x_1d, psi_1d, label=f'N={N} (C$^{N}$)', linewidth=2)
    
    ax.axvline(a, color='gray', linestyle='--', linewidth=1, label=f'Region boundary (a={a})')
    ax.axvline(b, color='gray', linestyle='--', linewidth=1, label=f'Region boundary (b={b})')
    ax.axvline(outer_start, color='red', linestyle=':', linewidth=1, label=f'Collar edge ({outer_start:.2f})')
    ax.axvline(outer_end, color='red', linestyle=':', linewidth=1, label=f'Collar edge ({outer_end:.2f})')
    
    # Shade the regions
    ax.axvspan(0, outer_start, alpha=0.1, color='blue', label='Exact 0 zone')
    ax.axvspan(outer_end, 1, alpha=0.1, color='blue')
    ax.axvspan(outer_start, a, alpha=0.1, color='orange', label='Collar (ramp)')
    ax.axvspan(b, outer_end, alpha=0.1, color='orange')
    ax.axvspan(a, b, alpha=0.1, color='green', label='Flat-top (=1)')
    
    ax.set_xlabel('x')
    ax.set_ylabel(f'$\\Psi_i(x, y={y_slice})$')
    ax.set_title(f'1D Slice of Compact Smoothstep Window at y={y_slice}')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    slice_path = output_dir / "window_1d_slice.png"
    plt.savefig(slice_path, dpi=150, bbox_inches='tight')
    print(f"Saved 1D slice: {slice_path}")
    plt.close()
    
    # --- Sanity checks ---
    print("\n--- Sanity Checks ---")
    for N in orders:
        psi_1d = psi_region(XY_1d, bounds_lower, bounds_upper, alpha, N)
        
        # Check flat-top region (inside [a, b] for both x and y, but y=0.5 is inside)
        inside_mask = (x_1d >= a) & (x_1d <= b)
        inside_vals = psi_1d[inside_mask]
        flat_top_ok = np.allclose(inside_vals, 1.0, atol=1e-10)
        
        # Check exactly zero outside collar
        outside_mask = (x_1d < outer_start) | (x_1d > outer_end)
        outside_vals = psi_1d[outside_mask]
        exact_zero_ok = np.allclose(outside_vals, 0.0, atol=1e-14)
        
        # Check collar ramps are in (0, 1)
        left_collar_mask = (x_1d > outer_start) & (x_1d < a)
        right_collar_mask = (x_1d > b) & (x_1d < outer_end)
        collar_mask = left_collar_mask | right_collar_mask
        collar_vals = psi_1d[collar_mask]
        collar_ok = np.all((collar_vals > 0) & (collar_vals < 1))
        
        status = "PASS" if (flat_top_ok and exact_zero_ok and collar_ok) else "FAIL"
        print(f"N={N}: flat-top={flat_top_ok}, exact-zero={exact_zero_ok}, collar-ramp={collar_ok} -> {status}")
    
    print(f"\nAll outputs saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    plot_window_demo()
