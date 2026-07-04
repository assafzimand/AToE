"""Visualization for linear probe analysis."""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch


def _safe_log_scale(ax, values_list):
    """Set log scale on y-axis only if all data has positive values.
    
    Returns:
        bool: True if log scale was applied, False if linear scale is used.
    """
    all_values = []
    for v in values_list:
        if isinstance(v, (list, np.ndarray)):
            all_values.extend(np.array(v).flatten())
        else:
            all_values.append(v)
    all_values = np.array(all_values)
    # Filter out NaN values for the check
    valid_values = all_values[~np.isnan(all_values)]
    if len(valid_values) > 0 and np.all(valid_values > 0):
        ax.set_yscale('log')
        return True
    return False


def plot_probe_metrics(probe_results: Dict, save_dir: Path):
    """
    Create line plots showing how probe metrics evolve across layers.
    
    Args:
        probe_results: Dictionary from probe_all_layers()
                      {layer_name: {'train_metrics', 'eval_metrics', ...}}
        save_dir: Directory to save plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract layer numbers and metrics
    layer_names = sorted(probe_results.keys())  # e.g., ['layer_1', 'layer_2', ...]
    layer_numbers = [int(name.split('_')[1]) for name in layer_names]
    
    train_rel_l2 = [probe_results[ln]['train_metrics']['rel_l2'] for ln in layer_names]
    train_inf_norm = [probe_results[ln]['train_metrics']['inf_norm'] for ln in layer_names]
    eval_rel_l2 = [probe_results[ln]['eval_metrics']['rel_l2'] for ln in layer_names]
    eval_inf_norm = [probe_results[ln]['eval_metrics']['inf_norm'] for ln in layer_names]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Relative L2 Error
    ax1.plot(layer_numbers, train_rel_l2, 'o-', linewidth=2, markersize=8, 
             label='Train', color='#3498db')
    ax1.plot(layer_numbers, eval_rel_l2, 's-', linewidth=2, markersize=8, 
             label='Eval', color='#e74c3c')
    ax1.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Relative L2 Error', fontsize=12, fontweight='bold')
    ax1.set_title('Linear Probe: Relative L2 Error vs Layer', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(layer_numbers)
    
    # Plot 2: Infinity Norm Error
    ax2.plot(layer_numbers, train_inf_norm, 'o-', linewidth=2, markersize=8, 
             label='Train', color='#3498db')
    ax2.plot(layer_numbers, eval_inf_norm, 's-', linewidth=2, markersize=8, 
             label='Eval', color='#e74c3c')
    ax2.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Infinity Norm Error', fontsize=12, fontweight='bold')
    ax2.set_title('Linear Probe: Infinity Norm Error vs Layer', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(layer_numbers)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = save_dir / "probe_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Probe metrics plot saved to {plot_path}")


def plot_layer_dimensions(probe_results: Dict, save_dir: Path):
    """
    Plot the dimensionality of each layer alongside probe performance.
    
    Args:
        probe_results: Dictionary from probe_all_layers()
        save_dir: Directory to save plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    layer_names = sorted(probe_results.keys())
    layer_numbers = [int(name.split('_')[1]) for name in layer_names]
    dimensions = [probe_results[ln]['hidden_dim'] for ln in layer_names]
    eval_rel_l2 = [probe_results[ln]['eval_metrics']['rel_l2'] for ln in layer_names]
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot dimensions on left axis
    color = '#2ecc71'
    ax1.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Layer Dimension', fontsize=12, fontweight='bold', color=color)
    ax1.bar(layer_numbers, dimensions, alpha=0.6, color=color, label='Dimension')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(layer_numbers)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot performance on right axis
    ax2 = ax1.twinx()
    color = '#e74c3c'
    ax2.set_ylabel('Probe Eval Rel-L2 Error', fontsize=12, fontweight='bold', color=color)
    ax2.plot(layer_numbers, eval_rel_l2, 'o-', linewidth=2, markersize=8, 
             color=color, label='Eval Error')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Layer Dimension vs Probe Performance', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    # Save plot
    plot_path = save_dir / "probe_dimensions.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Layer dimensions plot saved to {plot_path}")


def plot_probe_error_heatmaps(
    probe_results: Dict,
    eval_x: np.ndarray,
    eval_t: np.ndarray,
    eval_targets: np.ndarray,
    save_dir: Path
):
    """
    Generate error heatmaps for probe predictions (similar to residual heatmaps).
    
    Args:
        probe_results: Dictionary from probe_all_layers()
        eval_x: Spatial coordinates (N,)
        eval_t: Temporal coordinates (N,)
        eval_targets: Ground truth outputs (N, 2) - [u, v]
        save_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    
    save_dir.mkdir(parents=True, exist_ok=True)
    layer_names = sorted(probe_results.keys())
    
    # Check spatial dimension - for 2D spatial, skip heatmaps or use different visualization
    spatial_dim = eval_x.shape[1] if eval_x.ndim > 1 else 1
    if spatial_dim > 1:
        print(f"  Skipping probe error heatmaps (not supported for {spatial_dim}D spatial)")
        return
    
    # Flatten for 1D spatial
    eval_x_flat = eval_x.flatten() if eval_x.ndim > 1 else eval_x
    eval_t_flat = eval_t.flatten() if eval_t.ndim > 1 else eval_t
    
    for idx, layer_name in enumerate(layer_names, 1):
        print(f"  Generating error heatmaps for {layer_name} ({idx}/{len(layer_names)})")
        
        # Get predictions for this layer (ensure it's on CPU and numpy)
        eval_preds = probe_results[layer_name]['eval_predictions']  # (N, output_dim)
        if isinstance(eval_preds, torch.Tensor):
            eval_preds = eval_preds.cpu().numpy()
        output_dim = eval_preds.shape[1]
        
        # Compute errors
        error = eval_preds - eval_targets  # (N, output_dim)
        
        # Create grid for interpolation
        x_min, x_max = eval_x_flat.min(), eval_x_flat.max()
        t_min, t_max = eval_t_flat.min(), eval_t_flat.max()
        grid_x, grid_t = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(t_min, t_max, 200)
        )
        
        points = np.column_stack([eval_x_flat, eval_t_flat])
        
        # Create figure with output_dim subplots
        fig, axes = plt.subplots(output_dim, 1, figsize=(12, 5*output_dim))
        if output_dim == 1:
            axes = [axes]
        
        # Plot error for each output component
        if output_dim == 1:
            component_names = ['h(x,t)']
        elif output_dim == 2:
            component_names = ['u (real)', 'v (imag)']
        else:
            component_names = [f'h_{i}' for i in range(output_dim)]
        
        for i in range(output_dim):
            error_i = error[:, i]
            grid_error_i = griddata(points, error_i, (grid_x, grid_t),
                                    method='linear', fill_value=0.0)
            
            # Check if grid is valid for contourf
            if grid_error_i.shape[0] < 2 or grid_error_i.shape[1] < 2 or np.all(np.isnan(grid_error_i)):
                # Fall back to scatter plot
                axes[i].scatter(eval_x_flat, eval_t_flat, c=error_i, cmap='RdBu_r', s=10, alpha=0.6)
                axes[i].text(0.5, 0.95, '(Scatter - insufficient data for contour)',
                           ha='center', va='top', transform=axes[i].transAxes,
                           fontsize=9, color='gray')
            else:
                im = axes[i].contourf(grid_x, grid_t, grid_error_i, levels=50, cmap='RdBu_r')
                plt.colorbar(im, ax=axes[i])
            
            axes[i].set_xlabel('x', fontsize=11)
            axes[i].set_ylabel('t', fontsize=11)
            axes[i].set_title(f'Probe Error - {component_names[i]}', fontsize=12, fontweight='bold')
        
        # Overall title
        fig.suptitle(f'Probe Prediction Error Heatmaps - {layer_name}', 
                     fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        save_path = save_dir / f'probe_error_heatmaps_{layer_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    print(f"  Probe error heatmaps saved to {save_dir}")


def plot_probe_error_change_heatmaps(
    probe_results: Dict,
    eval_x: np.ndarray,
    eval_t: np.ndarray,
    eval_targets: np.ndarray,
    save_dir: Path
):
    """
    Plot probe error change heatmaps showing how errors evolve between layers.
    
    For each transition, computes: abs(error_{i+1}) / abs(error_i)
    - 0.0 (green): Error eliminated
    - 1.0 (white): No change
    - >1.0 (red): Error increased
    
    Args:
        probe_results: Dictionary from probe_all_layers()
        eval_x: Spatial coordinates (N,)
        eval_t: Temporal coordinates (N,)
        eval_targets: Ground truth outputs (N, 2) - [u, v]
        save_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from matplotlib.colors import TwoSlopeNorm
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Check spatial dimension - for 2D spatial, skip heatmaps
    spatial_dim = eval_x.shape[1] if eval_x.ndim > 1 else 1
    if spatial_dim > 1:
        print(f"  Skipping probe error change heatmaps (not supported for {spatial_dim}D spatial)")
        return
    
    layer_names = sorted(probe_results.keys(),
                        key=lambda x: int(x.split('_')[1]))
    n_layers = len(layer_names)
    
    if n_layers < 2:
        return  # Need at least 2 layers for changes
    
    # Flatten for 1D spatial
    eval_x_flat = eval_x.flatten() if eval_x.ndim > 1 else eval_x
    eval_t_flat = eval_t.flatten() if eval_t.ndim > 1 else eval_t
    
    # Pre-compute error grids for all layers
    x_min, x_max = eval_x_flat.min(), eval_x_flat.max()
    t_min, t_max = eval_t_flat.min(), eval_t_flat.max()
    grid_x, grid_t = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(t_min, t_max, 200)
    )
    
    points = np.column_stack([eval_x_flat, eval_t_flat])
    error_grids = {}
    
    # Determine output dimension from first layer
    first_layer = layer_names[0]
    output_dim = probe_results[first_layer]['eval_predictions'].shape[1]
    
    for layer_name in layer_names:
        eval_preds = probe_results[layer_name]['eval_predictions']  # (N, output_dim)
        if isinstance(eval_preds, torch.Tensor):
            eval_preds = eval_preds.cpu().numpy()
        error = eval_preds - eval_targets  # (N, output_dim)
        
        layer_grid = {}
        for i in range(output_dim):
            error_i = error[:, i]
            grid_error_i = griddata(points, error_i, (grid_x, grid_t),
                                    method='linear', fill_value=0.0)
            layer_grid[f'error_{i}'] = grid_error_i
        
        error_grids[layer_name] = layer_grid
    
    # Create figure for changes (N-1 transitions, output_dim subplots each)
    # Layout: stack components vertically, arrange transitions to make figure square
    n_changes = n_layers - 1
    
    # Calculate optimal grid layout for roughly square figure
    # Each transition needs output_dim rows (one per component)
    import math
    n_cols_transitions = max(1, int(math.ceil(math.sqrt(n_changes))))
    n_rows_groups = int(math.ceil(n_changes / n_cols_transitions))
    n_rows_total = n_rows_groups * output_dim  # output_dim rows per transition group
    
    fig, axes = plt.subplots(n_rows_total, n_cols_transitions,
                             figsize=(6 * n_cols_transitions, 5 * n_rows_groups))
    fig.suptitle('Probe Error Changes Between Layers\n'
                 '(Ratio: |error_{i+1}| / |error_i|, Green=Improved, Red=Worse)',
                 fontsize=14, fontweight='bold')
    
    # Ensure axes is 2D
    if n_rows_total == 1 and n_cols_transitions == 1:
        axes = np.array([[axes]])
    elif n_rows_total == 1:
        axes = axes.reshape(1, -1)
    elif n_cols_transitions == 1:
        axes = axes.reshape(-1, 1)
    
    # Define colormap normalization (center at 1.0)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=3.0)
    
    # Component names
    if output_dim == 1:
        component_names = ['h(x,t)']
    elif output_dim == 2:
        component_names = ['u (real)', 'v (imag)']
    else:
        component_names = [f'h_{i}' for i in range(output_dim)]
    
    for idx in range(n_changes):
        layer_prev = layer_names[idx]
        layer_curr = layer_names[idx + 1]
        
        # Calculate position in grid
        col_idx = idx % n_cols_transitions
        row_group = idx // n_cols_transitions
        
        # Plot change for each output component
        for comp_idx in range(output_dim):
            prev_error = error_grids[layer_prev][f'error_{comp_idx}']
            curr_error = error_grids[layer_curr][f'error_{comp_idx}']
            
            # Compute ratio: abs(curr) / abs(prev)
            eps = 1e-10
            ratio = np.abs(curr_error) / (np.abs(prev_error) + eps)
            ratio = np.clip(ratio, 0, 5)  # Cap extreme values
            
            # Get subplot position
            row_idx = row_group * output_dim + comp_idx
            ax = axes[row_idx, col_idx]
            
            # Check if grid is valid for contourf
            if ratio.shape[0] < 2 or ratio.shape[1] < 2 or np.all(np.isnan(ratio)):
                # Skip this subplot if data is insufficient
                ax.text(0.5, 0.5, 'Insufficient data',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='gray')
                ax.axis('off')
                continue
            
            # Plot
            im = ax.contourf(grid_x, grid_t, ratio, levels=50,
                           cmap='RdYlGn_r', norm=norm)
            ax.set_xlabel('x', fontsize=11)
            ax.set_ylabel('t', fontsize=11)
            ax.set_title(f'{layer_prev} -> {layer_curr}\n{component_names[comp_idx]} error',
                        fontsize=12, fontweight='bold')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Error Ratio', fontsize=10)
    
    # Hide unused subplots if any
    for row in range(n_rows_total):
        for col in range(n_cols_transitions):
            trans_idx = (row // 2) * n_cols_transitions + col
            if trans_idx >= n_changes:
                axes[row, col].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = save_dir / 'probe_error_change_heatmaps.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Probe error change heatmaps saved to {save_path}")


# Time slices for 2D spatial visualization
TIME_SLICES_2D = [0.0, 0.5, 1.0, 1.5, 2.0]


def plot_probe_error_heatmaps_2d(
    probe_results: Dict,
    eval_x: np.ndarray,
    eval_t: np.ndarray,
    eval_targets: np.ndarray,
    save_dir: Path
):
    """
    Generate probe error heatmaps for 2D spatial problems.
    
    Dual visualization:
    - Method A: 3D scatter (x0, x1, t) colored by error
    - Method B: 2D time-slice heatmaps
    
    Args:
        probe_results: Dictionary from probe_all_layers()
        eval_x: Spatial coordinates (N, 2)
        eval_t: Temporal coordinates (N, 1)
        eval_targets: Ground truth outputs (N, output_dim)
        save_dir: Directory to save plots
    """
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.gridspec import GridSpec
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    layer_names = sorted(probe_results.keys())
    n_layers = len(layer_names)
    
    # Get coordinates
    x0 = eval_x[:, 0] if eval_x.ndim > 1 else eval_x.flatten()
    x1 = eval_x[:, 1] if eval_x.ndim > 1 else np.zeros_like(x0)
    t_flat = eval_t.flatten() if eval_t.ndim > 1 else eval_t
    
    # Get output dimension
    output_dim = eval_targets.shape[1] if eval_targets.ndim > 1 else 1
    
    # =========================================================================
    # Method A: 3D Scatter - one figure with all layers
    # =========================================================================
    fig = plt.figure(figsize=(6*n_layers, 5))
    
    subsample = max(1, len(x0) // 2000)
    
    for idx, layer_name in enumerate(layer_names):
        eval_preds = probe_results[layer_name]['eval_predictions']
        if isinstance(eval_preds, torch.Tensor):
            eval_preds = eval_preds.cpu().numpy()
        
        # Compute absolute error magnitude
        error = np.abs(eval_preds - eval_targets)
        error_magnitude = np.sqrt(np.sum(error**2, axis=1)) if error.ndim > 1 else np.abs(error)
        
        ax = fig.add_subplot(1, n_layers, idx+1, projection='3d')
        sc = ax.scatter(
            x0[::subsample], x1[::subsample], t_flat[::subsample],
            c=error_magnitude[::subsample], cmap='Reds', s=2, alpha=0.5
        )
        ax.set_xlabel('x0', fontsize=9)
        ax.set_ylabel('x1', fontsize=9)
        ax.set_zlabel('t', fontsize=9)
        ax.set_title(f'{layer_name}', fontsize=11, fontweight='bold')
        plt.colorbar(sc, ax=ax, shrink=0.5, label='|error|')
    
    plt.suptitle('Probe Prediction Error (3D)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path_3d = save_dir / 'probe_error_3d.png'
    plt.savefig(save_path_3d, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Probe error 3D scatter saved to {save_path_3d}")
    
    # =========================================================================
    # Method B: Time Slices - one figure per layer (continuous contourf)
    # =========================================================================
    from scipy.interpolate import griddata
    
    n_grid = 50
    x0_grid_lin = np.linspace(0, 1, n_grid)
    x1_grid_lin = np.linspace(0, 1, n_grid)
    X0_grid, X1_grid = np.meshgrid(x0_grid_lin, x1_grid_lin)
    
    for layer_name in layer_names:
        eval_preds = probe_results[layer_name]['eval_predictions']
        if isinstance(eval_preds, torch.Tensor):
            eval_preds = eval_preds.cpu().numpy()
        
        error = np.abs(eval_preds - eval_targets)
        error_magnitude = np.sqrt(np.sum(error**2, axis=1)) if error.ndim > 1 else np.abs(error)
        
        fig = plt.figure(figsize=(22, 4))
        gs = GridSpec(1, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.08], wspace=0.25)
        
        cf_ref = None
        vmax = error_magnitude.max()
        
        for col_idx, t_slice in enumerate(TIME_SLICES_2D):
            ax = fig.add_subplot(gs[0, col_idx])
            
            t_tol = 0.15
            mask = np.abs(t_flat - t_slice) < t_tol
            
            if mask.sum() < 5:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f't = {t_slice}', fontsize=11, fontweight='bold')
                continue
            
            x0_slice = x0[mask]
            x1_slice = x1[mask]
            error_slice = error_magnitude[mask]
            
            # Interpolate to grid
            error_grid = griddata((x0_slice, x1_slice), error_slice, (X0_grid, X1_grid), method='linear')
            
            cf = ax.contourf(X0_grid, X1_grid, error_grid, levels=50, cmap='Reds', 
                            vmin=0, vmax=vmax)
            ax.set_title(f't = {t_slice}', fontsize=11, fontweight='bold')
            ax.set_xlabel('x0', fontsize=9)
            ax.set_ylabel('x1', fontsize=9)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_aspect('equal')
            
            if cf_ref is None:
                cf_ref = cf
        
        # Add colorbar
        cbar_ax = fig.add_subplot(gs[0, 5])
        fig.colorbar(cf_ref, cax=cbar_ax, label='|error|')
        
        plt.suptitle(f'Probe Error - {layer_name} (Time Slices)', fontsize=14, fontweight='bold', y=0.98)
        
        save_path = save_dir / f'probe_error_slices_{layer_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Probe error time slices saved to {save_dir}")


def plot_probe_error_change_heatmaps_2d(
    probe_results: Dict,
    eval_x: np.ndarray,
    eval_t: np.ndarray,
    eval_targets: np.ndarray,
    save_dir: Path
):
    """
    Plot probe error change heatmaps for 2D spatial problems.
    
    Shows how error changes between consecutive layers.
    Dual visualization: 3D scatter and 2D time slices.
    
    Args:
        probe_results: Dictionary from probe_all_layers()
        eval_x: Spatial coordinates (N, 2)
        eval_t: Temporal coordinates (N, 1)
        eval_targets: Ground truth outputs (N, output_dim)
        save_dir: Directory to save plots
    """
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import TwoSlopeNorm
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    layer_names = sorted(probe_results.keys(), key=lambda x: int(x.split('_')[1]))
    n_layers = len(layer_names)
    
    if n_layers < 2:
        print("  Skipping probe error change plots (need at least 2 layers)")
        return
    
    n_transitions = n_layers - 1
    
    # Get coordinates
    x0 = eval_x[:, 0] if eval_x.ndim > 1 else eval_x.flatten()
    x1 = eval_x[:, 1] if eval_x.ndim > 1 else np.zeros_like(x0)
    t_flat = eval_t.flatten() if eval_t.ndim > 1 else eval_t
    
    # Pre-compute error magnitudes for all layers
    error_magnitudes = {}
    for layer_name in layer_names:
        eval_preds = probe_results[layer_name]['eval_predictions']
        if isinstance(eval_preds, torch.Tensor):
            eval_preds = eval_preds.cpu().numpy()
        error = np.abs(eval_preds - eval_targets)
        error_magnitudes[layer_name] = np.sqrt(np.sum(error**2, axis=1)) if error.ndim > 1 else np.abs(error)
    
    # =========================================================================
    # Method A: 3D Scatter - error ratio for each transition
    # =========================================================================
    fig = plt.figure(figsize=(6*n_transitions, 5))
    
    subsample = max(1, len(x0) // 2000)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=3.0)
    
    for idx in range(n_transitions):
        layer_prev = layer_names[idx]
        layer_curr = layer_names[idx + 1]
        
        # Compute error ratio
        eps = 1e-10
        ratio = error_magnitudes[layer_curr] / (error_magnitudes[layer_prev] + eps)
        ratio = np.clip(ratio, 0, 5)
        
        ax = fig.add_subplot(1, n_transitions, idx+1, projection='3d')
        sc = ax.scatter(
            x0[::subsample], x1[::subsample], t_flat[::subsample],
            c=ratio[::subsample], cmap='RdYlGn_r', s=2, alpha=0.5, norm=norm
        )
        ax.set_xlabel('x0', fontsize=9)
        ax.set_ylabel('x1', fontsize=9)
        ax.set_zlabel('t', fontsize=9)
        ax.set_title(f'{layer_prev}→{layer_curr}', fontsize=11, fontweight='bold')
        plt.colorbar(sc, ax=ax, shrink=0.5, label='Error Ratio')
    
    plt.suptitle('Probe Error Changes (3D)\n(Green=Improved, Red=Worse)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path_3d = save_dir / 'probe_error_changes_3d.png'
    plt.savefig(save_path_3d, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Probe error changes 3D saved to {save_path_3d}")
    
    # =========================================================================
    # Method B: Time Slices - transitions x time_slices grid (continuous contourf)
    # =========================================================================
    from scipy.interpolate import griddata
    
    n_grid = 50
    x0_grid_lin = np.linspace(0, 1, n_grid)
    x1_grid_lin = np.linspace(0, 1, n_grid)
    X0_grid, X1_grid = np.meshgrid(x0_grid_lin, x1_grid_lin)
    
    fig = plt.figure(figsize=(22, 4*n_transitions))
    gs = GridSpec(n_transitions, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.08], wspace=0.25, hspace=0.3)
    
    cf_ref = None
    
    for row_idx in range(n_transitions):
        layer_prev = layer_names[row_idx]
        layer_curr = layer_names[row_idx + 1]
        
        eps = 1e-10
        ratio = error_magnitudes[layer_curr] / (error_magnitudes[layer_prev] + eps)
        ratio = np.clip(ratio, 0, 5)
        
        for col_idx, t_slice in enumerate(TIME_SLICES_2D):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            t_tol = 0.15
            mask = np.abs(t_flat - t_slice) < t_tol
            
            if mask.sum() < 5:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes)
                continue
            
            x0_slice = x0[mask]
            x1_slice = x1[mask]
            ratio_slice = ratio[mask]
            
            # Interpolate to grid
            ratio_grid = griddata((x0_slice, x1_slice), ratio_slice, (X0_grid, X1_grid), method='linear')
            
            cf = ax.contourf(X0_grid, X1_grid, ratio_grid, levels=50, cmap='RdYlGn_r', 
                            norm=norm)
            
            if row_idx == 0:
                ax.set_title(f't = {t_slice}', fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'{layer_prev}→{layer_curr}\nx1', fontsize=9)
            ax.set_xlabel('x0', fontsize=9)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_aspect('equal')
            
            if cf_ref is None:
                cf_ref = cf
    
    # Add colorbar
    cbar_ax = fig.add_subplot(gs[:, 5])
    fig.colorbar(cf_ref, cax=cbar_ax, label='Error Ratio')
    
    plt.suptitle('Probe Error Changes (Time Slices)\n(Green=Improved, Red=Worse)', 
                fontsize=14, fontweight='bold', y=0.98)
    
    save_path_slices = save_dir / 'probe_error_changes_slices.png'
    plt.savefig(save_path_slices, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Probe error changes time slices saved to {save_path_slices}")


def generate_all_probe_plots(
    probe_results: Dict, 
    save_dir: Path,
    eval_x: np.ndarray = None,
    eval_t: np.ndarray = None,
    eval_targets: np.ndarray = None
):
    """
    Generate all probe visualization plots.
    
    Args:
        probe_results: Dictionary from probe_all_layers()
        save_dir: Directory to save plots
        eval_x: Optional spatial coordinates for error heatmaps
        eval_t: Optional temporal coordinates for error heatmaps
        eval_targets: Optional ground truth for error heatmaps
    """
    print("\nGenerating probe plots...")
    plot_probe_metrics(probe_results, save_dir)
    plot_layer_dimensions(probe_results, save_dir)
    
    # Generate error heatmaps if data is provided
    if eval_x is not None and eval_t is not None and eval_targets is not None:
        # Detect spatial dimension
        spatial_dim = eval_x.shape[1] if eval_x.ndim > 1 else 1
        
        if spatial_dim == 1:
            # 1D spatial - use existing heatmap functions
            plot_probe_error_heatmaps(probe_results, eval_x, eval_t, eval_targets, save_dir)
            plot_probe_error_change_heatmaps(probe_results, eval_x, eval_t, eval_targets, save_dir)
        else:
            # 2D spatial - use new 3D scatter + time slice functions
            plot_probe_error_heatmaps_2d(probe_results, eval_x, eval_t, eval_targets, save_dir)
            plot_probe_error_change_heatmaps_2d(probe_results, eval_x, eval_t, eval_targets, save_dir)
    
    print("  All probe plots generated")


def plot_probe_history_shaded(history: List[tuple], save_dir: Path) -> None:
    """
    Overlay probe metrics across epochs with shaded progression.
    history: list of (epoch, metrics_summary) where metrics_summary matches run_probes return.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    history = sorted(history, key=lambda x: x[0])
    base_color = '#e74c3c'
    alphas = [min(0.45 + 0.15 * idx, 1.0) for idx in range(len(history))]

    # Rel-L2
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    all_rel_l2 = []
    for (epoch, metrics), alpha in zip(history, alphas):
        layers = list(range(1, len(metrics['train']['rel_l2']) + 1))
        axes[0].plot(layers, metrics['train']['rel_l2'], marker='o', color=base_color, alpha=alpha, label=f"Epoch {epoch}")
        axes[1].plot(layers, metrics['eval']['rel_l2'], marker='s', color=base_color, alpha=alpha, label=f"Epoch {epoch}")
        all_rel_l2.extend(metrics['train']['rel_l2'])
        all_rel_l2.extend(metrics['eval']['rel_l2'])
    is_log = _safe_log_scale(axes[0], [all_rel_l2])
    _safe_log_scale(axes[1], [all_rel_l2])
    scale_str = "[log]" if is_log else "[linear]"
    axes[0].set_title(f'Train Rel-L2 {scale_str}')
    axes[1].set_title(f'Eval Rel-L2 {scale_str}')
    for ax in axes:
        ax.set_xlabel('Layer')
        ax.set_ylabel('Rel-L2')
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    scale_label = "(Log Scale)" if is_log else "(Linear Scale)"
    fig.suptitle(f'Probe Rel-L2 Metrics {scale_label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "probe_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()

    # L_inf
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    all_inf = []
    for (epoch, metrics), alpha in zip(history, alphas):
        layers = list(range(1, len(metrics['train']['inf_norm']) + 1))
        axes[0].plot(layers, metrics['train']['inf_norm'], marker='o', color=base_color, alpha=alpha, label=f"Epoch {epoch}")
        axes[1].plot(layers, metrics['eval']['inf_norm'], marker='s', color=base_color, alpha=alpha, label=f"Epoch {epoch}")
        all_inf.extend(metrics['train']['inf_norm'])
        all_inf.extend(metrics['eval']['inf_norm'])
    is_log = _safe_log_scale(axes[0], [all_inf])
    _safe_log_scale(axes[1], [all_inf])
    scale_str = "[log]" if is_log else "[linear]"
    axes[0].set_title(f'Train L∞ {scale_str}')
    axes[1].set_title(f'Eval L∞ {scale_str}')
    for ax in axes:
        ax.set_xlabel('Layer')
        ax.set_ylabel('L∞')
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    scale_label = "(Log Scale)" if is_log else "(Linear Scale)"
    fig.suptitle(f'Probe L∞ Metrics {scale_label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "probe_metrics_inf.png", dpi=150, bbox_inches='tight')
    plt.close()

