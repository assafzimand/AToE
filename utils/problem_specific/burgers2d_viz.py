"""
2D Burgers equation specific visualizations.

Provides dual visualization methods for all plots:
1. Method A: 3D scatter plots with colormap (axes: x0, x1, t; color: h)
2. Method B: 2D time-slice heatmaps at t = 0, 0.5, 1.0, 1.5, 2.0

Functions:
- visualize_dataset: Dataset visualization
- visualize_evaluation: Model evaluation (GT vs Pred vs Error)
- visualize_ncc_dataset: NCC dataset distribution
- visualize_ncc_classification: NCC classification accuracy
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, List
from matplotlib.colors import TwoSlopeNorm

# Time slices for Method B
TIME_SLICES = [0.0, 0.5, 1.0, 1.5, 2.0]


# =============================================================================
# DATASET VISUALIZATION
# =============================================================================

def visualize_dataset(data_dict: Dict, save_dir: Path, config: Dict, split_name: str):
    """
    Visualize Burgers2D dataset with dual methods.
    
    Method A: 3D scatter (x0, x1, t) colored by h, with IC/BC highlighted
    Method B: 2D continuous heatmaps at 5 time slices, with IC/BC highlighted
    
    Args:
        data_dict: Dataset dictionary with 'x', 't', 'h_gt', 'mask' tensors
        save_dir: Directory to save visualization
        config: Configuration dictionary
        split_name: Name of split ('training' or 'evaluation')
    """
    from scipy.interpolate import griddata
    from matplotlib.gridspec import GridSpec
    
    # Extract data
    x = data_dict['x'].cpu().numpy()  # (N, 2)
    t = data_dict['t'].cpu().numpy()  # (N, 1)
    h_gt = data_dict['h_gt'].cpu().numpy()  # (N, 1)
    
    x0 = x[:, 0]
    x1 = x[:, 1]
    t_flat = t[:, 0]
    h = h_gt[:, 0]
    
    # Get IC/BC masks if available
    has_masks = 'mask' in data_dict
    if has_masks:
        mask_ic = data_dict['mask']['IC'].cpu().numpy().astype(bool)
        mask_bc = data_dict['mask']['BC'].cpu().numpy().astype(bool)
        mask_interior = ~(mask_ic | mask_bc)
    else:
        mask_interior = np.ones(len(x0), dtype=bool)
        mask_ic = np.zeros(len(x0), dtype=bool)
        mask_bc = np.zeros(len(x0), dtype=bool)
    
    # Color scale
    vmax = h.max()
    vmin = h.min()
    
    # =========================================================================
    # Method A: 3D Scatter with IC/BC highlighted
    # =========================================================================
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample for clarity
    subsample = max(1, len(x0) // 3000)
    
    # Plot interior points (smaller, semi-transparent)
    interior_idx = np.where(mask_interior)[0][::subsample]
    if len(interior_idx) > 0:
        scatter = ax.scatter(x0[interior_idx], x1[interior_idx], t_flat[interior_idx], 
                            c=h[interior_idx], cmap='viridis', s=2, alpha=0.4,
                            vmin=vmin, vmax=vmax, label='Interior')
    
    # Plot IC points (blue, larger)
    ic_idx = np.where(mask_ic)[0][::max(1, len(np.where(mask_ic)[0])//500)]
    if len(ic_idx) > 0:
        ax.scatter(x0[ic_idx], x1[ic_idx], t_flat[ic_idx], 
                  c='blue', s=15, alpha=0.9, marker='o', label='IC (t=0)')
    
    # Plot BC points (red, larger)
    bc_idx = np.where(mask_bc)[0][::max(1, len(np.where(mask_bc)[0])//500)]
    if len(bc_idx) > 0:
        ax.scatter(x0[bc_idx], x1[bc_idx], t_flat[bc_idx], 
                  c='red', s=15, alpha=0.9, marker='s', label='BC')
    
    ax.set_xlabel('x0', fontsize=12)
    ax.set_ylabel('x1', fontsize=12)
    ax.set_zlabel('t', fontsize=12)
    ax.set_title(f'Dataset Distribution - {split_name.capitalize()} (3D)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    
    if len(interior_idx) > 0:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.1)
        cbar.set_label('h', fontsize=11)
    
    save_path_3d = save_dir / f"dataset_{split_name.lower()}_3d.png"
    plt.savefig(save_path_3d, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {split_name.capitalize()} 3D scatter saved to {save_path_3d}")
    
    # =========================================================================
    # Method A2: 3D Surface plots (h as vertical axis) at different time slices
    # =========================================================================
    n_grid_surf = 40
    x0_surf = np.linspace(0, 1, n_grid_surf)
    x1_surf = np.linspace(0, 1, n_grid_surf)
    X0_surf, X1_surf = np.meshgrid(x0_surf, x1_surf)
    
    # Create 5 surface plots for each time slice
    fig = plt.figure(figsize=(20, 4))
    
    for idx, t_slice in enumerate(TIME_SLICES):
        ax = fig.add_subplot(1, 5, idx + 1, projection='3d')
        
        t_tol = 0.15
        mask_t = np.abs(t_flat - t_slice) < t_tol
        
        if mask_t.sum() > 5:
            x0_slice = x0[mask_t]
            x1_slice = x1[mask_t]
            h_slice = h[mask_t]
            
            # Interpolate to grid
            H_surf = griddata((x0_slice, x1_slice), h_slice, (X0_surf, X1_surf), method='linear')
            
            # Plot surface
            surf = ax.plot_surface(X0_surf, X1_surf, H_surf, cmap='viridis', 
                                  edgecolor='none', alpha=0.8, vmin=vmin, vmax=vmax)
            
            # Overlay IC points at t=0
            if t_slice == 0.0:
                ic_at_t = mask_ic & mask_t
                if ic_at_t.sum() > 0:
                    ic_sub = np.where(ic_at_t)[0][::max(1, ic_at_t.sum()//100)]
                    ax.scatter(x0[ic_sub], x1[ic_sub], h[ic_sub], 
                              c='blue', s=20, alpha=1.0, marker='o', label='IC')
            
            # Overlay BC points
            bc_at_t = mask_bc & mask_t
            if bc_at_t.sum() > 0:
                bc_sub = np.where(bc_at_t)[0][::max(1, bc_at_t.sum()//100)]
                ax.scatter(x0[bc_sub], x1[bc_sub], h[bc_sub], 
                          c='red', s=20, alpha=1.0, marker='s', label='BC')
        
        ax.set_xlabel('x0', fontsize=9)
        ax.set_ylabel('x1', fontsize=9)
        ax.set_zlabel('h', fontsize=9)
        ax.set_title(f't = {t_slice}', fontsize=11, fontweight='bold')
        ax.set_zlim([vmin, vmax])
    
    plt.suptitle(f'h(x0, x1) Surface - {split_name.capitalize()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path_surf = save_dir / f"dataset_{split_name.lower()}_3d_surface.png"
    plt.savefig(save_path_surf, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {split_name.capitalize()} 3D surface saved to {save_path_surf}")
    
    # =========================================================================
    # Method B: 2D Time Slices (continuous heatmap + IC/BC overlay)
    # =========================================================================
    n_grid = 50
    x0_grid_lin = np.linspace(0, 1, n_grid)
    x1_grid_lin = np.linspace(0, 1, n_grid)
    X0_grid, X1_grid = np.meshgrid(x0_grid_lin, x1_grid_lin)
    
    fig = plt.figure(figsize=(22, 4))
    gs = GridSpec(1, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.08], wspace=0.25)
    
    cf_ref = None
    
    for idx, t_slice in enumerate(TIME_SLICES):
        ax = fig.add_subplot(gs[0, idx])
        
        # Select points near this time slice
        t_tol = 0.15
        mask_t = np.abs(t_flat - t_slice) < t_tol
        
        if mask_t.sum() > 5:
            x0_slice = x0[mask_t]
            x1_slice = x1[mask_t]
            h_slice = h[mask_t]
            
            # Interpolate to grid
            h_grid = griddata((x0_slice, x1_slice), h_slice, (X0_grid, X1_grid), method='linear')
            
            cf = ax.contourf(X0_grid, X1_grid, h_grid, levels=50, cmap='viridis',
                            vmin=vmin, vmax=vmax)
            if cf_ref is None:
                cf_ref = cf
            
            # Overlay IC points (at t=0 slice only)
            if t_slice == 0.0:
                ic_at_t = mask_ic & mask_t
                if ic_at_t.sum() > 0:
                    ax.scatter(x0[ic_at_t], x1[ic_at_t], c='blue', s=8, alpha=0.8, 
                              marker='o', label='IC', edgecolors='white', linewidths=0.5)
            
            # Overlay BC points
            bc_at_t = mask_bc & mask_t
            if bc_at_t.sum() > 0:
                ax.scatter(x0[bc_at_t], x1[bc_at_t], c='red', s=8, alpha=0.8, 
                          marker='s', label='BC', edgecolors='white', linewidths=0.5)
        
        ax.set_xlabel('x0', fontsize=10)
        ax.set_ylabel('x1', fontsize=10)
        ax.set_title(f't = {t_slice}', fontsize=12, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
        
        # Add legend on first subplot
        if idx == 0 and (mask_ic.sum() > 0 or mask_bc.sum() > 0):
            ax.legend(loc='upper right', fontsize=8)
    
    # Add colorbar
    if cf_ref is not None:
        cbar_ax = fig.add_subplot(gs[0, 5])
        fig.colorbar(cf_ref, cax=cbar_ax, label='h')
    
    plt.suptitle(f'h(x0, x1) at Time Slices - {split_name.capitalize()}', 
                fontsize=14, fontweight='bold', y=0.98)
    
    save_path_slices = save_dir / f"dataset_{split_name.lower()}_slices.png"
    plt.savefig(save_path_slices, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {split_name.capitalize()} time slices saved to {save_path_slices}")


# =============================================================================
# EVALUATION VISUALIZATION
# =============================================================================

def visualize_evaluation(model: torch.nn.Module, eval_data_path: str, 
                        save_dir: Path, config: Dict):
    """
    Visualize Burgers2D model evaluation with dual methods.
    
    Method A: 3D scatter for GT, Pred, Error
    Method B: 5x3 grid of time-slice heatmaps (GT, Pred, Error) x 5 times
    
    Args:
        model: Trained neural network
        eval_data_path: Path to evaluation dataset
        save_dir: Directory to save visualization
        config: Configuration dictionary
    """
    from solvers.burgers2d_solver import analytical_solution
    
    # Get problem-specific config
    problem = config.get('problem', 'burgers2d')
    problem_config = config[problem]
    spatial_domain = problem_config['spatial_domain']
    temporal_domain = problem_config['temporal_domain']
    
    x0_min, x0_max = spatial_domain[0]
    x1_min, x1_max = spatial_domain[1]
    t_min, t_max = temporal_domain
    
    # Create dense evaluation grid
    n_x = 50  # Grid points per spatial dimension
    n_t = 50  # Time points
    
    x0_grid = np.linspace(x0_min, x0_max, n_x)
    x1_grid = np.linspace(x1_min, x1_max, n_x)
    t_grid = np.linspace(t_min, t_max, n_t)
    
    # Create 3D meshgrid
    X0, X1, T = np.meshgrid(x0_grid, x1_grid, t_grid, indexing='ij')
    
    # Flatten for model input (use model's dtype for precision compatibility)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    x0_flat = torch.tensor(X0.flatten(), dtype=dtype, device=device)
    x1_flat = torch.tensor(X1.flatten(), dtype=dtype, device=device)
    t_flat = torch.tensor(T.flatten(), dtype=dtype, device=device)
    
    # Model predictions
    model.eval()
    with torch.no_grad():
        x_input = torch.stack([x0_flat, x1_flat], dim=1)
        t_input = t_flat.view(-1, 1)
        xt_input = torch.cat([x_input, t_input], dim=1)
        h_pred = model(xt_input)[:, 0].cpu().numpy()
    
    # Ground truth
    h_gt = analytical_solution(X0.flatten(), X1.flatten(), T.flatten())
    
    # Compute error
    error = np.abs(h_pred - h_gt)
    
    # Reshape for visualization
    h_pred_3d = h_pred.reshape(n_x, n_x, n_t)
    h_gt_3d = h_gt.reshape(n_x, n_x, n_t)
    error_3d = error.reshape(n_x, n_x, n_t)
    
    # =========================================================================
    # Method A: 3D Scatter (subsampled for clarity)
    # =========================================================================
    subsample = 5  # Take every 5th point
    x0_sub = X0.flatten()[::subsample]
    x1_sub = X1.flatten()[::subsample]
    t_sub = T.flatten()[::subsample]
    h_gt_sub = h_gt[::subsample]
    h_pred_sub = h_pred[::subsample]
    error_sub = error[::subsample]
    
    fig = plt.figure(figsize=(18, 5))
    
    # GT
    ax1 = fig.add_subplot(131, projection='3d')
    sc1 = ax1.scatter(x0_sub, x1_sub, t_sub, c=h_gt_sub, cmap='viridis', s=1, alpha=0.5)
    ax1.set_xlabel('x0')
    ax1.set_ylabel('x1')
    ax1.set_zlabel('t')
    ax1.set_title('Ground Truth', fontsize=12, fontweight='bold')
    plt.colorbar(sc1, ax=ax1, shrink=0.5)
    
    # Pred
    ax2 = fig.add_subplot(132, projection='3d')
    sc2 = ax2.scatter(x0_sub, x1_sub, t_sub, c=h_pred_sub, cmap='viridis', s=1, alpha=0.5)
    ax2.set_xlabel('x0')
    ax2.set_ylabel('x1')
    ax2.set_zlabel('t')
    ax2.set_title('Prediction', fontsize=12, fontweight='bold')
    plt.colorbar(sc2, ax=ax2, shrink=0.5)
    
    # Error
    ax3 = fig.add_subplot(133, projection='3d')
    sc3 = ax3.scatter(x0_sub, x1_sub, t_sub, c=error_sub, cmap='Reds', s=1, alpha=0.5)
    ax3.set_xlabel('x0')
    ax3.set_ylabel('x1')
    ax3.set_zlabel('t')
    ax3.set_title('Error', fontsize=12, fontweight='bold')
    plt.colorbar(sc3, ax=ax3, shrink=0.5)
    
    plt.suptitle('Evaluation Comparison (3D Scatter)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path_3d = save_dir / "evaluation_comparison_3d.png"
    plt.savefig(save_path_3d, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Evaluation 3D scatter saved to {save_path_3d}")
    
    # =========================================================================
    # Method B: Time Slice Heatmaps (5x3 grid)
    # =========================================================================
    # Use gridspec for better control over colorbar placement
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(22, 12))
    # Main grid: 3 rows, 5 cols for plots + 1 col for colorbars
    gs = GridSpec(3, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.08], wspace=0.3, hspace=0.25)
    
    # Common color scales
    vmax_h = max(h_gt_3d.max(), h_pred_3d.max())
    vmin_h = min(h_gt_3d.min(), h_pred_3d.min())
    vmax_err = error_3d.max()
    
    X0_2d, X1_2d = np.meshgrid(x0_grid, x1_grid, indexing='ij')
    
    # Store one im reference per row for colorbars
    im_refs = [None, None, None]
    
    for col_idx, t_slice in enumerate(TIME_SLICES):
        # Find nearest time index
        t_idx = np.argmin(np.abs(t_grid - t_slice))
        
        # Row 0: Ground Truth
        ax_gt = fig.add_subplot(gs[0, col_idx])
        im_gt = ax_gt.contourf(X0_2d, X1_2d, h_gt_3d[:, :, t_idx], 
                              levels=50, cmap='viridis', vmin=vmin_h, vmax=vmax_h)
        ax_gt.set_title(f't = {t_slice}', fontsize=11, fontweight='bold')
        if col_idx == 0:
            ax_gt.set_ylabel('Ground Truth\nx1', fontsize=10)
        ax_gt.set_xlabel('x0', fontsize=9)
        ax_gt.set_aspect('equal')
        if col_idx == 0:
            im_refs[0] = im_gt
        
        # Row 1: Prediction
        ax_pred = fig.add_subplot(gs[1, col_idx])
        im_pred = ax_pred.contourf(X0_2d, X1_2d, h_pred_3d[:, :, t_idx], 
                                  levels=50, cmap='viridis', vmin=vmin_h, vmax=vmax_h)
        if col_idx == 0:
            ax_pred.set_ylabel('Prediction\nx1', fontsize=10)
        ax_pred.set_xlabel('x0', fontsize=9)
        ax_pred.set_aspect('equal')
        if col_idx == 0:
            im_refs[1] = im_pred
        
        # Row 2: Error
        ax_err = fig.add_subplot(gs[2, col_idx])
        im_err = ax_err.contourf(X0_2d, X1_2d, error_3d[:, :, t_idx], 
                                levels=50, cmap='Reds', vmin=0, vmax=vmax_err)
        if col_idx == 0:
            ax_err.set_ylabel('Error\nx1', fontsize=10)
        ax_err.set_xlabel('x0', fontsize=9)
        ax_err.set_aspect('equal')
        if col_idx == 0:
            im_refs[2] = im_err
    
    # Add colorbars in the last column
    cbar_ax0 = fig.add_subplot(gs[0, 5])
    fig.colorbar(im_refs[0], cax=cbar_ax0, label='h (GT)')
    
    cbar_ax1 = fig.add_subplot(gs[1, 5])
    fig.colorbar(im_refs[1], cax=cbar_ax1, label='h (Pred)')
    
    cbar_ax2 = fig.add_subplot(gs[2, 5])
    fig.colorbar(im_refs[2], cax=cbar_ax2, label='|error|')
    
    plt.suptitle('Evaluation Comparison (Time Slices)', fontsize=14, fontweight='bold', y=0.98)
    
    save_path_slices = save_dir / "evaluation_comparison_slices.png"
    plt.savefig(save_path_slices, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Evaluation time slices saved to {save_path_slices}")


# =============================================================================
# NCC DATASET VISUALIZATION
# =============================================================================

def visualize_ncc_dataset(ncc_data: Dict, dataset_dir: Path, config: Dict, prefix: str = ""):
    """
    Visualize NCC dataset distribution for Burgers2D.
    
    Method A: 3D scatter colored by h value
    Method B: 2D time-slice continuous heatmaps
    
    Args:
        ncc_data: NCC dataset dictionary
        dataset_dir: Directory to save visualization
        config: Configuration dictionary
        prefix: Prefix for filename
    """
    from scipy.interpolate import griddata
    from matplotlib.gridspec import GridSpec
    
    x = ncc_data['x'].cpu().numpy()  # (N, 2)
    t = ncc_data['t'].cpu().numpy()  # (N, 1)
    h_gt = ncc_data['h_gt'].cpu().numpy()  # (N, 1)
    
    x0 = x[:, 0]
    x1 = x[:, 1]
    t_flat = t[:, 0]
    h = h_gt[:, 0]
    
    vmin, vmax = h.min(), h.max()
    
    # =========================================================================
    # Method A: 3D Scatter
    # =========================================================================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    subsample = max(1, len(x0) // 3000)
    scatter = ax.scatter(x0[::subsample], x1[::subsample], t_flat[::subsample], 
                        c=h[::subsample], cmap='viridis', s=2, alpha=0.5)
    ax.set_xlabel('x0', fontsize=11)
    ax.set_ylabel('x1', fontsize=11)
    ax.set_zlabel('t', fontsize=11)
    ax.set_title('NCC Dataset Distribution (3D)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, shrink=0.6, label='h')
    
    save_path_3d = dataset_dir / f"{prefix}ncc_dataset_3d.png"
    plt.savefig(save_path_3d, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  NCC dataset 3D scatter saved to {save_path_3d}")
    
    # =========================================================================
    # Method B: 2D Time Slice Continuous Heatmaps
    # =========================================================================
    n_grid = 50
    x0_grid_lin = np.linspace(0, 1, n_grid)
    x1_grid_lin = np.linspace(0, 1, n_grid)
    X0_grid, X1_grid = np.meshgrid(x0_grid_lin, x1_grid_lin)
    
    fig = plt.figure(figsize=(22, 4))
    gs = GridSpec(1, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.08], wspace=0.25)
    
    cf_ref = None
    
    for idx, t_slice in enumerate(TIME_SLICES):
        ax = fig.add_subplot(gs[0, idx])
        
        t_tol = 0.15
        mask = np.abs(t_flat - t_slice) < t_tol
        
        if mask.sum() > 5:
            x0_slice = x0[mask]
            x1_slice = x1[mask]
            h_slice = h[mask]
            
            h_grid = griddata((x0_slice, x1_slice), h_slice, (X0_grid, X1_grid), method='linear')
            
            cf = ax.contourf(X0_grid, X1_grid, h_grid, levels=50, cmap='viridis',
                            vmin=vmin, vmax=vmax)
            if cf_ref is None:
                cf_ref = cf
        
        ax.set_xlabel('x0', fontsize=10)
        ax.set_ylabel('x1', fontsize=10)
        ax.set_title(f't = {t_slice}', fontsize=12, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
    
    if cf_ref is not None:
        cbar_ax = fig.add_subplot(gs[0, 5])
        fig.colorbar(cf_ref, cax=cbar_ax, label='h')
    
    plt.suptitle('NCC Dataset - h(x0, x1) at Time Slices', fontsize=14, fontweight='bold', y=0.98)
    
    save_path_slices = dataset_dir / f"{prefix}ncc_dataset_slices.png"
    plt.savefig(save_path_slices, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  NCC dataset slices saved to {save_path_slices}")
    
    # =========================================================================
    # Method C: Histogram of h values
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(h, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('h', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('NCC Dataset Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    save_path_hist = dataset_dir / f"{prefix}ncc_dataset_histogram.png"
    plt.savefig(save_path_hist, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  NCC dataset histogram saved to {save_path_hist}")


# =============================================================================
# NCC CLASSIFICATION VISUALIZATION
# =============================================================================

def visualize_ncc_classification(
    h_gt: np.ndarray,
    class_labels: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    bins: np.ndarray,
    save_path: Path
):
    """
    Visualize NCC classification in output space.
    
    Args:
        h_gt: Ground truth values (N, 1)
        class_labels: True class labels (N,)
        predictions_dict: Dict mapping layer_name -> predictions (N,)
        bins: Bin edges for classification
        save_path: Path to save figure
    """
    n_layers = len(predictions_dict)
    fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    
    h_gt_np = h_gt if isinstance(h_gt, np.ndarray) else h_gt.detach().cpu().numpy()
    labels_np = class_labels if isinstance(class_labels, np.ndarray) else class_labels.detach().cpu().numpy()
    indices = np.arange(len(h_gt_np))

    for ax, (layer_name, preds) in zip(axes, predictions_dict.items()):
        preds_np = preds if isinstance(preds, np.ndarray) else preds.detach().cpu().numpy()
        correct = (preds_np == labels_np).astype(float)
        colors = np.where(correct > 0.5, 'green', 'red')
        
        ax.scatter(h_gt_np[:, 0], indices, c=colors, s=2, alpha=0.5)
        ax.set_xlabel('h', fontsize=10)
        ax.set_ylabel('Sample index', fontsize=10)
        ax.set_title(f'{layer_name}\nAcc: {correct.mean():.3f}', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  NCC classification scatter saved to {save_path}")


def visualize_ncc_classification_input_space(
    x: np.ndarray,
    t: np.ndarray,
    class_labels: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    save_path: Path
):
    """
    Visualize NCC classification in input space with dual methods.
    
    Method A: 3D scatter (x0, x1, t) colored by correctness
    Method B: 2D time slices colored by correctness
    
    Args:
        x: Spatial coordinates (N, 2)
        t: Temporal coordinates (N, 1)
        class_labels: True class labels (N,)
        predictions_dict: Dict mapping layer_name -> predictions (N,)
        save_path: Path to save figure
    """
    n_layers = len(predictions_dict)
    
    x_np = x if isinstance(x, np.ndarray) else x.detach().cpu().numpy()
    t_np = t if isinstance(t, np.ndarray) else t.detach().cpu().numpy()
    labels_np = class_labels if isinstance(class_labels, np.ndarray) else class_labels.detach().cpu().numpy()
    
    x0 = x_np[:, 0]
    x1 = x_np[:, 1]
    t_flat = t_np[:, 0] if t_np.ndim > 1 else t_np
    
    # =========================================================================
    # Method A: 3D Scatter
    # =========================================================================
    fig = plt.figure(figsize=(6*n_layers, 5))
    
    for idx, (layer_name, preds) in enumerate(predictions_dict.items()):
        ax = fig.add_subplot(1, n_layers, idx+1, projection='3d')
        preds_np = preds if isinstance(preds, np.ndarray) else preds.detach().cpu().numpy()
        correct = (preds_np == labels_np).astype(float)
        colors = np.where(correct > 0.5, 'green', 'red')
        
        ax.scatter(x0, x1, t_flat, c=colors, s=2, alpha=0.5)
        ax.set_xlabel('x0', fontsize=9)
        ax.set_ylabel('x1', fontsize=9)
        ax.set_zlabel('t', fontsize=9)
        ax.set_title(f'{layer_name}\nAcc: {correct.mean():.3f}', fontsize=10)
    
    plt.suptitle('NCC Classification - Input Space (3D)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    save_path_3d = save_path.parent / (save_path.stem + '_3d.png')
    plt.savefig(save_path_3d, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  NCC classification 3D saved to {save_path_3d}")
    
    # =========================================================================
    # Method B: Time Slices
    # =========================================================================
    fig, axes = plt.subplots(n_layers, 5, figsize=(20, 4*n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, (layer_name, preds) in enumerate(predictions_dict.items()):
        preds_np = preds if isinstance(preds, np.ndarray) else preds.detach().cpu().numpy()
        correct = (preds_np == labels_np).astype(float)
        colors = np.where(correct > 0.5, 'green', 'red')
        
        for col_idx, t_slice in enumerate(TIME_SLICES):
            ax = axes[row_idx, col_idx]
            
            # Select points near this time slice
            t_tol = 0.15
            mask = np.abs(t_flat - t_slice) < t_tol
            
            if mask.sum() > 0:
                ax.scatter(x0[mask], x1[mask], c=np.array(colors)[mask], s=5, alpha=0.6)
            
            ax.set_xlabel('x0', fontsize=9)
            ax.set_ylabel('x1', fontsize=9)
            if row_idx == 0:
                ax.set_title(f't = {t_slice}', fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'{layer_name}\nx1', fontsize=10)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('NCC Classification - Input Space (Time Slices)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path_slices = save_path.parent / (save_path.stem + '_slices.png')
    plt.savefig(save_path_slices, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  NCC classification time slices saved to {save_path_slices}")


def visualize_ncc_classification_heatmap(
    h_gt: np.ndarray,
    class_labels: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    bins: np.ndarray,
    save_path: Path,
    config: Dict
):
    """
    Visualize NCC classification accuracy as line plot (1D binning by h value).
    
    Args:
        h_gt: Ground truth values (N, 1)
        class_labels: True class labels (N,)
        predictions_dict: Dict mapping layer_name -> predictions (N,)
        bins: Bin edges for classification
        save_path: Path to save figure
        config: Configuration dictionary
    """
    n_bin_vis = max(20, config.get('n_bin_visualize_ncc', 100))
    min_samples = config.get('min_samples_threshold', 1)
    
    n_layers = len(predictions_dict)
    fig, axes = plt.subplots(1, n_layers, figsize=(6*n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    
    h_gt_np = h_gt if isinstance(h_gt, np.ndarray) else h_gt.detach().cpu().numpy()
    labels_np = class_labels if isinstance(class_labels, np.ndarray) else class_labels.detach().cpu().numpy()

    for ax, (layer_name, preds) in zip(axes, predictions_dict.items()):
        h_values = h_gt_np[:, 0]
        h_min, h_max = h_values.min(), h_values.max()
        bin_edges = np.linspace(h_min, h_max, n_bin_vis + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        preds_np = preds if isinstance(preds, np.ndarray) else preds.detach().cpu().numpy()
        
        accuracies = []
        for i in range(n_bin_vis):
            mask = (h_values >= bin_edges[i]) & (h_values < bin_edges[i+1])
            if mask.sum() >= min_samples:
                acc = (preds_np[mask] == labels_np[mask]).astype(float).mean()
                accuracies.append(acc)
            else:
                accuracies.append(np.nan)
        
        valid_mask = ~np.isnan(accuracies)
        ax.plot(bin_centers[valid_mask], np.array(accuracies)[valid_mask], 
               linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('h', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_title(f'{layer_name}\nOverall Acc: {(preds_np == labels_np).astype(float).mean():.3f}', 
                    fontsize=11)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  NCC classification heatmap saved to {save_path}")


def visualize_ncc_accuracy_evolution(
    h_gt: np.ndarray,
    class_labels: np.ndarray,
    predictions_history: Dict[str, Dict[str, np.ndarray]],
    save_path: Path,
    config: Dict
):
    """
    Visualize how NCC classification accuracy evolves across layers.
    
    Args:
        h_gt: Ground truth values (N, 1)
        class_labels: True class labels (N,)
        predictions_history: Dict mapping layer -> dict of predictions at different epochs
        save_path: Path to save figure
        config: Configuration dictionary
    """
    labels_np = class_labels if isinstance(class_labels, np.ndarray) else class_labels.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layer_names = list(predictions_history.keys())
    layer_accuracies = []
    
    for layer_name in layer_names:
        preds = predictions_history[layer_name]
        if isinstance(preds, dict):
            preds = list(preds.values())[-1]  # Get latest
        preds_np = preds if isinstance(preds, np.ndarray) else preds.detach().cpu().numpy()
        acc = (preds_np == labels_np).astype(float).mean()
        layer_accuracies.append(acc)
    
    ax.plot(range(len(layer_names)), layer_accuracies, marker='o', linewidth=2, markersize=8)
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('NCC Classification Accuracy Evolution', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  NCC accuracy evolution saved to {save_path}")


def visualize_ncc_classification_input_space_heatmap(
    x: np.ndarray,
    t: np.ndarray,
    class_labels: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    save_path: Path,
    config: Dict
):
    """
    Visualize NCC classification accuracy as continuous heatmap in input space for 2D spatial.
    
    Uses smooth interpolation for truly continuous visualization.
    
    Args:
        x: Spatial coordinates (N, 2)
        t: Temporal coordinates (N, 1)
        class_labels: True class labels (N,)
        predictions_dict: Dict mapping layer_name -> predictions (N,)
        save_path: Path to save figure
        config: Configuration dictionary
    """
    from scipy.interpolate import griddata
    from matplotlib.gridspec import GridSpec
    
    n_layers = len(predictions_dict)
    min_samples = config.get('min_samples_threshold', 1)
    
    x_np = x if isinstance(x, np.ndarray) else x.detach().cpu().numpy()
    t_np = t if isinstance(t, np.ndarray) else t.detach().cpu().numpy()
    labels_np = class_labels if isinstance(class_labels, np.ndarray) else class_labels.detach().cpu().numpy()
    
    x0 = x_np[:, 0] if x_np.ndim > 1 else x_np.flatten()
    x1 = x_np[:, 1] if x_np.ndim > 1 else np.zeros_like(x0)
    t_flat = t_np.flatten() if t_np.ndim > 1 else t_np
    
    # Grid for interpolation
    n_grid = 50
    x0_grid_lin = np.linspace(0, 1, n_grid)
    x1_grid_lin = np.linspace(0, 1, n_grid)
    X0_grid, X1_grid = np.meshgrid(x0_grid_lin, x1_grid_lin)
    
    fig = plt.figure(figsize=(22, 4*n_layers))
    gs = GridSpec(n_layers, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.08], wspace=0.25, hspace=0.3)
    
    cf_ref = None
    
    for row_idx, (layer_name, preds) in enumerate(predictions_dict.items()):
        preds_np = preds if isinstance(preds, np.ndarray) else preds.detach().cpu().numpy()
        correct = (preds_np == labels_np).astype(float)
        
        for col_idx, t_slice in enumerate(TIME_SLICES):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            # Select points near this time slice
            t_tol = 0.15
            mask = np.abs(t_flat - t_slice) < t_tol
            
            if mask.sum() < min_samples:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes)
                if row_idx == 0:
                    ax.set_title(f't = {t_slice}', fontsize=11, fontweight='bold')
                continue
            
            x0_slice = x0[mask]
            x1_slice = x1[mask]
            acc_slice = correct[mask]
            
            # Interpolate accuracy to grid for smooth visualization
            acc_grid = griddata((x0_slice, x1_slice), acc_slice, (X0_grid, X1_grid), method='linear')
            
            cf = ax.contourf(X0_grid, X1_grid, acc_grid, levels=50, cmap='RdYlGn', 
                            vmin=0, vmax=1)
            if cf_ref is None:
                cf_ref = cf
            
            if row_idx == 0:
                ax.set_title(f't = {t_slice}', fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'{layer_name}\nx1', fontsize=10)
            ax.set_xlabel('x0', fontsize=9)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_aspect('equal')
    
    # Add single colorbar on the right
    cbar_ax = fig.add_subplot(gs[:, 5])
    fig.colorbar(cf_ref, cax=cbar_ax, label='Accuracy')
    
    plt.suptitle('NCC Classification Accuracy - Input Space Heatmap', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  NCC input space heatmap saved to {save_path}")


def visualize_ncc_classification_accuracy_changes(
    x: np.ndarray,
    t: np.ndarray,
    class_labels: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    save_path: Path,
    config: Dict
):
    """
    Visualize NCC classification accuracy changes between layers.
    
    Args:
        x: Spatial coordinates (N, 2)
        t: Temporal coordinates (N, 1)
        class_labels: True class labels (N,)
        predictions_dict: Dict mapping layer_name -> predictions (N,)
        save_path: Path to save figure
        config: Configuration dictionary
    """
    labels_np = class_labels if isinstance(class_labels, np.ndarray) else class_labels.detach().cpu().numpy()
    
    layer_names = list(predictions_dict.keys())
    if len(layer_names) < 2:
        print("  Skipping accuracy changes plot (need at least 2 layers)")
        return
    
    n_transitions = len(layer_names) - 1
    
    fig, axes = plt.subplots(1, n_transitions, figsize=(5*n_transitions, 4))
    if n_transitions == 1:
        axes = [axes]
    
    for idx in range(n_transitions):
        layer_prev = layer_names[idx]
        layer_curr = layer_names[idx + 1]
        
        preds_prev = predictions_dict[layer_prev]
        preds_curr = predictions_dict[layer_curr]
        
        preds_prev_np = preds_prev if isinstance(preds_prev, np.ndarray) else preds_prev.detach().cpu().numpy()
        preds_curr_np = preds_curr if isinstance(preds_curr, np.ndarray) else preds_curr.detach().cpu().numpy()
        
        correct_prev = (preds_prev_np == labels_np).astype(float)
        correct_curr = (preds_curr_np == labels_np).astype(float)
        
        # Change: positive = improved, negative = degraded
        change = correct_curr - correct_prev
        
        ax = axes[idx]
        
        # Plot histogram of changes
        ax.hist(change, bins=3, range=(-1.5, 1.5), alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Accuracy Change', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{layer_prev} -> {layer_curr}', fontsize=11, fontweight='bold')
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(['Degraded', 'No Change', 'Improved'])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('NCC Accuracy Changes Between Layers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  NCC accuracy changes saved to {save_path}")


def visualize_ncc_classification_input_space_accuracy_changes(
    x: np.ndarray,
    t: np.ndarray,
    class_labels: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    save_path: Path,
    config: Dict
):
    """
    Visualize NCC classification accuracy changes in input space for 2D spatial.
    
    Args:
        x: Spatial coordinates (N, 2)
        t: Temporal coordinates (N, 1)
        class_labels: True class labels (N,)
        predictions_dict: Dict mapping layer_name -> predictions (N,)
        save_path: Path to save figure
        config: Configuration dictionary
    """
    labels_np = class_labels if isinstance(class_labels, np.ndarray) else class_labels.detach().cpu().numpy()
    
    x_np = x if isinstance(x, np.ndarray) else x.detach().cpu().numpy()
    t_np = t if isinstance(t, np.ndarray) else t.detach().cpu().numpy()
    
    x0 = x_np[:, 0] if x_np.ndim > 1 else x_np.flatten()
    x1 = x_np[:, 1] if x_np.ndim > 1 else np.zeros_like(x0)
    t_flat = t_np.flatten() if t_np.ndim > 1 else t_np
    
    layer_names = list(predictions_dict.keys())
    if len(layer_names) < 2:
        print("  Skipping input space accuracy changes plot (need at least 2 layers)")
        return
    
    n_transitions = len(layer_names) - 1
    n_bin_vis = config.get('n_bin_visualize_ncc', 20)
    min_samples = config.get('min_samples_threshold', 1)
    
    # Create time slice plots for each transition with continuous heatmaps
    from matplotlib.gridspec import GridSpec
    from scipy.interpolate import griddata
    
    fig = plt.figure(figsize=(22, 4*n_transitions))
    gs = GridSpec(n_transitions, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.08], wspace=0.25, hspace=0.3)
    
    cf_ref = None
    
    # Grid for interpolation
    n_grid = 50
    x0_grid_lin = np.linspace(0, 1, n_grid)
    x1_grid_lin = np.linspace(0, 1, n_grid)
    X0_grid, X1_grid = np.meshgrid(x0_grid_lin, x1_grid_lin)
    
    for row_idx in range(n_transitions):
        layer_prev = layer_names[row_idx]
        layer_curr = layer_names[row_idx + 1]
        
        preds_prev = predictions_dict[layer_prev]
        preds_curr = predictions_dict[layer_curr]
        
        preds_prev_np = preds_prev if isinstance(preds_prev, np.ndarray) else preds_prev.detach().cpu().numpy()
        preds_curr_np = preds_curr if isinstance(preds_curr, np.ndarray) else preds_curr.detach().cpu().numpy()
        
        correct_prev = (preds_prev_np == labels_np).astype(float)
        correct_curr = (preds_curr_np == labels_np).astype(float)
        change = correct_curr - correct_prev
        
        for col_idx, t_slice in enumerate(TIME_SLICES):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            t_tol = 0.15
            mask = np.abs(t_flat - t_slice) < t_tol
            
            if mask.sum() < 5:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes)
                continue
            
            x0_slice = x0[mask]
            x1_slice = x1[mask]
            change_slice = change[mask]
            
            # Interpolate change values onto grid for continuous heatmap
            change_grid = griddata((x0_slice, x1_slice), change_slice, 
                                   (X0_grid, X1_grid), method='linear')
            
            # Use contourf for continuous visualization
            levels = np.linspace(-1, 1, 21)
            cf = ax.contourf(X0_grid, X1_grid, change_grid, levels=levels,
                            cmap='RdYlGn', extend='both')
            
            if cf_ref is None:
                cf_ref = cf
            
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_aspect('equal')
            
            if row_idx == 0:
                ax.set_title(f't = {t_slice}', fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'{layer_prev}->{layer_curr}\nx1', fontsize=9)
            ax.set_xlabel('x0', fontsize=9)
    
    # Add colorbar
    cbar_ax = fig.add_subplot(gs[:, 5])
    cbar = fig.colorbar(cf_ref, cax=cbar_ax, label='Accuracy Change')
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Degraded', 'No Change', 'Improved'])
    
    plt.suptitle('NCC Accuracy Changes - Input Space', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  NCC input space accuracy changes saved to {save_path}")

