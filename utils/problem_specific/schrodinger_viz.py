"""
Schrödinger equation specific visualizations.

Provides custom visualization for:
1. Dataset visualization: heatmaps of |h| and arg(h)
2. Evaluation visualization: comprehensive model performance analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict


def visualize_dataset(data_dict: Dict, save_dir: Path, config: Dict, split_name: str):
    """
    Visualize Schrödinger equation dataset with heatmaps.
    
    Creates two heatmaps:
    - |h| (magnitude) vs (x, t)
    - arg(h) (phase) vs (x, t)
    
    Args:
        data_dict: Dataset dictionary with 'x', 't', 'h_gt' tensors
        save_dir: Directory to save visualization
        config: Configuration dictionary
        split_name: Name of split ('training' or 'evaluation')
    """
    # Extract data
    x = data_dict['x'].cpu().numpy()  # (N, spatial_dim)
    t = data_dict['t'].cpu().numpy()  # (N, 1)
    h_gt = data_dict['h_gt'].cpu().numpy()  # (N, 2) where h = u + iv, [:, 0]=real, [:, 1]=imag
    
    # Flatten coordinates
    x_flat = x[:, 0]
    t_flat = t[:, 0]
    
    # Compute magnitude and phase (h = u + iv)
    u = h_gt[:, 0]
    v = h_gt[:, 1]
    magnitude = np.sqrt(u**2 + v**2)
    phase = np.arctan2(v, u)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Magnitude |h|
    scatter1 = axes[0].scatter(x_flat, t_flat, c=magnitude, cmap='viridis', s=5, alpha=0.6)
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('t', fontsize=12)
    axes[0].set_title(f'|h| - {split_name.capitalize()}', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('|h|', fontsize=11)
    
    # Plot 2: Phase arg(h)
    scatter2 = axes[1].scatter(x_flat, t_flat, c=phase, cmap='twilight', s=5, alpha=0.6, 
                              vmin=-np.pi, vmax=np.pi)
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('t', fontsize=12)
    axes[1].set_title(f'arg(h) - {split_name.capitalize()}', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('arg(h) [rad]', fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    save_path = Path(save_dir) / f"{split_name}_schrodinger_viz.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Schrodinger visualization saved to {save_path}")


def visualize_evaluation(model, eval_data_path: str, save_dir: Path, config: Dict):
    """
    Evaluation visualization for Schrödinger equation.
    Produces predictions_and_error_maps.png via the generic visualizer.

    Generates:
    1. Six heatmaps (2 rows × 3 columns):
       - Row 1: |h| - Ground truth, Prediction, Error
       - Row 2: arg(h) - Ground truth, Prediction, Error

    2. Six heatmaps for u and v components (2 rows × 3 columns):
       - Row 1: u (real) - Ground truth, Prediction, Error
       - Row 2: v (imaginary) - Ground truth, Prediction, Error
       
    3. Six fixed-time plots (2 rows × 3 columns):
       - Row 1: |h| at t=0, π/4, π/2
       - Row 2: arg(h) at t=0, π/4, π/2
       
    Args:
        model: Trained model
        eval_data_path: Path to evaluation dataset
        save_dir: Directory to save visualizations
        config: Configuration dictionary
    """
    from utils.problem_specific.generic_viz import plot_predictions_and_error_maps
    plot_predictions_and_error_maps(model, save_dir, config)

    from utils.dataset_gen import load_dataset

    device = torch.device('cuda' if config['cuda'] and torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load evaluation data
    eval_data = load_dataset(eval_data_path, device)
    
    # Get problem-specific config
    problem = config.get('problem', 'schrodinger')
    problem_config = config[problem]
    spatial_domain = problem_config['spatial_domain'][0]  # [x_min, x_max]
    temporal_domain = problem_config['temporal_domain']  # [t_min, t_max]
    
    x_min, x_max = spatial_domain
    t_min, t_max = temporal_domain
    
    # Create dense evaluation grid
    n_x = 256
    n_t = 200
    x_grid = np.linspace(x_min, x_max, n_x)
    t_grid = np.linspace(t_min, t_max, n_t)
    X, T = np.meshgrid(x_grid, t_grid)
    
    # Flatten for model input (use model's dtype for precision compatibility)
    dtype = next(model.parameters()).dtype
    x_flat = torch.tensor(X.flatten(), dtype=dtype, device=device).view(-1, 1)
    t_flat = torch.tensor(T.flatten(), dtype=dtype, device=device).view(-1, 1)
    
    # Model predictions
    with torch.no_grad():
        xt_input = torch.cat([x_flat, t_flat], dim=1)
        uv_pred = model(xt_input)
        u_pred = uv_pred[:, 0].cpu().numpy().reshape(n_t, n_x)
        v_pred = uv_pred[:, 1].cpu().numpy().reshape(n_t, n_x)
    
    # Get ground truth solver
    from solvers.schrodinger_solver import _get_interpolator
    interpolator = _get_interpolator(config)
    
    # Interpolate ground truth to grid
    h_true_flat = interpolator(x_flat.cpu().numpy().flatten(), t_flat.cpu().numpy().flatten())
    u_true = h_true_flat.real.reshape(n_t, n_x)
    v_true = h_true_flat.imag.reshape(n_t, n_x)
    
    # Compute magnitudes and phases
    mag_true = np.sqrt(u_true**2 + v_true**2)
    mag_pred = np.sqrt(u_pred**2 + v_pred**2)
    mag_error = np.abs(mag_pred - mag_true)
    
    phase_true = np.arctan2(v_true, u_true)
    phase_pred = np.arctan2(v_pred, u_pred)
    # Wrap phase error to [-π, π]
    phase_error = np.angle(np.exp(1j * (phase_pred - phase_true)))
    
    # ==================================================================
    # FIGURE 1: HEATMAPS (2 rows × 3 columns)
    # ==================================================================
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 8))
    
    # Row 1: Magnitude |h|
    im10 = axes1[0, 0].contourf(X, T, mag_true, levels=50, cmap='viridis')
    axes1[0, 0].set_title('|h| - Ground Truth', fontsize=12, fontweight='bold')
    axes1[0, 0].set_xlabel('x')
    axes1[0, 0].set_ylabel('t')
    plt.colorbar(im10, ax=axes1[0, 0])
    
    im11 = axes1[0, 1].contourf(X, T, mag_pred, levels=50, cmap='viridis')
    axes1[0, 1].set_title('|h| - Prediction', fontsize=12, fontweight='bold')
    axes1[0, 1].set_xlabel('x')
    axes1[0, 1].set_ylabel('t')
    plt.colorbar(im11, ax=axes1[0, 1])
    
    im12 = axes1[0, 2].contourf(X, T, mag_error, levels=50, cmap='Reds')
    axes1[0, 2].set_title('|h| - Absolute Error', fontsize=12, fontweight='bold')
    axes1[0, 2].set_xlabel('x')
    axes1[0, 2].set_ylabel('t')
    plt.colorbar(im12, ax=axes1[0, 2])
    
    # Row 2: Phase arg(h)
    im20 = axes1[1, 0].contourf(X, T, phase_true, levels=50, cmap='twilight', 
                                vmin=-np.pi, vmax=np.pi)
    axes1[1, 0].set_title('arg(h) - Ground Truth', fontsize=12, fontweight='bold')
    axes1[1, 0].set_xlabel('x')
    axes1[1, 0].set_ylabel('t')
    cbar20 = plt.colorbar(im20, ax=axes1[1, 0])
    cbar20.set_label('[rad]')
    
    im21 = axes1[1, 1].contourf(X, T, phase_pred, levels=50, cmap='twilight',
                                vmin=-np.pi, vmax=np.pi)
    axes1[1, 1].set_title('arg(h) - Prediction', fontsize=12, fontweight='bold')
    axes1[1, 1].set_xlabel('x')
    axes1[1, 1].set_ylabel('t')
    cbar21 = plt.colorbar(im21, ax=axes1[1, 1])
    cbar21.set_label('[rad]')
    
    im22 = axes1[1, 2].contourf(X, T, phase_error, levels=50, cmap='RdBu_r',
                                vmin=-np.pi, vmax=np.pi)
    axes1[1, 2].set_title('arg(h) - Phase Error', fontsize=12, fontweight='bold')
    axes1[1, 2].set_xlabel('x')
    axes1[1, 2].set_ylabel('t')
    cbar22 = plt.colorbar(im22, ax=axes1[1, 2])
    cbar22.set_label('[rad]')
    
    plt.tight_layout()
    save_path1 = Path(save_dir) / "schrodinger_heatmaps.png"
    plt.savefig(save_path1, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Heatmaps saved to {save_path1}")
    
    # ==================================================================
    # FIGURE 2: U AND V COMPONENT HEATMAPS (2 rows × 3 columns)
    # ==================================================================
    # Compute errors for u and v
    u_error = np.abs(u_pred - u_true)
    v_error = np.abs(v_pred - v_true)
    
    fig_uv, axes_uv = plt.subplots(2, 3, figsize=(15, 8))
    
    # Row 1: u (Real component)
    im_u0 = axes_uv[0, 0].contourf(X, T, u_true, levels=50, cmap='RdBu_r')
    axes_uv[0, 0].set_title('u (Real) - Ground Truth', fontsize=12,
                            fontweight='bold')
    axes_uv[0, 0].set_xlabel('x')
    axes_uv[0, 0].set_ylabel('t')
    plt.colorbar(im_u0, ax=axes_uv[0, 0])
    
    im_u1 = axes_uv[0, 1].contourf(X, T, u_pred, levels=50, cmap='RdBu_r')
    axes_uv[0, 1].set_title('u (Real) - Prediction', fontsize=12,
                            fontweight='bold')
    axes_uv[0, 1].set_xlabel('x')
    axes_uv[0, 1].set_ylabel('t')
    plt.colorbar(im_u1, ax=axes_uv[0, 1])
    
    im_u2 = axes_uv[0, 2].contourf(X, T, u_error, levels=50, cmap='Reds')
    axes_uv[0, 2].set_title('u (Real) - Absolute Error', fontsize=12,
                            fontweight='bold')
    axes_uv[0, 2].set_xlabel('x')
    axes_uv[0, 2].set_ylabel('t')
    plt.colorbar(im_u2, ax=axes_uv[0, 2])
    
    # Row 2: v (Imaginary component)
    im_v0 = axes_uv[1, 0].contourf(X, T, v_true, levels=50, cmap='RdBu_r')
    axes_uv[1, 0].set_title('v (Imaginary) - Ground Truth', fontsize=12,
                            fontweight='bold')
    axes_uv[1, 0].set_xlabel('x')
    axes_uv[1, 0].set_ylabel('t')
    plt.colorbar(im_v0, ax=axes_uv[1, 0])
    
    im_v1 = axes_uv[1, 1].contourf(X, T, v_pred, levels=50, cmap='RdBu_r')
    axes_uv[1, 1].set_title('v (Imaginary) - Prediction', fontsize=12,
                            fontweight='bold')
    axes_uv[1, 1].set_xlabel('x')
    axes_uv[1, 1].set_ylabel('t')
    plt.colorbar(im_v1, ax=axes_uv[1, 1])
    
    im_v2 = axes_uv[1, 2].contourf(X, T, v_error, levels=50, cmap='Reds')
    axes_uv[1, 2].set_title('v (Imaginary) - Absolute Error', fontsize=12,
                            fontweight='bold')
    axes_uv[1, 2].set_xlabel('x')
    axes_uv[1, 2].set_ylabel('t')
    plt.colorbar(im_v2, ax=axes_uv[1, 2])
    
    plt.tight_layout()
    save_path_uv = Path(save_dir) / "schrodinger_uv_components.png"
    plt.savefig(save_path_uv, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  u/v component heatmaps saved to {save_path_uv}")
    
    # ==================================================================
    # FIGURE 3: FIXED-TIME PLOTS (2 rows × 3 columns)
    # ==================================================================
    # Fixed times: t=0, π/4, π/2
    t_snapshots = [0.0, np.pi/4, np.pi/2]
    t_labels = ['t=0', 't=π/4', 't=π/2']
    
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    
    for col_idx, (t_val, t_label) in enumerate(zip(t_snapshots, t_labels)):
        # Find closest time index in grid
        t_idx = np.argmin(np.abs(t_grid - t_val))
        
        # Row 1: |h| at fixed time
        axes2[0, col_idx].plot(x_grid, mag_true[t_idx, :], 'b-', linewidth=2, label='Ground Truth')
        axes2[0, col_idx].plot(x_grid, mag_pred[t_idx, :], 'r--', linewidth=2, label='Prediction')
        axes2[0, col_idx].set_xlabel('x', fontsize=11)
        axes2[0, col_idx].set_ylabel('|h|', fontsize=11)
        axes2[0, col_idx].set_title(f'|h| at {t_label}', fontsize=12, fontweight='bold')
        axes2[0, col_idx].legend(fontsize=9)
        axes2[0, col_idx].grid(True, alpha=0.3)
        
        # Row 2: arg(h) at fixed time
        axes2[1, col_idx].plot(x_grid, phase_true[t_idx, :], 'b-', linewidth=2, label='Ground Truth')
        axes2[1, col_idx].plot(x_grid, phase_pred[t_idx, :], 'r--', linewidth=2, label='Prediction')
        axes2[1, col_idx].set_xlabel('x', fontsize=11)
        axes2[1, col_idx].set_ylabel('arg(h) [rad]', fontsize=11)
        axes2[1, col_idx].set_title(f'arg(h) at {t_label}', fontsize=12, fontweight='bold')
        axes2[1, col_idx].legend(fontsize=9)
        axes2[1, col_idx].grid(True, alpha=0.3)
        axes2[1, col_idx].set_ylim([-np.pi, np.pi])
    
    plt.tight_layout()
    save_path2 = Path(save_dir) / "schrodinger_fixed_time.png"
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Fixed-time plots saved to {save_path2}")


def visualize_ncc_dataset(ncc_data: Dict, dataset_dir: Path, config: Dict, prefix: str = 'ncc'):
    """
    Visualize NCC dataset distribution in (u, v) output space with bin grid.
    
    Creates scatter plot showing:
    - Sample distribution on (u, v) grid
    - Bin boundaries as grid lines
    - Color-coded by class ID
    
    Args:
        ncc_data: NCC dataset dictionary with 'h_gt' tensor (N, 2) where h = u + iv
        dataset_dir: Directory to save visualization
        config: Configuration dictionary with 'bins'
        prefix: Prefix for filename
    """
    # Extract u, v values (h = u + iv)
    h_gt = ncc_data['h_gt'].cpu().numpy()  # (N, 2)
    u = h_gt[:, 0]
    v = h_gt[:, 1]
    
    bins = config['bins']
    N = len(u)
    
    # Compute bin edges
    u_min, u_max = u.min(), u.max()
    v_min, v_max = v.min(), v.max()
    u_edges = np.linspace(u_min, u_max, bins + 1)
    v_edges = np.linspace(v_min, v_max, bins + 1)
    
    # Assign samples to classes for coloring
    u_bin = np.clip(np.digitize(u, u_edges) - 1, 0, bins - 1)
    v_bin = np.clip(np.digitize(v, v_edges) - 1, 0, bins - 1)
    class_ids = u_bin * bins + v_bin  # Flatten to class ID
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Scatter plot colored by class ID
    scatter = ax.scatter(u, v, c=class_ids, cmap='tab20', s=20, alpha=0.6, 
                        edgecolors='none')
    
    # Draw bin grid lines
    for u_edge in u_edges:
        ax.axvline(u_edge, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    for v_edge in v_edges:
        ax.axhline(v_edge, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    ax.set_xlabel('u (Real part)', fontsize=13)
    ax.set_ylabel('v (Imaginary part)', fontsize=13)
    ax.set_title(f'NCC Dataset Distribution ({bins}×{bins} bins, {N} samples)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Class ID', fontsize=11)
    
    plt.tight_layout()
    
    # Save
    save_path = Path(dataset_dir) / f"{prefix}_ncc_distribution_with_bins.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  NCC distribution visualization saved to {save_path}")


def visualize_ncc_classification(
    h_gt: torch.Tensor,
    class_labels: torch.Tensor,
    predictions_dict: Dict[str, torch.Tensor],
    bins: int,
    save_path: Path
):
    """
    Visualize NCC classification correctness per layer in (u, v) output space.
    
    Creates multi-panel figure with one subplot per layer showing:
    - Green dots: correctly classified samples
    - Red dots: misclassified samples
    - Bin grid overlay
    
    Args:
        h_gt: Ground truth outputs (N, 2) with h = u + iv as (real, imag)
        class_labels: True class labels (N,)
        predictions_dict: Dict mapping layer_name -> predictions (N,)
        bins: Number of bins per dimension
        save_path: Path to save figure
    """
    # Extract u, v values (h = u + iv)
    h_gt_np = h_gt.cpu().numpy()
    u = h_gt_np[:, 0]
    v = h_gt_np[:, 1]
    
    class_labels_np = class_labels.cpu().numpy()
    
    # Compute bin edges
    u_min, u_max = u.min(), u.max()
    v_min, v_max = v.min(), v.max()
    u_edges = np.linspace(u_min, u_max, bins + 1)
    v_edges = np.linspace(v_min, v_max, bins + 1)
    
    # Setup subplots (3 columns)
    layer_names = list(predictions_dict.keys())
    n_layers = len(layer_names)
    n_cols = 3
    n_rows = int(np.ceil(n_layers / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    if n_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, layer_name in enumerate(layer_names):
        ax = axes[idx]
        predictions = predictions_dict[layer_name].cpu().numpy()
        
        # Determine correctness
        correct = (predictions == class_labels_np)
        accuracy = correct.mean()
        
        # Plot correct (green) and incorrect (red)
        ax.scatter(u[correct], v[correct], c='green', s=15, alpha=0.6, 
                  label='Correct', edgecolors='none')
        ax.scatter(u[~correct], v[~correct], c='red', s=15, alpha=0.6, 
                  label='Incorrect', edgecolors='none')
        
        # Draw bin grid
        for u_edge in u_edges:
            ax.axvline(u_edge, color='gray', linestyle='--', linewidth=0.6, alpha=0.4)
        for v_edge in v_edges:
            ax.axhline(v_edge, color='gray', linestyle='--', linewidth=0.6, alpha=0.4)
        
        ax.set_xlabel('u (Real part)', fontsize=11)
        ax.set_ylabel('v (Imaginary part)', fontsize=11)
        ax.set_title(f'{layer_name} (Acc: {accuracy:.2%})', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.2)
    
    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  NCC classification diagnostic saved to {save_path}")


def visualize_ncc_classification_input_space(
    x: torch.Tensor,
    t: torch.Tensor,
    class_labels: torch.Tensor,
    predictions_dict: Dict[str, torch.Tensor],
    save_path: Path
):
    """
    Visualize NCC classification correctness per layer in (x, t) input space.
    
    Creates multi-panel figure with one subplot per layer showing:
    - Green dots: correctly classified samples
    - Red dots: misclassified samples
    
    Args:
        x: Spatial coordinates (N, 1)
        t: Temporal coordinates (N, 1)
        class_labels: True class labels (N,)
        predictions_dict: Dict mapping layer_name -> predictions (N,)
        save_path: Path to save figure
    """
    # Extract x, t values
    x_np = x.cpu().numpy().flatten()
    t_np = t.cpu().numpy().flatten()
    class_labels_np = class_labels.cpu().numpy()
    
    # Setup subplots (3 columns)
    layer_names = list(predictions_dict.keys())
    n_layers = len(layer_names)
    n_cols = 3
    n_rows = int(np.ceil(n_layers / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    if n_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, layer_name in enumerate(layer_names):
        ax = axes[idx]
        predictions = predictions_dict[layer_name].cpu().numpy()
        
        # Determine correctness
        correct = (predictions == class_labels_np)
        accuracy = correct.mean()
        
        # Plot correct (green) and incorrect (red)
        ax.scatter(x_np[correct], t_np[correct], c='green', s=15, alpha=0.6,
                  label='Correct', edgecolors='none')
        ax.scatter(x_np[~correct], t_np[~correct], c='red', s=15, alpha=0.6,
                  label='Incorrect', edgecolors='none')
        
        ax.set_xlabel('x (spatial)', fontsize=11)
        ax.set_ylabel('t (time)', fontsize=11)
        ax.set_title(f'{layer_name} (Acc: {accuracy:.2%})',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.2)
    
    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Input space classification diagnostic saved to {save_path}")


def visualize_ncc_classification_heatmap(
    h_gt: torch.Tensor,
    class_labels: torch.Tensor,
    predictions_dict: Dict[str, torch.Tensor],
    bins: int,
    save_path: Path,
    config: Dict
):
    """
    Visualize NCC classification accuracy as heatmaps in (u, v) output space.
    
    Creates multi-panel figure with one subplot per layer showing a heatmap
    where each grid cell's color represents the classification accuracy (0-1)
    in that region.
    
    Args:
        h_gt: Ground truth outputs (N, 2) with h = u + iv as (real, imag)
        class_labels: True class labels (N,)
        predictions_dict: Dict mapping layer_name -> predictions (N,)
        bins: Number of bins per dimension (for NCC classes)
        save_path: Path to save figure
        config: Configuration dictionary with 'n_bin_visualize_ncc' and 'n_samples_ncc'
    """
    # Get visualization parameters
    n_bin_viz = max(20, config.get('n_bin_visualize_ncc', 100))
    min_samples_threshold = 1  # Minimum 1 sample per bin
    
    # Extract u, v values (h = u + iv)
    h_gt_np = h_gt.cpu().numpy()
    u = h_gt_np[:, 0]
    v = h_gt_np[:, 1]
    
    class_labels_np = class_labels.cpu().numpy()
    
    # Compute visualization grid edges
    u_min, u_max = u.min(), u.max()
    v_min, v_max = v.min(), v.max()
    u_edges = np.linspace(u_min, u_max, n_bin_viz + 1)
    v_edges = np.linspace(v_min, v_max, n_bin_viz + 1)
    
    # Setup subplots (3 columns)
    layer_names = list(predictions_dict.keys())
    n_layers = len(layer_names)
    n_cols = 3
    n_rows = int(np.ceil(n_layers / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    if n_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Store accuracy grids for computing changes
    accuracy_grids = {}
    
    # Create dense grid for interpolation
    from scipy.interpolate import griddata
    u_grid = np.linspace(u_min, u_max, n_bin_viz)
    v_grid = np.linspace(v_min, v_max, n_bin_viz)
    U_grid, V_grid = np.meshgrid(u_grid, v_grid)
    
    for idx, layer_name in enumerate(layer_names):
        ax = axes[idx]
        predictions = predictions_dict[layer_name].cpu().numpy()
        
        # Compute per-point correctness
        correct = (predictions == class_labels_np).astype(float)
        
        # Interpolate to dense grid for smooth heatmap
        points = np.column_stack([u, v])
        
        # Try cubic first, fall back to linear if needed
        try:
            accuracy_grid = griddata(
                points, correct, (U_grid, V_grid),
                method='cubic', fill_value=0.5
            )
            # Check if cubic produced valid results
            if accuracy_grid.shape[0] < 2 or accuracy_grid.shape[1] < 2 or np.all(np.isnan(accuracy_grid)):
                raise ValueError("Cubic interpolation insufficient")
        except:
            # Fall back to linear interpolation (more robust)
            accuracy_grid = griddata(
                points, correct, (U_grid, V_grid),
                method='linear', fill_value=0.5
            )
        
        accuracy_grid = np.clip(accuracy_grid, 0, 1)
        
        # Store for computing changes
        accuracy_grids[layer_name] = accuracy_grid
        
        # Check if we have valid data to plot
        if accuracy_grid.shape[0] >= 2 and accuracy_grid.shape[1] >= 2 and not np.all(np.isnan(accuracy_grid)):
            im = ax.contourf(
                U_grid, V_grid, accuracy_grid,
                levels=20,
                cmap='RdYlGn',
                vmin=0.0,
                vmax=1.0
            )
            plt.colorbar(im, ax=ax, label='Local Accuracy')
        else:
            # Only if interpolation completely fails
            ax.text(0.5, 0.5, 'Insufficient data\nfor heatmap',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray')
        
        ax.set_xlabel('u (Real part)', fontsize=11)
        ax.set_ylabel('v (Imaginary part)', fontsize=11)
        overall_acc = correct.mean()
        ax.set_title(f'{layer_name} (Acc: {overall_acc:.2%})', 
                    fontsize=12, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  NCC classification heatmap saved to {save_path}")
    
    # Generate accuracy changes plot (N-1 transitions)
    if n_layers > 1:
        n_changes = n_layers - 1
        n_cols_changes = 3
        n_rows_changes = int(np.ceil(n_changes / n_cols_changes))
        
        fig_changes, axes_changes = plt.subplots(
            n_rows_changes, n_cols_changes, 
            figsize=(15, 5 * n_rows_changes)
        )
        
        if n_changes == 1:
            axes_changes = np.array([axes_changes])
        axes_changes = axes_changes.flatten()
        
        for idx in range(n_changes):
            ax = axes_changes[idx]
            layer_prev = layer_names[idx]
            layer_curr = layer_names[idx + 1]
            
            # Compute difference: current - previous
            diff_grid = accuracy_grids[layer_curr] - accuracy_grids[layer_prev]
            
            # Check if grid is valid for contourf
            if diff_grid.shape[0] >= 2 and diff_grid.shape[1] >= 2 and not np.all(np.isnan(diff_grid)):
                # Plot smooth difference heatmap
                from matplotlib.colors import TwoSlopeNorm
                norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
                contour = ax.contourf(
                    U_grid, V_grid, diff_grid,
                    levels=20,
                    cmap='RdYlGn',
                    norm=norm
                )
                
                ax.set_xlabel('u (Real part)', fontsize=11)
                ax.set_ylabel('v (Imaginary part)', fontsize=11)
                ax.set_title(f'{layer_prev} → {layer_curr}', 
                            fontsize=12, fontweight='bold')
                plt.colorbar(contour, ax=ax, label='Accuracy Change')
            else:
                # Skip this subplot if data is insufficient
                ax.text(0.5, 0.5, 'Insufficient data\nfor change visualization',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, color='gray')
                ax.set_xlabel('u (Real part)', fontsize=11)
                ax.set_ylabel('v (Imaginary part)', fontsize=11)
                ax.set_title(f'{layer_prev} → {layer_curr}', 
                            fontsize=12, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(n_changes, len(axes_changes)):
            axes_changes[idx].axis('off')
        
        plt.tight_layout()
        changes_path = save_path.parent / "ncc_classification_accuracy_changes.png"
        plt.savefig(changes_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  NCC accuracy changes heatmap saved to {changes_path}")


def visualize_ncc_classification_input_space_heatmap(
    x: torch.Tensor,
    t: torch.Tensor,
    class_labels: torch.Tensor,
    predictions_dict: Dict[str, torch.Tensor],
    save_path: Path,
    config: Dict
):
    """
    Visualize NCC classification accuracy as heatmaps in (x, t) input space.
    
    Creates multi-panel figure with one subplot per layer showing a heatmap
    where each grid cell's color represents the classification accuracy (0-1)
    in that region.
    
    Args:
        x: Spatial coordinates (N, 1)
        t: Temporal coordinates (N, 1)
        class_labels: True class labels (N,)
        predictions_dict: Dict mapping layer_name -> predictions (N,)
        save_path: Path to save figure
        config: Configuration dictionary with 'n_bin_visualize_ncc' and 'n_samples_ncc'
    """
    # Get visualization parameters
    n_bin_viz = max(20, config.get('n_bin_visualize_ncc', 100))
    min_samples_threshold = 1  # Minimum 1 sample per bin
    
    # Extract x, t values
    x_np = x.cpu().numpy().flatten()
    t_np = t.cpu().numpy().flatten()
    class_labels_np = class_labels.cpu().numpy()
    
    # Compute visualization grid edges
    x_min, x_max = x_np.min(), x_np.max()
    t_min, t_max = t_np.min(), t_np.max()
    x_edges = np.linspace(x_min, x_max, n_bin_viz + 1)
    t_edges = np.linspace(t_min, t_max, n_bin_viz + 1)
    
    # Setup subplots (3 columns)
    layer_names = list(predictions_dict.keys())
    n_layers = len(layer_names)
    n_cols = 3
    n_rows = int(np.ceil(n_layers / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    if n_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Store accuracy grids for computing changes
    accuracy_grids = {}
    
    # Create dense grid for interpolation
    from scipy.interpolate import griddata
    x_grid = np.linspace(x_min, x_max, n_bin_viz)
    t_grid = np.linspace(t_min, t_max, n_bin_viz)
    X_grid, T_grid = np.meshgrid(x_grid, t_grid)
    
    for idx, layer_name in enumerate(layer_names):
        ax = axes[idx]
        predictions = predictions_dict[layer_name].cpu().numpy()
        
        # Compute per-point correctness
        correct = (predictions == class_labels_np).astype(float)
        
        # Interpolate to dense grid for smooth heatmap
        points = np.column_stack([x_np, t_np])
        
        # Try cubic first, fall back to linear if needed
        try:
            accuracy_grid = griddata(
                points, correct, (X_grid, T_grid),
                method='cubic', fill_value=0.5
            )
            if accuracy_grid.shape[0] < 2 or accuracy_grid.shape[1] < 2 or np.all(np.isnan(accuracy_grid)):
                raise ValueError("Cubic interpolation insufficient")
        except:
            # Fall back to linear interpolation (more robust)
            accuracy_grid = griddata(
                points, correct, (X_grid, T_grid),
                method='linear', fill_value=0.5
            )
        
        accuracy_grid = np.clip(accuracy_grid, 0, 1)
        
        # Store for computing changes
        accuracy_grids[layer_name] = accuracy_grid
        
        # Check if we have valid data to plot
        if accuracy_grid.shape[0] >= 2 and accuracy_grid.shape[1] >= 2 and not np.all(np.isnan(accuracy_grid)):
            im = ax.contourf(
                X_grid, T_grid, accuracy_grid,
                levels=20,
                cmap='RdYlGn',
                vmin=0.0,
                vmax=1.0
            )
            plt.colorbar(im, ax=ax, label='Local Accuracy')
        else:
            # Only if interpolation completely fails
            ax.text(0.5, 0.5, 'Insufficient data\nfor heatmap',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray')
        
        ax.set_xlabel('x (spatial)', fontsize=11)
        ax.set_ylabel('t (time)', fontsize=11)
        overall_acc = correct.mean()
        ax.set_title(f'{layer_name} (Acc: {overall_acc:.2%})',
                    fontsize=12, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Input space classification heatmap saved to {save_path}")
    
    # Generate accuracy changes plot (N-1 transitions)
    if n_layers > 1:
        n_changes = n_layers - 1
        n_cols_changes = 3
        n_rows_changes = int(np.ceil(n_changes / n_cols_changes))
        
        fig_changes, axes_changes = plt.subplots(
            n_rows_changes, n_cols_changes, 
            figsize=(15, 5 * n_rows_changes)
        )
        
        if n_changes == 1:
            axes_changes = np.array([axes_changes])
        axes_changes = axes_changes.flatten()
        
        for idx in range(n_changes):
            ax = axes_changes[idx]
            layer_prev = layer_names[idx]
            layer_curr = layer_names[idx + 1]
            
            # Compute difference: current - previous
            diff_grid = accuracy_grids[layer_curr] - accuracy_grids[layer_prev]
            
            # Check if grid is valid for contourf
            if diff_grid.shape[0] >= 2 and diff_grid.shape[1] >= 2 and not np.all(np.isnan(diff_grid)):
                # Plot smooth difference heatmap
                from matplotlib.colors import TwoSlopeNorm
                norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
                contour = ax.contourf(
                    X_grid, T_grid, diff_grid,
                    levels=20,
                    cmap='RdYlGn',
                    norm=norm
                )
                plt.colorbar(contour, ax=ax, label='Accuracy Change')
            else:
                # Skip this subplot if data is insufficient
                ax.text(0.5, 0.5, 'Insufficient data\nfor change visualization',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, color='gray')
                ax.axis('off')
            
            ax.set_xlabel('x (spatial)', fontsize=11)
            ax.set_ylabel('t (time)', fontsize=11)
            ax.set_title(f'{layer_prev} → {layer_curr}', 
                        fontsize=12, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(n_changes, len(axes_changes)):
            axes_changes[idx].axis('off')
        
        plt.tight_layout()
        changes_path = save_path.parent / "ncc_classification_input_space_accuracy_changes.png"
        plt.savefig(changes_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Input space accuracy changes heatmap saved to {changes_path}")


def visualize_ncc_classification_accuracy_changes(
    x: torch.Tensor,
    t: torch.Tensor,
    class_labels: torch.Tensor,
    predictions_dict: Dict[str, torch.Tensor],
    save_path: Path,
    config: Dict
):
    """
    Visualize NCC accuracy changes between layers in output space (u, v).
    
    For Schrödinger (2D output space), this creates heatmaps showing how
    accuracy changes in the (u, v) plane between consecutive layers.
    
    Args:
        x: Spatial coordinates (N, 1) - not used for output space plot
        t: Temporal coordinates (N, 1) - not used for output space plot
        class_labels: True class labels (N,)
        predictions_dict: Dict mapping layer_name -> predictions (N,)
        save_path: Path to save figure
        config: Configuration dictionary
    """
    # This functionality is embedded in visualize_ncc_classification_heatmap
    # For now, create a placeholder that indicates it's handled elsewhere
    print(f"  NCC accuracy changes (output space) - see classification_heatmap")


def visualize_ncc_classification_input_space_accuracy_changes(
    x: torch.Tensor,
    t: torch.Tensor,
    class_labels: torch.Tensor,
    predictions_dict: Dict[str, torch.Tensor],
    save_path: Path,
    config: Dict
):
    """
    Visualize NCC accuracy changes between layers in input space (x, t).
    
    This functionality is embedded within visualize_ncc_classification_input_space_heatmap.
    Calling this function will trigger the generation of the changes plot.
    
    Args:
        x: Spatial coordinates (N, 1)
        t: Temporal coordinates (N, 1)
        class_labels: True class labels (N,)
        predictions_dict: Dict mapping layer_name -> predictions (N,)
        save_path: Path to save figure
        config: Configuration dictionary
    """
    # The changes plot is already generated by visualize_ncc_classification_input_space_heatmap
    # This is just a stub to match the expected interface
    print(f"  NCC accuracy changes (input space) - generated by input_space_heatmap function")