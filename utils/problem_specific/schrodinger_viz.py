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
    plot_predictions_and_error_maps(
        model, save_dir, config,
        filename="pred_final_schrodinger_relL2_{relL2}.png")

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
