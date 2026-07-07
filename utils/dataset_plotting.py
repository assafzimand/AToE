"""Plotting utilities for dataset visualization."""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import numpy as np


def plot_dataset(data: Dict[str, torch.Tensor], save_path: str, title: str = "Dataset Visualization") -> None:
    """
    Visualize a PINN dataset showing point distributions and ground truth.
    
    Args:
        data: Dataset dictionary with x, t, h_gt, and mask keys
        save_path: Path to save the figure
        title: Title for the plot
    """
    # Move data to CPU for plotting
    x = data['x'].cpu().numpy()
    t = data['t'].cpu().numpy()
    h_gt = data['h_gt'].cpu().numpy()
    mask_residual = data['mask']['residual'].cpu().numpy()
    mask_ic = data['mask']['IC'].cpu().numpy()
    mask_bc = data['mask']['BC'].cpu().numpy()
    
    spatial_dim = x.shape[1]
    output_dim = h_gt.shape[1]
    
    # Create figure based on spatial dimension
    if spatial_dim == 1:
        # Determine number of subplots based on output_dim
        n_plots = 1 + output_dim  # 1 for point distribution + 1 per output component
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        
        # Plot 1: Point distribution in (x, t) space
        ax = axes[0]
        ax.scatter(x[mask_residual, 0], t[mask_residual, 0], 
                  c='blue', s=1, alpha=0.5, label='Residual')
        ax.scatter(x[mask_ic, 0], t[mask_ic, 0], 
                  c='green', s=10, alpha=0.7, label='IC (t=0)')
        ax.scatter(x[mask_bc, 0], t[mask_bc, 0], 
                  c='red', s=10, alpha=0.7, label='BC (boundaries)')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title('Point Distribution')
        ax.legend(markerscale=3)
        ax.grid(True, alpha=0.3)
        
        # Plots 2+: Ground truth components
        cmaps = ['viridis', 'plasma', 'inferno', 'magma']
        for i in range(output_dim):
            ax = axes[1 + i]
            scatter = ax.scatter(x[:, 0], t[:, 0], c=h_gt[:, i], 
                               s=2, cmap=cmaps[i % len(cmaps)], alpha=0.6)
            ax.set_xlabel('x')
            ax.set_ylabel('t')
            ax.set_title(f'Ground Truth h_{i}(x,t)')
            plt.colorbar(scatter, ax=ax)
            ax.grid(True, alpha=0.3)
        
    else:
        # For higher dimensions, show projections or simplified views
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Point distribution - first 2 spatial dims
        ax = axes[0, 0]
        ax.scatter(x[mask_residual, 0], x[mask_residual, min(1, spatial_dim-1)], 
                  c='blue', s=1, alpha=0.5, label='Residual')
        ax.scatter(x[mask_ic, 0], x[mask_ic, min(1, spatial_dim-1)], 
                  c='green', s=10, alpha=0.7, label='IC')
        ax.scatter(x[mask_bc, 0], x[mask_bc, min(1, spatial_dim-1)], 
                  c='red', s=10, alpha=0.7, label='BC')
        ax.set_xlabel('x₀')
        ax.set_ylabel(f'x₁' if spatial_dim > 1 else 'x₀')
        ax.set_title('Spatial Point Distribution')
        ax.legend(markerscale=3)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Time distribution
        ax = axes[0, 1]
        ax.hist(t[mask_residual, 0], bins=30, alpha=0.5, label='Residual', color='blue')
        ax.hist(t[mask_ic, 0], bins=30, alpha=0.7, label='IC', color='green')
        ax.hist(t[mask_bc, 0], bins=30, alpha=0.7, label='BC', color='red')
        ax.set_xlabel('t')
        ax.set_ylabel('Count')
        ax.set_title('Temporal Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Ground truth component 0
        ax = axes[1, 0]
        scatter = ax.scatter(x[:, 0], t[:, 0], c=h_gt[:, 0], 
                           s=2, cmap='viridis', alpha=0.6)
        ax.set_xlabel('x₀')
        ax.set_ylabel('t')
        ax.set_title('Ground Truth h₀')
        plt.colorbar(scatter, ax=ax)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Ground truth component 1 (if exists)
        ax = axes[1, 1]
        if output_dim > 1:
            scatter = ax.scatter(x[:, 0], t[:, 0], c=h_gt[:, 1], 
                               s=2, cmap='plasma', alpha=0.6)
        else:
            scatter = ax.scatter(x[:, 0], t[:, 0], c=h_gt[:, 0], 
                               s=2, cmap='viridis', alpha=0.6)
        ax.set_xlabel('x₀')
        ax.set_ylabel('t')
        ax.set_title('Ground Truth h₁')
        plt.colorbar(scatter, ax=ax)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Dataset visualization saved to {save_path}")


def save_spawn_prediction_plot(
    model,
    domain_bounds,
    gt_grid: np.ndarray,
    grid_x: np.ndarray,
    grid_t: np.ndarray,
    output_path,
    epoch: int,
    cfg: dict,
    output_names=None,
    resolution: int = 200,
) -> None:
    """Save GT | Pred | |Error| heatmaps for the current model state.

    Thin wrapper over the unified renderer
    (utils.problem_specific.generic_viz.plot_predictions_and_error_maps),
    which evaluates on the solver's NATIVE solution grid. Interpolating the
    ground truth between grid nodes (the old behavior here) painted large
    fake error needles across steep fronts.

    Args:
        model: The PINN model (torch.nn.Module), will be called in eval+no_grad.
        domain_bounds: (Unused, kept for signature compatibility.)
        gt_grid: (Deprecated, unused.)
        grid_x: (Deprecated, unused.)
        grid_t: (Deprecated, unused.)
        output_path: File path to save the PNG.
        epoch: Current epoch (used in title).
        cfg: Full config dict.
        output_names: (Unused; labels come from the problem metadata.)
        resolution: Fallback dense-grid resolution when the solver has no
            cached native grid.
    """
    from utils.problem_specific.generic_viz import plot_predictions_and_error_maps

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_predictions_and_error_maps(
        model,
        output_path.parent,
        cfg,
        filename=output_path.name,
        n_x=resolution,
        n_t=resolution,
    )


def plot_dataset_statistics(data: Dict[str, torch.Tensor], save_path: str) -> None:
    """
    Plot statistical information about the dataset.
    
    Args:
        data: Dataset dictionary
        save_path: Path to save the figure
    """
    x = data['x'].cpu().numpy()
    t = data['t'].cpu().numpy()
    h_gt = data['h_gt'].cpu().numpy()
    
    mask_residual = data['mask']['residual'].cpu().numpy()
    mask_ic = data['mask']['IC'].cpu().numpy()
    mask_bc = data['mask']['BC'].cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Point type distribution
    ax = axes[0, 0]
    counts = [mask_residual.sum(), mask_ic.sum(), mask_bc.sum()]
    labels = ['Residual', 'IC', 'BC']
    colors = ['blue', 'green', 'red']
    ax.bar(labels, counts, color=colors, alpha=0.7)
    ax.set_ylabel('Number of Points')
    ax.set_title('Point Type Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (label, count) in enumerate(zip(labels, counts)):
        ax.text(i, count, f'{count}', ha='center', va='bottom')
    
    # Ground truth distribution
    ax = axes[0, 1]
    output_dim = h_gt.shape[1]
    colors = ['purple', 'orange', 'green', 'red']
    for i in range(output_dim):
        ax.hist(h_gt[:, i], bins=50, alpha=0.6, label=f'h_{i}', color=colors[i % len(colors)])
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Ground Truth Value Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Spatial coverage
    ax = axes[1, 0]
    for dim in range(x.shape[1]):
        ax.hist(x[:, dim], bins=50, alpha=0.5, label=f'x_{dim}')
    ax.set_xlabel('Spatial Coordinate')
    ax.set_ylabel('Frequency')
    ax.set_title('Spatial Coverage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Temporal coverage
    ax = axes[1, 1]
    ax.hist(t[:, 0], bins=50, alpha=0.7, color='teal')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Frequency')
    ax.set_title('Temporal Coverage')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Dataset statistics saved to {save_path}")

