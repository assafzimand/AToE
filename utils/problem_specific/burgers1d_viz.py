"""
1D Burgers equation specific visualizations.

Provides custom visualization for:
1. Dataset visualization: heatmap of h(x,t)
2. Evaluation visualization: comprehensive model performance analysis
3. NCC visualizations: classification heatmaps in output and input spaces
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict



def visualize_dataset(data_dict: Dict, save_dir: Path, config: Dict, split_name: str):
    """
    Visualize Burgers equation dataset with heatmap.
    
    Creates heatmap of h(x,t) - real-valued solution field.
    
    Args:
        data_dict: Dataset dictionary with 'x', 't', 'h_gt' tensors
        save_dir: Directory to save visualization
        config: Configuration dictionary
        split_name: Name of split ('training' or 'evaluation')
    """
    # Extract data
    x = data_dict['x'].cpu().numpy()  # (N, spatial_dim)
    t = data_dict['t'].cpu().numpy()  # (N, 1)
    h_gt = data_dict['h_gt'].cpu().numpy()  # (N, 1)
    
    # Flatten coordinates
    x_flat = x[:, 0]
    t_flat = t[:, 0]
    h = h_gt[:, 0]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot: h(x,t)
    scatter = ax.scatter(x_flat, t_flat, c=h, cmap='RdBu_r', s=5, alpha=0.6,
                        vmin=-np.abs(h).max(), vmax=np.abs(h).max())
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('t', fontsize=12)
    ax.set_title(f'h(x,t) - {split_name.capitalize()}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('h', fontsize=11)
    
    plt.tight_layout()
    
    # Save
    save_path = save_dir / f"dataset_{split_name.lower()}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  {split_name.capitalize()} dataset visualization saved to {save_path}")


def visualize_evaluation(model: torch.nn.Module, save_dir: Path, config: Dict):
    from utils.problem_specific.generic_viz import plot_predictions_and_error_maps
    plot_predictions_and_error_maps(
        model, save_dir, config,
        filename="pred_final_burgers1d_relL2_{relL2}.png")
