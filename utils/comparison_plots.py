"""Shared comparison plotting functions for multi-experiment analysis."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import colors as mcolors
from matplotlib.colors import LogNorm
import os
from typing import Dict


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


def _build_color_map(model_names):
    """
    Deterministic color assignment per model, shared across all comparison plots.
    Uses distinct, high-contrast colors for clear visual differentiation.
    """
    # Use a diverse set of distinct colors
    distinct_colors = [
        '#e41a1c',  # Red
        '#377eb8',  # Blue
        '#4daf4a',  # Green
        '#984ea3',  # Purple
        '#ff7f00',  # Orange
        '#ffff33',  # Yellow
        '#a65628',  # Brown
        '#f781bf',  # Pink
        '#999999',  # Gray
        '#66c2a5',  # Teal
        '#fc8d62',  # Salmon
        '#8da0cb',  # Light blue
        '#e78ac3',  # Light pink
        '#a6d854',  # Light green
        '#ffd92f',  # Gold
        '#e5c494',  # Tan
        '#b3b3b3',  # Light gray
        '#1b9e77',  # Dark teal
        '#d95f02',  # Dark orange
        '#7570b3',  # Dark purple
    ]
    
    sorted_names = sorted(model_names)
    color_map = {}
    for idx, name in enumerate(sorted_names):
        color_map[name] = distinct_colors[idx % len(distinct_colors)]
    return color_map


def _long_path(path: Path) -> Path:
    """Prefix Windows paths to bypass MAX_PATH limits when needed."""
    resolved = path.resolve()
    p_str = str(resolved)
    if os.name == "nt" and not p_str.startswith("\\\\?\\"):
        return Path("\\\\?\\" + p_str)
    return resolved


def generate_ncc_classification_plot(save_dir, ncc_data):
    """Generate NCC classification accuracy comparison across layers and epochs."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    model_names = sorted(ncc_data.keys())
    color_map = _build_color_map(model_names)
    
    for model_idx, model_name in enumerate(model_names):
        epochs_data = ncc_data[model_name]
        if 'final' not in epochs_data:
            continue
        ncc_metrics = epochs_data['final']
        layers = ncc_metrics['layers_analyzed']
        accuracies = [ncc_metrics['layer_accuracies'][layer] for layer in layers]
        base_color = color_map[model_name]
        ax.plot(
            layers,
            accuracies,
            marker='o',
            color=base_color,
            linewidth=2,
            markersize=6,
            alpha=0.9,
            label=model_name
        )
    
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('NCC Classification Accuracy Comparison\n(Darker shades = later epochs)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    save_path = _long_path(Path(save_dir) / "ncc_classification_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  NCC classification comparison saved to {save_path}")


def generate_ncc_compactness_plot(save_dir, ncc_data):
    """Generate NCC compactness (margin) comparison across layers and epochs."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    model_names = sorted(ncc_data.keys())
    color_map = _build_color_map(model_names)
    
    for model_idx, model_name in enumerate(model_names):
        epochs_data = ncc_data[model_name]
        if 'final' not in epochs_data:
            continue
        ncc_metrics = epochs_data['final']
        layers = ncc_metrics['layers_analyzed']
        
        margin_snrs = []
        for layer in layers:
            mean = ncc_metrics['layer_margins'][layer]['mean_margin']
            std = ncc_metrics['layer_margins'][layer]['std_margin']
            snr = mean / std if std > 0 else 0
            margin_snrs.append(snr)
            
        base_color = color_map[model_name]
            
        ax.plot(
            layers,
            margin_snrs,
            marker='o',
            color=base_color,
            linewidth=2,
            markersize=6,
            alpha=0.9,
            label=model_name
        )
    
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Margin SNR (mean/std)', fontsize=12, fontweight='bold')
    ax.set_title('NCC Compactness (Margin SNR) Comparison\n(Darker shades = later epochs)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    save_path = _long_path(Path(save_dir) / "ncc_compactness_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  NCC compactness comparison saved to {save_path}")


def generate_ncc_comparison_plots_only(save_dir, ncc_data):
    """
    Generate NCC comparison plots for multi-checkpoint evaluation.
    
    Args:
        save_dir: Directory to save plots
        ncc_data: Dict of {model_name: {epoch: ncc_metrics}}
    """
    print("\nGenerating NCC comparison plots...")
    generate_ncc_classification_plot(save_dir, ncc_data)
    generate_ncc_compactness_plot(save_dir, ncc_data)
    print(f"Comparison plots saved to {save_dir}")


def generate_probe_comparison_plots(save_dir, probe_data):
    """
    Generate linear probe comparison plots across experiments.
    
    Args:
        save_dir: Directory to save plots
        probe_data: Dict of {experiment_name: probe_metrics}
        where probe_metrics has 'train'/'eval' with 'rel_l2' and 'inf_norm' lists
    """
    print("\nGenerating probe comparison plots...")
    
    exp_names = sorted(probe_data.keys())
    color_map = _build_color_map(exp_names)
    
    # Create figure with 2 subplots (one for train, one for eval)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for exp_name in exp_names:
        metrics = probe_data[exp_name]
        color = color_map[exp_name]
        
        # Number of layers
        num_layers = len(metrics['train']['rel_l2'])
        layer_numbers = list(range(1, num_layers + 1))
        
        # Plot training metrics
        ax1.plot(layer_numbers, metrics['train']['rel_l2'], 
                marker='o', color=color, linewidth=2, markersize=7,
                label=exp_name, alpha=0.8)
        
        # Plot evaluation metrics
        ax2.plot(layer_numbers, metrics['eval']['rel_l2'], 
                marker='s', color=color, linewidth=2, markersize=7,
                label=exp_name, alpha=0.8)
    
    # Configure training plot
    ax1.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probe Rel-L2 Error', fontsize=12, fontweight='bold')
    ax1.set_title('Linear Probe Performance - Training Data', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='best')
    
    # Configure evaluation plot
    ax2.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probe Rel-L2 Error', fontsize=12, fontweight='bold')
    ax2.set_title('Linear Probe Performance - Evaluation Data', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    # Save plot
    save_path = _long_path(Path(save_dir) / "probe_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Probe comparison plot saved to {save_path}")
    
    # Infinity norm comparison (train/eval)
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
    for exp_name in exp_names:
        metrics = probe_data[exp_name]
        color = color_map[exp_name]
        num_layers = len(metrics['train']['inf_norm'])
        layer_numbers = list(range(1, num_layers + 1))
        
        ax3.plot(layer_numbers, metrics['train']['inf_norm'],
                 marker='o', color=color, linewidth=2, markersize=7,
                 label=exp_name, alpha=0.8)
        ax4.plot(layer_numbers, metrics['eval']['inf_norm'],
                 marker='s', color=color, linewidth=2, markersize=7,
                 label=exp_name, alpha=0.8)
    
    ax3.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Probe L∞ Error', fontsize=12, fontweight='bold')
    ax3.set_title('Linear Probe Infinity Norm - Training Data',
                  fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10, loc='best')
    
    ax4.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Probe L∞ Error', fontsize=12, fontweight='bold')
    ax4.set_title('Linear Probe Infinity Norm - Evaluation Data',
                  fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    inf_path = _long_path(Path(save_dir) / "probe_comparison_inf.png")
    plt.savefig(inf_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Probe infinity-norm comparison saved to {inf_path}")


def _lookup_metric(exp_data, section_key, split, layer_name, metric_name):
    if section_key is None:
        layer_metrics = exp_data.get(split, {}).get(layer_name, {})
    else:
        layer_metrics = (
            exp_data.get(section_key, {})
            .get(split, {})
            .get(layer_name, {})
        )
    return layer_metrics.get(metric_name, np.nan)


def _plot_derivative_comparison_grid(save_dir, derivatives_data, section_key, key_map, title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    exp_names = sorted(derivatives_data.keys())
    color_map = _build_color_map(exp_names)
    max_layers = max(len(data['layers_analyzed']) for data in derivatives_data.values())
    
    panel_cfg = [
        (axes[0, 0], 'train', 'l2', 'Train L2'),
        (axes[0, 1], 'eval', 'l2', 'Eval L2'),
        (axes[1, 0], 'train', 'linf', 'Train L∞'),
        (axes[1, 1], 'eval', 'linf', 'Eval L∞'),
    ]
    
    for panel_idx, (ax, split, metric_id, panel_title) in enumerate(panel_cfg):
        all_panel_values = []
        for exp_name in exp_names:
            exp_data = derivatives_data[exp_name]
            layers = exp_data['layers_analyzed']
            layer_indices = list(range(1, len(layers) + 1))
            values = []
            metric_key = key_map[metric_id]
            for layer_name in layers:
                values.append(
                    _lookup_metric(exp_data, section_key, split, layer_name, metric_key)
                )
            ax.plot(
                layer_indices,
                values,
                marker='o',
                linewidth=2,
                markersize=6,
                label=exp_name,
                color=color_map[exp_name]
            )
            all_panel_values.extend(values)
        
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel('Mean Norm', fontsize=11)
        ax.set_xticks(range(1, max_layers + 1))
        is_log = _safe_log_scale(ax, [all_panel_values])
        scale_str = "[log]" if is_log else "[linear]"
        ax.set_title(f'{panel_title} {scale_str}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if panel_idx == 0:
            ax.legend(fontsize=10, loc='best')
    
    # Determine overall scale for suptitle
    scale_label = "(Log Scale)" if is_log else "(Linear Scale)"
    fig.suptitle(f'{title} {scale_label}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    save_path = _long_path(Path(save_dir) / filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {title} saved to {save_path}")


def generate_derivatives_comparison_plots(save_dir, derivatives_data):
    """
    Generate comparison plots for derivatives tracking across experiments.
    """
    print("\nGenerating derivatives comparison plots...")
    
    _plot_derivative_comparison_grid(
        save_dir,
        derivatives_data,
        section_key=None,
        key_map={'l2': 'residual_norm', 'linf': 'residual_inf_norm'},
        title='Derivatives residual comparison',
        filename="derivatives_residual_comparison.png"
    )
    _plot_derivative_comparison_grid(
        save_dir,
        derivatives_data,
        section_key='ic',
        key_map={'l2': 'l2', 'linf': 'linf'},
        title='Derivatives IC comparison',
        filename="derivatives_ic_comparison.png"
    )
    _plot_derivative_comparison_grid(
        save_dir,
        derivatives_data,
        section_key='bc_value',
        key_map={'l2': 'l2', 'linf': 'linf'},
        title='Derivatives BC value comparison',
        filename="derivatives_bc_value_comparison.png"
    )
    _plot_derivative_comparison_grid(
        save_dir,
        derivatives_data,
        section_key='bc_derivative',
        key_map={'l2': 'l2', 'linf': 'linf'},
        title='Derivatives BC derivative comparison',
        filename="derivatives_bc_derivative_comparison.png"
    )


def generate_frequency_coverage_comparison(
    output_dir: Path,
    frequency_data: Dict[str, Dict]
) -> None:
    """
    Generate frequency coverage comparison as continuous line plot with GT reference.
    
    Creates a 2-subplot figure:
    - Top: Ground Truth radial power spectrum (shows where the signal is)
    - Bottom: Relative error per model (shows where errors are)
    
    Args:
        output_dir: Directory to save plot
        frequency_data: Dict mapping model_name -> frequency_metrics dict
    """
    # Collect data for all models
    model_names = list(frequency_data.keys())
    if not model_names:
        return
    
    # Check if we have spectral_efficiency data (from frequency_metrics.json)
    models_with_data = {}
    for model_name in model_names:
        metrics = frequency_data[model_name]
        if 'spectral_efficiency' in metrics and metrics['spectral_efficiency']:
            models_with_data[model_name] = metrics['spectral_efficiency']
    
    if not models_with_data:
        print("  No spectral_efficiency data found in frequency metrics")
        return
    
    # Check if GT radial power is available
    first_model_data = list(models_with_data.values())[0]
    has_gt_spectrum = 'gt_radial_power' in first_model_data
    
    # Create figure with 2 subplots if GT spectrum available, else 1
    if has_gt_spectrum:
        fig, (ax_gt, ax_err) = plt.subplots(2, 1, figsize=(12, 10), 
                                             gridspec_kw={'height_ratios': [1, 2]})
    else:
        fig, ax_err = plt.subplots(figsize=(12, 7))
        ax_gt = None
    
    n_models = len(models_with_data)
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    # Plot GT spectrum if available
    if has_gt_spectrum and ax_gt is not None:
        k_radial_bins = np.array(first_model_data['k_radial_bins'])
        gt_radial_power = np.array(first_model_data['gt_radial_power'])
        
        ax_gt.fill_between(k_radial_bins, 0, gt_radial_power, alpha=0.4, color='steelblue')
        ax_gt.plot(k_radial_bins, gt_radial_power, color='steelblue', linewidth=2, label='GT Power')
        
        ax_gt.set_xlabel('Radial Frequency |k| (Hz)', fontsize=11)
        ax_gt.set_ylabel('Power |FFT|²', fontsize=11)
        
        # Try log scale
        if np.all(gt_radial_power > 0):
            ax_gt.set_yscale('log')
            scale_str_gt = '[log]'
        else:
            scale_str_gt = '[linear]'
        
        ax_gt.set_title(f'Ground Truth Frequency Content: |FFT(h_gt)|² {scale_str_gt}', 
                       fontsize=11, fontweight='bold')
        ax_gt.grid(True, alpha=0.3)
        ax_gt.legend(loc='upper right', fontsize=9)
    
    # Plot model errors
    all_errors = []
    
    for idx, (model_name, spectral_data) in enumerate(models_with_data.items()):
        k_radial_bins = np.array(spectral_data['k_radial_bins'])
        error_matrix = np.array(spectral_data['error_matrix'])  # Shape: [layers, freq_bins]
        
        # Get final layer error (last row)
        final_layer_error = error_matrix[-1]
        
        all_errors.extend(final_layer_error[final_layer_error > 0].tolist())
        
        # Plot continuous line
        ax_err.plot(k_radial_bins, final_layer_error, color=colors[idx], 
                   label=model_name, linewidth=2.5, alpha=0.85)
    
    ax_err.set_xlabel('Radial Frequency |k| (Hz)', fontsize=12, fontweight='bold')
    ax_err.set_ylabel('Relative Error', fontsize=12, fontweight='bold')
    
    # Log scale for y-axis
    scale_str = '[linear]'
    if all_errors and min(all_errors) > 0:
        ax_err.set_yscale('log')
        scale_str = '[log]'
    
    ax_err.set_title(f'Model Errors: |FFT(ĥ - h_gt)|² / |FFT(h_gt)|² {scale_str}\n(Lower = Better)', 
                    fontsize=11, fontweight='bold')
    ax_err.legend(loc='upper left', fontsize=10, ncol=1)
    ax_err.grid(True, alpha=0.3)
    
    plt.suptitle('Frequency Coverage Comparison (Final Layer)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    save_path = _long_path(Path(output_dir) / 'frequency_coverage_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Frequency coverage comparison saved to {save_path}")


def plot_spectral_learning_efficiency_comparison(
    frequency_data: Dict[str, Dict],
    output_dir: Path
) -> None:
    """
    Generate side-by-side comparison of spectral learning efficiency for all models.
    
    Creates a grid layout (2 columns, multiple rows) showing each model's
    spectral learning efficiency heatmap.
    
    Args:
        frequency_data: Dict mapping model_name -> frequency_metrics dict
        output_dir: Directory to save plot
    """
    model_names = list(frequency_data.keys())
    if not model_names:
        return
    
    # Check if all models have spectral efficiency data
    models_with_data = {}
    for model_name, metrics in frequency_data.items():
        if 'spectral_efficiency' in metrics:
            models_with_data[model_name] = metrics['spectral_efficiency']
    
    if not models_with_data:
        print("  No spectral efficiency data found in frequency metrics")
        return
    
    n_models = len(models_with_data)
    
    # Calculate grid layout: 2 columns, multiple rows
    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 8 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else axes.reshape(1, 1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for idx, (model_name, spectral_data) in enumerate(models_with_data.items()):
        ax = axes_flat[idx]
        
        # Extract error matrix and k_radial bins
        error_matrix = np.array(spectral_data['error_matrix']).T  # Transpose: rows = freq, cols = layers
        k_radial_ref = np.array(spectral_data['k_radial_bins'])
        
        n_freq_bins, n_layers = error_matrix.shape
        
        # Get layer names from frequency_data
        layers = frequency_data[model_name]['layers_analyzed']
        
        # Create heatmap
        im = ax.imshow(error_matrix, aspect='auto', cmap='viridis_r', 
                       interpolation='bilinear', origin='lower')
        
        # Set ticks
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
        ax.set_xlabel('Layer', fontsize=10, fontweight='bold')
        
        # Y-axis: frequency bins
        y_ticks = np.linspace(0, n_freq_bins - 1, min(10, n_freq_bins))
        y_tick_labels = [f'{k_radial_ref[int(i)]:.1f}' for i in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        ax.set_ylabel('|k| (Hz)', fontsize=10, fontweight='bold')
        
        # Colorbar for each subplot
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Try log scale for colorbar
        scale_str = '[linear]'
        if np.all(error_matrix > 0):
            im.set_norm(LogNorm(vmin=error_matrix[error_matrix > 0].min(), 
                               vmax=error_matrix.max()))
            scale_str = '[log]'
        cbar.set_label(f'Relative Error {scale_str}', fontsize=9, rotation=270, labelpad=15)
        
        ax.set_title(f'{model_name}\n(Lower = Better)', 
                     fontsize=11, fontweight='bold', pad=10)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.suptitle('Spectral Learning Efficiency Comparison\nRelative Error |FFT(error)|²/|FFT(gt)|² by Layer and Frequency', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    save_path = _long_path(Path(output_dir) / 'spectral_learning_efficiency_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Spectral learning efficiency comparison saved to {save_path}")