"""Plotting utilities for frequency tracking."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from frequency_tracker.frequency_core import compute_binned_frequency_errors
from matplotlib.colors import LogNorm


def _safe_log_scale(ax, values_list):
    """
    Set log scale on y-axis only if all data has positive values.
    
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
    # Filter out NaN and zero values for the check
    valid_values = all_values[~np.isnan(all_values) & (all_values > 0)]
    if len(valid_values) > 0:
        ax.set_yscale('log')
        return True
    return False


def plot_learned_frequencies(
    freq_results: Dict[str, Dict],
    h_gt_spectrum: Dict,
    save_dir: Path,
    config: Dict
) -> None:
    """
    Generate the learned_frequencies visualization.
    
    Creates one figure with all layers overlaid:
    - Row 1: Cumulative Error - relative error at each frequency for layer i
    - Row 2: Error Change Ratio - error ratio E_i / E_{i-1} from layer i-1 to layer i
    - Columns: k_x, k_t, |k| (or k_x0, k_x1, k_t, |k| for 2D)
    
    All values are RELATIVE to ground truth power for comparability.
    
    Args:
        freq_results: Dictionary mapping layer_name -> results dict
        h_gt_spectrum: Ground truth spectrum dict
        save_dir: Directory to save plot
        config: Configuration dictionary
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    problem = config['problem']
    problem_cfg = config[problem]
    spatial_dim = problem_cfg.get('spatial_dim', 1)
    
    layer_names = sorted(freq_results.keys())
    n_layers = len(layer_names)
    
    # Determine column names based on spatial_dim
    if spatial_dim == 2:
        dim_names = ['x0', 'x1', 't', 'radial']
        n_cols = 4
    else:
        dim_names = ['x', 't', 'radial']
        n_cols = 3
    
    n_rows = 3  # GT Spectrum, Cumulative Error, Added (Improvement)
    n_bins = 40  # Number of bins for marginal spectra
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Use colormap for layers
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n_layers - 1, 1)) for i in range(n_layers)]
    
    # Pre-compute ground truth power spectrum (marginal) for each dimension
    gt_power = h_gt_spectrum['power']
    gt_freqs = h_gt_spectrum['freqs']
    n_dims_gt = len(gt_freqs)
    if gt_power.ndim > n_dims_gt:
        gt_power_avg = gt_power.mean(axis=-1)
    else:
        gt_power_avg = gt_power
    
    def compute_marginal(power_nd, freqs, dim_name, spatial_dim):
        """Compute 1D marginal spectrum for a specific dimension."""
        n_dims = len(freqs)
        if power_nd.ndim > n_dims:
            power_avg = power_nd.mean(axis=-1)
        else:
            power_avg = power_nd
        
        if dim_name == 'radial':
            mesh_freqs = np.meshgrid(*freqs, indexing='ij')
            k_magnitude = np.sqrt(sum(f**2 for f in mesh_freqs))
            k_max = k_magnitude.max()
            if k_max == 0:
                k_max = 1.0
            k_bin_edges = np.linspace(0, k_max, n_bins + 1)
            k_bin_centers = (k_bin_edges[:-1] + k_bin_edges[1:]) / 2
            
            power_binned = np.zeros(n_bins)
            for bin_idx in range(n_bins):
                if bin_idx == n_bins - 1:
                    mask = (k_magnitude >= k_bin_edges[bin_idx]) & (k_magnitude <= k_bin_edges[bin_idx + 1])
                else:
                    mask = (k_magnitude >= k_bin_edges[bin_idx]) & (k_magnitude < k_bin_edges[bin_idx + 1])
                if mask.sum() > 0:
                    power_binned[bin_idx] = power_avg[mask].mean()
            
            return k_bin_centers, power_binned
        else:
            if spatial_dim == 2:
                dim_idx = {'x0': 0, 'x1': 1, 't': 2}.get(dim_name, 0)
            else:
                dim_idx = {'x': 0, 't': 1}.get(dim_name, 0)
            
            freq_1d = freqs[dim_idx]
            axes_to_avg = tuple(j for j in range(n_dims) if j != dim_idx)
            if axes_to_avg:
                power_1d = power_avg.mean(axis=axes_to_avg)
            else:
                power_1d = power_avg
            return freq_1d, power_1d
    
    # Pre-compute GT marginals and error spectra for all layers
    # Only keep positive frequencies (negative frequencies mirror positive for real signals)
    gt_marginals = {}
    for dim_name in dim_names:
        freq_1d, power_1d = compute_marginal(gt_power, gt_freqs, dim_name, spatial_dim)
        if dim_name != 'radial':
            pos_mask = freq_1d >= 0
            freq_1d = freq_1d[pos_mask]
            power_1d = power_1d[pos_mask]
        gt_marginals[dim_name] = (freq_1d, power_1d)
    
    # Row 0: Ground Truth Frequency Content
    for col_idx, dim_name in enumerate(dim_names):
        ax = axes[0, col_idx]
        
        gt_freq, gt_power_1d = gt_marginals[dim_name]
        
        # Plot GT spectrum as filled area
        ax.fill_between(gt_freq, 0, gt_power_1d, alpha=0.4, color='steelblue', label='GT Power')
        ax.plot(gt_freq, gt_power_1d, color='steelblue', linewidth=2, alpha=0.9)
        
        # Formatting
        if dim_name == 'radial':
            ax.set_xlabel('|k| (Radial Frequency, Hz)', fontsize=11)
        else:
            ax.set_xlabel(f'k_{dim_name} (Hz)', fontsize=11)
        
        ax.set_ylabel('Power |FFT|²', fontsize=11)
        
        # Log scale for y-axis
        is_log = _safe_log_scale(ax, [gt_power_1d])
        scale_str = "[log]" if is_log else "[linear]"
        
        ax.set_title(f'Ground Truth Frequency Content: |FFT(h_gt)|²\n{scale_str}', 
                    fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if col_idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    # Compute error spectra: |FFT(ĥ_i - h_gt)|² for each layer
    # Note: leftover spectrum = |FFT(h_gt - ĥ_i)|² = |FFT(ĥ_i - h_gt)|² (same magnitude)
    # Only keep positive frequencies
    layer_error_marginals = {}
    for layer_name in layer_names:
        layer_data = freq_results[layer_name]
        error_spectrum = layer_data['leftover']  # This is |FFT(h_gt - ĥ_i)|²
        error_power = error_spectrum['power']
        error_freqs = error_spectrum['freqs']
        
        layer_error_marginals[layer_name] = {}
        for dim_name in dim_names:
            freq_1d, error_1d = compute_marginal(error_power, error_freqs, dim_name, spatial_dim)
            if dim_name != 'radial':
                pos_mask = freq_1d >= 0
                freq_1d = freq_1d[pos_mask]
                error_1d = error_1d[pos_mask]
            layer_error_marginals[layer_name][dim_name] = (freq_1d, error_1d)
    
    # Row 1: Cumulative Error (relative error at layer i)
    for col_idx, dim_name in enumerate(dim_names):
        ax = axes[1, col_idx]
        all_values = []
        
        gt_freq, gt_power_1d = gt_marginals[dim_name]
        gt_power_safe = np.where(gt_power_1d > 1e-15, gt_power_1d, 1e-15)
        
        for layer_idx, layer_name in enumerate(layer_names):
            freq_1d, error_1d = layer_error_marginals[layer_name][dim_name]
            
            # Relative error: error_power / gt_power
            relative_error = error_1d / gt_power_safe
            
            all_values.append(relative_error)
            
            ax.plot(freq_1d, relative_error, color=colors[layer_idx], 
                   label=layer_name, linewidth=2, alpha=0.8)
        
        # Formatting
        if dim_name == 'radial':
            ax.set_xlabel('|k| (Radial Frequency, Hz)', fontsize=11)
        else:
            ax.set_xlabel(f'k_{dim_name} (Hz)', fontsize=11)
        
        ax.set_ylabel('Relative Error', fontsize=11)
        
        # Log scale for y-axis
        is_log = _safe_log_scale(ax, all_values)
        scale_str = "[log]" if is_log else "[linear]"
        
        # Title with math expression
        ax.set_title(f'Error at Layer i: |FFT(ĥᵢ - h_gt)|² / |FFT(h_gt)|²\n'
                    f'(Lower = Better) {scale_str}', fontsize=10, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.legend(loc='upper right', fontsize=7, ncol=1)
    
    # Row 2: Error Change Ratio from layer i-1 to layer i
    for col_idx, dim_name in enumerate(dim_names):
        ax = axes[2, col_idx]
        all_values = []
        
        gt_freq, gt_power_1d = gt_marginals[dim_name]
        gt_power_safe = np.where(gt_power_1d > 1e-15, gt_power_1d, 1e-15)
        
        # Reference line at 1 (no change)
        ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='No change (ratio = 1)')
        
        prev_error = None
        for layer_idx, layer_name in enumerate(layer_names):
            freq_1d, error_1d = layer_error_marginals[layer_name][dim_name]
            
            # Relative error at this layer
            rel_error_i = error_1d / gt_power_safe
            
            if prev_error is None:
                # First layer: Compare to "nothing learned" (error = GT power = 1.0 relative)
                # Ratio = rel_error_i / 1.0 = rel_error_i
                error_ratio = rel_error_i
            else:
                # Error ratio: E_i / E_{i-1}
                # < 1 = improvement (error decreased), > 1 = degradation (error increased)
                eps = 1e-10
                error_ratio = rel_error_i / (prev_error + eps)
            
            all_values.append(error_ratio)
            
            ax.plot(freq_1d, error_ratio, color=colors[layer_idx], 
                   label=layer_name, linewidth=2, alpha=0.8)
            
            prev_error = rel_error_i
        
        # Formatting
        if dim_name == 'radial':
            ax.set_xlabel('|k| (Radial Frequency, Hz)', fontsize=11)
        else:
            ax.set_xlabel(f'k_{dim_name} (Hz)', fontsize=11)
        
        ax.set_ylabel('Error Ratio (Eᵢ / Eᵢ₋₁)', fontsize=11)
        
        # Scale with 1.0 as center (log scale might be better, but keep linear for now)
        all_flat = np.concatenate([np.array(v).flatten() for v in all_values])
        valid_values = all_flat[np.isfinite(all_flat) & (all_flat > 0)]
        if len(valid_values) > 0:
            min_val = valid_values.min()
            max_val = valid_values.max()
            # Symmetric log scale around 1
            margin = max(1.0 - min_val, max_val - 1.0) * 1.2
            ax.set_ylim(max(0.01, 1.0 - margin), 1.0 + margin)
        
        # Title
        ax.set_title(f'Error Change Ratio: Eᵢ / Eᵢ₋₁\n'
                    f'(< 1: Improvement, = 1: No change, > 1: Degradation)', 
                    fontsize=10, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        
        # Add shading for improvement (< 1) and degradation (> 1) regions
        ylim = ax.get_ylim()
        ax.axhspan(ylim[0], 1.0, alpha=0.1, color='green')
        ax.axhspan(1.0, ylim[1], alpha=0.1, color='red')
        
        if col_idx == 0:
            ax.legend(loc='upper right', fontsize=7, ncol=1)
    
    plt.suptitle('Frequency Learning Analysis by Layer', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    save_path = save_dir / 'learned_frequencies.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Learned frequencies plot saved to {save_path}")


def plot_spectral_learning_efficiency(
    freq_results: Dict[str, Dict],
    h_gt_spectrum: Dict,
    save_dir: Path,
    config: Dict
) -> None:
    """
    Generate 2D heatmap showing spectral learning efficiency.
    
    X-axis: Layer index
    Y-axis: Radial frequency |k| (binned)
    Color: Leftover error at (layer, |k|) - lower = better learned
    
    Args:
        freq_results: Dictionary mapping layer_name -> results dict
        h_gt_spectrum: Ground truth spectrum dict
        save_dir: Directory to save plot
        config: Configuration dictionary
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    layer_names = sorted(freq_results.keys())
    n_layers = len(layer_names)
    n_bins = 30
    
    # Compute GT radial spectrum
    gt_power = h_gt_spectrum['power']
    gt_freqs = h_gt_spectrum['freqs']
    n_dims = len(gt_freqs)
    if gt_power.ndim > n_dims:
        gt_power_avg = gt_power.mean(axis=-1)
    else:
        gt_power_avg = gt_power
    
    # Create radial frequency grid
    mesh_freqs = np.meshgrid(*gt_freqs, indexing='ij')
    k_magnitude = np.sqrt(sum(f**2 for f in mesh_freqs))
    k_max = k_magnitude.max()
    if k_max == 0:
        k_max = 1.0
    k_bin_edges = np.linspace(0, k_max, n_bins + 1)
    k_bin_centers = (k_bin_edges[:-1] + k_bin_edges[1:]) / 2
    
    # Compute GT radial power
    gt_radial = np.zeros(n_bins)
    for bin_idx in range(n_bins):
        if bin_idx == n_bins - 1:
            mask = (k_magnitude >= k_bin_edges[bin_idx]) & (k_magnitude <= k_bin_edges[bin_idx + 1])
        else:
            mask = (k_magnitude >= k_bin_edges[bin_idx]) & (k_magnitude < k_bin_edges[bin_idx + 1])
        if mask.sum() > 0:
            gt_radial[bin_idx] = gt_power_avg[mask].mean()
    gt_radial_safe = np.where(gt_radial > 1e-15, gt_radial, 1e-15)
    
    # Compute relative error for each layer
    error_matrix = np.zeros((n_bins, n_layers))
    
    for layer_idx, layer_name in enumerate(layer_names):
        layer_data = freq_results[layer_name]
        leftover = layer_data['leftover']
        leftover_power = leftover['power']
        leftover_freqs = leftover['freqs']
        
        if leftover_power.ndim > len(leftover_freqs):
            leftover_avg = leftover_power.mean(axis=-1)
        else:
            leftover_avg = leftover_power
        
        # Compute radial leftover power
        mesh_freqs_l = np.meshgrid(*leftover_freqs, indexing='ij')
        k_mag_l = np.sqrt(sum(f**2 for f in mesh_freqs_l))
        
        leftover_radial = np.zeros(n_bins)
        for bin_idx in range(n_bins):
            if bin_idx == n_bins - 1:
                mask = (k_mag_l >= k_bin_edges[bin_idx]) & (k_mag_l <= k_bin_edges[bin_idx + 1])
            else:
                mask = (k_mag_l >= k_bin_edges[bin_idx]) & (k_mag_l < k_bin_edges[bin_idx + 1])
            if mask.sum() > 0:
                leftover_radial[bin_idx] = leftover_avg[mask].mean()
        
        # Relative error
        error_matrix[:, layer_idx] = leftover_radial / gt_radial_safe
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(error_matrix, aspect='auto', cmap='viridis_r', 
                   interpolation='bilinear', origin='lower')
    
    # Set ticks
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    
    # Y-axis: frequency bins
    y_ticks = np.linspace(0, n_bins - 1, min(10, n_bins))
    y_tick_labels = [f'{k_bin_centers[int(i)]:.1f}' for i in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_ylabel('Radial Frequency |k| (Hz)', fontsize=12, fontweight='bold')
    
    # Colorbar with log scale
    cbar = plt.colorbar(im, ax=ax)
    scale_str = '[linear]'
    if np.all(error_matrix > 0):
        im.set_norm(LogNorm(vmin=error_matrix[error_matrix > 0].min(), 
                           vmax=error_matrix.max()))
        scale_str = '[log]'
    cbar.set_label(f'Relative Error (|FFT(error)|²/|FFT(gt)|²) {scale_str}', 
                   fontsize=11, rotation=270, labelpad=20)
    
    ax.set_title('Spectral Learning Efficiency\n(Relative Error by Layer and Frequency - Lower = Better)', 
                 fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    save_path = save_dir / 'spectral_learning_efficiency.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Spectral learning efficiency plot saved to {save_path}")


def generate_all_frequency_plots(
    freq_results: Dict[str, Dict],
    h_gt_spectrum: Dict,
    save_dir: Path,
    config: Dict
) -> None:
    """
    Generate all frequency analysis plots.
    
    Args:
        freq_results: Dictionary mapping layer_name -> results dict
        h_gt_spectrum: Ground truth spectrum dict
        save_dir: Directory to save plots
        config: Configuration dictionary
    """
    print("  Generating frequency plots...")
    
    # Main learned frequencies plot
    plot_learned_frequencies(
        freq_results=freq_results,
        h_gt_spectrum=h_gt_spectrum,
        save_dir=save_dir,
        config=config
    )
    
    # Spectral learning efficiency heatmap
    plot_spectral_learning_efficiency(
        freq_results=freq_results,
        h_gt_spectrum=h_gt_spectrum,
        save_dir=save_dir,
        config=config
    )
    
    print(f"  All frequency plots saved to {save_dir}")
