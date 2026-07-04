"""Core frequency analysis via FFT on probe outputs."""

import torch
import numpy as np
from typing import Dict, Tuple, List


def generate_frequency_grid(config: Dict) -> Tuple[torch.Tensor, Tuple[int, ...], int]:
    """
    Generate regular grid for FFT evaluation.
    
    Args:
        config: Configuration dict with problem domain info
        
    Returns:
        x_grid: (N_total, d_in) grid points where N_total = n_grid^d_in
        grid_shape: Tuple of grid dimensions (n_grid, n_grid, ...)
        n_dims: Number of input dimensions (spatial_dim + 1 for time)
    """
    problem_name = config['problem']
    problem_cfg = config[problem_name]
    n_grid = config.get('n_freq_grid', 32)
    
    # Build linspace for each dimension
    grids = []
    spatial_domain = problem_cfg['spatial_domain']
    temporal_domain = problem_cfg['temporal_domain']
    
    # Spatial dimensions
    for dom in spatial_domain:
        grids.append(torch.linspace(dom[0], dom[1], n_grid, dtype=torch.float64))
    
    # Time dimension
    grids.append(torch.linspace(temporal_domain[0], temporal_domain[1], n_grid, dtype=torch.float64))
    
    n_dims = len(grids)
    grid_shape = tuple([n_grid] * n_dims)
    
    # Create meshgrid and flatten
    mesh = torch.meshgrid(*grids, indexing='ij')
    
    # Stack: spatial dims first, then time
    # For burgers2d: [x0, x1, t] -> shape (N, 3)
    # For schrodinger/wave1d/burgers1d: [x, t] -> shape (N, 2)
    x_grid = torch.stack([m.flatten() for m in mesh], dim=1)
    
    return x_grid, grid_shape, n_dims


def compute_frequency_spectrum(
    values: np.ndarray,
    grid_shape: Tuple[int, ...],
    sample_spacings: List[float] = None
) -> Dict[str, np.ndarray]:
    """
    Compute N-D FFT and power spectrum.
    
    Args:
        values: (N,) or (N, d_o) probe outputs on flattened grid
        grid_shape: Shape to reshape to (n_grid, n_grid, ...)
        sample_spacings: List of sample spacings per dimension (for physical frequencies)
        
    Returns:
        Dict with:
            'power': Power spectrum |FFT|² with shape grid_shape (+ d_o if multi-output)
            'freqs': List of frequency arrays for each dimension (physical frequencies)
            'output_dim': Number of output dimensions
    """
    n_dims = len(grid_shape)
    
    # Default sample spacing = 1 (normalized frequencies)
    if sample_spacings is None:
        sample_spacings = [1.0] * n_dims
    
    # Handle multi-output
    if values.ndim == 1:
        values_grid = values.reshape(grid_shape)
        output_dim = 1
    else:
        # (N, d_o) -> (*grid_shape, d_o)
        output_dim = values.shape[1]
        values_grid = values.reshape(*grid_shape, output_dim)
    
    # Compute N-D FFT along spatial/temporal axes
    fft_axes = tuple(range(n_dims))  # (0, 1, ..., n_dims-1)
    fft_result = np.fft.fftn(values_grid, axes=fft_axes)
    fft_shifted = np.fft.fftshift(fft_result, axes=fft_axes)
    
    # Power spectrum
    power = np.abs(fft_shifted) ** 2
    
    # Frequency coordinates with physical units based on sample spacing
    # fftfreq returns normalized freq, divide by spacing to get physical freq
    freqs = []
    for i, n in enumerate(grid_shape):
        d = sample_spacings[i]
        freq = np.fft.fftshift(np.fft.fftfreq(n, d=d))
        freqs.append(freq)
    
    return {
        'power': power,
        'freqs': freqs,
        'output_dim': output_dim
    }


def compute_radial_spectrum(
    power: np.ndarray,
    freqs: List[np.ndarray],
    n_bins: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute radial (isotropic) power spectrum.
    
    Aggregates power by frequency magnitude, useful for scale analysis.
    
    Args:
        power: N-D power spectrum
        freqs: List of frequency arrays per dimension
        n_bins: Number of radial bins
        
    Returns:
        k_radial: Radial frequency values (bin centers)
        power_radial: Summed power in each radial bin
    """
    # Get spatial dimensions (exclude output_dim if present)
    n_spatial = len(freqs)
    
    # Create radial frequency grid
    mesh_freqs = np.meshgrid(*freqs, indexing='ij')
    k_magnitude = np.sqrt(sum(f**2 for f in mesh_freqs))
    
    # Bin edges
    k_max = k_magnitude.max()
    if k_max == 0:
        k_max = 1.0  # Avoid division by zero
    k_edges = np.linspace(0, k_max, n_bins + 1)
    k_centers = (k_edges[:-1] + k_edges[1:]) / 2
    
    # Handle multi-output: average over output dimension first
    if power.ndim > n_spatial:
        power_avg = power.mean(axis=-1)
    else:
        power_avg = power
    
    # Sum power in each bin
    power_radial = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (k_magnitude >= k_edges[i]) & (k_magnitude < k_edges[i+1])
        if mask.sum() > 0:
            power_radial[i] = power_avg[mask].sum()
    
    return k_centers, power_radial


def compute_marginal_spectra(
    power: np.ndarray,
    freqs: List[np.ndarray],
    spatial_dim: int = 1
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute 1D marginal spectra by averaging over other dimensions.
    
    Args:
        power: N-D power spectrum
        freqs: List of frequency arrays per dimension
        spatial_dim: Number of spatial dimensions (1 or 2)
        
    Returns:
        Dict mapping dim_name -> (freq_1d, power_1d)
    """
    n_spatial = len(freqs)
    
    # Name dimensions based on spatial_dim
    if spatial_dim == 2:
        dim_names = ['x0', 'x1', 't'][:n_spatial]
    else:
        dim_names = ['x', 't'][:n_spatial]
    
    # Handle multi-output
    if power.ndim > n_spatial:
        power_avg = power.mean(axis=-1)
    else:
        power_avg = power
    
    marginals = {}
    for i, name in enumerate(dim_names):
        # Average over all other dimensions
        axes_to_avg = tuple(j for j in range(n_spatial) if j != i)
        if axes_to_avg:
            power_1d = power_avg.mean(axis=axes_to_avg)
        else:
            power_1d = power_avg
        marginals[name] = (freqs[i], power_1d)
    
    return marginals


def analyze_all_layers_frequency(
    model: torch.nn.Module,
    probes_dict: Dict[str, torch.nn.Linear],
    freq_data: Dict,
    device: torch.device,
    config: Dict
) -> Tuple[Dict[str, Dict], List[float]]:
    """
    Analyze frequency content for all layers.
    
    For each layer, computes:
    - Cumulative spectrum: |FFT(ĥ_i)|²
    - Added spectrum: |FFT(ĥ_i - ĥ_{i-1})|²
    - Leftover spectrum: |FFT(h_gt - ĥ_i)|²
    
    Args:
        model: Trained neural network
        probes_dict: Dict mapping layer_name -> trained probe
        freq_data: Pre-loaded frequency grid data from frequency_grid.pt
        device: Device for computation
        config: Configuration dict
        
    Returns:
        Tuple of:
            - Dict mapping layer_name -> {
                'h_pred': (N, d_o) probe predictions on grid
                'cumulative': spectrum dict
                'added': spectrum dict (vs previous layer)
                'leftover': spectrum dict (h_gt - h_pred)
                'radial': (k, power) tuple for cumulative
                'radial_added': (k, power) tuple for added
                'radial_leftover': (k, power) tuple for leftover
                'marginal': dict of 1D spectra
              }
            - sample_spacings: List of sample spacings per dimension
    """
    model.eval()
    
    # Extract data from freq_data
    x_grid = freq_data['x_grid'].to(device)
    h_gt_grid = freq_data['h_gt_grid'].cpu().numpy()
    grid_shape = tuple(freq_data['grid_shape'])
    
    # Get problem info
    problem_cfg = config[config['problem']]
    spatial_dim = problem_cfg['spatial_dim']
    
    # Compute sample spacings for physical frequency units
    # sample_spacing[i] = domain_length[i] / n_grid
    n_grid = grid_shape[0]  # Assuming uniform grid
    sample_spacings = []
    
    spatial_domain = problem_cfg['spatial_domain']
    temporal_domain = problem_cfg['temporal_domain']
    
    # Spatial dimensions
    for dom in spatial_domain:
        domain_length = dom[1] - dom[0]
        sample_spacings.append(domain_length / n_grid)
    
    # Time dimension
    time_length = temporal_domain[1] - temporal_domain[0]
    sample_spacings.append(time_length / n_grid)
    
    results = {}
    prev_h_pred = None
    
    layer_names = sorted(probes_dict.keys())
    
    for layer_name in layer_names:
        print(f"  Analyzing frequency content for {layer_name}...")
        probe = probes_dict[layer_name]
        
        # Get layer activations
        handles = model.register_ncc_hooks([layer_name], keep_gradients=False)
        
        with torch.no_grad():
            # x_grid is already in format [x0, x1, t] or [x, t] - pass directly
            _ = model(x_grid)
        
        embeddings = model.activations[layer_name]
        model.remove_hooks()
        
        # Get probe predictions
        with torch.no_grad():
            h_pred = probe(embeddings).cpu().numpy()
        
        # Flatten h_pred if single output
        if h_pred.shape[1] == 1:
            h_pred_flat = h_pred.flatten()
            h_gt_flat = h_gt_grid.flatten()
        else:
            h_pred_flat = h_pred
            h_gt_flat = h_gt_grid
        
        # Compute spectra with physical frequency units
        cumulative = compute_frequency_spectrum(h_pred_flat, grid_shape, sample_spacings)
        leftover = compute_frequency_spectrum(h_gt_flat - h_pred_flat, grid_shape, sample_spacings)
        
        if prev_h_pred is not None:
            if prev_h_pred.shape[1] == 1 if prev_h_pred.ndim > 1 else True:
                prev_flat = prev_h_pred.flatten() if prev_h_pred.ndim > 1 else prev_h_pred
            else:
                prev_flat = prev_h_pred
            added = compute_frequency_spectrum(h_pred_flat - prev_flat, grid_shape, sample_spacings)
        else:
            added = cumulative  # First layer: everything is "added"
        
        # Radial spectra
        radial = compute_radial_spectrum(cumulative['power'], cumulative['freqs'])
        radial_added = compute_radial_spectrum(added['power'], added['freqs'])
        radial_leftover = compute_radial_spectrum(leftover['power'], leftover['freqs'])
        
        # Marginal spectra
        marginal = compute_marginal_spectra(cumulative['power'], cumulative['freqs'], spatial_dim)
        
        results[layer_name] = {
            'h_pred': h_pred,
            'cumulative': cumulative,
            'added': added,
            'leftover': leftover,
            'radial': radial,
            'radial_added': radial_added,
            'radial_leftover': radial_leftover,
            'marginal': marginal
        }
        
        prev_h_pred = h_pred.copy()
        
        # Print summary
        total_power = cumulative['power'].sum()
        leftover_power = leftover['power'].sum()
        print(f"    Total power: {total_power:.2e}, Leftover power: {leftover_power:.2e}")
    
    return results, sample_spacings


def compute_binned_frequency_errors(
    power_pred: np.ndarray,
    power_gt: np.ndarray,
    freqs: List[np.ndarray],
    spatial_dim: int,
    n_bins: int = 20,
    is_leftover: bool = False
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute mean relative error binned by frequency for each dimension.
    
    For each dimension, bins frequencies and computes:
    mean(|power_pred - power_gt| / power_gt) for each bin
    
    Args:
        power_pred: Predicted power spectrum (same shape as power_gt)
        power_gt: Ground truth power spectrum
        freqs: List of frequency arrays per dimension
        spatial_dim: Number of spatial dimensions (1 or 2)
        n_bins: Number of bins per dimension
        
    Returns:
        Dict mapping dim_name -> (bin_centers, mean_relative_error)
        Plus 'radial' for |k| binning
    """
    n_dims = len(freqs)
    
    # Handle multi-output: average over output dimension first
    if power_pred.ndim > n_dims:
        power_pred_avg = power_pred.mean(axis=-1)
        power_gt_avg = power_gt.mean(axis=-1)
    else:
        power_pred_avg = power_pred
        power_gt_avg = power_gt
    
    # Compute relative error: |pred - gt| / gt
    # For leftover: power_pred is already the leftover power, so we want leftover/gt
    # For others: we want |pred - gt| / gt
    power_gt_safe = np.where(power_gt_avg > 1e-10, power_gt_avg, 1e-10)
    if is_leftover:
        # Leftover: show leftover power normalized by ground truth
        relative_error = power_pred_avg / power_gt_safe
    else:
        # Standard relative error
        relative_error = np.abs(power_pred_avg - power_gt_avg) / power_gt_safe
    
    # Name dimensions
    if spatial_dim == 2:
        dim_names = ['x0', 'x1', 't'][:n_dims]
    else:
        dim_names = ['x', 't'][:n_dims]
    
    results = {}
    
    # Bin by each dimension
    for i, dim_name in enumerate(dim_names):
        freq_1d = freqs[i]
        freq_min, freq_max = freq_1d.min(), freq_1d.max()
        
        # Create bins
        bin_edges = np.linspace(freq_min, freq_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Average over all other dimensions
        axes_to_avg = tuple(j for j in range(n_dims) if j != i)
        if axes_to_avg:
            error_1d = relative_error.mean(axis=axes_to_avg)
        else:
            error_1d = relative_error
        
        # Bin the error
        mean_error_binned = np.zeros(n_bins)
        for bin_idx in range(n_bins):
            mask = (freq_1d >= bin_edges[bin_idx]) & (freq_1d < bin_edges[bin_idx + 1])
            if bin_idx == n_bins - 1:  # Include right edge for last bin
                mask = (freq_1d >= bin_edges[bin_idx]) & (freq_1d <= bin_edges[bin_idx + 1])
            if mask.sum() > 0:
                mean_error_binned[bin_idx] = error_1d[mask].mean()
        
        results[dim_name] = (bin_centers, mean_error_binned)
    
    # Radial binning: |k| = sqrt(sum(k_i^2))
    mesh_freqs = np.meshgrid(*freqs, indexing='ij')
    k_magnitude = np.sqrt(sum(f**2 for f in mesh_freqs))
    
    k_max = k_magnitude.max()
    if k_max == 0:
        k_max = 1.0
    k_bin_edges = np.linspace(0, k_max, n_bins + 1)
    k_bin_centers = (k_bin_edges[:-1] + k_bin_edges[1:]) / 2
    
    mean_error_radial = np.zeros(n_bins)
    for bin_idx in range(n_bins):
        mask = (k_magnitude >= k_bin_edges[bin_idx]) & (k_magnitude < k_bin_edges[bin_idx + 1])
        if bin_idx == n_bins - 1:
            mask = (k_magnitude >= k_bin_edges[bin_idx]) & (k_magnitude <= k_bin_edges[bin_idx + 1])
        if mask.sum() > 0:
            mean_error_radial[bin_idx] = relative_error[mask].mean()
    
    results['radial'] = (k_bin_centers, mean_error_radial)
    
    return results

