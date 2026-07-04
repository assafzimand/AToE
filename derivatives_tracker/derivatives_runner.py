"""Derivatives tracker orchestrator."""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict

from derivatives_tracker.derivatives_core import track_all_layers
from derivatives_tracker.derivatives_plotting import generate_all_derivative_plots
from probes.probe_runner import run_probes


def _compute_mean_norms(diff: np.ndarray) -> Dict[str, float]:
    if diff.size == 0:
        return {'l2': float('nan'), 'linf': float('nan')}
    l2 = np.linalg.norm(diff, axis=1)
    linf = np.max(np.abs(diff), axis=1)
    return {'l2': float(np.mean(l2)), 'linf': float(np.mean(linf))}


def _compute_ic_metrics(layer_results: Dict[str, Dict], data: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    mask_ic = data['mask']['IC'].detach().cpu().numpy().astype(bool)
    if not mask_ic.any():
        return {}
    h_gt = data['h_gt'].detach().cpu().numpy()
    metrics = {}
    for layer_name, results in layer_results.items():
        preds = results['h'][mask_ic]
        target = h_gt[mask_ic]
        metrics[layer_name] = _compute_mean_norms(preds - target)
    return metrics


def _pair_boundary_indices(x: np.ndarray, t: np.ndarray, mask: np.ndarray, boundary_value: float, atol: float = 1e-6):
    indices = np.where(mask & np.isclose(x, boundary_value, atol=atol))[0]
    time_map = {}
    for idx in indices:
        key = round(float(t[idx]), 8)
        time_map.setdefault(key, []).append(idx)
    return time_map


def _compute_bc_metrics(layer_results: Dict[str, Dict],
                        data: Dict[str, torch.Tensor],
                        config: Dict,
                        use_derivative: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Compute BC metrics based on problem type.
    
    - Schrödinger (periodic BC): Compare h_left to h_right (and h_x_left to h_x_right for derivatives)
    - Dirichlet BC problems (wave1d, burgers1d, burgers2d): Compare h_pred to h_gt at boundary points
    
    Args:
        layer_results: Dictionary of layer results with 'h' and 'h_x' keys
        data: Data dictionary with 'x', 't', 'h_gt', 'mask'
        config: Configuration dictionary
        use_derivative: If True, compute derivative metrics (only for periodic BC)
        
    Returns:
        Dictionary mapping layer_name to metrics dict with 'l2' and 'linf'
    """
    mask_bc = data['mask']['BC'].detach().cpu().numpy().astype(bool)
    if not mask_bc.any():
        return {}
    
    problem_name = config['problem']
    problem_cfg = config[problem_name]
    
    # Check if this is a periodic BC problem (only Schrödinger)
    is_periodic_bc = (problem_name == 'schrodinger')
    
    if is_periodic_bc:
        # PERIODIC BC: Compare h_left to h_right (Schrödinger)
        x = data['x'][:, 0].detach().cpu().numpy()
        t = data['t'][:, 0].detach().cpu().numpy()
        x_min, x_max = problem_cfg['spatial_domain'][0]
        
        left_map = _pair_boundary_indices(x, t, mask_bc, x_min)
        right_map = _pair_boundary_indices(x, t, mask_bc, x_max)
        
        # Build matched pairs using shared times
        pairs = []
        for time_key, left_indices in left_map.items():
            right_indices = right_map.get(time_key, [])
            while left_indices and right_indices:
                pairs.append((left_indices.pop(), right_indices.pop()))
        if not pairs:
            return {}
        
        left_idx = np.array([p[0] for p in pairs])
        right_idx = np.array([p[1] for p in pairs])
        
        metrics = {}
        value_key = 'h_x' if use_derivative else 'h'
        for layer_name, results in layer_results.items():
            values = results[value_key]
            left_vals = values[left_idx]
            right_vals = values[right_idx]
            metrics[layer_name] = _compute_mean_norms(left_vals - right_vals)
        return metrics
    else:
        # DIRICHLET BC: Compare h_pred to h_gt at boundary points
        # This applies to wave1d, burgers1d, burgers2d
        
        # For derivative metrics, Dirichlet BCs don't enforce derivative matching
        if use_derivative:
            return {}  # No derivative constraint for Dirichlet BCs
        
        h_gt = data['h_gt'].detach().cpu().numpy()
        h_gt_bc = h_gt[mask_bc]
        
        metrics = {}
        for layer_name, results in layer_results.items():
            h_pred = results['h']
            h_pred_bc = h_pred[mask_bc]
            error = h_pred_bc - h_gt_bc
            metrics[layer_name] = _compute_mean_norms(error)
        return metrics


def _extract_ic_profile(layer_results: Dict[str, Dict], data: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    mask_ic = data['mask']['IC'].detach().cpu().numpy().astype(bool)
    if not mask_ic.any():
        return {}
    
    x_data = data['x'].detach().cpu().numpy()
    spatial_dim = x_data.shape[1] if x_data.ndim > 1 else 1
    
    if spatial_dim == 1:
        # 1D spatial: sort by x for line plot
        x_vals = x_data[:, 0] if x_data.ndim > 1 else x_data.flatten()
        x_ic = x_vals[mask_ic]
        order = np.argsort(x_ic)
        x_sorted = x_ic[order]
        gt_real = 2.0 / np.cosh(x_sorted)
        gt_imag = np.zeros_like(x_sorted)
        profiles = {
            'x': x_sorted,
            'spatial_dim': 1,
            'gt_real': gt_real,
            'gt_imag': gt_imag,
            'layers': {}
        }
        for layer_name, results in layer_results.items():
            preds = results['h'][mask_ic][order]
            profiles['layers'][layer_name] = preds
    else:
        # 2D spatial: extract x0, x1 coordinates for surface plot
        x0_ic = x_data[mask_ic, 0]
        x1_ic = x_data[mask_ic, 1]
        profiles = {
            'x0': x0_ic,
            'x1': x1_ic,
            'spatial_dim': 2,
            'layers': {}
        }
        for layer_name, results in layer_results.items():
            preds = results['h'][mask_ic]
            profiles['layers'][layer_name] = preds
    
    return profiles


def run_derivatives_tracker(
    model: torch.nn.Module,
    train_data_path: str,
    eval_data_path: str,
    cfg: Dict,
    run_dir: Path,
    epoch_suffix: str = ""
) -> Dict:
    """
    Run complete derivatives tracking analysis.
    
    This function:
    1. First runs probe analysis to train linear probes for each layer
    2. Uses those probes to project each layer to 2D
    3. Computes derivatives and residual terms via autograd
    4. Generates comprehensive visualizations
    
    Args:
        model: Trained neural network model
        train_data_path: Path to training_data.pt
        eval_data_path: Path to eval_data.pt
        cfg: Configuration dictionary
        run_dir: Output directory for this run
        
    Returns:
        Dictionary with derivatives metrics summary
    """
    print("\n" + "=" * 60)
    print("Derivatives Tracker Analysis")
    print("=" * 60)
    
    device = torch.device('cuda' if cfg['cuda'] and 
                          torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = model.to(device)
    model.eval()
    
    # Step 1: Run probe analysis to get trained probes
    print("\n" + "=" * 60)
    print("Step 1: Training Linear Probes")
    print("=" * 60)
    
    probe_results = run_probes(
        model=model,
        train_data_path=train_data_path,
        eval_data_path=eval_data_path,
        cfg=cfg,
        run_dir=run_dir,
        epoch_suffix=epoch_suffix
    )
    
    # Extract trained probes from probe_results
    # We need to load probe_results from the actual probe_runner output
    # since run_probes returns a summary, not the full results
    probe_plots_dir = run_dir / "probe_plots"
    
    # Load data
    print("\n" + "=" * 60)
    print("Step 2: Loading Data for Derivatives")
    print("=" * 60)
    
    train_data = torch.load(train_data_path)
    eval_data = torch.load(eval_data_path)
    
    train_x = train_data['x'].to(device)
    train_t = train_data['t'].to(device)
    
    eval_x = eval_data['x'].to(device)
    eval_t = eval_data['t'].to(device)
    
    N_train = train_x.shape[0]
    N_eval = eval_x.shape[0]
    print(f"  Train samples: {N_train}")
    print(f"  Eval samples: {N_eval}")
    
    # Step 3: Re-train probes to get probe objects (needed for derivatives)
    # We need the actual probe objects, so we'll extract embeddings and train again
    print("\n" + "=" * 60)
    print("Step 3: Extracting Probes for Derivatives")
    print("=" * 60)
    
    all_layers = model.get_layer_names()
    hidden_layers = all_layers[:-1]
    print(f"  Analyzing layers: {hidden_layers}")
    
    # Register hooks to get embeddings (without gradients for probe training)
    handles = model.register_ncc_hooks(hidden_layers, keep_gradients=False)
    
    with torch.no_grad():
        train_inputs = torch.cat([train_x, train_t], dim=1)
        _ = model(train_inputs)
    
    train_embeddings = model.activations.copy()
    model.remove_hooks()
    
    # Train probes to get probe objects
    from probes.probe_core import train_linear_probe
    
    train_targets = train_data['h_gt'].to(device)
    probes_dict = {}
    
    print("  Training probes...")
    for layer_name in hidden_layers:
        embeddings = train_embeddings[layer_name]
        probe = train_linear_probe(embeddings, train_targets)
        probes_dict[layer_name] = probe
        print(f"    {layer_name}: probe trained ({embeddings.shape[1]} -> 2)")
    
    # Step 4: Track derivatives for train and eval datasets
    print("\n" + "=" * 60)
    print("Step 4: Computing Derivatives and Residuals")
    print("=" * 60)
    
    print("\nProcessing training data...")
    train_derivatives = track_all_layers(
        model=model,
        probes_dict=probes_dict,
        x=train_x,
        t=train_t,
        device=device,
        config=cfg
    )
    
    print("\nProcessing evaluation data...")
    eval_derivatives = track_all_layers(
        model=model,
        probes_dict=probes_dict,
        x=eval_x,
        t=eval_t,
        device=device,
        config=cfg
    )
    
    ic_train_metrics = _compute_ic_metrics(train_derivatives, train_data)
    ic_eval_metrics = _compute_ic_metrics(eval_derivatives, eval_data)
    
    # Pass callable ground-truth functions for plotting (problem-specific)
    ic_profile_eval = _extract_ic_profile(eval_derivatives, eval_data)
    if ic_profile_eval:
        problem_name = cfg.get('problem', 'schrodinger')
        if problem_name == 'schrodinger':
            ic_profile_eval['gt_real'] = lambda arr: 2.0 / np.cosh(arr)
            ic_profile_eval['gt_imag'] = lambda arr: np.zeros_like(arr)
            ic_profile_eval['gt_label_0'] = 'Ground Truth 2·sech(x)'
            ic_profile_eval['gt_label_1'] = 'Ground Truth 0'
        elif problem_name == 'wave1d':
            ic_profile_eval['gt_real'] = lambda arr: np.sin(arr)
            ic_profile_eval['gt_label_0'] = 'Ground Truth sin(x)'
        elif problem_name == 'burgers1d':
            ic_profile_eval['gt_real'] = lambda arr: -np.sin(np.pi * arr)
            ic_profile_eval['gt_label_0'] = 'Ground Truth -sin(πx)'
        elif problem_name == 'burgers2d':
            # For 2D spatial, gt_real takes (x0, x1) grid arrays
            ic_profile_eval['gt_real'] = lambda x0_arr, x1_arr: 1.0 / (1.0 + np.exp((x0_arr + x1_arr) / 0.2))
            ic_profile_eval['gt_label_0'] = 'Ground Truth h(x0,x1,0)'
    bc_value_train = _compute_bc_metrics(train_derivatives, train_data, cfg, use_derivative=False)
    bc_value_eval = _compute_bc_metrics(eval_derivatives, eval_data, cfg, use_derivative=False)
    bc_deriv_train = _compute_bc_metrics(train_derivatives, train_data, cfg, use_derivative=True)
    bc_deriv_eval = _compute_bc_metrics(eval_derivatives, eval_data, cfg, use_derivative=True)
    
    # Step 5: Generate visualizations (use eval data)
    print("\n" + "=" * 60)
    print("Step 5: Generating Visualizations")
    print("=" * 60)
    
    if epoch_suffix:
        # Periodic derivatives: save inside derivatives_plots/derivatives_plots_epoch_X/
        derivatives_plots_dir = run_dir / "derivatives_plots" / f"derivatives_plots{epoch_suffix}"
    else:
        # Final derivatives: save directly in derivatives_plots/
        derivatives_plots_dir = run_dir / "derivatives_plots"
    
    derivatives_plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute ground truth derivatives for comparison
    print("\nComputing ground truth derivatives...")
    from derivatives_tracker.derivatives_core import compute_ground_truth_derivatives
    
    ground_truth_derivatives = compute_ground_truth_derivatives(
        x=eval_x,
        t=eval_t,
        config=cfg
    )
    print("  Ground truth derivatives computed")
    
    # Convert to numpy for plotting (detach first to remove gradients)
    eval_x_np = eval_x.detach().cpu().numpy()
    eval_t_np = eval_t.detach().cpu().numpy()
    
    generate_all_derivative_plots(
        train_results=train_derivatives,
        eval_results=eval_derivatives,
        ic_metrics={'train': ic_train_metrics, 'eval': ic_eval_metrics},
        bc_value_metrics={'train': bc_value_train, 'eval': bc_value_eval},
        bc_derivative_metrics={'train': bc_deriv_train, 'eval': bc_deriv_eval},
        ic_profile=ic_profile_eval,
        x=eval_x_np,
        t=eval_t_np,
        ground_truth_derivatives=ground_truth_derivatives,
        save_dir=derivatives_plots_dir,
        config=cfg
    )
    
    # Step 6: Build and save metrics summary
    print("\n" + "=" * 60)
    print("Step 6: Saving Metrics")
    print("=" * 60)
    
    metrics_summary = {
        'layers_analyzed': hidden_layers,
        'num_layers': len(hidden_layers),
        'train': {},
        'eval': {},
        'layer_norms': {},
        'ic': {'train': ic_train_metrics, 'eval': ic_eval_metrics},
        'bc_value': {'train': bc_value_train, 'eval': bc_value_eval},
        'bc_derivative': {'train': bc_deriv_train, 'eval': bc_deriv_eval}
    }
    
    # Collect norms for each layer
    for layer_name in sorted(train_derivatives.keys()):
        metrics_summary['train'][layer_name] = train_derivatives[layer_name]['norms']
        metrics_summary['eval'][layer_name] = eval_derivatives[layer_name]['norms']
        metrics_summary['layer_norms'][layer_name] = {
            'train_residual_l2': train_derivatives[layer_name]['norms']['residual_norm'],
            'eval_residual_l2': eval_derivatives[layer_name]['norms']['residual_norm'],
            'train_residual_linf': train_derivatives[layer_name]['norms']['residual_inf_norm'],
            'eval_residual_linf': eval_derivatives[layer_name]['norms']['residual_inf_norm']
        }
    
    # Add final layer residual for easy comparison
    final_layer = hidden_layers[-1]
    metrics_summary['final_layer_train_residual'] = train_derivatives[final_layer]['norms']['residual_norm']
    metrics_summary['final_layer_eval_residual'] = eval_derivatives[final_layer]['norms']['residual_norm']
    metrics_summary['final_layer_train_residual_inf'] = train_derivatives[final_layer]['norms']['residual_inf_norm']
    metrics_summary['final_layer_eval_residual_inf'] = eval_derivatives[final_layer]['norms']['residual_inf_norm']
    metrics_summary['final_layer_train_ic'] = ic_train_metrics.get(final_layer, {})
    metrics_summary['final_layer_eval_ic'] = ic_eval_metrics.get(final_layer, {})
    metrics_summary['final_layer_train_bc_value'] = bc_value_train.get(final_layer, {})
    metrics_summary['final_layer_eval_bc_value'] = bc_value_eval.get(final_layer, {})
    metrics_summary['final_layer_train_bc_derivative'] = bc_deriv_train.get(final_layer, {})
    metrics_summary['final_layer_eval_bc_derivative'] = bc_deriv_eval.get(final_layer, {})
    # Store per-layer IC/BC metrics for history plotting
    metrics_summary['ic'] = {'train': ic_train_metrics, 'eval': ic_eval_metrics}
    metrics_summary['bc_value'] = {'train': bc_value_train, 'eval': bc_value_eval}
    metrics_summary['bc_derivative'] = {'train': bc_deriv_train, 'eval': bc_deriv_eval}
    
    # Save metrics
    metrics_path = derivatives_plots_dir / "derivatives_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  Derivatives metrics saved to {metrics_path}")
    
    # Step 7: Summary
    print("\n" + "=" * 60)
    print("Derivatives Tracker Analysis Complete")
    print("=" * 60)
    print(f"Results saved to: {derivatives_plots_dir}")
    print("  Generated plots:")
    print("    - residual_evolution_summary.png (most important!)")
    print("    - ic_summary.png")
    print("    - bc_value_summary.png")
    print("    - bc_derivative_summary.png")
    print("    - term_magnitudes.png")
    print("    - derivative_heatmaps_*.png (for selected layers)")
    print("    - residual_heatmaps_*.png (for selected layers)")
    print(f"Metrics: {metrics_path}")
    print("\nFinal Layer Residuals:")
    print(f"  Train: {metrics_summary['final_layer_train_residual']:.6e}")
    print(f"  Eval:  {metrics_summary['final_layer_eval_residual']:.6e}")
    print(f"  Train L_inf: {metrics_summary['final_layer_train_residual_inf']:.6e}")
    print(f"  Eval  L_inf: {metrics_summary['final_layer_eval_residual_inf']:.6e}")
    ic_train_final = metrics_summary['final_layer_train_ic']
    ic_eval_final = metrics_summary['final_layer_eval_ic']
    if ic_train_final:
        print(f"  IC Train L2/L_inf: {ic_train_final.get('l2', float('nan')):.6e} / "
              f"{ic_train_final.get('linf', float('nan')):.6e}")
    if ic_eval_final:
        print(f"  IC Eval  L2/L_inf: {ic_eval_final.get('l2', float('nan')):.6e} / "
              f"{ic_eval_final.get('linf', float('nan')):.6e}")
    bc_val_train_final = metrics_summary['final_layer_train_bc_value']
    bc_val_eval_final = metrics_summary['final_layer_eval_bc_value']
    if bc_val_train_final:
        print(f"  BC Value Train L2/L_inf: {bc_val_train_final.get('l2', float('nan')):.6e} / "
              f"{bc_val_train_final.get('linf', float('nan')):.6e}")
    if bc_val_eval_final:
        print(f"  BC Value Eval  L2/L_inf: {bc_val_eval_final.get('l2', float('nan')):.6e} / "
              f"{bc_val_eval_final.get('linf', float('nan')):.6e}")
    bc_der_train_final = metrics_summary['final_layer_train_bc_derivative']
    bc_der_eval_final = metrics_summary['final_layer_eval_bc_derivative']
    if bc_der_train_final:
        print(f"  BC Deriv Train L2/L_inf: {bc_der_train_final.get('l2', float('nan')):.6e} / "
              f"{bc_der_train_final.get('linf', float('nan')):.6e}")
    if bc_der_eval_final:
        print(f"  BC Deriv Eval  L2/L_inf: {bc_der_eval_final.get('l2', float('nan')):.6e} / "
              f"{bc_der_eval_final.get('linf', float('nan')):.6e}")
    print("=" * 60)
    
    return metrics_summary

