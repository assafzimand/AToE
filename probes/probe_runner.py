"""Linear probe analysis orchestrator."""

import torch
import json
from pathlib import Path
from typing import Dict

from probes.probe_core import probe_all_layers
from probes.probe_plotting import generate_all_probe_plots


def run_probes(
    model: torch.nn.Module,
    train_data_path: str,
    eval_data_path: str,
    cfg: Dict,
    run_dir: Path,
    epoch_suffix: str = ""
) -> Dict:
    """
    Run complete linear probe analysis on a trained model.
    
    Args:
        model: Trained neural network model
        train_data_path: Path to training_data.pt
        eval_data_path: Path to eval_data.pt
        cfg: Configuration dictionary
        run_dir: Output directory for this run
        epoch_suffix: Optional suffix for epoch-specific analysis (e.g., "_epoch_2")
        
    Returns:
        Dictionary with probe metrics summary
    """
    print("\n" + "=" * 60)
    print("Linear Probe Analysis")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if cfg['cuda'] and
                          torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True  # Optimize for GPU
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Load datasets
    print(f"\nLoading datasets...")
    train_data = torch.load(train_data_path)
    eval_data = torch.load(eval_data_path)
    
    # Move data to device
    train_x = train_data['x'].to(device)
    train_t = train_data['t'].to(device)
    train_targets = train_data['h_gt'].to(device)
    
    eval_x = eval_data['x'].to(device)
    eval_t = eval_data['t'].to(device)
    eval_targets = eval_data['h_gt'].to(device)
    
    N_train = train_x.shape[0]
    N_eval = eval_x.shape[0]
    print(f"  Train samples: {N_train}")
    print(f"  Eval samples: {N_eval}")
    
    # Get all layer names (exclude output layer)
    all_layers = model.get_layer_names()
    hidden_layers = all_layers[:-1]  # Exclude output layer
    print(f"  Analyzing layers: {hidden_layers}")
    
    # Register hooks to capture activations for training data
    print("\nCollecting training embeddings...")
    handles = model.register_ncc_hooks(hidden_layers)
    
    with torch.no_grad():
        train_inputs = torch.cat([train_x, train_t], dim=1)
        _ = model(train_inputs)
    
    train_embeddings = model.activations.copy()
    model.remove_hooks()
    
    print(f"  Collected embeddings from {len(train_embeddings)} layers")
    for layer_name, activations in train_embeddings.items():
        print(f"    {layer_name}: {activations.shape}")
    
    # Register hooks to capture activations for eval data
    print("\nCollecting evaluation embeddings...")
    _ = model.register_ncc_hooks(hidden_layers)  # noqa: F841
    
    with torch.no_grad():
        eval_inputs = torch.cat([eval_x, eval_t], dim=1)
        _ = model(eval_inputs)
    
    eval_embeddings = model.activations.copy()
    model.remove_hooks()
    
    print(f"  Collected embeddings from {len(eval_embeddings)} layers")
    
    # Train probes for all layers
    print("\n" + "=" * 60)
    probe_results = probe_all_layers(
        embeddings_dict=train_embeddings,
        train_targets=train_targets,
        eval_embeddings_dict=eval_embeddings,
        eval_targets=eval_targets,
        device=device
    )
    print("=" * 60)
    
    # Create output directory
    if epoch_suffix:
        # Periodic probes: save inside probe_plots/probe_plots_epoch_X/
        probe_plots_dir = run_dir / "probe_plots" / f"probe_plots{epoch_suffix}"
    else:
        # Final probes: save directly in probe_plots/
        probe_plots_dir = run_dir / "probe_plots"
    
    probe_plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots (include coordinates and targets for error heatmaps)
    generate_all_probe_plots(
        probe_results, 
        probe_plots_dir,
        eval_x=eval_x.cpu().numpy(),
        eval_t=eval_t.cpu().numpy(),
        eval_targets=eval_targets.cpu().numpy()
    )
    
    # Build metrics summary for JSON export
    metrics_summary = {
        'layers_analyzed': hidden_layers,
        'num_layers': len(hidden_layers),
        'train': {
            'rel_l2': [],
            'inf_norm': []
        },
        'eval': {
            'rel_l2': [],
            'inf_norm': []
        },
        'layer_dimensions': {}
    }
    
    for layer_name in sorted(probe_results.keys()):
        metrics_summary['train']['rel_l2'].append(
            probe_results[layer_name]['train_metrics']['rel_l2']
        )
        metrics_summary['train']['inf_norm'].append(
            probe_results[layer_name]['train_metrics']['inf_norm']
        )
        metrics_summary['eval']['rel_l2'].append(
            probe_results[layer_name]['eval_metrics']['rel_l2']
        )
        metrics_summary['eval']['inf_norm'].append(
            probe_results[layer_name]['eval_metrics']['inf_norm']
        )
        metrics_summary['layer_dimensions'][layer_name] = probe_results[layer_name]['hidden_dim']
    
    # Save metrics to JSON
    metrics_path = probe_plots_dir / "probe_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\n  Probe metrics saved to {metrics_path}")
    
    print("\n" + "=" * 60)
    print("Linear Probe Analysis Complete")
    print("=" * 60)
    print(f"Results saved to: {probe_plots_dir}")
    print("  - probe_metrics.png")
    print("  - probe_dimensions.png")
    print(f"Metrics: {metrics_path}")
    print("=" * 60)
    
    return metrics_summary

