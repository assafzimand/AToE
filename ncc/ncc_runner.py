"""NCC runner for analyzing trained models."""

import torch
import json
from pathlib import Path
from typing import Dict

from ncc.ncc_core import compute_all_ncc_metrics
from ncc.ncc_plotting import generate_all_ncc_plots


def run_ncc(
    model: torch.nn.Module,
    eval_data_path: str,
    cfg: Dict,
    run_dir: Path = None,
    epoch_suffix: str = "",
    suppress_plots: bool = False
) -> Dict:
    """
    Run complete NCC analysis on a trained model.

    Args:
        model: Trained neural network model
        eval_data_path: Path to NCC data (.pt file) - stratified dataset
        cfg: Configuration dictionary
        run_dir: Output directory for this run (None for suppress_plots mode)
        epoch_suffix: Optional suffix for epoch-specific analysis (e.g., "_epoch_1000")
        suppress_plots: If True, skip all plot generation and only return metrics

    Returns:
        Dictionary with NCC metrics summary
    """
    print("\n" + "=" * 60)
    print("NCC Analysis")
    print("=" * 60)

    # Setup device
    device = torch.device('cuda' if cfg['cuda'] and
                          torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load NCC data (stratified dataset)
    print(f"\nLoading NCC data from: {eval_data_path}")
    eval_data = torch.load(eval_data_path)

    # Move data to device
    x = eval_data['x'].to(device)
    t = eval_data['t'].to(device)
    h_gt = eval_data['h_gt'].to(device)

    N = x.shape[0]
    print(f"  NCC samples: {N}")

    # Get bins parameter
    bins = cfg['bins']
    print(f"  Bins per dimension: {bins}")

    # Get all layer names (exclude output layer for NCC analysis)
    all_layers = model.get_layer_names()
    hidden_layers = all_layers[:-1]  # Exclude output layer
    print(f"  Analyzing layers: {hidden_layers}")

    # Register hooks to capture activations
    print("\nRegistering hooks...")
    handles = model.register_ncc_hooks(hidden_layers)
    print(f"  Hooks registered on {len(handles)} layers")

    # Forward pass to collect activations
    print("\nRunning forward pass to collect activations...")
    with torch.no_grad():
        inputs = torch.cat([x, t], dim=1)
        h_pred = model(inputs)

    # Get activations from model
    embeddings_dict = model.activations.copy()
    print(f"  Collected activations from {len(embeddings_dict)} layers")

    # Verify activations
    for layer_name, activations in embeddings_dict.items():
        print(f"    {layer_name}: {activations.shape}")

    # Clean up hooks
    model.remove_hooks()

    # Compute NCC metrics
    print("\nComputing NCC metrics...")
    ncc_results = compute_all_ncc_metrics(
        embeddings_dict=embeddings_dict,
        ground_truth_outputs=h_gt,
        bins=bins,
        device=device
    )

    num_classes = ncc_results['bin_info']['num_classes']
    print(f"  Created {num_classes} classes from regression outputs")

    # Print metrics summary
    print("\n  Per-layer NCC accuracy:")
    for layer_name, metrics in ncc_results['layer_metrics'].items():
        acc = metrics['accuracy']
        print(f"    {layer_name}: {acc:.4f}")

    # Generate plots (unless suppressed for multi-eval)
    if not suppress_plots:
        # Generate plots
        if epoch_suffix:
            # Periodic NCC: save inside ncc_plots/ncc_plots_epoch_X/
            ncc_plots_dir = run_dir / "ncc_plots" / f"ncc_plots{epoch_suffix}"
        else:
            # Final NCC: save directly in ncc_plots/
            ncc_plots_dir = run_dir / "ncc_plots"
        
        ncc_plots_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating NCC plots...")
        generate_all_ncc_plots(ncc_results, ncc_plots_dir)

        # Generate problem-specific NCC classification diagnostic
        print("\nGenerating NCC classification diagnostic...")
        try:
            from utils.problem_specific import get_visualization_module
            viz_module = get_visualization_module(cfg['problem'])
            _, _, _, visualize_ncc_classification, visualize_ncc_classification_input_space = viz_module[:5]
            
            # Check if heatmap functions are available (new functions)
            # Each function is checked individually - None means not available
            try:
                visualize_ncc_classification_heatmap = viz_module[5]
                visualize_ncc_classification_input_space_heatmap = viz_module[6]
                visualize_ncc_classification_accuracy_changes = viz_module[7]
                visualize_ncc_classification_input_space_accuracy_changes = viz_module[8]
            except (IndexError, AttributeError):
                visualize_ncc_classification_heatmap = None
                visualize_ncc_classification_input_space_heatmap = None
                visualize_ncc_classification_accuracy_changes = None
                visualize_ncc_classification_input_space_accuracy_changes = None
            
            # Extract class labels from results (already a tensor on device)
            class_labels = ncc_results['class_labels']
            
            predictions_dict = {
                ln: torch.tensor(ncc_results['layer_metrics'][ln]['predictions'], device=device)
                for ln in hidden_layers
            }
            
            # Output space visualization (u, v) - scatter plot
            viz_path = ncc_plots_dir / "ncc_classification_diagnostic.png"
            visualize_ncc_classification(h_gt, class_labels, predictions_dict, bins, viz_path)
            print(f"  Classification diagnostic saved to {viz_path}")
            
            # Input space visualization (x, t) - scatter plot
            viz_path_input = ncc_plots_dir / "ncc_classification_input_space.png"
            visualize_ncc_classification_input_space(x, t, class_labels, predictions_dict, viz_path_input)
            print(f"  Input space classification diagnostic saved to {viz_path_input}")
            
            # Generate heatmap versions if available (each function checked individually)
            if visualize_ncc_classification_heatmap is not None:
                viz_path_heatmap = ncc_plots_dir / "ncc_classification_heatmap.png"
                visualize_ncc_classification_heatmap(h_gt, class_labels, predictions_dict, bins, viz_path_heatmap, cfg)
            
            if visualize_ncc_classification_input_space_heatmap is not None:
                viz_path_input_heatmap = ncc_plots_dir / "ncc_classification_input_space_heatmap.png"
                visualize_ncc_classification_input_space_heatmap(x, t, class_labels, predictions_dict, viz_path_input_heatmap, cfg)
            
            if visualize_ncc_classification_accuracy_changes is not None:
                viz_path_accuracy_changes = ncc_plots_dir / "ncc_classification_accuracy_changes.png"
                visualize_ncc_classification_accuracy_changes(x, t, class_labels, predictions_dict, viz_path_accuracy_changes, cfg)
            
            if visualize_ncc_classification_input_space_accuracy_changes is not None:
                viz_path_input_accuracy_changes = ncc_plots_dir / "ncc_classification_input_space_accuracy_changes.png"
                visualize_ncc_classification_input_space_accuracy_changes(x, t, class_labels, predictions_dict, viz_path_input_accuracy_changes, cfg)
        except (ValueError, AttributeError) as e:
            print(f"  Warning: Problem-specific NCC visualization not available: {e}")

    # Build metrics summary
    metrics_summary = {
        'bins': bins,
        'num_classes': num_classes,
        'num_samples': N,
        'layers_analyzed': hidden_layers,
        'layer_accuracies': {
            layer_name: metrics['accuracy']
            for layer_name, metrics in ncc_results['layer_metrics'].items()
        },
        'layer_compactness': {
            layer_name: {
                'intra_class': metrics['compactness']['intra_class_dist'],
                'intra_class_std': metrics['compactness']['intra_class_std'],
                'inter_class_mean': metrics['compactness']['inter_class_mean'],
                'inter_class_std': metrics['compactness']['inter_class_std']
            }
            for layer_name, metrics in ncc_results['layer_metrics'].items()
        },
        'layer_margins': {
            layer_name: {
                'mean_margin': metrics['margins']['mean_margin'],
                'std_margin': metrics['margins']['std_margin'],
                'fraction_positive': metrics['margins']['fraction_positive']
            }
            for layer_name, metrics in ncc_results['layer_metrics'].items()
        }
    }

    # Save metrics to file (only if not suppressed)
    if not suppress_plots:
        ncc_metrics_path = ncc_plots_dir / "ncc_metrics.json"
        with open(ncc_metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"  NCC metrics saved to {ncc_metrics_path}")

        print("\n" + "=" * 60)
        print("NCC Analysis Complete")
        print("=" * 60)
        print(f"Results saved to: {ncc_plots_dir}")
        print("  - ncc_layer_accuracy.png")
        print("  - ncc_compactness.png")
        print("  - ncc_center_geometry.png")
        print("  - ncc_margin.png")
        print("  - ncc_confusions.png")
        print(f"Metrics: {ncc_metrics_path}")
        print("=" * 60)

    return metrics_summary

