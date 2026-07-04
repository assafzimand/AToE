"""NCC visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
import seaborn as sns


def plot_ncc_layer_accuracy(
    layer_metrics: Dict,
    save_path: Path
) -> None:
    """
    Plot NCC accuracy across layers.

    Args:
        layer_metrics: Dict mapping layer_name -> metrics
        save_path: Path to save plot
    """
    layer_names = list(layer_metrics.keys())
    accuracies = [layer_metrics[ln]['accuracy'] for ln in layer_names]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(range(len(layer_names)), accuracies, 'o-', linewidth=2,
            markersize=8, color='#2E86AB')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('NCC Accuracy', fontsize=12)
    ax.set_title('NCC Classification Accuracy per Layer',
                fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Add value labels
    for i, acc in enumerate(accuracies):
        ax.text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_compactness(
    layer_metrics: Dict,
    save_path: Path
) -> None:
    """
    Plot intra-class vs inter-class distances.

    Args:
        layer_metrics: Dict mapping layer_name -> metrics
        save_path: Path to save plot
    """
    layer_names = list(layer_metrics.keys())
    intra_dists = [layer_metrics[ln]['compactness']['intra_class_dist']
                   for ln in layer_names]
    intra_stds = [layer_metrics[ln]['compactness']['intra_class_std']
                  for ln in layer_names]
    inter_means = [layer_metrics[ln]['compactness']['inter_class_mean']
                   for ln in layer_names]
    inter_stds = [layer_metrics[ln]['compactness']['inter_class_std']
                  for ln in layer_names]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(layer_names))
    ax.errorbar(x, intra_dists, yerr=intra_stds, fmt='o-', linewidth=2,
                markersize=8, label='Intra-class distance (mean ± std)',
                color='#E63946', capsize=5)
    ax.errorbar(x, inter_means, yerr=inter_stds, fmt='o-', linewidth=2,
                markersize=8, label='Inter-class distance (mean ± std)',
                color='#06A77D', capsize=5)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title('Class Compactness: Intra vs Inter-class Distances',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_center_geometry(
    layer_metrics: Dict,
    save_path: Path
) -> None:
    """
    Plot center norms and mean pairwise distances.

    Args:
        layer_metrics: Dict mapping layer_name -> metrics
        save_path: Path to save plot
    """
    layer_names = list(layer_metrics.keys())
    mean_norms = [layer_metrics[ln]['center_geometry']['mean_center_norm']
                  for ln in layer_names]
    std_norms = [layer_metrics[ln]['center_geometry']['std_center_norm']
                 for ln in layer_names]

    # Compute mean pairwise distance from pairwise_distances matrix
    mean_pairwise = []
    for ln in layer_names:
        pw_dists = layer_metrics[ln]['center_geometry']['pairwise_distances']
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(pw_dists), k=1).astype(bool)
        if mask.sum() > 0:
            mean_pairwise.append(pw_dists[mask].mean())
        else:
            mean_pairwise.append(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Center norms
    ax = axes[0]
    x = np.arange(len(layer_names))
    ax.errorbar(x, mean_norms, yerr=std_norms, fmt='o-', linewidth=2,
                markersize=8, color='#9D4EDD', capsize=5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Center Norm', fontsize=12)
    ax.set_title('Class Center Norms', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

    # Plot 2: Mean pairwise distances
    ax = axes[1]
    ax.plot(x, mean_pairwise, 'o-', linewidth=2, markersize=8,
            color='#F77F00')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Pairwise Distance', fontsize=12)
    ax.set_title('Mean Distance Between Centers',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_margin(
    layer_metrics: Dict,
    save_path: Path
) -> None:
    """
    Plot margin statistics and fraction positive.

    Args:
        layer_metrics: Dict mapping layer_name -> metrics
        save_path: Path to save plot
    """
    layer_names = list(layer_metrics.keys())
    mean_margins = [layer_metrics[ln]['margins']['mean_margin']
                    for ln in layer_names]
    std_margins = [layer_metrics[ln]['margins']['std_margin']
                   for ln in layer_names]
    frac_positive = [layer_metrics[ln]['margins']['fraction_positive']
                     for ln in layer_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(layer_names))

    # Plot 1: Mean margin with error bars
    ax = axes[0]
    ax.errorbar(x, mean_margins, yerr=std_margins, fmt='o-', linewidth=2,
                markersize=8, color='#D62828', capsize=5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Margin', fontsize=12)
    ax.set_title('Classification Margin (mean ± std)',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

    # Plot 2: Fraction with positive margin
    ax = axes[1]
    ax.bar(x, frac_positive, color='#06A77D', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Fraction Positive Margin', fontsize=12)
    ax.set_title('Fraction of Samples with Positive Margin',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, frac in enumerate(frac_positive):
        ax.text(i, frac + 0.02, f'{frac:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(
    layer_metrics: Dict,
    save_dir: Path,
    max_classes_display: int = None
) -> None:
    """
    Plot confusion matrices for each layer.

    Args:
        layer_metrics: Dict mapping layer_name -> metrics
        save_dir: Directory to save plots
        max_classes_display: Maximum number of classes to display (None = show all)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    layer_names = list(layer_metrics.keys())

    # Determine grid size
    n_layers = len(layer_names)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    if n_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, layer_name in enumerate(layer_names):
        ax = axes[idx]
        confusion = layer_metrics[layer_name]['confusion_matrix']

        # Show all classes (no limiting)
        confusion_subset = confusion
        title_suffix = ""

        # Plot heatmap with row normalization
        # Convert to numpy
        confusion_np = confusion_subset.cpu().numpy() if torch.is_tensor(confusion_subset) else confusion_subset
        
        # Row normalized: each row sums to 1.0, so vmax=1.0
        sns.heatmap(confusion_np, annot=False, cmap='Blues',
                    cbar=True, ax=ax, vmin=0, vmax=1.0)

        ax.set_xlabel('Predicted Class', fontsize=10)
        ax.set_ylabel('True Class', fontsize=10)
        ax.set_title(f'{layer_name}{title_suffix}\n(row normalized)',
                    fontsize=11, fontweight='bold')

    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    save_path = save_dir / 'ncc_confusions.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Confusion matrices saved to {save_path}")


def generate_all_ncc_plots(
    ncc_results: Dict,
    save_dir: Path
) -> None:
    """
    Generate all NCC plots.

    Args:
        ncc_results: Results from compute_all_ncc_metrics
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    layer_metrics = ncc_results['layer_metrics']

    print("  Generating NCC plots...")

    # Plot 1: Layer accuracy
    plot_ncc_layer_accuracy(
        layer_metrics,
        save_dir / 'ncc_layer_accuracy.png'
    )
    print(f"    Layer accuracy plot saved")

    # Plot 2: Compactness
    plot_compactness(
        layer_metrics,
        save_dir / 'ncc_compactness.png'
    )
    print(f"    Compactness plot saved")

    # Plot 3: Center geometry
    plot_center_geometry(
        layer_metrics,
        save_dir / 'ncc_center_geometry.png'
    )
    print(f"    Center geometry plot saved")

    # Plot 4: Margin
    plot_margin(
        layer_metrics,
        save_dir / 'ncc_margin.png'
    )
    print(f"    Margin plot saved")

    # Plot 5: Confusion matrices
    plot_confusion_matrices(
        layer_metrics,
        save_dir
    )


def plot_ncc_history_shaded(history: List[tuple], save_dir: Path) -> None:
    """
    Overlay NCC accuracy and margin SNR across epochs for a single model using shaded progression.
    history: list of (epoch, metrics_summary)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    # Sort by epoch
    history = sorted(history, key=lambda x: x[0])
    # Colors
    base_color = '#1f77b4'
    alphas = []
    for idx in range(len(history)):
        alphas.append(min(0.45 + 0.15 * idx, 1.0))

    # Accuracy plot (overwrite main file)
    fig, ax = plt.subplots(figsize=(10, 6))
    for (epoch, metrics), alpha in zip(history, alphas):
        layers = metrics['layers_analyzed']
        accs = [metrics['layer_accuracies'][ln] for ln in layers]
        label = f"Epoch {epoch}"
        ax.plot(layers, accs, marker='o', color=base_color, alpha=alpha, label=label)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Accuracy')
    ax.set_title('NCC Accuracy (mid-training shading)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "ncc_layer_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Margin SNR plot (line only, no bars; overwrite main file)
    fig, ax = plt.subplots(figsize=(10, 6))
    for (epoch, metrics), alpha in zip(history, alphas):
        layers = metrics['layers_analyzed']
        snrs = []
        for ln in layers:
            mean = metrics['layer_margins'][ln]['mean_margin']
            std = metrics['layer_margins'][ln]['std_margin']
            snrs.append(mean / std if std > 0 else 0)
        label = f"Epoch {epoch}"
        ax.plot(layers, snrs, marker='o', color=base_color, alpha=alpha, label=label)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Margin SNR')
    ax.set_title('NCC Margin SNR (mid-training shading)')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "ncc_margin.png", dpi=150, bbox_inches='tight')
    plt.close()
