"""Test NCC core functionality with synthetic data."""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ncc.ncc_core import (
    create_class_labels_from_regression,
    compute_class_centers,
    compute_ncc_predictions,
    compute_ncc_accuracy,
    compute_compactness_metrics,
    compute_center_geometry_metrics,
    compute_margin_metrics,
    compute_confusion_matrix,
    compute_all_ncc_metrics
)


def test_binning():
    """Test regression to class label conversion."""
    print("Testing binning...")

    device = torch.device('cpu')  # Test on CPU for simplicity

    # Create synthetic regression outputs
    torch.manual_seed(42)
    N = 100
    output_dim = 2
    outputs = torch.randn(N, output_dim, device=device)

    bins = 5

    class_labels, class_map, bin_info = create_class_labels_from_regression(
        outputs, bins, device
    )

    # Check outputs
    assert class_labels.shape == (N,), "Class labels shape incorrect"
    assert len(class_map) > 0, "Class map should not be empty"
    assert len(class_map) <= bins ** output_dim, \
        "Too many classes created"
    assert bin_info['bins'] == bins, "Bins mismatch"
    assert bin_info['output_dim'] == output_dim, "Output dim mismatch"

    print(f"  ✓ Created {len(class_map)} classes from {N} samples")
    print(f"  ✓ Bins: {bins}, Output dim: {output_dim}")


def test_class_centers():
    """Test class center computation."""
    print("\nTesting class center computation...")

    device = torch.device('cpu')
    torch.manual_seed(42)

    # Create synthetic embeddings
    N = 200
    embedding_dim = 50
    num_classes = 10

    embeddings = torch.randn(N, embedding_dim, device=device)
    class_labels = torch.randint(0, num_classes, (N,), device=device)

    # Compute centers
    centers = compute_class_centers(embeddings, class_labels, num_classes)

    # Check output
    assert centers.shape == (num_classes, embedding_dim), \
        "Centers shape incorrect"
    assert not torch.isnan(centers).any(), "Centers contain NaN"
    assert not torch.isinf(centers).any(), "Centers contain Inf"

    print(f"  ✓ Centers computed: shape {centers.shape}")
    print(f"  ✓ No NaN or Inf values")


def test_ncc_predictions():
    """Test NCC prediction."""
    print("\nTesting NCC predictions...")

    device = torch.device('cpu')
    torch.manual_seed(42)

    N = 100
    embedding_dim = 30
    num_classes = 5

    # Create well-separated clusters
    embeddings = torch.randn(N, embedding_dim, device=device)
    true_labels = torch.randint(0, num_classes, (N,), device=device)

    # Add class-specific offsets to create separation
    for c in range(num_classes):
        mask = (true_labels == c)
        offset = torch.randn(1, embedding_dim, device=device) * 5
        embeddings[mask] += offset

    # Compute centers
    centers = compute_class_centers(embeddings, true_labels, num_classes)

    # Predict
    predictions = compute_ncc_predictions(embeddings, centers)

    # Check output
    assert predictions.shape == (N,), "Predictions shape incorrect"
    assert predictions.min() >= 0, "Predictions should be >= 0"
    assert predictions.max() < num_classes, \
        f"Predictions should be < {num_classes}"

    # Compute accuracy
    accuracy = compute_ncc_accuracy(predictions, true_labels)

    print(f"  ✓ Predictions computed: shape {predictions.shape}")
    print(f"  ✓ Accuracy: {accuracy:.3f}")
    assert accuracy > 0.5, "Accuracy should be reasonable for well-separated data"


def test_compactness_metrics():
    """Test compactness metric computation."""
    print("\nTesting compactness metrics...")

    device = torch.device('cpu')
    torch.manual_seed(42)

    N = 150
    embedding_dim = 40
    num_classes = 8

    embeddings = torch.randn(N, embedding_dim, device=device)
    class_labels = torch.randint(0, num_classes, (N,), device=device)

    # Compute centers
    centers = compute_class_centers(embeddings, class_labels, num_classes)

    # Compute compactness
    compactness = compute_compactness_metrics(
        embeddings, class_labels, centers, num_classes
    )

    # Check outputs
    assert 'intra_class_dist' in compactness, "Missing intra_class_dist"
    assert 'inter_class_mean' in compactness, "Missing inter_class_mean"
    assert 'inter_class_std' in compactness, "Missing inter_class_std"

    assert compactness['intra_class_dist'] >= 0, "Intra-class dist should be >= 0"
    assert compactness['inter_class_mean'] >= 0, "Inter-class mean should be >= 0"
    assert compactness['inter_class_std'] >= 0, "Inter-class std should be >= 0"

    print(f"  ✓ Intra-class distance: {compactness['intra_class_dist']:.3f}")
    print(f"  ✓ Inter-class mean: {compactness['inter_class_mean']:.3f}")
    print(f"  ✓ Inter-class std: {compactness['inter_class_std']:.3f}")


def test_center_geometry():
    """Test center geometry metrics."""
    print("\nTesting center geometry metrics...")

    device = torch.device('cpu')
    torch.manual_seed(42)

    num_classes = 6
    embedding_dim = 25

    centers = torch.randn(num_classes, embedding_dim, device=device)

    geometry = compute_center_geometry_metrics(centers)

    # Check outputs
    assert 'center_norms' in geometry, "Missing center_norms"
    assert 'pairwise_distances' in geometry, "Missing pairwise_distances"
    assert 'mean_center_norm' in geometry, "Missing mean_center_norm"

    assert len(geometry['center_norms']) == num_classes, \
        "Wrong number of center norms"
    assert geometry['pairwise_distances'].shape == (num_classes, num_classes), \
        "Wrong pairwise distances shape"

    print(f"  ✓ Mean center norm: {geometry['mean_center_norm']:.3f}")
    print(f"  ✓ Std center norm: {geometry['std_center_norm']:.3f}")


def test_margin_metrics():
    """Test margin computation."""
    print("\nTesting margin metrics...")

    device = torch.device('cpu')
    torch.manual_seed(42)

    N = 120
    embedding_dim = 35
    num_classes = 7

    embeddings = torch.randn(N, embedding_dim, device=device)
    class_labels = torch.randint(0, num_classes, (N,), device=device)

    # Compute centers
    centers = compute_class_centers(embeddings, class_labels, num_classes)

    # Compute margins
    margins_result = compute_margin_metrics(
        embeddings, class_labels, centers, num_classes
    )

    # Check outputs
    assert 'margins' in margins_result, "Missing margins"
    assert 'mean_margin' in margins_result, "Missing mean_margin"
    assert 'fraction_positive' in margins_result, "Missing fraction_positive"

    assert len(margins_result['margins']) == N, "Wrong number of margins"
    assert 0 <= margins_result['fraction_positive'] <= 1, \
        "Fraction positive should be in [0, 1]"

    print(f"  ✓ Mean margin: {margins_result['mean_margin']:.3f}")
    print(f"  ✓ Fraction positive: {margins_result['fraction_positive']:.3f}")


def test_confusion_matrix():
    """Test confusion matrix computation."""
    print("\nTesting confusion matrix...")

    device = torch.device('cpu')
    torch.manual_seed(42)

    N = 100
    num_classes = 5

    # Create predictions and labels
    predictions = torch.randint(0, num_classes, (N,), device=device)
    true_labels = torch.randint(0, num_classes, (N,), device=device)

    confusion = compute_confusion_matrix(predictions, true_labels, num_classes)

    # Check output
    assert confusion.shape == (num_classes, num_classes), \
        "Confusion matrix shape incorrect"
    assert torch.allclose(confusion.sum(dim=1),
                         torch.ones(num_classes, device=device),
                         atol=1e-6), \
        "Confusion matrix rows should sum to 1 (normalized)"

    print(f"  ✓ Confusion matrix shape: {confusion.shape}")
    print(f"  ✓ Rows sum to 1 (normalized)")


def test_all_metrics_integration():
    """Test complete NCC pipeline."""
    print("\nTesting complete NCC pipeline...")

    device = torch.device('cpu')
    torch.manual_seed(42)

    # Create synthetic data
    N = 200
    embedding_dim_1 = 50
    embedding_dim_2 = 100
    output_dim = 2

    # Create embeddings for two layers
    embeddings_dict = {
        'layer_1': torch.randn(N, embedding_dim_1, device=device),
        'layer_2': torch.randn(N, embedding_dim_2, device=device)
    }

    # Create ground truth outputs
    ground_truth = torch.randn(N, output_dim, device=device)

    # Run complete NCC analysis
    results = compute_all_ncc_metrics(
        embeddings_dict,
        ground_truth,
        bins=4,
        device=device
    )

    # Check output structure
    assert 'class_labels' in results, "Missing class_labels"
    assert 'class_map' in results, "Missing class_map"
    assert 'bin_info' in results, "Missing bin_info"
    assert 'layer_metrics' in results, "Missing layer_metrics"

    # Check layer metrics
    assert 'layer_1' in results['layer_metrics'], "Missing layer_1 metrics"
    assert 'layer_2' in results['layer_metrics'], "Missing layer_2 metrics"

    # Check each layer has all metrics
    for layer_name in ['layer_1', 'layer_2']:
        metrics = results['layer_metrics'][layer_name]
        assert 'accuracy' in metrics, f"{layer_name}: missing accuracy"
        assert 'compactness' in metrics, f"{layer_name}: missing compactness"
        assert 'center_geometry' in metrics, \
            f"{layer_name}: missing center_geometry"
        assert 'margins' in metrics, f"{layer_name}: missing margins"
        assert 'confusion_matrix' in metrics, \
            f"{layer_name}: missing confusion_matrix"

        print(f"  ✓ {layer_name}: accuracy = {metrics['accuracy']:.3f}")

    print(f"  ✓ Complete pipeline successful")
    print(f"  ✓ Number of classes: {results['bin_info']['num_classes']}")


def test_cuda_compatibility():
    """Test CUDA compatibility (if available)."""
    print("\nTesting CUDA compatibility...")

    if not torch.cuda.is_available():
        print("  ⚠ CUDA not available, skipping CUDA test")
        return

    device = torch.device('cuda')
    torch.manual_seed(42)

    N = 100
    embedding_dim = 40
    output_dim = 2
    num_classes = 8

    # Create data on CUDA
    embeddings = torch.randn(N, embedding_dim, device=device)
    outputs = torch.randn(N, output_dim, device=device)
    class_labels = torch.randint(0, num_classes, (N,), device=device)

    # Test binning on CUDA
    labels, class_map, bin_info = create_class_labels_from_regression(
        outputs, bins=5, device=device
    )
    assert labels.device.type == 'cuda', "Labels should be on CUDA"

    # Test centers on CUDA
    centers = compute_class_centers(embeddings, class_labels, num_classes)
    assert centers.device.type == 'cuda', "Centers should be on CUDA"

    # Test predictions on CUDA
    predictions = compute_ncc_predictions(embeddings, centers)
    assert predictions.device.type == 'cuda', "Predictions should be on CUDA"

    print("  ✓ All operations work on CUDA")
    print("  ✓ No CPU fallback detected")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 7 — NCC core — Tests")
    print("=" * 60)

    try:
        test_binning()
        test_class_centers()
        test_ncc_predictions()
        test_compactness_metrics()
        test_center_geometry()
        test_margin_metrics()
        test_confusion_matrix()
        test_all_metrics_integration()
        test_cuda_compatibility()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

