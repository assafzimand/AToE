"""Test NCC runner integration."""

import sys
import torch
from pathlib import Path
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io import load_config
from models.fc_model import FCNet
from ncc.ncc_runner import run_ncc


def test_ncc_runner():
    """Test NCC runner with mock model."""
    print("Testing NCC runner...")

    # Load config
    config = load_config()
    device = torch.device('cuda' if config['cuda'] and
                          torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Create a small model
    architecture = [2, 20, 30, 20, 2]  # Small for testing
    activation = 'tanh'

    test_config = config.copy()
    test_config['base_architecture'] = architecture
    test_config['problem'] = 'schrodinger'

    model = FCNet(architecture, activation, test_config)
    model = model.to(device)

    print(f"  ✓ Model created with {len(model.get_layer_names())} layers")

    # Create synthetic eval data
    N = 100
    spatial_dim = test_config['schrodinger']['spatial_dim']
    output_dim = architecture[-1]

    eval_data = {
        'x': torch.randn(N, spatial_dim, device=device),
        't': torch.randn(N, 1, device=device),
        'h_gt': torch.randn(N, output_dim, device=device),
        'mask': {
            'residual': torch.ones(N, dtype=torch.bool, device=device),
            'IC': torch.zeros(N, dtype=torch.bool, device=device),
            'BC': torch.zeros(N, dtype=torch.bool, device=device)
        }
    }

    # Save eval data
    test_data_dir = Path("datasets") / "test_ncc"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    eval_data_path = test_data_dir / "eval_data.pt"
    torch.save(eval_data, eval_data_path)

    # Create run directory
    run_dir = Path("outputs") / "test_ncc_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Run NCC analysis
    print("\n  Running NCC analysis...")
    metrics_summary = run_ncc(
        model=model,
        eval_data_path=str(eval_data_path),
        cfg=test_config,
        run_dir=run_dir
    )

    # Verify outputs
    print("\n  Verifying outputs...")

    # Check ncc_plots directory
    ncc_plots_dir = run_dir / "ncc_plots"
    assert ncc_plots_dir.exists(), "ncc_plots directory should exist"
    print(f"    ✓ ncc_plots directory exists")

    # Check individual plots
    required_plots = [
        'ncc_layer_accuracy.png',
        'ncc_compactness.png',
        'ncc_center_geometry.png',
        'ncc_margin.png',
        'ncc_confusions.png'
    ]

    for plot_name in required_plots:
        plot_path = ncc_plots_dir / plot_name
        assert plot_path.exists(), f"{plot_name} should exist"
        print(f"    ✓ {plot_name} exists")

    # Check metrics JSON
    ncc_metrics_path = run_dir / "ncc_metrics.json"
    assert ncc_metrics_path.exists(), "ncc_metrics.json should exist"
    print(f"    ✓ ncc_metrics.json exists")

    # Check metrics content
    import json
    with open(ncc_metrics_path, 'r') as f:
        saved_metrics = json.load(f)

    assert 'bins' in saved_metrics, "Missing bins in metrics"
    assert 'num_classes' in saved_metrics, "Missing num_classes in metrics"
    assert 'layer_accuracies' in saved_metrics, \
        "Missing layer_accuracies in metrics"
    assert 'layer_compactness' in saved_metrics, \
        "Missing layer_compactness in metrics"
    assert 'layer_margins' in saved_metrics, \
        "Missing layer_margins in metrics"

    print(f"    ✓ Metrics JSON has correct structure")
    print(f"      Bins: {saved_metrics['bins']}")
    print(f"      Classes: {saved_metrics['num_classes']}")
    print(f"      Layers analyzed: {saved_metrics['layers_analyzed']}")

    # Verify metrics summary return value
    assert metrics_summary == saved_metrics, \
        "Returned metrics should match saved metrics"

    # Check that hidden layers were analyzed (not output layer)
    expected_hidden_layers = ['layer_1', 'layer_2', 'layer_3']
    assert saved_metrics['layers_analyzed'] == expected_hidden_layers, \
        f"Should analyze hidden layers only: {expected_hidden_layers}"
    print(f"    ✓ Hidden layers analyzed (output layer excluded)")

    # Cleanup
    shutil.rmtree(test_data_dir)
    shutil.rmtree(run_dir)
    print(f"\n    ✓ Test data cleaned up")


def test_ncc_with_different_bins():
    """Test NCC runner with different bin configurations."""
    print("\nTesting NCC with different bin values...")

    config = load_config()
    device = torch.device('cuda' if config['cuda'] and
                          torch.cuda.is_available() else 'cpu')

    # Test with different bin sizes
    for bins in [3, 5, 8]:
        print(f"\n  Testing with bins={bins}...")

        architecture = [2, 15, 15, 2]
        test_config = config.copy()
        test_config['base_architecture'] = architecture
        test_config['bins'] = bins
        test_config['problem'] = 'schrodinger'

        model = FCNet(architecture, activation='tanh', config=test_config)
        model = model.to(device)

        # Create eval data
        N = 50
        spatial_dim = test_config['schrodinger']['spatial_dim']
        output_dim = architecture[-1]

        eval_data = {
            'x': torch.randn(N, spatial_dim, device=device),
            't': torch.randn(N, 1, device=device),
            'h_gt': torch.randn(N, output_dim, device=device),
            'mask': {
                'residual': torch.ones(N, dtype=torch.bool, device=device),
                'IC': torch.zeros(N, dtype=torch.bool, device=device),
                'BC': torch.zeros(N, dtype=torch.bool, device=device)
            }
        }

        test_data_dir = Path("datasets") / f"test_ncc_bins_{bins}"
        test_data_dir.mkdir(parents=True, exist_ok=True)
        eval_data_path = test_data_dir / "eval_data.pt"
        torch.save(eval_data, eval_data_path)

        run_dir = Path("outputs") / f"test_ncc_bins_{bins}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Run NCC
        metrics = run_ncc(
            model=model,
            eval_data_path=str(eval_data_path),
            cfg=test_config,
            run_dir=run_dir
        )

        # Verify bins were used
        assert metrics['bins'] == bins, f"Bins should be {bins}"
        print(f"    ✓ bins={bins} created {metrics['num_classes']} classes")

        # Cleanup
        shutil.rmtree(test_data_dir)
        shutil.rmtree(run_dir)


if __name__ == "__main__":
    print("=" * 60)
    print("Step 8 — NCC runner — Tests")
    print("=" * 60)

    try:
        test_ncc_runner()
        test_ncc_with_different_bins()

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

