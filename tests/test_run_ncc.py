"""Test run_ncc.py orchestrator."""

import sys
import torch
from pathlib import Path
import shutil
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io import load_config


def test_run_ncc_with_training():
    """Test run_ncc.py with training mode."""
    print("Testing run_ncc with training mode...")

    # Load and modify config for quick test
    config = load_config()

    test_config = config.copy()
    test_config['epochs'] = 2
    test_config['batch_size'] = 32
    test_config['print_every'] = 1
    test_config['save_every'] = 1
    test_config['problem'] = 'schrodinger'
    test_config['eval_only'] = False
    test_config['resume_from'] = None
    test_config['bins'] = 4

    # Small dataset for speed
    test_config['n_residual_train'] = 150
    test_config['n_initial_train'] = 15
    test_config['n_boundary_train'] = 15
    test_config['n_residual_eval'] = 50
    test_config['n_initial_eval'] = 10
    test_config['n_boundary_eval'] = 10

    # Temporarily replace config
    import utils.io as io_module
    original_load = io_module.load_config

    def load_test_config(path="config/config.yaml"):
        return test_config

    io_module.load_config = load_test_config

    # Clean up any existing test data
    problem = test_config['problem']
    test_dataset_dir = Path("datasets") / problem
    if test_dataset_dir.exists():
        shutil.rmtree(test_dataset_dir)

    try:
        # Run orchestrator
        print("\n  Running orchestrator...")
        import run_ncc
        import importlib
        importlib.reload(run_ncc)

        run_ncc.main()

        # Verify outputs
        print("\n  Verifying outputs...")

        # Check run directory
        architecture = test_config['base_architecture']
        activation = test_config['activation']
        layers_str = "-".join(map(str, architecture))
        run_dir = Path("outputs") / f"{problem}_layers-{layers_str}_act-{activation}"

        assert run_dir.exists(), f"Run directory should exist: {run_dir}"
        print(f"    ✓ Run directory exists: {run_dir}")

        # Check NCC plots
        ncc_plots_dir = run_dir / "ncc_plots"
        assert ncc_plots_dir.exists(), "ncc_plots directory should exist"

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

        # Check NCC metrics
        ncc_metrics_path = run_dir / "ncc_metrics.json"
        assert ncc_metrics_path.exists(), "ncc_metrics.json should exist"
        print(f"    ✓ ncc_metrics.json exists")

        # Check training outputs also exist
        training_plots = run_dir / "training_plots"
        assert training_plots.exists(), "training_plots should exist"
        print(f"    ✓ training_plots exists")

        summary = run_dir / "summary.txt"
        assert summary.exists(), "summary.txt should exist"
        print(f"    ✓ summary.txt exists")

        # Cleanup
        shutil.rmtree(test_dataset_dir)
        shutil.rmtree(run_dir)
        print(f"\n    ✓ Test data cleaned up")

    finally:
        io_module.load_config = original_load


def test_run_ncc_eval_only():
    """Test run_ncc.py in eval-only mode."""
    print("\nTesting run_ncc in eval-only mode...")

    # First create a checkpoint by training briefly
    config = load_config()
    test_config = config.copy()
    test_config['epochs'] = 1
    test_config['batch_size'] = 32
    test_config['print_every'] = 1
    test_config['save_every'] = 1
    test_config['problem'] = 'schrodinger'
    test_config['n_residual_train'] = 100
    test_config['n_initial_train'] = 10
    test_config['n_boundary_train'] = 10
    test_config['n_residual_eval'] = 50
    test_config['n_initial_eval'] = 5
    test_config['n_boundary_eval'] = 5

    problem = test_config['problem']
    test_dataset_dir = Path("datasets") / problem
    if test_dataset_dir.exists():
        shutil.rmtree(test_dataset_dir)

    # Generate datasets and train
    from utils.dataset_gen import generate_and_save_datasets
    from models.fc_model import FCNet
    from losses.schrodinger_loss import build_loss
    from trainer.trainer import train
    from utils.io import make_run_dir

    generate_and_save_datasets(test_config)

    model = FCNet(test_config['base_architecture'],
                  test_config['activation'], test_config)
    loss_fn = build_loss(**test_config)

    run_dir = make_run_dir(problem, test_config['base_architecture'],
                          test_config['activation'])

    checkpoint_path = train(
        model=model,
        loss_fn=loss_fn,
        train_data_path=str(test_dataset_dir / "training_data.pt"),
        eval_data_path=str(test_dataset_dir / "eval_data.pt"),
        cfg=test_config,
        run_dir=run_dir
    )

    print(f"  ✓ Training checkpoint created: {checkpoint_path}")

    # Now test eval-only mode with NCC
    test_config['eval_only'] = True
    test_config['resume_from'] = str(checkpoint_path)

    # Temporarily replace config
    import utils.io as io_module
    original_load = io_module.load_config

    def load_test_config(path="config/config.yaml"):
        return test_config

    io_module.load_config = load_test_config

    try:
        # Create new run dir for NCC-only
        ncc_run_dir = Path("outputs") / "test_ncc_eval_only"
        if ncc_run_dir.exists():
            shutil.rmtree(ncc_run_dir)

        # Monkey patch make_run_dir
        def test_make_run_dir(problem, layers, act):
            ncc_run_dir.mkdir(parents=True, exist_ok=True)
            (ncc_run_dir / "training_plots").mkdir(exist_ok=True)
            (ncc_run_dir / "ncc_plots").mkdir(exist_ok=True)
            return ncc_run_dir

        original_make_run_dir = io_module.make_run_dir
        io_module.make_run_dir = test_make_run_dir

        # Run orchestrator in eval-only mode
        import run_ncc
        import importlib
        importlib.reload(run_ncc)

        run_ncc.main()

        # Verify NCC outputs
        print("\n  Verifying NCC outputs...")

        ncc_plots_dir = ncc_run_dir / "ncc_plots"
        assert ncc_plots_dir.exists(), "ncc_plots should exist"

        ncc_metrics = ncc_run_dir / "ncc_metrics.json"
        assert ncc_metrics.exists(), "ncc_metrics.json should exist"

        print(f"    ✓ NCC analysis completed in eval-only mode")

        # Cleanup
        shutil.rmtree(test_dataset_dir)
        shutil.rmtree(run_dir)
        shutil.rmtree(ncc_run_dir)
        print(f"    ✓ Test data cleaned up")

        io_module.make_run_dir = original_make_run_dir

    finally:
        io_module.load_config = original_load


def test_run_ncc_requires_checkpoint():
    """Test that eval-only mode requires resume_from."""
    print("\nTesting that eval-only requires checkpoint...")

    config = load_config()
    test_config = config.copy()
    test_config['eval_only'] = True
    test_config['resume_from'] = None  # Not set

    import utils.io as io_module
    original_load = io_module.load_config

    def load_test_config(path="config/config.yaml"):
        return test_config

    io_module.load_config = load_test_config

    try:
        import run_ncc
        import importlib
        importlib.reload(run_ncc)

        # This should raise ValueError
        try:
            run_ncc.main()
            assert False, "Should have raised ValueError for missing resume_from"
        except ValueError as e:
            assert "resume_from" in str(e).lower(), \
                "Error should mention resume_from"
            print(f"  ✓ Correctly raised error: {str(e)[:60]}...")

    finally:
        io_module.load_config = original_load


if __name__ == "__main__":
    print("=" * 60)
    print("Step 9 — NCC orchestrator — Tests")
    print("=" * 60)

    try:
        test_run_ncc_with_training()
        test_run_ncc_eval_only()
        test_run_ncc_requires_checkpoint()

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

