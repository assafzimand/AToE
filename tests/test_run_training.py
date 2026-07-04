"""Test run_training.py orchestrator."""

import sys
import torch
from pathlib import Path
import shutil
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io import load_config


def test_training_orchestrator():
    """Test run_training.py with 1-epoch smoke run."""
    print("Testing training orchestrator (1-epoch smoke run)...")

    # Load and modify config for quick test
    config = load_config()
    
    # Create a temporary config file for testing
    test_config = config.copy()
    test_config['epochs'] = 1
    test_config['batch_size'] = 64
    test_config['print_every'] = 1
    test_config['save_every'] = 1
    test_config['problem'] = 'schrodinger'
    test_config['eval_only'] = False
    test_config['resume_from'] = None

    # Small dataset for speed
    test_config['n_residual_train'] = 300
    test_config['n_initial_train'] = 30
    test_config['n_boundary_train'] = 30
    test_config['n_residual_eval'] = 100
    test_config['n_initial_eval'] = 10
    test_config['n_boundary_eval'] = 10

    # Save temporary config
    temp_config_path = Path("config") / "test_config.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False)

    # Clean up any existing test outputs
    problem = test_config['problem']
    test_dataset_dir = Path("datasets") / problem
    if test_dataset_dir.exists():
        shutil.rmtree(test_dataset_dir)

    # Run training script by importing and calling main
    # Temporarily replace config path
    import utils.io as io_module
    original_load = io_module.load_config

    def load_test_config(path="config/config.yaml"):
        return test_config

    io_module.load_config = load_test_config

    try:
        # Import and run
        print("\n  Running orchestrator...")
        import run_training
        
        # Reload to get fresh import
        import importlib
        importlib.reload(run_training)
        
        # Run main
        run_training.main()

        # Verify outputs
        print("\n  Verifying outputs...")

        # Check run directory exists
        architecture = test_config['base_architecture']
        activation = test_config['activation']
        layers_str = "-".join(map(str, architecture))
        run_dir_name = f"{problem}_layers-{layers_str}_act-{activation}"
        run_dir = Path("outputs") / run_dir_name

        assert run_dir.exists(), f"Run directory should exist: {run_dir}"
        print(f"    ✓ Run directory exists: {run_dir}")

        # Check config_used.yaml
        config_used = run_dir / "config_used.yaml"
        assert config_used.exists(), "config_used.yaml should exist"
        print(f"    ✓ config_used.yaml exists")

        # Check metrics.json
        metrics_json = run_dir / "metrics.json"
        assert metrics_json.exists(), "metrics.json should exist"
        print(f"    ✓ metrics.json exists")

        # Check training_plots
        training_plots = run_dir / "training_plots"
        assert training_plots.exists(), "training_plots/ should exist"
        
        curves = training_plots / "training_curves.png"
        assert curves.exists(), "training_curves.png should exist"
        print(f"    ✓ training_curves.png exists")

        predictions = training_plots / "final_predictions.png"
        assert predictions.exists(), "final_predictions.png should exist"
        print(f"    ✓ final_predictions.png exists")

        # Check summary.txt
        summary = run_dir / "summary.txt"
        assert summary.exists(), "summary.txt should exist"
        print(f"    ✓ summary.txt exists")

        # Check datasets were generated
        assert test_dataset_dir.exists(), "Dataset directory should exist"
        train_data = test_dataset_dir / "training_data.pt"
        eval_data = test_dataset_dir / "eval_data.pt"
        assert train_data.exists(), "training_data.pt should exist"
        assert eval_data.exists(), "eval_data.pt should exist"
        print(f"    ✓ Datasets generated")

        # Check checkpoints (now inside run_dir)
        checkpoint_dir = run_dir / "checkpoints"
        assert checkpoint_dir.exists(), "Checkpoint directory should exist"
        
        best_ckpt = checkpoint_dir / "best_model.pt"
        assert best_ckpt.exists(), "best_model.pt should exist"
        print(f"    ✓ Best checkpoint saved")

        final_ckpt = checkpoint_dir / "final_model.pt"
        assert final_ckpt.exists(), "final_model.pt should exist"
        print(f"    ✓ Final checkpoint saved")

        # Cleanup
        shutil.rmtree(test_dataset_dir)
        shutil.rmtree(run_dir)
        temp_config_path.unlink()
        print(f"\n    ✓ Test data cleaned up")

    finally:
        # Restore original load_config
        io_module.load_config = original_load


def test_eval_only_mode():
    """Test eval_only mode."""
    print("\nTesting eval_only mode...")

    # First, create a checkpoint by running a quick training
    config = load_config()
    test_config = config.copy()
    test_config['epochs'] = 1
    test_config['batch_size'] = 64
    test_config['print_every'] = 1
    test_config['save_every'] = 1
    test_config['problem'] = 'schrodinger'
    test_config['eval_only'] = False
    test_config['resume_from'] = None
    test_config['n_residual_train'] = 100
    test_config['n_initial_train'] = 10
    test_config['n_boundary_train'] = 10
    test_config['n_residual_eval'] = 50
    test_config['n_initial_eval'] = 5
    test_config['n_boundary_eval'] = 5

    # Generate datasets and train briefly
    from utils.dataset_gen import generate_and_save_datasets
    from models.fc_model import FCNet
    from losses.schrodinger_loss import build_loss
    from trainer.trainer import train
    from utils.io import make_run_dir

    # Clean up
    problem = test_config['problem']
    test_dataset_dir = Path("datasets") / problem
    if test_dataset_dir.exists():
        shutil.rmtree(test_dataset_dir)

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

    # Now test eval_only mode
    test_config['eval_only'] = True
    test_config['resume_from'] = str(checkpoint_path)

    # Temporarily replace config
    import utils.io as io_module
    original_load = io_module.load_config

    def load_test_config(path="config/config.yaml"):
        return test_config

    io_module.load_config = load_test_config

    try:
        # Run in eval mode
        import run_training
        import importlib
        importlib.reload(run_training)

        run_training.main()

        # The run_dir will be the same as before (using make_run_dir)
        architecture = test_config['base_architecture']
        activation = test_config['activation']
        layers_str = "-".join(map(str, architecture))
        eval_run_dir = Path("outputs") / f"{problem}_layers-{layers_str}_act-{activation}"

        # Verify eval outputs
        metrics_json = eval_run_dir / "metrics.json"
        assert metrics_json.exists(), "metrics.json should exist in eval mode"

        import json
        with open(metrics_json, 'r') as f:
            metrics = json.load(f)

        assert 'eval_loss' in metrics, "Should have eval_loss"
        assert 'eval_rel_l2' in metrics, "Should have eval_rel_l2"
        print(f"  ✓ Eval metrics saved")
        print(f"    Eval loss: {metrics['eval_loss']:.6f}")
        print(f"    Eval rel-L2: {metrics['eval_rel_l2']:.6f}")

        # Cleanup
        shutil.rmtree(test_dataset_dir)
        if run_dir.exists():
            shutil.rmtree(run_dir)
        if eval_run_dir.exists():
            shutil.rmtree(eval_run_dir)
        print(f"  ✓ Test data cleaned up")

    finally:
        io_module.load_config = original_load


if __name__ == "__main__":
    print("=" * 60)
    print("Step 6 — Training orchestrator — Tests")
    print("=" * 60)

    try:
        test_training_orchestrator()
        test_eval_only_mode()

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

