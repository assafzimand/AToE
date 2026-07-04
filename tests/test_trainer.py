"""Test trainer with small training run."""

import sys
import torch
from pathlib import Path
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io import load_config, make_run_dir
from utils.dataset_gen import generate_and_save_datasets
from models.fc_model import FCNet
from losses.schrodinger_loss import build_loss
from trainer.trainer import train


def test_trainer_small_run():
    """Test trainer with very small run (3 epochs)."""
    print("Testing trainer with small run...")

    # Load config and modify for quick test
    config = load_config()
    config['epochs'] = 3
    config['batch_size'] = 32
    config['print_every'] = 1
    config['save_every'] = 2
    config['problem'] = 'schrodinger'

    # Use tiny dataset for testing
    config['n_residual_train'] = 200
    config['n_initial_train'] = 20
    config['n_boundary_train'] = 20
    config['n_residual_eval'] = 50
    config['n_initial_eval'] = 10
    config['n_boundary_eval'] = 10

    device = torch.device('cuda' if config['cuda'] and
                          torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Generate test datasets
    print("  Generating test datasets...")
    
    # Clean up any existing test data
    test_data_dir = Path("datasets") / "schrodinger_test"
    if test_data_dir.exists():
        shutil.rmtree(test_data_dir)
    
    config_test = config.copy()
    config_test['problem'] = 'schrodinger'
    
    # Generate datasets using solver
    from solvers.schrodinger_solver import generate_dataset
    
    train_data = generate_dataset(
        n_residual=config['n_residual_train'],
        n_ic=config['n_initial_train'],
        n_bc=config['n_boundary_train'],
        device=device,
        config=config
    )
    
    eval_data = generate_dataset(
        n_residual=config['n_residual_eval'],
        n_ic=config['n_initial_eval'],
        n_bc=config['n_boundary_eval'],
        device=device,
        config=config
    )
    
    # Save datasets
    test_data_dir.mkdir(parents=True, exist_ok=True)
    train_path = test_data_dir / "training_data.pt"
    eval_path = test_data_dir / "eval_data.pt"
    torch.save(train_data, train_path)
    torch.save(eval_data, eval_path)

    # Build model
    print("  Building model...")
    model = FCNet(config['base_architecture'], config['activation'], config)

    # Build loss
    print("  Building loss...")
    loss_fn = build_loss(**config)

    # Create run directory
    run_dir = Path("outputs") / "test_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_plots").mkdir(exist_ok=True)

    # Run training
    print("  Running training...")
    checkpoint_path = train(
        model=model,
        loss_fn=loss_fn,
        train_data_path=str(train_path),
        eval_data_path=str(eval_path),
        cfg=config,
        run_dir=run_dir
    )

    # Verify outputs
    print("\n  Verifying outputs...")

    # Check checkpoint exists
    assert checkpoint_path is not None, "Checkpoint path should not be None"
    assert checkpoint_path.exists(), f"Checkpoint should exist: {checkpoint_path}"

    # Check checkpoint contents
    checkpoint = torch.load(checkpoint_path)
    assert 'model_state_dict' in checkpoint, "Checkpoint missing model_state_dict"
    assert 'optimizer_state_dict' in checkpoint, "Checkpoint missing optimizer_state_dict"
    assert 'epoch' in checkpoint, "Checkpoint missing epoch"
    assert 'config' in checkpoint, "Checkpoint missing config"
    assert 'metrics' in checkpoint, "Checkpoint missing metrics"

    print(f"    ✓ Checkpoint saved with all info: {checkpoint_path}")

    # Check plots exist
    training_curves = run_dir / "training_plots" / "training_curves.png"
    assert training_curves.exists(), "Training curves plot should exist"
    print(f"    ✓ Training curves plot exists")

    final_pred = run_dir / "training_plots" / "final_predictions.png"
    assert final_pred.exists(), "Final predictions plot should exist"
    print(f"    ✓ Final predictions plot exists")

    # Check metrics JSON
    metrics_path = run_dir / "metrics.json"
    assert metrics_path.exists(), "Metrics JSON should exist"
    print(f"    ✓ Metrics JSON exists")

    # Check summary
    summary_path = run_dir / "summary.txt"
    assert summary_path.exists(), "Summary file should exist"
    print(f"    ✓ Summary file exists")

    # Check config
    config_path = run_dir / "config_used.yaml"
    assert config_path.exists(), "Config file should exist"
    print(f"    ✓ Config file exists")

    # Check periodic checkpoints (now inside run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    epoch_checkpoint = checkpoint_dir / "checkpoint_epoch_2.pt"
    assert epoch_checkpoint.exists(), "Periodic checkpoint should exist"
    print(f"    ✓ Periodic checkpoint exists (epoch 2)")

    # Verify metrics have expected structure
    import json
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    assert 'epochs' in metrics, "Metrics missing epochs"
    assert 'train_loss' in metrics, "Metrics missing train_loss"
    assert 'eval_loss' in metrics, "Metrics missing eval_loss"
    assert 'eval_rel_l2' in metrics, "Metrics missing eval_rel_l2"
    assert len(metrics['epochs']) == 3, "Should have 3 epochs of metrics"

    print(f"    ✓ Metrics have correct structure with {len(metrics['epochs'])} epochs")

    # Cleanup test data
    shutil.rmtree(test_data_dir)
    shutil.rmtree(run_dir)
    print(f"\n  ✓ Test data cleaned up")


def test_cuda_batching():
    """Test that batching works correctly on CUDA."""
    print("\nTesting CUDA batching...")

    config = load_config()
    device = torch.device('cuda' if config['cuda'] and
                          torch.cuda.is_available() else 'cpu')

    # Create tiny batch
    from solvers.schrodinger_solver import generate_dataset
    data = generate_dataset(
        n_residual=100,
        n_ic=10,
        n_bc=10,
        device=device,
        config=config
    )

    # Create DataLoader
    from trainer.trainer import _create_dataloader
    loader = _create_dataloader(data, batch_size=32, shuffle=True)

    # Test iteration
    for i, batch in enumerate(loader):
        # Check batch structure
        assert 'x' in batch, "Batch missing x"
        assert 't' in batch, "Batch missing t"
        assert 'h_gt' in batch, "Batch missing h_gt"
        assert 'mask' in batch, "Batch missing mask"

        # Check device
        assert batch['x'].device.type == device.type, f"x should be on {device}"
        assert batch['t'].device.type == device.type, f"t should be on {device}"

        # Check shapes
        assert batch['x'].dim() == 2, "x should be 2D"
        assert batch['t'].dim() == 2, "t should be 2D"
        assert batch['h_gt'].dim() == 2, "h_gt should be 2D"

        if i == 0:
            print(f"  ✓ Batch structure correct")
            print(f"  ✓ Batch size: {batch['x'].shape[0]}")
            print(f"  ✓ Device: {batch['x'].device}")
            break

    print(f"  ✓ DataLoader works correctly on {device}")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 5 — Trainer — Tests")
    print("=" * 60)

    try:
        test_cuda_batching()
        test_trainer_small_run()

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

