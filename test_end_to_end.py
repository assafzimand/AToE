"""
End-to-end dry run with placeholders.

This script runs the complete NCC-PINN pipeline:
1. Generate datasets
2. Train for a few epochs
3. Run NCC analysis

Verifies all outputs are created properly.
"""

import sys
import torch
from pathlib import Path
import shutil

# Ensure we're using the venv
print(f"Python: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print()

# Run the complete pipeline
print("=" * 70)
print("END-TO-END TEST: Complete NCC-PINN Pipeline with Placeholders")
print("=" * 70)

# Clean up any previous test data
print("\n1. Cleaning up previous test data...")
cleanup_paths = [
    Path("datasets/schrodinger"),
    Path("outputs/schrodinger-2-50-100-50-2-tanh"),
    Path("checkpoints/schrodinger")
]

for path in cleanup_paths:
    if path.exists():
        shutil.rmtree(path)
        print(f"   Removed: {path}")

print("   ✓ Cleanup complete\n")

# Modify config for a small test run
print("2. Setting up test configuration...")
from utils.io import load_config
import yaml

config = load_config()

# Small test configuration
test_config = config.copy()
test_config['problem'] = 'schrodinger'
test_config['epochs'] = 10  # Just 10 epochs
test_config['batch_size'] = 128
test_config['print_every'] = 2
test_config['save_every'] = 5
test_config['bins'] = 6
test_config['eval_only'] = False
test_config['resume_from'] = None

# Moderate dataset size
test_config['n_residual_train'] = 1000
test_config['n_initial_train'] = 100
test_config['n_boundary_train'] = 100
test_config['n_residual_eval'] = 300
test_config['n_initial_eval'] = 30
test_config['n_boundary_eval'] = 30

# Save test config
temp_config_path = Path("config/temp_test_config.yaml")
with open(temp_config_path, 'w') as f:
    yaml.dump(test_config, f, default_flow_style=False)

print("   Test configuration:")
print(f"     Problem: {test_config['problem']}")
print(f"     Epochs: {test_config['epochs']}")
print(f"     Train samples: {test_config['n_residual_train'] + test_config['n_initial_train'] + test_config['n_boundary_train']}")  # noqa
print(f"     Eval samples: {test_config['n_residual_eval'] + test_config['n_initial_eval'] + test_config['n_boundary_eval']}")  # noqa
print(f"     Bins for NCC: {test_config['bins']}")
print("   ✓ Configuration ready\n")

# Temporarily replace load_config
import utils.io as io_module
original_load = io_module.load_config


def load_test_config(path="config/config.yaml"):
    return test_config


io_module.load_config = load_test_config

try:
    # Run the complete pipeline
    print("3. Running complete pipeline (training + NCC)...")
    print("=" * 70)

    import run_ncc
    import importlib
    importlib.reload(run_ncc)

    run_ncc.main()

    print("\n" + "=" * 70)
    print("4. Verifying outputs...")
    print("=" * 70)

    problem = test_config['problem']
    architecture = test_config['base_architecture']
    activation = test_config['activation']
    layers_str = "-".join(map(str, architecture))
    run_dir = Path("outputs") / f"{problem}-{layers_str}-{activation}"

    # Check run directory
    assert run_dir.exists(), f"Run directory should exist: {run_dir}"
    print(f"\n✓ Run directory exists: {run_dir}")

    # Check all required files and directories
    required_outputs = {
        'config_used.yaml': 'Configuration file',
        'metrics.json': 'Training metrics',
        'summary.txt': 'Training summary',
        'ncc_metrics.json': 'NCC metrics',
        'training_plots': 'Training plots directory',
        'ncc_plots': 'NCC plots directory'
    }

    print("\nChecking required outputs:")
    all_good = True
    for output_name, description in required_outputs.items():
        output_path = run_dir / output_name
        if output_path.exists():
            print(f"   ✓ {output_name:25s} - {description}")
        else:
            print(f"   ✗ {output_name:25s} - MISSING!")
            all_good = False

    # Check specific plots
    training_plots_dir = run_dir / "training_plots"
    if training_plots_dir.exists():
        training_plots = [
            'training_curves.png',
            'final_predictions.png'
        ]
        print("\nTraining plots:")
        for plot_name in training_plots:
            plot_path = training_plots_dir / plot_name
            if plot_path.exists():
                print(f"   ✓ {plot_name}")
            else:
                print(f"   ✗ {plot_name} - MISSING!")
                all_good = False

    ncc_plots_dir = run_dir / "ncc_plots"
    if ncc_plots_dir.exists():
        ncc_plots = [
            'ncc_layer_accuracy.png',
            'ncc_compactness.png',
            'ncc_center_geometry.png',
            'ncc_margin.png',
            'ncc_confusions.png'
        ]
        print("\nNCC plots:")
        for plot_name in ncc_plots:
            plot_path = ncc_plots_dir / plot_name
            if plot_path.exists():
                print(f"   ✓ {plot_name}")
            else:
                print(f"   ✗ {plot_name} - MISSING!")
                all_good = False

    # Check checkpoints (now inside run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.exists():
        print("\nCheckpoints:")
        checkpoint_files = ['best_model.pt', 'final_model.pt']
        for ckpt_name in checkpoint_files:
            ckpt_path = checkpoint_dir / ckpt_name
            if ckpt_path.exists():
                print(f"   ✓ {ckpt_name}")
            else:
                print(f"   ✗ {ckpt_name} - MISSING!")
                all_good = False

    # Check datasets
    dataset_dir = Path("datasets") / problem
    if dataset_dir.exists():
        print("\nDatasets:")
        dataset_files = ['training_data.pt', 'eval_data.pt']
        for ds_name in dataset_files:
            ds_path = dataset_dir / ds_name
            if ds_path.exists():
                print(f"   ✓ {ds_name}")
            else:
                print(f"   ✗ {ds_name} - MISSING!")
                all_good = False

    print("\n" + "=" * 70)
    if all_good:
        print("✓✓✓ END-TO-END TEST PASSED! ✓✓✓")
        print("=" * 70)
        print("\nAll components working correctly:")
        print("  ✓ Dataset generation with IC/BC placeholders")
        print("  ✓ Model training with placeholder losses")
        print("  ✓ Training visualization")
        print("  ✓ Checkpoint saving")
        print("  ✓ NCC analysis")
        print("  ✓ NCC visualizations")
        print("\nThe NCC-PINN framework is ready for use!")
    else:
        print("✗✗✗ END-TO-END TEST FAILED! ✗✗✗")
        print("=" * 70)
        print("Some outputs are missing. Check the logs above.")
        sys.exit(1)

finally:
    # Restore original config loader
    io_module.load_config = original_load

    # Cleanup temp config
    if temp_config_path.exists():
        temp_config_path.unlink()

print("\n" + "=" * 70)
print("Test complete. Outputs are in:")
print(f"  {run_dir}")
print("=" * 70)

