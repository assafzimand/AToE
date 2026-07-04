"""Test configuration loading and run directory creation."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io import load_config, make_run_dir


def test_config_loading():
    """Test that config loads correctly and has all required keys."""
    print("Testing config loading...")
    
    # Load config
    config = load_config()
    
    # Check that config is a dictionary
    assert isinstance(config, dict), "Config should be a dictionary"
    
    # Required keys from PRD
    required_keys = [
        'problem', 'base_architecture', 'activation', 'epochs', 'batch_size', 
        'lr', 'cuda', 'eval_only', 'resume_from', 'bins', 'seed',
        'n_residual_train', 'n_initial_train', 'n_boundary_train',
        'n_residual_eval', 'n_initial_eval', 'n_boundary_eval'
    ]
    
    # Required problem-specific keys
    problem_keys = ['spatial_dim', 'spatial_domain', 'temporal_domain']
    
    # Check all keys exist
    for key in required_keys:
        assert key in config, f"Missing required key: {key}"
    
    # Check types
    assert isinstance(config['problem'], str), "problem should be a string"
    assert isinstance(config['base_architecture'], list), "base_architecture should be a list"
    assert isinstance(config['activation'], str), "activation should be a string"
    assert isinstance(config['epochs'], int), "epochs should be an integer"
    assert isinstance(config['batch_size'], int), "batch_size should be an integer"
    assert isinstance(config['lr'], float), "lr should be a float"
    assert isinstance(config['cuda'], bool), "cuda should be a boolean"
    assert isinstance(config['eval_only'], bool), "eval_only should be a boolean"
    assert config['resume_from'] is None or isinstance(config['resume_from'], str), \
        "resume_from should be None or string"
    assert isinstance(config['bins'], int), "bins should be an integer"
    
    # Check dataset size keys are integers
    for key in ['n_residual_train', 'n_initial_train', 'n_boundary_train',
                'n_residual_eval', 'n_initial_eval', 'n_boundary_eval']:
        assert isinstance(config[key], int), f"{key} should be an integer"
    
    # Check seed
    assert isinstance(config['seed'], int), "seed should be an integer"
    
    # Check problem-specific configuration
    problem = config['problem']
    assert problem in config, f"Problem '{problem}' configuration not found in config"
    problem_config = config[problem]
    
    for key in problem_keys:
        assert key in problem_config, f"Missing problem-specific key: {key}"
    
    assert isinstance(problem_config['spatial_dim'], int), "spatial_dim should be an integer"
    assert isinstance(problem_config['spatial_domain'], list), "spatial_domain should be a list"
    assert isinstance(problem_config['temporal_domain'], list), "temporal_domain should be a list"
    
    print("✓ Config loaded successfully")
    print(f"✓ Problem: {config['problem']}")
    print(f"✓ Architecture: {config['base_architecture']}")
    print(f"✓ Activation: {config['activation']}")
    print(f"✓ All {len(required_keys)} required keys present with correct types")
    print(f"✓ Problem-specific config validated: spatial_dim={problem_config['spatial_dim']}, "
          f"spatial_domain={problem_config['spatial_domain']}, temporal_domain={problem_config['temporal_domain']}")


def test_run_dir_creation():
    """Test run directory creation with proper naming."""
    print("\nTesting run directory creation...")
    
    # Load config to get parameters
    config = load_config()
    
    # Create run directory
    run_dir = make_run_dir(
        problem=config['problem'],
        layers=config['base_architecture'],
        act=config['activation']
    )
    
    # Check that directory exists
    assert run_dir.exists(), "Run directory should exist"
    assert run_dir.is_dir(), "Run directory should be a directory"
    
    # Check subdirectories
    assert (run_dir / "training_plots").exists(), "training_plots subdirectory should exist"
    assert (run_dir / "ncc_plots").exists(), "ncc_plots subdirectory should exist"
    
    # Check naming format
    expected_name = f"{config['problem']}-2-50-100-50-2-{config['activation']}"
    assert run_dir.name == expected_name, f"Directory name should be {expected_name}"
    
    print(f"✓ Run directory created: {run_dir}")
    print(f"✓ Directory name format correct: {run_dir.name}")
    print(f"✓ Subdirectories created: training_plots/, ncc_plots/")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 1 — Base config + IO helpers — Tests")
    print("=" * 60)
    
    try:
        test_config_loading()
        test_run_dir_creation()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)
