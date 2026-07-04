"""Derivatives tracker orchestrator for trained models."""

import sys
import torch
import importlib
from pathlib import Path

from utils.io import load_config, make_run_dir
from models.fc_model import FCNet
from derivatives_tracker.derivatives_runner import run_derivatives_tracker
from trainer.trainer import train


def main():
    """Orchestrate derivatives tracking analysis workflow."""
    print("=" * 60)
    print("Derivatives Tracker Orchestrator")
    print("=" * 60)

    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config()
    problem = config['problem']
    architecture = config['base_architecture']
    activation = config['activation']
    eval_only = config['eval_only']
    resume_from = config['resume_from']
    
    print(f"  Problem: {problem}")
    print(f"  Architecture: {architecture}")
    print(f"  Activation: {activation}")
    print(f"  Eval only: {eval_only}")
    print(f"  Resume from: {resume_from}")

    # Create run directory
    print("\n2. Creating run directory...")
    run_dir = make_run_dir(problem, architecture, activation)
    print(f"  Run directory: {run_dir}")

    # Save config to run directory
    import yaml
    config_path = run_dir / "config_used.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"  Config saved to {config_path}")

    # Determine checkpoint path
    checkpoint_path = None

    if not eval_only:
        # Training mode - train first, then run derivatives analysis
        print("\n3. Training mode - will train then run derivatives analysis")

        # Check datasets
        print("\n4. Checking datasets...")
        dataset_dir = Path("datasets") / problem
        train_data_path = dataset_dir / "training_data.pt"
        eval_data_path = dataset_dir / "eval_data.pt"

        if not train_data_path.exists() or not eval_data_path.exists():
            print("  Datasets not found. Generating...")
            from utils.dataset_gen import generate_and_save_datasets
            generate_and_save_datasets(config)
        else:
            print(f"  Datasets found:")
            print(f"    Train: {train_data_path}")
            print(f"    Eval: {eval_data_path}")

        # Build model
        print("\n5. Building model...")
        model = FCNet(architecture, activation, config)
        print(f"  Model created: {len(model.get_layer_names())} layers")

        # Load checkpoint if resume_from is specified
        if resume_from is not None:
            print(f"\n  Loading checkpoint from: {resume_from}")
            from run_ncc import fix_long_path
            resume_checkpoint_path = fix_long_path(Path(resume_from))

            if not resume_checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {resume_from}")

            # Load checkpoint (with legacy support)
            try:
                checkpoint = torch.load(resume_checkpoint_path,
                                        map_location='cpu')
            except Exception:
                print("  Warning: Standard load failed, trying legacy mode...")
                checkpoint = torch.load(resume_checkpoint_path,
                                        map_location='cpu',
                                        weights_only=False)
                print("  Legacy checkpoint loaded")

            # Extract state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Remap keys for legacy checkpoints
            remapped_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('layer_') or key.startswith('output.'):
                    if key.startswith('output.'):
                        layer_num = len(architecture) - 1
                        new_key = key.replace('output.',
                                              f'network.layer_{layer_num}.')
                    else:
                        new_key = f'network.{key}'
                    remapped_state_dict[new_key] = value
                else:
                    remapped_state_dict[key] = value

            # Load weights
            try:
                model.load_state_dict(remapped_state_dict)
                print("  Model weights loaded - continuing from checkpoint")
            except RuntimeError:
                print("  Warning: Remapped keys didn't match, trying original...")
                model.load_state_dict(state_dict)
                print("  Model weights loaded - continuing from checkpoint")

        # Build loss function
        print("\n6. Building loss function...")
        loss_module = importlib.import_module(f"losses.{problem}_loss")
        loss_fn = loss_module.build_loss(**config)
        print(f"  Loss function built for {problem}")

        # Train
        print("\n7. Training...")
        checkpoint_path = train(
            model=model,
            loss_fn=loss_fn,
            train_data_path=str(train_data_path),
            eval_data_path=str(eval_data_path),
            cfg=config,
            run_dir=run_dir
        )
        print(f"\n  Training complete")
        print(f"  Best checkpoint: {checkpoint_path}")

        # Load best checkpoint for derivatives analysis
        print(f"\n8. Loading best checkpoint for derivatives analysis...")
        from run_ncc import fix_long_path
        checkpoint_path = fix_long_path(Path(checkpoint_path))
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print("  Best model loaded")

    else:
        # Evaluation-only mode - load checkpoint and run derivatives analysis
        print("\n3. Evaluation-only mode - derivatives analysis only")
        
        if resume_from is None:
            raise ValueError("eval_only=True requires resume_from to be set.")
        
        from run_ncc import fix_long_path
        checkpoint_path = fix_long_path(Path(resume_from))
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
        
        print(f"  Using checkpoint: {checkpoint_path}")

        # Check datasets
        print("\n4. Checking datasets...")
        dataset_dir = Path("datasets") / problem
        train_data_path = dataset_dir / "training_data.pt"
        eval_data_path = dataset_dir / "eval_data.pt"

        if not train_data_path.exists() or not eval_data_path.exists():
            print("  Datasets not found. Generating...")
            from utils.dataset_gen import generate_and_save_datasets
            generate_and_save_datasets(config)
        else:
            print(f"  Datasets found:")
            print(f"    Train: {train_data_path}")
            print(f"    Eval: {eval_data_path}")

        # Build model
        print(f"\n5. Building model...")
        model = FCNet(architecture, activation, config)
        print(f"  Model created: {len(model.get_layer_names())} layers")

        # Load checkpoint
        print(f"  Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception:
            print("  Warning: Standard load failed, trying legacy mode...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu',
                                    weights_only=False)
            print("  Legacy checkpoint loaded")

        # Extract and remap state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        remapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('layer_') or key.startswith('output.'):
                if key.startswith('output.'):
                    layer_num = len(architecture) - 1
                    new_key = key.replace('output.',
                                          f'network.layer_{layer_num}.')
                else:
                    new_key = f'network.{key}'
                remapped_state_dict[new_key] = value
            else:
                remapped_state_dict[key] = value

        try:
            model.load_state_dict(remapped_state_dict)
            print("  Model weights loaded")
        except RuntimeError:
            print("  Warning: Remapped keys didn't match, trying original...")
            model.load_state_dict(state_dict)
            print("  Model weights loaded")

    # Run derivatives tracker analysis
    step_num = 9 if not eval_only else 6
    print(f"\n{step_num}. Running derivatives tracker analysis...")
    
    derivatives_metrics = run_derivatives_tracker(
        model=model,
        train_data_path=str(train_data_path),
        eval_data_path=str(eval_data_path),
        cfg=config,
        run_dir=run_dir
    )

    # Final summary
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"Output directory: {run_dir}")
    print(f"  - config_used.yaml")
    if not eval_only:
        print(f"  - metrics.json (training)")
        print(f"  - training_plots/")
        print(f"  - summary.txt")
    print(f"  - probe_plots/ (2 plots)")
    print(f"  - derivatives_plots/ (4 plot types: residual, magnitudes, heatmaps)")
    print("\nDerivatives Summary:")
    print(f"  Layers analyzed: {derivatives_metrics['layers_analyzed']}")
    print(f"  Final layer train residual: {derivatives_metrics['final_layer_train_residual']:.6e}")
    print(f"  Final layer eval residual: {derivatives_metrics['final_layer_eval_residual']:.6e}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

