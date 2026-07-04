"""NCC analysis orchestrator for trained models."""

import sys
import os
import torch
import importlib
from pathlib import Path

# Windows consoles default to a non-UTF-8 code page (e.g. cp1255 on Hebrew
# Windows), which raises UnicodeEncodeError when the trainer prints unicode
# (->, Sigma, psi). Force UTF-8 stdout/stderr so logging never crashes a run.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from utils.io import load_config, make_run_dir, resolve_experts_architecture, log_architectures
from utils.config_validation import (
    validate_problem_config,
    merge_problem_features_to_toplevel,
)
from utils.logging_config import setup_logging, get_logger, update_log_file, close_logging
from models.fc_model import FCNet
from models.network_factory import create_network
from ncc.ncc_runner import run_ncc
from trainer.trainer import train


def fix_long_path(path):
    r"""
    Fix Windows long path issues by adding \\?\ prefix when needed.
    Only applies on Windows for absolute paths longer than 260 chars.
    """
    if sys.platform != 'win32':
        return path
    
    # Convert to Path object if string
    if isinstance(path, str):
        path = Path(path)
    
    # Get absolute path
    abs_path = path.resolve()
    abs_path_str = str(abs_path)
    
    # If path is too long and doesn't already have the prefix
    if len(abs_path_str) > 260 and not abs_path_str.startswith('\\\\?\\'):
        return Path(f'\\\\?\\{abs_path_str}')
    
    return abs_path


def run_multi_eval(checkpoints_dict, config, run_dir):
    """
    Run NCC analysis on multiple checkpoints and generate comparison plots.
    
    Args:
        checkpoints_dict: Dict of {model_name: checkpoint_path}
        config: Configuration dict
        run_dir: Directory to save results
    """
    logger = get_logger(__name__)
    logger.info(f"Running NCC analysis on {len(checkpoints_dict)} checkpoints...")
    
    # Storage for aggregated results
    ncc_data = {}
    
    problem = config['problem']
    ncc_data_path = Path("datasets") / problem / "ncc_data.pt"
    
    # Process each checkpoint
    for model_idx, (model_name, checkpoint_path) in enumerate(checkpoints_dict.items(), 1):
        logger.info(f"[{model_idx}/{len(checkpoints_dict)}] Processing {model_name}...")
        logger.info(f"  Checkpoint: {checkpoint_path}")
        
        checkpoint_path = fix_long_path(Path(checkpoint_path))
        if not checkpoint_path.exists():
            logger.error(f"  ERROR: Checkpoint not found, skipping...")
            continue
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception:
            logger.warning("  Warning: Standard load failed, trying legacy mode...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            logger.info("  Legacy checkpoint loaded")
        
        # Extract architecture and activation from checkpoint config if available
        if 'config' in checkpoint:
            checkpoint_config = checkpoint['config']
            architecture = checkpoint_config.get('base_architecture',
                            checkpoint_config.get('architecture', config['base_architecture']))
            activation = checkpoint_config.get('activation', config['activation'])
            logger.info(f"  Loaded architecture from checkpoint: {architecture}")
        elif 'architecture' in checkpoint:
            architecture = checkpoint['architecture']
            activation = checkpoint.get('activation', config['activation'])
            logger.info(f"  Loaded architecture from checkpoint: {architecture}")
        else:
            # Fallback: use from config (might not match)
            architecture = config['base_architecture']
            activation = config['activation']
            logger.warning(f"  Warning: Architecture not in checkpoint, using config: {architecture}")
        
        # Build model
        model_config = config.copy()
        model_config['base_architecture'] = architecture
        model_config['activation'] = activation
        et = config.get('adaptive_pinn', {}).get('expert_type', 'mlp')
        model = create_network(architecture, activation, model_config,
                               is_base=True, expert_type=et)
        
        # Load model weights
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
                    new_key = key.replace('output.', f'network.layer_{layer_num}.')
                else:
                    new_key = f'network.{key}'
                remapped_state_dict[new_key] = value
            else:
                remapped_state_dict[key] = value
        
        # Load weights
        try:
            model.load_state_dict(remapped_state_dict)
            logger.info("  Model weights loaded")
        except RuntimeError:
            logger.warning("  Warning: Remapped keys didn't match, trying original...")
            model.load_state_dict(state_dict)
            logger.info("  Model weights loaded")
        
        # Run NCC analysis (without saving plots, just get metrics)
        logger.info(f"  Running NCC analysis...")
        ncc_metrics = run_ncc(
            model=model,
            eval_data_path=str(ncc_data_path),
            cfg=model_config,
            run_dir=None,  # Don't save per-model plots
            suppress_plots=True  # Add flag to suppress individual plots
        )
        
        # Store results in format expected by comparison plots
        # Structure: {model_name: {epoch: ncc_metrics}}
        ncc_data[model_name] = {
            'final': ncc_metrics
        }
        
        logger.info(f"  Complete - Accuracy: {ncc_metrics['layer_accuracies']}")
    
    # Generate comparison plots
    logger.info("=" * 60)
    logger.info("Generating comparison plots...")
    logger.info("=" * 60)
    
    from utils.comparison_plots import generate_ncc_comparison_plots_only
    generate_ncc_comparison_plots_only(run_dir, ncc_data)
    
    # Save summary
    import yaml
    summary_path = run_dir / "multi_eval_summary.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump(ncc_data, f, default_flow_style=False)
    logger.info(f"Summary saved to {summary_path}")
    
    # Final summary
    logger.info("=" * 60)
    logger.info("Multi-Evaluation Complete!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {run_dir}")
    logger.info(f"  - ncc_classification_comparison.png")
    logger.info(f"  - ncc_compactness_comparison.png")
    logger.info(f"  - multi_eval_summary.yaml")
    logger.info("Models evaluated:")
    for model_name in ncc_data.keys():
        metrics = ncc_data[model_name]['final']
        logger.info(f"  {model_name}:")
        logger.info(f"    Layers: {metrics['layers_analyzed']}")
        for layer, acc in metrics['layer_accuracies'].items():
            logger.info(f"      {layer}: {acc:.4f}")
    logger.info("=" * 60)


def main():
    """Orchestrate NCC analysis workflow."""
    # Initialize logging (console only until run_dir is created)
    setup_logging(run_dir=None)
    logger = get_logger(__name__)
    
    print("=" * 60)
    print("NCC Analysis Orchestrator")
    print("=" * 60)

    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config()
    
    # Validate per-problem config (all features must be explicitly specified)
    validate_problem_config(config)
    
    # Copy per-problem features to top-level for backward compatibility with
    # model creation code that reads config['fourier_features'], config['rwf'], etc.
    merge_problem_features_to_toplevel(config)
    
    problem = config['problem']
    architecture = config['base_architecture']
    activation = config['activation']
    eval_only = config['eval_only']
    resume_from = config['resume_from']
    
    # Detect multi-eval mode
    is_multi_eval = isinstance(resume_from, dict)

    print(f"  Problem: {problem}")
    log_architectures(config)
    print(f"  Activation: {activation}")
    print(f"  Eval only: {eval_only}")
    if is_multi_eval:
        print(f"  Multi-eval mode: {len(resume_from)} checkpoints")
        for name, path in resume_from.items():
            print(f"    - {name}: {path}")
    else:
        print(f"  Resume from: {resume_from}")

    # Create run directory
    print("\n2. Creating run directory...")
    if is_multi_eval:
        # Multi-eval mode - special directory naming
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("outputs") / "multi_eval_comparison" / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Single model mode - standard naming
        run_dir = make_run_dir(
            problem, architecture, activation,
            experts_layers=resolve_experts_architecture(config),
        )
        # Record run_dir for run_experiments.py to find without mtime search
        _run_dir_record = Path("outputs") / ".last_run_dir.txt"
        _run_dir_record.parent.mkdir(parents=True, exist_ok=True)
        _run_dir_record.write_text(str(run_dir.resolve()))
    print(f"  Run directory: {run_dir}")
    
    # Now set up file logging to the run directory
    update_log_file(run_dir)
    logger.info(f"Logging initialized. Log file: {run_dir / 'training_logs.log'}")

    # Save config to run directory
    import yaml
    from utils.io import get_git_info
    config['git'] = get_git_info()
    config_path = run_dir / "config_used.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"  Config saved to {config_path}")

    # Determine checkpoint path
    checkpoint_path = None

    if is_multi_eval:
        # Multi-eval mode - evaluate multiple checkpoints and compare
        logger.info("3. Multi-evaluation mode - comparing multiple checkpoints")
        
        if not eval_only:
            raise ValueError(
                "Multi-evaluation mode (dict resume_from) requires eval_only=True. "
                "Cannot train with multiple checkpoints."
            )
        
        # Check NCC dataset
        logger.info("4. Checking NCC dataset...")
        dataset_dir = Path("datasets") / problem
        ncc_data_path = dataset_dir / "ncc_data.pt"

        if not ncc_data_path.exists():
            logger.info("  NCC data not found. Generating...")
            from utils.dataset_gen import generate_and_save_datasets
            generate_and_save_datasets(config)
        else:
            logger.info(f"  NCC data found: {ncc_data_path}")
        
        # Run multi-eval
        run_multi_eval(resume_from, config, run_dir)
        
        # Exit early - multi-eval handles everything
        close_logging()
        return
    
    elif not eval_only:
        # Training mode - train first, then run NCC on best checkpoint
        logger.info("3. Training mode - will train then run NCC analysis")

        # Check datasets
        logger.info("4. Checking datasets...")
        dataset_dir = Path("datasets") / problem
        train_data_path = dataset_dir / "training_data.pt"
        eval_data_path = dataset_dir / "eval_data.pt"
        ncc_data_path = dataset_dir / "ncc_data.pt"

        if not train_data_path.exists() or not eval_data_path.exists() or not ncc_data_path.exists():
            logger.info("  Datasets not found. Generating...")
            from utils.dataset_gen import generate_and_save_datasets
            generate_and_save_datasets(config)
        else:
            logger.info(f"  Datasets found:")
            logger.info(f"    Train: {train_data_path}")
            logger.info(f"    Eval: {eval_data_path}")
            logger.info(f"    NCC: {ncc_data_path}")

        # Check if adaptive PINN is enabled
        adaptive_cfg = config.get('adaptive_pinn', {})
        is_adaptive = adaptive_cfg.get('enabled', False)
        
        # Handle precision configuration (float32 or float64)
        precision = config.get('precision', 'float32')
        if precision == 'float64':
            torch.set_default_dtype(torch.float64)
            logger.info(f"  [Precision] Using float64 (double precision)")
        else:
            torch.set_default_dtype(torch.float32)
        
        # Check if time marching is enabled for this problem
        tm_cfg = config.get(problem, {}).get('time_marching', {})
        use_time_marching = tm_cfg.get('enabled', False)
        
        if use_time_marching:
            # Time marching mode - train separate models for each window
            logger.info("5. Time marching mode enabled")
            logger.info(f"  Windows: {tm_cfg.get('num_windows', 5)}")
            logger.info(f"  M distribution: {tm_cfg.get('m_distribution', 'quadratic')}")
            logger.info(f"  Freeze previous: {tm_cfg.get('freeze_previous_windows', True)}")
            
            if not is_adaptive:
                raise ValueError(
                    "Time marching requires adaptive PINN (adaptive_pinn.enabled=true). "
                    "Time marching trains AToE/ANT models for each window."
                )
            
            # Determine model class
            model_type = config.get('model', 'AToE')
            if model_type == 'ANT':
                from models.ant import ANT
                model_class = ANT
            elif model_type == 'AToELeaves':
                from models.atoe_leaves import AToELeaves
                model_class = AToELeaves
            else:
                from models.atoe import AToE
                model_class = AToE
            
            # Get device
            cuda_available = config.get('cuda', True) and torch.cuda.is_available()
            device = torch.device('cuda' if cuda_available else 'cpu')
            
            # Train with time marching
            from trainer.time_marching import train_with_time_marching
            model, checkpoint_path = train_with_time_marching(
                model_class=model_class,
                architecture=architecture,
                activation=activation,
                config=config,
                adaptive_cfg=adaptive_cfg,
                run_dir=run_dir,
                device=device,
            )
            
            logger.info(f"  Time marching training complete")
            logger.info(f"  Combined model: {model.num_windows} windows, {model.total_experts} total experts")
            logger.info(f"  Last checkpoint: {checkpoint_path}")
        
        else:
            # Standard training mode (no time marching)
            # Build model
            logger.info("5. Building model...")
            
            if is_adaptive:
                model_type = config.get('model', 'AToE')
                experts_arch = resolve_experts_architecture(config)
                if model_type == 'ANT':
                    from models.ant import ANT
                    model = ANT(architecture, activation, config, adaptive_cfg)
                elif model_type == 'AToELeaves':
                    from models.atoe_leaves import AToELeaves
                    model = AToELeaves(
                        architecture, activation, config, adaptive_cfg,
                        experts_architecture=experts_arch,
                    )
                else:
                    from models.atoe import AToE
                    model = AToE(
                        architecture, activation, config, adaptive_cfg,
                        experts_architecture=experts_arch,
                    )
                logger.info(f"  {type(model).__name__} created: {len(model.get_layer_names())} base layers")
            else:
                expert_type = adaptive_cfg.get('expert_type', 'mlp')
                model = create_network(architecture, activation, config,
                                       is_base=True, expert_type=expert_type)
                logger.info(f"  Model created: {len(model.get_layer_names())} layers")

            # Convert model to double precision if configured
            if precision == 'float64':
                model = model.double()

            # Load checkpoint if resume_from is specified
            if resume_from is not None:
                logger.info(f"  Loading checkpoint from: {resume_from}")
                resume_checkpoint_path = fix_long_path(Path(resume_from))

                if not resume_checkpoint_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {resume_from}")

                # Load checkpoint (with legacy support)
                try:
                    checkpoint = torch.load(resume_checkpoint_path,
                                            map_location='cpu')
                except Exception:
                    logger.warning("  Warning: Standard load failed, trying legacy mode...")
                    checkpoint = torch.load(resume_checkpoint_path,
                                            map_location='cpu',
                                            weights_only=False)
                    logger.info("  Legacy checkpoint loaded")

                # Extract state dict
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint

                # Check if this is an adaptive checkpoint
                if is_adaptive and checkpoint.get('is_adaptive', False):
                    # Load adaptive state including experts and regions
                    model.load_state_dict_extended(checkpoint['adaptive_state'])
                    logger.info(f"  Adaptive model weights loaded ({model.num_experts} experts)")
                else:
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
                        logger.info("  Model weights loaded - continuing from checkpoint")
                    except RuntimeError:
                        logger.warning("  Warning: Remapped keys didn't match, trying original...")
                        model.load_state_dict(state_dict)
                    logger.info("  Model weights loaded - continuing from checkpoint")

            # Build loss
            logger.info("6. Building loss function...")
            loss_module = importlib.import_module(f"losses.{problem}_loss")
            loss_fn = loss_module.build_loss(**config)
            logger.info(f"  Loss function built for {problem}")

            # Train
            logger.info("7. Training...")
            checkpoint_path = train(
                model=model,
                loss_fn=loss_fn,
                train_data_path=str(train_data_path),
                eval_data_path=str(eval_data_path),
                cfg=config,
                run_dir=run_dir
            )

            logger.info(f"  Training complete")
            logger.info(f"  Best checkpoint: {checkpoint_path}")

    else:
        # Eval-only mode - require resume_from
        logger.info("3. Evaluation-only mode - NCC analysis only")

        if resume_from is None:
            raise ValueError(
                "eval_only=True requires resume_from to be set. "
                "Please specify the path to a trained model checkpoint."
            )

        checkpoint_path = fix_long_path(Path(resume_from))
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {resume_from}"
            )

        logger.info(f"  Using checkpoint: {checkpoint_path}")

        # Check NCC dataset
        logger.info("4. Checking NCC dataset...")
        dataset_dir = Path("datasets") / problem
        ncc_data_path = dataset_dir / "ncc_data.pt"

        if not ncc_data_path.exists():
            logger.info("  NCC data not found. Generating...")
            from utils.dataset_gen import generate_and_save_datasets
            generate_and_save_datasets(config)
        else:
            logger.info(f"  NCC data found: {ncc_data_path}")

    # Run NCC analysis
        adaptive_cfg = config.get('adaptive_pinn', {})
        inner_metrics_disabled = False
        if adaptive_cfg.get('enabled', False):
            inner_metrics_disabled = not adaptive_cfg.get('inner_metrics_calculation', True)

        if inner_metrics_disabled:
            logger.info("[Skipping NCC analysis: inner_metrics_calculation is False]")
        else:
            logger.info(f"{'8' if not eval_only else '5'}. Running NCC analysis...")

            # Load checkpoint
            logger.info(f"  Loading checkpoint: {checkpoint_path}")
            try:
                # Try loading with default settings (weights_only=True in PyTorch 2.6+)
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            except Exception:
                # Fallback for legacy checkpoints with custom classes
                logger.warning("  Warning: Standard load failed, trying legacy mode...")
                checkpoint = torch.load(checkpoint_path, map_location='cpu',
                                        weights_only=False)
                logger.info("  Legacy checkpoint loaded")

            # Build model - check for adaptive PINN
            is_adaptive = adaptive_cfg.get('enabled', False) or checkpoint.get('is_adaptive', False)
            if is_adaptive:
                model_type = config.get('model', 'AToE')
                experts_arch = resolve_experts_architecture(config)
                if model_type == 'ANT':
                    from models.ant import ANT
                    model = ANT(architecture, activation, config, adaptive_cfg)
                elif model_type == 'AToELeaves':
                    from models.atoe_leaves import AToELeaves
                    model = AToELeaves(
                        architecture, activation, config, adaptive_cfg,
                        experts_architecture=experts_arch,
                    )
                else:
                    from models.atoe import AToE
                    model = AToE(
                        architecture, activation, config, adaptive_cfg,
                        experts_architecture=experts_arch,
                    )
            else:
                et = adaptive_cfg.get('expert_type', 'mlp')
                model = create_network(architecture, activation, config,
                                       is_base=True, expert_type=et)

            # Load model weights - handle different checkpoint formats
            if is_adaptive and checkpoint.get('is_adaptive', False):
                # Load adaptive state including experts and regions
                model.load_state_dict_extended(checkpoint['adaptive_state'])
                logger.info(f"  Adaptive model weights loaded ({model.num_experts} experts)")
            else:
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    # Checkpoint might be just the state dict itself
                    state_dict = checkpoint

                # Remap keys for legacy checkpoints with different layer naming
                remapped_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('layer_') or key.startswith('output.'):
                        if key.startswith('output.'):
                            layer_num = len(architecture) - 1
                            new_key = key.replace('output.', f'network.layer_{layer_num}.')
                        else:
                            new_key = f'network.{key}'
                        remapped_state_dict[new_key] = value
                    else:
                        remapped_state_dict[key] = value

                try:
                    model.load_state_dict(remapped_state_dict)
                    logger.info("  Model weights loaded")
                except RuntimeError:
                    logger.warning("  Warning: Remapped keys didn't match, trying original...")
                    model.load_state_dict(state_dict)
                    logger.info("  Model weights loaded")

            # Get NCC data path
            ncc_data_path = Path("datasets") / problem / "ncc_data.pt"

            # Run NCC
            ncc_metrics = run_ncc(
                model=model,
                eval_data_path=str(ncc_data_path),  # Using stratified NCC dataset
                cfg=config,
                run_dir=run_dir
            )

            # If there's a history from training, regenerate shaded plots
            if not eval_only:
                metrics_path = run_dir / "metrics.json"
                if metrics_path.exists():
                    import json
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    if 'ncc_history' in metrics and metrics['ncc_history']:
                        from ncc.ncc_plotting import plot_ncc_history_shaded
                        final_epoch = config.get('epochs', 0)
                        metrics['ncc_history'].append((final_epoch, ncc_metrics))
                        history = [(epoch, mets) for epoch, mets in metrics['ncc_history']]
                        plot_ncc_history_shaded(history, run_dir / "ncc_plots")
                        logger.info(f"  Shaded NCC plots generated from {len(history)} epochs")

            # In eval-only mode, also run problem-specific evaluation visualization
            if eval_only:
                logger.info("Generating problem-specific evaluation visualizations...")

                eval_data_path = Path("datasets") / problem / "eval_data.pt"
                if not eval_data_path.exists():
                    logger.info("  Eval data not found. Generating...")
                    from utils.dataset_gen import generate_and_save_datasets
                    generate_and_save_datasets(config)

                try:
                    from utils.problem_specific import get_visualization_module
                    viz_module = get_visualization_module(problem)
                    visualize_evaluation = viz_module[1]
                    visualize_evaluation(model, str(eval_data_path), run_dir, config)
                except ValueError:
                    logger.info(f"  (No custom evaluation visualization for {problem})")
                except Exception as e:
                    logger.warning(f"  Warning: Could not generate evaluation visualization: {e}")

            logger.info("=" * 60)
            logger.info("Complete!")
            logger.info("=" * 60)
            logger.info(f"Output directory: {run_dir}")
            logger.info(f"  - config_used.yaml")
            if not eval_only:
                logger.info(f"  - metrics.json (training)")
                logger.info(f"  - training_plots/")
                logger.info(f"  - summary.txt")
            logger.info(f"  - ncc_plots/ (5 plots)")
            logger.info(f"  - ncc_metrics.json")
            logger.info("NCC Summary:")
            logger.info(f"  Classes: {ncc_metrics['num_classes']}")
            logger.info(f"  Layers analyzed: {ncc_metrics['layers_analyzed']}")
            logger.info(f"  Layer accuracies:")
            for layer, acc in ncc_metrics['layer_accuracies'].items():
                logger.info(f"    {layer}: {acc:.4f}")
            logger.info("=" * 60)
    
    # Clean up logging
    close_logging()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        close_logging()
        sys.exit(1)

