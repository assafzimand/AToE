"""Training orchestrator: builds the model and loss for the configured problem
and runs the AToE-Leaves training flow (root -> M-term tree -> leaf experts ->
joint fine-tune), optionally wrapped in time-marching windows."""

import sys
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
from models.network_factory import create_network
from trainer.trainer import train


def fix_long_path(path):
    r"""Add the Windows \\?\ prefix to absolute paths longer than 260 chars."""
    if sys.platform != 'win32':
        return path

    if isinstance(path, str):
        path = Path(path)

    abs_path = path.resolve()
    abs_path_str = str(abs_path)

    if len(abs_path_str) > 260 and not abs_path_str.startswith('\\\\?\\'):
        return Path(f'\\\\?\\{abs_path_str}')

    return abs_path


def _build_model(config, adaptive_cfg, architecture, activation, logger):
    """Build the model configured by `model` / `adaptive_pinn`."""
    is_adaptive = adaptive_cfg.get('enabled', False)
    if is_adaptive:
        from models.atoe_leaves import AToELeaves
        experts_arch = resolve_experts_architecture(config)
        model = AToELeaves(
            architecture, activation, config, adaptive_cfg,
            experts_architecture=experts_arch,
        )
        logger.info(f"  {type(model).__name__} created: {len(model.get_layer_names())} base layers")
    else:
        expert_type = adaptive_cfg.get('expert_type', 'mlp')
        model = create_network(architecture, activation, config,
                               is_base=True, expert_type=expert_type)
        logger.info(f"  Model created: {len(model.get_layer_names())} layers")
    return model


def _load_checkpoint_into_model(model, checkpoint_path, architecture, is_adaptive, logger):
    """Load a checkpoint file into a freshly built model.

    Supports plain state dicts, {'model_state_dict': ...} wrappers, adaptive
    checkpoints ({'is_adaptive': True, 'adaptive_state': ...}), and legacy key
    layouts (layer_* / output.* remapped to network.layer_*).
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception:
        logger.warning("  Warning: Standard load failed, trying legacy mode...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        logger.info("  Legacy checkpoint loaded")

    if is_adaptive and checkpoint.get('is_adaptive', False):
        model.load_state_dict_extended(checkpoint['adaptive_state'])
        logger.info(f"  Adaptive model weights loaded ({model.num_experts} experts)")
        return

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
                new_key = key.replace('output.', f'network.layer_{layer_num}.')
            else:
                new_key = f'network.{key}'
            remapped_state_dict[new_key] = value
        else:
            remapped_state_dict[key] = value

    try:
        model.load_state_dict(remapped_state_dict)
    except RuntimeError:
        logger.warning("  Warning: Remapped keys didn't match, trying original...")
        model.load_state_dict(state_dict)
    logger.info("  Model weights loaded")


def main():
    """Orchestrate the training (or eval-only visualization) workflow."""
    # Initialize logging (console only until run_dir is created)
    setup_logging(run_dir=None)
    logger = get_logger(__name__)

    print("=" * 60)
    print("AToE Training Orchestrator")
    print("=" * 60)

    print("\n1. Loading configuration...")
    config = load_config()

    # Validate per-problem config (all features must be explicitly specified)
    validate_problem_config(config)

    # Copy per-problem features to top-level for model creation code that
    # reads config['fourier_features'], config['rwf'], etc.
    merge_problem_features_to_toplevel(config)

    problem = config['problem']
    architecture = config['base_architecture']
    activation = config['activation']
    eval_only = config['eval_only']
    resume_from = config['resume_from']

    print(f"  Problem: {problem}")
    log_architectures(config)
    print(f"  Activation: {activation}")
    print(f"  Eval only: {eval_only}")
    print(f"  Resume from: {resume_from}")

    print("\n2. Creating run directory...")
    run_dir = make_run_dir(
        problem, architecture, activation,
        experts_layers=resolve_experts_architecture(config),
    )
    # Record run_dir for run_experiments.py to find without mtime search
    _run_dir_record = Path("outputs") / ".last_run_dir.txt"
    _run_dir_record.parent.mkdir(parents=True, exist_ok=True)
    _run_dir_record.write_text(str(run_dir.resolve()))
    print(f"  Run directory: {run_dir}")

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

    adaptive_cfg = config.get('adaptive_pinn', {})
    is_adaptive = adaptive_cfg.get('enabled', False)

    # Precision (float32 or float64)
    precision = config.get('precision', 'float32')
    if precision == 'float64':
        torch.set_default_dtype(torch.float64)
        logger.info(f"  [Precision] Using float64 (double precision)")
    else:
        torch.set_default_dtype(torch.float32)

    if not eval_only:
        logger.info("3. Training mode")

        # Ensure datasets exist
        logger.info("4. Checking datasets...")
        dataset_dir = Path("datasets") / problem
        train_data_path = dataset_dir / "training_data.pt"
        eval_data_path = dataset_dir / "eval_data.pt"

        if not train_data_path.exists() or not eval_data_path.exists():
            logger.info("  Datasets not found. Generating...")
            from utils.dataset_gen import generate_and_save_datasets
            generate_and_save_datasets(config)
        else:
            logger.info(f"  Datasets found:")
            logger.info(f"    Train: {train_data_path}")
            logger.info(f"    Eval: {eval_data_path}")

        # Time-marching: separate models trained per temporal window
        tm_cfg = config.get(problem, {}).get('time_marching', {})
        use_time_marching = tm_cfg.get('enabled', False)

        if use_time_marching:
            logger.info("5. Time marching mode enabled")
            logger.info(f"  Windows: {tm_cfg.get('num_windows', 5)}")
            logger.info(f"  M distribution: {tm_cfg.get('m_distribution', 'quadratic')}")
            logger.info(f"  Freeze previous: {tm_cfg.get('freeze_previous_windows', True)}")

            if not is_adaptive:
                raise ValueError(
                    "Time marching requires adaptive PINN (adaptive_pinn.enabled=true)."
                )

            from models.atoe_leaves import AToELeaves

            cuda_available = config.get('cuda', True) and torch.cuda.is_available()
            device = torch.device('cuda' if cuda_available else 'cpu')

            from trainer.time_marching import train_with_time_marching
            model, checkpoint_path = train_with_time_marching(
                model_class=AToELeaves,
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
            logger.info("5. Building model...")
            model = _build_model(config, adaptive_cfg, architecture, activation, logger)

            if precision == 'float64':
                model = model.double()

            if resume_from is not None:
                logger.info(f"  Loading checkpoint from: {resume_from}")
                resume_checkpoint_path = fix_long_path(Path(resume_from))
                if not resume_checkpoint_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
                _load_checkpoint_into_model(
                    model, resume_checkpoint_path, architecture, is_adaptive, logger)

            logger.info("6. Building loss function...")
            loss_module = importlib.import_module(f"losses.{problem}_loss")
            loss_fn = loss_module.build_loss(**config)
            logger.info(f"  Loss function built for {problem}")

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
        # Eval-only: load a checkpoint and produce evaluation visualizations
        logger.info("3. Evaluation-only mode")

        if resume_from is None:
            raise ValueError(
                "eval_only=True requires resume_from to be set. "
                "Please specify the path to a trained model checkpoint."
            )

        checkpoint_path = fix_long_path(Path(resume_from))
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
        logger.info(f"  Using checkpoint: {checkpoint_path}")

        model = _build_model(config, adaptive_cfg, architecture, activation, logger)
        if precision == 'float64':
            model = model.double()
        _load_checkpoint_into_model(model, checkpoint_path, architecture, is_adaptive, logger)

        eval_data_path = Path("datasets") / problem / "eval_data.pt"
        if not eval_data_path.exists():
            logger.info("  Eval data not found. Generating...")
            from utils.dataset_gen import generate_and_save_datasets
            generate_and_save_datasets(config)

        logger.info("Generating problem-specific evaluation visualizations...")
        try:
            from utils.problem_specific import get_visualization_module
            viz_module = get_visualization_module(problem)
            visualize_evaluation = viz_module[1]
            visualize_evaluation(model, str(eval_data_path), run_dir, config)
        except ValueError:
            logger.info(f"  (No custom evaluation visualization for {problem})")
        except Exception as e:
            logger.warning(f"  Warning: Could not generate evaluation visualization: {e}")

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
