"""Main training orchestrator for NCC-PINN framework."""

import sys
import torch
import importlib
from pathlib import Path

# Force UTF-8 stdout/stderr so unicode in logs doesn't crash on non-UTF-8
# Windows consoles (e.g. cp1255).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from utils.io import load_config, make_run_dir, resolve_experts_architecture, log_architectures
from utils.dataset_gen import generate_and_save_datasets
from models.fc_model import FCNet
from models.network_factory import create_network
from trainer.trainer import train
from utils.logging_config import setup_logging, get_logger


def main():
    """Orchestrate the complete training workflow."""
    # Initial setup logging (console only until run_dir is created)
    setup_logging(run_dir=None)
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("NCC-PINN Training Orchestrator")
    logger.info("=" * 60)

    # Load configuration
    logger.info("1. Loading configuration...")
    config = load_config()
    problem = config['problem']
    architecture = config['base_architecture']
    activation = config['activation']
    eval_only = config['eval_only']
    resume_from = config['resume_from']

    logger.info(f"  Problem: {problem}")
    log_architectures(config, logger=logger)
    logger.info(f"  Activation: {activation}")
    logger.info(f"  Eval only: {eval_only}")
    logger.info(f"  Resume from: {resume_from}")

    # Create run directory
    logger.info("2. Creating run directory...")
    run_dir = make_run_dir(
        problem, architecture, activation,
        experts_layers=resolve_experts_architecture(config),
    )
    logger.info(f"  Run directory: {run_dir}")
    
    # Now set up logging to write to the run directory
    from utils.logging_config import update_log_file
    update_log_file(run_dir)
    logger.info(f"  Logging redirected to: {run_dir / 'training_logs.log'}")

    # Save config to run directory
    import yaml
    config_path = run_dir / "config_used.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"  Config saved to {config_path}")

    # Generate datasets if missing
    logger.info("3. Checking datasets...")
    dataset_dir = Path("datasets") / problem
    train_data_path = dataset_dir / "training_data.pt"
    eval_data_path = dataset_dir / "eval_data.pt"

    if not train_data_path.exists() or not eval_data_path.exists():
        logger.info("  Datasets not found. Generating...")
        generate_and_save_datasets(config)
    else:
        logger.info(f"  Datasets found:")
        logger.info(f"    Train: {train_data_path}")
        logger.info(f"    Eval: {eval_data_path}")

    # Build model
    logger.info("4. Building model...")
    device = torch.device('cuda' if config['cuda'] and
                          torch.cuda.is_available() else 'cpu')
    
    adaptive_cfg = config.get('adaptive_pinn', {})
    is_adaptive = adaptive_cfg.get('enabled', False)
    model_type = config.get('model', 'AToE')
    
    if is_adaptive:
        experts_arch = config.get('experts_architecture', architecture)
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
        logger.info(f"  Model type: {type(model).__name__}")
        logger.info(f"  Base layers: {len(model.get_layer_names())}")
        logger.info(f"    Max experts: {adaptive_cfg.get('max_experts', 5)}")
        logger.info(f"    Blending mode: {adaptive_cfg.get('blending_mode', 'hard')}")
    else:
        expert_type = adaptive_cfg.get('expert_type', 'mlp')
        model = create_network(architecture, activation, config,
                               is_base=True, expert_type=expert_type)
        logger.info(f"  Model created: {len(model.get_layer_names())} layers")
    logger.info(f"  Device: {device}")

    # Build loss function
    logger.info("5. Building loss function...")
    loss_module = importlib.import_module(f"losses.{problem}_loss")
    loss_fn = loss_module.build_loss(**config)
    logger.info(f"  Loss function built for {problem}")

    # Handle resume_from checkpoint
    start_epoch = 0
    optimizer = None

    if resume_from is not None:
        logger.info(f"6. Loading checkpoint from: {resume_from}")
        checkpoint_path = Path(resume_from)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if is_adaptive and checkpoint.get('is_adaptive', False):
            model.load_state_dict_extended(checkpoint['adaptive_state'])
            logger.info(f"  Adaptive model weights loaded ({model.num_experts} experts)")
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"  Model weights loaded")

        # Load optimizer state for fine-tuning
        if not eval_only:
            optimizer_state = checkpoint['optimizer_state_dict']
            start_epoch = checkpoint['epoch']
            logger.info(f"  Optimizer state loaded (resuming from epoch {start_epoch})")
        else:
            logger.info(f"  Model loaded for evaluation only")

        # Log checkpoint info
        if 'train_loss' in checkpoint:
            logger.info(f"  Checkpoint info:")
            logger.info(f"    Epoch: {checkpoint['epoch']}")
            logger.info(f"    Train loss: {checkpoint['train_loss']:.6f}")
            logger.info(f"    Eval loss: {checkpoint['eval_loss']:.6f}")
    else:
        logger.info("6. No checkpoint to resume from")

    # Training or evaluation
    if eval_only:
        logger.info("7. Evaluation-only mode")

        if resume_from is None:
            raise ValueError(
                "eval_only=True requires resume_from to be set"
            )

        # Run evaluation
        logger.info("  Running evaluation on eval dataset...")
        model = model.to(device)
        model.eval()

        # Load eval data
        eval_data = torch.load(eval_data_path)

        # Move to device
        eval_data_device = {
            'x': eval_data['x'].to(device),
            't': eval_data['t'].to(device),
            'h_gt': eval_data['h_gt'].to(device),
            'mask': {
                'residual': eval_data['mask']['residual'].to(device),
                'IC': eval_data['mask']['IC'].to(device),
                'BC': eval_data['mask']['BC'].to(device)
            }
        }

        with torch.no_grad():
            # Compute loss
            eval_loss = loss_fn(model, eval_data_device)

            # Compute relative L2
            from trainer.utils import compute_relative_l2_error
            inputs = torch.cat([eval_data_device['x'], eval_data_device['t']],
                             dim=1)
            h_pred = model(inputs)
            eval_rel_l2 = compute_relative_l2_error(
                h_pred,
                eval_data_device['h_gt']
            )

        logger.info(f"  Eval loss: {eval_loss.item():.6f}")
        logger.info(f"  Eval relative L2: {eval_rel_l2.item():.6f}")

        # Save metrics
        import json
        metrics = {
            'eval_loss': eval_loss.item(),
            'eval_rel_l2': eval_rel_l2.item(),
            'checkpoint_loaded': resume_from
        }

        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"  Metrics saved to {metrics_path}")

        # Generate evaluation plots
        from trainer.plotting import plot_final_comparison
        training_plots_dir = run_dir / "training_plots"

        plot_final_comparison(
            h_pred.cpu().numpy(),
            eval_data_device['h_gt'].cpu().numpy(),
            eval_data_device['x'].cpu().numpy(),
            eval_data_device['t'].cpu().numpy(),
            training_plots_dir
        )

        logger.info(f"  Evaluation complete")

    else:
        logger.info("7. Training mode")

        # Create optimizer
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
            logger.info(f"  Optimizer created (Adam, lr={config['lr']})")
        else:
            # Reconstruct optimizer with loaded state
            optimizer_new = torch.optim.Adam(model.parameters(),
                                           lr=config['lr'])
            optimizer_new.load_state_dict(optimizer_state)
            optimizer = optimizer_new
            logger.info(f"  Optimizer resumed")

        # Adjust epochs if resuming
        if start_epoch > 0:
            remaining_epochs = config['epochs'] - start_epoch
            logger.info(f"  Continuing for {remaining_epochs} more epochs " +
                  f"(total: {config['epochs']})")

        # Call trainer
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

    # Final summary
    logger.info("=" * 60)
    logger.info("Run complete!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {run_dir}")
    logger.info(f"  - config_used.yaml")
    logger.info(f"  - metrics.json")
    logger.info(f"  - training_plots/")
    if not eval_only:
        logger.info(f"  - summary.txt")
        logger.info(f"Best checkpoint saved to: checkpoints/{problem}/")
    logger.info("=" * 60)
    
    # Close logging
    from utils.logging_config import close_logging
    close_logging()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
