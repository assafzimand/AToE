"""Training loop for PINN models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, Callable, Tuple
import json
import math
import time
import copy
import numpy as np

from utils.logging_config import get_logger

logger = get_logger(__name__)

from trainer.plotting import (
    plot_training_curves, plot_final_comparison,
    plot_per_expert_curves,
)
from trainer.utils import compute_infinity_norm_error
from trainer.timing import EpochTimer
from trainer.training_context import TrainingContext, SegmentResult
from models.atoe_leaves import AToELeaves
from utils.dataset_gen import (
    regenerate_training_data,
    resample_residual_inplace,
    _save_adaptive_sampling_heatmap,
)
from utils.dataset_plotting import save_spawn_prediction_plot
from utils.config_validation import (
    validate_problem_config,
    validate_adaptive_staged_config,
)
from losses.causal_weighting import advance_causal_schedule, create_causal_state
from losses.lra import LRAWeights
import losses.ks_loss as _ks_loss_module
from losses.split_loss import build_split_loss
from adaptive.subdomain_data import build_subdomain_data, KIND_NAMES


def _override_ic_for_time_marching(
    train_data: Dict[str, torch.Tensor],
    cfg: Dict,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Override IC h_gt values with previous model predictions for time marching.
    Also updates IC t values to window.t_start so they aren't filtered out.
    
    Called after each regenerate_training_data to ensure IC values come from
    the previous window's model, not the analytical IC.
    
    Args:
        train_data: Freshly resampled training data
        cfg: Config with _time_marching_window info
        device: Device for inference
    
    Returns:
        train_data with IC h_gt overridden (if time marching window > 0)
    """
    tm_window = cfg.get('_time_marching_window', {})
    if not tm_window.get('enabled', False):
        return train_data
    
    window_idx = tm_window.get('idx', 0)
    prev_model = tm_window.get('prev_model', None)
    t_start = tm_window.get('t_start', 0)
    
    # Window 0 uses analytical IC, no override needed
    if window_idx == 0 or prev_model is None:
        return train_data
    
    # Get IC mask and points
    ic_mask = train_data['mask']['IC']
    if ic_mask.sum() == 0:
        logger.info(f"  [IC Override] Window {window_idx}: No IC points found in dataset, skipping")
        return train_data
    
    x_ic = train_data['x'][ic_mask]
    h_gt_original = train_data['h_gt'][ic_mask].clone()
    
    # Create t values at window.t_start for querying previous model
    t_query = torch.full_like(train_data['t'][ic_mask], t_start)
    
    # Diagnostic: print input stats
    logger.info(f"  [IC Override] Window {window_idx}: Overriding {ic_mask.sum().item()} IC points at t={t_start:.4f}")
    logger.info(f"    x_ic: shape={x_ic.shape}, min={x_ic.min().item():.4f}, max={x_ic.max().item():.4f}, mean={x_ic.mean().item():.4f}")
    logger.info(f"    h_gt (original): min={h_gt_original.min().item():.4f}, max={h_gt_original.max().item():.4f}, mean={h_gt_original.mean().item():.4f}")
    
    # Query previous model for IC values
    prev_model.eval()
    with torch.no_grad():
        inputs = torch.cat([x_ic, t_query], dim=1).to(device)
        h_pred = prev_model(inputs)
    
    # Diagnostic: print prediction stats
    has_nan = torch.isnan(h_pred).any().item()
    has_inf = torch.isinf(h_pred).any().item()
    logger.info(f"    h_pred: min={h_pred.min().item():.4f}, max={h_pred.max().item():.4f}, mean={h_pred.mean().item():.4f}")
    logger.info(f"    h_pred contains NaN: {has_nan}, Inf: {has_inf}")
    
    if has_nan or has_inf:
        logger.info(f"    [WARNING] Previous model produced invalid values! This will cause NaN divergence.")
        num_nan = torch.isnan(h_pred).sum().item()
        num_inf = torch.isinf(h_pred).sum().item()
        logger.info(f"    Number of NaN: {num_nan}, Number of Inf: {num_inf}")
    
    # Override h_gt AND t for IC points
    train_data['h_gt'][ic_mask] = h_pred.to(train_data['h_gt'].device)
    train_data['t'][ic_mask] = t_start  # Set IC t to window start
    
    return train_data


class _NumpySafeEncoder(json.JSONEncoder):
    """Handles numpy scalars and PyTorch tensors that stdlib json cannot serialize."""
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _create_adam_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    """Create Adam optimizer with config parameters.
    
    Only includes trainable parameters (requires_grad=True) to avoid
    wasting memory/compute on frozen parameters (e.g., pretrained base model).
    """
    betas = tuple(cfg['adam_betas'])
    eps = cfg['adam_eps']
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(
        trainable_params,
        lr=cfg['lr'],
        betas=betas,
        eps=eps
    )


def _create_lbfgs_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    """Create LBFGS optimizer. Should be used with full-batch training.
    
    Only includes trainable parameters (requires_grad=True) to avoid
    wasting memory/compute on frozen parameters (e.g., pretrained base model).
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.LBFGS(
        trainable_params,
        lr=cfg['lbfgs_lr'],
        max_iter=cfg['lbfgs_max_iter'],
        max_eval=None,  # Default: max_iter * 1.25
        history_size=cfg['lbfgs_history_size'],
        line_search_fn=cfg['lbfgs_line_search'],
        tolerance_grad=cfg['lbfgs_tolerance_grad'],
        tolerance_change=cfg['lbfgs_tolerance_change']
    )


def _create_soap_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    """Create SOAP optimizer (quasi-second-order, Shampoo-preconditioned Adam).

    Only includes trainable parameters (requires_grad=True).
    """
    from optimizers.soap import SOAP
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return SOAP(
        trainable_params,
        lr=cfg['lr'],
        betas=tuple(cfg['soap_betas']),
        eps=cfg['adam_eps'],
        precondition_frequency=cfg['soap_precondition_frequency'],
        weight_decay=cfg['soap_weight_decay'],
    )


def _create_ssbroyden_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    """Create SSBroyden (Self-Scaled Broyden) quasi-Newton optimizer via scimba.

    Falls back to LBFGS with a warning if scimba is not installed.
    Only includes trainable parameters.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    try:
        from scimba_torch.optimizers.ssbroyden import SSBroyden
        return SSBroyden(
            trainable_params,
            lr=cfg.get('ssbroyden_lr', 1.0),
            tolerance_grad=cfg.get('ssbroyden_tolerance_grad', 1e-10),
            method='ssbroyden',
        )
    except ImportError:
        logger.info("  [Warning] scimba not installed — SSBroyden unavailable, falling back to LBFGS.")
        logger.info("            Install with: pip install scimba")
        return _create_lbfgs_optimizer(model, cfg)


def _create_optimizer_by_name(name: str, model: nn.Module, cfg: Dict) -> Tuple[torch.optim.Optimizer, str]:
    """Create an optimizer by name string. Returns (optimizer, display_name)."""
    name = name.lower()
    if name == 'soap':
        return _create_soap_optimizer(model, cfg), 'SOAP'
    elif name == 'lbfgs':
        return _create_lbfgs_optimizer(model, cfg), 'LBFGS'
    elif name == 'ssbroyden':
        opt = _create_ssbroyden_optimizer(model, cfg)
        return opt, 'SSBroyden' if opt.__class__.__name__ != 'LBFGS' else 'LBFGS'
    else:
        return _create_adam_optimizer(model, cfg), 'Adam'


def _debug_print_model_state(model: nn.Module, segment_name: str, 
                             eval_data: Dict = None) -> None:
    """Print comprehensive model state at segment start for debugging."""
    logger.info(f"\n[DEBUG] Model state at start of segment '{segment_name}':")
    logger.info(f"  Model type: {type(model).__name__}")
    
    # Basic model info
    base = getattr(model, 'base_model', None)
    experts = getattr(model, 'experts', [])
    regions = getattr(model, 'regions', [])
    
    logger.info(f"  Has base_model: {base is not None}")
    logger.info(f"  Num experts: {len(experts)}")
    logger.info(f"  Num regions: {len(regions)}")
    
    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Base model state
    base_arch = getattr(model, 'base_architecture', None)
    if base is not None:
        base_params = sum(p.numel() for p in base.parameters())
        base_trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
        base_grad_status = "TRAINABLE" if base_trainable > 0 else "FROZEN"
        logger.info(f"  Base: {base_params:,} params, {base_grad_status}"
                    + (f", arch={list(base_arch)}" if base_arch else ""))
    if hasattr(model, 'experts_architecture'):
        exp_arch = model.experts_architecture
        if base_arch is None or list(exp_arch) != list(base_arch):
            logger.info(f"  Experts architecture (spawn): {list(exp_arch)}")
    
    # Expert states
    for idx, expert in enumerate(experts):
        exp_params = sum(p.numel() for p in expert.parameters())
        exp_trainable = sum(p.numel() for p in expert.parameters() if p.requires_grad)
        exp_grad_status = "TRAINABLE" if exp_trainable > 0 else "FROZEN"
        region = regions[idx] if idx < len(regions) else None
        depth = region.depth if region else "?"
        parent = region.parent_idx if region else "?"
        logger.info(f"  Expert[{idx}]: {exp_params:,} params, {exp_grad_status}, depth={depth}, parent={parent}")
        if region:
            logger.info(f"    Region: {region.bounds_lower} -> {region.bounds_upper}")
    
    # Leaf indices
    if hasattr(model, 'leaf_indices'):
        logger.info(f"  Leaf indices: {sorted(model.leaf_indices)}")

    # Call model-specific debug_composition if available and has experts
    if hasattr(model, 'debug_composition') and len(experts) > 0 and eval_data is not None:
        try:
            sample_inputs = torch.cat([eval_data['x'][:100], eval_data['t'][:100]], dim=1)
            model.debug_composition(sample_inputs)
        except Exception as e:
            logger.info(f"  [DEBUG] debug_composition failed: {e}")
    
    # ── Per-expert output magnitude (shows contribution magnitudes) ──
    if eval_data is not None and hasattr(model, 'forward_decomposed'):
        try:
            with torch.no_grad():
                sample_inputs = torch.cat([eval_data['x'][:200], eval_data['t'][:200]], dim=1)
                decomp = model.forward_decomposed(sample_inputs)
                
                # Log base output magnitude
                if 'base' in decomp:
                    base_out = decomp['base']
                    logger.info(f"\n[DEBUG] Output magnitudes (N=200 sample points):")
                    logger.info(f"  Base: norm={base_out.norm().item():.4f}, "
                              f"mean={base_out.mean().item():.6f}, "
                              f"std={base_out.std().item():.6f}")
                
                # Log each expert's output magnitude
                for i in range(len(experts)):
                    key = f'expert_{i}'
                    if key in decomp:
                        exp_out = decomp[key]
                        logger.info(f"  Expert[{i}]: norm={exp_out.norm().item():.4f}, "
                                  f"mean={exp_out.mean().item():.6f}, "
                                  f"std={exp_out.std().item():.6f}")
                
                # Log composed output
                composed_out = model(sample_inputs)
                logger.info(f"  Composed: norm={composed_out.norm().item():.4f}, "
                          f"mean={composed_out.mean().item():.6f}, "
                          f"std={composed_out.std().item():.6f}")
        except Exception as e:
            logger.info(f"  [DEBUG] Output magnitude computation failed: {e}")
    
    logger.info("")  # Blank line for readability


def _create_primary_optimizer(model: nn.Module, cfg: Dict) -> Tuple[torch.optim.Optimizer, str]:
    """Create the primary (first-order) optimizer based on config.

    Supports new optimizer_1 key and legacy optimizer key.
    Returns (optimizer, name_string).
    """
    opt_name = cfg['optimizer_1'].lower()
    return _create_optimizer_by_name(opt_name, model, cfg)


def _create_lr_scheduler(optimizer, cfg, total_steps):
    """Create an LR scheduler composed of optional warmup + decay.

    Uses standard PyTorch schedulers:
    - LinearLR for warmup (ramps from ~0 to base lr)
    - StepLR for exponential decay (multiplies lr by decay_rate every decay_steps)
    - CosineAnnealingLR for cosine schedule

    Returns None if no scheduling is configured.
    """
    from torch.optim.lr_scheduler import LinearLR, StepLR, CosineAnnealingLR, SequentialLR

    schedule = cfg['lr_schedule']
    warmup_steps = cfg['lr_warmup_steps']

    if schedule == 'none' and warmup_steps <= 0:
        return None

    schedulers = []
    milestones = []

    if warmup_steps > 0:
        start_factor = cfg['lr_warmup_start_factor']
        schedulers.append(LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_steps))
        milestones.append(warmup_steps)

    if schedule == 'exponential':
        decay_rate = cfg['lr_decay_rate']
        decay_steps = cfg['lr_decay_steps']
        schedulers.append(StepLR(optimizer, step_size=decay_steps, gamma=decay_rate))
    elif schedule == 'cosine':
        remaining = max(total_steps - warmup_steps, 1)
        schedulers.append(CosineAnnealingLR(optimizer, T_max=remaining))

    if len(schedulers) == 0:
        return None
    elif len(schedulers) == 1:
        return schedulers[0]
    else:
        return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


def _get_optimizer_snapshot(optimizer, lr_scheduler, step_count):
    """Return a compact dict of optimizer/scheduler state for metrics logging at key events."""
    lrs = [pg['lr'] for pg in optimizer.param_groups]
    sched_type = type(lr_scheduler).__name__ if lr_scheduler is not None else None
    sched_last_epoch = getattr(lr_scheduler, 'last_epoch', None) if lr_scheduler is not None else None
    sched_base_lrs = None
    if lr_scheduler is not None:
        sched_base_lrs = getattr(lr_scheduler, 'base_lrs', None)
        if sched_base_lrs is None:
            first_sub = (getattr(lr_scheduler, '_schedulers', None) or [None])[0]
            sched_base_lrs = getattr(first_sub, 'base_lrs', None) if first_sub else None
    return {
        'step_count': step_count,
        'num_param_groups': len(optimizer.param_groups),
        'lr_per_group': lrs,
        'scheduler_type': sched_type,
        'scheduler_last_epoch': sched_last_epoch,
        'scheduler_base_lrs': sched_base_lrs,
    }


def train(
    model: nn.Module,
    loss_fn: Callable,
    train_data_path: str,
    eval_data_path: str,
    cfg: Dict,
    run_dir: Path
) -> Path:
    """
    Train a PINN model with CUDA acceleration and vectorized operations.

    Thin wrapper over three phases:
      1. ``_setup_training``    — build data/optimizer/metrics/adaptive state.
      2. ``train_orchestrator`` — per-variant segment + staged-spawning driver.
      3. ``_finalize_training`` — checkpoints, plots, metrics, summary.

    Args:
        model: Neural network model
        loss_fn: Loss function (model, batch) -> scalar
        train_data_path: Path to training_data.pt
        eval_data_path: Path to eval_data.pt
        cfg: Configuration dictionary
        run_dir: Output directory for this run

    Returns:
        Path to best checkpoint (or None on NaN divergence)
    """
    ctx = _setup_training(model, loss_fn, train_data_path, eval_data_path, cfg, run_dir)
    train_orchestrator(ctx)
    return _finalize_training(ctx)


def _setup_training(
    model: nn.Module,
    loss_fn: Callable,
    train_data_path: str,
    eval_data_path: str,
    cfg: Dict,
    run_dir: Path
) -> TrainingContext:
    """Phase 1 of :func:`train`: build datasets, optimizer, metrics, adaptive state.

    Behavior-preserving extraction of the original setup block (no logic changes).
    Returns a :class:`TrainingContext` carrying all state into the loop + finalize.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)

    # Validate per-problem config (all features must be explicitly specified)
    validate_problem_config(cfg)
    validate_adaptive_staged_config(cfg)
    problem = cfg['problem']
    problem_cfg = cfg[problem]
    
    # Copy per-problem features to top-level for backward compatibility with
    # functions that read cfg['init'], cfg['fourier_features'], etc.
    for key in ['rwf', 'fourier_features', 'init', 'lra', 'adaptive_sampling',
                'grad_clip_norm', 'expert_grad_clip_norm']:
        if key in problem_cfg:
            cfg[key] = problem_cfg[key]

    # Setup device
    device = torch.device('cuda' if cfg['cuda'] and
                          torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # GPU optimization and monitoring
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"Initial GPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.3f} GB")
        torch.backends.cudnn.benchmark = True
        logger.info("CUDNN benchmark enabled for GPU optimization")

    # Set seed for reproducibility
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg['seed'])

    # Move model to device
    model = model.to(device)

    # DIAGNOSTIC: Verify model is on correct device (configurable)
    if cfg['adaptive_pinn']['enable_gradient_diagnostics']:
        logger.info(f"\n{'='*40} GPU DIAGNOSTIC {'='*40}")
        logger.info(f"Target device: {device}")
        if hasattr(model, 'base_model'):
            logger.info(f"Base model device: {next(model.base_model.parameters()).device}")
        else:
            logger.info(f"Model device: {next(model.parameters()).device}")
        logger.info(f"{'='*80}\n")

    # Load datasets
    logger.info(f"\nLoading datasets...")
    train_data = torch.load(train_data_path)
    eval_data = torch.load(eval_data_path)

    # Move data to device
    train_data = _move_batch_to_device(train_data, device)
    eval_data = _move_batch_to_device(eval_data, device)

    # Cast data to configured precision (float32 or float64)
    precision = cfg.get('precision', 'float32')
    target_dtype = torch.float64 if precision == 'float64' else torch.float32
    train_data = _cast_data_to_dtype(train_data, target_dtype)
    eval_data = _cast_data_to_dtype(eval_data, target_dtype)

    # Filter train and eval data by window temporal bounds if time marching is enabled
    time_marching_window = cfg.get('_time_marching_window', {})
    if time_marching_window.get('enabled', False):
        t_start = time_marching_window['t_start']
        t_end = time_marching_window['t_end']
        window_idx = time_marching_window['idx']
        
        # IMPORTANT: For windows 1+, override IC BEFORE filtering
        # This updates IC t values from t=0 to t=window.t_start, so they survive filtering
        train_data = _override_ic_for_time_marching(train_data, cfg, device)
        eval_data = _override_ic_for_time_marching(eval_data, cfg, device)
        
        # --- Filter TRAINING data ---
        t_train = train_data['t'].squeeze()
        train_mask = (t_train >= t_start) & (t_train < t_end)
        # Handle edge case: last window should include t_end
        if t_train.max() <= t_end:
            train_mask = train_mask | (t_train == t_end)
        
        n_train_original = train_data['x'].shape[0]
        n_train_filtered = train_mask.sum().item()
        
        logger.info(f"  [Time Marching] Filtering train data for window {window_idx}: "
              f"t in [{t_start:.4f}, {t_end:.4f}]")
        logger.info(f"  [Time Marching] Train data: {n_train_original} → {n_train_filtered} points")
        
        # Apply mask to train_data
        filtered_train_data = {}
        for key, value in train_data.items():
            if torch.is_tensor(value):
                filtered_train_data[key] = value[train_mask]
            elif key == 'mask':
                filtered_train_data[key] = {
                    k: v[train_mask] if torch.is_tensor(v) else v
                    for k, v in value.items()
                }
            else:
                filtered_train_data[key] = value
        train_data = filtered_train_data
        
        # --- Filter EVAL data ---
        t_eval = eval_data['t'].squeeze()
        eval_mask = (t_eval >= t_start) & (t_eval < t_end)
        # Handle edge case: last window should include t_end
        if t_eval.max() <= t_end:
            eval_mask = eval_mask | (t_eval == t_end)
        
        n_eval_original = eval_data['x'].shape[0]
        n_eval_filtered = eval_mask.sum().item()
        
        logger.info(f"  [Time Marching] Filtering eval data for window {window_idx}: "
              f"t in [{t_start:.4f}, {t_end:.4f}]")
        logger.info(f"  [Time Marching] Eval data: {n_eval_original} → {n_eval_filtered} points")
        
        # Apply mask to eval_data
        filtered_eval_data = {}
        for key, value in eval_data.items():
            if torch.is_tensor(value):
                filtered_eval_data[key] = value[eval_mask]
            elif key == 'mask':
                filtered_eval_data[key] = {
                    k: v[eval_mask] if torch.is_tensor(v) else v
                    for k, v in value.items()
                }
            else:
                filtered_eval_data[key] = value
        eval_data = filtered_eval_data

    logger.info(f"  Train size: {train_data['x'].shape[0]}")
    logger.info(f"  Eval size: {eval_data['x'].shape[0]}")
    logger.info(f"  Train data device: {train_data['x'].device}")
    logger.info(f"  Eval data device: {eval_data['x'].device}")

    # Reset default device context to CPU before creating DataLoaders.
    # This fixes a PyTorch issue where CUDA inference (e.g., prev_model forward pass
    # in time marching) can corrupt the global device context, causing RandomSampler
    # to fail with "Expected 'cuda' device type for generator but found 'cpu'".
    # This does NOT affect training device - model and data are already on CUDA
    # via explicit .to(device) calls; this only affects internal generator creation.
    torch.set_default_device(None)

    # Create DataLoaders
    train_loader = _create_dataloader(train_data, cfg['batch_size'],
                                      shuffle=True)
    eval_loader = _create_dataloader(eval_data, cfg['batch_size'],
                                     shuffle=False)

    # ── 3-phase logic (root -> M-term tree spawn -> leaf training) ──
    adaptive_cfg_init = cfg['adaptive_pinn']
    is_adaptive_init = adaptive_cfg_init['enabled']
    initial_train_cfg = adaptive_cfg_init.get('initial_train', None)
    pretrained_base_checkpoint = problem_cfg.get('pretrained_base_checkpoint', None)
    _pretrained_force_spawn = False  # set True to force first-epoch spawn (checkpoint flow)

    if not is_adaptive_init:
        # Non-adaptive base-only training: single phase, no spawning.
        active_cfg = cfg
        epochs = cfg['epochs']
        phase3_epochs = 0
        current_phase = 0
        use_three_phase = False
    elif pretrained_base_checkpoint is not None:
        # Phase 1 supplied as a checkpoint: load the base, skip Phase-1 training,
        # force the spawn on the first loop epoch, then transition to Phase 3.
        _load_pretrained_base(model, pretrained_base_checkpoint, cfg)
        phase3_epochs = cfg['epochs']
        active_cfg = cfg
        epochs = 1            # one loop epoch to trigger the forced spawn
        current_phase = 1
        use_three_phase = True
        _pretrained_force_spawn = True
        logger.info(f"\n  [3-Phase] Phase 1 skipped: base loaded from "
              f"{pretrained_base_checkpoint}")
        logger.info(f"  [3-Phase] Phase 3 will run for {phase3_epochs} epochs after spawning")
    else:
        # Phase 1 trains the base for initial_train.epochs.
        if initial_train_cfg is None:
            raise ValueError(
                "adaptive_pinn.initial_train is required "
                "when pretrained_base_checkpoint is null.")
        phase1_cfg = dict(cfg)
        for k, v in initial_train_cfg.items():
            phase1_cfg[k] = v
        phase1_epochs = initial_train_cfg['epochs']
        phase3_epochs = cfg['epochs']
        active_cfg = phase1_cfg
        epochs = phase1_epochs
        current_phase = 1
        use_three_phase = True
        logger.info(f"\n  [3-Phase] Phase 1: initial training for {phase1_epochs} epochs")
        logger.info(f"  [3-Phase] Phase 3 will run for {phase3_epochs} epochs after spawning")

    # Determine optimizer strategy (new config: optimizer_1/optimizer_2/optimizer_switch_epoch)
    optimizer_1_name = active_cfg['optimizer_1'].lower()
    optimizer_2_name_cfg = active_cfg.get('optimizer_2', None)
    optimizer_2_name = optimizer_2_name_cfg.lower() if optimizer_2_name_cfg else None

    # optimizer_switch_epoch: when to switch. None / ignored when optimizer_2 is null.
    if optimizer_2_name is not None:
        switch_epoch = active_cfg['optimizer_switch_epoch']
    else:
        switch_epoch = epochs + 1  # never switch

    # Estimate total optimizer steps for LR scheduler
    n_train_samples = train_data['x'].shape[0]
    batches_per_epoch = max(1, (n_train_samples + cfg['batch_size'] - 1) // cfg['batch_size'])
    total_steps_estimate = epochs * batches_per_epoch

    # Patience tracking: only active after switch (or from epoch 1 if no switch)
    patience_start_epoch = switch_epoch if optimizer_2_name is not None else 1

    # Setup initial optimizer + scheduler
    full_batch_opt1 = optimizer_1_name in ('lbfgs', 'ssbroyden')
    if full_batch_opt1:
        optimizer, current_optimizer_name = _create_optimizer_by_name(optimizer_1_name, model, active_cfg)
        lr_scheduler = None
        logger.info(f"Using {current_optimizer_name} optimizer (full-batch) for all epochs")
    else:
        optimizer, current_optimizer_name = _create_primary_optimizer(model, active_cfg)
        lr_scheduler = _create_lr_scheduler(optimizer, active_cfg, total_steps_estimate)
        if optimizer_2_name is not None:
            logger.info(f"Using {current_optimizer_name} until epoch {switch_epoch}, "
                  f"then {optimizer_2_name.upper()} (full-batch)")
            logger.info(f"  Patience early-stopping active from epoch {patience_start_epoch}")
        else:
            logger.info(f"Using {current_optimizer_name} optimizer (mini-batch) for all epochs")
        if lr_scheduler is not None:
            sched_name = active_cfg['lr_schedule']
            warmup = active_cfg['lr_warmup_steps']
            logger.info(f"  LR schedule: {sched_name} (warmup={warmup} steps, ~{total_steps_estimate} total steps)")

    step_count = 0  # global optimizer step counter for LR scheduler

    # Training setup
    print_every = cfg['print_every']
    eval_every = cfg['eval_every']
    save_every = cfg['save_every']

    # Metrics storage
    # Note: train_loss is stored every epoch, eval metrics only every print_every
    metrics = {
        'train_loss_epochs': [],  # All epochs
        'train_loss': [],          # All epochs
        'epochs': [],              # Evaluation epochs only
        'eval_loss': [],
        'eval_rel_l2': [],
        'eval_inf_norm': [],
        'causal_history': [],      # Causal training state (tol, min_weight, stage) at eval epochs
        'lra_history': [],         # LRA weights and grad norms at eval epochs
        'resample_events': [],     # Track resampling/skipping events
        # 'freeze_events' removed - staged freezing now uses requires_grad=False per level
        'plateau_events': [],      # Plateau check outcomes (deferred / triggered)
        'optimizer_events': [],    # Optimizer switch events
        'optimizer_snapshots': [],  # Optimizer/scheduler state at spawn, freeze, unfreeze, resample
        'loss_components_history': [],  # Per-component losses at eval epochs
        'exception_events': [],    # Caught Python exceptions with traceback
        # Term-wise loss components for plotting (populated during evaluation)
        'loss_components': {
            'epochs': [],      # Epochs where components were recorded
            'residual': [],    # PDE residual loss
            'ic': [],          # Initial condition loss
            'bc': [],          # Boundary condition loss
        },
        # Gradient norm history
        'gradient_norms': {
            'epochs': [],
            'total_grad_norm': [],  # Overall gradient norm
            'base_grad_norm': [],   # Base model gradient norm
            'experts_grad_norm': [], # Experts gradient norm (sum)
        },
        # Learning rate history
        'lr_history': {
            'epochs': [],
            'lr': [],
        },
    }

    best_eval_loss = float('inf')
    best_train_loss = float('inf')
    best_checkpoint_path = None
    patience_epochs = cfg['patience_epochs']
    min_epochs = cfg['min_epochs']
    # Relative-improvement threshold for the plateau test: an epoch only counts as
    # an improvement if it beats the anchored best by at least this fraction.
    patience_rel_delta = cfg.get('patience_rel_delta', 0.0)
    epochs_without_improvement = 0

    # LRA: adaptive loss component weighting (read from per-problem config)
    lra_cfg = problem_cfg['lra']
    lra_enabled = lra_cfg['enabled']
    if lra_enabled:
        initial_loss_weights = problem_cfg['loss_weights']
        lra_weights = LRAWeights(
            alpha=lra_cfg['alpha'],
            update_every=lra_cfg['update_every'],
            initial_weights=initial_loss_weights,
            scheme=lra_cfg['scheme'],
            scheme_cfg=lra_cfg,
        )
    else:
        lra_weights = None

    # Wrap loss_fn to apply LRA weights when enabled
    if lra_weights is not None:
        _orig_loss_fn = loss_fn

        def _lra_loss_fn(model, batch, for_tree_spawning=False, return_components=False, update_causal_state=True):
            if for_tree_spawning or return_components:
                return _orig_loss_fn(model, batch,
                                     for_tree_spawning=for_tree_spawning,
                                     return_components=return_components,
                                     update_causal_state=False)
            comps = _orig_loss_fn(model, batch, return_components=True, update_causal_state=update_causal_state)
            w = lra_weights.weights
            return sum(w.get(k, 1.0) * v for k, v in comps.items())

        _lra_loss_fn.causal_state = getattr(loss_fn, 'causal_state', None)
        loss_fn = _lra_loss_fn

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Adaptive PINN setup
    adaptive_cfg = cfg['adaptive_pinn']
    is_adaptive = adaptive_cfg['enabled']
    region_detector = None
    max_experts = adaptive_cfg['max_experts']

    # Read configurable norm variables
    variable_for_node_accept = adaptive_cfg['variable_for_node_accept']
    variable_for_expert_size = adaptive_cfg['variable_for_expert_size']

    if is_adaptive:
        tree_max_depth = adaptive_cfg['tree_max_depth']
        tree_min_samples_leaf = adaptive_cfg['tree_min_samples_leaf']

        logger.info(f"\nAdaptive PINN enabled (M-term tree by norm):")
        logger.info(f"  Max experts: {max_experts}")
        logger.info(f"  Tree max depth: {tree_max_depth}")
        logger.info(f"  Tree min samples leaf: {tree_min_samples_leaf}")
        logger.info(f"  M experts num: {adaptive_cfg['M_experts_num']}")
        logger.info(f"  Blending mode: {adaptive_cfg['blending_mode']}")
        logger.info(f"  Model type: {type(model).__name__}")
        enable_timing_cfg = adaptive_cfg['enable_timing']
        logger.info(f"  Timing profiling: {'enabled' if enable_timing_cfg else 'disabled'}")

        from adaptive.region_detector import RegionDetector
        from adaptive.visualization import prepare_ground_truth_grid

        domain_bounds = model.get_domain_bounds()
        gt_grid, gt_x, gt_t = prepare_ground_truth_grid(eval_data, domain_bounds)

        region_detector = RegionDetector(
            n_estimators=1,
            max_depth=tree_max_depth,
            min_samples_leaf=tree_min_samples_leaf,
            domain_bounds=domain_bounds
        )
        
        # Create directory for adaptive outputs
        adaptive_plots_dir = run_dir / "adaptive_plots"
        adaptive_plots_dir.mkdir(exist_ok=True)
    
    rejected_regions = []
    leaf_loss_history = []

    # Training loop
    total_epochs = epochs  # may extend when transitioning to Phase 3
    logger.info(f"\nTraining for {total_epochs} epochs...")
    start_time = time.time()
    
    # Epoch timer for fine-grained performance profiling
    enable_timing = adaptive_cfg['enable_timing'] if is_adaptive else False
    timer = EpochTimer(enabled=enable_timing, print_every=eval_every)
    if enable_timing:
        model._timer = timer

    train_loss = 0.0
    eval_loss = 0.0
    eval_rel_l2 = 0.0
    eval_inf_norm = 0.0

    resample_every = cfg['sampling']['resample_every_epochs']
    base_seed = cfg['seed']
    # Gradient clipping (read from per-problem config)
    grad_clip_norm = problem_cfg['grad_clip_norm']
    # Tighter clip for all expert params (separate from base); only active when experts exist.
    # When no experts exist (base-only phase), grad_clip_norm applies to all params as usual.
    expert_grad_clip_norm = problem_cfg['expert_grad_clip_norm']

    # Consolidated feature summary
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE SUMMARY")
    logger.info("=" * 60)
    
    # Fourier Features (read from per-problem config)
    ff_cfg = problem_cfg['fourier_features']
    ff_enabled = ff_cfg['enabled']
    if ff_enabled:
        ff_dim = ff_cfg['dim']
        ff_scale = ff_cfg['scale']
        _base_for_ff = model.base_model if hasattr(model, 'base_model') else model
        _ff_out = _base_for_ff.ff_emb.output_dim if (hasattr(_base_for_ff, 'ff_emb') and _base_for_ff.ff_emb is not None) else 2 * ff_dim
        _periodic = ff_cfg['periodic']
        logger.info(f"  Fourier Features: enabled (dim={ff_dim}, scale={ff_scale}, output_dim={_ff_out}, periodic={_periodic})")
    else:
        logger.info(f"  Fourier Features: disabled")
    
    # RWF (read from per-problem config)
    rwf_enabled = problem_cfg['rwf']
    if rwf_enabled:
        logger.info(f"  RWF: enabled")
    else:
        logger.info(f"  RWF: disabled")
    
    # Causal Training
    _cs_init = getattr(loss_fn, 'causal_state', None)
    if _cs_init is not None:
        logger.info(f"  Causal Training: enabled (schedule={_cs_init['schedule']}, chunks={_cs_init['num_chunks']}, threshold={_cs_init['threshold']})")
    else:
        logger.info(f"  Causal Training: disabled")
    
    # LRA
    if lra_enabled:
        init_w = lra_weights.weights
        init_w_str = ', '.join(f'{k}={v:.1f}' for k, v in init_w.items())
        logger.info(f"  LRA: enabled (scheme={lra_weights.scheme}, alpha={lra_weights.alpha}, update_every={lra_weights.update_every}, "
              f"init_weights={{{init_w_str}}})")
    else:
        logger.info(f"  LRA: disabled")
    
    # Resampling & Adaptive Sampling (read from per-problem config)
    adaptive_sampling_cfg = problem_cfg['adaptive_sampling']
    adaptive_sampling_enabled = adaptive_sampling_cfg['enabled']
    if resample_every > 0:
        if adaptive_sampling_enabled:
            as_ratio = adaptive_sampling_cfg['adaptive_ratio']
            logger.info(f"  Resampling: every {resample_every} epochs (adaptive: enabled, ratio={as_ratio})")
        else:
            logger.info(f"  Resampling: every {resample_every} epochs (adaptive: disabled)")
    else:
        logger.info(f"  Resampling: disabled")
    
    # Optimizer schedule
    opt1_name = cfg['optimizer_1']
    opt2_name = cfg.get('optimizer_2', 'null')
    if opt2_name and opt2_name != 'null':
        switch_epoch = cfg['optimizer_switch_epoch']
        logger.info(f"  Optimizer: {opt1_name} → {opt2_name} at epoch {switch_epoch}")
    else:
        logger.info(f"  Optimizer: {opt1_name}")
    
    # Early stopping
    if patience_epochs > 0:
        logger.info(f"  Early stopping: enabled (patience={patience_epochs}, min_epochs={min_epochs})")
    else:
        logger.info(f"  Early stopping: disabled")
    
    logger.info("=" * 60 + "\n")

    # Smart initialization (Glorot hidden + zero/LS output) — base model only
    # Skip when a pretrained base was loaded (init would destroy the trained weights).
    from trainer.init import apply_hidden_init, apply_output_init
    _init_target = model.base_model if is_adaptive else model
    _init_cfg = cfg.get('init', {})
    if pretrained_base_checkpoint is not None:
        logger.info("[Init] Skipped — base loaded from pretrained checkpoint.")
    elif _init_cfg.get('hidden', 'default') != 'default' or _init_cfg.get('output', 'default') != 'default':
        logger.info("[Init] Applying smart initialization to base model...")
        # parent_weights is expert-only; the base model uses glorot instead
        # (except for resnet, whose glorot zeros fc2 for the identity start).
        _base_init_cfg = cfg
        _expert_type = cfg['adaptive_pinn']['expert_type']
        if cfg['init']['hidden'] == 'parent_weights' and _expert_type != 'resnet':
            _base_init_cfg = {**cfg, 'init': {**cfg['init'], 'hidden': 'glorot'}}
        apply_hidden_init(_init_target, _base_init_cfg)
        apply_output_init(_init_target, train_data, cfg, device)
        logger.info("")

    # ── Build the context that carries all state into the loop + finalize ──
    return TrainingContext(
        model=model,
        loss_fn=loss_fn,
        cfg=cfg,
        problem=problem,
        problem_cfg=problem_cfg,
        device=device,
        run_dir=run_dir,
        train_data=train_data,
        eval_data=eval_data,
        train_loader=train_loader,
        eval_loader=eval_loader,
        active_cfg=active_cfg,
        epochs=epochs,
        phase3_epochs=phase3_epochs,
        current_phase=current_phase,
        use_three_phase=use_three_phase,
        optimizer_2_name=optimizer_2_name,
        switch_epoch=switch_epoch,
        batches_per_epoch=batches_per_epoch,
        total_steps_estimate=total_steps_estimate,
        patience_start_epoch=patience_start_epoch,
        optimizer=optimizer,
        current_optimizer_name=current_optimizer_name,
        lr_scheduler=lr_scheduler,
        step_count=step_count,
        print_every=print_every,
        eval_every=eval_every,
        save_every=save_every,
        metrics=metrics,
        best_eval_loss=best_eval_loss,
        best_train_loss=best_train_loss,
        best_checkpoint_path=best_checkpoint_path,
        patience_epochs=patience_epochs,
        min_epochs=min_epochs,
        patience_rel_delta=patience_rel_delta,
        epochs_without_improvement=epochs_without_improvement,
        lra_weights=lra_weights,
        checkpoint_dir=checkpoint_dir,
        adaptive_cfg=adaptive_cfg,
        is_adaptive=is_adaptive,
        initial_train_cfg=initial_train_cfg,
        pretrained_base_checkpoint=pretrained_base_checkpoint,
        _pretrained_force_spawn=_pretrained_force_spawn,
        region_detector=region_detector,
        max_experts=max_experts,
        variable_for_node_accept=variable_for_node_accept,
        variable_for_expert_size=variable_for_expert_size,
        domain_bounds=domain_bounds,
        gt_grid=gt_grid,
        gt_x=gt_x,
        gt_t=gt_t,
        adaptive_plots_dir=adaptive_plots_dir,
        rejected_regions=rejected_regions,
        leaf_loss_history=leaf_loss_history,
        total_epochs=total_epochs,
        start_time=start_time,
        timer=timer,
        train_loss=train_loss,
        eval_loss=eval_loss,
        eval_rel_l2=eval_rel_l2,
        eval_inf_norm=eval_inf_norm,
        resample_every=resample_every,
        base_seed=base_seed,
        grad_clip_norm=grad_clip_norm,
        expert_grad_clip_norm=expert_grad_clip_norm,
        adaptive_sampling_enabled=adaptive_sampling_enabled,
        epoch=0,
        _nan_detected=False,
    )


def _train_segment(
    ctx: TrainingContext,
    segment_name: str,
    epoch_budget: int,
    segment_cfg: Dict,
    *,
    lr_override=None,
    min_epochs_override=None,
) -> SegmentResult:
    """Run one training segment: a self-contained epoch loop with no spawning.

    Builds a fresh optimizer + scheduler from ``segment_cfg`` over the model's
    currently-trainable params (freezing is set by the caller), advances the
    GLOBAL epoch counter ``ctx.epoch`` by up to ``epoch_budget`` epochs, and
    handles the in-segment optimizer_1->optimizer_2 switch + patience early-stop
    (with optimizer_1 fast-forward). Shared state is read from ``ctx``; values
    reassigned during the segment are written back to ``ctx`` at the end.

    The forward pass always evaluates the FULL model composition (base + every
    spawned expert, frozen or not); only ``requires_grad`` controls which params
    the optimizer updates.
    """
    # ── Unpack shared / per-epoch state from ctx (segment-local state is built below) ──
    model = ctx.model
    loss_fn = ctx.loss_fn
    cfg = ctx.cfg
    problem_cfg = ctx.problem_cfg
    device = ctx.device
    run_dir = ctx.run_dir
    train_data = ctx.train_data
    train_loader = ctx.train_loader
    eval_loader = ctx.eval_loader
    batches_per_epoch = ctx.batches_per_epoch
    print_every = ctx.print_every
    eval_every = ctx.eval_every
    save_every = ctx.save_every
    metrics = ctx.metrics
    best_eval_loss = ctx.best_eval_loss
    best_checkpoint_path = ctx.best_checkpoint_path
    patience_epochs = ctx.patience_epochs
    patience_rel_delta = ctx.patience_rel_delta
    lra_weights = ctx.lra_weights
    checkpoint_dir = ctx.checkpoint_dir
    adaptive_cfg = ctx.adaptive_cfg
    is_adaptive = ctx.is_adaptive
    timer = ctx.timer
    start_time = ctx.start_time
    train_loss = ctx.train_loss
    eval_loss = ctx.eval_loss
    eval_rel_l2 = ctx.eval_rel_l2
    eval_inf_norm = ctx.eval_inf_norm
    resample_every = ctx.resample_every
    base_seed = ctx.base_seed
    grad_clip_norm = ctx.grad_clip_norm
    expert_grad_clip_norm = ctx.expert_grad_clip_norm
    adaptive_sampling_enabled = ctx.adaptive_sampling_enabled
    # Diagnostic residual-heatmap plot cadence (defaults to the resample cadence).
    plot_samples_every = cfg.get('sampling', {}).get(
        'plot_samples_every', resample_every) or resample_every

    # ── Segment setup: fresh optimizer + scheduler over current requires_grad params ──
    seg_cfg = dict(segment_cfg)
    if lr_override is not None:
        seg_cfg['lr'] = lr_override
    seg_min_epochs = (min_epochs_override
                      if min_epochs_override is not None else ctx.min_epochs)

    optimizer_1_name = seg_cfg['optimizer_1'].lower()
    optimizer_2_name_cfg = seg_cfg.get('optimizer_2', None)
    optimizer_2_name = optimizer_2_name_cfg.lower() if optimizer_2_name_cfg else None

    segment_start_epoch = ctx.epoch
    total_epochs = segment_start_epoch + epoch_budget

    if optimizer_2_name is not None:
        switch_epoch = segment_start_epoch + seg_cfg['optimizer_switch_epoch']
    else:
        switch_epoch = total_epochs + 1  # never switch

    total_steps_estimate = max(1, epoch_budget) * batches_per_epoch

    full_batch_opt1 = optimizer_1_name in ('lbfgs', 'ssbroyden')
    # Default-device context for this segment. Full-batch optimizers (LBFGS/
    # SSBroyden) need it on CUDA so their state tensors are allocated on the GPU;
    # mini-batch optimizers (Adam/SOAP) need it reset to None so the DataLoader's
    # sampler generator (CPU) matches torch.randperm during iteration. A previous
    # segment may have left the default on CUDA (e.g. after an optimizer switch),
    # so we always re-establish it here at the segment boundary.
    if full_batch_opt1:
        torch.set_default_device(device)
        optimizer, current_optimizer_name = _create_optimizer_by_name(
            optimizer_1_name, model, seg_cfg)
        lr_scheduler = None
    else:
        torch.set_default_device(None)
        optimizer, current_optimizer_name = _create_primary_optimizer(model, seg_cfg)
        lr_scheduler = _create_lr_scheduler(optimizer, seg_cfg, total_steps_estimate)

    step_count = 0
    best_train_loss = float('inf')
    epochs_without_improvement = 0
    # optimizer_1 is watched from the segment start; reset to switch_epoch at the switch.
    patience_start_epoch = segment_start_epoch
    _nan_detected = False
    _stopped_early = False
    _stop_reason = 'budget'

    _n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _switch_str = (f" -> {optimizer_2_name.upper()}@{switch_epoch}"
                   if optimizer_2_name is not None else "")
    logger.info(f"\n[Segment:{segment_name}] start | epochs "
          f"{segment_start_epoch + 1}..{total_epochs} (budget {epoch_budget}) | "
          f"optimizer={current_optimizer_name}{_switch_str} | lr={seg_cfg['lr']} | "
          f"trainable_params={_n_train_params}")
    
    # ── DEBUG: Print comprehensive model state at segment start ──
    _debug_print_model_state(model, segment_name, ctx.eval_data)
    
    metrics.setdefault('segment_events', []).append({
        'segment': segment_name,
        'start_epoch': segment_start_epoch + 1,
        'epoch_budget': epoch_budget,
        'optimizer_1': current_optimizer_name,
        'optimizer_2': optimizer_2_name,
        'switch_epoch': switch_epoch if optimizer_2_name is not None else None,
        'lr': seg_cfg['lr'],
        'trainable_params': _n_train_params,
    })

    epoch = segment_start_epoch
    while epoch < total_epochs:
        epoch += 1
        ctx.epoch = epoch  # keep ctx in sync for the orchestrator's emergency save
        timer.start_epoch(epoch, num_experts=model.num_experts if (is_adaptive and hasattr(model, 'num_experts')) else 0)

        # Enable residual caching for adaptive sampling if needed
        # Cache THIS epoch's residuals for NEXT epoch's resampling
        # (adaptive_sampling_enabled already set from problem_cfg above)
        causal_state = getattr(loss_fn, 'causal_state', None)
        
        will_cache_for_resample = (
            adaptive_sampling_enabled
            and resample_every > 0
            and epoch > 0 and epoch % resample_every == 0
        )
        # Cache residuals for the diagnostic heatmap even when adaptive sampling is
        # off. Cadence is controlled by sampling.plot_samples_every (independent of
        # the resample cadence; defaults to it).
        _problem_spatial_dim = problem_cfg['spatial_dim']
        will_cache_for_plot = (
            not adaptive_sampling_enabled
            and plot_samples_every > 0
            and epoch > 0 and epoch % plot_samples_every == 0
            and _problem_spatial_dim == 1
        )
        if will_cache_for_resample or will_cache_for_plot:
            model._residual_cache = []
            model._residual_cache_enabled = True
            # Log when adaptive sampling first activates
            if will_cache_for_resample and not hasattr(model, '_adaptive_sampling_activated'):
                model._adaptive_sampling_activated = True
                logger.info(f"  [Adaptive Sampling] Activated at epoch {epoch} (causal training reached final stage)")

        # Enable residual caching in split_loss_fn for diagnostic heatmap plots.
        # (The model-level cache is not populated in the split path; the loss fn
        # owns the cache instead.)
        _split_loss_fn = loss_fn if hasattr(loss_fn, '_residual_cache') else None
        _will_cache_split = (
            _split_loss_fn is not None
            and plot_samples_every > 0
            and epoch > 0 and epoch % plot_samples_every == 0
            and _problem_spatial_dim == 1
        )
        if _split_loss_fn is not None:
            _split_loss_fn._cache_residuals = _will_cache_split
            if _will_cache_split:
                _split_loss_fn._residual_cache.clear()

        # Resample training data periodically (in-memory, no disk I/O)
        # Skip resampling during L-BFGS/SSBroyden (they need stable loss landscape)
        allow_resample_optimizer = current_optimizer_name not in ('LBFGS', 'SSBroyden')
        _split_ctx = getattr(ctx, '_split_context', None)
        if resample_every > 0 and epoch > 1 and (epoch - 1) % resample_every == 0 and allow_resample_optimizer:
            resample_seed = base_seed + epoch
            if _split_ctx is not None:
                logger.info(f"  [Resample-Split] Rebuilding subdomain data at epoch {epoch}")
                # Use the frozen snapshot for stable interface targets
                train_data = build_subdomain_data(
                    _split_ctx['model_snapshot'], _split_ctx['new_expert_indices'],
                    _split_ctx['regions'], cfg, device, seed=resample_seed,
                    interface_model=_split_ctx.get('interface_model'),
                )
                ctx.train_data = train_data
                torch.set_default_device(None)
                train_loader = _create_split_dataloader(
                    train_data, cfg['batch_size'], shuffle=True)
                ctx.train_loader = train_loader
                metrics['resample_events'].append({
                    'epoch': epoch, 'action': 'split_resampled',
                    'optimizer': current_optimizer_name,
                })
                # Diagnostic residual heatmap: drain the per-expert residual cache
                # collected this epoch (union of all experts' local residual points).
                if (_split_loss_fn is not None
                        and _split_loss_fn._residual_cache
                        and _problem_spatial_dim == 1
                        and (epoch - 1) % plot_samples_every == 0):
                    _rc = _split_loss_fn._residual_cache
                    all_x = torch.cat([r[0] for r in _rc], dim=0)
                    all_t = torch.cat([r[1] for r in _rc], dim=0)
                    all_r2 = torch.cat([r[2] for r in _rc], dim=0)
                    _save_adaptive_sampling_heatmap(
                        all_x, all_t, all_r2,
                        None, None,
                        run_dir, epoch, cfg,
                        causal_state=None,
                    )
                    _split_loss_fn._residual_cache.clear()
            else:
                cached_residuals = getattr(model, '_residual_cache', [])
                model._residual_cache_enabled = False
                if (not adaptive_sampling_enabled and cached_residuals
                        and _problem_spatial_dim == 1
                        and (epoch - 1) % plot_samples_every == 0):
                    all_x = torch.cat([r[0] for r in cached_residuals], dim=0)
                    all_t = torch.cat([r[1] for r in cached_residuals], dim=0)
                    all_r2 = torch.cat([r[2] for r in cached_residuals], dim=0)
                    _save_adaptive_sampling_heatmap(
                        all_x, all_t, all_r2,
                        None, None,
                        run_dir, epoch, cfg,
                        causal_state=causal_state,
                    )
                train_data = resample_residual_inplace(
                    train_data, cfg, device,
                    resample_seed=resample_seed,
                    cached_residuals=cached_residuals,
                    run_dir=run_dir,
                    epoch=epoch,
                    causal_state=causal_state,
                )
                torch.set_default_device(None)
                train_loader = _create_dataloader(train_data, cfg['batch_size'], shuffle=True)
                metrics['resample_events'].append({
                    'epoch': epoch,
                    'action': 'resampled',
                    'optimizer': current_optimizer_name
                })
                metrics['optimizer_snapshots'].append({
                    'epoch': epoch,
                    'event': 'resample',
                    **_get_optimizer_snapshot(optimizer, lr_scheduler, step_count),
                })
        elif resample_every > 0 and epoch > 1 and (epoch - 1) % resample_every == 0 and not allow_resample_optimizer:
            # Log when resampling is skipped due to optimizer
            if not hasattr(model, '_resample_skip_logged'):
                model._resample_skip_logged = True
                logger.info(f"  [Resample] Skipping resampling during {current_optimizer_name} (loss landscape stability required)")
            # Save skip event to metrics
            metrics['resample_events'].append({
                'epoch': epoch,
                'action': 'skipped',
                'optimizer': current_optimizer_name,
                'reason': 'optimizer_stability'
            })

        # Train phase
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        _ks_loss_module._nan_ctx[0] = f"epoch {epoch}"

        if current_optimizer_name in ('Adam', 'SOAP'):
            # Adam/SOAP: Mini-batch training (GPU parallelized)
            for batch in train_loader:
                optimizer.zero_grad()
                timer.start('train.loss_fn')
                loss = loss_fn(model, batch)
                timer.stop('train.loss_fn')
                timer.start('train.backward')
                loss.backward()
                timer.stop('train.backward')
                
                # ── Track gradient norms (first batch only, at eval epochs) ──
                if n_train_batches == 0 and (epoch % eval_every == 0 or epoch == 1):
                    _total_gn = 0.0
                    _base_gn = 0.0
                    _exp_gn = 0.0
                    
                    # Base model gradient norm
                    if hasattr(model, 'base_model'):
                        for p in model.base_model.parameters():
                            if p.grad is not None:
                                _base_gn += p.grad.data.norm().item() ** 2
                        _base_gn = _base_gn ** 0.5
                    
                    # Experts gradient norm
                    if hasattr(model, 'experts') and model.experts:
                        for exp in model.experts:
                            for p in exp.parameters():
                                if p.grad is not None:
                                    _exp_gn += p.grad.data.norm().item() ** 2
                        _exp_gn = _exp_gn ** 0.5
                    
                    # Total gradient norm
                    for p in model.parameters():
                        if p.grad is not None:
                            _total_gn += p.grad.data.norm().item() ** 2
                    _total_gn = _total_gn ** 0.5
                    
                    # Store for this epoch (will be logged in should_evaluate block)
                    ctx._epoch_grad_norms = {
                        'total': _total_gn,
                        'base': _base_gn,
                        'experts': _exp_gn
                    }

                # DIAGNOSTIC: Gradient flow analysis (gated by debug_prints, every 100 epochs)
                if cfg.get('debug_prints', False) and n_train_batches == 0 and epoch % 100 == 0:
                    _net = getattr(model, 'base_model', model)
                    
                    # Alpha gradients (PirateNet specific)
                    _alpha_grads = []
                    _alpha_vals = []
                    for name, param in _net.named_parameters():
                        if 'alpha' in name and param.grad is not None:
                            _alpha_grads.append((name, param.grad.norm().item(), param.item()))
                            _alpha_vals.append(param.item())
                    if _alpha_grads:
                        _ag_str = ', '.join(f'{g:.2e}' for _, g, _ in _alpha_grads)
                        logger.info(f"  [GradDiag] alpha grads: [{_ag_str}]")
                    
                    # Per-layer gradient norms (top 5 smallest non-zero)
                    _layer_grads = []
                    for name, param in _net.named_parameters():
                        if param.grad is not None:
                            _gn = param.grad.norm().item()
                            if _gn > 0:
                                _layer_grads.append((name, _gn, param.data.norm().item()))
                    if _layer_grads:
                        _layer_grads.sort(key=lambda x: x[1])  # sort by grad norm
                        _smallest = _layer_grads[:3]
                        _largest = _layer_grads[-3:]
                        _sm_str = ', '.join(f'{n.split(".")[-1]}={g:.2e}' for n, g, _ in _smallest)
                        _lg_str = ', '.join(f'{n.split(".")[-1]}={g:.2e}' for n, g, _ in _largest)
                        logger.info(f"  [GradDiag] smallest grads: [{_sm_str}]")
                        logger.info(f"  [GradDiag] largest grads: [{_lg_str}]")
                        
                        # Gradient/weight ratio (indicates update magnitude)
                        _ratios = [(n, g/w if w > 0 else 0) for n, g, w in _layer_grads]
                        _ratios.sort(key=lambda x: x[1])
                        _ratio_str = ', '.join(f'{n.split(".")[-1]}={r:.2e}' for n, r in _ratios[:3])
                        logger.info(f"  [GradDiag] grad/weight ratios (smallest): [{_ratio_str}]")

                # DIAGNOSTIC: Check gradients immediately after backward (early epochs only, configurable)
                enable_grad_diag = adaptive_cfg.get('enable_gradient_diagnostics', False) if is_adaptive else False
                if enable_grad_diag and n_train_batches == 0 and hasattr(model, 'num_experts') and model.num_experts > 0 and epoch <= 10:
                    logger.info(f"\n[DIAG Epoch {epoch}] Checking gradients after backward pass:")
                    for i, expert in enumerate(model.experts):
                        layer_names = expert.get_layer_names()
                        if layer_names:
                            first_layer = expert.network[layer_names[0]]
                            final_layer = expert.network[layer_names[-1]]
                            first_grad = first_layer.weight.grad
                            final_grad = final_layer.weight.grad
                            logger.info(f"  Expert {i}: first_layer.grad={'None' if first_grad is None else f'norm={first_grad.norm().item():.6f}'}, "
                                  f"final_layer.grad={'None' if final_grad is None else f'norm={final_grad.norm().item():.6f}'}")

                # Split clip: experts at expert_grad_clip_norm (tighter), base at grad_clip_norm.
                # When no experts exist (base-only phase), falls back to grad_clip_norm for all.
                _exp_clip_ps = ([p for exp in model.experts for p in exp.parameters()
                                  if p.requires_grad]
                                 if hasattr(model, 'experts') and model.experts else [])
                _base_clip_ps = ([p for p in model.base_model.parameters() if p.requires_grad]
                                  if hasattr(model, 'base_model') else [])
                if _exp_clip_ps and expert_grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(_exp_clip_ps, expert_grad_clip_norm)
                    if _base_clip_ps and grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(_base_clip_ps, grad_clip_norm)
                elif grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip_norm)
                timer.start('train.optim_step')
                
                # DIAGNOSTIC: Track parameter values before step for update magnitude calculation
                _param_before = None
                if cfg.get('debug_prints', False) and n_train_batches == 0 and epoch % 100 == 0:
                    _net = getattr(model, 'base_model', model)
                    _param_before = {name: param.data.clone() for name, param in _net.named_parameters() if param.requires_grad}
                
                optimizer.step()
                timer.stop('train.optim_step')
                
                # DIAGNOSTIC: Compute actual parameter update magnitudes
                if _param_before is not None:
                    _net = getattr(model, 'base_model', model)
                    _update_norms = []
                    _alpha_updates = []
                    for name, param in _net.named_parameters():
                        if name in _param_before:
                            _delta = (param.data - _param_before[name]).norm().item()
                            _update_norms.append((name, _delta, param.data.norm().item()))
                            if 'alpha' in name:
                                _alpha_updates.append((name, _delta, param.item()))
                    
                    # Report alpha updates specifically
                    if _alpha_updates:
                        _au_str = ', '.join(f'{d:.2e}' for _, d, _ in _alpha_updates)
                        logger.info(f"  [UpdateDiag] alpha update magnitudes: [{_au_str}]")
                    
                    # Overall update stats
                    if _update_norms:
                        _total_update = sum(d for _, d, _ in _update_norms)
                        _total_weight = sum(w for _, _, w in _update_norms)
                        logger.info(f"  [UpdateDiag] total update norm: {_total_update:.4e}, "
                              f"total weight norm: {_total_weight:.2f}, "
                              f"ratio: {_total_update/_total_weight:.2e}")

                step_count += 1
                if lr_scheduler is not None and current_optimizer_name != 'LBFGS':
                    lr_scheduler.step()

                train_loss += loss.item()
                n_train_batches += 1

                # DIAGNOSTIC: Track expert gradients and outputs (first batch only per epoch, configurable)
                enable_grad_diag = adaptive_cfg.get('enable_gradient_diagnostics', False) if is_adaptive else False
                if enable_grad_diag and n_train_batches == 1 and is_adaptive and hasattr(model, 'num_experts') and model.num_experts > 0:
                    with torch.no_grad():
                        # Check expert gradients
                        expert_grad_norms = []
                        for i, expert in enumerate(model.experts):
                            layer_names = expert.get_layer_names()
                            if layer_names and hasattr(expert.network[layer_names[0]], 'weight'):
                                first_layer = expert.network[layer_names[0]]
                                if first_layer.weight.grad is not None:
                                    grad_norm = first_layer.weight.grad.norm().item()
                                    expert_grad_norms.append(grad_norm)

                        # Check expert outputs vs base
                        inputs = torch.cat([batch['x'], batch['t']], dim=1)
                        decomp = model.forward_decomposed(inputs)
                        base_norm = decomp['base'].norm().item()
                        expert_norms = [decomp[f'expert_{i}'].norm().item() for i in range(model.num_experts)]
                        total_expert_contrib = sum(expert_norms)

                        # Store for this epoch
                        if not hasattr(model, '_diag_data'):
                            model._diag_data = []
                        model._diag_data.append({
                            'epoch': epoch,
                            'base_norm': base_norm,
                            'expert_norms': expert_norms,
                            'expert_grad_norms': expert_grad_norms,
                            'total_expert_contrib': total_expert_contrib
                        })

        else:
            # LBFGS: Full-batch training with memory error handling
            # Process entire dataset in single forward pass (no batching)
            def closure():
                optimizer.zero_grad()
                # Single forward pass with ALL training data at once
                loss = loss_fn(model, train_data)
                loss.backward()
                # Split clip: experts at expert_grad_clip_norm (tighter), base at grad_clip_norm.
                # When no experts exist (base-only phase), falls back to grad_clip_norm for all.
                _exp_clip_ps = ([p for exp in model.experts for p in exp.parameters()
                                  if p.requires_grad]
                                 if hasattr(model, 'experts') and model.experts else [])
                _base_clip_ps = ([p for p in model.base_model.parameters() if p.requires_grad]
                                  if hasattr(model, 'base_model') else [])
                if _exp_clip_ps and expert_grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(_exp_clip_ps, expert_grad_clip_norm)
                    if _base_clip_ps and grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(_base_clip_ps, grad_clip_norm)
                elif grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip_norm)
                return loss
            
            try:
                timer.start('train.lbfgs_step')
                # LBFGS step processes entire dataset via closure
                loss = optimizer.step(closure)
                timer.stop('train.lbfgs_step')
                train_loss = loss.item()
                n_train_batches = 1

                # DIAGNOSTIC: Track expert gradients and outputs (LBFGS, configurable)
                enable_grad_diag = adaptive_cfg.get('enable_gradient_diagnostics', False) if is_adaptive else False
                if enable_grad_diag and is_adaptive and hasattr(model, 'num_experts') and model.num_experts > 0:
                    with torch.no_grad():
                        # Check expert gradients
                        expert_grad_norms = []
                        for i, expert in enumerate(model.experts):
                            layer_names = expert.get_layer_names()
                            if layer_names and hasattr(expert.network[layer_names[0]], 'weight'):
                                first_layer = expert.network[layer_names[0]]
                                if first_layer.weight.grad is not None:
                                    grad_norm = first_layer.weight.grad.norm().item()
                                    expert_grad_norms.append(grad_norm)

                        # Check expert outputs vs base
                        inputs = torch.cat([train_data['x'][:512], train_data['t'][:512]], dim=1)  # Sample for speed
                        decomp = model.forward_decomposed(inputs)
                        base_norm = decomp['base'].norm().item()
                        expert_norms = [decomp[f'expert_{i}'].norm().item() for i in range(model.num_experts)]
                        total_expert_contrib = sum(expert_norms)

                        # Store for this epoch
                        if not hasattr(model, '_diag_data'):
                            model._diag_data = []
                        model._diag_data.append({
                            'epoch': epoch,
                            'base_norm': base_norm,
                            'expert_norms': expert_norms,
                            'expert_grad_norms': expert_grad_norms,
                            'total_expert_contrib': total_expert_contrib
                        })
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # GPU OOM - stop training and trigger finalize for training curves
                    opt_name = current_optimizer_name.upper()
                    error_msg = (
                        f"\n{'='*60}\n"
                        f"MEMORY ERROR at epoch {epoch}\n"
                        f"{opt_name} ran out of GPU memory. Stopping training.\n"
                        f"Consider: reducing batch_size, dataset size, or\n"
                        f"using a different optimizer.\n"
                        f"{'='*60}\n"
                    )
                    logger.info(error_msg)
                    
                    # Save warning to persistent file
                    warning_log = run_dir / "optimizer_fallback_warning.txt"
                    with open(warning_log, 'a') as f:
                        from datetime import datetime
                        f.write(f"[{datetime.now()}] Epoch {epoch}:\n")
                        f.write(error_msg)
                        f.write(f"Error details: {str(e)}\n\n")
                    
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Signal OOM stop - will trigger finalize for training curves
                    ctx.oom_stopped = True
                    _stop_reason = 'oom'
                    break
                else:
                    raise  # Re-raise other errors

        train_loss /= n_train_batches
        
        # Check for optimizer switch (optimizer_1 → optimizer_2)
        if epoch == switch_epoch and optimizer_2_name is not None:
            logger.info(f"\n{'='*60}")
            logger.info(f"OPTIMIZER SWITCH: {current_optimizer_name} -> {optimizer_2_name.upper()} at epoch {epoch}")
            logger.info(f"{'='*60}\n")
            # Restore default device to CUDA before creating SSBroyden/LBFGS optimizer.
            # This ensures optimizer state tensors (e.g., Hessian approximation) are created
            # on the correct device, not CPU (which can happen if default was reset earlier
            # for DataLoader compatibility in time-marching windows 1+).
            torch.set_default_device(device)
            _prev_opt = current_optimizer_name
            optimizer, current_optimizer_name = _create_optimizer_by_name(
                optimizer_2_name, model, seg_cfg)
            lr_scheduler = None  # optimizer_2 uses its own LR / line search
            # Reset patience at the switch; optimizer_2 gets a fresh grace window.
            epochs_without_improvement = 0
            best_train_loss = float('inf')
            patience_start_epoch = switch_epoch
            metrics['optimizer_events'].append({
                'epoch': epoch,
                'from': _prev_opt,
                'to': current_optimizer_name,
            })
        
        # Store train loss every epoch
        metrics['train_loss_epochs'].append(epoch)
        metrics['train_loss'].append(train_loss)

        # NaN early-stop: save everything and break so the next experiment can run
        if math.isnan(train_loss) or math.isinf(train_loss):
            logger.info(f"\n{'!'*60}")
            logger.info(f"  [NaN] Training diverged at epoch {epoch} — saving diagnostics and stopping.")

            # Diagnose which loss component went NaN
            try:
                with torch.no_grad():
                    _diag_batch = next(iter(train_loader))
                    _comps = loss_fn(model, _diag_batch, return_components=True)
                    logger.info(f"  [NaN] Loss components: " +
                          ", ".join(f"{k}={v.item():.6g}" for k, v in _comps.items()))
                    metrics['nan_components'] = {k: float(v.item()) for k, v in _comps.items()}
            except Exception as _e:
                logger.info(f"  [NaN] Could not compute loss components: {_e}")

            metrics['nan_divergence'] = {'epoch': epoch, 'train_loss': train_loss}
            metrics['training_time_seconds'] = time.time() - start_time

            # Save metrics JSON so the run is inspectable
            _nan_metrics_path = run_dir / "metrics.json"
            with open(_nan_metrics_path, 'w') as _f:
                json.dump(metrics, _f, indent=2, cls=_NumpySafeEncoder)
            logger.info(f"  [NaN] Metrics saved to {_nan_metrics_path}")

            # Save a NaN-state checkpoint for post-mortem inspection
            _nan_ckpt_path = checkpoint_dir / f"nan_checkpoint_epoch_{epoch}.pt"
            _save_checkpoint(_nan_ckpt_path, model, optimizer, current_optimizer_name,
                             epoch, train_loss, eval_loss, cfg, metrics)
            logger.info(f"  [NaN] Checkpoint saved to {_nan_ckpt_path}")
            logger.info(f"{'!'*60}\n")
            _nan_detected = True
            break

        # LRA: update adaptive loss weights periodically
        if lra_weights is not None and epoch > 0 and epoch % lra_weights.update_every == 0:
            try:
                batch_for_lra = next(iter(train_loader))
                lra_weights.update(model, loss_fn, batch_for_lra)
                if epoch % print_every == 0:
                    w_str = ', '.join(f'{k}={v:.4f}' for k, v in lra_weights.weights.items())
                    logger.info(f"  [LRA] weights: {w_str}")
            except Exception as e:
                logger.info(f"  [LRA] Weight update failed at epoch {epoch}: {e}")

        # Causal weighting: check if epsilon should advance
        causal_state = getattr(loss_fn, 'causal_state', None)
        causal_epoch_min_weight = None
        if causal_state is not None:
            causal_epoch_min_weight = causal_state['min_weight']
        if advance_causal_schedule(causal_state):
            cs = loss_fn.causal_state
            logger.info(f"  [Causal] epsilon advanced to "
                  f"{cs['tol']:.2f} "
                  f"(stage {cs['schedule_idx']+1}/"
                  f"{len(cs['schedule'])}, "
                  f"prev_min_w={causal_epoch_min_weight:.6f})")
        # Reset min_weight AFTER advance check so it sees the true minimum.
        if causal_state is not None:
            causal_state['min_weight'] = 1.0

        # Compute evaluation metrics only every print_every epochs or last epoch
        # This speeds up training significantly for physics-informed losses
        should_evaluate = (epoch % eval_every == 0 or epoch == 1 or epoch == total_epochs)
        
        if should_evaluate:
            # Compute train rel-L2 and infinity norm errors

            # Eval phase
            model.eval()
            eval_loss = 0.0
            # Accumulate squared sums for correct global rel-L2 computation
            # (averaging per-batch rel-L2 is mathematically incorrect)
            total_diff_sq = 0.0
            total_gt_sq = 0.0
            eval_inf_norm = 0.0  # Track max across all batches
            n_eval_batches = 0

            # Eval metric uses the configured blending_mode (composed forward),
            # so the rel-L2 curve reflects the actual inference-time composition.
            for batch in eval_loader:
                # Note: For physics-informed losses, we need gradients w.r.t. inputs
                # even during evaluation (for computing derivatives in PDE residuals).
                # We still use model.eval() to disable dropout/batchnorm training behavior.
                timer.start('eval.loss_fn')
                loss = loss_fn(model, batch, update_causal_state=False)
                timer.stop('eval.loss_fn')

                with torch.no_grad():
                    inputs = torch.cat([batch['x'], batch['t']], dim=1)
                    timer.start('eval.h_pred')
                    h_pred = model(inputs)
                    timer.stop('eval.h_pred')
                    # Accumulate squared differences and GT norms for global rel-L2
                    diff = h_pred - batch['h_gt']
                    total_diff_sq += (diff ** 2).sum().item()
                    total_gt_sq += (batch['h_gt'] ** 2).sum().item()
                    # Track max inf_norm across all batches
                    inf_norm = compute_infinity_norm_error(h_pred, batch['h_gt'])
                    eval_inf_norm = max(eval_inf_norm, inf_norm.item())

                eval_loss += loss.item()
                n_eval_batches += 1

            eval_loss /= n_eval_batches
            # Compute global rel-L2: ||pred - gt||_2 / ||gt||_2
            eval_rel_l2 = math.sqrt(total_diff_sq) / (math.sqrt(total_gt_sq) + 1e-10)

            # Store evaluation metrics (train_loss already stored above for all epochs)
            metrics['epochs'].append(epoch)
            metrics['eval_loss'].append(eval_loss)
            metrics['eval_rel_l2'].append(eval_rel_l2)
            metrics['eval_inf_norm'].append(eval_inf_norm)
            
            # ── Compute and log term-wise loss components ──
            try:
                _comp_batch = next(iter(eval_loader))
                _loss_comps = loss_fn(model, _comp_batch, return_components=True, update_causal_state=False)
                _comp_dict = {k: float(v.item()) if isinstance(v, torch.Tensor) else float(v) 
                              for k, v in _loss_comps.items()}
                
                # Store in metrics for plotting
                metrics['loss_components']['epochs'].append(epoch)
                for term in ['residual', 'ic', 'bc']:
                    val = _comp_dict.get(term, 0.0)
                    metrics['loss_components'][term].append(val)
                
                # Log components at evaluation epochs
                _comp_str = ', '.join(f'{k}={v:.6f}' for k, v in _comp_dict.items())
                logger.info(f"  [LossTerms] {_comp_str}")
                
                # Store in history with full details
                metrics['loss_components_history'].append({
                    'epoch': epoch,
                    **_comp_dict
                })
            except Exception as _comp_err:
                logger.info(f"  [LossTerms] Failed to compute: {_comp_err}")

            # Per-expert split-loss breakdown
            _split_ctx = getattr(ctx, '_split_context', None)
            if _split_ctx is not None and hasattr(loss_fn, '_per_expert_history'):
                _peh = loss_fn._per_expert_history
                for _eidx in sorted(_peh.keys()):
                    _eh = _peh[_eidx]
                    _last = {k: v[-1] for k, v in _eh.items() if v}
                    _s = ', '.join(
                        f'{k}={v:.6f}' for k, v in _last.items()
                    )
                    logger.info(
                        f"  [SplitTerms] expert={_eidx} {_s}"
                    )

        # End epoch timing (handles printing based on print_every)
        timer.end_epoch()

        # Print progress
        if should_evaluate:
            elapsed = time.time() - start_time
            batch_mode = "mini" if current_optimizer_name in ('Adam', 'SOAP') else "full"
            logger.info(f"Epoch [{epoch}/{total_epochs}] ({elapsed:.1f}s) [{current_optimizer_name}/{batch_mode}] | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Eval Loss: {eval_loss:.6f} | "
                  f"Eval Rel-L2: {eval_rel_l2:.6f} | "
                  f"Eval Inf: {eval_inf_norm:.6f}")

            # DIAGNOSTIC: Causal weight progression
            if causal_state is not None and causal_epoch_min_weight is not None:
                cs = causal_state
                stage_str = f"{cs['schedule_idx']+1}/{len(cs['schedule'])}"
                logger.info(f"  [Causal] tol={cs['tol']:.2f}, stage={stage_str}, min_weight={causal_epoch_min_weight:.6f}")
                metrics['causal_history'].append({
                    'epoch': epoch,
                    'tol': float(cs['tol']),
                    'stage': int(cs['schedule_idx']),
                    'stage_total': len(cs['schedule']),
                    'min_weight': float(causal_epoch_min_weight),
                    'threshold': float(cs['threshold'])
                })

            # DIAGNOSTIC: LRA weights and gradient norms
            if lra_weights is not None:
                w = lra_weights.weights
                g = lra_weights.last_grad_norms
                w_str = ', '.join(f'{k}={v:.4f}' for k, v in w.items())
                g_str = ', '.join(f'{k}={g.get(k, 0):.6f}' for k in w)
                logger.info(f"  [LRA] weights: {w_str} | grads: {g_str}")
                # Save to metrics
                metrics['lra_history'].append({
                    'epoch': epoch,
                    'weights': {k: float(v) for k, v in w.items()},
                    'grad_norms': {k: float(g.get(k, 0)) for k in w},
                })
            
            # ── Log gradient norms (computed during backward pass) ──
            _gn = getattr(ctx, '_epoch_grad_norms', None)
            if _gn is not None:
                logger.info(f"  [GradNorm] total={_gn['total']:.4e}, base={_gn['base']:.4e}, experts={_gn['experts']:.4e}")
                metrics['gradient_norms']['epochs'].append(epoch)
                metrics['gradient_norms']['total_grad_norm'].append(_gn['total'])
                metrics['gradient_norms']['base_grad_norm'].append(_gn['base'])
                metrics['gradient_norms']['experts_grad_norm'].append(_gn['experts'])
            
            # ── Log current learning rate ──
            _current_lr = seg_cfg['lr']  # Default from config
            if lr_scheduler is not None:
                try:
                    _current_lr = lr_scheduler.get_last_lr()[0]
                except:
                    pass
            elif hasattr(optimizer, 'param_groups'):
                _current_lr = optimizer.param_groups[0].get('lr', _current_lr)
            logger.info(f"  [LR] current={_current_lr:.6e}")
            metrics['lr_history']['epochs'].append(epoch)
            metrics['lr_history']['lr'].append(_current_lr)

            # DIAGNOSTIC: Full loss-term breakdown (raw → grad → weight → weighted-grad)
            # Shows exactly what the optimizer sees, to diagnose why updates are tiny.
            # ||sum|| << individual weighted grads ⇒ terms cancel (gradient conflict).
            if cfg.get('debug_prints', False) and lra_weights is not None:
                try:
                    _dbg_batch = next(iter(train_loader))
                    _dbg_params = [p for p in model.parameters() if p.requires_grad]
                    _raw_comps = loss_fn(model, _dbg_batch, return_components=True)
                    _w = lra_weights.weights
                    _raw_vals, _raw_gn, _wtd_gn = {}, {}, {}
                    _weighted_grad_flats = []
                    for _k, _v in _raw_comps.items():
                        _raw_vals[_k] = _v.item()
                        if isinstance(_v, torch.Tensor) and _v.requires_grad:
                            _grads = torch.autograd.grad(
                                _v, _dbg_params, retain_graph=True, allow_unused=True)
                            _flat = torch.cat([gg.flatten() for gg in _grads if gg is not None])
                            _raw_gn[_k] = _flat.norm().item()
                            _wk = _w.get(_k, 1.0)
                            _wtd_gn[_k] = _wk * _raw_gn[_k]
                            _weighted_grad_flats.append(_wk * _flat)
                        else:
                            _raw_gn[_k] = 0.0
                            _wtd_gn[_k] = 0.0
                    model.zero_grad(set_to_none=True)
                    # Norm of the summed weighted gradient = actual update-direction magnitude
                    _total_wg = 0.0
                    if _weighted_grad_flats:
                        _total_wg = torch.stack(_weighted_grad_flats, dim=0).sum(dim=0).norm().item()
                    _keys = list(_raw_comps.keys())
                    logger.info("  [LossDiag] raw terms:      " +
                          ', '.join(f'{k}={_raw_vals[k]:.4e}' for k in _keys))
                    logger.info("  [LossDiag] raw grad norms: " +
                          ', '.join(f'{k}={_raw_gn[k]:.4e}' for k in _keys))
                    logger.info("  [LossDiag] LRA weights:    " +
                          ', '.join(f'{k}={_w.get(k, 1.0):.4f}' for k in _keys))
                    logger.info("  [LossDiag] weighted terms: " +
                          ', '.join(f'{k}={_w.get(k, 1.0) * _raw_vals[k]:.4e}' for k in _keys))
                    logger.info("  [LossDiag] weighted grads: " +
                          ', '.join(f'{k}={_wtd_gn[k]:.4e}' for k in _keys) +
                          f"  (||sum||={_total_wg:.4e})")
                except Exception as _e:
                    logger.info(f"  [LossDiag] failed: {_e}")

            # DIAGNOSTIC: PirateNet alphas, causal chunks, LR
            if cfg.get('debug_prints', False):
                # PirateNet alpha cold-start check
                _net = getattr(model, 'base_model', model)
                if hasattr(_net, 'debug_state'):
                    _ds = _net.debug_state()
                    _alphas_str = ', '.join(
                        f'{a:.4f}' for a in _ds['alphas'])
                    _wn0 = (
                        _ds['block_w_norms'][0]
                        if _ds['block_w_norms'] else []
                    )
                    _wn0_str = '/'.join(f'{w:.3f}' for w in _wn0)
                    logger.info(
                        f"  [PirateNet] alphas=[{_alphas_str}] | "
                        f"W-norms(block0)=[{_wn0_str}]"
                    )

                # Per-chunk causal breakdown
                _cs = causal_state
                if _cs is not None and 'last_weights' in _cs:
                    _w_str = ', '.join(
                        f'{w:.3f}' for w in _cs['last_weights'])
                    _cl_str = ', '.join(
                        f'{cl:.2e}'
                        for cl in _cs['last_chunk_losses']
                    )
                    _t_str = ', '.join(
                        f'{t:.3f}' for t in _cs['last_chunk_tmax'])
                    logger.info(f"  [CausalChunks] w=[{_w_str}]")
                    logger.info(f"  [CausalChunks] L=[{_cl_str}]")
                    logger.info(f"  [CausalChunks] tmax=[{_t_str}]")

                # LR schedule sanity check (extended)
                _cur_lr = optimizer.param_groups[0]['lr']
                _warmup_steps = cfg.get('lr_warmup_steps', 0)
                _decay_steps = cfg.get('lr_decay_steps', 2000)
                _decay_rate = cfg.get('lr_decay_rate', 0.9)
                _base_lr = cfg.get('lr', 0.001)
                
                # Calculate expected LR
                if step_count <= _warmup_steps:
                    _phase = "warmup"
                    _expected_lr = _base_lr * (cfg.get('lr_warmup_start_factor', 0.01) + 
                                               (1 - cfg.get('lr_warmup_start_factor', 0.01)) * step_count / _warmup_steps)
                else:
                    _steps_after_warmup = step_count - _warmup_steps
                    _num_decays = _steps_after_warmup // _decay_steps
                    _expected_lr = _base_lr * (_decay_rate ** _num_decays)
                    _phase = f"decay (n={_num_decays})"
                
                _lr_match = "✓" if abs(_cur_lr - _expected_lr) / _expected_lr < 0.01 else "✗"
                logger.info(
                    f"  [LR] lr={_cur_lr:.2e} (expected={_expected_lr:.2e} {_lr_match}) | "
                    f"step={step_count} | phase={_phase}"
                )

            # DIAGNOSTIC: Unweighted loss component breakdown
            # Compute on a sample eval batch
            try:
                sample_batch = next(iter(eval_loader))
                # Get the original loss function (unwrap LRA if present)
                orig_loss_fn = getattr(loss_fn, '__wrapped__', loss_fn)
                if hasattr(orig_loss_fn, '__self__'):  # It's a method/closure
                    # For the LRA wrapper, we need to access _orig_loss_fn from the closure
                    if '_orig_loss_fn' in dir(loss_fn):
                        orig_loss_fn = loss_fn.__code__.co_consts  # This won't work, need different approach
                # Actually, just call with return_components=True which the wrapper forwards
                with torch.no_grad():
                    components = loss_fn(model, sample_batch, return_components=True)
                    comps_str = ', '.join(f'{k}={v:.6f}' for k, v in components.items())
                    logger.info(f"  [Loss] components: {comps_str} (unweighted)")
                    metrics['loss_components_history'].append({
                        'epoch': epoch,
                        'residual': float(components['residual'].item()),
                        'ic': float(components['ic'].item()),
                        'bc': float(components['bc'].item()),
                    })
            except Exception as e:
                # Don't crash if component breakdown fails
                pass

            # DIAGNOSTIC: Print expert contributions (configurable)
            enable_grad_diag = adaptive_cfg.get('enable_gradient_diagnostics', False) if is_adaptive else False
            if enable_grad_diag and is_adaptive and hasattr(model, 'num_experts') and model.num_experts > 0 and hasattr(model, '_diag_data') and model._diag_data:
                latest_diag = model._diag_data[-1]
                base_norm = latest_diag['base_norm']
                total_expert = latest_diag['total_expert_contrib']
                expert_norms = latest_diag['expert_norms']
                expert_grads = latest_diag['expert_grad_norms']

                logger.info(f"  [DIAG] Base norm: {base_norm:.6f} | Expert contrib: {total_expert:.6f} | Ratio: {total_expert/base_norm if base_norm > 0 else 0:.4f}")
                logger.info(f"  [DIAG] Expert norms: {[f'{x:.4f}' for x in expert_norms[:5]]}" + ("..." if len(expert_norms) > 5 else ""))
                if expert_grads:
                    logger.info(f"  [DIAG] Expert grad norms: {[f'{x:.6f}' for x in expert_grads[:5]]}" + ("..." if len(expert_grads) > 5 else ""))

        # Save checkpoint periodically (only when we have eval metrics)
        if epoch % save_every == 0 and eval_loss is not None:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            _save_checkpoint(checkpoint_path, model, optimizer, current_optimizer_name, epoch,
                           train_loss, eval_loss, cfg, metrics)
            logger.info(f"  Checkpoint saved: {checkpoint_path}")

        # Save best model (only when we have eval metrics)
        if eval_loss is not None and eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_checkpoint_path = checkpoint_dir / "best_model.pt"
            _save_checkpoint(best_checkpoint_path, model, optimizer, current_optimizer_name, epoch,
                           train_loss, eval_loss, cfg, metrics)

        # Patience-based early stopping on train loss. Active for BOTH optimizers
        # (Step 5): the relative-improvement plateau test runs from the segment
        # start. On an optimizer_1 plateau we fast-forward to the switch epoch (so
        # the existing switch handler fires and optimizer_2 keeps its full budget)
        # rather than stopping; an optimizer_2 (or no-switch) plateau stops the
        # segment. The relative min-delta means a loss creeping down by a negligible
        # amount each epoch still counts as "no improvement".
        if (train_loss is not None and patience_epochs > 0
                and epoch >= patience_start_epoch):
            if train_loss < best_train_loss * (1.0 - patience_rel_delta):
                best_train_loss = train_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            # seg_min_epochs is a grace period measured within the active window.
            if (epoch - patience_start_epoch >= seg_min_epochs
                    and epochs_without_improvement >= patience_epochs):
                _in_optimizer_1 = (optimizer_2_name is not None
                                   and epoch < switch_epoch)
                if _in_optimizer_1 and switch_epoch < total_epochs:
                    # Fast-forward to the switch; preserves optimizer_2's budget.
                    logger.info(f"\n  [Patience] optimizer_1 plateau "
                          f">{patience_rel_delta:.1%} for "
                          f"{epochs_without_improvement} epochs at epoch {epoch}; "
                          f"fast-forwarding to switch epoch {switch_epoch}.")
                    metrics['plateau_events'].append({
                        'epoch': epoch,
                        'action': 'optimizer_1_fast_forward',
                        'switch_epoch': switch_epoch,
                    })
                    epoch = switch_epoch - 1
                    ctx.epoch = epoch
                    epochs_without_improvement = 0
                    best_train_loss = float('inf')
                    continue
                else:
                    logger.info(f"\n  [EarlyStop] No train loss improvement "
                          f">{patience_rel_delta:.1%} for "
                          f"{epochs_without_improvement} epochs "
                          f"(best={best_train_loss:.6f}). "
                          f"Stopping segment at epoch {epoch}.")
                    _stopped_early = True
                    _stop_reason = 'early_stop'
                    break

    # ── Write reassigned segment state back to ctx ──
    # (objects mutated in place — model, metrics, timer — need no write-back.)
    ctx.epoch = epoch
    ctx.total_epochs = epoch
    ctx.optimizer = optimizer
    ctx.current_optimizer_name = current_optimizer_name
    ctx.lr_scheduler = lr_scheduler
    ctx.step_count = step_count
    ctx.switch_epoch = switch_epoch
    ctx.optimizer_2_name = optimizer_2_name
    ctx.best_eval_loss = best_eval_loss
    ctx.best_train_loss = best_train_loss
    ctx.best_checkpoint_path = best_checkpoint_path
    ctx.train_loss = train_loss
    ctx.eval_loss = eval_loss
    ctx.eval_rel_l2 = eval_rel_l2
    ctx.eval_inf_norm = eval_inf_norm
    ctx.train_data = train_data
    ctx.train_loader = train_loader
    ctx._nan_detected = _nan_detected

    if _nan_detected:
        _stop_reason = 'nan'
    if not _nan_detected:
        _save_segment_pred_plot(ctx, segment_name)
    _final_tl = train_loss if train_loss is not None else float('nan')
    _final_el = eval_loss if eval_loss is not None else float('nan')
    _oom_stopped = getattr(ctx, 'oom_stopped', False)
    
    # Save segment-end checkpoint
    _save_segment_checkpoint(ctx, segment_name, epoch, optimizer, current_optimizer_name,
                             train_loss, eval_loss, metrics, cfg)
    
    logger.info(f"[Segment:{segment_name}] done | ran {epoch - segment_start_epoch} "
          f"epochs (stop={_stop_reason}) | "
          f"train_loss={_final_tl:.6f} eval_loss={_final_el:.6f}")
    return SegmentResult(
        nan_detected=_nan_detected,
        stopped_early=_stopped_early,
        stop_reason=_stop_reason,
        epochs_run=epoch - segment_start_epoch,
        final_train_loss=_final_tl,
        final_eval_loss=_final_el,
        oom_stopped=_oom_stopped,
    )


# ======================================================================
# Staged-spawning helpers (orchestrator level; called between segments)
# ======================================================================

def _save_segment_checkpoint(ctx: TrainingContext, segment_name: str, epoch: int,
                             optimizer, optimizer_name: str, train_loss: float,
                             eval_loss: float, metrics: Dict, cfg: Dict) -> None:
    """Save checkpoint at the end of a training segment.
    
    Creates a checkpoint file named `checkpoint_after_<segment>.pt` in the 
    checkpoint directory. This captures the model state at each stage boundary
    (root, level_1, level_2, ..., fine_tune, phase3) for debugging and recovery.
    """
    checkpoint_dir = ctx.checkpoint_dir
    if checkpoint_dir is None:
        return
    
    try:
        checkpoint_path = checkpoint_dir / f"checkpoint_after_{segment_name}.pt"
        _save_checkpoint(checkpoint_path, ctx.model, optimizer, optimizer_name, epoch,
                        train_loss, eval_loss, cfg, metrics)
        logger.info(f"  [Segment:{segment_name}] saved checkpoint_after_{segment_name}.pt")
    except Exception as e:
        logger.info(f"  [Segment:{segment_name}] checkpoint save failed: {e}")


def _save_segment_pred_plot(ctx: TrainingContext, segment_name: str) -> None:
    """Save ``pred_after_<segment>.png`` (1D problems with ground truth).

    Captures the full-composition prediction at the end of a segment so each
    stage (root, every level, fine-tune, joint) leaves a visual checkpoint.
    """
    gt_grid = ctx.gt_grid
    if gt_grid is None or ctx.problem_cfg.get('spatial_dim', None) != 1:
        return
    out_dir = ctx.adaptive_plots_dir or ctx.run_dir
    if out_dir is None:
        return
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_spawn_prediction_plot(
            model=ctx.model,
            domain_bounds=ctx.domain_bounds,
            gt_grid=gt_grid,
            grid_x=ctx.gt_x,
            grid_t=ctx.gt_t,
            output_path=out_dir / f"pred_after_{segment_name}.png",
            epoch=ctx.epoch,
            cfg=ctx.cfg,
        )
        logger.info(f"  [Segment:{segment_name}] saved pred_after_{segment_name}.png")
    except Exception as _e:
        logger.info(f"  [Segment:{segment_name}] prediction plot failed: {_e}")


def _set_trainable(model: nn.Module, which: str, verbose: bool = True) -> int:
    """Set ``requires_grad`` across the model for a training segment.

    ``which``:
      * ``'all'``      — every parameter trainable (root w/o experts, joint fine-tune).
      * ``'base'``     — only the base/root network (Phase-1 root segment).
      * ``'leaves'``   — all leaf experts trainable, base frozen (Phase 3).

    Frozen params still participate in the forward composition; only the
    optimizer (which filters on ``requires_grad``) skips them. Returns the
    count of trainable param tensors.
    """
    trainable_details = []
    
    if which == 'all':
        for p in model.parameters():
            p.requires_grad = True
        trainable_details.append("ALL params trainable")
    else:
        for p in model.parameters():
            p.requires_grad = False
        base = getattr(model, 'base_model', None)
        if which == 'base':
            if base is not None:
                for p in base.parameters():
                    p.requires_grad = True
                trainable_details.append("base_model: TRAINABLE")
            else:
                for p in model.parameters():
                    p.requires_grad = True
                trainable_details.append("no base_model attr; all params: TRAINABLE")
        elif which == 'leaves':
            experts = getattr(model, 'experts', [])
            trainable_details.append("base_model: FROZEN")
            for idx, expert in enumerate(experts):
                for p in expert.parameters():
                    p.requires_grad = True
                trainable_details.append(f"  expert[{idx}]: TRAINABLE")
        else:
            raise ValueError(f"_set_trainable: unknown which={which!r}")
    
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    n_total = sum(1 for _ in model.parameters())
    
    if verbose:
        logger.info(f"\n[DEBUG] _set_trainable(which='{which}'):")
        logger.info(f"  Total params: {n_total}, Trainable: {n_trainable}")
        for detail in trainable_details:
            logger.info(f"  {detail}")
    
    return n_trainable


def _build_tree_once(ctx: TrainingContext, retain_siblings: bool) -> Dict:
    """Fit the M-term tree once from the current model's eval prediction.

    No experts are spawned here. Returns a dict carrying the accepted nodes and
    the tree maps needed to select levels and link parent experts.
    """
    model = ctx.model
    eval_data = ctx.eval_data
    region_detector = ctx.region_detector
    adaptive_cfg = ctx.adaptive_cfg
    variable_for_node_accept = ctx.variable_for_node_accept
    M = adaptive_cfg['M_experts_num']

    model.eval()
    with torch.no_grad():
        eval_inputs = torch.cat([eval_data['x'], eval_data['t']], dim=1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        u_pred = model(eval_inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    X_eval = eval_inputs.cpu().numpy()
    y_eval = u_pred.cpu().numpy()

    closure_desc = ("ancestors-only" if not retain_siblings
                    else "ancestors+siblings")
    logger.info(f"\n[Tree] Computing M-term tree (retain_siblings={retain_siblings}) — "
          f"closure: {closure_desc}")
    logger.info(f"  [M-term Tree] Fitting full tree (max_depth={region_detector.max_depth}, "
          f"min_samples_leaf={region_detector.min_samples_leaf}), selecting top M={M}...")
    accepted_nodes, prune_depth_stats = region_detector.fit_full_tree_and_prune(
        X=X_eval, y=y_eval, M=M,
        variable_for_node_accept=variable_for_node_accept,
        verbose=True, retain_siblings=retain_siblings,
    )

    tree = region_detector.rf.estimators_[0].tree_
    children_left = tree.children_left
    children_right = tree.children_right

    from collections import deque as _deque
    parent_map = {}
    node_tree_depth = {0: 0}
    _bfs = _deque([0])
    while _bfs:
        nid = _bfs.popleft()
        for child in (children_left[nid], children_right[nid]):
            if child != -1:
                parent_map[child] = nid
                node_tree_depth[child] = node_tree_depth[nid] + 1
                _bfs.append(child)

    logger.info(f"  [Tree] Accepted {len(accepted_nodes)} node(s) of "
          f"{int(tree.node_count)} tree nodes.")
    return {
        'accepted_nodes': accepted_nodes,
        'tree': tree,
        'children_left': children_left,
        'parent_map': parent_map,
        'node_tree_depth': node_tree_depth,
        'prune_depth_stats': prune_depth_stats,
    }


def _select_levels(ctx: TrainingContext, build_result: Dict, leaves_only: bool):
    """Group spawnable nodes into ordered levels (coarse → fine).

    Returns ``(levels, nodes_to_spawn)`` where ``levels`` is a list (ordered by
    increasing tree depth) of lists of ``(node, parent_tree_id)``.
    """
    accepted_nodes = build_result['accepted_nodes']
    children_left = build_result['children_left']
    node_tree_depth = build_result['node_tree_depth']
    if leaves_only:
        accepted_ids = {n.node_id for n, _ in accepted_nodes}
        nodes_to_spawn = [
            (node, parent_id) for node, parent_id in accepted_nodes
            if children_left[node.node_id] == -1
            or children_left[node.node_id] not in accepted_ids
        ]
    else:
        nodes_to_spawn = list(accepted_nodes)

    from collections import defaultdict as _dd
    by_depth = _dd(list)
    for node, parent_id in nodes_to_spawn:
        by_depth[node_tree_depth.get(node.node_id, 1)].append((node, parent_id))
    levels = [by_depth[d] for d in sorted(by_depth)]
    logger.info(f"  [Tree] {len(nodes_to_spawn)} node(s) to spawn across "
          f"{len(levels)} level(s): {[len(lv) for lv in levels]}")
    return levels, nodes_to_spawn


def _record_tree_diagnostics(ctx: TrainingContext, build_result: Dict,
                             nodes_to_spawn) -> None:
    """Append a full per-node tree-diagnostics record to ``metrics``."""
    metrics = ctx.metrics
    region_detector = ctx.region_detector
    epoch = ctx.epoch
    accepted_nodes = build_result['accepted_nodes']
    tree = build_result['tree']
    parent_map = build_result['parent_map']
    node_tree_depth = build_result['node_tree_depth']
    prune_depth_stats = build_result['prune_depth_stats']

    accepted_ids = {n.node_id for n, _ in accepted_nodes}
    spawned_ids = {n.node_id for n, _ in nodes_to_spawn}
    tree_diag_nodes = []
    for nd in region_detector.compute_wavelet_norms():
        if nd.node_id == 0:
            continue
        tree_diag_nodes.append({
            'node_id': nd.node_id,
            'parent_node_id': parent_map.get(nd.node_id, -1),
            'wavelet_norm_squared': nd.wavelet_norm_squared,
            'n_samples': nd.n_samples,
            'is_leaf': bool(nd.is_leaf),
            'bounds_lower': nd.bounds_lower,
            'bounds_upper': nd.bounds_upper,
            'accepted': bool(nd.node_id in accepted_ids),
            'spawned_as_expert': bool(nd.node_id in spawned_ids),
            'tree_depth': node_tree_depth.get(nd.node_id, -1),
        })
    metrics.setdefault('spawning_diagnostics', []).append({
        'epoch': epoch,
        'method': 'M_term_tree_by_norm',
        'M_experts_num': ctx.adaptive_cfg['M_experts_num'],
        'variable_for_node_accept': ctx.variable_for_node_accept,
        'total_tree_nodes': int(tree.node_count),
        'accepted_count': len(accepted_ids),
        'spawned_count': len(spawned_ids),
        'depth_stats': {str(k): v for k, v in prune_depth_stats.items()},
        'nodes': tree_diag_nodes,
    })


def _spawn_nodes(ctx: TrainingContext, level_nodes, copy_output: bool,
                 node_to_expert: Dict, node_tree_depth: Dict):
    """Spawn one level's experts, link parents, init, and apply spectral norm.

    ``node_to_expert`` maps tree-node-id → expert index and is updated in place
    so finer levels can resolve their parent expert. Returns
    ``(num_spawned, new_expert_indices)``.
    """
    model = ctx.model
    cfg = ctx.cfg
    problem_cfg = ctx.problem_cfg
    metrics = ctx.metrics
    epoch = ctx.epoch

    from trainer.init import apply_expert_init, apply_parent_copy_init
    from adaptive.indicators import RegionDescriptor

    init_mode = problem_cfg['init']['hidden']

    new_expert_indices = []
    for node, parent_tree_id in level_nodes:
        parent_expert_idx = node_to_expert.get(parent_tree_id, -1)
        depth = node_tree_depth.get(node.node_id, 1)
        child_region = RegionDescriptor(
            bounds_lower=node.bounds_lower,
            bounds_upper=node.bounds_upper,
            wavelet_norm_squared=node.wavelet_norm_squared,
            new_wavelet_norm_squared=node.new_wavelet_norm_squared,
            spawn_epoch=epoch,
            depth=depth,
            parent_idx=parent_expert_idx,
            smoothness_alpha=node.smoothness_alpha,
        )
        expert_idx = model.spawn_expert(child_region,
                                        copy_from_idx=parent_expert_idx)
        if expert_idx >= 0:
            node_to_expert[node.node_id] = expert_idx
            new_expert_indices.append(expert_idx)
            metrics.setdefault('expert_spawns', []).append({
                'epoch': epoch,
                'expert_idx': expert_idx,
                'region': child_region.to_dict(),
                'depth': depth,
                'parent_idx': parent_expert_idx,
                **({'num_experts': model.num_experts}
                   if hasattr(model, 'num_experts') else {}),
            })

    # Init newly spawned experts (after spawn so copy-init parents already exist).
    logger.info(f"\n[DEBUG] _spawn_nodes: Initializing {len(new_expert_indices)} new experts")
    logger.info(f"  copy_output={copy_output}, init_mode='{init_mode}'")

    for expert_idx in new_expert_indices:
        new_exp = model.experts[expert_idx]
        region = model.regions[expert_idx] if hasattr(model, 'regions') and expert_idx < len(model.regions) else None
        
        # Print region info
        if region:
            logger.info(f"\n  [Expert {expert_idx}] Region bounds: {region.bounds_lower} -> {region.bounds_upper}")
            logger.info(f"    depth={region.depth}, parent_idx={region.parent_idx}, spawn_epoch={region.spawn_epoch}")
        
        if init_mode == 'parent_weights':
            if hasattr(model, 'regions') and expert_idx < len(model.regions):
                par_idx = model.regions[expert_idx].parent_idx
            else:
                par_idx = -1
            parent_model = (model.base_model if par_idx == -1
                            else model.experts[par_idx])
            par_label = 'base' if par_idx == -1 else f'expert {par_idx}'
            apply_parent_copy_init(new_exp, parent_model, cfg,
                                   copy_output=copy_output)
            logger.info(f"    [ParentInit] Expert {expert_idx}: copied from {par_label}, copy_output={copy_output}")
        else:
            zero_output = not copy_output
            apply_expert_init(new_exp, cfg, zero_output=zero_output)
            output_init = 'zeroed' if zero_output else f'{init_mode}'
            logger.info(f"    [Init] Expert {expert_idx}: hidden='{init_mode}', output='{output_init}'")

        # Print output layer state after init
        from trainer.init import _get_output_layer
        out_layer = _get_output_layer(new_exp)
        out_weight_norm = out_layer.weight.data.norm().item()
        out_bias_val = out_layer.bias.data.mean().item() if out_layer.bias is not None else None
        logger.info(f"    After init: output_weight_norm={out_weight_norm:.6f}, output_bias_mean={out_bias_val}")

    return len(new_expert_indices), new_expert_indices


def _plot_after_spawn(ctx: TrainingContext, tag: str) -> None:
    """Save expert-region (and soft-weight) plots after a spawn event."""
    model = ctx.model
    if (not ctx.is_adaptive or not hasattr(model, 'num_experts')
            or model.num_experts == 0):
        return
    from adaptive.visualization import (plot_expert_regions,
                                        plot_expert_soft_weights)
    domain_bounds = ctx.domain_bounds
    adaptive_plots_dir = ctx.adaptive_plots_dir
    gt_grid, gt_x, gt_t = ctx.gt_grid, ctx.gt_x, ctx.gt_t
    adaptive_cfg = ctx.adaptive_cfg
    problem_type = '2d' if len(domain_bounds['lower']) == 2 else '3d'
    num_experts_str = f" ({model.num_experts} experts)"
    leaf_info = model.get_leaf_info()
    leaf_expert_indices = [idx for _, idx in leaf_info if idx >= 0]
    regions_to_plot = (
        [model.regions[i] for i in leaf_expert_indices]
        if isinstance(model, AToELeaves) else model.regions
    )
    plot_expert_regions(
        regions=regions_to_plot,
        domain_bounds=domain_bounds,
        output_path=adaptive_plots_dir / f"expert_regions_{tag}.png",
        problem_type=problem_type,
        title=f"Expert Regions ({tag}){num_experts_str}",
        ground_truth=gt_grid, grid_x=gt_x, grid_t=gt_t,
    )
    if adaptive_cfg['blending_mode'] == 'soft' and problem_type == '2d':
        leaf_indices_set = (set(leaf_expert_indices)
                            if isinstance(model, (AToELeaves, ANT)) else None)
        plot_expert_soft_weights(
            model=model, domain_bounds=domain_bounds,
            output_path=adaptive_plots_dir / f"soft_weights_{tag}.png",
            title_prefix=f"{tag}: ", leaf_indices=leaf_indices_set,
        )


def _check_output_continuity(ctx: TrainingContext, label: str = "spawn") -> Dict:
    """Compute model output on sample points for continuity checking.
    
    Call before and after spawning to verify output doesn't change unexpectedly.
    Returns dict with output statistics that can be compared.
    """
    model = ctx.model
    eval_data = ctx.eval_data
    
    if eval_data is None:
        return {}
    
    try:
        with torch.no_grad():
            sample_inputs = torch.cat([eval_data['x'][:200], eval_data['t'][:200]], dim=1)
            output = model(sample_inputs)
            
            stats = {
                'label': label,
                'output_norm': output.norm().item(),
                'output_mean': output.mean().item(),
                'output_std': output.std().item(),
                'output_min': output.min().item(),
                'output_max': output.max().item(),
            }
            return stats
    except Exception as e:
        logger.info(f"  [Continuity] Failed to compute {label}: {e}")
        return {}


def _log_continuity_diff(before: Dict, after: Dict) -> None:
    """Log the difference between before and after spawning outputs."""
    if not before or not after:
        return
    
    norm_diff = abs(after['output_norm'] - before['output_norm'])
    mean_diff = abs(after['output_mean'] - before['output_mean'])
    
    # Compute relative difference
    rel_norm_diff = norm_diff / (before['output_norm'] + 1e-10)
    rel_mean_diff = mean_diff / (abs(before['output_mean']) + 1e-10)
    
    logger.info(f"\n[Continuity Check] Before vs After Spawning:")
    logger.info(f"  Before: norm={before['output_norm']:.6f}, mean={before['output_mean']:.6f}, "
                f"std={before['output_std']:.6f}")
    logger.info(f"  After:  norm={after['output_norm']:.6f}, mean={after['output_mean']:.6f}, "
                f"std={after['output_std']:.6f}")
    logger.info(f"  Diff:   norm_change={norm_diff:.6f} ({rel_norm_diff*100:.2f}%), "
                f"mean_change={mean_diff:.6f} ({rel_mean_diff*100:.2f}%)")
    
    # Warn if output changed significantly
    if rel_norm_diff > 0.01:  # More than 1% change
        logger.info(f"  [WARNING] Output norm changed by {rel_norm_diff*100:.2f}% after spawning!")
    if rel_mean_diff > 0.01:
        logger.info(f"  [WARNING] Output mean changed by {rel_mean_diff*100:.2f}% after spawning!")


def train_orchestrator(ctx: TrainingContext) -> None:
    """Drive training as a sequence of segments.

    Dispatch:
      * non-adaptive → one ``main`` segment.
      * AToE-Leaves  → root (Phase 1, or pretrained) → M-term tree → spawn all
                       leaves → per-leaf split training (``phase3``) → joint
                       PoU ``fine_tune``.
    """
    cfg = ctx.cfg
    model = ctx.model

    # Emergency save spanning all segments (reads live ctx state).
    import atexit as _atexit

    def _emergency_metrics_save():
        if _emergency_metrics_save.done:
            return
        import traceback as _tb_mod
        exc = _tb_mod.format_exc()
        ctx.metrics['exception_events'].append({
            'epoch': ctx.epoch,
            'note': 'process_exit_or_exception',
            'traceback': exc if exc.strip() != 'NoneType: None' else None,
        })
        ctx.metrics['training_time_seconds'] = time.time() - ctx.start_time
        _p = ctx.run_dir / "metrics.json"
        try:
            with open(_p, 'w') as _f:
                json.dump(ctx.metrics, _f, indent=2, cls=_NumpySafeEncoder)
            logger.info(f"\n[Emergency] Metrics saved to {_p}")
        except Exception as _se:
            logger.info(f"\n[Emergency] Could not save metrics: {_se}")

    _emergency_metrics_save.done = False
    _atexit.register(_emergency_metrics_save)
    ctx._emergency_metrics_save = _emergency_metrics_save
    ctx._atexit = _atexit

    # ── Variant detection ──
    variant = 'AToE-Leaves' if isinstance(model, AToELeaves) else 'base'
    logger.info(f"\n[Orchestrator] variant={variant} | adaptive={ctx.is_adaptive}")

    # ── Non-adaptive: single segment over all params ──
    if not ctx.is_adaptive:
        _set_trainable(model, 'all')
        res = _train_segment(ctx, 'main', ctx.epochs, cfg)
        ctx.total_epochs = ctx.epoch
        return

    # ── Root / base training (Phase 1) ──
    if ctx.pretrained_base_checkpoint is not None:
        logger.info("[Orchestrator] Root skipped — base loaded from "
              f"{ctx.pretrained_base_checkpoint}.")
    else:
        _set_trainable(model, 'base')
        root_cfg = dict(cfg)
        root_cfg.update(ctx.initial_train_cfg or {})
        root_budget = ctx.initial_train_cfg['epochs']
        logger.info(f"[Orchestrator] [3-Phase] Phase 1: training root/base for "
              f"{root_budget} epochs")
        res = _train_segment(ctx, 'root', root_budget, root_cfg)
        if res.nan_detected or res.oom_stopped:
            return

    # ── Root rel-L2 baseline for the training-curve reference line ──
    # base_model holds the root (loaded or Phase-1 trained), no experts yet.
    try:
        _root_net = getattr(model, 'base_model', model)
        if ctx.eval_data is not None:
            model.eval()
            with torch.no_grad():
                _ev = ctx.eval_data
                _pred = _root_net(torch.cat([_ev['x'], _ev['t']], dim=1))
                _num = torch.sqrt(((_pred - _ev['h_gt']) ** 2).sum())
                _den = torch.sqrt((_ev['h_gt'] ** 2).sum()) + 1e-10
                ctx.metrics['root_rel_l2'] = (_num / _den).item()
            logger.info(f"[Orchestrator] Root rel-L2 = "
                        f"{ctx.metrics['root_rel_l2']:.6e} (training-curve baseline)")
    except Exception as _e:
        logger.info(f"[Orchestrator] Could not compute root rel-L2: {_e}")

    # ── Tree build (once) + level selection ──
    retain_siblings = True  # full binary tiling gives a complete PoU over the domain
    leaves_only = True
    # Non-additive leaf composition: each leaf owns its subdomain, so it
    # starts from a copy of the root's output layer (PoU continuity).
    copy_output = True
    build_result = _build_tree_once(ctx, retain_siblings)
    levels, nodes_to_spawn = _select_levels(ctx, build_result, leaves_only)
    _record_tree_diagnostics(ctx, build_result, nodes_to_spawn)
    node_tree_depth = build_result['node_tree_depth']

    if not nodes_to_spawn:
        logger.info("[Orchestrator] No nodes accepted — finishing after root.")
        ctx.total_epochs = ctx.epoch
        return

    node_to_expert: Dict = {}

    # ── Spawn all leaves at once, then joint Phase 3 ──
    split_enabled = ctx.adaptive_cfg.get('split_icbc', {}).get('enabled', False)
    _before_spawn = _check_output_continuity(ctx, "before_spawn")

    total = 0
    for level in levels:
        spawned, _ = _spawn_nodes(ctx, level, copy_output,
                                  node_to_expert, node_tree_depth)
        total += spawned
    logger.info(f"[FullTree] Spawning complete. {total} leaves spawned.")

    _after_spawn = _check_output_continuity(ctx, "after_spawn")
    _log_continuity_diff(_before_spawn, _after_spawn)

    _plot_after_spawn(ctx, f"epoch_{ctx.epoch}")
    if total == 0:
        logger.info("[Orchestrator] Zero experts spawned — finishing after root.")
        ctx.total_epochs = ctx.epoch
        return
    logger.info(f"[Phase 3] Training {total} leaf experts (base retired from composition)")
    _set_trainable(model, 'leaves')

    if split_enabled:
        _run_split_segment(ctx, 'phase3', cfg['epochs'], cfg, variant='AToE-Leaves')
    else:
        res = _train_segment(ctx, 'phase3', cfg['epochs'], cfg)

    # ── Final joint fine-tune with the PoU-composed loss ──
    fine_tune_cfg = ctx.adaptive_cfg.get('fine_tune', None)
    if fine_tune_cfg:
        blending = model.blending_mode if hasattr(model, 'blending_mode') else 'soft'
        logger.info("[FineTune] Unfreezing ALL params for final joint fine-tune.")
        logger.info(f"[FineTune] Using composed loss with blending_mode='{blending}' (matches inference)")
        _set_trainable(model, 'all')

        # Ensure split_context is cleared so eval uses configured blending_mode
        ctx._split_context = None

        # L2-SP anchoring: snapshot weights and wrap loss
        l2sp_lambda = fine_tune_cfg.get('l2sp_lambda', 0.0)
        orig_loss_fn = ctx.loss_fn
        if l2sp_lambda > 0:
            ctx._l2sp_anchor = {
                name: p.clone().detach()
                for name, p in model.named_parameters()
                if p.requires_grad
            }
            _anchor = ctx._l2sp_anchor
            _lam = l2sp_lambda

            def _l2sp_loss(model, batch, **kw):
                loss = orig_loss_fn(model, batch, **kw)
                if isinstance(loss, dict) or kw.get('return_components', False):
                    return loss
                penalty = sum(
                    (p - _anchor[n]).pow(2).sum()
                    for n, p in model.named_parameters()
                    if n in _anchor
                )
                return loss + (_lam / 2.0) * penalty

            ctx.loss_fn = _l2sp_loss
            logger.info(f"[L2-SP] Anchoring enabled with lambda={l2sp_lambda}")

        ft_cfg = dict(cfg)
        ft_cfg.update(fine_tune_cfg)
        ft_min = fine_tune_cfg.get('min_epochs', ctx.min_epochs)
        res = _train_segment(ctx, 'fine_tune', fine_tune_cfg['epochs'], ft_cfg,
                       min_epochs_override=ft_min)

        # Restore original loss function
        if l2sp_lambda > 0:
            ctx.loss_fn = orig_loss_fn
            ctx._l2sp_anchor = None

    ctx.total_epochs = ctx.epoch


def _run_split_segment(
    ctx: TrainingContext,
    segment_name: str,
    epoch_budget: int,
    segment_cfg: Dict,
    *,
    variant: str,
    lr_override=None,
    min_epochs_override=None,
) -> SegmentResult:
    """Swap to split-loss data/loss, run _train_segment, then restore originals.

    Returns:
        SegmentResult from the inner _train_segment call.
    """
    model = ctx.model
    cfg = ctx.cfg

    # Fix 5: Snapshot model BEFORE training for stable interface targets
    # This frozen snapshot is used to mint targets for interface points
    model_snapshot = copy.deepcopy(model)
    model_snapshot.eval()
    for p in model_snapshot.parameters():
        p.requires_grad = False
    logger.info(f"[SplitLoss] Created frozen model snapshot for interface targets")

    # Identify the leaf experts being trained in this segment
    leaf_info = model.get_leaf_info()
    new_expert_indices = [idx for _, idx in leaf_info if idx >= 0]

    regions_list = model.regions

    logger.info(f"[SplitLoss] Building subdomain data for {len(new_expert_indices)} "
                f"new expert(s): {new_expert_indices}")

    # The leaves tile the domain and share the base (root) as their common
    # parent, so mint interface targets from the frozen base — good root
    # predictions regardless of expert architecture.
    interface_model = model_snapshot.base_model
    logger.info("[SplitLoss] Interface targets minted from frozen base (root).")

    # Use snapshot for interface target minting
    split_data = build_subdomain_data(
        model_snapshot, new_expert_indices, regions_list, cfg,
        ctx.device, seed=ctx.epoch,
        interface_model=interface_model,
    )

    _log_subdomain_summary(new_expert_indices, regions_list, split_data)

    # Freeze/trainable confirmation
    trainable = [n for n, p in model.named_parameters()
                 if p.requires_grad]
    frozen = [n for n, p in model.named_parameters()
              if not p.requires_grad]
    logger.info(
        f"[SplitLoss] NO-PoU mode: each expert trained "
        f"on its local output only"
    )
    logger.info(
        f"[SplitLoss] trainable params: {len(trainable)}, "
        f"frozen params: {len(frozen)}"
    )

    # Stash original context state
    orig_loss_fn = ctx.loss_fn
    orig_train_data = ctx.train_data
    orig_train_loader = ctx.train_loader

    # Build split loss with original loss as fallback for eval batches
    split_loss = build_split_loss(
        model, cfg, variant=variant, orig_loss_fn=orig_loss_fn,
    )

    # Swap to split data/loss
    ctx.loss_fn = split_loss
    ctx.train_data = split_data
    ctx.train_loader = _create_split_dataloader(
        split_data, segment_cfg.get('batch_size', cfg['batch_size']), shuffle=True,
    )
    ctx._split_context = {
        'model': model,
        'model_snapshot': model_snapshot,  # frozen snapshot reused on resample
        'new_expert_indices': new_expert_indices,
        'regions': regions_list,
        'variant': variant,
        'interface_model': interface_model,  # frozen base for interface targets
    }

    res = _train_segment(ctx, segment_name, epoch_budget, segment_cfg,
                         lr_override=lr_override,
                         min_epochs_override=min_epochs_override)

    # Save per-expert loss history into metrics
    peh = getattr(split_loss, '_per_expert_history', {})
    if peh:
        if 'split_expert_losses' not in ctx.metrics:
            ctx.metrics['split_expert_losses'] = {}
        ctx.metrics['split_expert_losses'][segment_name] = peh

    # Per-expert training curves + region panel
    def _to_numpy(x):
        """Convert to numpy, handling both Tensors and arrays."""
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x  # already numpy
    
    try:
        training_plots_dir = ctx.run_dir / 'training_plots'
        training_plots_dir.mkdir(exist_ok=True)
        plot_path = training_plots_dir / f'expert_curves_after_{segment_name}.png'
        plot_per_expert_curves(
            peh,
            list(regions_list),
            plot_path,
            domain_bounds=ctx.domain_bounds,
            gt_grid=_to_numpy(ctx.gt_grid),
            grid_x=_to_numpy(ctx.gt_x),
            grid_t=_to_numpy(ctx.gt_t),
            segment_name=segment_name,
            split_data=split_data,
        )
        logger.info(
            f"[SplitPlot] Saved training_plots/{plot_path.name}"
        )
    except Exception as e:
        logger.warning(
            f"[SplitPlot] Failed: {e}"
        )

    # Restore original context
    ctx.loss_fn = orig_loss_fn
    ctx.train_data = orig_train_data
    ctx.train_loader = orig_train_loader
    ctx._split_context = None

    return res


def _log_subdomain_summary(new_expert_indices, regions, split_data):
    """Log per-expert point summaries for the subdomain dataset."""
    expert_ids = split_data['expert_id']
    kinds = split_data['kind']
    cont_neighbors = split_data.get('cont_neighbor', None)

    for eidx in new_expert_indices:
        emask = (expert_ids == eidx)
        n_total = emask.sum().item()
        region = regions[eidx]
        counts = {}
        for k_val, k_name in KIND_NAMES.items():
            counts[k_name] = ((kinds[emask] == k_val).sum().item() if n_total > 0 else 0)
        logger.info(
            f"[SplitData] expert={eidx} depth={region.depth} parent={region.parent_idx} "
            f"bounds=[{region.bounds_lower}..{region.bounds_upper}] "
            f"total={n_total} {counts}"
        )
        if counts.get('residual', 0) == 0:
            logger.warning(f"[SplitData] expert={eidx} has 0 residual points!")
        if counts.get('ic_true', 0) + counts.get('interface_ic', 0) == 0:
            logger.warning(f"[SplitData] expert={eidx} has 0 IC/interface points!")
    
    # Log continuity pair summary
    if cont_neighbors is not None:
        from adaptive.subdomain_data import KIND_CONTINUITY
        cont_mask = (kinds == KIND_CONTINUITY)
        n_cont_total = cont_mask.sum().item()
        if n_cont_total > 0:
            # Count unique pairs
            unique_pairs = set()
            for i in range(len(cont_neighbors)):
                if kinds[i] == KIND_CONTINUITY:
                    a = expert_ids[i].item()
                    b = cont_neighbors[i].item()
                    unique_pairs.add((min(a, b), max(a, b)))
            logger.info(
                f"[SplitData] Continuity: {n_cont_total} points across "
                f"{len(unique_pairs)} neighbor pairs"
            )


def _finalize_training(ctx: TrainingContext) -> Path:
    """Phase 3 of :func:`train`: final checkpoints, plots, metrics, summary.

    Behavior-preserving extraction of the original post-loop block. Returns the
    best checkpoint path, or ``None`` on NaN divergence (matching the original).
    """
    # ── Unpack ctx into locals (finalize body below is unchanged) ──
    model = ctx.model
    cfg = ctx.cfg
    device = ctx.device
    run_dir = ctx.run_dir
    eval_data = ctx.eval_data
    metrics = ctx.metrics
    timer = ctx.timer
    checkpoint_dir = ctx.checkpoint_dir
    optimizer = ctx.optimizer
    current_optimizer_name = ctx.current_optimizer_name
    epochs = ctx.epochs
    total_epochs = ctx.total_epochs
    train_loss = ctx.train_loss
    eval_loss = ctx.eval_loss
    eval_rel_l2 = ctx.eval_rel_l2
    eval_inf_norm = ctx.eval_inf_norm
    best_eval_loss = ctx.best_eval_loss
    best_checkpoint_path = ctx.best_checkpoint_path
    switch_epoch = ctx.switch_epoch
    optimizer_2_name = ctx.optimizer_2_name
    _nan_detected = ctx._nan_detected
    is_adaptive = ctx.is_adaptive
    adaptive_cfg = ctx.adaptive_cfg
    adaptive_plots_dir = ctx.adaptive_plots_dir
    domain_bounds = ctx.domain_bounds
    gt_grid = ctx.gt_grid
    gt_x = ctx.gt_x
    gt_t = ctx.gt_t
    rejected_regions = ctx.rejected_regions
    leaf_loss_history = ctx.leaf_loss_history
    max_experts = ctx.max_experts
    start_time = ctx.start_time

    # Finalize-only imports (originally imported in setup under is_adaptive)
    if is_adaptive:
        from adaptive.visualization import (
            plot_expert_regions, save_regions_metadata, plot_expert_soft_weights
        )

    # Disable emergency save (loop done or NaN exit)
    _emergency_metrics_save = ctx._emergency_metrics_save
    _atexit = ctx._atexit
    _emergency_metrics_save.done = True
    _atexit.unregister(_emergency_metrics_save)

    if _nan_detected:
        logger.info("[NaN] Generating partial training curves before exit...")
        try:
            training_plots_dir = run_dir / "training_plots"
            switch_epoch_to_plot = switch_epoch if (optimizer_2_name is not None and switch_epoch <= epochs) else None
            plot_training_curves(metrics, training_plots_dir, optimizer_switch_epoch=switch_epoch_to_plot)
        except Exception as _plot_err:
            logger.info(f"  [NaN] Could not generate training curves: {_plot_err}")
        logger.info("[NaN] Skipping remaining post-training cleanup — moving to next experiment.")
        return

    # Save final model
    final_checkpoint_path = checkpoint_dir / "final_model.pt"
    _save_checkpoint(final_checkpoint_path, model, optimizer, current_optimizer_name, total_epochs,
                    train_loss, eval_loss, cfg, metrics)

    logger.info(f"\nTraining completed in {time.time() - start_time:.1f}s")
    logger.info(f"  Best eval loss: {best_eval_loss:.6f}")
    logger.info(f"  Best checkpoint: {best_checkpoint_path}")
    logger.info(f"  Final checkpoint: {final_checkpoint_path}")
    
    # Save timing data and print summary
    timer.save(run_dir / "timing.json")
    timer.print_summary()

    # Save expert diagnostics to CSV
    if is_adaptive and hasattr(model, 'num_experts') and model.num_experts > 0 and hasattr(model, '_diag_data') and model._diag_data:
        import pandas as pd
        diag_csv_path = run_dir / "expert_diagnostics.csv"

        # Flatten diagnostic data for CSV
        diag_rows = []
        for diag in model._diag_data:
            row = {
                'epoch': diag['epoch'],
                'base_norm': diag['base_norm'],
                'total_expert_contrib': diag['total_expert_contrib'],
                'ratio_expert_to_base': diag['total_expert_contrib'] / diag['base_norm'] if diag['base_norm'] > 0 else 0
            }
            # Add individual expert norms
            for i, norm in enumerate(diag['expert_norms']):
                row[f'expert_{i}_norm'] = norm
            # Add individual expert gradient norms
            for i, grad_norm in enumerate(diag['expert_grad_norms']):
                row[f'expert_{i}_grad_norm'] = grad_norm
            diag_rows.append(row)

        df = pd.DataFrame(diag_rows)
        df.to_csv(diag_csv_path, index=False)
        logger.info(f"  Expert diagnostics saved: {diag_csv_path}")

    # Plot training curves
    logger.info(f"\nGenerating training plots...")
    training_plots_dir = run_dir / "training_plots"
    # Extract all optimizer switch epochs and segment start epochs from metrics
    optimizer_switch_epochs = [e['epoch'] for e in metrics.get('optimizer_events', [])]
    segment_start_epochs = [s['start_epoch'] for s in metrics.get('segment_events', [])]
    plot_training_curves(metrics, training_plots_dir,
                         optimizer_switch_epochs=optimizer_switch_epochs,
                         segment_start_epochs=segment_start_epochs)

    # Plot final predictions
    model.eval()
    with torch.no_grad():
        inputs_eval = torch.cat([eval_data['x'], eval_data['t']], dim=1)
        h_pred_eval = model(inputs_eval)

    plot_final_comparison(
        h_pred_eval.cpu().numpy(),
        eval_data['h_gt'].cpu().numpy(),
        eval_data['x'].detach().cpu().numpy(),
        eval_data['t'].detach().cpu().numpy(),
        training_plots_dir
    )

    # Final adaptive PINN outputs
    if is_adaptive and hasattr(model, 'num_experts') and model.num_experts > 0:
        logger.info("\n" + "=" * 60)
        logger.info("Adaptive PINN Final Summary")
        logger.info("=" * 60)
        logger.info(f"  Total experts spawned: {model.num_experts}")
        
        problem_type = '2d' if len(domain_bounds['lower']) == 2 else '3d'
        is_leaves_model = isinstance(model, (AToELeaves, ANT))
        leaf_info = model.get_leaf_info()
        leaf_expert_indices = [idx for _, idx in leaf_info if idx >= 0]
        regions_to_plot = (
            [model.regions[i] for i in leaf_expert_indices]
            if is_leaves_model else model.regions
        )
        label = 'leaves' if is_leaves_model else 'experts'
        plot_expert_regions(
            regions=regions_to_plot,
            domain_bounds=domain_bounds,
            output_path=adaptive_plots_dir / "expert_regions_final.png",
            problem_type=problem_type,
            title=f"Final Expert Regions ({len(regions_to_plot)} {label})",
            ground_truth=gt_grid,
            grid_x=gt_x,
            grid_t=gt_t
        )

        if adaptive_cfg['blending_mode'] == 'soft' and problem_type == '2d':
            leaf_indices_set = set(leaf_expert_indices) if is_leaves_model else None
            plot_expert_soft_weights(
                model=model,
                domain_bounds=domain_bounds,
                output_path=adaptive_plots_dir / "soft_weights_final.png",
                title_prefix="Final: ",
                leaf_indices=leaf_indices_set
            )
        
        save_regions_metadata(
            regions=model.regions,
            output_path=adaptive_plots_dir / "expert_regions.json",
            rejected_regions=rejected_regions,
            leaf_loss_history=leaf_loss_history,
            spawning_method='M_term_tree_by_norm',
            spawning_diagnostics=metrics.get('spawning_diagnostics', []),
        )
        
        base_params = sum(p.numel() for p in model.base_model.parameters())
        expert_full_params = []
        for i, expert in enumerate(model.experts):
            expert_full_params.append(
                sum(p.numel() for p in expert.parameters()))
        expert_archs = [
            e.layers if hasattr(e, 'layers') else []
            for e in model.experts]

        leaf_info = model.get_leaf_info()
        leaf_expert_indices = set(
            idx for _, idx in leaf_info if idx >= 0)

        is_leaves_only = isinstance(model, AToELeaves)
        expert_params = expert_full_params

        leaf_params = sum(
            expert_params[i] for i in leaf_expert_indices
            if i < len(expert_params))

        metrics['adaptive_pinn'] = {
            'num_experts': model.num_experts,
            'max_experts': max_experts,
            'spawning_method': 'M_term_tree_by_norm',
            'regions': [r.to_dict() for r in model.regions],
            'base_params': base_params,
            'expert_params': expert_params,
            'expert_architectures': expert_archs,
            'total_params': base_params + sum(expert_params),
            'leaf_expert_indices': sorted(leaf_expert_indices),
            'leaf_params': leaf_params,
            'forward_params': base_params + (
                leaf_params if is_leaves_only
                else sum(expert_params)),
        }

    total_model_params = sum(p.numel() for p in model.parameters())
    metrics['total_params'] = total_model_params
    metrics['training_time_seconds'] = time.time() - start_time

    # Save metrics to JSON
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, cls=_NumpySafeEncoder)
    logger.info(f"  Metrics saved to {metrics_path}")

    # Save summary
    summary_path = run_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Training Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Problem: {cfg['problem']}\n")
        f.write(f"Base architecture: {cfg['base_architecture']}\n")
        exp_arch = cfg.get('experts_architecture', cfg['base_architecture'])
        f.write(f"Experts architecture: {exp_arch}\n")
        f.write(f"Activation: {cfg['activation']}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {cfg['batch_size']}\n")
        f.write(f"Learning rate: {cfg['lr']}\n")
        f.write(f"Device: {device}\n\n")
        f.write(f"Final train loss: {train_loss:.6f}\n")
        f.write(f"Final eval loss: {eval_loss:.6f}\n" if eval_loss is not None else "Final eval loss: N/A\n")
        f.write(f"Final eval rel-L2: {eval_rel_l2:.6f}\n" if eval_rel_l2 is not None else "Final eval rel-L2: N/A\n")
        f.write(f"Final eval inf-norm: {eval_inf_norm:.6f}\n" if eval_inf_norm is not None else "Final eval inf-norm: N/A\n")
        f.write(f"Best eval loss: {best_eval_loss:.6f}\n\n")
        f.write(f"Best checkpoint: {best_checkpoint_path}\n")
        f.write(f"Final checkpoint: {final_checkpoint_path}\n")
    logger.info(f"  Summary saved to {summary_path}")

    # Save config used
    from utils.io import get_git_info
    cfg['git'] = get_git_info()
    config_path = run_dir / "config_used.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    logger.info(f"  Config saved to {config_path}")
    
    # Problem-specific final evaluation visualization
    logger.info("\nGenerating problem-specific evaluation visualizations...")
    try:
        from utils.problem_specific import get_visualization_module
        viz_module = get_visualization_module(cfg['problem'])
        visualize_evaluation = viz_module[1]  # Second element is visualize_evaluation
        visualize_evaluation(model, eval_data_path, run_dir, cfg)
    except ValueError as e:
        logger.info(f"  (No custom evaluation visualization for {cfg['problem']})")
        logger.info(f"  ValueError details: {e}")
    except Exception as e:
        logger.info(f"  Warning: Could not generate evaluation visualization: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    return best_checkpoint_path


def _move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    """Move a batch dictionary to specified device."""
    result = {
        'x': batch['x'].to(device),
        't': batch['t'].to(device),
        'h_gt': batch['h_gt'].to(device),
        'mask': {
            'residual': batch['mask']['residual'].to(device),
            'IC': batch['mask']['IC'].to(device),
            'BC': batch['mask']['BC'].to(device)
        }
    }
    return result


def _cast_data_to_dtype(batch: Dict, dtype: torch.dtype) -> Dict:
    """Cast floating-point tensors in a batch dictionary to specified dtype."""
    result = {
        'x': batch['x'].to(dtype) if batch['x'].is_floating_point() else batch['x'],
        't': batch['t'].to(dtype) if batch['t'].is_floating_point() else batch['t'],
        'h_gt': batch['h_gt'].to(dtype) if batch['h_gt'].is_floating_point() else batch['h_gt'],
        'mask': batch['mask']  # masks are boolean, don't cast
    }
    return result


def _create_split_dataloader(
    data: Dict,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create DataLoader for split-loss subdomain data (expert_id + kind + bc_face_id + continuity schema)."""
    dataset = TensorDataset(
        data['x'], data['t'], data['h_gt'],
        data['expert_id'], data['kind'], data['bc_face_id'],
        data['cont_neighbor'], data['cont_dim'],
    )

    def collate_fn(batch_list):
        return {
            'x': torch.stack([b[0] for b in batch_list]),
            't': torch.stack([b[1] for b in batch_list]),
            'h_gt': torch.stack([b[2] for b in batch_list]),
            'expert_id': torch.stack([b[3] for b in batch_list]),
            'kind': torch.stack([b[4] for b in batch_list]),
            'bc_face_id': torch.stack([b[5] for b in batch_list]),
            'cont_neighbor': torch.stack([b[6] for b in batch_list]),
            'cont_dim': torch.stack([b[7] for b in batch_list]),
        }

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, pin_memory=False, num_workers=0,
    )


def _create_dataloader(
    data: Dict,
    batch_size: int,
    shuffle: bool
) -> DataLoader:
    """
    Create DataLoader from data dictionary.

    Args:
        data: Dictionary with 'x', 't', 'h_gt', 'mask'
        batch_size: Batch size
        shuffle: Whether to shuffle

    Returns:
        DataLoader
    """
    dataset = TensorDataset(
        data['x'],
        data['t'],
        data['h_gt'],
        data['mask']['residual'],
        data['mask']['IC'],
        data['mask']['BC']
    )

    # Custom collate function to reconstruct dict format
    def collate_fn(batch_list):
        x_batch = torch.stack(tuple(item[0] for item in batch_list))
        t_batch = torch.stack(tuple(item[1] for item in batch_list))
        h_gt_batch = torch.stack(tuple(item[2] for item in batch_list))
        mask_res_batch = torch.stack(tuple(item[3] for item in batch_list))
        mask_ic_batch = torch.stack(tuple(item[4] for item in batch_list))
        mask_bc_batch = torch.stack(tuple(item[5] for item in batch_list))

        result = {
            'x': x_batch,
            't': t_batch,
            'h_gt': h_gt_batch,
            'mask': {
                'residual': mask_res_batch,
                'IC': mask_ic_batch,
                'BC': mask_bc_batch
            }
        }
        
        return result

    # No explicit generator - uses global random state from torch.manual_seed()
    # This avoids device mismatch issues while maintaining reproducibility
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=False,  # Data already on device
        num_workers=0,  # Keep data on GPU
    )


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_name: str,
    epoch: int,
    train_loss: float,
    eval_loss: float,
    cfg: Dict,
    metrics: Dict
) -> None:
    """Save model checkpoint with full information."""
    # In time-marching mode, cfg['_time_marching_window'] carries a transient
    # 'prev_model' reference that must not be serialized into the checkpoint
    # (time_marching.py manages it in memory).
    cfg_to_save = cfg
    if ('_time_marching_window' in cfg
            and cfg['_time_marching_window'].get('prev_model') is not None):
        cfg_to_save = dict(cfg)
        tm = dict(cfg['_time_marching_window'])
        tm['prev_model'] = None
        cfg_to_save['_time_marching_window'] = tm

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer_name,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'eval_loss': eval_loss,
        'config': cfg_to_save,
        'metrics': metrics
    }
    
    # For adaptive models, also save extended state
    if hasattr(model, 'state_dict_extended'):
        checkpoint['adaptive_state'] = model.state_dict_extended()
        checkpoint['is_adaptive'] = True

    torch.save(checkpoint, path)


def _infer_base_arch_from_state_dict(sd: Dict) -> list:
    """Best-effort base architecture from a plain FCNet state dict.

    Counts ``network.layer_{i}.weight`` tensors. Only used as a fallback when the
    checkpoint does not store its nominal ``base_architecture`` (e.g. an old vanilla
    base checkpoint). Note: with Fourier features the inferred input dim reflects the
    expanded input, so a saved nominal arch is always preferred when available.
    """
    arch = []
    i = 1
    while f'network.layer_{i}.weight' in sd:
        w = sd[f'network.layer_{i}.weight']
        if i == 1:
            arch.append(int(w.shape[1]))
        arch.append(int(w.shape[0]))
        i += 1
    return arch


def _load_pretrained_base(model: nn.Module, ckpt_path: str, cfg: Dict) -> None:
    """Load the BASE network from a checkpoint into ``model.base_model``.

    Supplies Phase 1 without training (the ``pretrained_base_checkpoint`` flow).
    Accepts either an adaptive/MoE checkpoint (takes only ``adaptive_state['base_model']``,
    ignoring its experts) or a plain base checkpoint (uses ``model_state_dict``).

    If the checkpoint's base architecture differs from the run's, the base is rebuilt
    to the checkpoint's architecture so weights load correctly. Expert architecture
    (``experts_architecture`` / config) is not changed.
    """
    from pathlib import Path as _Path
    p = _Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(
            f"pretrained_base_checkpoint not found: {ckpt_path}")
    # Trusted local checkpoint: weights_only=False (PyTorch 2.6+ defaults to True,
    # which rejects the numpy scalars stored in the saved config/metrics).
    try:
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
    except TypeError:  # older torch without the weights_only kwarg
        ckpt = torch.load(p, map_location='cpu')

    adaptive_state = ckpt.get('adaptive_state') if isinstance(ckpt, dict) else None
    if adaptive_state and 'base_model' in adaptive_state:
        base_sd = adaptive_state['base_model']
        saved_arch = adaptive_state.get('base_architecture')
        saved_activation = adaptive_state.get('activation')
        saved_expert_type = (adaptive_state.get('adaptive_config') or {}).get('expert_type')
        logger.info(f"  [PretrainedBase] Source is an adaptive/MoE checkpoint; "
              f"loading its base only (ignoring {adaptive_state.get('num_experts', '?')} experts).")
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        base_sd = ckpt['model_state_dict']
        _cfg_in = ckpt.get('config') or {}
        saved_arch = _cfg_in.get('base_architecture')
        saved_activation = _cfg_in.get('activation')
        saved_expert_type = (_cfg_in.get('adaptive_pinn') or {}).get('expert_type')
    else:
        raise ValueError(
            f"Could not find base weights in checkpoint {ckpt_path} "
            f"(expected 'adaptive_state.base_model' or 'model_state_dict').")

    if not saved_arch:
        saved_arch = _infer_base_arch_from_state_dict(base_sd)
        logger.info(f"  [PretrainedBase] Checkpoint has no stored base_architecture; "
              f"inferred {saved_arch} from weights.")

    # Adopt the checkpoint's base architecture if it differs from the run's.
    if list(saved_arch) != list(model.base_architecture):
        from models.network_factory import create_network
        _old = next(model.base_model.parameters())
        device, dtype = _old.device, _old.dtype
        activation = saved_activation or getattr(model, 'activation', cfg.get('activation'))
        expert_type = saved_expert_type or cfg['adaptive_pinn'].get('expert_type', 'mlp')
        logger.info(f"  [PretrainedBase] Adopting checkpoint base architecture: "
              f"{model.base_architecture} -> {list(saved_arch)}")
        model.base_model = create_network(
            list(saved_arch), activation, cfg, is_base=True, expert_type=expert_type
        ).to(device=device, dtype=dtype)
        model.base_architecture = list(saved_arch)

    model.base_model.load_state_dict(base_sd)
    n_params = sum(q.numel() for q in model.base_model.parameters())
    logger.info(f"  [PretrainedBase] Loaded base weights from {ckpt_path} ({n_params} params)")
    if hasattr(model, 'experts_architecture'):
        logger.info(f"  [PretrainedBase] Base architecture: {list(model.base_architecture)}; "
              f"experts architecture unchanged: {list(model.experts_architecture)}")
    # Re-sync AToE's batched container so the forward pass sees the loaded base.
    if hasattr(model, 'batched_models'):
        model.batched_models.sync_from_models(model.base_model, model.experts)
