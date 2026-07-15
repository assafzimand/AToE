"""Training setup: config/data preparation, optimizer and dataloader factories, checkpoint IO."""

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
    plot_training_curves,
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


def _set_default_torch_device(device: torch.device, full_batch: bool) -> None:
    """Re-establish torch's global default device at a training boundary.

    Works around a PyTorch device-context issue observed in time-marching
    runs: full-batch optimizers (LBFGS/SSBroyden) allocate their internal
    state via the default device, so it must point at the training device,
    while DataLoader sampling (Adam/SOAP) requires the default back on CPU so
    the sampler's generator matches torch.randperm. A previous window or
    segment may have left either state behind, so every boundary (segment
    start, optimizer switch, loader rebuild) calls this explicitly.
    """
    torch.set_default_device(device if full_batch else None)


def _opt_cfg(cfg: Dict, opt_name: str, key: str, legacy_key: str, default=None):
    """Read an optimizer hyperparameter from the per-optimizer sub-dict.

    Preferred layout is ``cfg[opt_name][key]`` (e.g. ``adam: {betas: ...}``);
    the flat legacy key (e.g. ``adam_betas``) is accepted as a fallback.
    """
    sub = cfg.get(opt_name)
    if isinstance(sub, dict) and key in sub:
        return sub[key]
    return cfg.get(legacy_key, default)


def _create_adam_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    """Create Adam optimizer with config parameters.

    Only includes trainable parameters (requires_grad=True) to avoid
    wasting memory/compute on frozen parameters (e.g., pretrained base model).
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(
        trainable_params,
        lr=cfg['lr'],
        betas=tuple(_opt_cfg(cfg, 'adam', 'betas', 'adam_betas', (0.9, 0.999))),
        eps=_opt_cfg(cfg, 'adam', 'eps', 'adam_eps', 1e-8),
    )


def _create_lbfgs_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    """Create LBFGS optimizer. Should be used with full-batch training.

    Only includes trainable parameters (requires_grad=True) to avoid
    wasting memory/compute on frozen parameters (e.g., pretrained base model).
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.LBFGS(
        trainable_params,
        lr=_opt_cfg(cfg, 'lbfgs', 'lr', 'lbfgs_lr', 1.0),
        max_iter=_opt_cfg(cfg, 'lbfgs', 'max_iter', 'lbfgs_max_iter', 1),
        max_eval=None,  # Default: max_iter * 1.25
        history_size=_opt_cfg(cfg, 'lbfgs', 'history_size', 'lbfgs_history_size', 100),
        line_search_fn=_opt_cfg(cfg, 'lbfgs', 'line_search', 'lbfgs_line_search', 'strong_wolfe'),
        tolerance_grad=_opt_cfg(cfg, 'lbfgs', 'tolerance_grad', 'lbfgs_tolerance_grad', 0.0),
        tolerance_change=_opt_cfg(cfg, 'lbfgs', 'tolerance_change', 'lbfgs_tolerance_change', 0.0),
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
        betas=tuple(_opt_cfg(cfg, 'soap', 'betas', 'soap_betas', (0.95, 0.95))),
        eps=_opt_cfg(cfg, 'adam', 'eps', 'adam_eps', 1e-8),
        precondition_frequency=_opt_cfg(cfg, 'soap', 'precondition_frequency',
                                        'soap_precondition_frequency', 10),
        weight_decay=_opt_cfg(cfg, 'soap', 'weight_decay', 'soap_weight_decay', 0.0),
    )


def _create_ssbroyden_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    """Create SSBroyden (Self-Scaled Broyden) quasi-Newton optimizer via scimba.

    Falls back to LBFGS if scimba is not installed — loudly, since the two
    optimizers reach very different accuracies on the benchmark problems.
    Only includes trainable parameters.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    try:
        from scimba_torch.optimizers.ssbroyden import SSBroyden
        return SSBroyden(
            trainable_params,
            lr=_opt_cfg(cfg, 'ssbroyden', 'lr', 'ssbroyden_lr', 1.0),
            tolerance_grad=_opt_cfg(cfg, 'ssbroyden', 'tolerance_grad',
                                    'ssbroyden_tolerance_grad', 1e-10),
            method='ssbroyden',
        )
    except ImportError:
        logger.warning("!" * 70)
        logger.warning("[Optimizer] scimba NOT installed — SSBroyden unavailable.")
        logger.warning("[Optimizer] FALLING BACK TO LBFGS: results are NOT comparable")
        logger.warning("[Optimizer] to SSBroyden runs. Install with: pip install scimba")
        logger.warning("!" * 70)
        return _create_lbfgs_optimizer(model, cfg)


def _create_optimizer_by_name(name: str, model: nn.Module, cfg: Dict) -> Tuple[torch.optim.Optimizer, str]:
    """Create an optimizer by name string. Returns (optimizer, display_name).

    The display name reflects what was ACTUALLY created (an SSBroyden request
    that fell back to LBFGS is reported as LBFGS everywhere downstream).
    """
    name = name.lower()
    if name == 'soap':
        return _create_soap_optimizer(model, cfg), 'SOAP'
    elif name == 'lbfgs':
        return _create_lbfgs_optimizer(model, cfg), 'LBFGS'
    elif name == 'ssbroyden':
        opt = _create_ssbroyden_optimizer(model, cfg)
        actual = 'SSBroyden' if opt.__class__.__name__ != 'LBFGS' else 'LBFGS'
        return opt, actual
    else:
        return _create_adam_optimizer(model, cfg), 'Adam'


def _debug_print_model_state(model: nn.Module, segment_name: str,
                             sample_data: Dict = None) -> None:
    """Print comprehensive model state at segment start for debugging.

    ``sample_data`` is any dict with 'x'/'t' tensors (e.g. the training set);
    a few of its points probe the composition's output magnitudes.
    """
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
    if hasattr(model, 'debug_composition') and len(experts) > 0 and sample_data is not None:
        try:
            sample_inputs = torch.cat([sample_data['x'][:100], sample_data['t'][:100]], dim=1)
            model.debug_composition(sample_inputs)
        except Exception as e:
            logger.info(f"  [DEBUG] debug_composition failed: {e}")

    # ── Per-expert output magnitude (shows contribution magnitudes) ──
    if sample_data is not None and hasattr(model, 'forward_decomposed'):
        try:
            with torch.no_grad():
                sample_inputs = torch.cat([sample_data['x'][:200], sample_data['t'][:200]], dim=1)
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
    rel_l2: float,
    cfg: Dict,
    metrics: Dict
) -> None:
    """Save model checkpoint with full information.

    ``rel_l2`` is the solver-grid rel-L2 at save time (may be None before the
    first evaluation).
    """
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

    # Optimizer STATE is intentionally not stored: nothing reloads it
    # (reconciliation and resume restore weights only; optimizers are
    # rebuilt per segment), and SSBroyden's dense Hessian approximation
    # alone is n_params^2 floats (multi-GB per save).
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer_name,
        'train_loss': train_loss,
        'rel_l2': rel_l2,
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
    # Converted root checkpoints are plain tensor dicts (safe load); legacy
    # checkpoints with pickled config/metrics need weights_only=False.
    try:
        ckpt = torch.load(p, map_location='cpu', weights_only=True)
    except Exception:
        logger.info("  [PretrainedBase] Not a plain state-dict checkpoint; "
                    "loading legacy format (weights_only=False). Consider "
                    "converting with scripts/convert_root_checkpoint.py.")
        ckpt = torch.load(p, map_location='cpu', weights_only=False)

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

    # The configured base architecture must match the checkpoint exactly —
    # silently adopting a different architecture would make the run's config
    # lie about what actually trained.
    if list(saved_arch) != list(model.base_architecture):
        msg = (
            f"[PretrainedBase] ARCHITECTURE MISMATCH — checkpoint {ckpt_path} "
            f"holds base architecture {list(saved_arch)} but the config requests "
            f"{list(model.base_architecture)}. Set base_architecture to match "
            f"the checkpoint (or point pretrained_base_checkpoint at the right "
            f"file / set it to null to train a fresh root)."
        )
        logger.error("!" * 70)
        logger.error(msg)
        logger.error("!" * 70)
        raise ValueError(msg)

    model.base_model.load_state_dict(base_sd)
    n_params = sum(q.numel() for q in model.base_model.parameters())
    logger.info(f"  [PretrainedBase] Loaded base weights from {ckpt_path} ({n_params} params)")
    if hasattr(model, 'experts_architecture'):
        logger.info(f"  [PretrainedBase] Base architecture: {list(model.base_architecture)}; "
              f"experts architecture unchanged: {list(model.experts_architecture)}")
    # Re-sync AToE's batched container so the forward pass sees the loaded base.
    if hasattr(model, 'batched_models'):
        model.batched_models.sync_from_models(model.base_model, model.experts)


def _load_pretrained_experts(model: nn.Module, ckpt_path: str, cfg: Dict) -> None:
    """Load a FULL AToE-Leaves state (base + leaf experts + regions) from a
    checkpoint — the ``pretrained_local_expert_checkpoint`` flow.

    The checkpoint must carry ``adaptive_state`` (any checkpoint saved by
    ``_save_checkpoint`` for an adaptive model, e.g. ``best_model_phase3.pt``).
    Root and Phase 3 are then skipped by the orchestrator and training goes
    straight to fine-tune.
    """
    from pathlib import Path as _Path
    p = _Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(
            f"pretrained_local_expert_checkpoint not found: {ckpt_path}")
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    adaptive_state = ckpt.get('adaptive_state') if isinstance(ckpt, dict) else None
    if not adaptive_state or not adaptive_state.get('experts'):
        raise ValueError(
            f"pretrained_local_expert_checkpoint {ckpt_path} has no "
            f"'adaptive_state' with experts — need a full AToE checkpoint "
            f"(e.g. best_model_phase3.pt), not a plain base checkpoint.")
    model.load_state_dict_extended(adaptive_state)
    n_params = sum(q.numel() for q in model.parameters())
    logger.info(f"  [PretrainedExperts] Loaded base + {len(model.experts)} "
                f"expert(s) from {ckpt_path} ({n_params} params, "
                f"leaves={sorted(model.leaf_indices)})")
    if ckpt.get('rel_l2') is not None:
        logger.info(f"  [PretrainedExperts] Checkpoint's recorded rel-L2: "
                    f"{ckpt['rel_l2']:.6e} (epoch {ckpt.get('epoch')})")


def _setup_training(
    model: nn.Module,
    loss_fn: Callable,
    train_data_path: str,
    cfg: Dict,
    run_dir: Path
) -> TrainingContext:
    """Phase 1 of :func:`train`: build datasets, optimizer, metrics, adaptive state.

    There is no eval dataset: all rel-L2 / inf-norm metrics are computed on
    the ground-truth solver's native grid, which also supplies the GT heatmap
    background for plots.

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
        # Force true float32 matmuls. On Ampere+ GPUs TF32 (~10-bit mantissa)
        # can otherwise silently distort forward passes and logged metrics:
        # the KdV root run on an A10G reported rel-L2 0.1205 while the same
        # checkpoint evaluated in real float32 scores 0.157 — metrics must be
        # reproducible when checkpoints are re-evaluated on other devices.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        logger.info("TF32 disabled (float32 matmul precision = highest)")

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

    # Load dataset (training points only; metrics come from the solver grid)
    logger.info(f"\nLoading datasets...")
    train_data = torch.load(train_data_path)

    # Move data to device
    train_data = _move_batch_to_device(train_data, device)

    # Cast data to configured precision (float32 or float64)
    precision = cfg.get('precision', 'float32')
    target_dtype = torch.float64 if precision == 'float64' else torch.float32
    train_data = _cast_data_to_dtype(train_data, target_dtype)

    # Filter train data by window temporal bounds if time marching is enabled
    time_marching_window = cfg.get('_time_marching_window', {})
    if time_marching_window.get('enabled', False):
        t_start = time_marching_window['t_start']
        t_end = time_marching_window['t_end']
        window_idx = time_marching_window['idx']

        # IMPORTANT: For windows 1+, override IC BEFORE filtering
        # This updates IC t values from t=0 to t=window.t_start, so they survive filtering
        train_data = _override_ic_for_time_marching(train_data, cfg, device)

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

    logger.info(f"  Train size: {train_data['x'].shape[0]}")
    logger.info(f"  Train data device: {train_data['x'].device}")

    # Reset the default device before creating DataLoaders (see
    # _set_default_torch_device). Model and data stay on CUDA via explicit
    # .to(device); this only affects the sampler's generator creation.
    _set_default_torch_device(device, full_batch=False)

    # Create DataLoader
    train_loader = _create_dataloader(train_data, cfg['batch_size'],
                                      shuffle=True)

    # ── 3-phase logic (root -> M-term tree spawn -> leaf training) ──
    adaptive_cfg_init = cfg['adaptive_pinn']
    is_adaptive_init = adaptive_cfg_init['enabled']
    initial_train_cfg = adaptive_cfg_init.get('initial_train', None)
    pretrained_base_checkpoint = problem_cfg.get('pretrained_base_checkpoint', None)
    pretrained_local_expert_checkpoint = problem_cfg.get(
        'pretrained_local_expert_checkpoint', None)
    _pretrained_force_spawn = False  # set True to force first-epoch spawn (checkpoint flow)

    if not is_adaptive_init:
        # Non-adaptive base-only training: single phase, no spawning.
        active_cfg = cfg
        epochs = cfg['epochs']
        phase3_epochs = 0
        current_phase = 0
        use_three_phase = False
    elif pretrained_local_expert_checkpoint is not None:
        # Full leaf-expert model supplied as a checkpoint (end-of-phase-3
        # state): root AND phase 3 are skipped; only fine-tune runs.
        if adaptive_cfg_init.get('fine_tune', None) is None:
            raise ValueError(
                "adaptive_pinn.fine_tune is required when "
                "pretrained_local_expert_checkpoint is set (it is the only "
                "segment that runs).")
        _load_pretrained_experts(model, pretrained_local_expert_checkpoint, cfg)
        active_cfg = cfg
        epochs = cfg['epochs']
        phase3_epochs = 0
        current_phase = 3
        use_three_phase = True
        if pretrained_base_checkpoint is not None:
            logger.info("  [3-Phase] pretrained_base_checkpoint ignored — the "
                        "expert checkpoint already contains the base.")
        logger.info(f"\n  [3-Phase] Phases 1+3 skipped: full leaf-expert model "
                    f"loaded from {pretrained_local_expert_checkpoint}")
        logger.info("  [3-Phase] Only the fine-tune segment will run.")
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

    # Optimizers and schedulers are built fresh per segment in _train_segment
    # (over the segment's trainable params); ctx carries only placeholders here.
    optimizer = None
    current_optimizer_name = optimizer_1_name.capitalize()
    lr_scheduler = None
    if optimizer_2_name is not None:
        logger.info(f"Optimizer plan: {optimizer_1_name} until epoch {switch_epoch}, "
              f"then {optimizer_2_name.upper()} (full-batch)")
    else:
        logger.info(f"Optimizer plan: {optimizer_1_name} for all epochs")

    step_count = 0  # global optimizer step counter for LR scheduler

    # Training setup
    print_every = cfg['print_every']
    eval_every = cfg['eval_every']
    save_every = cfg['save_every']

    # Metrics storage
    # Note: train_loss is stored every epoch; rel-L2/inf-norm (computed on the
    # ground-truth solver's native grid) only every eval_every epochs.
    metrics = {
        'train_loss_epochs': [],  # All epochs
        'train_loss': [],          # All epochs
        'epochs': [],              # Evaluation epochs only
        'rel_l2': [],              # Solver-grid rel-L2
        'inf_norm': [],            # Solver-grid inf-norm
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
            'l2sp': [],        # L2-SP anchor penalty (0 unless l2sp_lambda > 0)
            'l2sp_drift': [],  # ||theta - theta_0|| weight drift (anchor runs)
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

    best_rel_l2 = float('inf')
    best_checkpoint_path = None
    # Patience counts consecutive RESAMPLE INTERVALS in which the train loss
    # failed to improve by patience_rel_delta (start-vs-end of each interval,
    # where the point set is fixed and losses are comparable). Preferred key:
    # patience_intervals. Legacy patience_epochs / patience_evals configs are
    # converted to an equivalent interval count in the epoch loop.
    patience_intervals = cfg.get('patience_intervals')
    if 'patience_epochs' in cfg:
        patience_epochs = cfg['patience_epochs']
    elif 'patience_evals' in cfg:
        patience_epochs = cfg['patience_evals'] * max(1, cfg['eval_every'])
    else:
        patience_epochs = 0
    min_epochs = cfg['min_epochs']
    # Relative-improvement threshold for the plateau test: an eval only counts as
    # an improvement if it beats the anchored best by at least this fraction.
    patience_rel_delta = cfg.get('patience_rel_delta', 0.0)

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
            if for_tree_spawning:
                return _orig_loss_fn(model, batch,
                                     for_tree_spawning=True,
                                     update_causal_state=False)
            comps = _orig_loss_fn(model, batch, return_components=True, update_causal_state=update_causal_state)
            w = lra_weights.weights
            lra_total = sum(w.get(k, 1.0) * v for k, v in comps.items() if k != 'total')
            if return_components:
                comps['total'] = lra_total
                return comps
            return lra_total

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
        from trainer.utils import native_ground_truth_grid

        domain_bounds = model.get_domain_bounds()
        # GT heatmap background straight from the solver's native grid
        # (no eval sample, no interpolation).
        _native_gt = native_ground_truth_grid(cfg)
        if _native_gt is not None:
            gt_grid, gt_x, gt_t = _native_gt
        else:
            gt_grid, gt_x, gt_t = None, None, None
            logger.warning("  [GT] Solver native grid unavailable — plots "
                           "will render without a ground-truth background.")

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
    _pat_active = (patience_intervals if patience_intervals is not None
                   else patience_epochs)
    if _pat_active and _pat_active > 0:
        if patience_intervals is not None:
            logger.info(f"  Early stopping: enabled ({patience_intervals} consecutive "
                        f"flat resample intervals on train loss, min_epochs={min_epochs})")
        else:
            logger.info(f"  Early stopping: enabled (legacy patience_epochs="
                        f"{patience_epochs} — converted to resample intervals, "
                        f"min_epochs={min_epochs})")
    else:
        logger.info(f"  Early stopping: disabled")
    
    logger.info("=" * 60 + "\n")

    # Smart initialization (Glorot hidden + zero/LS output) — base model only
    # Skip when a pretrained base was loaded (init would destroy the trained weights).
    from trainer.init import apply_hidden_init, apply_output_init
    _init_target = model.base_model if is_adaptive else model
    _init_cfg = cfg.get('init', {})
    if pretrained_base_checkpoint is not None or pretrained_local_expert_checkpoint is not None:
        logger.info("[Init] Skipped — weights loaded from pretrained checkpoint.")
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
        train_loader=train_loader,
        plain_train_data=train_data,
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
        best_rel_l2=best_rel_l2,
        best_checkpoint_path=best_checkpoint_path,
        patience_epochs=patience_epochs,
        patience_intervals=patience_intervals,
        min_epochs=min_epochs,
        patience_rel_delta=patience_rel_delta,
        lra_weights=lra_weights,
        checkpoint_dir=checkpoint_dir,
        adaptive_cfg=adaptive_cfg,
        is_adaptive=is_adaptive,
        initial_train_cfg=initial_train_cfg,
        pretrained_base_checkpoint=pretrained_base_checkpoint,
        pretrained_local_expert_checkpoint=pretrained_local_expert_checkpoint,
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
        resample_every=resample_every,
        base_seed=base_seed,
        grad_clip_norm=grad_clip_norm,
        expert_grad_clip_norm=expert_grad_clip_norm,
        adaptive_sampling_enabled=adaptive_sampling_enabled,
        epoch=0,
        _nan_detected=False,
    )
