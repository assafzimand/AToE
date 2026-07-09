"""The per-segment epoch loop: train/eval/resample/optimizer-switch/patience."""

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
from trainer.utils import compute_infinity_norm_error, compute_native_grid_metrics
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

from trainer.setup import (
    _NumpySafeEncoder, _create_optimizer_by_name, _create_primary_optimizer,
    _create_lr_scheduler, _get_optimizer_snapshot, _create_dataloader,
    _create_split_dataloader, _save_checkpoint, _debug_print_model_state,
    _set_default_torch_device,
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
    batches_per_epoch = ctx.batches_per_epoch
    print_every = ctx.print_every
    eval_every = ctx.eval_every
    save_every = ctx.save_every
    metrics = ctx.metrics
    best_rel_l2 = ctx.best_rel_l2
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
    rel_l2 = ctx.rel_l2
    inf_norm = ctx.inf_norm
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
    # Re-establish the default-device context at the segment boundary
    # (see _set_default_torch_device for why this is needed).
    _set_default_torch_device(device, full_batch=full_batch_opt1)
    if full_batch_opt1:
        optimizer, current_optimizer_name = _create_optimizer_by_name(
            optimizer_1_name, model, seg_cfg)
        lr_scheduler = None
    else:
        optimizer, current_optimizer_name = _create_primary_optimizer(model, seg_cfg)
        lr_scheduler = _create_lr_scheduler(optimizer, seg_cfg, total_steps_estimate)

    step_count = 0
    best_patience_train_loss = float('inf')
    epochs_without_improvement = 0
    # optimizer_1 is watched from the segment start; reset to switch_epoch at the switch.
    patience_start_epoch = segment_start_epoch
    _nan_detected = False
    _stopped_early = False
    _stop_reason = 'budget'
    _lra_updated_epoch = -1
    _resample_skip_logged = False  # log the LBFGS/SSBroyden skip once per segment
    _native_fallback_logged = False  # log the native-grid fallback once per segment

    _n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _switch_str = (f" -> {optimizer_2_name.upper()}@{switch_epoch}"
                   if optimizer_2_name is not None else "")
    logger.info(f"\n[Segment:{segment_name}] start | epochs "
          f"{segment_start_epoch + 1}..{total_epochs} (budget {epoch_budget}) | "
          f"optimizer={current_optimizer_name}{_switch_str} | lr={seg_cfg['lr']} | "
          f"trainable_params={_n_train_params}")
    
    # ── DEBUG: Print comprehensive model state at segment start ──
    _debug_print_model_state(model, segment_name, ctx.train_data)
    
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
                logger.info(f"  [Adaptive Sampling] Residual caching active from epoch {epoch}")

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
                logger.info(f"  [Resample-Split] Redrawing residual interiors at epoch {epoch}")
                # Static faces + interface targets are cached for the segment;
                # only the residual collocation points are redrawn.
                train_data = build_subdomain_data(
                    _split_ctx['model_snapshot'], _split_ctx['new_expert_indices'],
                    _split_ctx['regions'], cfg, device, seed=resample_seed,
                    interface_model=_split_ctx.get('interface_model'),
                    static=_split_ctx.get('static'),
                )
                ctx.train_data = train_data
                _set_default_torch_device(device, full_batch=False)
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
                # In-place mutation: the existing DataLoader's TensorDataset holds
                # references to these same tensors, so no loader rebuild is needed.
                train_data = resample_residual_inplace(
                    train_data, cfg, device,
                    resample_seed=resample_seed,
                    cached_residuals=cached_residuals,
                    run_dir=run_dir,
                    epoch=epoch,
                    causal_state=causal_state,
                )
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
            # Log when resampling is skipped due to optimizer (once per segment)
            if not _resample_skip_logged:
                _resample_skip_logged = True
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
                        'epoch': epoch,
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
                            logger.info(f"  Expert {i}: first_layer.grad={'None' if first_grad is None else f'norm={first_grad.norm().item():.3e}'}, "
                                  f"final_layer.grad={'None' if final_grad is None else f'norm={final_grad.norm().item():.3e}'}")

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
            # Optimizer_2 is full-batch: its state tensors (e.g. Hessian
            # approximation) must be allocated on the training device.
            _set_default_torch_device(device, full_batch=True)
            _prev_opt = current_optimizer_name
            optimizer, current_optimizer_name = _create_optimizer_by_name(
                optimizer_2_name, model, seg_cfg)
            lr_scheduler = None  # optimizer_2 uses its own LR / line search
            # Reset patience at the switch; optimizer_2 gets a fresh grace window.
            epochs_without_improvement = 0
            best_patience_train_loss = float('inf')
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

            # Diagnose which loss component went NaN. No torch.no_grad() here:
            # physics losses need input autograd for the residual — the graph
            # is simply discarded without a backward pass.
            try:
                _diag_batch = next(iter(train_loader))
                _comps = loss_fn(model, _diag_batch, return_components=True,
                                 update_causal_state=False)
                _flat = {k: float(v.item()) for k, v in _comps.items()
                         if isinstance(v, torch.Tensor)}
                logger.info(f"  [NaN] Loss components: " +
                      ", ".join(f"{k}={v:.6g}" for k, v in _flat.items()))
                metrics['nan_components'] = _flat
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
                             epoch, train_loss, rel_l2, cfg, metrics)
            logger.info(f"  [NaN] Checkpoint saved to {_nan_ckpt_path}")
            logger.info(f"{'!'*60}\n")
            _nan_detected = True
            break

        # LRA: update adaptive loss weights periodically
        if lra_weights is not None and epoch > 0 and epoch % lra_weights.update_every == 0:
            try:
                batch_for_lra = next(iter(train_loader))
                if lra_weights.update(model, loss_fn, batch_for_lra):
                    _lra_updated_epoch = epoch
                if epoch % print_every == 0:
                    w_str = ', '.join(f'{k}={v:.4e}' for k, v in lra_weights.weights.items())
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
        should_evaluate = (epoch % eval_every == 0
                           or epoch == segment_start_epoch + 1
                           or epoch == total_epochs)
        
        if should_evaluate:
            # Evaluation = rel-L2 / inf-norm on the ground-truth solver's
            # NATIVE grid (the single reported metric; paper-comparable, same
            # metric as finalize, plot filenames and the comparison reports),
            # plus a full-batch loss-component snapshot on the plain training
            # set for the [LossTerms] log and the components training curve.
            model.eval()

            timer.start('eval.native_grid')
            _native = compute_native_grid_metrics(model, cfg, device)
            timer.stop('eval.native_grid')
            if _native is not None:
                rel_l2 = _native['rel_l2']
                inf_norm = _native['inf_norm']
            else:
                rel_l2 = float('nan')
                inf_norm = float('nan')
                if not _native_fallback_logged:
                    logger.warning("  [Eval] Solver native grid unavailable — "
                                   "rel-L2/inf-norm cannot be computed.")
                    _native_fallback_logged = True

            # The metric uses the model's CURRENT blending_mode (composed
            # forward): the configured mode normally, but hard indicators
            # during split segments (set by _run_split_segment) so the rel-L2
            # curve reflects what is actually being trained.

            # ── Loss-component snapshot on the plain training set ──
            # During split segments ctx.train_data holds the split schema, so
            # the snapshot probes ctx.plain_train_data through the composed
            # loss (split_loss falls back to it for plain batches). Physics
            # losses need gradients w.r.t. inputs even in eval mode.
            _probe = (train_data if isinstance(train_data, dict)
                      and 'mask' in train_data else ctx.plain_train_data)
            comp_means = {}
            if _probe is not None:
                timer.start('eval.loss_fn')
                comps = loss_fn(model, _probe, return_components=True,
                                update_causal_state=False)
                timer.stop('eval.loss_fn')
                comp_means = {
                    k: float(v.item()) if isinstance(v, torch.Tensor) else float(v)
                    for k, v in comps.items()
                }
                comp_means.pop('total', None)

            # Store evaluation metrics (train_loss already stored above for all epochs)
            metrics['epochs'].append(epoch)
            metrics['rel_l2'].append(rel_l2)
            metrics['inf_norm'].append(inf_norm)

            # ── Term-wise loss components (from the same eval pass) ──
            metrics['loss_components']['epochs'].append(epoch)
            for term in ['residual', 'ic', 'bc']:
                metrics['loss_components'][term].append(comp_means.get(term, 0.0))
            _comp_str = ', '.join(f'{k}={v:.6e}' for k, v in comp_means.items())
            logger.info(f"  [LossTerms] {_comp_str}")
            metrics['loss_components_history'].append({
                'epoch': epoch,
                **comp_means,
            })

            # Per-expert split-loss breakdown
            _split_ctx = getattr(ctx, '_split_context', None)
            if _split_ctx is not None and hasattr(loss_fn, '_per_expert_history'):
                _peh = loss_fn._per_expert_history
                for _eidx in sorted(_peh.keys()):
                    _eh = _peh[_eidx]
                    _last = {k: v[-1] for k, v in _eh.items() if v}
                    _s = ', '.join(
                        f'{k}={v:.6e}' for k, v in _last.items()
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
                  f"Train Loss: {train_loss:.6e} | "
                  f"Rel-L2 (grid): {rel_l2:.6e} | "
                  f"Inf (grid): {inf_norm:.6e}")

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

            # DIAGNOSTIC: LRA weights (+ gradient norms when updated this epoch)
            if lra_weights is not None:
                w = lra_weights.weights
                w_str = ', '.join(f'{k}={v:.4e}' for k, v in w.items())
                if _lra_updated_epoch == epoch:
                    g = lra_weights.last_grad_norms
                    g_str = ', '.join(f'{k}={g.get(k, 0):.6e}' for k in w)
                    logger.info(f"  [LRA] weights: {w_str} | grads: {g_str}")
                else:
                    logger.info(f"  [LRA] weights: {w_str}")
                # Save to metrics
                g = lra_weights.last_grad_norms
                metrics['lra_history'].append({
                    'epoch': epoch,
                    'weights': {k: float(v) for k, v in w.items()},
                    'grad_norms': {k: float(g.get(k, 0)) for k in w},
                    'updated_this_epoch': _lra_updated_epoch == epoch,
                })
            
            # ── Log gradient norms (only when computed THIS epoch; full-batch
            # optimizers don't refresh them, so stale values are not repeated) ──
            _gn = getattr(ctx, '_epoch_grad_norms', None)
            if _gn is not None and _gn.get('epoch') == epoch:
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
                          ', '.join(f'{k}={_w.get(k, 1.0):.4e}' for k in _keys))
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
                    logger.info(f"  [DIAG] Expert grad norms: {[f'{x:.3e}' for x in expert_grads[:5]]}" + ("..." if len(expert_grads) > 5 else ""))

        # Save checkpoint periodically (only when we have grid metrics)
        if epoch % save_every == 0 and rel_l2 is not None:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            _save_checkpoint(checkpoint_path, model, optimizer, current_optimizer_name, epoch,
                           train_loss, rel_l2, cfg, metrics)
            logger.info(f"  Checkpoint saved: {checkpoint_path}")

        # Save best model on the solver-grid rel-L2 (checked at eval epochs)
        if (should_evaluate and rel_l2 is not None
                and math.isfinite(rel_l2) and rel_l2 < best_rel_l2):
            best_rel_l2 = rel_l2
            best_checkpoint_path = checkpoint_dir / "best_model.pt"
            _save_checkpoint(best_checkpoint_path, model, optimizer, current_optimizer_name, epoch,
                           train_loss, rel_l2, cfg, metrics)

        # Patience-based early stopping on TRAIN loss, counted in epochs
        # (checked every epoch — train loss exists every epoch, unlike the
        # eval metrics). Active for BOTH optimizers: on an optimizer_1
        # plateau we fast-forward to the switch epoch (so the existing switch
        # handler fires and optimizer_2 keeps its full budget) rather than
        # stopping; an optimizer_2 (or no-switch) plateau stops the segment.
        # The relative min-delta means a loss creeping down by a negligible
        # amount still counts as "no improvement".
        if (patience_epochs > 0 and math.isfinite(train_loss)
                and epoch >= patience_start_epoch):
            if train_loss < best_patience_train_loss * (1.0 - patience_rel_delta):
                best_patience_train_loss = train_loss
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
                    best_patience_train_loss = float('inf')
                    continue
                else:
                    logger.info(f"\n  [EarlyStop] No train loss improvement "
                          f">{patience_rel_delta:.1%} for "
                          f"{epochs_without_improvement} epochs "
                          f"(best={best_patience_train_loss:.6e}). "
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
    ctx.best_rel_l2 = best_rel_l2
    ctx.best_checkpoint_path = best_checkpoint_path
    ctx.train_loss = train_loss
    ctx.rel_l2 = rel_l2
    ctx.inf_norm = inf_norm
    ctx.train_data = train_data
    ctx.train_loader = train_loader
    ctx._nan_detected = _nan_detected

    if _nan_detected:
        _stop_reason = 'nan'
    if not _nan_detected:
        _save_segment_pred_plot(ctx, segment_name)
    _final_tl = train_loss if train_loss is not None else float('nan')
    _final_rl2 = rel_l2 if rel_l2 is not None else float('nan')
    _oom_stopped = getattr(ctx, 'oom_stopped', False)

    # Save segment-end checkpoint
    _save_segment_checkpoint(ctx, segment_name, epoch, optimizer, current_optimizer_name,
                             train_loss, rel_l2, metrics, cfg)

    logger.info(f"[Segment:{segment_name}] done | ran {epoch - segment_start_epoch} "
          f"epochs (stop={_stop_reason}) | "
          f"train_loss={_final_tl:.6e} rel_l2={_final_rl2:.6e}")
    return SegmentResult(
        nan_detected=_nan_detected,
        stopped_early=_stopped_early,
        stop_reason=_stop_reason,
        epochs_run=epoch - segment_start_epoch,
        final_train_loss=_final_tl,
        final_rel_l2=_final_rl2,
        oom_stopped=_oom_stopped,
    )


# ======================================================================
# Staged-spawning helpers (orchestrator level; called between segments)
# ======================================================================


def _save_segment_checkpoint(ctx: TrainingContext, segment_name: str, epoch: int,
                             optimizer, optimizer_name: str, train_loss: float,
                             rel_l2: float, metrics: Dict, cfg: Dict) -> None:
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
                        train_loss, rel_l2, cfg, metrics)
        logger.info(f"  [Segment:{segment_name}] saved checkpoint_after_{segment_name}.pt")
    except Exception as e:
        logger.info(f"  [Segment:{segment_name}] checkpoint save failed: {e}")


def _save_segment_pred_plot(ctx: TrainingContext, segment_name: str) -> None:
    """Save ``pred_after_<segment>.png`` (1D problems with ground truth).

    Captures the full-composition prediction at the end of a segment so each
    stage (root, every level, fine-tune, joint) leaves a visual checkpoint.
    """
    if ctx.problem_cfg.get('spatial_dim', None) != 1:
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
            # {relL2} placeholder is filled in by the renderer
            output_path=(out_dir / f"pred_after_{segment_name}_ep{ctx.epoch}"
                                   f"_relL2_{{relL2}}.png"),
            epoch=ctx.epoch,
            cfg=ctx.cfg,
        )
        logger.info(f"  [Segment:{segment_name}] saved pred_after_{segment_name} plot")
    except Exception as _e:
        logger.info(f"  [Segment:{segment_name}] prediction plot failed: {_e}")
