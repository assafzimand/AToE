"""Post-training finalization: final checkpoint, metrics, summary, and plots."""

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

from trainer.setup import _save_checkpoint, _NumpySafeEncoder


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
    metrics = ctx.metrics
    timer = ctx.timer
    checkpoint_dir = ctx.checkpoint_dir
    optimizer = ctx.optimizer
    current_optimizer_name = ctx.current_optimizer_name
    epochs = ctx.epochs
    total_epochs = ctx.total_epochs
    train_loss = ctx.train_loss
    rel_l2 = ctx.rel_l2
    inf_norm = ctx.inf_norm
    best_rel_l2 = ctx.best_rel_l2
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
            _switch_epochs = [e['epoch'] for e in metrics.get('optimizer_events', [])]
            _segment_markers = [(s['start_epoch'], s.get('segment', ''))
                                for s in metrics.get('segment_events', [])]
            plot_training_curves(metrics, training_plots_dir,
                                 optimizer_switch_epochs=_switch_epochs,
                                 segment_markers=_segment_markers)
        except Exception as _plot_err:
            logger.info(f"  [NaN] Could not generate training curves: {_plot_err}")
        logger.info("[NaN] Skipping remaining post-training cleanup — moving to next experiment.")
        return

    # No separate final checkpoint: segment-end reconciliation guarantees the
    # in-memory model == best_model_<last_segment>.pt (the run's result).
    logger.info(f"\nTraining completed in {time.time() - start_time:.1f}s")
    logger.info(f"  Best rel-L2 (solver grid, last segment): {best_rel_l2:.6e}")
    logger.info(f"  Best checkpoint: {best_checkpoint_path}")

    # ── Headline metric: rel-L2 on the solver's dense solution grid ──
    dense_rel_l2 = None
    _dense = compute_native_grid_metrics(model, cfg, device)
    if _dense is None:
        logger.warning(f"  [DenseRelL2] Solver grid unavailable for '{cfg['problem']}'")
    else:
        dense_rel_l2 = _dense['rel_l2']
        _dense_shape = _dense['grid_shape']
        metrics['final_dense_rel_l2'] = dense_rel_l2
        metrics['final_dense_grid_shape'] = list(_dense_shape)
        logger.info(f"  Final dense-grid rel-L2: {dense_rel_l2:.6e} "
                    f"(grid {_dense_shape[0]}x{_dense_shape[1]} = {_dense['n_points']:,} points)")
    
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
    # Extract all optimizer switch epochs and segment markers from metrics
    optimizer_switch_epochs = [e['epoch'] for e in metrics.get('optimizer_events', [])]
    segment_markers = [(s['start_epoch'], s.get('segment', ''))
                       for s in metrics.get('segment_events', [])]
    # Run metadata goes into the filenames (paper-ready; captions carry it)
    _n_exp = model.num_experts if hasattr(model, 'num_experts') else 0
    _curves_suffix = f"{cfg.get('problem', '')}_ep{total_epochs}_E{_n_exp}"
    plot_training_curves(metrics, training_plots_dir,
                         optimizer_switch_epochs=optimizer_switch_epochs,
                         segment_markers=segment_markers,
                         name_suffix=_curves_suffix)

    # Final adaptive PINN outputs
    if is_adaptive and hasattr(model, 'num_experts') and model.num_experts > 0:
        logger.info("\n" + "=" * 60)
        logger.info("Adaptive PINN Final Summary")
        logger.info("=" * 60)
        logger.info(f"  Total experts spawned: {model.num_experts}")
        
        problem_type = '2d' if len(domain_bounds['lower']) == 2 else '3d'
        is_leaves_model = isinstance(model, AToELeaves)
        leaf_info = model.get_leaf_info()
        leaf_expert_indices = [idx for _, idx in leaf_info if idx >= 0]
        regions_to_plot = (
            [model.regions[i] for i in leaf_expert_indices]
            if is_leaves_model else model.regions
        )
        _n_final = len(regions_to_plot)
        plot_expert_regions(
            regions=regions_to_plot,
            domain_bounds=domain_bounds,
            output_path=adaptive_plots_dir / f"expert_regions_final_E{_n_final}.png",
            problem_type=problem_type,
            ground_truth=gt_grid,
            grid_x=gt_x,
            grid_t=gt_t
        )

        if adaptive_cfg['blending_mode'] == 'soft' and problem_type == '2d':
            leaf_indices_set = set(leaf_expert_indices) if is_leaves_model else None
            plot_expert_soft_weights(
                model=model,
                domain_bounds=domain_bounds,
                output_path=adaptive_plots_dir / f"soft_weights_final_E{_n_final}.png",
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
        f.write(f"Final train loss: {train_loss:.6e}\n")
        f.write(f"Final rel-L2 (solver grid): {dense_rel_l2:.6e}\n" if dense_rel_l2 is not None else "Final rel-L2 (solver grid): N/A\n")
        f.write(f"Final inf-norm (solver grid): {inf_norm:.6e}\n" if inf_norm is not None else "Final inf-norm (solver grid): N/A\n")
        # Per-segment bests (each == its best_model_<segment>.pt)
        for _ev in metrics.get('segment_reconcile_events', []):
            _kept = _ev.get('best_rel_l2') if _ev.get('kept') == 'best' else _ev.get('final_rel_l2')
            if _kept is not None:
                f.write(f"Best rel-L2 [{_ev['segment']}]: {_kept:.6e}\n")
        f.write(f"\nBest checkpoint: {best_checkpoint_path}\n")
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
        visualize_evaluation(model, run_dir, cfg)
    except ValueError as e:
        logger.info(f"  (No custom evaluation visualization for {cfg['problem']})")
        logger.info(f"  ValueError details: {e}")
    except Exception as e:
        logger.info(f"  Warning: Could not generate evaluation visualization: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    return best_checkpoint_path
