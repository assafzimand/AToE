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

from trainer.setup import _save_checkpoint, _NumpySafeEncoder


def _compute_dense_grid_rel_l2(model, cfg, device, chunk_size: int = 65536):
    """Rel-L2 of the model against the solver's native dense solution grid.

    This is the headline number comparable to the literature (which reports
    dense-grid rel-L2, not a random-subsample estimate). The grid is restricted
    to the config's temporal domain so time-marching windows are scored on
    their own window.

    Returns:
        (rel_l2, n_points, grid_shape) or None if the solver grid is unavailable.
    """
    import importlib

    problem = cfg['problem']
    try:
        solver = importlib.import_module(f'solvers.{problem}_solver')
        x_grid, t_grid, h_sol = solver._get_solution_cached(cfg)
    except Exception as e:
        logger.warning(f"  [DenseRelL2] Solver grid unavailable for '{problem}': {e}")
        return None

    t0, t1 = cfg[problem]['temporal_domain']
    t_mask = (t_grid >= t0 - 1e-12) & (t_grid <= t1 + 1e-12)
    t_grid = t_grid[t_mask]
    h_sol = h_sol[t_mask]  # (nt, nx), complex for schrodinger

    # Flatten grid to (N, 2) inputs and (N, output_dim) ground truth
    T, X = np.meshgrid(t_grid, x_grid, indexing='ij')
    xt = np.column_stack([X.ravel(), T.ravel()])
    if np.iscomplexobj(h_sol):
        gt = np.column_stack([h_sol.real.ravel(), h_sol.imag.ravel()])
    else:
        gt = h_sol.reshape(-1, 1)

    dtype = next(model.parameters()).dtype
    model.eval()
    total_diff_sq = 0.0
    total_gt_sq = 0.0
    with torch.no_grad():
        for start in range(0, xt.shape[0], chunk_size):
            xb = torch.tensor(xt[start:start + chunk_size], dtype=dtype, device=device)
            gb = torch.tensor(gt[start:start + chunk_size], dtype=dtype, device=device)
            pred = model(xb)
            total_diff_sq += ((pred - gb) ** 2).sum().item()
            total_gt_sq += (gb ** 2).sum().item()

    rel_l2 = math.sqrt(total_diff_sq) / (math.sqrt(total_gt_sq) + 1e-10)
    return rel_l2, xt.shape[0], (len(t_grid), len(x_grid))


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
            _switch_epochs = [e['epoch'] for e in metrics.get('optimizer_events', [])]
            _segment_starts = [s['start_epoch'] for s in metrics.get('segment_events', [])]
            plot_training_curves(metrics, training_plots_dir,
                                 optimizer_switch_epochs=_switch_epochs,
                                 segment_start_epochs=_segment_starts)
        except Exception as _plot_err:
            logger.info(f"  [NaN] Could not generate training curves: {_plot_err}")
        logger.info("[NaN] Skipping remaining post-training cleanup — moving to next experiment.")
        return

    # Save final model
    final_checkpoint_path = checkpoint_dir / "final_model.pt"
    _save_checkpoint(final_checkpoint_path, model, optimizer, current_optimizer_name, total_epochs,
                    train_loss, eval_loss, cfg, metrics)

    logger.info(f"\nTraining completed in {time.time() - start_time:.1f}s")
    logger.info(f"  Best eval loss: {best_eval_loss:.6e}")
    logger.info(f"  Best checkpoint: {best_checkpoint_path}")
    logger.info(f"  Final checkpoint: {final_checkpoint_path}")

    # ── Headline metric: rel-L2 on the solver's dense solution grid ──
    dense_rel_l2 = None
    _dense = _compute_dense_grid_rel_l2(model, cfg, device)
    if _dense is not None:
        dense_rel_l2, _n_dense, _dense_shape = _dense
        metrics['final_dense_rel_l2'] = dense_rel_l2
        metrics['final_dense_grid_shape'] = list(_dense_shape)
        logger.info(f"  Final dense-grid rel-L2: {dense_rel_l2:.6e} "
                    f"(grid {_dense_shape[0]}x{_dense_shape[1]} = {_n_dense:,} points)")
    
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
        is_leaves_model = isinstance(model, AToELeaves)
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
        f.write(f"Final train loss: {train_loss:.6e}\n")
        f.write(f"Final eval loss: {eval_loss:.6e}\n" if eval_loss is not None else "Final eval loss: N/A\n")
        f.write(f"Final eval rel-L2 (subsample): {eval_rel_l2:.6e}\n" if eval_rel_l2 is not None else "Final eval rel-L2 (subsample): N/A\n")
        f.write(f"Final dense-grid rel-L2: {dense_rel_l2:.6e}\n" if dense_rel_l2 is not None else "Final dense-grid rel-L2: N/A\n")
        f.write(f"Final eval inf-norm: {eval_inf_norm:.6e}\n" if eval_inf_norm is not None else "Final eval inf-norm: N/A\n")
        f.write(f"Best eval loss: {best_eval_loss:.6e}\n\n")
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
        eval_data_path = Path("datasets") / cfg['problem'] / "eval_data.pt"
        visualize_evaluation(model, str(eval_data_path), run_dir, cfg)
    except ValueError as e:
        logger.info(f"  (No custom evaluation visualization for {cfg['problem']})")
        logger.info(f"  ValueError details: {e}")
    except Exception as e:
        logger.info(f"  Warning: Could not generate evaluation visualization: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    return best_checkpoint_path
