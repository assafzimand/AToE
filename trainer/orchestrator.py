"""Training orchestration: phase sequencing, M-term tree build, and leaf-expert spawning."""

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

from trainer.setup import _setup_training, _NumpySafeEncoder
from trainer.epoch_loop import _train_segment
from trainer.split_segment import _run_split_segment
from trainer.finalize import _finalize_training


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
        _run_split_segment(ctx, 'phase3', cfg['epochs'], cfg)
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
                            if isinstance(model, AToELeaves) else None)
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
