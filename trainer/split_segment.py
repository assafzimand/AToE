"""Per-leaf split training segment: swaps in subdomain data + split loss around the epoch loop."""

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
from adaptive.subdomain_data import (
    build_subdomain_data, build_subdomain_static, KIND_NAMES,
)

from trainer.setup import _create_split_dataloader
from trainer.epoch_loop import _train_segment


def _run_split_segment(
    ctx: TrainingContext,
    segment_name: str,
    epoch_budget: int,
    segment_cfg: Dict,
    *,
    lr_override=None,
    min_epochs_override=None,
) -> SegmentResult:
    """Swap to split-loss data/loss, run _train_segment, then restore originals.

    Returns:
        SegmentResult from the inner _train_segment call.
    """
    model = ctx.model
    cfg = ctx.cfg

    # Snapshot the model BEFORE training so interface targets stay stable
    # across the whole segment (and across resamples).
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

    # Static faces (IC/BC/interface/continuity + minted targets) are constant
    # within the segment: built once here, reused on every resample.
    split_static = build_subdomain_static(
        model_snapshot, new_expert_indices, regions_list, cfg,
        ctx.device, seed=ctx.epoch,
        interface_model=interface_model,
    )
    split_data = build_subdomain_data(
        model_snapshot, new_expert_indices, regions_list, cfg,
        ctx.device, seed=ctx.epoch,
        interface_model=interface_model,
        static=split_static,
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
        model, cfg, orig_loss_fn=orig_loss_fn,
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
        'interface_model': interface_model,  # frozen base for interface targets
        'static': split_static,  # cached faces + targets; resample redraws residuals only
    }

    # During split training each expert learns its OWN region (hard ownership
    # + interface losses), so the composed forward used by eval rel-L2 and by
    # pred_after_<segment>.png must use HARD indicators to reflect what is
    # actually being trained. Restored to the configured blending after the
    # segment (fine-tune / inference use the configured mode).
    orig_blending = getattr(model, 'blending_mode', None)
    if orig_blending is not None and orig_blending != 'hard':
        model.blending_mode = 'hard'
        logger.info("[SplitLoss] Eval/plots use HARD indicators for this "
                    f"segment (configured '{orig_blending}' restored after).")

    try:
        res = _train_segment(ctx, segment_name, epoch_budget, segment_cfg,
                             lr_override=lr_override,
                             min_epochs_override=min_epochs_override)
    finally:
        if orig_blending is not None:
            model.blending_mode = orig_blending

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
