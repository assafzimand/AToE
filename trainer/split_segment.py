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

    # Frozen snapshot reused on resample (kept for API stability; the
    # split dataset no longer mints any targets from it).
    model_snapshot = copy.deepcopy(model)
    model_snapshot.eval()
    for p in model_snapshot.parameters():
        p.requires_grad = False

    # Identify the leaf experts being trained in this segment
    leaf_info = model.get_leaf_info()
    new_expert_indices = [idx for _, idx in leaf_info if idx >= 0]

    regions_list = model.regions

    logger.info(f"[SplitLoss] Building subdomain data for {len(new_expert_indices)} "
                f"new expert(s): {new_expert_indices}")

    # Static rows (optional continuity pairs) are constant within the
    # segment: built once here, reused on every resample.
    split_static = build_subdomain_static(
        model_snapshot, new_expert_indices, regions_list, cfg,
        ctx.device, seed=ctx.epoch,
    )
    split_data = build_subdomain_data(
        model_snapshot, new_expert_indices, regions_list, cfg,
        ctx.device, seed=ctx.epoch,
        static=split_static,
    )

    _log_subdomain_summary(new_expert_indices, regions_list, split_data, cfg)

    # Freeze/trainable confirmation
    trainable = [n for n, p in model.named_parameters()
                 if p.requires_grad]
    frozen = [n for n, p in model.named_parameters()
              if not p.requires_grad]
    logger.info(
        "[SplitLoss] Grouped composed-residual mode: PDE residual on the "
        "blended PoU composition as a sum of per-group means (one per "
        "expert solo zone + one collar mean); no u0 guide terms"
    )
    logger.info(
        f"[SplitLoss] trainable params: {len(trainable)}, "
        f"frozen params: {len(frozen)}"
    )

    # Stash original context state
    orig_loss_fn = ctx.loss_fn
    orig_train_data = ctx.train_data
    orig_train_loader = ctx.train_loader

    # Composition IC/BC batch: the plain training set's IC/BC rows with an
    # all-false residual mask. The split loss runs the ORIGINAL global loss
    # on it through the blended PoU composition — exact physics (periodic
    # pairing included) enforced once, on the reported object, at full
    # weight. (In time-marching windows these rows already carry the
    # previous window's IC override.)
    pd = ctx.plain_train_data
    _sel = pd['mask']['IC'] | pd['mask']['BC']
    ic_bc_batch = {
        'x': pd['x'][_sel],
        't': pd['t'][_sel],
        'h_gt': pd['h_gt'][_sel],
        'mask': {
            'residual': torch.zeros(int(_sel.sum()), dtype=torch.bool,
                                    device=pd['x'].device),
            'IC': pd['mask']['IC'][_sel],
            'BC': pd['mask']['BC'][_sel],
        },
    }
    logger.info(f"[SplitLoss] composition IC/BC term: "
                f"n_ic={int(ic_bc_batch['mask']['IC'].sum())}, "
                f"n_bc={int(ic_bc_batch['mask']['BC'].sum())} "
                f"(exact physics on the blended PoU, full weight)")

    # Build split loss with original loss as fallback for eval batches and
    # as the source of the composition IC/BC terms.
    split_loss = build_split_loss(
        model, cfg, orig_loss_fn=orig_loss_fn, ic_bc_batch=ic_bc_batch,
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
        'static': split_static,  # cached continuity rows; resample redraws residuals only
    }

    # D2 reporting: experts train on their inflated boxes (region + collar),
    # exactly the support of their blending windows, so eval rel-L2, best-
    # checkpoint selection, and pred_after_<segment>.png all use the blended
    # POU composition — the metric curve matches the final model throughout.
    logger.info("[SplitLoss] Eval/plots use the blended "
                f"'{getattr(model, 'blending_mode', 'soft')}' PoU "
                "composition (D2 reporting).")

    res = _train_segment(ctx, segment_name, epoch_budget, segment_cfg,
                         lr_override=lr_override,
                         min_epochs_override=min_epochs_override)

    # Save per-expert and composition loss histories into metrics
    peh = getattr(split_loss, '_per_expert_history', {})
    if peh:
        if 'split_expert_losses' not in ctx.metrics:
            ctx.metrics['split_expert_losses'] = {}
        ctx.metrics['split_expert_losses'][segment_name] = peh
    gh = getattr(split_loss, '_global_history', {})
    if gh and any(gh.values()):
        ctx.metrics.setdefault('split_composition_losses', {})[
            segment_name] = gh

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
        plot_path = (training_plots_dir /
                     f'expert_curves_after_{segment_name}'
                     f'_E{len(new_expert_indices)}.png')
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


def _log_subdomain_summary(new_expert_indices, regions, split_data, cfg):
    """Log per-expert point summaries for the subdomain dataset.

    Residual rows are one uniform draw over the whole domain, tagged with
    the loss GROUP: the leaf whose exclusive (solo) box contains the
    point, or -1 for collar points. The grouped residual takes one mean
    per group at weight 1 each. Guide faces live on each expert's
    exclusive box; swallowed leaves have none. The summary logs each
    expert's exclusive box next to its hard region, its solo-group point
    count, the per-kind face counts, and the collar-group size.
    """
    from adaptive.subdomain_data import (
        _domain_box, KIND_RESIDUAL, exclusive_bounds, is_swallowed,
    )

    expert_ids = split_data['expert_id']
    kinds = split_data['kind']
    cont_neighbors = split_data.get('cont_neighbor', None)

    problem = cfg['problem']
    pc = cfg[problem]
    spatial_dim = pc['spatial_dim']
    sigma_fraction = cfg['adaptive_pinn']['sigma_fraction']
    g_lo, g_hi = _domain_box(pc)

    rmask_all = (kinds == KIND_RESIDUAL)
    n_res_total = int(rmask_all.sum())
    n_collar = int((rmask_all & (expert_ids == -1)).sum())
    logger.info(f"[SplitData] grouped composed residual: {n_res_total} "
                f"uniform points; collar group={n_collar} points, solo "
                f"groups get one weight-1 mean per expert")

    for eidx in new_expert_indices:
        emask = (expert_ids == eidx)
        n_total = emask.sum().item()
        region = regions[eidx]
        excl_bl, excl_bu = exclusive_bounds(
            eidx, new_expert_indices, regions, sigma_fraction, g_lo, g_hi)
        swallowed = is_swallowed(excl_bl, excl_bu)
        counts = {}
        for k_val, k_name in KIND_NAMES.items():
            counts[k_name] = ((kinds[emask] == k_val).sum().item() if n_total > 0 else 0)
        n_solo = counts.get('residual', 0)

        _fmt = lambda v: [round(float(x), 6) for x in v]
        logger.info(
            f"[SplitData] expert={eidx} depth={region.depth} parent={region.parent_idx} "
            f"hard=[{region.bounds_lower}..{region.bounds_upper}] "
            f"exclusive=[{_fmt(excl_bl)}..{_fmt(excl_bu)}]"
            f"{' SWALLOWED' if swallowed else ''} "
            f"total={n_total} residual_solo_group={n_solo} {counts}"
        )
        if n_solo == 0 and not swallowed:
            logger.warning(f"[SplitData] expert={eidx} solo group is EMPTY "
                           f"this draw (no residual mean of its own until "
                           f"the next resample)")
        if swallowed:
            logger.info(f"[SplitData] expert={eidx} is swallowed: trained "
                        f"via the collar group of the composed residual")
    
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
