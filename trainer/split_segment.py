"""Owner-imitator training segment: swaps in the tagged dataset + loss
around the epoch loop (docs/New_phase_3.tex).

Every loss term is evaluated on an expert's own raw output u_j; the PoU
composition is readout only. Locality exchange happens through minted
targets u* re-minted at every refresh (the epoch loop's resample arm):
points redrawn, lambda stepped 1 -> lambda_min, targets re-baked from
the LIVE weights, quasi-Newton optimizer state reset.
"""

import torch
from typing import Dict

from utils.logging_config import get_logger

logger = get_logger(__name__)

from trainer.plotting import plot_per_expert_curves
from trainer.training_context import TrainingContext, SegmentResult
from losses.split_loss import build_owner_imitator_loss, _pde_spatial_order
from adaptive.subdomain_data import (
    build_owner_imitator_data, build_owner_imitator_static, mint_targets,
    imit_order, KIND_NAMES, KIND_RESIDUAL, KIND_IMIT, _domain_box,
    inflated_bounds,
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
    """Swap to owner-imitator data/loss, run _train_segment, restore.

    Returns:
        SegmentResult from the inner _train_segment call.
    """
    model = ctx.model
    cfg = ctx.cfg
    problem = cfg['problem']

    leaf_info = model.get_leaf_info()
    new_expert_indices = [idx for _, idx in leaf_info if idx >= 0]
    regions_list = model.regions

    # The root supplies the lambda-weighted half of the minted targets.
    root_model = model.base_model
    assert all(not p.requires_grad for p in root_model.parameters()), (
        "[OwnerImitator] base (root) model must be frozen in phase 3")

    q = imit_order(cfg)
    m = _pde_spatial_order(problem)
    oi_cfg = (ctx.adaptive_cfg.get('owner_imitator', {}) or {})
    lambda_min = float(oi_cfg.get('mint_lambda_min', 0.0) or 0.0)
    epochs_on_min = int(oi_cfg.get('epochs_on_min_lambda', 0) or 0)

    # Refresh schedule: mirror the epoch loop's resample gate exactly
    # (epoch > 1 and (epoch - 1) % resample_every == 0) over this
    # segment's epoch range. The linear 1 -> lambda_min ramp spans only
    # the refreshes up to (segment end - epochs_on_min_lambda); the
    # remaining refreshes hold lambda_min (pure exchange window).
    seg_start = ctx.epoch
    resample_every = ctx.resample_every
    if resample_every and resample_every > 0:
        _refresh_epochs = [
            e for e in range(seg_start + 1, seg_start + epoch_budget + 1)
            if e > 1 and (e - 1) % resample_every == 0]
        n_refresh_total = len(_refresh_epochs)
        ramp_end = seg_start + epoch_budget - epochs_on_min
        n_ramp_refreshes = sum(1 for e in _refresh_epochs if e <= ramp_end)
    else:
        n_refresh_total = 0
        n_ramp_refreshes = 0
    if n_refresh_total > 0 and n_ramp_refreshes == 0:
        logger.warning(
            "[OwnerImitator] epochs_on_min_lambda leaves no refreshes in "
            "the ramp window — lambda jumps to lambda_min at the first "
            "refresh.")

    _slot_names = (['value']
                   + [f'u_x^{p}' if p > 1 else 'u_x' for p in range(1, q + 1)]
                   + [f'u_t^{p}' if p > 1 else 'u_t' for p in range(1, q + 1)])
    _ladder = [1.0] + [
        max(lambda_min,
            1.0 - k * (1.0 - lambda_min) / max(1, n_ramp_refreshes))
        for k in range(1, n_refresh_total + 1)]
    logger.info(
        f"[OwnerImitator] config: lambda_min={lambda_min}, "
        f"epochs_on_min_lambda={epochs_on_min} "
        f"(ramp over {n_ramp_refreshes}/{n_refresh_total} refreshes), "
        f"q={q} (slots {_slot_names}; PDE order m={m}), "
        f"R={resample_every} epochs, "
        f"lambda ladder {[round(v, 4) for v in _ladder]}")
    logger.info(
        f"[OwnerImitator] {len(new_expert_indices)} experts; every term "
        f"on the expert's own output; composition u_theta is readout only")
    if ctx.lra_weights is not None:
        logger.warning("[OwnerImitator] LRA is enabled — untested with the "
                       "owner-imitator loss component structure")

    # ── Data: static IC/BC-or-PER rows (fixed) + dynamic draw ──
    static = build_owner_imitator_static(
        ctx.plain_train_data, new_expert_indices, regions_list, cfg,
        ctx.device, seed=ctx.epoch)
    split_data = build_owner_imitator_data(
        new_expert_indices, regions_list, cfg, ctx.device,
        seed=ctx.epoch, static=static)

    # Initial mint at lambda=1: bit-for-bit root targets.
    _stats = mint_targets(split_data, model, root_model, 1.0, q)
    logger.info(
        f"[OwnerImitator] initial mint at lambda=1.0 (== root targets): "
        f"n_minted={_stats['n_minted']}, "
        f"max|mint - root| = {_stats['mean_abs_dev_from_root']:.1e}")

    _log_subdomain_summary(new_expert_indices, regions_list, split_data, cfg)

    # Freeze/trainable confirmation
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    logger.info(f"[OwnerImitator] trainable params: {len(trainable)}, "
                f"frozen params: {len(frozen)}")

    # Stash original context state
    orig_loss_fn = ctx.loss_fn
    orig_train_data = ctx.train_data
    orig_train_loader = ctx.train_loader

    oi_loss = build_owner_imitator_loss(
        model, cfg, orig_loss_fn=orig_loss_fn)
    oi_loss._mint_lambda = 1.0

    # Swap to owner-imitator data/loss
    ctx.loss_fn = oi_loss
    ctx.train_data = split_data
    ctx.train_loader = _create_split_dataloader(
        split_data, segment_cfg.get('batch_size', cfg['batch_size']),
        shuffle=True,
    )
    ctx._split_context = {
        'model': model,
        'root_model': root_model,
        'new_expert_indices': new_expert_indices,
        'regions': regions_list,
        'static': static,          # fixed rows; dynamic part redrawn on refresh
        'lambda': 1.0,
        'lambda_min': lambda_min,
        'refresh_count': 0,
        'n_refresh_total': n_refresh_total,
        'n_ramp_refreshes': n_ramp_refreshes,
        'imit_order': q,
    }

    logger.info("[OwnerImitator] Eval/plots use the blended "
                f"'{getattr(model, 'blending_mode', 'soft')}' PoU "
                "composition (readout).")

    res = _train_segment(ctx, segment_name, epoch_budget, segment_cfg,
                         lr_override=lr_override,
                         min_epochs_override=min_epochs_override)

    # Save per-expert loss histories into metrics
    peh = getattr(oi_loss, '_per_expert_history', {})
    if peh:
        ctx.metrics.setdefault('split_expert_losses', {})[segment_name] = peh

    # Per-expert training curves + sampling-map panel
    def _to_numpy(v):
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return v.cpu().numpy()
        return v

    try:
        pc = cfg[problem]
        g_lo, g_hi = _domain_box(pc)
        sigma_fraction = cfg['adaptive_pinn']['sigma_fraction']
        inflated_boxes = {
            eidx: inflated_bounds(regions_list[eidx], sigma_fraction,
                                  g_lo, g_hi)
            for eidx in new_expert_indices
        }
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
            split_data=ctx.train_data,   # last refresh's draw
            inflated_boxes=inflated_boxes,
        )
        logger.info(f"[OIPlot] Saved training_plots/{plot_path.name}")
    except Exception as e:
        logger.warning(f"[OIPlot] Failed: {e}")

    # Restore original context
    ctx.loss_fn = orig_loss_fn
    ctx.train_data = orig_train_data
    ctx.train_loader = orig_train_loader
    ctx._split_context = None

    return res


def _log_subdomain_summary(new_expert_indices, regions, split_data, cfg):
    """Per-expert point summary of the owner-imitator dataset.

    One line per expert: hard tile, inflated box, per-kind counts, and
    the imitation rows' owner distribution — verifies the ownership
    partition, the collar filtering, and that IC/BC/PER rows attach only
    to boundary-touching tiles.
    """
    expert_ids = split_data['expert_id']
    kinds = split_data['kind']
    mint_owner = split_data['mint_owner']

    problem = cfg['problem']
    pc = cfg[problem]
    sigma_fraction = cfg['adaptive_pinn']['sigma_fraction']
    g_lo, g_hi = _domain_box(pc)

    n_res_total = int((kinds == KIND_RESIDUAL).sum())
    n_imit_total = int((kinds == KIND_IMIT).sum())
    logger.info(f"[OIData] dataset: {int(kinds.shape[0])} rows total — "
                f"{n_res_total} residual (owner-partitioned), "
                f"{n_imit_total} imit (duplicated per covering expert)")

    _fmt = lambda v: [round(float(x), 6) for x in v]
    for eidx in new_expert_indices:
        emask = (expert_ids == eidx)
        region = regions[eidx]
        infl_lo, infl_hi = inflated_bounds(
            region, sigma_fraction, g_lo, g_hi)
        counts = {}
        for k_val, k_name in KIND_NAMES.items():
            counts[k_name] = int((kinds[emask] == k_val).sum())
        imit_mask = emask & (kinds == KIND_IMIT)
        owners = mint_owner[imit_mask]
        owner_counts = {int(o): int((owners == o).sum())
                        for o in owners.unique().tolist()}
        active_terms = [k for k, v in counts.items() if v > 0]
        logger.info(
            f"[OIData] expert={eidx} depth={region.depth} "
            f"tile=[{_fmt(region.bounds_lower)}..{_fmt(region.bounds_upper)}] "
            f"inflated=[{_fmt(infl_lo)}..{_fmt(infl_hi)}] "
            f"counts={counts} imit_owners={owner_counts} "
            f"active_terms={active_terms}"
        )
