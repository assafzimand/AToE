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
    plot_per_expert_region_report,
)
from trainer.utils import (
    compute_infinity_norm_error, compute_native_grid_metrics,
    per_region_rel_l2,
)
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


def _build_rad_pool(model, loss_fn, cfg, device, train_data):
    """Fresh uniform candidate pool + its PDE residual, for RAD re-targeting.

    Wu et al. (CMAME 2023) draw a large uniform pool each RAD cycle and select
    the collocation set from it. Reusing the previous epoch's own points, as
    this path used to, is a bootstrap filter: the support feeds on itself and
    only the uniform half rejuvenates it.

    Returns ``[(x, t, r2)]`` in the shape the sampler expects, or None if the
    residual probe fails (in which case the caller keeps the previous pool and
    training is unaffected).
    """
    n_pool = int((cfg.get('sampling', {}) or {}).get(
        'rad_pool_size', 100000) or 100000)
    problem = cfg['problem']
    pc = cfg[problem]
    sd = pc['spatial_dim']
    dtype = train_data['x'].dtype
    try:
        x = torch.empty(n_pool, sd, device=device, dtype=dtype)
        for d in range(sd):
            lo, hi = pc['spatial_domain'][d]
            x[:, d].uniform_(float(lo), float(hi))
        t = torch.empty(n_pool, 1, device=device, dtype=dtype)
        t.uniform_(float(pc['temporal_domain'][0]), float(pc['temporal_domain'][1]))

        # Chunked: the residual needs 2nd derivatives, and in the fine-tune the
        # forward is the full PoU composition (every leaf evaluated at every
        # point). A single 100k-point graph over ~10 experts is a large
        # allocation for one diagnostic draw, and this process has been
        # OOM-killed before. Chunks are independent -- no cross-chunk graph --
        # so this is numerically identical to one shot.
        chunk = int((cfg.get('sampling', {}) or {}).get('rad_pool_chunk', 20000)
                    or 20000)
        was_training = model.training
        model.eval()
        r2_parts = []
        for s in range(0, n_pool, chunk):
            e = min(s + chunk, n_pool)
            m = e - s
            batch = {
                'x': x[s:e], 't': t[s:e],
                'h_gt': torch.zeros(m, pc['output_dim'],
                                    device=device, dtype=dtype),
                'mask': {
                    'residual': torch.ones(m, dtype=torch.bool, device=device),
                    'IC': torch.zeros(m, dtype=torch.bool, device=device),
                    'BC': torch.zeros(m, dtype=torch.bool, device=device),
                },
            }
            out = loss_fn(model, batch, for_tree_spawning=True,
                          update_causal_state=False)
            r2_parts.append(out['residual'].detach())
        if was_training:
            model.train()
        r2 = torch.cat(r2_parts)
    except Exception as exc:                                    # noqa: BLE001
        logger.warning(f"  [RAD] pool refresh failed ({type(exc).__name__}: "
                       f"{exc}); keeping the previous pool.")
        return None
    return [(x.detach(), t.detach(), r2)]


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
    # Best-model tracking is PER SEGMENT: reset at every segment start; the
    # segment's best is reconciled against the end-of-segment weights below.
    best_rel_l2 = float('inf')
    _best_epoch = None
    best_checkpoint_path = (ctx.checkpoint_dir / f"best_model_{segment_name}.pt"
                            if ctx.checkpoint_dir is not None else None)
    patience_evals = ctx.patience_evals
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
    # Per-segment cadence override: a flat `resample_every_epochs` inside a
    # segment block (fine_tune / initial_train) lands in segment_cfg via the
    # segment-config flatten (e.g. ft_cfg.update(fine_tune_cfg)); it takes
    # precedence over the global sampling.resample_every_epochs. Phase 3's
    # segment cfg is the full config, which has no flat key -> global value.
    _seg_resample = segment_cfg.get('resample_every_epochs')
    if _seg_resample is not None:
        resample_every = int(_seg_resample)
        logger.info(f"  [Sampling] {segment_name}: segment-local resample "
                    f"cadence = {resample_every} (overrides global)")
    base_seed = ctx.base_seed
    grad_clip_norm = ctx.grad_clip_norm
    expert_grad_clip_norm = ctx.expert_grad_clip_norm
    adaptive_sampling_enabled = ctx.adaptive_sampling_enabled
    # ── Phase 3 always samples UNIFORMLY, whatever the config says ──────────
    # Phase 3 trains each leaf on its own subdomain, with the interfaces
    # mounted from the root's prediction, so the expert only has to learn a
    # local PDE with no transition inside its box. The decomposition has
    # ALREADY done the spatial adaptation: the wavelet tree makes boxes small
    # where the solution is hard and large where it is smooth, and each box is
    # accepted only once it is locally smooth enough. Residual-weighted
    # sampling inside such a box has little left to concentrate on.
    #
    # It would also be near-useless here even if it did: the split loss reduces
    # per expert (`{eidx: r2[s:e].mean()}`, summed with equal weight across
    # experts), so drawing more points into one leaf does not raise its weight
    # -- only the within-box distribution could change. And routing through the
    # global-draw-then-filter path to get it would give up the direct
    # per-expert draw, whose whole point is avoiding the data-dependent mask
    # shapes that cost ~163 device->host copies per step.
    #
    # Adaptive sampling stays on for the root and, deliberately, for the joint
    # fine-tune: that is the only segment where an expert is evaluated OUTSIDE
    # its own box (the PoU blends it across the collar), so it is the only
    # segment where residual-weighted sampling can teach it its collar
    # behaviour. On the phase-3 checkpoints, 99.7-99.97% of the composed
    # residual sits in those collars, so the fine-tune's adaptive draw lands
    # there without being aimed.
    if segment_name == 'phase3' and adaptive_sampling_enabled:
        logger.info(
            "  [Sampling] phase3: adaptive sampling ENABLED (LOCALIZED: each "
            "expert keeps its own point budget and draws its adaptive share "
            "from its OWN cached residuals; redraws follow the resample "
            "cadence instead of the static-draw default).")
    # ── Phase 3 under a full-batch optimizer trains on a STATIC draw ────────
    # Redrawing the per-expert interiors mid-segment turned phase 3 into a
    # moving-target problem the experts cannot converge on. Measured on the
    # full_flow_root64x5_rad burgers run (500-epoch cadence, SSBroyden):
    # each 921-param expert ground its ~3k-point draw to ~2e-8 loss, and the
    # SAME weights scored ~1e-6 on the next fresh draw — a 50-250x
    # memorization gap, repeating at every redraw, with rel-L2 plateauing at
    # ~1e-5 (50x worse than the static-draw regime, which reached ≈ root
    # accuracy). The ROOT survives the same cadence because its redraw
    # perturbs one RAD-targeted 20k-point global mean; phase 3's objective is
    # a sum of small per-expert means with shock-concentrated tails, so the
    # per-redraw jolt is orders larger. Adam/SOAP keep their every-epoch
    # refresh (stochastic steps average over draws — the proven warmup
    # regime); only the deterministic full-batch phase goes static.
    # Static remains the DEFAULT for phase 3 (the measured burgers failure
    # above). adaptive_sampling.enabled=true opts phase 3 into the normal
    # resample cadence, with the LOCALIZED per-expert adaptive redraws
    # (see _sample_per_expert_localized_adaptive) instead of uniform ones.
    _phase3_static = (segment_name == 'phase3'
                      and not adaptive_sampling_enabled)
    if _phase3_static and resample_every > 0:
        logger.info(
            "  [Sampling] phase3: residual draw is STATIC under full-batch "
            "optimizers for this segment (one draw at segment start; "
            "Adam/SOAP would still refresh every epoch).")
    # Diagnostic residual-heatmap plot cadence. Absent -> follow the resample
    # cadence; 0 -> never plot. The old `... or resample_every` turned an
    # explicit 0 back into the resample cadence, so the documented way to
    # disable plotting silently enabled it — and under Adam/SOAP, which
    # resample EVERY epoch, that meant a matplotlib savefig per step (measured
    # at ~450 ms/step wall against ~9 ms/step of actual GPU work).
    _pse = cfg.get('sampling', {}).get('plot_samples_every', None)
    plot_samples_every = resample_every if _pse is None else int(_pse)

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

    # ── Segment-scoped collar_data_ratio (fine-tune only) ──────────────────
    # The samplers read collar_data_ratio from the GLOBAL sampling cfg, so a
    # plan-level value would also reroute PHASE 3 onto collar-focused draws
    # (and off the proven direct per-expert path). A flat `collar_data_ratio`
    # inside the fine_tune block applies to THIS segment only: written into
    # cfg['sampling'] here, restored at segment end. The annealing block
    # below reads its start value AFTER this override, so the two compose
    # (start = the fine-tune-local ratio).
    _cdr_override = (segment_cfg.get('collar_data_ratio')
                     if segment_name == 'fine_tune' else None)
    _cdr_saved = None
    if _cdr_override is not None:
        _samp0 = cfg.setdefault('sampling', {})
        _cdr_saved = _samp0.get('collar_data_ratio', 0.0)
        _samp0['collar_data_ratio'] = float(_cdr_override)
        logger.info(f"  [Sampling] fine_tune: segment-local collar_data_ratio"
                    f" = {float(_cdr_override)} (global stays {_cdr_saved})")

    # ── Collar-data annealing (fine-tune only) ─────────────────────────────
    # fine_tune.collar_data_annealing: null = off; a number in [0, 1] = the
    # TARGET collar_data_ratio, reached linearly at the END of the segment's
    # epoch budget. The ramp starts from the configured
    # sampling.collar_data_ratio (which the segment's initial draw uses
    # as-is) and the current value is applied at each resample by writing it
    # into cfg['sampling'] just before the redraw (the samplers read it from
    # there); the original value is restored after the segment. Inert when
    # no redraw can fire inside the budget (resampling off / cadence too
    # large), per design.
    _cda_target = (segment_cfg.get('collar_data_annealing')
                   if segment_name == 'fine_tune' else None)
    _cda_start = None
    _cda_saved = None
    if _cda_target is not None:
        _cda_target = float(_cda_target)
        if not 0.0 <= _cda_target <= 1.0:
            raise ValueError(
                f"fine_tune.collar_data_annealing must be null or a number "
                f"in [0, 1], got {_cda_target}")
        _samp = cfg.get('sampling', {})
        _cda_saved = _samp.get('collar_data_ratio', 0.0)
        _cda_start = float(_cda_saved or 0.0)
        if resample_every <= 0 or resample_every >= epoch_budget:
            logger.info(
                f"  [CollarAnneal] disabled: no resample fires within the "
                f"{epoch_budget}-epoch budget (cadence {resample_every}); "
                f"collar_data_ratio stays {_cda_start}.")
            _cda_target = None
            _cda_saved = None
        else:
            logger.info(
                f"  [CollarAnneal] collar_data_ratio {_cda_start} -> "
                f"{_cda_target} linearly over {epoch_budget} epochs, applied "
                f"at each resample (every {resample_every}).")

    if optimizer_2_name is not None:
        switch_epoch = segment_start_epoch + seg_cfg['optimizer_switch_epoch']
    else:
        switch_epoch = total_epochs + 1  # never switch

    # ── Per-optimizer residual pool (ALWAYS on; global segments only) ──
    # Fixed rule, no config flag: Adam/SOAP train on `batch_size` residual
    # points per step, as ONE full batch that also carries ALL IC/BC points
    # — 1 step = 1 epoch, and the every-epoch Adam resample keeps the draw
    # fresh (when resampling is disabled the subsample simply stays fixed).
    # LBFGS/SSBroyden train on the full sampling.n_residual_train pool,
    # which is kept aside untouched here and swapped back in at the
    # optimizer switch. So: `batch_size` = points per Adam/SOAP step;
    # `n_residual_train` = the full-batch pool (and the on-disk dataset).
    # If batch_size >= the pool, the subsample is the whole pool and this
    # is a no-op. Split phase 3 has its own per-expert data path.
    _full_train_data = train_data
    if (optimizer_1_name in ('adam', 'soap')
            and getattr(ctx, '_split_context', None) is None):
        from utils.dataset_gen import subsample_residual_rows
        _n_full = int(_full_train_data['mask']['residual'].sum())
        if int(cfg['batch_size']) < _n_full:
            train_data = subsample_residual_rows(
                train_data, int(cfg['batch_size']), seed=int(cfg.get('seed', 0)))
        train_loader = _create_dataloader(train_data, 10 ** 9, shuffle=True)
        batches_per_epoch = 1
        ctx.train_data = train_data
        ctx.train_loader = train_loader
        logger.info(
            f"  [Sampling] {optimizer_1_name} phase: "
            f"{int(train_data['mask']['residual'].sum())} residual points "
            f"(min(batch_size, pool)) + all IC/BC as one full batch per "
            f"step; full pool ({_n_full} residual) reserved for a "
            f"full-batch optimizer phase.")

    total_steps_estimate = max(1, epoch_budget) * batches_per_epoch

    # Release the PREVIOUS segment's optimizer before allocating this one:
    # a full-batch 2nd-order optimizer holds a dense n x n Hk (2.05 GiB at
    # 18 leaves, float64). ctx.optimizer has no readers — it only keeps that
    # matrix alive across the boundary, adding a full extra copy to the new
    # segment's first-step peak (measured OOM: fine-tune after phase 3,
    # 18 leaves, A10G, 2026-08-14).
    if getattr(ctx, 'optimizer', None) is not None:
        ctx.optimizer = None
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("  [Segment] Released previous segment's optimizer "
                        "state and emptied the CUDA cache.")

    full_batch_opt1 = optimizer_1_name in ('lbfgs', 'ssbroyden')
    # Re-establish the default-device context at the segment boundary
    # (see _set_default_torch_device for why this is needed).
    _set_default_torch_device(device, full_batch=full_batch_opt1)
    if full_batch_opt1:
        optimizer, current_optimizer_name = _create_optimizer_by_name(
            optimizer_1_name, model, seg_cfg)
        from trainer.multi_optimizer import maybe_wrap_multi_optimizer
        optimizer = maybe_wrap_multi_optimizer(
            optimizer, current_optimizer_name, model, cfg, seg_cfg,
            ctx, loss_fn, segment_name)
        lr_scheduler = None
    else:
        optimizer, current_optimizer_name = _create_primary_optimizer(model, seg_cfg)
        lr_scheduler = _create_lr_scheduler(optimizer, seg_cfg, total_steps_estimate)

    step_count = 0
    # ── rel-L2 patience (evaluated every eval_every) ──
    # rel-L2 is computed on the fixed solver grid, so it is directly comparable
    # across evals (unlike the train loss, which jumps at each resample). An
    # eval "improves" only if it beats the best-so-far by >= patience_rel_delta;
    # patience_evals consecutive non-improving evals trip the plateau action
    # (optimizer_1 -> fast-forward to switch; optimizer_2 / no-switch -> stop).
    best_rel_l2_pat = float('inf')   # patience-local best (reset at the switch)
    evals_no_improve = 0
    # optimizer_1 is watched from the segment start; reset to switch_epoch at the switch.
    patience_start_epoch = segment_start_epoch
    # ── freeze monitor (freeze_stop_epochs / refresh_2nd_order_optimizer) ──
    # A full-batch 2nd-order optimizer whose strong-Wolfe line search returns
    # t=0 poisons its own curvature state (s=y=0 -> NaN Hessian update is
    # silently skipped) and then repeats bit-identically forever. Exact float
    # equality of consecutive epoch losses detects it; checked every epoch,
    # far more frequently than the rel-L2 patience (eval_every-based).
    # freeze_stop_epochs: N>0 -> STOP the segment after N bit-identical
    # epochs (frozen epochs are pure waste and ~2x slower: 25 rejected
    # line-search evals per step; measured 2026-08-11-12).
    # refresh_2nd_order_optimizer (retired 2026-08-11: restarts insta-
    # refreeze at the wall) -> rebuild the optimizer instead; freeze-stop
    # takes precedence when both are set.
    _refresh_2nd = bool(cfg.get('refresh_2nd_order_optimizer', False))
    _freeze_stop_n = int(cfg.get('freeze_stop_epochs', 0) or 0)
    _refresh_hist = []
    _nan_detected = False
    _stopped_early = False
    _stop_reason = 'budget'
    _lra_updated_epoch = -1
    _native_fallback_logged = False  # log the native-grid fallback once per segment
    # RAD candidate pool, rebuilt every sampling.resample_every_epochs when
    # adaptive sampling is on. Segment-local: each segment re-targets from
    # scratch rather than inheriting the previous segment's residual field.
    _rad_pool = None
    _rad_pool_epoch = -1
    _rad_pool_logged = None

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

    # ── Per-expert (per Ω_j) region rel-L2 tracking ──
    # Leaf tiles are fixed within a segment; root/base-only segments have
    # no leaves (num_experts == 0 or base still in the leaf set) and are
    # skipped. Collected at every eval; rendered at segment end.
    _track_regions = (
        is_adaptive and problem_cfg['spatial_dim'] == 1
        and hasattr(model, 'leaf_indices') and hasattr(model, 'regions')
        and getattr(model, 'num_experts', 0) > 0
        and -1 not in model.leaf_indices)
    _leaf_track = (sorted(i for i in model.leaf_indices if i >= 0)
                   if _track_regions else [])
    _bounds_lo = [list(model.regions[i].bounds_lower) for i in _leaf_track]
    _bounds_up = [list(model.regions[i].bounds_upper) for i in _leaf_track]
    _seg_err_start = None   # (err_grid, epoch) at the segment's first eval
    _seg_gt_ref = None      # (gt_sq_grid, x_grid, t_grid)

    # Bounded torch.profiler window for this segment. Disabled by default and
    # fail-safe: any profiler error logs once and training continues.
    from trainer.profiling import make_segment_profiler
    _profiler = make_segment_profiler(cfg, segment_name, segment_start_epoch,
                                      run_dir)

    # ── Initial collar seeding ──
    # When collar-focused sampling is active (sampling.collar_data_ratio > 0),
    # draw the FIRST collocation with collar focus too, not only at resamples.
    # Without this the segment spends its first resample interval on the plain
    # uniform set (and, if resampling is disabled, the WHOLE segment). Plain
    # path only — the split path builds its own collar-aware data upstream.
    _seed_collar_ratio = (cfg.get('sampling', {}) or {}).get(
        'collar_data_ratio', 0.0) or 0.0
    if (_seed_collar_ratio > 0
            and getattr(ctx, '_split_context', None) is None
            and hasattr(model, 'regions') and hasattr(model, 'leaf_indices')):
        _seed_leaf_idx = [i for i in sorted(model.leaf_indices) if i >= 0]
        if len(_seed_leaf_idx) >= 2:
            from utils.dataset_gen import build_collar_info, collar_deltas_for
            _seed_d_lo, _seed_d_hi = collar_deltas_for(model, _seed_leaf_idx)
            _seed_collar_info = build_collar_info(
                [model.regions[i] for i in _seed_leaf_idx],
                getattr(model, 'sigma_fraction', 0.2),
                plot=False,
                margin=(cfg.get('sampling', {}) or {}).get(
                    'collar_margin', 1.0) or 1.0,
                delta_lo=_seed_d_lo, delta_hi=_seed_d_hi,
            )
            train_data = resample_residual_inplace(
                train_data, cfg, device,
                resample_seed=base_seed + segment_start_epoch,
                cached_residuals=[], run_dir=run_dir,
                epoch=segment_start_epoch, causal_state=None,
                collar_info=_seed_collar_info,
            )
            ctx.train_data = train_data
            logger.info(f"  [Segment:{segment_name}] seeded initial collar "
                        f"collocation (collar_data_ratio={_seed_collar_ratio})")

    epoch = segment_start_epoch
    while epoch < total_epochs:
        epoch += 1
        ctx.epoch = epoch  # keep ctx in sync for the orchestrator's emergency save
        timer.start_epoch(epoch, num_experts=model.num_experts if (is_adaptive and hasattr(model, 'num_experts')) else 0)

        causal_state = getattr(loss_fn, 'causal_state', None)
        _problem_spatial_dim = problem_cfg['spatial_dim']
        # Split loss fn owns its own residual cache (the model-level cache is
        # not populated in the split path).
        _split_loss_fn = loss_fn if hasattr(loss_fn, '_residual_cache') else None

        # Resample training data periodically (in-memory, no disk I/O).
        # The cadence is optimizer-dependent. Adam/SOAP (stochastic mini-batch)
        # resample EVERY epoch: refreshing the collocation each step avoids
        # overfitting the sampled set. The full-batch quasi-Newton optimizers
        # (LBFGS, SSBroyden) warm-start their curvature state across steps, so
        # they refresh only every `resample_every` epochs — as in the
        # SSBroyden-for-PINNs literature: Urbán et al. (JCP 2025) refresh the
        # sampled points every ~500 iterations and warm-start across the
        # refresh, and the Kiyani et al. 2025 official code carries hess_inv
        # across RAD resamples. `resample_every <= 0` disables resampling for
        # every optimizer. This applies in all segments (initial train, the
        # phase-3 split, and fine-tune); the optimizer_1 -> optimizer_2 switch
        # flips the cadence mid-segment along with current_optimizer_name.
        _effective_resample_every = (
            1 if current_optimizer_name in ('Adam', 'SOAP') else resample_every)
        # Phase 3 static draw: no periodic redraw under the full-batch
        # optimizers (see the segment-setup comment). 0 disables the redraw
        # below without touching the Adam/SOAP every-epoch branch above.
        if _phase3_static and current_optimizer_name not in ('Adam', 'SOAP'):
            _effective_resample_every = 0
        _resampled_this_epoch = False
        _split_ctx = getattr(ctx, '_split_context', None)
        if (resample_every > 0 and _effective_resample_every > 0
                and epoch > 1 and (epoch - 1) % _effective_resample_every == 0):
            resample_seed = base_seed + epoch
            _resampled_this_epoch = True
            # Collar sampling (phase 3 + fine-tune only, i.e. whenever leaf
            # experts exist): draw sampling.collar_data_ratio of the residual
            # budget uniformly from the >= 2-overlapping-supports region.
            _collar_info = None
            # Collar-data annealing: write the current ramp value into the
            # sampling cfg so this block AND the samplers downstream
            # (sample_residual_points reads cfg) see the same ratio.
            if _cda_target is not None:
                _t = min(1.0, max(0.0, (epoch - segment_start_epoch)
                                  / float(epoch_budget)))
                cfg['sampling']['collar_data_ratio'] = (
                    _cda_start + (_cda_target - _cda_start) * _t)
            _collar_ratio = (cfg.get('sampling', {}) or {}).get(
                'collar_data_ratio', 0.0) or 0.0
            if (_collar_ratio > 0 and hasattr(model, 'regions')
                    and hasattr(model, 'leaf_indices')):
                _leaf_idx = [i for i in sorted(model.leaf_indices) if i >= 0]
                if len(_leaf_idx) >= 2:
                    from utils.dataset_gen import (
                        build_collar_info, collar_deltas_for)
                    _d_lo, _d_hi = collar_deltas_for(model, _leaf_idx)
                    _collar_info = build_collar_info(
                        [model.regions[i] for i in _leaf_idx],
                        getattr(model, 'sigma_fraction', 0.2),
                        plot=(plot_samples_every > 0
                              and (epoch - 1) % plot_samples_every == 0
                              and _problem_spatial_dim == 1),
                        margin=(cfg.get('sampling', {}) or {}).get(
                            'collar_margin', 1.0) or 1.0,
                        delta_lo=_d_lo, delta_hi=_d_hi,
                    )
            if _split_ctx is not None:
                # Adaptive sampling: hand last epoch's per-expert residual
                # cache to the sampler (uniform/adaptive mix, same as the
                # non-split path). Empty/absent cache → pure uniform draw.
                _split_cached = (list(_split_loss_fn._residual_cache)
                                 if (adaptive_sampling_enabled
                                     and _split_loss_fn is not None
                                     and _split_loss_fn._residual_cache)
                                 else None)
                _mode = 'adaptive' if _split_cached else 'uniform'
                # Fires on every resample, i.e. every epoch under Adam/SOAP.
                # Report it on the same cadence as the sampling plots so a
                # 100k-epoch segment does not emit 100k identical lines.
                if (epoch <= segment_start_epoch + 1
                        or (plot_samples_every > 0
                            and (epoch - 1) % plot_samples_every == 0)):
                    logger.info(f"  [Resample-Split] Redrawing residual "
                                f"interiors ({_mode}) at epoch {epoch}")
                # Static faces + interface targets are cached for the segment;
                # only the residual collocation points are redrawn.
                train_data = build_subdomain_data(
                    _split_ctx['model_snapshot'], _split_ctx['new_expert_indices'],
                    _split_ctx['regions'], cfg, device, seed=resample_seed,
                    interface_model=_split_ctx.get('interface_model'),
                    static=_split_ctx.get('static'),
                    cached_residuals=_split_cached,
                    run_dir=run_dir, epoch=epoch,
                    collar_info=_collar_info,
                )
                if _split_cached is not None:
                    _split_loss_fn._residual_cache.clear()
                ctx.train_data = train_data
                _set_default_torch_device(device, full_batch=False)
                train_loader = _create_split_dataloader(
                    train_data, cfg['batch_size'], shuffle=True)
                ctx.train_loader = train_loader
                # Restore the full-batch default-device context when a
                # quasi-Newton optimizer is active (the loader rebuild above
                # needs the CPU default; SSBroyden/LBFGS state allocation
                # needs the training device — see _set_default_torch_device).
                _set_default_torch_device(
                    device,
                    full_batch=current_optimizer_name in ('LBFGS', 'SSBroyden'))
                metrics['resample_events'].append({
                    'epoch': epoch, 'action': 'split_resampled',
                    'optimizer': current_optimizer_name,
                })
                # Diagnostic residual heatmap: drain the per-expert residual cache
                # collected this epoch (union of all experts' local residual points).
                # With adaptive sampling on, the sampler already saved the richer
                # heatmap (cache + sampled points) above — skip the plain one.
                if (not adaptive_sampling_enabled
                        and _split_loss_fn is not None
                        and _split_loss_fn._residual_cache
                        and _problem_spatial_dim == 1
                        and plot_samples_every > 0
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
                # ── RAD re-target (adaptive sampling only) ──────────────────
                # Two cadences, deliberately different:
                #   * the UNIFORM half is redrawn every epoch (jaxpi/SOAP
                #     behaviour -- stops the net memorising a frozen set),
                #   * the residual DISTRIBUTION the adaptive half draws from
                #     is rebuilt only every sampling.resample_every_epochs.
                # Re-targeting every epoch, as this did before, meant chasing a
                # one-step-old single-forward-pass residual estimate while
                # SOAP's preconditioner EMA tried to track a moving objective.
                # Wu et al. (CMAME 2023), the RAD reference, re-target every
                # 1000 iterations from a FRESH uniform pool; drawing from last
                # epoch's own points instead makes it a bootstrap filter.
                if adaptive_sampling_enabled:
                    _pool_every = max(1, int(resample_every))
                    _need_pool = (
                        _rad_pool is None
                        or (epoch - _rad_pool_epoch) >= _pool_every)
                    if _need_pool:
                        _new_pool = _build_rad_pool(
                            model, loss_fn, cfg, device, train_data)
                        if _new_pool is not None:
                            _rad_pool, _rad_pool_epoch = _new_pool, epoch
                            if _rad_pool_logged != _pool_every:
                                _rad_pool_logged = _pool_every
                                logger.info(
                                    f"  [RAD] re-targeting every {_pool_every} "
                                    f"epochs from a fresh pool of "
                                    f"{_rad_pool[0][0].shape[0]} uniform points")
                    if _rad_pool is not None:
                        cached_residuals = _rad_pool
                if (not adaptive_sampling_enabled and cached_residuals
                        and _problem_spatial_dim == 1
                        and plot_samples_every > 0
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
                    collar_info=_collar_info,
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

        # ── Arm residual caching for THIS epoch (consumed at the NEXT
        # resample). Must run AFTER the resample block: arming clears the
        # cache, and with _effective_resample_every == 1 (Adam/SOAP) the arming
        # epoch and the consuming epoch coincide — arming first would wipe the
        # cache the resample is about to use. Uses the same optimizer-dependent
        # cadence as the resample gate above so caching tracks it exactly.
        will_cache_for_resample = (
            adaptive_sampling_enabled
            and resample_every > 0 and _effective_resample_every > 0
            and epoch > 0 and epoch % _effective_resample_every == 0
        )
        # Cache residuals for the diagnostic heatmap even when adaptive sampling is
        # off. Cadence is controlled by sampling.plot_samples_every (independent of
        # the resample cadence; defaults to it).
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

        # Same for the split-loss cache: diagnostic heatmaps and — when
        # adaptive sampling is on — the next split resample's adaptive redraw.
        _will_cache_split = (
            _split_loss_fn is not None
            and plot_samples_every > 0
            and epoch > 0 and epoch % plot_samples_every == 0
            and _problem_spatial_dim == 1
        )
        _will_cache_split_resample = (
            _split_loss_fn is not None
            and adaptive_sampling_enabled
            and resample_every > 0 and _effective_resample_every > 0
            and epoch > 0 and epoch % _effective_resample_every == 0
        )
        if _split_loss_fn is not None:
            _cache_split_now = _will_cache_split or _will_cache_split_resample
            _split_loss_fn._cache_residuals = _cache_split_now
            if _cache_split_now:
                _split_loss_fn._residual_cache.clear()
            if _will_cache_split_resample and not hasattr(
                    _split_loss_fn, '_adaptive_sampling_activated'):
                _split_loss_fn._adaptive_sampling_activated = True
                logger.info(f"  [Adaptive Sampling] Split residual caching "
                            f"active from epoch {epoch}")

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
            # multi_optimizers (phase-3 only): per-expert-group SSBroyden —
            # each group gets its own Hessian/line-search/stopping instead
            # of one shared set across experts at incompatible scales.
            from trainer.multi_optimizer import maybe_wrap_multi_optimizer
            optimizer = maybe_wrap_multi_optimizer(
                optimizer, current_optimizer_name, model, cfg, seg_cfg,
                ctx, loss_fn, segment_name)
            lr_scheduler = None  # optimizer_2 uses its own LR / line search
            # Reset patience at the switch; optimizer_2 gets a fresh grace window.
            best_rel_l2_pat = float('inf')
            evals_no_improve = 0
            patience_start_epoch = switch_epoch
            _refresh_hist.clear()
            # LRA freezes at the switch: the quasi-Newton phase needs a fixed
            # objective (shifting weights corrupt curvature pairs and the
            # strong-Wolfe comparisons). The LAST adapted weights keep
            # weighting the loss from here on.
            if lra_weights is not None:
                _w_str = ', '.join(f'{k}={v:.4e}'
                                   for k, v in lra_weights.weights.items())
                logger.info(f"  [LRA] FROZEN for the {current_optimizer_name} "
                            f"phase at: {_w_str}")
            # Causal weighting is an Adam-phase curriculum for the same
            # reason: its weights are a function of the current residuals, so
            # they shift under the line search. Disable at the switch — a
            # completed ladder has weights ~1 anyway; an incomplete one hands
            # the quasi-Newton phase the clean plain-MSE residual.
            _cs = getattr(loss_fn, 'causal_state', None)
            if _cs is not None and _cs.get('enabled', True):
                _cs['enabled'] = False
                logger.info(
                    f"  [Causal] DISABLED for the {current_optimizer_name} "
                    f"phase (ladder stage {_cs['schedule_idx'] + 1}/"
                    f"{len(_cs['schedule'])}, tol={_cs['tol']:.2f}, "
                    f"min_w={_cs['min_weight']:.4f}) — residual reverts to "
                    f"plain MSE.")
            metrics['optimizer_events'].append({
                'epoch': epoch,
                'from': _prev_opt,
                'to': current_optimizer_name,
            })
            # Per-optimizer pool rule: the full-batch optimizer takes over —
            # swap the untouched full residual pool back in (static from here:
            # the Adam-only every-epoch resample no longer applies).
            # GLOBAL segments only: split (phase-3) segments reassign
            # train_data on every per-expert redraw, so the identity check
            # alone would wrongly fire here and clobber the split data with
            # a plain loader (measured: KeyError 'mask' at the switch,
            # 2026-08-17 phase-3 ablation run).
            if (getattr(ctx, '_split_context', None) is None
                    and train_data is not _full_train_data
                    and current_optimizer_name in ('LBFGS', 'SSBroyden')):
                _set_default_torch_device(device, full_batch=False)
                train_data = _full_train_data
                train_loader = _create_dataloader(train_data, 10 ** 9,
                                                  shuffle=True)
                _set_default_torch_device(device, full_batch=True)
                ctx.train_data = train_data
                ctx.train_loader = train_loader
                logger.info(
                    f"  [Sampling] {current_optimizer_name} phase: switched "
                    f"to the full pool "
                    f"({int(train_data['mask']['residual'].sum())} residual "
                    f"points, static).")
        
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

        # Freeze monitor (see the init comment above): stop the segment on a
        # deterministic line-search freeze (freeze_stop_epochs), or — retired
        # fallback — rebuild the optimizer (refresh_2nd_order_optimizer).
        # The multi-opt wrapper monitors its own groups for refresh, but the
        # stop applies to it too: a bit-identical TOTAL means every group is
        # frozen.
        if ((_freeze_stop_n > 0 or _refresh_2nd)
                and current_optimizer_name in ('LBFGS', 'SSBroyden')):
            _refresh_hist.append(train_loss)
            _keep = max(_freeze_stop_n, 3)
            if len(_refresh_hist) > _keep:
                _refresh_hist.pop(0)
            if (_freeze_stop_n > 0 and len(_refresh_hist) >= _freeze_stop_n
                    and len(set(_refresh_hist[-_freeze_stop_n:])) == 1):
                logger.info(
                    f"\n  [FreezeStop] train loss bit-identical for "
                    f"{_freeze_stop_n} epochs ({train_loss:.6e}) — "
                    f"deterministic line-search freeze; stopping segment at "
                    f"epoch {epoch} (rel_l2 best={best_rel_l2_pat:.6e}).")
                metrics['plateau_events'].append({
                    'epoch': epoch,
                    'action': 'freeze_stop',
                    'metric': 'train_loss',
                })
                _stopped_early = True
                _stop_reason = 'freeze_stop'
                break
            if (_refresh_2nd and _freeze_stop_n == 0
                    and not hasattr(optimizer, 'opts')
                    and len(_refresh_hist) >= 3
                    and len(set(_refresh_hist[-3:])) == 1):
                logger.info(
                    f"  [Refresh] train loss bit-identical for 3 epochs "
                    f"({train_loss:.6e}) — rebuilding {current_optimizer_name} "
                    f"with fresh state at epoch {epoch}.")
                _set_default_torch_device(device, full_batch=True)
                optimizer, current_optimizer_name = _create_optimizer_by_name(
                    current_optimizer_name.lower(), model, seg_cfg)
                metrics['optimizer_events'].append(
                    {'epoch': epoch, 'refresh': current_optimizer_name})
                _refresh_hist.clear()

        # LRA: update adaptive loss weights periodically — first-order phase
        # ONLY. Quasi-Newton phases (LBFGS/SSBroyden) train under the frozen
        # last-adapted weights (see the switch block above).
        if (lra_weights is not None and epoch > 0
                and epoch % lra_weights.update_every == 0
                and current_optimizer_name not in ('LBFGS', 'SSBroyden')):
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
            _native = compute_native_grid_metrics(model, cfg, device,
                                                  return_grids=_track_regions)
            timer.stop('eval.native_grid')
            if _native is not None:
                rel_l2 = _native['rel_l2']
                inf_norm = _native['inf_norm']
                # Per-expert region rel-L2 from the same eval pass
                if _track_regions and 'err_grid' in _native:
                    _region_rels = per_region_rel_l2(
                        _native['err_grid'], _native['gt_sq_grid'],
                        _native['x_grid'], _native['t_grid'],
                        _bounds_lo, _bounds_up)
                    _pe = metrics.setdefault('per_expert_rel_l2', {}).setdefault(
                        segment_name, {
                            'epochs': [],
                            'experts': {str(i): [] for i in _leaf_track},
                            'leaf_indices': list(_leaf_track),
                            'bounds_lower': _bounds_lo,
                            'bounds_upper': _bounds_up,
                        })
                    _pe['epochs'].append(epoch)
                    for _i, _r in zip(_leaf_track, _region_rels):
                        _pe['experts'][str(_i)].append(_r)
                    if _seg_err_start is None:
                        _seg_err_start = (_native['err_grid'], epoch)
                        _seg_gt_ref = (_native['gt_sq_grid'],
                                       _native['x_grid'], _native['t_grid'])
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

            # rel-L2 patience: an eval "improves" only if it beats the
            # patience-local best by >= patience_rel_delta. The plateau ACTION
            # is taken at the bottom of the loop (so this eval's logging still
            # runs). Only real (finite) rel-L2 values participate.
            if patience_evals > 0 and math.isfinite(rel_l2):
                if rel_l2 < best_rel_l2_pat * (1.0 - patience_rel_delta):
                    best_rel_l2_pat = rel_l2
                    evals_no_improve = 0
                else:
                    evals_no_improve += 1

            # ── Term-wise loss components (from the same eval pass) ──
            metrics['loss_components']['epochs'].append(epoch)
            for term in ['residual', 'ic', 'bc', 'bc_dx', 'bc_dxx', 'bc_dxxx',
                         'l2sp', 'l2sp_drift']:
                metrics['loss_components'].setdefault(term, []).append(
                    comp_means.get(term, 0.0))
            _comp_str = ', '.join(f'{k}={v:.6e}' for k, v in comp_means.items())
            logger.info(f"  [LossTerms] {_comp_str}")
            metrics['loss_components_history'].append({
                'epoch': epoch,
                **comp_means,
            })

            # ── Per-expert loss terms on the COMPOSED u_θ, filtered to Ω_j ──
            # Per-sample losses from the problem loss (for_tree_spawning path),
            # averaged over the probe points of each kind inside each leaf
            # tile. Rendered in the per-expert region report at segment end.
            if _track_regions and _probe is not None:
                try:
                    _ps = loss_fn(model, _probe, for_tree_spawning=True,
                                  update_causal_state=False)
                    _px = _probe['x'][:, 0]
                    _pt = _probe['t'][:, 0]
                    _pmask = _probe['mask']
                    _kind_of = {'residual': _pmask['residual'],
                                'ic': _pmask['IC'], 'bc': _pmask['BC']}
                    _lw = problem_cfg['loss_weights']
                    _pel = metrics.setdefault(
                        'per_expert_loss_terms', {}).setdefault(
                        segment_name, {
                            'epochs': [],
                            'experts': {str(i): {'residual': [], 'ic': [],
                                                 'bc': [], 'total': []}
                                        for i in _leaf_track},
                        })
                    _pel['epochs'].append(epoch)
                    for _i, _lo, _hi in zip(_leaf_track, _bounds_lo, _bounds_up):
                        _in_box = ((_px >= _lo[0]) & (_px <= _hi[0])
                                   & (_pt >= _lo[1]) & (_pt <= _hi[1]))
                        _tot = 0.0
                        _any = False
                        for _term in ('residual', 'ic', 'bc'):
                            _arr = _ps.get(_term)
                            _m = _in_box & _kind_of[_term]
                            if _arr is not None and _m.any():
                                _v = float(_arr[_m].detach().mean().item())
                                _tot += _lw.get(_term, 1.0) * _v
                                _any = True
                            else:
                                _v = float('nan')
                            _pel['experts'][str(_i)][_term].append(_v)
                        _pel['experts'][str(_i)]['total'].append(
                            _tot if _any else float('nan'))
                except Exception as _pel_err:
                    logger.info(f"  [PerExpertTerms] failed: {_pel_err}")

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

        # Bounded profiling window (no-op unless profiling.enabled). Placed
        # before the plateau block so a fast-forward `continue` cannot skip it.
        if _profiler is not None:
            _profiler.step(epoch)

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

        # ── Incremental metrics dump ──────────────────────────────────────
        # metrics.json used to be written only by finalize(), so anything that
        # killed the process discarded the whole run's history — training
        # curves, per-expert rel-L2, LR/grad-norm traces. A SIGKILL (the OOM
        # killer) cannot be caught, so no amount of exception handling helps;
        # the only fix is to have the file already on disk. Rewriting it at
        # every eval caps the loss at one eval interval.
        #
        # Written to a temp file and renamed, so a kill mid-write leaves the
        # previous complete metrics.json rather than a truncated one. Compact
        # (no indent) because this runs ~200x per segment; finalize still
        # writes the pretty version at the end.
        if should_evaluate and run_dir is not None:
            try:
                _mp = run_dir / "metrics.json"
                _mp_tmp = run_dir / "metrics.json.tmp"
                with open(_mp_tmp, 'w') as _f:
                    json.dump(metrics, _f, cls=_NumpySafeEncoder)
                _mp_tmp.replace(_mp)
            except Exception as _e:                             # noqa: BLE001
                logger.warning(f"  [Metrics] incremental write failed "
                               f"({type(_e).__name__}: {_e}); the end-of-run "
                               f"write is unaffected.")

        # Save checkpoint periodically (only when we have grid metrics)
        if epoch % save_every == 0 and rel_l2 is not None:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            _save_checkpoint(checkpoint_path, model, optimizer, current_optimizer_name, epoch,
                           train_loss, rel_l2, cfg, metrics)
            logger.info(f"  Checkpoint saved: {checkpoint_path}")

        # Save the SEGMENT's best model on the solver-grid rel-L2 (checked at
        # eval epochs). Reconciled against the end-of-segment weights below.
        if (should_evaluate and best_checkpoint_path is not None
                and rel_l2 is not None
                and math.isfinite(rel_l2) and rel_l2 < best_rel_l2):
            best_rel_l2 = rel_l2
            _save_checkpoint(best_checkpoint_path, model, optimizer, current_optimizer_name, epoch,
                           train_loss, rel_l2, cfg, metrics)
            _best_epoch = epoch

        # rel-L2 plateau patience (counter updated in the eval block above).
        # Acted on only at eval epochs, after this epoch's logging. Active for
        # BOTH optimizers: on an optimizer_1 plateau we fast-forward to the
        # switch epoch (so the existing switch handler fires and optimizer_2
        # keeps its full budget) rather than stopping; an optimizer_2 (or
        # no-switch) plateau stops the segment. seg_min_epochs is a grace
        # period measured from the start of the active optimizer window.
        if (patience_evals > 0 and should_evaluate
                and epoch - patience_start_epoch >= seg_min_epochs
                and evals_no_improve >= patience_evals):
            _in_optimizer_1 = (optimizer_2_name is not None and epoch < switch_epoch)
            if _in_optimizer_1 and switch_epoch < total_epochs:
                logger.info(f"\n  [Patience] optimizer_1 plateau: {evals_no_improve} "
                      f"consecutive evals without >{patience_rel_delta:.1%} rel-L2 "
                      f"improvement (best={best_rel_l2_pat:.6e}) at epoch {epoch}; "
                      f"fast-forwarding to switch epoch {switch_epoch}.")
                metrics['plateau_events'].append({
                    'epoch': epoch,
                    'action': 'optimizer_1_fast_forward',
                    'switch_epoch': switch_epoch,
                    'metric': 'rel_l2',
                })
                epoch = switch_epoch - 1
                ctx.epoch = epoch
                best_rel_l2_pat = float('inf')
                evals_no_improve = 0
                continue
            else:
                logger.info(f"\n  [EarlyStop] {evals_no_improve} consecutive evals "
                      f"without >{patience_rel_delta:.1%} rel-L2 improvement "
                      f"(best={best_rel_l2_pat:.6e}). Stopping segment at epoch "
                      f"{epoch} (rel_l2={rel_l2:.6e}).")
                metrics['plateau_events'].append({
                    'epoch': epoch,
                    'action': 'early_stop',
                    'metric': 'rel_l2',
                })
                _stopped_early = True
                _stop_reason = 'early_stop'
                break

    # Collar-data annealing / segment-local ratio: restore what we found (the
    # sampling cfg dict is shared with the global config). Order matters:
    # the annealing's saved value is the segment-local one, so the _cdr
    # restore must come LAST to reinstate the true global.
    if _cda_saved is not None:
        cfg['sampling']['collar_data_ratio'] = _cda_saved
    if _cdr_saved is not None:
        cfg['sampling']['collar_data_ratio'] = _cdr_saved

    # ── Reconcile the segment's best with the end-of-segment weights ──
    # After this, the in-memory model == best_model_<segment>.pt == the
    # segment's best; the next segment (tree build, interface snapshot,
    # fine-tune) continues from it.
    _seg_grids_final = None
    _seg_grids_best = None
    if _nan_detected:
        _stop_reason = 'nan'
    else:
        # End-of-segment error field (the 'final' weights, captured before
        # reconciliation may replace them with the best checkpoint).
        if _track_regions:
            _seg_grids_final = compute_native_grid_metrics(
                model, cfg, device, return_grids=True)
        rel_l2, inf_norm = _reconcile_segment_best(
            model, optimizer, current_optimizer_name, segment_name, epoch,
            train_loss, best_rel_l2, _best_epoch, best_checkpoint_path,
            cfg, metrics, device, ctx=ctx)
        best_rel_l2 = min(best_rel_l2, rel_l2) if math.isfinite(rel_l2) else best_rel_l2
        # Error field of the kept (best) model after reconciliation.
        if _track_regions:
            _seg_grids_best = compute_native_grid_metrics(
                model, cfg, device, return_grids=True)

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

    if not _nan_detected:
        # 'best' = the reconciled model that continues in the pipeline;
        # label it with the epoch of the kept weights (best checkpoint's
        # epoch when it was restored, else the segment-end epoch).
        _rec = metrics.get('segment_reconcile_events', [])
        _ev = (_rec[-1] if _rec and _rec[-1].get('segment') == segment_name
               else None)
        _best_label = (_ev['best_epoch']
                       if _ev and _ev.get('kept') == 'best'
                       and _ev.get('best_epoch') is not None else epoch)
        _save_segment_pred_plot(ctx, segment_name, tag='best',
                                epoch_label=_best_label)

        # ── Per-expert region report (curve + start/best/final heatmaps) ──
        if (_track_regions and _seg_grids_best is not None
                and _seg_err_start is not None and _seg_gt_ref is not None):
            _grids = {
                'start': _seg_err_start,
                'best': (_seg_grids_best['err_grid'], _best_label),
            }
            # 'final' only when the end weights lost to an earlier best
            if (_ev is not None and _ev.get('kept') == 'best'
                    and _seg_grids_final is not None):
                _grids['final'] = (_seg_grids_final['err_grid'], epoch)
            try:
                _tp_dir = run_dir / 'training_plots'
                _tp_dir.mkdir(exist_ok=True)
                _pe = metrics['per_expert_rel_l2'][segment_name]

                def _np(a):
                    return a.cpu().numpy() if isinstance(a, torch.Tensor) else a

                _pel = metrics.get('per_expert_loss_terms', {}).get(
                    segment_name, {})

                # Frozen-series guard: a per-expert term that never changes
                # across >=4 evals despite the model training is a probe/
                # metrics bug (e.g. the eval-batch fallback in split_loss_fn
                # not reflecting live weights), not a real plateau — flag it
                # so a stale panel doesn't get read as a training signal.
                for _fi, _fterms in _pel.get('experts', {}).items():
                    for _fterm, _fvals in _fterms.items():
                        if len(_fvals) >= 4 and len(set(_fvals)) == 1 and _fvals[0] not in (0.0, float('nan')):
                            logger.warning(
                                f"  [PerExpertTerms] expert={_fi} term='{_fterm}' "
                                f"identical ({_fvals[0]:.6e}) across all "
                                f"{len(_fvals)} evals of segment '{segment_name}' "
                                f"— likely a stale/non-live probe, not a real "
                                f"plateau; do not read this term off the plot.")

                plot_per_expert_region_report(
                    epochs=_pe['epochs'],
                    series=_pe['experts'],
                    leaf_indices=_leaf_track,
                    bounds_lower=_bounds_lo,
                    bounds_upper=_bounds_up,
                    grids=_grids,
                    gt_sq_grid=_seg_gt_ref[0],
                    x_grid=_seg_gt_ref[1],
                    t_grid=_seg_gt_ref[2],
                    out_path=_tp_dir / f'per_expert_rel_l2_{segment_name}.png',
                    segment_name=segment_name,
                    gt_grid=_np(ctx.gt_grid),
                    gt_x=_np(ctx.gt_x),
                    gt_t=_np(ctx.gt_t),
                    loss_series=_pel.get('experts'),
                    loss_epochs=_pel.get('epochs'),
                )
                logger.info(f"  [Segment:{segment_name}] saved "
                            f"training_plots/per_expert_rel_l2_{segment_name}.png")
            except Exception as _pe_err:
                logger.warning(f"  [Segment:{segment_name}] per-expert region "
                               f"report failed: {_pe_err}")
    # Report a window the segment ended in the middle of (early stop, NaN,
    # or a budget shorter than start_epoch + warmup + active).
    if _profiler is not None:
        _profiler.close()

    _final_tl = train_loss if train_loss is not None else float('nan')
    _final_rl2 = rel_l2 if rel_l2 is not None else float('nan')
    _oom_stopped = getattr(ctx, 'oom_stopped', False)

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


def _reconcile_segment_best(model, optimizer, optimizer_name: str,
                            segment_name: str, epoch: int, train_loss: float,
                            best_rel_l2: float, best_epoch,
                            best_checkpoint_path, cfg: Dict, metrics: Dict,
                            device, ctx=None) -> tuple:
    """End-of-segment reconciliation: keep the segment's best model.

    Recomputes the in-memory (end-of-segment) rel-L2 fresh on the solver
    grid (on early stop the last logged value can be up to eval_every-1
    epochs stale), then:

      * best checkpoint better  -> restore ``best_model_<segment>.pt`` into
        the model, so the next segment continues from it;
      * end-of-segment better   -> overwrite ``best_model_<segment>.pt``
        with the in-memory weights.

    Either way the invariant holds: in-memory model == the segment's best ==
    ``best_model_<segment>.pt`` (there is no separate final checkpoint).
    The comparison uses the model's CURRENT blending mode (hard indicators
    during split segments), i.e. like-with-like within the segment.

    Returns:
        (rel_l2, inf_norm) of the reconciled model on the solver grid.
    """
    final_metrics = compute_native_grid_metrics(model, cfg, device)
    final_rel = final_metrics['rel_l2'] if final_metrics else float('nan')

    restore_best = (
        best_checkpoint_path is not None
        and Path(best_checkpoint_path).exists()
        and math.isfinite(best_rel_l2)
        and (not math.isfinite(final_rel) or best_rel_l2 < final_rel)
    )

    if restore_best:
        # The end-of-segment weights are about to be discarded in favor of
        # the best checkpoint — save their prediction plot first so both
        # states are visible ('final' tag = the non-kept end weights).
        if ctx is not None:
            _save_segment_pred_plot(ctx, segment_name, tag='final',
                                    epoch_label=epoch)
        try:
            ckpt = torch.load(best_checkpoint_path, map_location=device,
                              weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            if hasattr(model, 'batched_models'):
                model.batched_models.sync_from_models(model.base_model, model.experts)
            chosen_metrics = compute_native_grid_metrics(model, cfg, device)
            logger.info(f"  [Segment:{segment_name}] restored best "
                        f"(epoch {best_epoch}, rel_l2={best_rel_l2:.6e}) over "
                        f"end-of-segment (rel_l2={final_rel:.6e})")
            chosen = 'best'
        except Exception as e:
            logger.warning(f"  [Segment:{segment_name}] best-model restore "
                           f"failed ({e}); keeping end-of-segment weights.")
            chosen_metrics = final_metrics
            chosen = 'final'
    else:
        chosen_metrics = final_metrics
        chosen = 'final'
        if best_checkpoint_path is not None:
            try:
                _save_checkpoint(best_checkpoint_path, model, optimizer,
                                 optimizer_name, epoch, train_loss, final_rel,
                                 cfg, metrics)
                logger.info(f"  [Segment:{segment_name}] end-of-segment is the "
                            f"segment best (rel_l2={final_rel:.6e}) — saved "
                            f"{Path(best_checkpoint_path).name}")
            except Exception as e:
                logger.warning(f"  [Segment:{segment_name}] best-model save "
                               f"failed: {e}")

    metrics.setdefault('segment_reconcile_events', []).append({
        'segment': segment_name,
        'end_epoch': epoch,
        'best_epoch': best_epoch,
        'best_rel_l2': best_rel_l2 if math.isfinite(best_rel_l2) else None,
        'final_rel_l2': final_rel if math.isfinite(final_rel) else None,
        'kept': chosen,
    })

    if chosen_metrics is not None:
        return chosen_metrics['rel_l2'], chosen_metrics['inf_norm']
    return float('nan'), float('nan')


def _save_segment_pred_plot(ctx: TrainingContext, segment_name: str,
                            tag: str = 'best', epoch_label=None) -> None:
    """Save ``pred_after_<segment>_<tag>_ep<N>.png`` (1D problems with GT).

    ``tag`` is ``'best'`` for the reconciled model that continues in the
    pipeline, or ``'final'`` for discarded end-of-segment weights (saved only
    when they differ from the best). ``epoch_label`` overrides the epoch in
    the filename (e.g. the best checkpoint's epoch); defaults to ctx.epoch.
    """
    if ctx.problem_cfg.get('spatial_dim', None) != 1:
        return
    out_dir = ctx.adaptive_plots_dir or ctx.run_dir
    if out_dir is None:
        return
    ep = epoch_label if epoch_label is not None else ctx.epoch
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_spawn_prediction_plot(
            model=ctx.model,
            domain_bounds=ctx.domain_bounds,
            gt_grid=ctx.gt_grid,
            grid_x=ctx.gt_x,
            grid_t=ctx.gt_t,
            # {relL2} placeholder is filled in by the renderer
            output_path=(out_dir / f"pred_after_{segment_name}_{tag}_ep{ep}"
                                   f"_relL2_{{relL2}}.png"),
            epoch=ctx.epoch,
            cfg=ctx.cfg,
        )
        logger.info(f"  [Segment:{segment_name}] saved "
                    f"pred_after_{segment_name}_{tag} plot (epoch {ep})")
    except Exception as _e:
        logger.info(f"  [Segment:{segment_name}] prediction plot failed: {_e}")
