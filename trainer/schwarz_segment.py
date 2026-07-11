"""Schwarz-scheduled phase 3: distill warm start + freeze/unfreeze sweeps.

Implements the FBPINN-as-Schwarz training schedule (Dolean/Heinlein/
Moseley, arXiv:2211.05560) on the AToE leaf composition:

1. DISTILL warm start — every leaf expert is fitted individually to the
   frozen root (u0) on its whole window support by plain MSE, so any
   frozen expert is a root-accurate boundary-data source for its active
   neighbors. Replaces the removed u0 interface-guide terms.
2. SCHWARZ BLOCKS — the leaf overlap graph (inflated boxes) is greedily
   colored; blocks activate one color at a time (all other experts
   frozen) and minimize the CLASSIC global PINN loss of the composition
   u_theta: PDE residual restricted to the active supports (one weight-1
   mean per active expert — per-region weighting for free, since same-
   color supports are disjoint) + the exact IC/BC terms on u_theta. In
   the collars, the frozen neighbors' window-weighted outputs act as
   boundary data baked into the composed residual; that data improves
   every sweep instead of being pinned at root accuracy.

Every block runs through the standard ``_train_segment`` (fresh
optimizer over the currently-trainable params each block — quasi-Newton
curvature history never leaks across a freeze/unfreeze swap), with
``reconcile_best=False`` so block-end weights carry forward.

The loss object is the (guide-free) split loss: its sum-of-per-group
means over ``expert_id`` tags is exactly one mean per active support
when fed Schwarz block data, and its ``ic_bc_batch`` carries the exact
IC/BC on the composition. One instance lives across all blocks so the
per-expert history/curves span the whole phase.
"""

import torch
from typing import Dict, List

from utils.logging_config import get_logger

logger = get_logger(__name__)

from trainer.plotting import plot_per_expert_curves
from trainer.training_context import TrainingContext, SegmentResult
from trainer.setup import _create_split_dataloader
from trainer.epoch_loop import _train_segment
from losses.split_loss import build_split_loss
from losses.distill_loss import build_distill_loss
from adaptive.subdomain_data import (
    build_distill_data, sample_schwarz_residuals, _domain_box,
)
from adaptive.indicators import inflated_bounds


def compute_expert_coloring(
    expert_indices: List[int],
    regions,
    sigma_fraction: float,
    g_lo: List[float],
    g_hi: List[float],
    mode: str = 'colored',
    tol: float = 1e-12,
) -> List[List[int]]:
    """Color the leaf overlap graph for the Schwarz schedule.

    Two experts are adjacent iff their window supports (inflated boxes)
    overlap with positive volume in EVERY dim — supports that merely
    touch are independent (windows vanish at the support edge). Greedy
    coloring in descending-degree order; same-color supports are
    pairwise disjoint, so a color can train in one block with one
    residual mean per expert and no shared points.

    ``mode='sequential'`` returns one color per expert (multiplicative
    Schwarz — one active expert per block).

    Returns a list of colors, each a sorted list of expert indices.
    """
    if mode == 'sequential':
        return [[e] for e in sorted(expert_indices)]

    boxes = {}
    for eidx in expert_indices:
        boxes[eidx] = inflated_bounds(
            regions[eidx], sigma_fraction, g_lo, g_hi)

    def _overlaps(a, b):
        (a_lo, a_hi), (b_lo, b_hi) = boxes[a], boxes[b]
        return all(
            b_lo[d] < a_hi[d] - tol and b_hi[d] > a_lo[d] + tol
            for d in range(len(a_lo))
        )

    adj = {e: set() for e in expert_indices}
    for i, a in enumerate(expert_indices):
        for b in expert_indices[i + 1:]:
            if _overlaps(a, b):
                adj[a].add(b)
                adj[b].add(a)

    # Greedy: highest degree first, smallest available color.
    order = sorted(expert_indices, key=lambda e: (-len(adj[e]), e))
    color_of = {}
    for e in order:
        taken = {color_of[n] for n in adj[e] if n in color_of}
        c = 0
        while c in taken:
            c += 1
        color_of[e] = c

    n_colors = max(color_of.values()) + 1
    colors = [sorted(e for e in expert_indices if color_of[e] == c)
              for c in range(n_colors)]

    # Sanity: same-color supports must be disjoint.
    for c, members in enumerate(colors):
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                assert not _overlaps(a, b), (
                    f"coloring bug: experts {a},{b} share color {c} but "
                    f"their supports overlap")
    return colors


def _set_trainable_expert_subset(model, active_indices: List[int]) -> int:
    """Freeze everything, then unfreeze ONLY the given experts.

    The base and all other experts stay frozen but keep participating in
    the forward composition — the optimizer (which filters on
    ``requires_grad``) sees only the active experts. Returns the number
    of trainable param tensors.
    """
    for p in model.parameters():
        p.requires_grad = False
    active = set(active_indices)
    for idx, expert in enumerate(model.experts):
        if idx in active:
            for p in expert.parameters():
                p.requires_grad = True
    return sum(1 for p in model.parameters() if p.requires_grad)


def _frozen_param_signature(model, frozen_indices: List[int]) -> Dict[int, float]:
    """Cheap per-expert signature of frozen params (abs-sum), for the
    post-block immutability check."""
    sig = {}
    for idx in frozen_indices:
        s = 0.0
        for p in model.experts[idx].parameters():
            s += float(p.detach().abs().sum())
        sig[idx] = s
    return sig


def _run_schwarz_phase3(
    ctx: TrainingContext,
    epoch_budget: int,
    cfg: Dict,
) -> SegmentResult:
    """Distill warm start + Schwarz freeze/unfreeze blocks for phase 3."""
    model = ctx.model
    device = ctx.device
    problem = cfg['problem']
    pc = cfg[problem]
    scfg = ctx.adaptive_cfg.get('schwarz', {}) or {}

    leaf_info = model.get_leaf_info()
    expert_indices = sorted(idx for _, idx in leaf_info if idx >= 0)
    regions = model.regions
    sigma_fraction = ctx.adaptive_cfg['sigma_fraction']
    g_lo, g_hi = _domain_box(pc)

    # Stash originals (restored at the end; fine-tune uses them).
    orig_loss_fn = ctx.loss_fn
    orig_train_data = ctx.train_data
    orig_train_loader = ctx.train_loader

    # ════════════════════════════════════════════════════════════════
    # 1) DISTILL warm start: each expert -> u0 on its own support
    # ════════════════════════════════════════════════════════════════
    distill_cfg = scfg.get('distill', {}) or {}
    d_epochs = int(distill_cfg.get('epochs', 2000))
    if d_epochs > 0:
        from trainer.orchestrator import _set_trainable
        _set_trainable(model, 'leaves')
        logger.info(f"[Schwarz] DISTILL warm start: {len(expert_indices)} "
                    f"experts -> frozen root (u0) on their supports, "
                    f"{d_epochs} epochs, "
                    f"optimizer={distill_cfg.get('optimizer_1', 'adam')}")
        distill_data = build_distill_data(
            model.base_model, expert_indices, regions, cfg, device,
            seed=ctx.base_seed + ctx.epoch)
        distill_loss = build_distill_loss(
            model, cfg, orig_loss_fn=orig_loss_fn)

        ctx.loss_fn = distill_loss
        ctx.train_data = distill_data
        ctx.train_loader = _create_split_dataloader(
            distill_data, cfg['batch_size'], shuffle=True)
        ctx._schwarz_context = {
            'mode': 'distill',
            'expert_indices': expert_indices,
            'regions': regions,
            'base_model': model.base_model,
        }

        seg_cfg = dict(cfg)
        seg_cfg['optimizer_1'] = distill_cfg.get('optimizer_1', 'adam')
        seg_cfg['optimizer_2'] = distill_cfg.get('optimizer_2', None)
        seg_cfg['optimizer_switch_epoch'] = distill_cfg.get(
            'optimizer_switch_epoch', 10 ** 9)
        seg_cfg['lr'] = distill_cfg.get('lr', 0.001)
        res_d = _train_segment(ctx, 'distill', d_epochs, seg_cfg)
        logger.info(f"[Schwarz] distill done: rel_l2={res_d.final_rel_l2:.6e} "
                    f"(root territory expected)")
        _plot_expert_curves(ctx, distill_loss, regions, 'distill',
                            len(expert_indices))
    else:
        logger.info("[Schwarz] distill warm start DISABLED (epochs=0)")

    # ════════════════════════════════════════════════════════════════
    # 2) SCHWARZ blocks: colored freeze/unfreeze sweeps
    # ════════════════════════════════════════════════════════════════
    group_mode = scfg.get('group_mode', 'colored')
    coloring = compute_expert_coloring(
        expert_indices, regions, sigma_fraction, g_lo, g_hi,
        mode=group_mode)
    n_colors = len(coloring)
    _fmt = lambda v: [round(float(x), 6) for x in v]
    logger.info(f"[Schwarz] coloring ({group_mode}): {n_colors} colors "
                f"for {len(expert_indices)} experts")
    for c, members in enumerate(coloring):
        for e in members:
            b_lo, b_hi = inflated_bounds(
                regions[e], sigma_fraction, g_lo, g_hi)
            logger.info(f"[Schwarz]   color {c}: expert={e} "
                        f"support=[{_fmt(b_lo)}..{_fmt(b_hi)}]")

    p_block = int(scfg.get('epochs_per_block', 500))
    n_blocks = max(1, epoch_budget // p_block)
    n_sweeps = (n_blocks + n_colors - 1) // n_colors
    logger.info(f"[Schwarz] schedule: {n_blocks} blocks x {p_block} epochs "
                f"(~{n_sweeps} sweeps over {n_colors} colors), "
                f"optimizer={scfg.get('optimizer', 'ssbroyden')}")

    # Composition IC/BC batch: the plain training set's IC/BC rows (exact
    # physics on u_theta, every block, full weight) — same extraction as
    # the split segment.
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
    logger.info(f"[Schwarz] composition IC/BC term: "
                f"n_ic={int(ic_bc_batch['mask']['IC'].sum())}, "
                f"n_bc={int(ic_bc_batch['mask']['BC'].sum())} "
                f"(present in every block, full weight)")

    # ONE loss object across all blocks: per-expert history spans the
    # whole phase. Its grouped residual (sum of per-expert_id means) is
    # exactly one weight-1 mean per active support on Schwarz block data.
    schwarz_loss = build_split_loss(
        model, cfg, orig_loss_fn=orig_loss_fn, ic_bc_batch=ic_bc_batch,
    )
    ctx.loss_fn = schwarz_loss

    block_summary = []
    res = SegmentResult()
    for b in range(n_blocks):
        sweep = b // n_colors
        ci = b % n_colors
        active = coloring[ci]
        frozen_experts = [e for e in expert_indices if e not in active]

        n_trainable = _set_trainable_expert_subset(model, active)
        n_frozen = sum(1 for pp in model.parameters()
                       if not pp.requires_grad)
        logger.info(
            f"[Schwarz] block {b + 1}/{n_blocks} (sweep {sweep}, color {ci}): "
            f"ACTIVE experts={active}, FROZEN experts={frozen_experts} "
            f"(+base); trainable param tensors={n_trainable}, "
            f"frozen={n_frozen}")

        block_data = sample_schwarz_residuals(
            active, regions, cfg, device,
            seed=ctx.base_seed + ctx.epoch)
        ctx.train_data = block_data
        ctx.train_loader = _create_split_dataloader(
            block_data, cfg['batch_size'], shuffle=True)
        ctx._schwarz_context = {
            'mode': 'block',
            'active_indices': active,
            'regions': regions,
        }

        seg_cfg = dict(cfg)
        seg_cfg['optimizer_1'] = scfg.get('optimizer', 'ssbroyden')
        seg_cfg['optimizer_2'] = None
        seg_cfg['optimizer_switch_epoch'] = 10 ** 9
        if 'lr' in scfg:
            seg_cfg['lr'] = scfg['lr']

        frozen_sig = _frozen_param_signature(model, frozen_experts)
        res = _train_segment(
            ctx, f'phase3_s{sweep}_c{ci}', p_block, seg_cfg,
            reconcile_best=False)

        # Freeze-correctness check: frozen experts must be bitwise idle.
        sig_after = _frozen_param_signature(model, frozen_experts)
        bad = [e for e in frozen_experts
               if sig_after[e] != frozen_sig[e]]
        if bad:
            logger.error(f"[SchwarzCheck] FROZEN PARAMS CHANGED for "
                         f"experts {bad} during block {b + 1} — freeze "
                         f"logic is broken!")
        else:
            logger.info(f"[SchwarzCheck] frozen params unchanged: OK "
                        f"({len(frozen_experts)} frozen experts)")

        block_summary.append({
            'block': b + 1, 'sweep': sweep, 'color': ci,
            'active': list(active), 'rel_l2': res.final_rel_l2,
        })
        logger.info(f"[Schwarz] block {b + 1} end: "
                    f"rel_l2={res.final_rel_l2:.6e}")
        if res.nan_detected or res.oom_stopped:
            logger.error(f"[Schwarz] block {b + 1} aborted "
                         f"({res.stop_reason}) — stopping the schedule")
            break

    # Per-sweep summary table (verify sweep-over-sweep improvement).
    logger.info("[Schwarz] ══ per-block summary ══")
    for row in block_summary:
        logger.info(f"[Schwarz]   block {row['block']:>3} sweep {row['sweep']} "
                    f"color {row['color']} active={row['active']} "
                    f"rel_l2={row['rel_l2']:.6e}")
    ctx.metrics.setdefault('schwarz_blocks', []).extend(block_summary)

    _plot_expert_curves(ctx, schwarz_loss, regions, 'phase3_schwarz',
                        len(expert_indices))

    # Save loss histories into metrics (same keys the split path uses).
    peh = getattr(schwarz_loss, '_per_expert_history', {})
    if peh:
        ctx.metrics.setdefault('split_expert_losses', {})['phase3'] = peh
    gh = getattr(schwarz_loss, '_global_history', {})
    if gh and any(gh.values()):
        ctx.metrics.setdefault('split_composition_losses', {})['phase3'] = gh

    # Restore context for fine-tune (all leaves trainable is set there).
    ctx.loss_fn = orig_loss_fn
    ctx.train_data = orig_train_data
    ctx.train_loader = orig_train_loader
    ctx._schwarz_context = None
    return res


def _plot_expert_curves(ctx, loss_fn, regions, tag, n_experts):
    """Per-expert training-curve panel (same artifact the split path saves)."""
    peh = getattr(loss_fn, '_per_expert_history', {})
    if not peh:
        return

    def _to_numpy(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    try:
        training_plots_dir = ctx.run_dir / 'training_plots'
        training_plots_dir.mkdir(exist_ok=True)
        plot_path = (training_plots_dir /
                     f'expert_curves_after_{tag}_E{n_experts}.png')
        plot_per_expert_curves(
            peh,
            list(regions),
            plot_path,
            domain_bounds=ctx.domain_bounds,
            gt_grid=_to_numpy(ctx.gt_grid),
            grid_x=_to_numpy(ctx.gt_x),
            grid_t=_to_numpy(ctx.gt_t),
            segment_name=tag,
            split_data=ctx.train_data,
        )
        logger.info(f"[SchwarzPlot] Saved training_plots/{plot_path.name}")
    except Exception as e:
        logger.warning(f"[SchwarzPlot] Failed: {e}")
