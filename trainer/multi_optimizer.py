"""Per-expert-group optimizers for the phase-3 split segment.

Motivation (measured, 2026-08-11, W0 order-1 run): the split loss is
block-separable across experts (verified: exact-zero cross-gradients; the
only coupling is the periodic BC pairing between the two boundary-touching
experts), yet a SINGLE full-batch quasi-Newton optimizer imposes three
global objects on all experts at once — one inverse-Hessian estimate, one
line-search step size/Wolfe test on the TOTAL loss (hard-coded
tolerance_change=1e-9, max_ls=25 in scimba), and one stopping decision.
When the fast experts reach their float64 floor, the shared line search
collapses and rejects every step while the laggard still holds a healthy
gradient (measured max|g| = 6.7e-6 at the freeze, 8 orders above
tolerance_grad).

``multi_optimizers: true`` (base_config, phase-3 only) replaces the single
SSBroyden with one SSBroyden PER GROUP: each interior expert alone, and the
two periodic-boundary experts together (they share the pairing term). Each
group's closure evaluates only its own experts' rows, so the summed work per
step is about one full-batch closure — approximately runtime-neutral — and
each group gets its own Hessian, step size and stopping condition.
"""
from typing import Callable, Dict, List

import torch

from utils.logging_config import get_logger

logger = get_logger(__name__)


def compute_expert_groups(model, cfg: Dict) -> List[List[int]]:
    """Group leaf experts: singletons, except the periodic-boundary pair.

    For periodic problems the two experts whose regions touch the physical
    x-boundary share the cross-expert pairing term and must live in one
    group; everyone else is independent (verified zero cross-gradients).
    """
    from losses.split_loss import PERIODIC_PROBLEMS

    problem = cfg['problem']
    pc = cfg[problem]
    leaf_idx = sorted(i for i in model.leaf_indices if i >= 0)
    if problem not in PERIODIC_PROBLEMS:
        return [[i] for i in leaf_idx]

    x_lo, x_hi = pc['spatial_domain'][0]
    eps = 1e-12
    boundary, interior = [], []
    for i in leaf_idx:
        r = model.regions[i]
        touches = (abs(float(r.bounds_lower[0]) - float(x_lo)) < eps
                   or abs(float(r.bounds_upper[0]) - float(x_hi)) < eps)
        (boundary if touches else interior).append(i)
    groups = ([boundary] if boundary else []) + [[i] for i in interior]
    return groups


class MultiExpertOptimizer:
    """K independent SSBroyden optimizers stepping per-group sub-batches.

    Drop-in for the full-batch ``optimizer.step(closure)`` call site: the
    passed closure is ignored; each group's own closure evaluates the split
    loss on that group's rows only (the loss is block-separable, so the
    per-group gradient equals its block of the full gradient, and the sum of
    group losses equals the full loss). Returns the summed loss tensor.
    """

    def __init__(self, model, groups: List[List[int]], cfg: Dict,
                 seg_cfg: Dict, data_fn: Callable[[], Dict],
                 loss_fn: Callable):
        from scimba_torch.optimizers.ssbroyden import SSBroyden
        from trainer.setup import _opt_cfg

        self.model = model
        self.groups = groups
        self.data_fn = data_fn
        self.loss_fn = loss_fn
        self._cache_key = None
        self._subbatches = None

        lr = _opt_cfg(seg_cfg, 'ssbroyden', 'lr', 'ssbroyden_lr', 1.0)
        tol = _opt_cfg(seg_cfg, 'ssbroyden', 'tolerance_grad',
                       'ssbroyden_tolerance_grad', 1e-10)
        self.opts = []
        for g in groups:
            params = [p for i in g for p in model.experts[i].parameters()
                      if p.requires_grad]
            self.opts.append(SSBroyden(params, lr=lr, tolerance_grad=tol,
                                       method='ssbroyden'))
        n_par = [sum(p.numel() for i in g for p in model.experts[i].parameters())
                 for g in groups]
        logger.info(f"[MultiOpt] phase3 per-group SSBroyden: "
                    f"{len(groups)} groups {groups} "
                    f"(params {n_par}, lr={lr}, tolerance_grad={tol})")
        if (cfg.get('grad_clip_norm') is not None
                or cfg.get(cfg['problem'], {}).get('grad_clip_norm') is not None):
            logger.warning("[MultiOpt] grad clipping is not applied inside "
                           "per-group closures (unsupported in multi mode).")

    # ── torch.optim-compatible surface used by the trainer ──
    @property
    def param_groups(self):
        return self.opts[0].param_groups

    def zero_grad(self, set_to_none: bool = True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def _group_batches(self):
        data = self.data_fn()
        key = id(data)
        if self._cache_key == key:
            return self._subbatches
        eid = data['expert_id']
        subs = []
        for g in self.groups:
            m = torch.zeros_like(eid, dtype=torch.bool)
            for i in g:
                m |= (eid == i)
            subs.append({k: (v[m] if torch.is_tensor(v)
                             and v.dim() >= 1 and v.shape[0] == eid.shape[0]
                             else v)
                         for k, v in data.items()})
        self._cache_key, self._subbatches = key, subs
        return subs

    def step(self, closure: Callable = None):  # noqa: ARG002 — see class doc
        total = None
        for opt, sub in zip(self.opts, self._group_batches()):
            def group_closure(sub=sub, opt=opt):
                opt.zero_grad()
                loss = self.loss_fn(self.model, sub)
                loss.backward()
                return loss
            l = opt.step(group_closure)
            l = l.detach() if torch.is_tensor(l) else torch.tensor(float(l))
            total = l if total is None else total + l
        return total


def maybe_wrap_multi_optimizer(optimizer, current_optimizer_name: str,
                               model, cfg: Dict, seg_cfg: Dict,
                               ctx, loss_fn, segment_name: str):
    """Replace a freshly created SSBroyden with the per-group version when
    ``multi_optimizers: true`` and we are inside the phase-3 split segment.
    Returns the (possibly wrapped) optimizer unchanged otherwise."""
    if not cfg.get('multi_optimizers', False):
        return optimizer
    if segment_name != 'phase3':
        return optimizer
    if getattr(ctx, '_split_context', None) is None:
        return optimizer
    if current_optimizer_name != 'SSBroyden':
        if current_optimizer_name == 'LBFGS':
            logger.warning("[MultiOpt] requested but the optimizer resolved "
                           "to LBFGS (scimba missing?) — keeping single "
                           "optimizer.")
        return optimizer
    groups = compute_expert_groups(model, cfg)
    if len(groups) < 2:
        logger.info("[MultiOpt] <2 groups — single optimizer kept.")
        return optimizer
    return MultiExpertOptimizer(model, groups, cfg, seg_cfg,
                                data_fn=lambda: ctx.train_data,
                                loss_fn=loss_fn)
