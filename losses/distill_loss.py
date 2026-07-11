"""Distillation loss: per-expert supervised MSE to the frozen root (u0).

Schwarz warm start — before the freeze/unfreeze sweeps begin, every leaf
expert is fitted INDIVIDUALLY to u0 on its whole window support (the
inflated box), so that any frozen expert is a root-accurate boundary-
data source for its active neighbors in the collars.

Deliberately NOT a composed MSE(u_theta, u0): that would only constrain
the window-weighted sum, letting collar experts mutually compensate
(individually wrong, sum right) — useless as frozen boundary data.

Loss = sum over experts of mean_j ||u_j(x) - u0(x)||^2 over the
expert's own distill rows (one weight-1 mean per expert; the problems
are independent — no gradient crosses experts). No PDE terms.
"""

import torch
from typing import Dict, Callable
from adaptive.subdomain_data import KIND_DISTILL
from utils.logging_config import get_logger

logger = get_logger(__name__)


def build_distill_loss(
    model,
    cfg: Dict,
    *,
    orig_loss_fn: Callable = None,
) -> Callable:
    """Build the per-expert u0-distillation loss.

    Returns a callable ``loss_fn(model, batch)`` compatible with
    ``_train_segment``. Eval batches (no ``expert_id``/``kind`` keys)
    fall back to ``orig_loss_fn`` so the rel-L2 eval path is unchanged.
    """
    per_expert_history: Dict[int, Dict[str, list]] = {}

    def distill_loss_fn(model, batch, return_components=False, **kw):
        if 'expert_id' not in batch or 'kind' not in batch:
            if orig_loss_fn is not None:
                return orig_loss_fn(model, batch,
                                    return_components=return_components,
                                    **kw)
            x = batch['x']
            t = batch['t']
            h_gt = batch['h_gt']
            h_pred = model(torch.cat([x, t], dim=1))
            return torch.mean((h_pred - h_gt) ** 2)

        x = batch['x']
        t = batch['t']
        h_gt = batch['h_gt']
        expert_ids = batch['expert_id']
        kinds = batch['kind']
        device = x.device

        record_now = distill_loss_fn._record_next
        if record_now:
            distill_loss_fn._record_next = False

        dmask = (kinds == KIND_DISTILL)
        total_loss = torch.tensor(0.0, device=device)
        all_comps = {}
        for eidx in [e for e in expert_ids[dmask].unique().tolist()
                     if e >= 0]:
            emask = dmask & (expert_ids == eidx)
            xt_e = torch.cat([x[emask], t[emask]], dim=1)
            u_e = model.forward_single_expert(eidx, xt_e)
            mse_e = torch.mean((u_e - h_gt[emask]) ** 2)
            total_loss = total_loss + mse_e
            if record_now:
                per_expert_history.setdefault(
                    eidx, {'distill': [], 'total': []})
                per_expert_history[eidx]['distill'].append(
                    float(mse_e.detach()))
                per_expert_history[eidx]['total'].append(
                    float(mse_e.detach()))
            if return_components:
                all_comps[eidx] = {'distill': mse_e, 'total': mse_e}

        if return_components:
            return all_comps
        return total_loss

    distill_loss_fn._per_expert_history = per_expert_history
    distill_loss_fn._record_next = True  # trainer re-arms once per epoch
    return distill_loss_fn
