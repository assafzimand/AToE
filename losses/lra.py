"""Adaptive loss component weighting for PINNs.

Two schemes:
  'lra'       — Loss Rate Annealing (Wang et al. 2021): residual weight is
                fixed; IC and BC are boosted to match residual gradient scale.
                λ_i = max_θ|∂L_res/∂θ| / mean_θ|∂L_i/∂θ|
                Known failure mode: when IC/BC are quickly satisfied their
                gradients collapse → weights explode without bound.

  'grad_norm' — Symmetric gradient-norm balancing (jaxpi / Wang et al. 2024):
                ALL weights adapt so each term's weighted gradient has the
                same L2 norm.
                λ_i = mean_j(‖∇L_j‖₂) / ‖∇L_i‖₂
                No anchoring, no runaway: if IC is well-satisfied its weight
                grows but residual weight also adapts symmetrically, keeping
                all three contributions equal.
"""

import torch
from typing import Dict, Callable


class LRAWeights:
    """Adaptive loss weights supporting 'lra' and 'grad_norm' schemes.

    Args:
        alpha: EMA factor. 0 = frozen, 1 = instant update. Paper default 0.1.
        update_every: Epochs between updates (each costs 3 backward passes).
        initial_weights: Starting weights {'residual', 'ic', 'bc'}.
        scheme: 'lra' (Wang 2021, residual anchored) or
                'grad_norm' (jaxpi, all terms symmetric).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        update_every: int = 100,
        initial_weights: Dict[str, float] = None,
        scheme: str = 'grad_norm',
        scheme_cfg: Dict[str, float] = None,
    ):
        self.alpha = alpha
        self.update_every = update_every
        self.scheme = scheme
        self.scheme_cfg = scheme_cfg if scheme_cfg is not None else {}

        if initial_weights is not None:
            self.weights: Dict[str, float] = dict(initial_weights)
        else:
            self.weights: Dict[str, float] = {'residual': 1.0, 'ic': 1.0}

        self.fixed_residual_weight = self.weights['residual']
        self.last_grad_norms: Dict[str, float] = {k: 0.0 for k in self.weights}

    def update(
        self,
        model: torch.nn.Module,
        loss_fn: Callable,
        batch: Dict[str, torch.Tensor],
    ) -> bool:
        """Recompute weights from per-term gradient norms.

        Returns True if an update happened, False when skipped (split-loss
        batches, which return per-expert nested dicts).
        """
        components = loss_fn(model, batch, return_components=True)
        if any(isinstance(v, dict) for v in components.values()):
            # Split-loss batches return per-expert nested dicts; LRA balances
            # the global composed loss only, so skip the update here.
            return False
        # 'total' is the weighted sum of the other terms, not a component
        components = {k: v for k, v in components.items() if k != 'total'}
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        grad_norms: Dict[str, float] = {}

        for key, loss_val in components.items():
            if not isinstance(loss_val, torch.Tensor) or not loss_val.requires_grad:
                grad_norms[key] = 1e-8
                continue
            grads = torch.autograd.grad(
                loss_val, trainable_params,
                retain_graph=True,
                allow_unused=True,
            )
            if self.scheme == 'grad_norm':
                # Full L2 norm of the gradient vector (jaxpi formula)
                flat = torch.cat([g.flatten() for g in grads if g is not None])
                grad_norms[key] = max(flat.norm().item(), 1e-8)
            else:
                # LRA: max|grad| for residual, mean|grad| for others
                if key == 'residual':
                    grad_norms[key] = max(
                        max(g.abs().max().item() for g in grads if g is not None),
                        1e-8,
                    )
                else:
                    grad_norms[key] = max(
                        torch.mean(torch.stack([
                            g.abs().mean() for g in grads if g is not None
                        ])).item(),
                        1e-8,
                    )

        self.last_grad_norms = grad_norms.copy()
        model.zero_grad()

        if self.scheme == 'grad_norm':
            # jaxpi formula: w_i = (Σ_j ‖∇L_j‖) / ‖∇L_i‖; epsilon prevents
            # division-by-zero when a term is already well-satisfied
            # (e.g. IC after LS-init).
            sum_norm = sum(grad_norms.values())
            for key in grad_norms:
                epsilon = self.scheme_cfg['epsilon']
                target = sum_norm / (grad_norms[key] + epsilon * sum_norm)
                self.weights[key] = (
                    (1.0 - self.alpha) * self.weights.get(key, 1.0)
                    + self.alpha * target
                )
        else:
            # LRA: only non-residual terms adapt; residual stays fixed
            max_grad_res = grad_norms['residual']
            for key in grad_norms:
                if key == 'residual':
                    continue
                epsilon = self.scheme_cfg['epsilon']
                target = max_grad_res / (grad_norms[key] + epsilon * max_grad_res)
                self.weights[key] = (
                    (1.0 - self.alpha) * self.weights.get(key, 1.0)
                    + self.alpha * target
                )
            self.weights['residual'] = self.fixed_residual_weight

        return True

    def get(self, key: str) -> float:
        return self.weights.get(key, 1.0)

    def __repr__(self) -> str:
        w_str = ', '.join(f'{k}={v:.4f}' for k, v in self.weights.items())
        return (
            f"LRAWeights(scheme={self.scheme}, {w_str}, "
            f"alpha={self.alpha}, update_every={self.update_every})"
        )
