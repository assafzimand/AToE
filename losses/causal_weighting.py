"""
Causal weighting for PINN residual loss.

Implements the causal training strategy from
"Respecting Causality is All You Need" (Wang et al., 2022):
sort residual collocation points by time, split into temporal
chunks, and apply exponentially decaying weights:

    w_i = exp(-epsilon * sum_{j<i} L_j)

where L_j is the raw mean-squared residual of chunk j.
As training reduces losses, min(w_i) rises toward 1.0,
triggering advancement to the next epsilon level.

The paper recommends an annealing schedule for epsilon:
    [0.01, 0.1, 1, 10, 100]
advancing to the next level when min(w_i) > delta (e.g. 0.99).
Epsilon values must be tuned per-PDE since they depend on
residual magnitude (smaller for PDEs with large residuals).
"""

import torch
from typing import Dict, Optional, List


def create_causal_state(
    problem_config: Dict,
) -> Optional[Dict]:
    """
    Build a mutable causal-state dict from problem config.

    Returns None when causal training is disabled, which
    downstream functions interpret as "use plain MSE".
    """
    causal_cfg = problem_config.get('causal_training', {})
    if not causal_cfg.get('enabled', False):
        return None

    schedule = causal_cfg.get(
        'tol_schedule', [0.01, 0.1, 1.0, 10.0, 100.0])
    return {
        'enabled': True,
        'num_chunks': causal_cfg['num_chunks'],
        'schedule': list(schedule),
        'schedule_idx': 0,
        'tol': float(schedule[0]),
        'min_weight': 1.0,
        'threshold': causal_cfg.get(
            'min_weight_threshold', 0.99),
    }


def advance_causal_schedule(causal_state: Optional[Dict]) -> bool:
    """
    Check convergence and advance epsilon if appropriate.

    Call this once per epoch from the trainer. Returns True
    if epsilon was advanced, False otherwise.
    """
    if causal_state is None:
        return False
    idx = causal_state['schedule_idx']
    schedule = causal_state['schedule']
    if idx >= len(schedule) - 1:
        return False
    if causal_state['min_weight'] > causal_state['threshold']:
        causal_state['schedule_idx'] = idx + 1
        causal_state['tol'] = float(schedule[idx + 1])
        causal_state['min_weight'] = 0.0
        return True
    return False


def compute_causal_residual(
    residual_squared: torch.Tensor,
    t_residual: torch.Tensor,
    causal_state: Optional[Dict],
    update_state: bool = True,
) -> torch.Tensor:
    """
    Compute residual MSE, optionally with causal weighting.

    When causal_state is None (disabled), returns plain mean.
    Otherwise applies temporal causal weights and updates
    causal_state['min_weight'] for the annealing check (if update_state=True).
    
    Args:
        residual_squared: Squared residuals
        t_residual: Time values for residual points
        causal_state: Causal training state dict (or None if disabled)
        update_state: If False, don't update causal_state (use during eval)
    """
    if causal_state is None:
        return torch.mean(residual_squared)
    return _apply_causal_weights(
        residual_squared,
        t_residual,
        causal_state['num_chunks'],
        causal_state['tol'],
        causal_state if update_state else None,
    )


def _apply_causal_weights(
    residual_squared: torch.Tensor,
    t_residual: torch.Tensor,
    num_chunks: int,
    causal_tol: float,
    causal_state: Optional[Dict],
) -> torch.Tensor:
    """
    Sort residual points by time, bin into chunks, and
    weight each chunk by exp(-epsilon * cumulative_loss).
    """
    t_flat = t_residual.view(-1)
    N = t_flat.shape[0]

    if N == 0 or num_chunks <= 1 or N < num_chunks:
        return torch.mean(residual_squared)

    sort_idx = torch.argsort(t_flat)
    r2_sorted = residual_squared[sort_idx]

    chunk_size = max(1, N // num_chunks)
    chunk_losses: List[torch.Tensor] = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = (start + chunk_size
               if i < num_chunks - 1 else N)
        chunk_losses.append(
            torch.mean(r2_sorted[start:end]))

    chunk_losses_t = torch.stack(chunk_losses)

    cumsum = torch.cumsum(chunk_losses_t.detach(), dim=0)
    zero = torch.zeros(1, device=cumsum.device)
    shifted = torch.cat([zero, cumsum[:-1]])
    weights = torch.exp(-causal_tol * shifted).clamp(min=1e-8).detach()

    if causal_state is not None:
        batch_min = weights.min().item()
        causal_state['min_weight'] = min(
            causal_state['min_weight'], batch_min)
        causal_state['last_weights'] = weights.detach().tolist()
        causal_state['last_chunk_losses'] = chunk_losses_t.detach().tolist()
        t_sorted = t_flat[sort_idx]
        causal_state['last_chunk_tmax'] = [
            t_sorted[min((i + 1) * chunk_size, N) - 1].item()
            for i in range(num_chunks)
        ]

    weighted = torch.sum(weights * chunk_losses_t) / num_chunks
    return weighted

