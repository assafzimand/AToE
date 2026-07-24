"""Shared training state container for the trainer package.

``TrainingContext`` carries all state that crosses the boundaries between the
three phases of ``trainer.orchestrator.train``:

  * ``trainer.setup._setup_training`` builds and returns a fully-populated context,
  * ``trainer.epoch_loop._train_segment`` consumes it per segment (reading
    state, writing back the values it reassigns), and
  * ``trainer.finalize._finalize_training`` reads post-loop state to emit
    checkpoints/plots.

This is purely a state bag; it holds no behavior.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class SegmentResult:
    """Outcome of a single :func:`trainer.trainer._train_segment` call.

    ``stop_reason`` is one of: ``'budget'`` (ran the full epoch budget),
    ``'early_stop'`` (patience plateau), ``'nan'`` (divergence), ``'oom'`` (GPU OOM).
    """
    nan_detected: bool = False
    stopped_early: bool = False
    stop_reason: str = 'budget'
    epochs_run: int = 0
    final_train_loss: float = float('inf')
    final_rel_l2: float = float('inf')
    oom_stopped: bool = False


@dataclass
class TrainingContext:
    # ── Core objects ────────────────────────────────────────────────────────
    model: Any = None
    loss_fn: Optional[Callable] = None
    cfg: Optional[Dict] = None
    problem: Optional[str] = None
    problem_cfg: Optional[Dict] = None
    device: Any = None
    run_dir: Optional[Path] = None

    # Data / loaders. ``plain_train_data`` always points at the plain-format
    # (x/t/h_gt/mask) training set, even while a split segment has swapped
    # ``train_data`` to the split schema — it is the probe set for the
    # loss-component snapshots logged at eval epochs.
    train_data: Any = None
    train_loader: Any = None
    plain_train_data: Any = None

    # ── Phase / optimizer lifecycle ─────────────────────────────────────────
    active_cfg: Optional[Dict] = None
    epochs: int = 0
    phase3_epochs: int = 0
    current_phase: int = 0
    use_three_phase: bool = False
    optimizer_2_name: Optional[str] = None
    switch_epoch: int = 0
    batches_per_epoch: int = 1
    total_steps_estimate: int = 0
    patience_start_epoch: int = 1
    optimizer: Any = None
    current_optimizer_name: Optional[str] = None
    lr_scheduler: Any = None
    step_count: int = 0

    # ── Reporting intervals ─────────────────────────────────────────────────
    print_every: int = 0
    eval_every: int = 0
    save_every: int = 0

    # ── Metrics / best-model tracking ───────────────────────────────────────
    # All rel-L2 / inf-norm values are computed on the ground-truth solver's
    # NATIVE grid (the single reported metric — there is no eval dataset).
    # Best tracking is PER SEGMENT (best_model_<segment>.pt, reconciled with
    # the end-of-segment weights); these fields hold the LATEST segment's
    # best, so after the last segment they are the run's result.
    metrics: Dict = field(default_factory=dict)
    best_rel_l2: float = float('inf')
    best_checkpoint_path: Any = None
    # Patience: consecutive EVALS (every eval_every) in which the rel-L2 metric
    # failed to beat the best-so-far by at least patience_rel_delta. On an
    # optimizer_1 plateau the epoch loop fast-forwards to the switch; on an
    # optimizer_2 / no-switch plateau it stops the segment. patience_epochs /
    # patience_intervals are retained for backward-compatible config parsing.
    patience_evals: int = 0
    patience_intervals: Any = None
    patience_epochs: int = 0
    min_epochs: int = 0
    patience_rel_delta: float = 0.0

    # LRA (adaptive loss weighting)
    lra_weights: Any = None

    # Output dirs
    checkpoint_dir: Any = None

    # ── Adaptive PINN / spawning state ──────────────────────────────────────
    adaptive_cfg: Optional[Dict] = None
    is_adaptive: bool = False
    initial_train_cfg: Any = None
    pretrained_base_checkpoint: Any = None
    pretrained_local_expert_checkpoint: Any = None
    _pretrained_force_spawn: bool = False
    region_detector: Any = None
    max_experts: int = 0
    variable_for_node_accept: Any = None
    variable_for_expert_size: Any = None
    domain_bounds: Any = None
    gt_grid: Any = None
    gt_x: Any = None
    gt_t: Any = None
    adaptive_plots_dir: Any = None
    rejected_regions: List = field(default_factory=list)
    leaf_loss_history: List = field(default_factory=list)

    # ── Loop control / live metrics ─────────────────────────────────────────
    total_epochs: int = 0
    start_time: float = 0.0
    timer: Any = None
    train_loss: float = 0.0
    rel_l2: Any = None
    inf_norm: Any = None
    resample_every: int = 0
    base_seed: int = 0
    grad_clip_norm: Any = None
    expert_grad_clip_norm: Any = None
    adaptive_sampling_enabled: bool = False
    epoch: int = 0
    _nan_detected: bool = False
    oom_stopped: bool = False

    # ── Split-loss context (set during _run_split_segment, else None) ──────
    _split_context: Any = None

    # ── Closure handles (created in the loop, consumed by finalize) ──────────
    _emergency_metrics_save: Any = None
    _atexit: Any = None
