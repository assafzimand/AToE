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
    final_eval_loss: float = float('inf')
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

    # Data / loaders
    train_data: Any = None
    eval_data: Any = None
    train_loader: Any = None
    eval_loader: Any = None

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
    metrics: Dict = field(default_factory=dict)
    best_eval_loss: float = float('inf')
    best_checkpoint_path: Any = None
    patience_epochs: int = 0  # train-loss plateau window, counted in epochs
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
    eval_loss: float = 0.0
    eval_rel_l2: float = 0.0
    eval_inf_norm: float = 0.0
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
