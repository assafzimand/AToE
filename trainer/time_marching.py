"""
Time Marching Module for PINN Training.

Implements sequential training over temporal windows, essential for chaotic PDEs
like Kuramoto-Sivashinsky where standard PINNs fail due to error accumulation.

Key idea: Split temporal domain into windows, train AToE on each window sequentially,
using the previous window's terminal prediction as the next window's initial condition.
"""

import copy
import json
import shutil
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import importlib

from utils.logging_config import get_logger, add_window_log_handler, remove_window_log_handler
from utils.io import resolve_experts_architecture

logger = get_logger(__name__)


@dataclass
class TimeWindow:
    """Represents a single time window for time marching."""
    idx: int          # 0, 1, 2, ...
    t_start: float    # start of window
    t_end: float      # end of window
    is_first: bool    # True for window 0
    M: int            # experts allocated to this window


def compute_m_per_window(global_M: int, num_windows: int, distribution: str) -> List[int]:
    """
    Distribute global_M experts across windows based on distribution strategy.
    
    Args:
        global_M: Total number of experts to distribute
        num_windows: Number of time windows
        distribution: 'equal' | 'linear' | 'quadratic'
    
    Returns:
        List of M values for each window, summing to global_M
    
    Examples (global_M=40, num_windows=5):
        - equal: [8, 8, 8, 8, 8]
        - linear: [3, 5, 8, 11, 13]
        - quadratic: [1, 4, 7, 13, 15]
    """
    if distribution == 'equal':
        base = global_M // num_windows
        remainder = global_M % num_windows
        result = [base] * num_windows
        result[-1] += remainder  # add remainder to last window
        return result
    
    elif distribution == 'linear':
        # M_i proportional to (i+1).  Sum of 1+2+...+n = n*(n+1)/2.
        # Guarantee >= 1 per window: give each window 1 first, then distribute
        # the remainder with the largest-remainder (Hamilton) method so the
        # rounding correction is never dumped onto a single window.
        weights = [(i + 1) for i in range(num_windows)]
        total_weight = sum(weights)
        remaining = global_M - num_windows
        raw = [remaining * w / total_weight for w in weights]
        floors = [int(r) for r in raw]
        leftover = remaining - sum(floors)
        order = sorted(range(num_windows), key=lambda i: raw[i] - floors[i], reverse=True)
        for i in order[:leftover]:
            floors[i] += 1
        return [1 + f for f in floors]

    elif distribution == 'quadratic':
        # M_i proportional to (i+1)^2.  Sum of 1^2+...+n^2 = n*(n+1)*(2n+1)/6.
        # Same Hamilton approach as linear to prevent any window from getting 0.
        weights = [(i + 1) ** 2 for i in range(num_windows)]
        total_weight = sum(weights)
        remaining = global_M - num_windows
        raw = [remaining * w / total_weight for w in weights]
        floors = [int(r) for r in raw]
        leftover = remaining - sum(floors)
        order = sorted(range(num_windows), key=lambda i: raw[i] - floors[i], reverse=True)
        for i in order[:leftover]:
            floors[i] += 1
        return [1 + f for f in floors]

    elif distribution in ('linear_zero', 'quadratic_zero'):
        # ZERO-BASED variants: weights i^p (i = 0..n-1, p = 1 or 2), Hamilton
        # over the FULL M, no per-window floor — window 0 always gets M=0.
        # Semantics of an M=0 window: ONE expert spanning the whole window
        # (no tree fit). Implemented for only_for_tree_structure mode (the
        # orchestrator's windowed tree spawns the whole-slice expert); in
        # REAL time marching an M=0 window would train root-only — use with
        # care there.
        power = 1 if distribution == 'linear_zero' else 2
        weights = [i ** power for i in range(num_windows)]
        total_weight = sum(weights)
        raw = [global_M * w / total_weight for w in weights]
        floors = [int(r) for r in raw]
        leftover = global_M - sum(floors)
        order = sorted(range(num_windows), key=lambda i: raw[i] - floors[i], reverse=True)
        for i in order[:leftover]:
            floors[i] += 1
        return floors

    else:
        raise ValueError(f"Unknown m_distribution: {distribution}. Use 'equal', "
                         f"'linear', 'quadratic', 'linear_zero', or 'quadratic_zero'.")


def compute_time_windows(
    temporal_domain: List[float], 
    num_windows: int,
    global_M: int,
    m_distribution: str
) -> List[TimeWindow]:
    """
    Split [t_min, t_max] into num_windows equal, non-overlapping windows with M allocation.
    
    Args:
        temporal_domain: [t_min, t_max] from config
        num_windows: Number of windows to create
        global_M: Total experts to distribute
        m_distribution: Distribution strategy
    
    Returns:
        List of TimeWindow objects
    """
    t_min, t_max = temporal_domain
    dt = (t_max - t_min) / num_windows
    m_values = compute_m_per_window(global_M, num_windows, m_distribution)
    
    windows = []
    for i in range(num_windows):
        windows.append(TimeWindow(
            idx=i,
            t_start=t_min + i * dt,
            t_end=t_min + (i + 1) * dt,
            is_first=(i == 0),
            M=m_values[i]
        ))
    return windows


def narrow_config_for_window(cfg: Dict, window: TimeWindow, prev_model: nn.Module = None) -> Dict:
    """
    Create a copy of cfg with temporal_domain and M narrowed for this window.
    
    This is the key trick that makes everything work:
    - Dataset generation uses temporal_domain → generates points in [t_start, t_end]
    - Tree spawning uses domain_bounds from data → automatically matches window
    - Resampling uses temporal_domain → stays within window
    - M_experts_num is set per-window for variable expert allocation
    
    Args:
        cfg: Full configuration dictionary
        window: TimeWindow to narrow to
        prev_model: Model from previous window (for IC override during resampling)
    
    Returns:
        Deep copy of cfg with temporal_domain and M_experts_num updated
    """
    window_cfg = copy.deepcopy(cfg)
    problem = window_cfg['problem']
    
    # Save original temporal domain BEFORE narrowing — solvers need it to compute
    # the full-domain numerical solution once and cache it, then serve each window
    # from the correct time slice rather than re-solving with a wrong per-window IC.
    original_temporal_domain = cfg[problem]['temporal_domain'][:]

    # Narrow temporal domain
    window_cfg[problem]['temporal_domain'] = [window.t_start, window.t_end]

    # Set window-specific M
    window_cfg['adaptive_pinn']['M_experts_num'] = window.M

    # Add flag to indicate time marching is active (for eval filtering and IC override)
    # prev_model is stored as reference for IC override after resampling
    window_cfg['_time_marching_window'] = {
        'enabled': True,
        't_start': window.t_start,
        't_end': window.t_end,
        'idx': window.idx,
        'prev_model': prev_model,  # None for window 0, model for windows 1+
        'original_temporal_domain': original_temporal_domain,
    }
    
    return window_cfg


def resolve_window_pretrained_checkpoint(
    ckpt_value,
    window: TimeWindow,
    num_windows: int,
) -> Optional[str]:
    """Resolve a pretrained-checkpoint config value for one time window.

    Non-time-marching runs point ``pretrained_base_checkpoint`` /
    ``pretrained_local_expert_checkpoint`` at a single ``.pt`` file. For
    time-marching runs the value may instead be a DIRECTORY (e.g.
    ``roots_checkpoints/<name>/``) holding one checkpoint per window — one
    training segment across all windows, as written by
    :func:`train_with_time_marching` into ``run_dir/checkpoints/<segment>/``.
    Whether each file holds a plain root or a full leaf-expert model is up
    to the producer — the config key it is supplied under decides how the
    trainer loads it.

    The window's file is located by the folder's ``manifest.json`` when
    present (which also validates the run's window layout: window count and
    per-window time bounds must match); otherwise by filename —
    ``window_{idx}.pt`` or a unique ``window_{idx}_*.pt`` match.

    Returns:
        The per-window checkpoint path (str); the value unchanged when it
        is a single file (the same checkpoint is then reused for every
        window); or None when ``ckpt_value`` is None.
    """
    if ckpt_value is None:
        return None
    p = Path(ckpt_value)
    if not p.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint path not found: {ckpt_value}")
    if p.is_file():
        logger.info(f"    [Time Marching] Pretrained checkpoint {p} is a "
                    f"single file — reusing it for every window.")
        return str(p)

    fname = f"window_{window.idx}.pt"
    manifest_path = p / 'manifest.json'
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        saved_windows = manifest.get('num_windows')
        if saved_windows is not None and saved_windows != num_windows:
            raise ValueError(
                f"Pretrained checkpoint folder {p} was saved with "
                f"{saved_windows} windows but this run uses {num_windows}. "
                f"Match time_marching.num_windows to the checkpoint folder.")
        for entry in manifest.get('windows', []):
            if entry.get('idx') != window.idx:
                continue
            for key, have in (('t_start', window.t_start),
                              ('t_end', window.t_end)):
                want = entry.get(key)
                if want is not None and abs(want - have) > 1e-9:
                    raise ValueError(
                        f"Pretrained checkpoint folder {p}: window "
                        f"{window.idx} covers {key}={want} but this run's "
                        f"window {window.idx} has {key}={have}.")
            fname = entry.get('file', fname)
            break

    ckpt_path = p / fname
    if not ckpt_path.exists():
        # No manifest entry and no bare window_{idx}.pt — accept a unique
        # window_{idx}_*.pt (the collected-segment naming, e.g.
        # window_0_best_model_root.pt).
        matches = sorted(p.glob(f"window_{window.idx}_*.pt"))
        if len(matches) == 1:
            ckpt_path = matches[0]
        elif len(matches) > 1:
            raise ValueError(
                f"Pretrained checkpoint folder {p} has multiple candidates "
                f"for window {window.idx}: {[m.name for m in matches]}. "
                f"Keep one file per window (or add a manifest.json).")
        else:
            # SOFT miss: a partially-populated folder means "resume from
            # this stage only for the windows that have a file". The window
            # without one simply trains this stage from scratch — this is
            # what makes arbitrary window/segment resume combinations work
            # (e.g. phase3/window_0 + root/window_1 + nothing for window 2).
            logger.info(
                f"    [Time Marching] {p.name}: no checkpoint for window "
                f"{window.idx} — this window trains the stage from scratch.")
            return None
    return str(ckpt_path)


def override_ic_with_model(
    dataset: Dict[str, torch.Tensor],
    prev_model: nn.Module,
    window: TimeWindow,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Replace h_gt for IC points with predictions from prev_model.
    
    This is the key trick: IC loss in *_loss.py uses h_gt from batch.
    By replacing h_gt for IC points, we get predicted IC loss for free.
    
    The existing loss_weights.ic from the problem config is used automatically
    since the loss computation mechanism stays unchanged.
    
    Args:
        dataset: Training or eval dataset dict with 'x', 't', 'h_gt', 'mask'
        prev_model: Model from previous window to query
        window: Current window (to check if first)
        device: Device to run inference on
    
    Returns:
        Modified dataset with h_gt overridden for IC points
    """
    if window.is_first:
        return dataset  # Window 0 uses analytical IC
    
    # Get IC mask
    ic_mask = dataset['mask']['IC']
    if ic_mask.sum() == 0:
        logger.info(f"    [IC Override] Window {window.idx}: No IC points found in dataset, skipping")
        return dataset
    
    x_ic = dataset['x'][ic_mask]  # (n_ic, spatial_dim)
    t_ic = dataset['t'][ic_mask]  # (n_ic, 1) - all at window.t_start
    h_gt_original = dataset['h_gt'][ic_mask].clone()
    
    # Diagnostic: print input stats
    logger.info(f"    [IC Override] Window {window.idx}: Overriding {ic_mask.sum().item()} IC points")
    logger.info(f"      x_ic: shape={x_ic.shape}, min={x_ic.min().item():.4f}, max={x_ic.max().item():.4f}, mean={x_ic.mean().item():.4f}")
    logger.info(f"      t_ic: min={t_ic.min().item():.4f}, max={t_ic.max().item():.4f}")
    logger.info(f"      h_gt (original): min={h_gt_original.min().item():.4f}, max={h_gt_original.max().item():.4f}, mean={h_gt_original.mean().item():.4f}")
    
    # Query previous model (no gradients)
    prev_model.eval()
    with torch.no_grad():
        inputs = torch.cat([x_ic, t_ic], dim=1).to(device)
        h_pred = prev_model(inputs)
    
    # Diagnostic: print prediction stats
    has_nan = torch.isnan(h_pred).any().item()
    has_inf = torch.isinf(h_pred).any().item()
    logger.info(f"      h_pred: min={h_pred.min().item():.4f}, max={h_pred.max().item():.4f}, mean={h_pred.mean().item():.4f}")
    logger.info(f"      h_pred contains NaN: {has_nan}, Inf: {has_inf}")
    
    if has_nan or has_inf:
        logger.info(f"      [WARNING] Previous model produced invalid values! This will cause NaN divergence.")
        num_nan = torch.isnan(h_pred).sum().item()
        num_inf = torch.isinf(h_pred).sum().item()
        logger.info(f"      Number of NaN: {num_nan}, Number of Inf: {num_inf}")
    
    # Override h_gt for IC points
    dataset['h_gt'][ic_mask] = h_pred.to(dataset['h_gt'].device)
    
    logger.info(f"    [IC Override] Completed: overrode {ic_mask.sum().item()} IC points")
    
    return dataset


def _compute_full_domain_rel_l2(
    combined_model: nn.Module,
    config: Dict,
    device: torch.device,
) -> float:
    """Compute rel-L2 of the combined model over the full temporal domain.

    Scored on the solver's NATIVE solution grid (compute_native_grid_metrics)
    — the same metric as the per-window training curves, just un-windowed.
    Ground truth is never interpolated: off-node linear interpolation of the
    reference adds an artificial error floor across steep fronts (measured
    ~1.5e-3 on KdV's 512-point grid, swamping model errors of ~1e-7).
    """
    from trainer.utils import compute_native_grid_metrics

    metrics = compute_native_grid_metrics(combined_model, config, device)
    if metrics is None:
        raise RuntimeError(
            "Solver native grid unavailable — cannot compute full-domain rel-L2")
    return float(metrics['rel_l2'])


def _plot_combined_loss_curves(
    windows: List[TimeWindow],
    run_dir: Path,
    final_rel_l2: float = None,
) -> None:
    """
    Create a combined loss curve plot showing all windows concatenated.

    Reads metrics.json from each window and creates a single plot with:
    - Train loss and solver-grid rel-L2 curves concatenated across windows
    - Vertical lines showing window boundaries
    - (Optional) horizontal marker for the final full-domain rel-L2

    Args:
        windows: List of TimeWindow objects
        run_dir: Root run directory containing window subdirectories
        final_rel_l2: Full-domain rel-L2 of the combined model (added as annotation)
    """
    logger.info(f"\n  Creating combined loss curve plot...")
    
    all_train_epochs = []
    all_train_loss = []
    all_eval_epochs = []
    all_eval_rel_l2 = []

    epoch_offset = 0
    window_boundaries = [0]  # Epoch boundaries between windows

    for window in windows:
        window_metrics_path = run_dir / f"window_{window.idx}" / "metrics.json"
        if not window_metrics_path.exists():
            logger.info(f"    Warning: metrics.json not found for window {window.idx}")
            continue

        with open(window_metrics_path, 'r') as f:
            metrics = json.load(f)

        # Offset epochs to create continuous timeline
        train_epochs = np.array(metrics['train_loss_epochs']) + epoch_offset
        eval_epochs = np.array(metrics['epochs']) + epoch_offset

        all_train_epochs.extend(train_epochs)
        all_train_loss.extend(metrics['train_loss'])
        all_eval_epochs.extend(eval_epochs)
        # 'rel_l2' is the solver-grid metric ('eval_rel_l2' in older runs)
        all_eval_rel_l2.extend(metrics.get('rel_l2', metrics.get('eval_rel_l2', [])))
        
        # Update offset for next window
        if len(train_epochs) > 0:
            epoch_offset = train_epochs[-1]
            window_boundaries.append(epoch_offset)
    
    if len(all_train_epochs) == 0:
        logger.info(f"    Warning: No metrics found for any window")
        return
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Plot 1: Loss curve
    ax = axes[0]
    ax.plot(all_train_epochs, all_train_loss, 'b-', label='Train Loss',
            linewidth=2, alpha=0.8)
    
    # Add window boundary markers
    for i, boundary in enumerate(window_boundaries[1:-1], start=1):
        ax.axvline(x=boundary, color='gray', linestyle='--', 
                   linewidth=1.5, alpha=0.5)
        ax.text(boundary, ax.get_ylim()[1]*0.95, f'W{i}', 
                ha='center', va='top', fontsize=9, alpha=0.7)
    
    ax.set_xlabel('Epoch (Cumulative)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_title(f'Time Marching: Combined Loss Curves [{len(windows)} windows]', 
                 fontsize=14, fontweight='bold')
    
    # Plot 2: Relative L2 error (solver grid, per window)
    ax = axes[1]
    ax.plot(all_eval_epochs, all_eval_rel_l2, 'r-', label='Per-window Rel-L2 (grid)',
            linewidth=2, alpha=0.8)

    if final_rel_l2 is not None:
        ax.axhline(y=final_rel_l2, color='black', linestyle='--', linewidth=2,
                   label=f'Full-domain Rel-L2 (grid): {final_rel_l2:.4e}',
                   alpha=0.9)

    # Add window boundary markers
    for i, boundary in enumerate(window_boundaries[1:-1], start=1):
        ax.axvline(x=boundary, color='gray', linestyle='--',
                   linewidth=1.5, alpha=0.5)
        ax.text(boundary, ax.get_ylim()[1]*0.95, f'W{i}',
                ha='center', va='top', fontsize=9, alpha=0.7)

    ax.set_xlabel('Epoch (Cumulative)', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_title('Time Marching: Combined Relative L2 Error',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    save_path = run_dir / 'time_marching_combined_loss_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"    Combined loss curves saved to {save_path}")


def _plot_combined_heatmap(
    combined_model: nn.Module,
    config: Dict,
    run_dir: Path,
    device: torch.device
) -> None:
    """
    Create a global heatmap showing prediction and error vs ground truth.
    
    Uses the combined TimeMarchingModel to generate predictions across the
    entire temporal domain and compares with ground truth.
    
    Args:
        combined_model: TimeMarchingModel wrapping all windows
        config: Full configuration dictionary
        run_dir: Directory to save the plot
        device: Device for inference
    """
    from utils.problem_specific.generic_viz import plot_predictions_and_error_maps
    
    logger.info(f"\n  Creating combined prediction heatmap...")
    
    try:
        plot_predictions_and_error_maps(
            model=combined_model,
            save_dir=run_dir,
            config=config,
            filename="time_marching_combined_heatmap.png",
            n_x=256,
            n_t=200
        )
        logger.info(f"    Combined heatmap saved")
    except Exception as e:
        logger.info(f"    Warning: Could not create combined heatmap: {e}")


def _load_prev_window_model(
    ckpt_path: str,
    model_class,
    architecture: List[int],
    activation: str,
    config: Dict,
    prev_window: TimeWindow,
    device: torch.device,
) -> nn.Module:
    """Frozen stand-in for the previous window's trained model (debug mode).

    Used when ``time_marching.debug_windows`` starts at window k > 0: the
    checkpoint supplies window k-1's terminal prediction for the IC override
    and the IC-face target re-minting, exactly like an in-run predecessor.

    Accepts either a root checkpoint (base weights only — the prediction is
    the base forward) or a full AToE-Leaves checkpoint (``adaptive_state``
    with experts, e.g. best_model_phase3.pt / best_model_fine_tune.pt — the
    prediction is the soft PoU composition).
    """
    from trainer.setup import _load_pretrained_base, _load_pretrained_experts

    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(
            f"time_marching.last_window_checkpoint not found: {ckpt_path}")

    ckpt = torch.load(p, map_location='cpu', weights_only=False)

    # Construct the stand-in from the CHECKPOINT's own stored config when
    # available. The composition geometry (fixed sigma_fraction collars,
    # window smoothness/type) is baked in at construction from the config —
    # NOT from the weights — so building with the current plan's config
    # would blend the loaded experts with different overlap widths than
    # they were trained/reconciled under and silently predict differently
    # (measured: a sigma 0.05-trained fine_tune checkpoint rebuilt at
    # sigma 0.2 degraded from ~1e-6 to 2.9e-4 at the handoff slice).
    _ckpt_cfg = ckpt.get('config') if isinstance(ckpt, dict) else None
    if _ckpt_cfg:
        prev_cfg = copy.deepcopy(_ckpt_cfg)
        architecture = prev_cfg.get('base_architecture', architecture)
        activation = prev_cfg.get('activation', activation)
        logger.info("  [Debug] Stand-in built from the checkpoint's stored "
                    "config (composition geometry matches its training run).")
    else:
        prev_cfg = narrow_config_for_window(config, prev_window,
                                            prev_model=None)
        logger.warning("  [Debug] Checkpoint has no stored config — building "
                       "stand-in from the CURRENT plan config; if collar/"
                       "window settings changed since that run, its blended "
                       "prediction will differ from the original.")

    # Constructing + loading under the run's precision: modules created
    # during load (load_state_dict_extended recreates experts) must already
    # be float64, or load_state_dict would silently round the checkpoint
    # through float32. In-run this is a no-op (run_training already set the
    # default dtype); it matters when this helper is used standalone.
    _prev_dtype = torch.get_default_dtype()
    if prev_cfg.get('precision', config.get('precision', 'float32')) == 'float64':
        torch.set_default_dtype(torch.float64)
    try:
        model = model_class(
            architecture, activation, prev_cfg, prev_cfg['adaptive_pinn'],
            experts_architecture=resolve_experts_architecture(prev_cfg),
        )
        astate = ckpt.get('adaptive_state') if isinstance(ckpt, dict) else None
        if astate and astate.get('experts'):
            _load_pretrained_experts(model, str(p), prev_cfg)
            kind = (f"full AToE ({len(astate['experts'])} experts, "
                    f"soft-blend prediction)")
        else:
            _load_pretrained_base(model, str(p), prev_cfg)
            kind = "root (base-only prediction)"
    finally:
        torch.set_default_dtype(_prev_dtype)

    model = model.to(device)
    for q in model.parameters():
        q.requires_grad = False
    model.eval()
    logger.info(f"  [Debug] last_window_checkpoint loaded as window "
                f"{prev_window.idx} stand-in: {kind} — {ckpt_path}")
    return model


def train_with_time_marching(
    model_class,
    architecture: List[int],
    activation: str,
    config: Dict,
    adaptive_cfg: Dict,
    run_dir: Path,
    device: torch.device,
) -> Tuple[nn.Module, Path]:
    """
    Train separate AToE models for each time window, then combine.
    
    This orchestrator:
    1. Computes time windows with M allocation
    2. For each window:
       - Narrows config (temporal_domain, M_experts_num)
       - Resolves pretrained_base_checkpoint / pretrained_local_expert_checkpoint
         per window (a directory value maps to its window_{idx}.pt — see
         resolve_window_pretrained_checkpoint)
       - Generates datasets for narrowed domain
       - If not first window: overrides IC h_gt with prev_model predictions
       - Creates fresh model
       - Calls existing train() as black box
       - Collects the window's best_model_<segment>.pt files into
         run_dir/checkpoints/<segment>/window_{idx}_best_model_<segment>.pt
         (+ manifest.json per segment folder) — copy ONE segment folder to
         roots_checkpoints/<name>/ to reuse that stage for all windows via
         the pretrained checkpoint keys in a later run
       - Optionally freezes model
    3. Wraps all window models in TimeMarchingModel
    
    Args:
        model_class: Class to instantiate (AToE, ANT, or AToELeaves)
        architecture: Base architecture
        activation: Activation function name
        config: Full configuration dictionary
        adaptive_cfg: Adaptive PINN config section
        run_dir: Output directory for this run
        device: CUDA device
    
    Returns:
        Tuple of (combined_model, best_checkpoint_path)
    """
    from trainer.trainer import train
    from utils.dataset_gen import generate_and_save_datasets
    from models.time_marching_model import TimeMarchingModel
    
    problem = config['problem']
    tm_cfg = config[problem]['time_marching']
    global_M = config['adaptive_pinn']['M_experts_num']
    
    # Compute time windows with M allocation
    windows = compute_time_windows(
        config[problem]['temporal_domain'],
        tm_cfg['num_windows'],
        global_M,
        tm_cfg['m_distribution']
    )
    
    # Log M distribution
    logger.info(f"\n{'='*60}")
    logger.info(f"  TIME MARCHING: {len(windows)} windows, global_M={global_M}")
    logger.info(f"  Distribution ({tm_cfg['m_distribution']}): {[w.M for w in windows]}")
    logger.info(f"  Temporal ranges:")
    for w in windows:
        logger.info(f"    Window {w.idx}: t in [{w.t_start:.4f}, {w.t_end:.4f}], M={w.M}")
    logger.info(f"{'='*60}")
    
    # Per-RUN checkpoint collection: run_dir/checkpoints/<segment>/ gathers
    # every window's best_model_<segment>.pt (root / phase3 / fine_tune), so
    # ONE folder holds the same training stage for ALL windows. Copy a
    # segment folder to roots_checkpoints/<name>/ and point
    # pretrained_base_checkpoint (roots) or pretrained_local_expert_checkpoint
    # (full leaf-expert models) at it — each window of the new run then
    # loads its own file for that segment (resolve_window_pretrained_checkpoint).
    run_ckpt_root = run_dir / 'checkpoints'
    manifest_base = {
        'problem': problem,
        'temporal_domain': list(config[problem]['temporal_domain']),
        'num_windows': len(windows),
        'm_distribution': tm_cfg['m_distribution'],
        'global_M': global_M,
    }
    segment_manifests: Dict[str, Dict] = {}
    window_segment_paths: List[Dict[str, Path]] = []

    # ── Debug window selection (time_marching.debug_windows) ──────────────
    # Run only the listed window indices (contiguous, e.g. [1] or [1, 2]);
    # every other window is skipped outright — no datasets, no training, no
    # checkpoint collection. When the first executed window is > 0, its
    # predecessor's terminal prediction must be supplied via
    # time_marching.last_window_checkpoint (root OR full-AToE checkpoint;
    # see _load_prev_window_model).
    debug_windows = tm_cfg.get('debug_windows') or None
    if debug_windows is not None:
        debug_windows = sorted({int(i) for i in debug_windows})
        _valid_idx = {w.idx for w in windows}
        _bad = [i for i in debug_windows if i not in _valid_idx]
        if _bad:
            raise ValueError(
                f"time_marching.debug_windows contains invalid indices "
                f"{_bad} (valid: 0..{len(windows) - 1}).")
        if any(b - a != 1 for a, b in zip(debug_windows, debug_windows[1:])):
            raise ValueError(
                f"time_marching.debug_windows must be contiguous (got "
                f"{debug_windows}): a later window's IC comes from its "
                f"immediate predecessor, and only the FIRST executed window "
                f"may take it from last_window_checkpoint.")
        logger.info(f"  [Debug] debug_windows active: running only "
                    f"{debug_windows} of {[w.idx for w in windows]}")

    window_models: List[Tuple[TimeWindow, nn.Module]] = []
    prev_model = None
    last_checkpoint_path = None

    _lwc = tm_cfg.get('last_window_checkpoint')
    if debug_windows is not None and debug_windows[0] > 0:
        if not _lwc:
            raise ValueError(
                f"time_marching.debug_windows starts at window "
                f"{debug_windows[0]} (> 0): set "
                f"time_marching.last_window_checkpoint to a window-"
                f"{debug_windows[0] - 1} checkpoint (best_model_root.pt for "
                f"a root IC, or best_model_phase3/fine_tune.pt for a full "
                f"AToE IC) so the window has an IC source.")
        prev_model = _load_prev_window_model(
            _lwc, model_class, architecture, activation, config,
            windows[debug_windows[0] - 1], device)
    elif _lwc:
        logger.info("  [Debug] last_window_checkpoint set but not needed "
                    "(first executed window is 0) — ignored.")

    for window in windows:
        if debug_windows is not None and window.idx not in debug_windows:
            logger.info(f"\n  [Debug] Skipping window {window.idx} "
                        f"(not in debug_windows={debug_windows})")
            continue
        logger.info(f"\n{'='*60}")
        logger.info(f"  WINDOW {window.idx + 1}/{len(windows)}: t in [{window.t_start:.4f}, {window.t_end:.4f}]")
        logger.info(f"  M_experts_num = {window.M}")
        logger.info(f"{'='*60}")
        
        # 1. Narrow config for this window (pass prev_model for IC override during resampling)
        window_cfg = narrow_config_for_window(config, window, prev_model=prev_model)
        window_run_dir = run_dir / f"window_{window.idx}"
        window_run_dir.mkdir(parents=True, exist_ok=True)

        # 1b. Resolve pretrained checkpoints for this window: a directory
        # value maps to its window_{idx}.pt; a single file is reused as-is.
        # train() then applies the standard skip logic (base → skip Phase 1,
        # experts → fine-tune only) within the window.
        for _ckpt_key in ('pretrained_base_checkpoint',
                          'pretrained_local_expert_checkpoint'):
            _resolved = resolve_window_pretrained_checkpoint(
                window_cfg[problem].get(_ckpt_key), window, len(windows))
            if _resolved is not None:
                logger.info(f"  [Time Marching] {_ckpt_key} for window "
                            f"{window.idx}: {_resolved}")
            window_cfg[problem][_ckpt_key] = _resolved
        
        # 2. Generate datasets with narrowed domain
        logger.info(f"\n  Generating datasets for window {window.idx}...")
        # force=True: each window needs its OWN draw over its narrowed
        # t-range. The skip-if-exists default silently reused window 0's
        # file for every window (each window then saw only the ~1/N of the
        # points its domain filter kept — and with window-restricted
        # generation, later windows would keep none at all).
        generate_and_save_datasets(window_cfg, force=True)
        
        # NOTE: IC override for windows 1+ is now handled in-memory by trainer.py
        # (_override_ic_for_time_marching) which runs before filtering and after each resample.
        # This avoids corrupting the disk dataset if a previous window diverged with NaN.
        
        # 3. Create fresh model for this window
        logger.info(f"\n  Creating model for window {window.idx}...")
        window_model = model_class(
            architecture, activation, window_cfg, window_cfg['adaptive_pinn'],
            experts_architecture=resolve_experts_architecture(window_cfg),
        )
        window_model = window_model.to(device)
        
        # Convert to double precision if configured
        precision = window_cfg.get('precision', 'float32')
        if precision == 'float64':
            window_model = window_model.double()

        logger.info(f"  {type(window_model).__name__} created")

        # Optional warm start (time_marching.warm_start_from_previous):
        # initialize this window's BASE from the previous window's trained
        # base — the literature-standard transfer init for sequential
        # windows. The IC override already supplies the previous prediction
        # as DATA; this additionally starts the optimizer near it. Base
        # weights only (experts, if any, still spawn/train normally).
        if (tm_cfg.get('warm_start_from_previous', False)
                and prev_model is not None):
            _prev_base = getattr(prev_model, 'base_model', prev_model)
            _new_base = getattr(window_model, 'base_model', window_model)
            try:
                _new_base.load_state_dict(_prev_base.state_dict())
                logger.info(f"  [WarmStart] Window {window.idx} base "
                            f"initialized from the previous window's weights.")
            except Exception as _ws_err:  # noqa: BLE001
                logger.warning(f"  [WarmStart] SKIPPED — incompatible base "
                               f"state dict: {_ws_err}")
        
        # 4. Build loss function for this window
        loss_module = importlib.import_module(f"losses.{problem}_loss")
        loss_fn = loss_module.build_loss(**window_cfg)
        
        # 5. Call existing train() as black box
        logger.info(f"\n  Training window {window.idx}...")
        train_data_path = f"datasets/{problem}/training_data.pt"

        # Mirrors this window's slice of the run-wide log into its own
        # window_<idx>/training_logs.log, live, the same way window_<idx>/
        # metrics.json is already per-window and incrementally written.
        _window_log_handler = add_window_log_handler(window_run_dir)
        try:
            checkpoint_path = train(
                model=window_model,
                loss_fn=loss_fn,
                train_data_path=train_data_path,
                cfg=window_cfg,
                run_dir=window_run_dir,
            )
        finally:
            remove_window_log_handler(_window_log_handler)

        # 6. Collect this window's per-segment bests into the run-level
        # checkpoints/<segment>/ folders. The files come straight from
        # _save_checkpoint, so they carry adaptive_state and are consumable
        # by BOTH pretrained flows (_load_pretrained_base takes just the
        # base, _load_pretrained_experts the full leaf-expert state).
        collected: Dict[str, Path] = {}
        for src in sorted((window_run_dir / 'checkpoints').glob('best_model_*.pt')):
            segment = src.stem[len('best_model_'):]
            seg_dir = run_ckpt_root / segment
            seg_dir.mkdir(parents=True, exist_ok=True)
            dst = seg_dir / f"window_{window.idx}_{src.stem}.pt"
            shutil.copy2(src, dst)
            collected[segment] = dst
            manifest = segment_manifests.setdefault(
                segment, {**manifest_base, 'segment': segment, 'windows': []})
            manifest['windows'].append({
                'idx': window.idx,
                't_start': window.t_start,
                't_end': window.t_end,
                'M': window.M,
                'file': dst.name,
            })
            # Rewrite after every window so a partial run still leaves a
            # valid (truncated) segment folder behind.
            with open(seg_dir / 'manifest.json', 'w') as f:
                json.dump(manifest, f, indent=2)
        if collected:
            logger.info(f"  Window {window.idx} segment checkpoints collected "
                        f"into {run_ckpt_root}: {sorted(collected)}")
        else:
            logger.info(f"  Warning: no best_model_*.pt found for window "
                        f"{window.idx} in {window_run_dir / 'checkpoints'}")
        window_segment_paths.append(collected)

        # 6b. Live mirror for mid-run downloads: copy the COMPLETED window's
        # folder (and a snapshot of the run-level log + collected
        # checkpoints) into <_experiment_live_dir>/<run>_partial/ so the
        # standard download flow, which tars the latest outputs/experiments
        # folder, sees finished windows while later windows still train.
        # Removed at run end (the real run dir is moved in then). Never
        # allowed to break training.
        _live_root = config.get('_experiment_live_dir')
        if _live_root:
            try:
                _mirror = Path(_live_root) / f"{run_dir.name}_partial"
                _mirror.mkdir(parents=True, exist_ok=True)
                _wdst = _mirror / f"window_{window.idx}"
                if _wdst.exists():
                    shutil.rmtree(_wdst)
                shutil.copytree(window_run_dir, _wdst)
                shutil.copytree(run_ckpt_root, _mirror / 'checkpoints',
                                dirs_exist_ok=True)
                _log_src = run_dir / 'training_logs.log'
                if _log_src.exists():
                    shutil.copy2(_log_src, _mirror / 'training_logs.log')
                logger.info(f"  [LiveMirror] window {window.idx} copied to "
                            f"{_mirror}")
            except Exception as _lm_err:                       # noqa: BLE001
                logger.warning(f"  [LiveMirror] copy failed (non-fatal): "
                               f"{_lm_err}")

        # 7. Optionally freeze for memory savings
        if tm_cfg['freeze_previous_windows']:
            logger.info(f"  Freezing window {window.idx} model parameters")
            for p in window_model.parameters():
                p.requires_grad = False
            window_model.eval()
        
        window_models.append((window, window_model))
        prev_model = window_model
        last_checkpoint_path = checkpoint_path

    # ── Partial debug run: skip the full-domain composition stages ────────
    # The combined model, full-domain rel-L2 and combined plots all assume
    # every window trained; with a debug subset they would be undefined (or
    # crash on missing window_i folders). Per-window outputs and the
    # run-level checkpoints/<segment>/ collection above are already written.
    _all_idx = [w.idx for w in windows]
    if debug_windows is not None and debug_windows != _all_idx:
        logger.info(f"\n{'='*60}")
        logger.info(f"  [Debug] Ran windows {debug_windows} of {_all_idx} — "
                    f"skipping combined model checkpoint, full-domain rel-L2 "
                    f"and combined plots (undefined for a partial run).")
        logger.info(f"{'='*60}")
        combined_model = TimeMarchingModel(window_models)
        import json as _json
        final_metrics = {
            'debug_windows': debug_windows,
            'full_domain_rel_l2': None,
            'num_windows_total': len(windows),
            'num_windows_run': len(window_models),
            'm_per_window_run': [w.M for w, _ in window_models],
            'problem': config['problem'],
        }
        with open(run_dir / 'time_marching_final_metrics.json', 'w') as _f:
            _json.dump(final_metrics, _f, indent=2)
        _live_root = config.get('_experiment_live_dir')
        if _live_root:
            try:
                _mirror = Path(_live_root) / f"{run_dir.name}_partial"
                if _mirror.exists():
                    shutil.rmtree(_mirror)
                    logger.info(f"  [LiveMirror] removed {_mirror} "
                                f"(run complete)")
            except Exception as _lm_err:                        # noqa: BLE001
                logger.warning(f"  [LiveMirror] cleanup failed (non-fatal): "
                               f"{_lm_err}")
        logger.info(f"\n{'='*60}")
        logger.info(f"  Time marching training complete (debug subset)!")
        logger.info(f"{'='*60}")
        return combined_model, last_checkpoint_path

    # 9. Combine into TimeMarchingModel
    logger.info(f"\n{'='*60}")
    logger.info(f"  Creating combined TimeMarchingModel")
    logger.info(f"{'='*60}")
    combined_model = TimeMarchingModel(window_models)
    
    # 10. Save combined model checkpoint. Each window is represented by its
    # latest collected segment (the reconciled best of the last stage run).
    _segment_priority = ('fine_tune', 'phase3', 'root', 'main')

    def _final_segment_file(seg_paths: Dict[str, Path]):
        for seg in _segment_priority:
            if seg in seg_paths:
                return str(seg_paths[seg])
        return None

    combined_checkpoint = {
        'is_time_marching': True,
        'num_windows': len(windows),
        'windows': [
            {'idx': w.idx, 't_start': w.t_start, 't_end': w.t_end, 'M': w.M}
            for w, _ in window_models
        ],
        'window_checkpoints': [
            _final_segment_file(seg_paths)
            for seg_paths in window_segment_paths
        ],
    }
    combined_checkpoint_path = run_dir / "time_marching_combined.pt"
    torch.save(combined_checkpoint, combined_checkpoint_path)
    logger.info(f"  Combined checkpoint saved: {combined_checkpoint_path}")
    
    # 11. Compute full-domain rel-L2 using the combined model
    logger.info(f"\n{'='*60}")
    logger.info(f"  Computing full-domain rel-L2...")
    logger.info(f"{'='*60}")
    final_rel_l2 = None
    try:
        final_rel_l2 = _compute_full_domain_rel_l2(combined_model, config, device)
        logger.info(f"  Full-domain Rel-L2: {final_rel_l2:.6e}")
    except Exception as e:
        logger.info(f"  Warning: Could not compute full-domain rel-L2: {e}")

    # Save final metrics file
    import json as _json
    final_metrics = {
        'full_domain_rel_l2': final_rel_l2,
        'num_windows': len(windows),
        'm_per_window': [w.M for w in windows],
        'total_m': sum(w.M for w in windows),
        'problem': config['problem'],
    }
    final_metrics_path = run_dir / 'time_marching_final_metrics.json'
    with open(final_metrics_path, 'w') as _f:
        _json.dump(final_metrics, _f, indent=2)
    logger.info(f"  Final metrics saved to {final_metrics_path}")

    # 12. Create combined visualizations
    logger.info(f"\n{'='*60}")
    logger.info(f"  Generating time marching visualizations")
    logger.info(f"{'='*60}")

    # Plot combined loss curves from all windows (with full-domain rel-L2 marker)
    _plot_combined_loss_curves(windows, run_dir, final_rel_l2=final_rel_l2)

    # Plot combined prediction heatmap vs ground truth
    _plot_combined_heatmap(combined_model, config, run_dir, device)
    
    # Remove the live mirror: the run completed, so run_experiments moves the
    # REAL run dir into the experiments folder right after this returns — the
    # _partial copy would only duplicate it.
    _live_root = config.get('_experiment_live_dir')
    if _live_root:
        try:
            _mirror = Path(_live_root) / f"{run_dir.name}_partial"
            if _mirror.exists():
                shutil.rmtree(_mirror)
                logger.info(f"  [LiveMirror] removed {_mirror} (run complete)")
        except Exception as _lm_err:                            # noqa: BLE001
            logger.warning(f"  [LiveMirror] cleanup failed (non-fatal): "
                           f"{_lm_err}")

    logger.info(f"\n{'='*60}")
    logger.info(f"  Time marching training complete!")
    logger.info(f"{'='*60}")

    return combined_model, last_checkpoint_path
