"""
Epoch-level timing instrumentation for diagnosing performance bottlenecks.

Collects fine-grained sub-step timings during training, organized by epoch,
and saves a timing.json at the end of the run.

Usage:
    timer = EpochTimer()
    for epoch in range(1, epochs+1):
        timer.start_epoch(epoch, num_experts=model.num_experts)
        
        # In training step:
        timer.start("train.loss_fn")
        loss = loss_fn(model, batch)
        timer.stop("train.loss_fn")
        
        timer.end_epoch()
    
    timer.save(run_dir / "timing.json")
"""

import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

from utils.logging_config import get_logger

logger = get_logger(__name__)


class EpochTimer:
    """
    Collects hierarchical per-epoch timing measurements.
    
    Timings are stored per-epoch with metadata (num_experts, num_batches).
    At save time, per-epoch data + summary statistics are written to JSON.
    """
    
    def __init__(self, enabled: bool = True, print_every: int = 0):
        """
        Args:
            enabled: If False, all operations are no-ops (zero overhead).
            print_every: Print epoch timing summary every N epochs (0 = never).
        """
        self.enabled = enabled
        self.print_every = print_every
        
        # Per-epoch records: list of dicts
        self._epochs: List[Dict] = []
        
        # Current epoch state
        self._current_epoch: Optional[int] = None
        self._current_meta: Dict = {}
        self._accumulators: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
        self._active_timers: Dict[str, float] = {}
        self._epoch_start: float = 0.0
    
    def start_epoch(self, epoch: int, **meta) -> None:
        """Begin timing a new epoch. Meta can include num_experts, etc."""
        if not self.enabled:
            return
        self._current_epoch = epoch
        self._current_meta = meta
        self._accumulators = defaultdict(float)
        self._counts = defaultdict(int)
        self._active_timers = {}
        self._epoch_start = time.perf_counter()
    
    def start(self, key: str) -> None:
        """Start timing a named sub-step (e.g., 'train.loss_fn')."""
        if not self.enabled:
            return
        self._active_timers[key] = time.perf_counter()
    
    def stop(self, key: str) -> None:
        """Stop timing a named sub-step and accumulate."""
        if not self.enabled:
            return
        t0 = self._active_timers.pop(key, None)
        if t0 is not None:
            dt = time.perf_counter() - t0
            self._accumulators[key] += dt
            self._counts[key] += 1
    
    def record(self, key: str, dt: float) -> None:
        """Directly record a timing value (for external measurements)."""
        if not self.enabled:
            return
        self._accumulators[key] += dt
        self._counts[key] += 1
    
    def end_epoch(self) -> Optional[Dict]:
        """
        Finalize current epoch, store record, optionally print summary.
        
        Returns:
            The epoch timing record dict, or None if disabled.
        """
        if not self.enabled or self._current_epoch is None:
            return None
        
        epoch_total = time.perf_counter() - self._epoch_start
        
        record = {
            'epoch': self._current_epoch,
            **self._current_meta,
            'total_s': round(epoch_total, 6),
            'steps': {}
        }
        
        for key in sorted(self._accumulators.keys()):
            record['steps'][key] = {
                'total_s': round(self._accumulators[key], 6),
                'count': self._counts[key],
                'avg_s': round(self._accumulators[key] / max(self._counts[key], 1), 6),
            }
        
        self._epochs.append(record)
        
        # Print if requested
        if self.print_every > 0 and self._current_epoch % self.print_every == 0:
            self._print_epoch(record)
        
        self._current_epoch = None
        return record
    
    def _print_epoch(self, record: Dict) -> None:
        """Print a human-readable summary of one epoch's timing."""
        epoch = record['epoch']
        n_exp = record.get('num_experts', '?')
        total = record['total_s']
        
        parts = []
        for key, info in record['steps'].items():
            parts.append(f"{key}={info['total_s']:.4f}s({info['count']}x)")
        
        detail = " | ".join(parts) if parts else "no sub-steps recorded"
        logger.info(f"  [TIMING] Epoch {epoch} (K={n_exp}): {total:.4f}s | {detail}")
    
    def save(self, path: Path) -> None:
        """
        Save all timing data to JSON.
        
        Includes per-epoch records and summary statistics grouped by
        num_experts count for easy analysis of scaling.
        """
        if not self.enabled:
            return
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build summary: group epochs by num_experts
        by_k: Dict[int, List[Dict]] = defaultdict(list)
        for rec in self._epochs:
            k = rec.get('num_experts', 0)
            by_k[k].append(rec)
        
        summary = {}
        for k in sorted(by_k.keys()):
            recs = by_k[k]
            totals = [r['total_s'] for r in recs]
            step_keys = set()
            for r in recs:
                step_keys.update(r['steps'].keys())
            
            step_summary = {}
            for sk in sorted(step_keys):
                vals = [r['steps'][sk]['total_s'] for r in recs if sk in r['steps']]
                if vals:
                    step_summary[sk] = {
                        'mean_s': round(statistics.mean(vals), 6),
                        'median_s': round(statistics.median(vals), 6),
                        'min_s': round(min(vals), 6),
                        'max_s': round(max(vals), 6),
                        'n_epochs': len(vals),
                    }
            
            summary[f'K={k}'] = {
                'num_epochs': len(recs),
                'epoch_total': {
                    'mean_s': round(statistics.mean(totals), 6),
                    'median_s': round(statistics.median(totals), 6),
                    'min_s': round(min(totals), 6),
                    'max_s': round(max(totals), 6),
                },
                'steps': step_summary,
            }
        
        output = {
            'summary_by_num_experts': summary,
            'epochs': self._epochs,
        }
        
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"  Timing data saved to {path} ({len(self._epochs)} epochs)")
    
    def print_summary(self) -> None:
        """Print a compact summary table grouped by num_experts."""
        if not self.enabled or not self._epochs:
            return
        
        # Group by num_experts
        by_k: Dict[int, List[Dict]] = defaultdict(list)
        for rec in self._epochs:
            k = rec.get('num_experts', 0)
            by_k[k].append(rec)
        
        logger.info(f"\n{'='*70}")
        logger.info("TIMING SUMMARY (by num_experts)")
        logger.info(f"{'='*70}")
        
        for k in sorted(by_k.keys()):
            recs = by_k[k]
            totals = [r['total_s'] for r in recs]
            logger.info(f"\n  K={k} ({len(recs)} epochs): "
                  f"mean={statistics.mean(totals):.4f}s  "
                  f"median={statistics.median(totals):.4f}s  "
                  f"min={min(totals):.4f}s  max={max(totals):.4f}s")
            
            # Collect all step keys
            step_keys = set()
            for r in recs:
                step_keys.update(r['steps'].keys())
            
            for sk in sorted(step_keys):
                vals = [r['steps'][sk]['total_s'] for r in recs if sk in r['steps']]
                if vals:
                    logger.info(f"    {sk:40s}: mean={statistics.mean(vals):.4f}s  "
                          f"median={statistics.median(vals):.4f}s")
        
        logger.info(f"{'='*70}\n")
