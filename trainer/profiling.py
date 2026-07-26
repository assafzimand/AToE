"""Bounded torch.profiler window inside a training segment.

Profiling a full 100k-epoch segment is neither affordable nor useful. This
records a short window once the segment has reached steady state, writes a
RELATIVE breakdown (percent of device/CPU time per op) next to metrics.json,
and then disables itself for the remainder of the segment.

Design constraints, in priority order:

1. It must not change results. torch.profiler is an observer: it touches no
   RNG and no tensor math, so the trained weights are bit-identical with it
   on or off.
2. It must not kill the run. CUPTI (the CUDA profiling layer) can fail to
   initialise on some driver/toolkit combinations. Every entry point here is
   wrapped: the first failure logs once, permanently disables profiling, and
   training continues untouched.
3. It must be cheap. Only ``warmup + active`` epochs are recorded out of the
   whole segment, and ``profile_memory``/``with_stack`` stay off by default —
   those are the settings that add real overhead and can balloon a trace to
   gigabytes.

Absolute times inside the window are inflated by the profiler itself, which
is why the report leads with percentages: use the relative split ("QR is 40%
of the step"), not the raw milliseconds.

Config (top-level ``profiling`` block, all optional)::

    profiling:
      enabled: false
      segments: [root, phase3, fine_tune]   # or 'all'
      start_epoch: 2000     # epochs INTO the segment before recording
      warmup: 5             # discarded steps (graph caching, cuBLAS autotune)
      active: 50            # recorded steps
      top_k: 30             # rows in the report
      record_shapes: false
      emit_trace: false     # chrome://tracing json (large)
"""

from pathlib import Path
from typing import Dict, Optional

import torch

from utils.logging_config import get_logger

logger = get_logger(__name__)

_DEFAULTS = {
    'enabled': False,
    'segments': 'all',
    'start_epoch': 2000,
    'warmup': 5,
    'active': 50,
    'top_k': 30,
    'record_shapes': False,
    'emit_trace': False,
}


def _cfg(cfg: Dict) -> Dict:
    block = (cfg.get('profiling') or {}) if isinstance(cfg, dict) else {}
    out = dict(_DEFAULTS)
    out.update({k: v for k, v in block.items() if k in _DEFAULTS})
    return out


class SegmentProfiler:
    """One bounded profiling window for one training segment.

    ``step(epoch)`` is called once per epoch; everything else is internal.
    A disabled or finished profiler is a no-op costing one attribute check.
    """

    def __init__(self, cfg: Dict, segment_name: str,
                 segment_start_epoch: int, run_dir) -> None:
        c = _cfg(cfg)
        segs = c['segments']
        wanted = (segs == 'all' or segs is None
                  or (isinstance(segs, (list, tuple)) and segment_name in segs)
                  or segs == segment_name)
        self._enabled = bool(c['enabled']) and wanted
        self._segment = segment_name
        self._seg_start = segment_start_epoch
        self._run_dir = Path(run_dir) if run_dir is not None else None
        self._start_epoch = max(0, int(c['start_epoch']))
        self._warmup = max(0, int(c['warmup']))
        self._active = max(1, int(c['active']))
        self._top_k = max(1, int(c['top_k']))
        self._record_shapes = bool(c['record_shapes'])
        self._emit_trace = bool(c['emit_trace'])
        self._prof = None
        self._steps = 0
        self._done = not self._enabled

        if self._enabled:
            logger.info(
                f"  [Profiler] armed for segment '{segment_name}': "
                f"{self._warmup} warmup + {self._active} recorded steps "
                f"starting {self._start_epoch} epochs into the segment")

    # ── public ────────────────────────────────────────────────────────────
    def step(self, epoch: int) -> None:
        """Advance the window. Safe to call every epoch, always."""
        if self._done:
            return
        try:
            if self._prof is None:
                if (epoch - self._seg_start) < self._start_epoch:
                    return
                self._begin()
                return          # this epoch's work is already done; record from the next
            self._prof.step()
            self._steps += 1
            if self._steps >= self._warmup + self._active:
                self._finish()
        except Exception as exc:                                # noqa: BLE001
            self._abort(exc)

    def close(self) -> None:
        """Stop and report if the segment ended mid-window."""
        if self._done or self._prof is None:
            self._done = True
            return
        try:
            self._finish()
        except Exception as exc:                                # noqa: BLE001
            self._abort(exc)

    # ── internals ─────────────────────────────────────────────────────────
    def _begin(self) -> None:
        acts = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        self._prof = torch.profiler.profile(
            activities=acts,
            schedule=torch.profiler.schedule(
                wait=0, warmup=self._warmup, active=self._active, repeat=1),
            record_shapes=self._record_shapes,
            profile_memory=False,
            with_stack=False,
        )
        self._prof.start()
        self._steps = 0
        logger.info(f"  [Profiler] recording {self._segment} from epoch "
                    f"{self._seg_start + self._start_epoch + 1} "
                    f"(activities={[a.name for a in acts]})")

    def _finish(self) -> None:
        prof, self._prof = self._prof, None
        self._done = True
        prof.stop()
        self._write_report(prof)

    def _abort(self, exc: Exception) -> None:
        logger.warning(
            f"  [Profiler] disabled for the rest of this run after an error "
            f"({type(exc).__name__}: {exc}). Training is unaffected.")
        try:
            if self._prof is not None:
                self._prof.stop()
        except Exception:                                       # noqa: BLE001
            pass
        self._prof = None
        self._done = True

    @staticmethod
    def _total(evt, kind: str) -> float:
        """Self time in us, tolerating the 2.x device/cuda attribute rename."""
        for name in (f'self_{kind}_time_total', f'self_cuda_time_total'):
            v = getattr(evt, name, None)
            if v is not None:
                return float(v)
        return 0.0

    def _write_report(self, prof) -> None:
        """Dump PyTorch's own key_averages table.

        The percentage columns (Self CUDA % / Self CPU %) come straight from
        torch.profiler rather than from arithmetic here, so the report is the
        canonical format and there is no custom measurement code in the path.
        """
        if self._run_dir is None:
            return
        ka = prof.key_averages()
        on_cuda = torch.cuda.is_available()
        label = 'CUDA' if on_cuda else 'CPU'
        sort_by = 'self_device_time_total' if on_cuda else 'self_cpu_time_total'

        path = self._run_dir / f"profile_{self._segment}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"Profile - segment '{self._segment}'\n")
            f.write(f"  window : {self._active} recorded steps "
                    f"({self._warmup} warmup) starting "
                    f"{self._start_epoch} epochs into the segment\n")
            f.write(f"  device : {label}   (sorted by {sort_by})\n\n")
            f.write("Read the PERCENT columns. Absolute times inside a profiled\n"
                    "window are inflated by the profiler itself.\n\n")
            f.write(ka.table(sort_by=sort_by, row_limit=self._top_k))
            f.write("\n")

        kind = 'device' if on_cuda else 'cpu'
        rows = sorted(((e.key, self._total(e, kind)) for e in ka),
                      key=lambda r: r[1], reverse=True)
        total = sum(r[1] for r in rows) or 1.0
        top = ", ".join(f"{k} {100.0 * d / total:.0f}%" for k, d in rows[:3])
        logger.info(f"  [Profiler] {self._segment}: wrote {path.name} "
                    f"({len(rows)} ops, {total / 1e3 / self._active:.3f} ms/step "
                    f"self-{label} in-window). Top: {top}")

        if self._emit_trace:
            try:
                tpath = self._run_dir / f"profile_{self._segment}_trace.json"
                prof.export_chrome_trace(str(tpath))
                logger.info(f"  [Profiler] chrome trace -> {tpath.name}")
            except Exception as exc:                            # noqa: BLE001
                logger.warning(f"  [Profiler] trace export failed: {exc}")


def make_segment_profiler(cfg: Dict, segment_name: str,
                          segment_start_epoch: int,
                          run_dir) -> Optional['SegmentProfiler']:
    """Build a profiler, never raising — returns a disabled one on error."""
    try:
        return SegmentProfiler(cfg, segment_name, segment_start_epoch, run_dir)
    except Exception as exc:                                    # noqa: BLE001
        logger.warning(f"  [Profiler] could not be created ({exc}); "
                       f"continuing without profiling.")
        return None
