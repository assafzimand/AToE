"""Backwards-compatible facade for the trainer package.

The training implementation lives in focused modules:
  trainer.setup        — config/data preparation, optimizer factories, checkpoint IO
  trainer.epoch_loop   — the per-segment epoch loop
  trainer.split_segment — per-leaf split training segment
  trainer.orchestrator — phase sequencing, tree build, expert spawning
  trainer.finalize     — post-training outputs

``from trainer.trainer import train`` remains the public entry point.
"""

from trainer.orchestrator import train, train_orchestrator
from trainer.setup import _setup_training, _create_dataloader, _save_checkpoint
from trainer.epoch_loop import _train_segment
from trainer.split_segment import _run_split_segment
from trainer.finalize import _finalize_training

__all__ = ['train']
