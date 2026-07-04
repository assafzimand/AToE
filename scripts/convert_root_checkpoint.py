"""One-time conversion of legacy root checkpoints to plain state-dict files.

Legacy checkpoints were saved with the full training payload (optimizer state,
metrics, pickled config objects referencing old module paths), which makes them
huge and forces torch.load(weights_only=False). This script extracts only the
base network weights plus the minimal metadata `_load_pretrained_base` needs:

    {'model_state_dict': <plain FCNet state dict>,
     'config': {'base_architecture': [...], 'activation': '...'}}

Usage:
    python scripts/convert_root_checkpoint.py roots_checkpoints/burgers_root.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def convert(path: Path) -> None:
    print(f"Loading {path} ...")
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    # Prefer the adaptive_state base (plain FCNet keys); fall back to a
    # model_state_dict, stripping a base_model. prefix if present.
    adaptive_state = ckpt.get('adaptive_state')
    if adaptive_state and 'base_model' in adaptive_state:
        base_sd = adaptive_state['base_model']
        base_arch = adaptive_state.get('base_architecture')
        activation = adaptive_state.get('activation')
    elif 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
        if any(k.startswith('base_model.') for k in sd):
            base_sd = {k[len('base_model.'):]: v for k, v in sd.items()
                       if k.startswith('base_model.')}
        else:
            base_sd = sd
        cfg = ckpt.get('config') or {}
        base_arch = cfg.get('base_architecture')
        activation = cfg.get('activation')
    else:
        raise ValueError(
            "Checkpoint has neither adaptive_state.base_model nor model_state_dict")

    base_sd = {k: v.clone() for k, v in base_sd.items()}
    n_params = sum(v.numel() for v in base_sd.values())
    print(f"  Extracted base state dict: {len(base_sd)} tensors, {n_params:,} params")
    print(f"  base_architecture={base_arch}, activation={activation}")

    out = {
        'model_state_dict': base_sd,
        'config': {
            'base_architecture': list(base_arch) if base_arch else None,
            'activation': activation,
        },
    }

    backup = path.with_suffix('.pt.legacy')
    path.rename(backup)
    torch.save(out, path)
    print(f"  Saved plain checkpoint to {path} "
          f"({path.stat().st_size / 1024:.0f} KB; legacy kept at {backup.name})")

    # Verify the converted file loads with the safe default.
    reloaded = torch.load(path, map_location='cpu', weights_only=True)
    assert set(reloaded['model_state_dict']) == set(base_sd)
    for k in base_sd:
        assert torch.equal(reloaded['model_state_dict'][k], base_sd[k])
    print("  Verified: weights_only load OK, tensors identical.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        convert(Path(arg))
