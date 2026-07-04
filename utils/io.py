"""IO utilities for configuration and directory management."""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import subprocess
import yaml
import logging


def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        path: Path to the config YAML file (default: "config/config.yaml")

    Returns:
        Dictionary containing all configuration parameters
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_git_info() -> Dict[str, Any]:
    """Capture current git branch, commit hash, and dirty-tree status.

    Returns an empty dict if git is unavailable or the cwd is not a repo.
    """
    info: Dict[str, Any] = {}
    try:
        info['git_branch'] = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info['git_dirty'] = len(dirty) > 0
    except Exception:
        pass
    return info


def resolve_experts_architecture(config: Dict[str, Any]) -> List[int]:
    """Expert spawn architecture; defaults to base_architecture when omitted."""
    base = config['base_architecture']
    return config.get('experts_architecture', base)


def architecture_dir_layers_str(
    base_layers: List[int],
    experts_layers: Optional[List[int]] = None,
) -> str:
    """Folder label: base only, or base-{b}-experts-{e} when they differ."""
    base_str = "-".join(map(str, base_layers))
    if experts_layers is not None and list(experts_layers) != list(base_layers):
        exp_str = "-".join(map(str, experts_layers))
        return f"base-{base_str}-experts-{exp_str}"
    return base_str


def log_architectures(
    config: Dict[str, Any],
    logger: Optional[Union[logging.Logger, Any]] = None,
    prefix: str = "  ",
) -> None:
    """Log base vs experts architecture (when they differ)."""
    base = config['base_architecture']
    experts = resolve_experts_architecture(config)
    log_fn = logger.info if logger is not None and hasattr(logger, 'info') else print
    log_fn(f"{prefix}Base architecture: {list(base)}")
    if list(experts) != list(base):
        log_fn(f"{prefix}Experts architecture: {list(experts)}")
    else:
        log_fn(f"{prefix}Experts architecture: (same as base)")


def make_run_dir(
    problem: str,
    layers: list,
    act: str,
    experts_layers: Optional[list] = None,
) -> Path:
    """
    Create a run directory with standardized naming and timestamp.

    Args:
        problem: Problem name (e.g., "schrodinger")
        layers: List of layer sizes for base (e.g., [2, 50, 100, 50, 2])
        act: Activation function name (e.g., "tanh")
        experts_layers: Optional expert architecture; when different from layers,
            folder name includes both base and experts sizes.

    Returns:
        Path object to the created run directory
        Structure: outputs/<problem>_layers-<...>_act-<act>/<timestamp>/
    """
    layers_str = architecture_dir_layers_str(layers, experts_layers)

    # Create architecture directory name: <problem>-<...>-<activation>
    arch_dir_name = f"{problem}-{layers_str}-{act}"
    
    # Create timestamp for unique run: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the full path: outputs/<architecture>/<timestamp>/
    arch_dir = Path("outputs") / arch_dir_name
    run_dir = arch_dir / timestamp

    # Create the directory and subdirectories
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_plots").mkdir(exist_ok=True)
    (run_dir / "ncc_plots").mkdir(exist_ok=True)

    return run_dir
