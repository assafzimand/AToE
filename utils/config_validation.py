"""Config validation for per-problem required fields.

All features that were previously global defaults are now required to be
explicitly specified in each problem section. This ensures full transparency
of what actually ran from the config file alone.
"""

from typing import Dict, Any


# Required per-problem feature keys (no global defaults allowed)
REQUIRED_PROBLEM_FEATURES = [
    'rwf',
    'fourier_features',
    'init',
    'lra',
    'adaptive_sampling',
    'grad_clip_norm',
    'expert_grad_clip_norm',
]

# Required nested keys within each feature
REQUIRED_NESTED_KEYS = {
    'rwf': ['enabled'],
    'fourier_features': ['enabled', 'dim', 'scale', 'periodic'],
    'init': ['hidden', 'output', 'ls_use_bias'],
    'lra': ['enabled', 'update_every', 'alpha'],
    'adaptive_sampling': ['enabled', 'adaptive_ratio'],
}


def validate_problem_config(cfg: Dict[str, Any]) -> None:
    """Validate that the problem section has all required feature keys.

    Raises:
        ValueError: If any required keys are missing from the problem section.
    """
    problem = cfg.get('problem')
    if not problem:
        raise ValueError("Config missing 'problem' key")

    problem_cfg = cfg.get(problem)
    if not problem_cfg:
        raise ValueError(f"Config missing problem section for '{problem}'")

    missing = []
    nested_missing = []

    for key in REQUIRED_PROBLEM_FEATURES:
        if key not in problem_cfg:
            missing.append(key)
        elif key in REQUIRED_NESTED_KEYS:
            nested = problem_cfg[key]
            if not isinstance(nested, dict):
                nested_missing.append(
                    f"{key} (expected dict, got {type(nested).__name__})")
            else:
                for nested_key in REQUIRED_NESTED_KEYS[key]:
                    if nested_key not in nested:
                        nested_missing.append(f"{key}.{nested_key}")

    if missing or nested_missing:
        error_parts = []
        if missing:
            error_parts.append(f"Missing top-level keys: {missing}")
        if nested_missing:
            error_parts.append(f"Missing nested keys: {nested_missing}")

        raise ValueError(
            f"Problem '{problem}' config is missing required feature keys.\n"
            f"{chr(10).join(error_parts)}\n\n"
            f"All per-problem features must be explicitly specified.\n"
            f"Required keys: {REQUIRED_PROBLEM_FEATURES}\n"
            f"Add missing keys to '{problem}' section in your "
            f"experiments_plan.yaml."
        )


# Config keys removed by the staged-training refactor (error if still present).
REMOVED_ADAPTIVE_KEYS = ['freeze_mode', 'freeze_epochs_after_spawn']


def validate_adaptive_staged_config(cfg: Dict[str, Any]) -> None:
    """Validate the adaptive-PINN training config surface.

      * ``adaptive_pinn.initial_train`` (with ``epochs``) is required for the
        root segment unless a ``pretrained_base_checkpoint`` is supplied.
      * Removed keys (``freeze_mode``, ``freeze_epochs_after_spawn``) raise.

    Optional AToE-Leaves keys:
      * ``loss_weights.continuity`` (float, default 1.0): weight for the
        neighbor-continuity loss term in split_icbc training.

    Raises:
        ValueError: on a missing required key or a present removed key.
    """
    adaptive_cfg = cfg.get('adaptive_pinn', {})
    if not adaptive_cfg or not adaptive_cfg.get('enabled', False):
        return

    errors = []

    for k in REMOVED_ADAPTIVE_KEYS:
        if k in adaptive_cfg:
            errors.append(
                f"adaptive_pinn.{k} was removed by the staged-training refactor "
                f"(staged freezing is now per-level requires_grad). Remove it.")

    problem = cfg.get('problem')
    problem_cfg = cfg.get(problem, {}) if problem else {}
    pretrained = problem_cfg.get('pretrained_base_checkpoint', None)

    # Root segment config (skipped only when a pretrained base is loaded).
    if pretrained is None:
        initial_train = adaptive_cfg.get('initial_train', None)
        if not isinstance(initial_train, dict):
            errors.append(
                "adaptive_pinn.initial_train (dict with 'epochs') is required "
                "for the root segment when pretrained_base_checkpoint is null.")
        elif 'epochs' not in initial_train:
            errors.append("adaptive_pinn.initial_train.epochs is required.")

    if errors:
        raise ValueError(
            "Invalid adaptive_pinn training config:\n  - "
            + "\n  - ".join(errors)
        )


def get_problem_feature(
    cfg: Dict[str, Any],
    feature: str,
    nested_key: str = None
) -> Any:
    """Get a feature value from the problem config section.

    This replaces cfg.get() patterns for moved features. Raises KeyError if
    the feature is not found (no silent defaults).

    Args:
        cfg: Full config dictionary
        feature: Top-level feature name (e.g., 'rwf', 'fourier_features')
        nested_key: Optional nested key within the feature dict

    Returns:
        The feature value

    Raises:
        KeyError: If the feature or nested key is not found
    """
    problem = cfg['problem']
    problem_cfg = cfg[problem]

    if feature not in problem_cfg:
        raise KeyError(
            f"Feature '{feature}' not found in problem '{problem}' config. "
            f"This is a required per-problem field."
        )

    value = problem_cfg[feature]

    if nested_key is not None:
        if not isinstance(value, dict):
            raise KeyError(
                f"Feature '{feature}' in problem '{problem}' is not a dict, "
                f"cannot access nested key '{nested_key}'"
            )
        if nested_key not in value:
            raise KeyError(
                f"Nested key '{nested_key}' not found in "
                f"'{feature}' for problem '{problem}'"
            )
        return value[nested_key]

    return value


def merge_problem_features_to_toplevel(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Copy per-problem features to top-level for backward compatibility.

    This is a transitional helper that copies the per-problem features to
    the top-level config so existing code that reads cfg['rwf'] etc. still
    works. Should be called after validation but before trainer runs.

    Returns:
        Modified config dict with features copied to top-level
    """
    problem = cfg['problem']
    problem_cfg = cfg[problem]

    for feature in REQUIRED_PROBLEM_FEATURES:
        if feature in problem_cfg:
            cfg[feature] = problem_cfg[feature]

    return cfg
