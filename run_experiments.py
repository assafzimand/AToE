"""Automated experiment runner.

Reads experiments_plan.yaml, runs each experiment as a training subprocess,
collects the run outputs under one experiment directory, and generates the
cross-experiment comparison report.
"""

import yaml
import shutil
from pathlib import Path
from datetime import datetime
import subprocess
import sys

# Force UTF-8 stdout/stderr so unicode in logs doesn't crash on non-UTF-8
# Windows consoles (e.g. cp1255).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from utils.logging_config import setup_logging, get_logger, update_log_file
from utils.io import architecture_dir_layers_str, log_architectures, resolve_experts_architecture

logger = get_logger(__name__)


def load_experiment_plan(plan_path="experiments_plan.yaml"):
    """Load experiment plan from YAML file."""
    with open(plan_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_experiment_dir(plan):
    """Create parent experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = plan['experiment_name']
    parent_dir = Path("outputs") / "experiments" / f"{exp_name}_{timestamp}"
    parent_dir.mkdir(parents=True, exist_ok=True)
    return parent_dir


def _deep_merge(base, override):
    """Deep-merge override into base: nested dicts are merged, not replaced."""
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def run_single_experiment(exp_config, base_config, exp_name, parent_dir):
    """Run one experiment and move its run directory under parent_dir.

    Returns the destination run directory, or None on failure.
    """
    logger.info(f"{'='*70}")
    logger.info(f"Running Experiment: {exp_name}")
    log_architectures(exp_config, logger=logger, prefix="")
    logger.info(f"{'='*70}")

    # Deep-merge so nested dicts (adaptive_pinn, etc.) are merged, not replaced
    config = _deep_merge(base_config, exp_config)
    # Carry the plan's experiment name into config_used.yaml so comparison
    # reports can label runs by experiment instead of model class.
    config['experiment_tag'] = exp_name

    # Architecture-based folder name (aligned with make_run_dir)
    layers_str = architecture_dir_layers_str(
        config['base_architecture'],
        resolve_experts_architecture(config),
    )
    arch_folder_name = f"{config['problem']}-{layers_str}-{config['activation']}"
    exp_output_dir = parent_dir / arch_folder_name

    # Temporarily replace config/config.yaml with this experiment's config
    config_backup_path = Path('config/config.yaml.backup')
    shutil.copy('config/config.yaml', config_backup_path)

    try:
        with open('config/config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        result = subprocess.run([sys.executable, 'run_training.py'])

        if result.returncode != 0:
            logger.warning(f"{exp_name} exited with code {result.returncode} — attempting to save partial output anyway.")

        # Locate the run's output dir via the record written by the orchestrator
        run_dir = None
        _run_dir_record = Path("outputs") / ".last_run_dir.txt"
        if _run_dir_record.exists():
            _recorded = Path(_run_dir_record.read_text().strip())
            logger.info(f"  [Move] Recorded run_dir: {_recorded}")
            if _recorded.exists():
                run_dir = _recorded
            else:
                logger.warning(f"  [Move] recorded run_dir missing on disk: {_recorded}")

        if run_dir is None:
            logger.error(f"  [Move] could not find output dir for {exp_name}")
            return None

        exp_output_dir.mkdir(parents=True, exist_ok=True)
        dest_dir = exp_output_dir / run_dir.name
        logger.info(f"  [Move] Moving {run_dir} -> {dest_dir}")
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.move(str(run_dir), str(dest_dir))
        return dest_dir

    finally:
        # Restore original config
        if config_backup_path.exists():
            shutil.move(str(config_backup_path), 'config/config.yaml')


def main():
    """Main experiment runner. Optional argv[1]: path to a plan YAML."""
    # Set up logging (console only initially)
    setup_logging(run_dir=None)

    logger.info("="*70)
    logger.info("AToE: Automated Experiments")
    logger.info("="*70)

    # Load experiment plan (default experiments_plan.yaml)
    plan_path = sys.argv[1] if len(sys.argv) > 1 else "experiments_plan.yaml"
    logger.info(f"Plan: {plan_path}")
    plan = load_experiment_plan(plan_path)
    parent_dir = create_experiment_dir(plan)

    # Set up logging to write to the experiment directory
    update_log_file(parent_dir, log_filename="experiment_runner.log")

    # Save experiment plan with git info
    from utils.io import get_git_info
    plan['git'] = get_git_info()
    with open(parent_dir / "experiments_plan.yaml", 'w') as f:
        yaml.dump(plan, f, default_flow_style=False)

    logger.info(f"Experiment Directory: {parent_dir}")
    logger.info(f"Total Experiments: {len(plan['experiments'])}")

    results = {}

    # Run each experiment
    for i, exp in enumerate(plan['experiments'], 1):
        logger.info(f"[{i}/{len(plan['experiments'])}]")
        try:
            result = run_single_experiment(
                exp,
                plan['base_config'],
                exp['name'],
                parent_dir
            )
        except Exception as _exp_err:
            logger.error(f"Experiment {exp['name']} raised an exception: {_exp_err}")
            import traceback
            traceback.print_exc()
            result = None
        results[exp['name']] = result

    # Generate comparison report using the shared regenerate script
    from regenerate_comparison_plots import generate_comparison_for_batch
    generate_comparison_for_batch(parent_dir)

    logger.info(f"{'='*70}")
    logger.info("All Experiments Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Results saved to: {parent_dir}")

    # Close logging
    from utils.logging_config import close_logging
    close_logging()


if __name__ == "__main__":
    main()
