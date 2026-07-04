"""Automated experiment runner for architecture search."""

import yaml
import shutil
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import sys
import torch

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
    base = plan['base_config']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create descriptive folder name (drop training-param suffix for shorter paths)
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
    """Run one experiment."""
    logger.info(f"{'='*70}")
    logger.info(f"Running Experiment: {exp_name}")
    log_architectures(exp_config, logger=logger, prefix="")
    logger.info(f"{'='*70}")
    
    # Deep-merge so nested dicts (adaptive_pinn, etc.) are merged, not replaced
    config = _deep_merge(base_config, exp_config)
    
    # Generate architecture-based folder name (aligned with make_run_dir)
    layers_str = architecture_dir_layers_str(
        config['base_architecture'],
        resolve_experts_architecture(config),
    )
    arch_folder_name = f"{config['problem']}-{layers_str}-{config['activation']}"
    exp_output_dir = parent_dir / arch_folder_name
    
    # Temporarily update config.yaml
    config_backup_path = Path('config/config.yaml.backup')
    shutil.copy('config/config.yaml', config_backup_path)
    
    # Temporarily update outputs path in the environment
    original_outputs_path = Path("outputs")
    temp_outputs_path = exp_output_dir / "temp_outputs"
    
    try:
        # Step 1: Train model and run NCC analysis (run_ncc.py trains once)
        with open('config/config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        result = subprocess.run([sys.executable, 'run_ncc.py'])

        if result.returncode != 0:
            logger.warning(f"{exp_name} exited with code {result.returncode} — attempting to save partial output anyway.")

        # Find the checkpoint inside the run's output dir
        outputs_root = Path("outputs")
        arch_folder_name = f"{config['problem']}-{layers_str}-{config['activation']}"
        arch_dir = outputs_root / arch_folder_name
        best_checkpoint = None
        if arch_dir.exists():
            ts_dirs = sorted(
                [d for d in arch_dir.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime)
            if ts_dirs:
                run_ckpt_dir = ts_dirs[-1] / "checkpoints"
                # Prefer final_model.pt (end-of-training weights) over best_model.pt
                # so eval/probe plots reflect the same model as pred_after_fine_tune.
                for name in ['final_model.pt', 'best_model.pt']:
                    c = run_ckpt_dir / name
                    if c.exists():
                        best_checkpoint = c
                        break
        
        if best_checkpoint is None:
            logger.warning(f"No checkpoint found for {exp_name}, skipping inner metrics")
            best_checkpoint = Path("nonexistent")
        
        # Check if we should skip inner metrics analysis for adaptive PINN
        adaptive_cfg = config.get('adaptive_pinn', {})
        skip_inner_metrics = (
            adaptive_cfg.get('enabled', False) and 
            not adaptive_cfg.get('inner_metrics_calculation', False)
        )
        
        # Track how many analysis steps we ran (for directory handling)
        num_analysis_steps = 1  # NCC always runs
        
        if skip_inner_metrics:
            logger.info(f"{'='*70}")
            logger.info(f"Skipping Probes/Derivatives/Frequency for: {exp_name}")
            logger.info(f"(adaptive_pinn.inner_metrics_calculation is False)")
            logger.info(f"{'='*70}")
        else:
            # Step 2: Run probes analysis in eval-only mode on the trained checkpoint
            logger.info(f"{'='*70}")
            logger.info(f"Running Probe Analysis for: {exp_name} (eval-only mode)")
            logger.info(f"{'='*70}")
            
            # Update config to eval_only mode with resume_from
            eval_config = config.copy()
            eval_config['eval_only'] = True
            eval_config['resume_from'] = str(best_checkpoint)
            
            with open('config/config.yaml', 'w') as f:
                yaml.dump(eval_config, f, default_flow_style=False)
            
            result_probes = subprocess.run([sys.executable, 'run_probes.py'])
            
            if result_probes.returncode != 0:
                logger.warning(f"{exp_name} Probes: Process exited with code {result_probes.returncode}")
            num_analysis_steps += 1
            
            # Step 3: Run derivatives tracker analysis in eval-only mode on the trained checkpoint
            logger.info(f"{'='*70}")
            logger.info(f"Running Derivatives Tracker for: {exp_name} (eval-only mode)")
            logger.info(f"{'='*70}")
            
            with open('config/config.yaml', 'w') as f:
                yaml.dump(eval_config, f, default_flow_style=False)
            
            result_derivatives = subprocess.run([sys.executable, 'run_derivatives_tracker.py'])
            
            if result_derivatives.returncode != 0:
                logger.warning(f"{exp_name} Derivatives: Process exited with code {result_derivatives.returncode}")
            num_analysis_steps += 1
            
            # Step 4: Run frequency tracker analysis in eval-only mode on the trained checkpoint
            logger.info(f"{'='*70}")
            logger.info(f"Running Frequency Tracker for: {exp_name} (eval-only mode)")
            logger.info(f"{'='*70}")
            
            with open('config/config.yaml', 'w') as f:
                yaml.dump(eval_config, f, default_flow_style=False)
            
            result_frequency = subprocess.run([sys.executable, 'run_frequency_tracker.py'])
            
            if result_frequency.returncode != 0:
                logger.warning(f"{exp_name} Frequency: Process exited with code {result_frequency.returncode}")
            num_analysis_steps += 1
        
        # Move outputs to experiment directory
        outputs_root = Path("outputs")
        layers_str = architecture_dir_layers_str(
            config['base_architecture'],
            resolve_experts_architecture(config),
        )
        arch_folder_name = f"{config['problem']}-{layers_str}-{config['activation']}"

        if skip_inner_metrics:
            # Primary: use the run_dir recorded by run_ncc.py to avoid mtime races
            ncc_dir = None
            _run_dir_record = outputs_root / ".last_run_dir.txt"
            if _run_dir_record.exists():
                _recorded = Path(_run_dir_record.read_text().strip())
                logger.info(f"  [Move] Recorded run_dir: {_recorded}")
                if _recorded.exists():
                    ncc_dir = _recorded
                else:
                    logger.warning(f"  [Move] recorded run_dir missing on disk: {_recorded}")

            # Fallback: mtime search
            if ncc_dir is None:
                logger.info(f"  [Move] Falling back to mtime search for {arch_folder_name}")
                arch_dir = outputs_root / arch_folder_name
                logger.info(f"  [Move] arch_dir exists={arch_dir.exists()}")
                if arch_dir.exists():
                    ts_dirs = sorted(
                        [d for d in arch_dir.glob("*/") if d.is_dir()],
                        key=lambda x: x.stat().st_mtime
                    )
                    logger.info(f"  [Move] Found {len(ts_dirs)} dirs: {[d.name for d in ts_dirs]}")
                    if ts_dirs:
                        ncc_dir = ts_dirs[-1]

            if ncc_dir is None:
                logger.error(f"  [Move] could not find output dir for {exp_name}")
                return None

            exp_output_dir.mkdir(parents=True, exist_ok=True)
            dest_dir = exp_output_dir / ncc_dir.name
            logger.info(f"  [Move] Moving {ncc_dir} -> {dest_dir}")
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.move(str(ncc_dir), str(dest_dir))
            return dest_dir

        else:
            # All four analysis steps ran — need 4 most-recent dirs from arch_dir
            arch_dir = outputs_root / arch_folder_name
            if not arch_dir.exists():
                logger.error(f"  [Move] arch_dir not found: {arch_dir}")
                return None
            timestamp_dirs = sorted(
                [d for d in arch_dir.glob("*/") if d.is_dir()],
                key=lambda x: x.stat().st_mtime
            )
            if len(timestamp_dirs) < num_analysis_steps:
                logger.error(f"  [Move] expected {num_analysis_steps} dirs, found {len(timestamp_dirs)}")
                return None

            exp_output_dir.mkdir(parents=True, exist_ok=True)
            ncc_dir = timestamp_dirs[-4]
            probe_dir = timestamp_dirs[-3]
            deriv_dir = timestamp_dirs[-2]
            freq_dir = timestamp_dirs[-1]

            dest_dir = exp_output_dir / ncc_dir.name

            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.move(str(ncc_dir), str(dest_dir))

            # Merge probe results
            probe_plots_src = probe_dir / "probe_plots"
            if probe_plots_src.exists():
                probe_plots_dest = dest_dir / "probe_plots"
                probe_plots_dest.mkdir(parents=True, exist_ok=True)
                for item in probe_plots_src.iterdir():
                    if item.is_file() and item.suffix == '.json':
                        shutil.copy2(item, probe_plots_dest / item.name)
            if probe_dir.exists():
                shutil.rmtree(probe_dir)

            # Merge derivatives results
            deriv_plots_src = deriv_dir / "derivatives_plots"
            if deriv_plots_src.exists():
                deriv_plots_dest = dest_dir / "derivatives_plots"
                deriv_plots_dest.mkdir(parents=True, exist_ok=True)
                for item in deriv_plots_src.iterdir():
                    if item.is_file() and item.suffix == '.json':
                        shutil.copy2(item, deriv_plots_dest / item.name)
            if deriv_dir.exists():
                shutil.rmtree(deriv_dir)

            # Merge frequency results
            freq_plots_src = freq_dir / "frequency_plots"
            if freq_plots_src.exists():
                freq_plots_dest = dest_dir / "frequency_plots"
                freq_plots_dest.mkdir(parents=True, exist_ok=True)
                for item in freq_plots_src.iterdir():
                    if item.is_file() and item.suffix == '.json':
                        shutil.copy2(item, freq_plots_dest / item.name)
            if freq_dir.exists():
                shutil.rmtree(freq_dir)

            return dest_dir

        return None
        
    finally:
        # Restore original config
        if config_backup_path.exists():
            shutil.move(str(config_backup_path), 'config/config.yaml')


def generate_comparison_report(parent_dir, results):
    """Generate comparison plots and tables."""
    logger.info(f"{'='*70}")
    logger.info("Generating Comparison Report")
    logger.info(f"{'='*70}")
    
    # Collect training metrics
    metrics_data = []
    ncc_data = {}  # Store all NCC data for periodic plots
    probe_data = {}  # Store all probe data for comparison plots
    derivatives_data = {}  # Store all derivatives data for comparison plots
    frequency_data = {}  # Store all frequency data for comparison plots
    expert_regions_data = {}  # Store expert regions for adaptive PINN comparison
    
    for exp_name, result_path in results.items():
        if result_path is None:
            continue
            
        # Load training metrics
        metrics_file = result_path / "metrics.json"
        if not metrics_file.exists():
            continue
            
        with open(metrics_file) as f:
            train_metrics = json.load(f)
        
        # Collect NCC metrics from all epochs (periodic + final)
        ncc_plots_dir = result_path / "ncc_plots"
        ncc_epochs = {}
        
        if ncc_plots_dir.exists():
            # Load final NCC
            final_ncc_file = ncc_plots_dir / "ncc_metrics.json"
            if final_ncc_file.exists():
                with open(final_ncc_file) as f:
                    ncc_epochs['final'] = json.load(f)
            
            # Load periodic NCCs
            for subdir in ncc_plots_dir.iterdir():
                if subdir.is_dir() and subdir.name.startswith("ncc_plots_epoch_"):
                    epoch_num = int(subdir.name.split("_")[-1])
                    epoch_file = subdir / "ncc_metrics.json"
                    if epoch_file.exists():
                        with open(epoch_file) as f:
                            ncc_epochs[epoch_num] = json.load(f)
        
        if not ncc_epochs:
            continue
        
        # Store for table
        final_ncc = ncc_epochs.get('final', list(ncc_epochs.values())[-1])
        
        # Load probe metrics
        probe_file = result_path / "probe_plots" / "probe_metrics.json"
        probe_metrics = None
        if probe_file.exists():
            with open(probe_file) as f:
                probe_metrics = json.load(f)
                probe_data[exp_name] = probe_metrics
        
        # Load derivatives metrics
        deriv_file = result_path / "derivatives_plots" / "derivatives_metrics.json"
        deriv_metrics = None
        if deriv_file.exists():
            with open(deriv_file) as f:
                deriv_metrics = json.load(f)
                derivatives_data[exp_name] = deriv_metrics
        
        # Load frequency metrics
        freq_file = result_path / "frequency_plots" / "frequency_metrics.json"
        freq_metrics = None
        if freq_file.exists():
            with open(freq_file) as f:
                freq_metrics = json.load(f)
                frequency_data[exp_name] = freq_metrics
        
        # Load expert regions for adaptive PINN
        expert_regions_file = result_path / "adaptive_plots" / "expert_regions.json"
        if expert_regions_file.exists():
            from adaptive.indicators import RegionDescriptor
            from adaptive.visualization import load_regions_metadata
            regions = load_regions_metadata(expert_regions_file)
            if regions:
                expert_regions_data[exp_name] = regions
        
        # Extract margin SNR for final layer
        final_layer = list(final_ncc['layer_accuracies'].keys())[-1]
        margin_mean = final_ncc['layer_margins'][final_layer]['mean_margin']
        margin_std = final_ncc['layer_margins'][final_layer]['std_margin']
        margin_snr = margin_mean / margin_std if margin_std > 0 else 0
        
        # Build metrics data row
        metrics_row = {
            'experiment': exp_name,
            'final_train_loss': train_metrics['train_loss'][-1],
            'final_eval_loss': train_metrics['eval_loss'][-1],
            'final_eval_rel_l2': train_metrics['eval_rel_l2'][-1],
            'final_eval_inf_norm': train_metrics['eval_inf_norm'][-1],
            'ncc_final_accuracy': final_ncc['layer_accuracies'][list(final_ncc['layer_accuracies'].keys())[-1]],
            'margin_snr': margin_snr
        }
        
        # Add probe metrics if available (last layer probe)
        if probe_metrics:
            metrics_row['probe_final_eval_rel_l2'] = probe_metrics['eval']['rel_l2'][-1]
        
        # Add derivatives metrics if available
        if deriv_metrics:
            metrics_row['deriv_final_train_residual'] = deriv_metrics['final_layer_train_residual']
            metrics_row['deriv_final_eval_residual'] = deriv_metrics['final_layer_eval_residual']
        
        metrics_data.append(metrics_row)
        
        # Store for NCC plots
        ncc_data[exp_name] = ncc_epochs
    
    if not metrics_data:
        logger.info("  No valid results to compare.")
        return
    
    # Create comparison table
    df = pd.DataFrame(metrics_data)
    df.to_csv(parent_dir / "comparison_summary.csv", index=False)
    logger.info(f"  Comparison table saved to comparison_summary.csv")
    
    # Generate the three plots
    _generate_training_results_plot(parent_dir, df)
    
    # Use shared comparison plot functions
    from utils.comparison_plots import generate_ncc_classification_plot, generate_ncc_compactness_plot
    generate_ncc_classification_plot(parent_dir, ncc_data)
    generate_ncc_compactness_plot(parent_dir, ncc_data)
    
    # Generate probe comparison plots if probe data available
    if probe_data:
        from utils.comparison_plots import generate_probe_comparison_plots
        generate_probe_comparison_plots(parent_dir, probe_data)
    
    # Generate derivatives comparison plots if derivatives data available
    if derivatives_data:
        from utils.comparison_plots import generate_derivatives_comparison_plots
        generate_derivatives_comparison_plots(parent_dir, derivatives_data)
    
    # Generate frequency comparison plots if frequency data available
    if frequency_data:
        from utils.comparison_plots import generate_frequency_coverage_comparison, plot_spectral_learning_efficiency_comparison
        
        generate_frequency_coverage_comparison(parent_dir, frequency_data)
        plot_spectral_learning_efficiency_comparison(frequency_data, parent_dir)
    
    # Generate expert regions comparison if adaptive PINN data available
    if expert_regions_data:
        logger.info(f"  Generating expert regions comparison ({len(expert_regions_data)} experiments)...")
        from adaptive.visualization import (
            plot_expert_regions_comparison, prepare_ground_truth_grid
        )
        
        # Get domain bounds from the first experiment's config
        # (assuming all experiments use the same problem/domain)
        first_result_path = list(results.values())[0]
        if first_result_path is not None:
            config_file = first_result_path / "config_used.yaml"
            if config_file.exists():
                import yaml
                with open(config_file) as f:
                    exp_config = yaml.safe_load(f)
                problem = exp_config.get('problem', 'burgers1d')
                problem_config = exp_config.get(problem, {})
                spatial_domain = problem_config.get('spatial_domain', [[-1, 1]])
                temporal_domain = problem_config.get('temporal_domain', [0, 1])
                
                # Build domain bounds
                if len(spatial_domain) == 1:
                    domain_bounds = {
                        'lower': [spatial_domain[0][0], temporal_domain[0]],
                        'upper': [spatial_domain[0][1], temporal_domain[1]]
                    }
                    problem_type = '2d'
                else:
                    domain_bounds = {
                        'lower': [spatial_domain[0][0], spatial_domain[1][0], temporal_domain[0]],
                        'upper': [spatial_domain[0][1], spatial_domain[1][1], temporal_domain[1]]
                    }
                    problem_type = '3d'
                
                # Load eval data for ground truth background
                gt_grid, gt_x, gt_t = None, None, None
                if problem_type == '2d':
                    eval_data_path = Path("datasets") / problem / "eval_data.pt"
                    if eval_data_path.exists():
                        try:
                            eval_data = torch.load(eval_data_path, map_location='cpu')
                            gt_grid, gt_x, gt_t = prepare_ground_truth_grid(
                                eval_data, domain_bounds
                            )
                        except Exception as e:
                            logger.warning(f"  Could not load ground truth: {e}")
                
                plot_expert_regions_comparison(
                    experiment_regions=expert_regions_data,
                    domain_bounds=domain_bounds,
                    output_path=parent_dir / "expert_regions_comparison.png",
                    problem_type=problem_type,
                    ground_truth=gt_grid,
                    grid_x=gt_x,
                    grid_t=gt_t
                )
    
    logger.info(f"Comparison report saved to {parent_dir}")


def _generate_training_results_plot(parent_dir, df):
    """Generate training and results comparison table (no bar charts)."""
    fig = plt.figure(figsize=(16, 6))
    ax3 = fig.add_subplot(111)
    ax3.axis('off')
    
    # Create colored table
    table_data = []
    col_labels = ['Experiment', 'Train Loss', 'Eval Loss',
                  'Eval Rel-L2', 'Eval Inf', 'NCC Final Acc', 'Margin SNR', 
                  'Deriv Train Res', 'Deriv Eval Res']
    
    for _, row in df.iterrows():
        row_data = [
            row['experiment'],
            f"{row['final_train_loss']:.6f}",
            f"{row['final_eval_loss']:.6f}",
            f"{row['final_eval_rel_l2']:.6f}",
            f"{row['final_eval_inf_norm']:.6f}",
            f"{row['ncc_final_accuracy']:.6f}",
            f"{row['margin_snr']:.2f}"
        ]
        # Add derivatives if available
        if 'deriv_final_train_residual' in row and not pd.isna(row['deriv_final_train_residual']):
            row_data.append(f"{row['deriv_final_train_residual']:.2e}")
        else:
            row_data.append("N/A")
        if 'deriv_final_eval_residual' in row and not pd.isna(row['deriv_final_eval_residual']):
            row_data.append(f"{row['deriv_final_eval_residual']:.2e}")
        else:
            row_data.append("N/A")
        table_data.append(row_data)
    
    table = ax3.table(cellText=table_data, colLabels=col_labels,
                     cellLoc='center', loc='center',
                     bbox=[0.05, 0.1, 0.9, 0.8])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)
    
    # Color cells per column (green=best, red=worst)
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    # Create green-to-red colormap
    cmap = LinearSegmentedColormap.from_list('GreenRed', ['#2ecc71', '#f1c40f', '#e74c3c'])
    
    # Color coding for each column
    num_cols = len(col_labels)
    for col_idx in range(1, min(num_cols, len(df.columns) + 1)):
        # Check if this column exists in the dataframe
        if col_idx >= len(df.columns):
            continue
        
        col_name = df.columns[col_idx]
        values = df[col_name].values
        
        # Skip if all NaN
        if pd.isna(values).all():
            continue
        
        # For losses/errors/residuals, lower is better; for accuracy and margin SNR, higher is better
        if col_idx == 7 or col_idx == 8:  # NCC accuracy and Margin SNR - higher is better
            norm_values = 1 - (values - values.min()) / (values.max() - values.min() + 1e-10)
        else:  # Losses, errors, residuals - lower is better
            norm_values = (values - values.min()) / (values.max() - values.min() + 1e-10)
        
        for row_idx, norm_val in enumerate(norm_values):
            if not pd.isna(norm_val):
                cell = table[(row_idx + 1, col_idx)]
                color = cmap(norm_val)
                cell.set_facecolor(color)
                cell.set_alpha(0.7)
    
    # Style header
    for col_idx in range(num_cols):
        cell = table[(0, col_idx)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    fig.suptitle('Training and Results Comparison', fontsize=16, fontweight='bold', y=0.92)
    
    plt.savefig(parent_dir / "training_and_results_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Training and results comparison saved to training_and_results_comparison.png")




def main():
    """Main experiment runner."""
    # Set up logging (console only initially)
    setup_logging(run_dir=None)
    
    logger.info("="*70)
    logger.info("NCC-PINN: Automated Experiments")
    logger.info("="*70)
    
    # Load experiment plan
    plan = load_experiment_plan()
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
