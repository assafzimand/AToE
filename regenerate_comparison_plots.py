"""Regenerate comparison plots for experiment batches.

Supports two directory structures:
  1. Multi-PDE batch: root → PDE dirs → timestamp dirs (each with a different model)
  2. Single batch:    root → architecture dirs → timestamp dirs
"""

import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

import sys
sys.path.insert(0, str(Path(__file__).parent))

_TIMESTAMP_RE = re.compile(r'\d{8}_\d{6}$')


def _load_ckpt_helpers():
    """Import model-build/checkpoint-load helpers from the plot script
    (scripts/ is not a package, so load it by file path)."""
    import importlib.util
    helper_path = Path(__file__).parent / 'scripts' / 'plot_experts_predictions.py'
    spec = importlib.util.spec_from_file_location(
        'plot_experts_predictions', helper_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pin_float32_precision() -> None:
    """Force true float32 matmuls for checkpoint re-evaluation.

    On Ampere+ GPUs TF32 (~10-bit mantissa) silently distorts forward
    passes: the same checkpoint can score differently here than during
    training. Pinning makes recomputed metrics device-reproducible.
    """
    import torch
    torch.set_float32_matmul_precision('highest')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


_SEGMENT_ORDER = {'root': 0, 'phase3': 2, 'fine_tune': 3, 'main': 4}


def _segment_sort_key(seg: str):
    """Canonical pipeline order: root -> level_* -> phase3 -> fine_tune."""
    if seg.startswith('level'):
        return (1, seg)
    return (_SEGMENT_ORDER.get(seg, 5), seg)


def _segment_checkpoints(ckpt_dir: Path) -> Dict:
    """Map segment name -> its best checkpoint, in pipeline order.

    New runs store one checkpoint per segment (``best_model_<segment>.pt``,
    reconciled so best == end-of-segment). Old runs fall back to their
    ``checkpoint_after_<segment>.pt`` files.
    """
    seg_ckpts = {}
    for p in sorted(ckpt_dir.glob('checkpoint_after_*.pt')):
        seg_ckpts[p.stem.replace('checkpoint_after_', '')] = p
    for p in sorted(ckpt_dir.glob('best_model_*.pt')):
        seg_ckpts[p.stem.replace('best_model_', '')] = p  # new format wins
    return dict(sorted(seg_ckpts.items(),
                       key=lambda kv: _segment_sort_key(kv[0])))


def _eval_segment_rel_l2s(result_path: Path, helpers) -> Dict:
    """Reload every segment's best checkpoint and recompute its rel-L2 on
    the solver's native grid.

    These are the numbers to trust when reproducing results: metrics.json
    stores training-time values, which carry the training environment's
    numerics (e.g. TF32 on Ampere GPUs). phase3 checkpoints are scored with
    HARD indicators, matching how that segment trains and evaluates.

    Returns {segment: {'rel_l2', 'epoch', 'ckpt'}} (possibly empty).
    """
    import torch
    import yaml

    cfg_path = result_path / 'config_used.yaml'
    ckpt_dir = result_path / 'checkpoints'
    if not cfg_path.exists() or not ckpt_dir.exists():
        return {}
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get(cfg.get('problem', ''), {}).get('spatial_dim', 1) != 1:
        return {}  # native-grid metric only exists for 1D problems

    from trainer.utils import compute_native_grid_metrics
    is_adaptive = cfg.get('adaptive_pinn', {}).get('enabled', False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out = {}
    for segment, ckpt_path in _segment_checkpoints(ckpt_dir).items():
        try:
            model = helpers._build_model(cfg)
            epoch = helpers._load_checkpoint(model, ckpt_path, is_adaptive)
            model = model.to(device).eval()
            if segment.startswith('phase3') and hasattr(model, 'blending_mode'):
                model.blending_mode = 'hard'
            metrics = compute_native_grid_metrics(model, cfg, device)
            if metrics is not None:
                out[segment] = {'rel_l2': metrics['rel_l2'],
                                'epoch': epoch, 'ckpt': ckpt_path.name}
        except Exception as _eval_err:
            print(f"  [CkptEval] {result_path.name} [{segment}]: "
                  f"failed — {_eval_err}")
    return out


def _regen_segment_plots(result_path: Path, helpers) -> None:
    """Re-render pred_after_<segment>.png in adaptive_plots/ from each
    segment's best checkpoint (best_model_<segment>.pt; legacy runs:
    checkpoint_after_<segment>.pt), replacing the in-training plots with the
    unified native-grid renderer."""
    import yaml

    cfg_path = result_path / 'config_used.yaml'
    ckpt_dir = result_path / 'checkpoints'
    if not cfg_path.exists() or not ckpt_dir.exists():
        return
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    problem = cfg.get('problem', '')
    is_adaptive = cfg.get('adaptive_pinn', {}).get('enabled', False)
    out_dir = result_path / 'adaptive_plots'

    from utils.problem_specific.generic_viz import plot_predictions_and_error_maps

    for segment, ckpt_path in _segment_checkpoints(ckpt_dir).items():
        try:
            model = helpers._build_model(cfg)
            epoch = helpers._load_checkpoint(model, ckpt_path, is_adaptive)
            model.eval()
            # Split-segment (phase3) checkpoints are trained with hard region
            # ownership — render them with hard indicators so the regenerated
            # plot matches the in-training pred_after_phase3.png.
            if segment.startswith('phase3') and hasattr(model, 'blending_mode'):
                model.blending_mode = 'hard'
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_predictions_and_error_maps(
                model, out_dir, cfg,
                filename=f'pred_after_{segment}_ep{epoch}_relL2_{{relL2}}.png')
            print(f"  [SegmentPlots] regenerated pred_after_{segment} "
                  f"(epoch {epoch})")
        except Exception as _seg_err:
            print(f"  [SegmentPlots] {segment}: failed — {_seg_err}")


def _is_timestamp_dir(d: Path) -> bool:
    return d.is_dir() and bool(_TIMESTAMP_RE.match(d.name))


def _get_model_name(ts_dir: Path) -> str:
    """Run label from config_used.yaml: the plan's experiment name when
    available (experiment_tag), else the model class, else the dir name."""
    config_file = ts_dir / "config_used.yaml"
    if config_file.exists():
        try:
            import yaml
            with open(config_file) as f:
                cfg = yaml.safe_load(f)
            return cfg.get('experiment_tag') or cfg.get('model', ts_dir.name)
        except Exception:
            pass
    return ts_dir.name


def _build_run_name(ts_dir: Path) -> str:
    """Build a descriptive experiment name from a timestamp run directory.

    Reads config_used.yaml and extracts key training parameters to
    differentiate runs of the same architecture.
    Falls back to the timestamp folder name if config is unavailable.
    """
    config_file = ts_dir / "config_used.yaml"
    if not config_file.exists():
        return ts_dir.name

    try:
        import yaml
        with open(config_file) as f:
            cfg = yaml.safe_load(f)

        parts = []

        # Training params
        parts.append(f"ep{cfg.get('epochs', '?')}")
        lr = cfg.get('lr', None)
        if lr is not None:
            parts.append(f"lr{lr}")

        # Adaptive params (if present)
        adaptive = cfg.get('adaptive_pinn', {})
        if adaptive.get('enabled', False):
            spawn = adaptive.get('spawn_every_epochs')
            if spawn is not None:
                parts.append(f"sp{spawn}")
            problem_cfg = cfg.get(cfg.get('problem', ''), {})
            wt = problem_cfg.get('wavelet_threshold')
            if wt is not None:
                parts.append(f"wt{wt}")
            if adaptive.get('only_leaves', False):
                parts.append("leaves")

        # Optimizer switch
        switch = cfg.get('optimizer_switch_fraction')
        if switch is not None:
            parts.append(f"sw{switch}")

        name = "_".join(str(p) for p in parts)
        # Append short timestamp to guarantee uniqueness
        name += f"_{ts_dir.name[-6:]}"
        return name

    except Exception:
        return ts_dir.name


def _extract_run_info(result_path: Path) -> Dict:
    """Extract model/optimizer/capacity info from a run directory."""
    info = {
        'pde': '-',
        'model_type': '-',
        'expert_type': '-',
        'total_params': '-',
        'n_experts': 0,
        'expert_sizes': '-',
        'optimizer': '-',
        'lr_sched': '-',
        'spawning': '-',
    }
    config_file = result_path / 'config_used.yaml'
    if config_file.exists():
        try:
            import yaml
            with open(config_file) as f:
                cfg = yaml.safe_load(f)
            adaptive = cfg.get('adaptive_pinn', {})
            info['pde'] = cfg.get('problem', '-')
            info['model_type'] = cfg.get('model', '-')
            info['expert_type'] = adaptive.get(
                'expert_type', 'mlp')

            opt = cfg.get('optimizer', 'adam')
            sw = cfg.get('optimizer_switch_at', 1.0)
            if opt == 'soap':
                info['optimizer'] = 'SOAP'
            elif sw < 1.0:
                info['optimizer'] = f'Adam→LBFGS@{sw}'
            else:
                info['optimizer'] = 'Adam'

            lr = cfg.get('lr', '?')
            sched = cfg.get('lr_schedule', 'none')
            bs = cfg.get('batch_size', '?')
            info['lr_sched'] = (
                f"lr={lr}\n{sched}\nbs={bs}")
            info['spawning'] = adaptive.get(
                'spawning_method', '-')
        except Exception:
            pass

    metrics_file = result_path / 'metrics.json'
    if metrics_file.exists():
        try:
            with open(metrics_file) as f:
                met = json.load(f)
            ap = met.get('adaptive_pinn', {})
            info['n_experts'] = ap.get('num_experts', 0)

            forward_p = ap.get('forward_params')
            total_p = met.get('total_params',
                              ap.get('total_params'))
            if forward_p is not None:
                info['total_params'] = f'{forward_p:,}'
            elif total_p is not None:
                info['total_params'] = f'{total_p:,}'

            expert_params = ap.get('expert_params', [])
            n_exp = info['n_experts']
            if expert_params and n_exp > 0:
                leaf_idx = set(
                    ap.get('leaf_expert_indices', []))
                is_leaves = info['model_type'] in (
                    'AToELeaves', 'AToE-Leaves')
                if is_leaves and leaf_idx:
                    active_params = [
                        expert_params[i]
                        for i in leaf_idx
                        if i < len(expert_params)]
                    label = 'leaves'
                else:
                    active_params = expert_params
                    label = 'experts'
                if active_params:
                    mx = max(active_params)
                    mn = min(active_params)
                    avg = sum(active_params) // len(
                        active_params)
                    ct = len(active_params)
                    if mx == mn:
                        info['expert_sizes'] = (
                            f"{ct} {label}\n"
                            f"fixed: {mx:,}p each")
                    else:
                        info['expert_sizes'] = (
                            f"{ct} {label}\n"
                            f"max:{mx:,} avg:{avg:,}p")
                else:
                    info['expert_sizes'] = (
                        f"{n_exp} experts")
            elif n_exp > 0:
                info['expert_sizes'] = (
                    f"{n_exp} experts")
            else:
                info['expert_sizes'] = 'base only'
        except Exception:
            pass

    return info


def _generate_training_results_plot(parent_dir, df,
                                     run_infos=None):
    """Generate comparison table with model/optimizer info
    and colored result columns."""
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    info_cols = [
        'PDE', 'Model', 'Capacity', 'Optimizer',
        'LR / Sched', 'Spawning']
    # Per-segment best-checkpoint columns (best_<segment>_rel_l2), in
    # pipeline order — shows which stage produced which accuracy.
    seg_keys = sorted(
        [c for c in df.columns
         if c.startswith('best_') and c.endswith('_rel_l2')],
        key=lambda c: _segment_sort_key(c[len('best_'):-len('_rel_l2')]))
    result_keys = ['final_train_loss', 'final_rel_l2', 'final_inf_norm'] + seg_keys
    result_cols = ['Train\nLoss', 'Rel-L2\n(final)', 'Inf\n(final)'] + [
        f"Best\n{k[len('best_'):-len('_rel_l2')]}" for k in seg_keys]
    col_labels = ['Experiment'] + info_cols + result_cols
    n_info = len(info_cols)
    first_result_col = 1 + n_info

    def _fmt(v):
        import math
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return 'N/A'
        return f'{v:.4e}'

    table_data = []
    for _, row in df.iterrows():
        exp = row['experiment']
        info = (run_infos or {}).get(exp, {})
        model_str = (
            f"{info.get('model_type', '-')}\n"
            f"{info.get('expert_type', '-')}")
        capacity_str = (
            f"{info.get('total_params', '-')} params\n"
            f"{info.get('expert_sizes', '-')}")
        row_data = [
            exp,
            info.get('pde', '-'),
            model_str,
            capacity_str,
            info.get('optimizer', '-'),
            info.get('lr_sched', '-'),
            info.get('spawning', '-'),
        ] + [_fmt(row.get(k)) for k in result_keys]
        table_data.append(row_data)

    n_rows = len(table_data)
    n_cols = len(col_labels)
    fig_w = max(22, n_cols * 2.0)
    row_h = 0.55
    fig_h = max(4, 1.2 + n_rows * row_h)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    table = ax.table(
        cellText=table_data, colLabels=col_labels,
        cellLoc='center', loc='center',
        bbox=[0.01, 0.01, 0.98, 0.92])

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.auto_set_column_width(list(range(n_cols)))

    cell_h = 1.0 / (n_rows + 1)
    for (r, c), cell in table.get_celld().items():
        cell.set_height(cell_h)

    cmap = LinearSegmentedColormap.from_list(
        'GreenRed', ['#2ecc71', '#f1c40f', '#e74c3c'])

    for ri, key in enumerate(result_keys):
        ci = first_result_col + ri
        if key not in df.columns:
            continue
        vals = df[key].values.astype(float)
        if pd.isna(vals).all():
            continue
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        rng = vmax - vmin
        if rng < 1e-15:
            continue
        for row_idx in range(n_rows):
            v = vals[row_idx]
            if pd.isna(v):
                continue
            norm_v = (v - vmin) / (rng + 1e-15)
            cell = table[(row_idx + 1, ci)]
            cell.set_facecolor(cmap(norm_v))
            cell.set_alpha(0.7)

    for ci in range(n_cols):
        cell = table[(0, ci)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(
            weight='bold', color='white', fontsize=7)

    for row_idx in range(n_rows):
        cell_exp = table[(row_idx + 1, 0)]
        cell_exp.set_text_props(fontsize=6)
        for ci in range(1, first_result_col):
            cell = table[(row_idx + 1, ci)]
            cell.set_facecolor('#f8f9fa')
            cell.set_text_props(fontsize=6.5)

    fig.suptitle(
        'Training and Results Comparison',
        fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(
        parent_dir / "training_and_results_comparison.png",
        dpi=150, bbox_inches='tight')
    plt.close()
    print("  Training and results comparison saved")


def generate_comparison_for_batch(batch_dir: Path, label: str = None):
    """Generate comparison plots for a single experiment batch.

    Handles two layouts:
      Flat   – batch_dir contains timestamp dirs directly (e.g. per-PDE dir
               where each timestamp is a different model).
      Nested – batch_dir contains architecture dirs, each with timestamp subdirs.
    """
    display_name = label or batch_dir.name
    print(f"\n{'='*70}")
    print(f"Processing: {display_name}")
    print(f"{'='*70}\n")

    child_dirs = sorted([d for d in batch_dir.iterdir()
                         if d.is_dir() and d.name != 'checkpoints'])

    if not child_dirs:
        print(f"  No subdirectories found in {batch_dir}")
        return

    # --- Detect flat structure (timestamps directly under batch_dir) ---
    direct_ts_dirs = [d for d in child_dirs
                      if _is_timestamp_dir(d) and (d / 'metrics.json').exists()]

    results = {}
    if direct_ts_dirs:
        print(f"  Found {len(direct_ts_dirs)} experiment runs (flat / per-PDE structure)")
        for ts_dir in direct_ts_dirs:
            exp_name = _get_model_name(ts_dir)
            results[exp_name] = ts_dir
    else:
        # --- Nested structure (architecture dirs → timestamp subdirs) ---
        model_dirs = child_dirs
        print(f"  Found {len(model_dirs)} architecture dirs: {[d.name for d in model_dirs]}")

        for model_dir in model_dirs:
            timestamp_dirs = sorted(
                [d for d in model_dir.iterdir()
                 if d.is_dir() and d.name != 'checkpoints'
                 and (d / 'metrics.json').exists()]
            )
            if not timestamp_dirs:
                results[model_dir.name] = model_dir
                continue

            for ts_dir in timestamp_dirs:
                exp_name = _get_model_name(ts_dir)
                if exp_name in results:
                    exp_name = f"{exp_name}_{ts_dir.name[-6:]}"
                results[exp_name] = ts_dir

    # Helpers for loading checkpoints (model build); optional — comparison
    # still works without them.
    try:
        _ckpt_helpers = _load_ckpt_helpers()
    except Exception as _h_err:
        print(f"  [SegmentPlots] helpers unavailable ({_h_err}); "
              f"skipping segment-plot regeneration")
        _ckpt_helpers = None

    # Collect training metrics
    metrics_data = []
    for exp_name, result_path in results.items():
        if result_path is None:
            continue

        # Re-render the per-segment prediction plots from their checkpoints
        if _ckpt_helpers is not None:
            _regen_segment_plots(result_path, _ckpt_helpers)

        # Load training metrics
        metrics_file = result_path / "metrics.json"
        if not metrics_file.exists():
            print(f"  Warning: No metrics.json found for {exp_name}")
            continue

        with open(metrics_file) as f:
            train_metrics = json.load(f)

        def _last(lst):
            return lst[-1] if lst else float('nan')

        # Headline rel-L2: the reconciled end-of-run value recomputed by
        # finalize ('final_dense_rel_l2'); older runs fall back to the last
        # training-curve point.
        dense_rel_l2 = train_metrics.get('final_dense_rel_l2', float('nan'))
        if dense_rel_l2 == dense_rel_l2:  # not nan
            final_rel_l2 = dense_rel_l2
        else:
            final_rel_l2 = _last(train_metrics.get(
                'rel_l2', train_metrics.get('eval_rel_l2', [])))

        # Recompute rel-L2 of EVERY segment's best checkpoint (authoritative
        # for reproduction — training-time metrics carry the training
        # environment's numerics, e.g. TF32 on Ampere GPUs). One column per
        # segment shows which stage produced which accuracy.
        seg_rel = {}
        if _ckpt_helpers is not None:
            seg_rel = _eval_segment_rel_l2s(result_path, _ckpt_helpers)
            for _seg, _info in seg_rel.items():
                print(f"  {exp_name}: best[{_seg}] rel-L2 = "
                      f"{_info['rel_l2']:.6e} ({_info['ckpt']} "
                      f"@ epoch {_info['epoch']})")

        _row = {
            'experiment': exp_name,
            'final_train_loss': _last(train_metrics.get('train_loss', [])),
            'final_rel_l2': final_rel_l2,
            # 'inf_norm' is the solver-grid metric; older runs stored it
            # as 'eval_inf_norm'.
            'final_inf_norm': _last(train_metrics.get(
                'inf_norm', train_metrics.get('eval_inf_norm', []))),
        }
        for _seg, _info in seg_rel.items():
            _row[f'best_{_seg}_rel_l2'] = _info['rel_l2']
        metrics_data.append(_row)

    if not metrics_data:
        print(f"  No valid results to compare for batch {batch_dir.name}")
        return

    # Extract run info (model/optimizer/capacity) for each experiment
    run_infos = {}
    for exp_name, result_path in results.items():
        if result_path is not None:
            run_infos[exp_name] = _extract_run_info(result_path)

    # Create comparison table
    df = pd.DataFrame(metrics_data)
    df.to_csv(batch_dir / "comparison_summary.csv", index=False)
    print(f"  Comparison table saved to comparison_summary.csv")

    # Generate plots
    _generate_training_results_plot(batch_dir, df, run_infos)

    print(f"\n  [OK] Comparison plots saved to {batch_dir}")


def _detect_structure(target_path: Path):
    """Detect the directory layout.

    Returns one of:
      'multi_pde'   – target has PDE child dirs, each with timestamp subdirs
      'single_batch'– target is a single batch (arch dirs → timestamp subdirs,
                       or flat timestamps)
      'multi_batch' – target contains multiple independent batch dirs
    """
    child_dirs = [d for d in target_path.iterdir() if d.is_dir()]
    if not child_dirs:
        return 'multi_batch'

    # Multi-PDE: each child dir has ≥2 timestamp subdirs with metrics.json
    pde_like = 0
    for cd in child_dirs:
        ts_dirs = [d for d in cd.iterdir()
                   if _is_timestamp_dir(d) and (d / 'metrics.json').exists()]
        if len(ts_dirs) >= 2:
            pde_like += 1
    if pde_like >= 2:
        return 'multi_pde'

    # Single batch: any child (or grandchild) has metrics.json
    for cd in child_dirs:
        if (cd / 'metrics.json').exists():
            return 'single_batch'
        for sub in cd.iterdir():
            if sub.is_dir() and (sub / 'metrics.json').exists():
                return 'single_batch'

    return 'multi_batch'


def main():
    """Main entry point."""
    _pin_float32_precision()
    if len(sys.argv) > 1:
        target_path = Path(sys.argv[1])
    else:
        target_path = Path("outputs/experiments/AToE-New")

    if not target_path.exists():
        print(f"Error: Directory not found: {target_path}")
        return

    structure = _detect_structure(target_path)
    print(f"Detected structure: {structure}")

    if structure == 'multi_pde':
        pde_dirs = sorted([d for d in target_path.iterdir() if d.is_dir()])
        print(f"Found {len(pde_dirs)} PDE group(s):")
        for pd_dir in pde_dirs:
            pde_label = pd_dir.name.split('-')[0]
            print(f"  - {pde_label} ({pd_dir.name})")

        for pd_dir in pde_dirs:
            pde_label = pd_dir.name.split('-')[0]
            try:
                generate_comparison_for_batch(
                    pd_dir, label=f"{pde_label} ({pd_dir.name})")
            except Exception as e:
                print(f"\nError processing {pd_dir.name}: {e}")
                import traceback
                traceback.print_exc()

    elif structure == 'single_batch':
        generate_comparison_for_batch(target_path)

    else:
        batch_dirs = sorted([d for d in target_path.iterdir() if d.is_dir()])
        if not batch_dirs:
            print(f"No experiment batches found in {target_path}")
            return
        print(f"Found {len(batch_dirs)} experiment batch(es):")
        for bd in batch_dirs:
            print(f"  - {bd.name}")
        for bd in batch_dirs:
            try:
                generate_comparison_for_batch(bd)
            except Exception as e:
                print(f"\nError processing {bd.name}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*70}")
    print("Done! All comparison plots regenerated.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
