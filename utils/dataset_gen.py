"""Dataset generation utilities."""

import math
import torch
from pathlib import Path
from typing import Dict
import importlib
from utils.dataset_plotting import (
    plot_dataset,
    plot_dataset_statistics
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


def calculate_dataset_sizes(config: Dict) -> Dict[str, int]:
    """
    Calculate dataset sizes from ratios and problem domain.
    
    Args:
        config: Configuration dictionary containing problem name,
                sampling ratios, and problem-specific domain info.
    
    Returns:
        Dictionary with calculated dataset sizes.
    """
    problem = config['problem']
    problem_cfg = config[problem]
    sampling = config['sampling']
    
    # Get dimensionality: d = spatial_dim + 1 (time)
    spatial_dim = problem_cfg['spatial_dim']
    d = spatial_dim + 1
    
    # Calculate volume V = product of all domain ranges
    spatial_domain = problem_cfg['spatial_domain']
    temporal_domain = problem_cfg['temporal_domain']
    
    V = 1.0
    for i in range(spatial_dim):
        V *= (spatial_domain[i][1] - spatial_domain[i][0])
    V *= (temporal_domain[1] - temporal_domain[0])
    
    # Calculate n_residual_train from: ratio = S^(1/d) / V^(1/d)
    # Solving: S = (ratio * V^(1/d))^d
    ratio = sampling['sample_volume_ratio']
    # Allow explicit override via sampling.n_residual_train; default 10000.
    # n_residual_train = int(round((ratio * (V ** (1/d))) ** d))
    n_residual_train = sampling.get('n_residual_train', 10000)
    # Allow explicit override via sampling.n_initial_train / n_boundary_train
    # (absolute counts); falls back to the ratio-based calculation otherwise.
    n_initial_train = sampling.get('n_initial_train')
    if n_initial_train is None:
        n_initial_train = int(round(n_residual_train * sampling['initial_train_ratio']))
    n_boundary_train = sampling.get('n_boundary_train')
    if n_boundary_train is None:
        n_boundary_train = int(round(n_residual_train * sampling['boundary_train_ratio']))
    # Calculate other sizes from ratios
    # The eval sizes are used only by the standalone tree/analysis scripts
    # (the pipeline has no eval dataset); eval_train_ratio is optional.
    _eval_ratio = sampling.get('eval_train_ratio', 0.25)
    sizes = {
        'n_residual_train': n_residual_train,
        'n_initial_train': n_initial_train,
        'n_boundary_train': n_boundary_train,
        'n_residual_eval': int(round(n_residual_train * _eval_ratio)),
        'n_initial_eval': int(round(n_initial_train * _eval_ratio)),
        'n_boundary_eval': int(round(n_boundary_train * _eval_ratio)),
    }
    
    # Print calculated values
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset Size Calculation for {problem}")
    logger.info(f"{'='*60}")
    logger.info(f"  Dimensionality (d): {d} ({spatial_dim} spatial + 1 time)")
    logger.info(f"  Domain Volume (V): {V:.4f}")
    logger.info(f"  Target Ratio (S^(1/d) / V^(1/d)): {ratio}")
    calculated_ratio = (sizes['n_residual_train'] ** (1/d)) / (V ** (1/d))
    logger.info(f"  Calculated Ratio: {calculated_ratio:.2f}")
    logger.info(f"\n  Dataset Sizes:")
    logger.info(f"    n_residual_train: {sizes['n_residual_train']:,}")
    logger.info(f"    n_initial_train:  {sizes['n_initial_train']:,}")
    logger.info(f"    n_boundary_train: {sizes['n_boundary_train']:,}")
    logger.info(f"{'='*60}\n")
    
    return sizes


def generate_and_save_datasets(config: Dict) -> None:
    """
    Generate the training dataset if it doesn't exist.

    There is no eval dataset — all metrics are computed on the ground-truth
    solver's native grid.

    Args:
        config: Configuration dictionary containing problem name,
                sampling ratios, etc.
    """
    problem = config['problem']
    cuda_available = config['cuda'] and torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')

    # Calculate dataset sizes from ratios
    sizes = calculate_dataset_sizes(config)

    # Create datasets directory
    dataset_dir = Path("datasets") / problem
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_path = dataset_dir / "training_data.pt"

    # Dynamically import the solver for the problem
    solver_module = importlib.import_module(f"solvers.{problem}_solver")

    # Generate training data if missing
    if not train_path.exists():
        logger.info(f"Generating training data for {problem}...")
        train_data = solver_module.generate_dataset(
            n_residual=sizes['n_residual_train'],
            n_ic=sizes['n_initial_train'],
            n_bc=sizes['n_boundary_train'],
            device=device,
            config=config
        )
        torch.save(train_data, train_path)
        logger.info(f"  Saved to {train_path}")

        # Create visualizations
        plot_path = dataset_dir / "training_data_visualization.png"
        title = f"{problem} - Training Data"
        plot_dataset(train_data, str(plot_path), title=title)

        stats_path = dataset_dir / "training_data_statistics.png"
        plot_dataset_statistics(train_data, str(stats_path))
        
        # Problem-specific visualization
        try:
            from utils.problem_specific import get_visualization_module
            viz_funcs = get_visualization_module(problem)
            visualize_dataset = viz_funcs[0]
            visualize_dataset(train_data, dataset_dir, config, 'training')
        except ValueError:
            pass  # No custom visualization for this problem
    else:
        logger.info(f"Training data already exists: {train_path}")


def _analytic_ic(problem: str, x: torch.Tensor, pc: Dict) -> torch.Tensor:
    """Return analytical IC values h(x, t=0).  Shape: (N, output_dim)."""
    if problem == 'allen_cahn':
        return x[:, 0:1] ** 2 * torch.cos(math.pi * x[:, 0:1])
    if problem == 'burgers1d':
        return -torch.sin(math.pi * x[:, 0:1])
    if problem == 'kdv':
        return torch.cos(math.pi * x[:, 0:1])
    if problem == 'ks':
        return torch.cos(x[:, 0:1]) * (1.0 + torch.sin(x[:, 0:1]))
    if problem == 'schrodinger':
        real = 2.0 / torch.cosh(x[:, 0:1])
        imag = torch.zeros_like(real)
        return torch.cat([real, imag], dim=1)
    raise ValueError(f"No analytic IC for problem '{problem}'")


def _analytic_bc(problem: str, x: torch.Tensor, t: torch.Tensor,
                 pc: Dict) -> torch.Tensor:
    """Return analytical BC h_gt at boundary points.  Shape: (N, output_dim).

    For problems whose loss hardcodes the target (allen_cahn, burgers1d) or
    uses periodic matching (kdv, ks, schrodinger), the returned values are
    never read by the loss — we fill zeros."""
    if problem == 'schrodinger':
        return torch.zeros(x.shape[0], 2, device=x.device)
    return torch.zeros(x.shape[0], 1, device=x.device)


def _compute_phi_pdf(residuals: torch.Tensor, phi_cfg: Dict) -> torch.Tensor:
    """Compute sampling probability from residual magnitudes using potential Φ.

    Args:
        residuals: Absolute residual values per candidate point, shape (M,).
        phi_cfg: Dict with keys 'phi', 'phi_epsilon', 'phi_power'.

    Returns:
        Normalized probability tensor of shape (M,).
    """
    phi = phi_cfg['phi']
    if phi == 'exponential':
        eps = phi_cfg['phi_epsilon']
        w = torch.exp(residuals / eps)
    elif phi == 'power':
        p = phi_cfg['phi_power']
        w = residuals ** p
    else:  # default: quadratic
        w = residuals ** 2
    w = w + 1e-10  # avoid all-zero
    return w / w.sum()


def build_collar_info(leaf_regions, sigma_fraction: float, plot: bool = False) -> Dict:
    """Pack leaf-region geometry for collar sampling.

    The collar is the set of points covered by >= 2 leaf indicator supports
    Ω̃_j = ∏_d [a_d − δ_d, b_d + δ_d] with δ_d = max(σ_frac·(b_d − a_d), 1e-6)
    (matching BatchedIndicators' smoothstep support).

    Args:
        leaf_regions: RegionDescriptors of the LEAF experts only.
        sigma_fraction: The model's collar fraction (adaptive_pinn.sigma_fraction).
        plot: Whether the caller wants the collar diagnostic plot this resample.
    """
    lower = torch.tensor([list(r.bounds_lower) for r in leaf_regions],
                         dtype=torch.get_default_dtype())
    upper = torch.tensor([list(r.bounds_upper) for r in leaf_regions],
                         dtype=torch.get_default_dtype())
    return {
        'bounds_lower': lower,
        'bounds_upper': upper,
        'sigma_fraction': float(sigma_fraction),
        'plot': bool(plot),
    }


def _collar_support_bounds(collar_info: Dict, device) -> tuple:
    """Expanded (support) bounds of each leaf window: (lower−δ, upper+δ), each (K, D)."""
    lower = collar_info['bounds_lower'].to(device)
    upper = collar_info['bounds_upper'].to(device)
    delta = (collar_info['sigma_fraction'] * (upper - lower)).clamp(min=1e-6)
    return lower - delta, upper + delta


def _support_overlap_count(pts: torch.Tensor, lo_exp: torch.Tensor,
                           hi_exp: torch.Tensor) -> torch.Tensor:
    """Number of leaf supports containing each point. pts (N, D) → (N,) int."""
    inside = ((pts.unsqueeze(1) >= lo_exp.unsqueeze(0))
              & (pts.unsqueeze(1) <= hi_exp.unsqueeze(0))).all(dim=2)  # (N, K)
    return inside.sum(dim=1)


def _sample_collar_points(
    n_points: int,
    collar_info: Dict,
    config: Dict,
    device: torch.device,
) -> tuple:
    """Uniform draw restricted to the collar (>= 2 overlapping leaf supports).

    Rejection sampling from the uniform domain draw; if the collar volume is
    too small to fill the quota in 50 rounds, the remainder is drawn uniformly
    over the whole domain (with a warning).
    """
    problem = config['problem']
    pc = config[problem]
    spatial_dim = pc['spatial_dim']
    spatial_domain = pc['spatial_domain']
    t_min, t_max = pc['temporal_domain']

    lo_exp, hi_exp = _collar_support_bounds(collar_info, device)

    def _uniform_draw(n):
        x = torch.zeros(n, spatial_dim, device=device)
        for d in range(spatial_dim):
            lo, hi = spatial_domain[d]
            x[:, d] = torch.rand(n, device=device) * (hi - lo) + lo
        t = torch.rand(n, 1, device=device) * (t_max - t_min) + t_min
        return x, t

    xs, ts = [], []
    n_left = n_points
    for _ in range(50):
        n_draw = max(2048, 4 * n_left)
        x, t = _uniform_draw(n_draw)
        counts = _support_overlap_count(torch.cat([x, t], dim=1), lo_exp, hi_exp)
        keep = counts >= 2
        n_keep = int(keep.sum().item())
        if n_keep > 0:
            take = min(n_keep, n_left)
            xs.append(x[keep][:take])
            ts.append(t[keep][:take])
            n_left -= take
        if n_left <= 0:
            break
    if n_left > 0:
        logger.warning(f"  [Collar] Drew only {n_points - n_left}/{n_points} "
                       f"collar points after 50 rounds; filling remainder "
                       f"uniformly over the domain.")
        x, t = _uniform_draw(n_left)
        xs.append(x)
        ts.append(t)
    return torch.cat(xs, dim=0), torch.cat(ts, dim=0)


def _save_collar_sampling_plot(blocks: Dict, collar_info: Dict,
                               run_dir, epoch, config: Dict) -> None:
    """Plot the collar region + all residual samples color-coded by source.

    Shows the leaf tiles (black rectangles), the collar (shaded: >= 2
    overlapping supports), and the uniform / adaptive / collar sample blocks.
    1D-spatial problems only.

    Args:
        blocks: {'uniform'|'adaptive'|'collar': (x, t) tensors or None}.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Rectangle
        from pathlib import Path

        output_dir = Path(run_dir) / "collar_sampling"
        output_dir.mkdir(exist_ok=True, parents=True)

        pc = config[config['problem']]
        x_lo, x_hi = pc['spatial_domain'][0]
        t_min, t_max = pc['temporal_domain']

        lower = collar_info['bounds_lower'].cpu().numpy()
        upper = collar_info['bounds_upper'].cpu().numpy()
        delta = np.maximum(collar_info['sigma_fraction'] * (upper - lower), 1e-6)
        lo_exp = lower - delta
        hi_exp = upper + delta

        gx = np.linspace(x_lo, x_hi, 400)
        gt = np.linspace(t_min, t_max, 400)
        XX, TT = np.meshgrid(gx, gt)
        pts = np.stack([XX.ravel(), TT.ravel()], axis=1)  # (N, 2)
        inside = ((pts[:, None, :] >= lo_exp[None]) &
                  (pts[:, None, :] <= hi_exp[None])).all(axis=2)  # (N, K)
        collar_mask = (inside.sum(axis=1) >= 2).reshape(XX.shape)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contourf(XX, TT, collar_mask.astype(float),
                    levels=[0.5, 1.5], colors=['#ff9999'], alpha=0.45)
        for k in range(lower.shape[0]):
            ax.add_patch(Rectangle(
                (lower[k, 0], lower[k, 1]),
                upper[k, 0] - lower[k, 0], upper[k, 1] - lower[k, 1],
                fill=False, edgecolor='black', linewidth=0.8))

        styles = {
            'uniform': ('0.55', 2, 0.35),
            'adaptive': ('tab:blue', 3, 0.5),
            'collar': ('tab:red', 3, 0.6),
        }
        for label, xt in blocks.items():
            if xt is None or xt[0] is None or xt[0].shape[0] == 0:
                continue
            x_np = xt[0][:, 0].detach().cpu().numpy()
            t_np = xt[1][:, 0].detach().cpu().numpy()
            c, s, a = styles.get(label, ('tab:green', 2, 0.4))
            ax.scatter(x_np, t_np, c=c, s=s, alpha=a,
                       label=f'{label} ({len(x_np)})')

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(t_min, t_max)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title(f'Collar sampling (epoch {epoch})')
        ax.legend(loc='upper right', markerscale=3)
        plt.tight_layout()
        plt.savefig(output_dir / f"collar_epoch_{epoch}.png",
                    dpi=100, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        logger.info(f"  [Warning] Failed to save collar sampling plot: {e}")


def _filter_cache_to_region(cached_residuals: list, region) -> list:
    """Filter (x, t, r²) cache tuples to points within region's spatial+temporal bounds."""
    filtered = []
    for x_batch, t_batch, r2_batch in cached_residuals:
        spatial_dim = x_batch.shape[1]
        mask = torch.ones(x_batch.shape[0], dtype=torch.bool, device=x_batch.device)
        for d in range(spatial_dim):
            mask &= (x_batch[:, d] >= region.bounds_lower[d]) & \
                    (x_batch[:, d] < region.bounds_upper[d])
        mask &= (t_batch[:, 0] >= region.bounds_lower[spatial_dim]) & \
                (t_batch[:, 0] < region.bounds_upper[spatial_dim])
        if mask.sum() > 0:
            filtered.append((x_batch[mask], t_batch[mask], r2_batch[mask]))
    return filtered


def _uniform_in_region(region, n_points: int, spatial_dim: int, device) -> tuple:
    """Sample uniform (x, t) points within region's bounds (fallback for empty leaf cache)."""
    x = torch.zeros(n_points, spatial_dim, device=device)
    for d in range(spatial_dim):
        lo, hi = region.bounds_lower[d], region.bounds_upper[d]
        x[:, d] = torch.rand(n_points, device=device) * (hi - lo) + lo
    t_lo = region.bounds_lower[spatial_dim]
    t_hi = region.bounds_upper[spatial_dim]
    t = torch.rand(n_points, 1, device=device) * (t_hi - t_lo) + t_lo
    return x, t


def _sample_adaptive_residual_points(
    cached_residuals: list,
    config: Dict,
    device: torch.device,
    n_points: int,
    phi_cfg: Dict,
    run_dir=None,
    epoch=None,
    causal_state: dict = None,
) -> tuple:
    """Sample residual points biased toward high-residual regions using cached PDE residuals.

    Uses particle-filter-style resampling: draws from cached coordinates weighted by
    their actual PDE residual magnitude, then adds small Gaussian noise to avoid duplicates.

    Args:
        cached_residuals: List of (x, t, r²) tuples from previous epoch's training batches
        config: Full config dict
        device: Target device
        n_points: Number of adaptive points to return
        phi_cfg: Dict with phi type and parameters
        run_dir: Optional path to save diagnostic heatmap
        epoch: Current epoch (for heatmap filename)

    Returns:
        (x_adaptive, t_adaptive): tensors of shape (n_points, spatial_dim) and (n_points, 1)
    """
    problem = config['problem']
    pc = config[problem]
    spatial_dim = pc['spatial_dim']
    spatial_domain = pc['spatial_domain']
    t_min, t_max = pc['temporal_domain']

    # Concatenate all cached residuals
    x_cached_list = []
    t_cached_list = []
    r2_cached_list = []
    
    for x_batch, t_batch, r2_batch in cached_residuals:
        x_cached_list.append(x_batch)
        t_cached_list.append(t_batch)
        r2_cached_list.append(r2_batch)
    
    x_cached = torch.cat(x_cached_list, dim=0)
    t_cached = torch.cat(t_cached_list, dim=0)
    r2_cached = torch.cat(r2_cached_list, dim=0)
    
    # Compute sampling probability from residual magnitudes
    residuals = torch.sqrt(r2_cached + 1e-10)  # sqrt to get |r| from r²
    probs = _compute_phi_pdf(residuals, phi_cfg)
    
    # Draw n_points indices (with replacement) weighted by residual
    indices = torch.multinomial(probs, num_samples=n_points, replacement=True)
    
    # Get selected coordinates
    x_selected = x_cached[indices]
    t_selected = t_cached[indices]
    
    # Add Gaussian noise to avoid exact duplicates
    # Noise std = domain_range / sqrt(n_cached) as a reasonable perturbation scale
    n_cached = len(x_cached)
    for d in range(spatial_dim):
        lo, hi = spatial_domain[d]
        domain_range = hi - lo
        noise_std = domain_range / (n_cached ** 0.5)
        x_selected[:, d] += torch.randn(n_points, device=device) * noise_std
        # Clamp to domain bounds
        x_selected[:, d] = torch.clamp(x_selected[:, d], min=lo, max=hi)
    
    t_range = t_max - t_min
    t_noise_std = t_range / (n_cached ** 0.5)
    t_selected += torch.randn(n_points, 1, device=device) * t_noise_std
    t_selected = torch.clamp(t_selected, min=t_min, max=t_max)
    
    # Diagnostic logging
    r_min = residuals.min().item()
    r_max = residuals.max().item()
    r_mean = residuals.mean().item()
    logger.info(f"  [Resample] Adaptive: residual_pdf min={r_min:.6f}, max={r_max:.6f}, mean={r_mean:.6f} (from cached PDE residuals)")
    
    # Save diagnostic heatmap (only for 1D spatial problems)
    if run_dir is not None and epoch is not None and spatial_dim == 1:
        _save_adaptive_sampling_heatmap(
            x_cached, t_cached, r2_cached,
            x_selected, t_selected,
            run_dir, epoch, config,
            causal_state=causal_state,
        )
    
    return x_selected, t_selected


def _save_adaptive_sampling_heatmap(
    x_cached, t_cached, r2_cached,
    x_sampled, t_sampled,
    run_dir, epoch, config,
    causal_state=None,
):
    """Save diagnostic heatmap for residual distribution (and adaptive sampling if active).

    x_sampled / t_sampled may be None when adaptive sampling is disabled —
    in that case the sampled-points panel is omitted.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path

        output_dir = Path(run_dir) / "adaptive_sampling"
        output_dir.mkdir(exist_ok=True, parents=True)

        x_cached_np = x_cached[:, 0].cpu().numpy()
        t_cached_np = t_cached[:, 0].cpu().numpy()
        r2_cached_np = r2_cached.cpu().numpy()

        log_r2 = np.log10(r2_cached_np + 1e-10)

        has_adaptive = x_sampled is not None and t_sampled is not None
        has_causal = (causal_state is not None and causal_state.get('enabled', False))
        n_panels = 1 + int(has_adaptive) + int(has_causal)
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]
        panel_iter = iter(axes)
        ax1 = next(panel_iter)
        ax2 = next(panel_iter) if has_adaptive else None
        ax3 = next(panel_iter) if has_causal else None

        # Panel 1: raw (pure) PDE residuals (epoch is in the filename)
        sc1 = ax1.scatter(x_cached_np, t_cached_np, c=log_r2, cmap='hot', s=1, alpha=0.6)
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        ax1.set_title('Pure PDE residual')
        plt.colorbar(sc1, ax=ax1, label='log10(r²)')

        # Panel 2: adaptive sampling points (only when adaptive sampling is active)
        if has_adaptive and ax2 is not None:
            x_sampled_np = x_sampled[:, 0].cpu().numpy()
            t_sampled_np = t_sampled[:, 0].cpu().numpy()
            ax2.scatter(x_sampled_np, t_sampled_np, c='blue', s=3, alpha=0.4)
            ax2.set_xlabel('x')
            ax2.set_ylabel('t')
            ax2.set_title('Adaptive sampling by pure residual')

        # Panel 3: causal-weighted residuals
        if has_causal and ax3 is not None:
            num_chunks = causal_state['num_chunks']
            causal_tol = causal_state['tol']

            t_flat = t_cached[:, 0].cpu()
            r2_flat = r2_cached.cpu()
            N = len(t_flat)
            sort_idx = torch.argsort(t_flat)
            r2_sorted = r2_flat[sort_idx]

            chunk_size = max(1, N // num_chunks)
            per_point_weights = torch.ones(N)
            chunk_losses = []
            for i in range(num_chunks):
                start = i * chunk_size
                end = start + chunk_size if i < num_chunks - 1 else N
                chunk_losses.append(r2_sorted[start:end].mean().item())

            cumsum = np.cumsum(chunk_losses)
            shifted = np.concatenate([[0.0], cumsum[:-1]])
            chunk_weights = np.exp(-causal_tol * shifted)

            for i in range(num_chunks):
                start = i * chunk_size
                end = start + chunk_size if i < num_chunks - 1 else N
                per_point_weights[sort_idx[start:end]] = chunk_weights[i]

            weighted_r2 = r2_flat.numpy() * per_point_weights.numpy()
            log_weighted = np.log10(weighted_r2 + 1e-10)
            tol_str = f'{causal_tol:.4g}'
            panel_title = f'Causal-Weighted Residual (ε={tol_str}, epoch {epoch-1})'

            sc3 = ax3.scatter(x_cached_np, t_cached_np, c=log_weighted, cmap='hot', s=1, alpha=0.6)
            ax3.set_xlabel('x')
            ax3.set_ylabel('t')
            ax3.set_title(panel_title)
            plt.colorbar(sc3, ax=ax3, label='log10(w·r²)')
        
        problem = config['problem']
        pc = config[problem]
        spatial_domain = pc['spatial_domain']
        spatial_dim = pc['spatial_dim']
        t_min, t_max = pc['temporal_domain']
        x_lo, x_hi = spatial_domain[0]
        for ax in [a for a in [ax1, ax2, ax3] if a is not None]:
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(t_min, t_max)

        plt.tight_layout()
        output_path = output_dir / f"resample_epoch_{epoch}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        logger.info(f"  [Warning] Failed to save adaptive sampling heatmap: {e}")


def regenerate_training_data(
    config: Dict,
    device: torch.device,
    resample_seed: int = 0,
    cached_residuals: list = None,
    run_dir=None,
    epoch=None,
    causal_state: dict = None,
) -> Dict[str, torch.Tensor]:
    """Lightweight resampling: fresh random coordinates + analytical IC/BC.

    Unlike the initial dataset generation this does **not** run any
    numerical solver — only random (x, t) sampling plus trivial
    analytical formulas for IC and BC ground truth.

    Args:
        config: Full configuration dict
        device: Target device
        resample_seed: Random seed for reproducibility
        cached_residuals: Optional list of (x, t, r²) tuples from previous epoch's training
        run_dir: Optional path for saving diagnostic plots
        epoch: Current epoch (for diagnostic filenames)
        causal_state: Optional causal training state dict for diagnostic heatmaps
    """
    problem = config['problem']
    pc = config[problem]
    spatial_dim = pc['spatial_dim']
    spatial_domain = pc['spatial_domain']
    t_min, t_max = pc['temporal_domain']
    output_dim = pc['output_dim']

    sizes = calculate_dataset_sizes(config)
    n_res = sizes['n_residual_train']
    n_ic = sizes['n_initial_train']
    n_bc = sizes['n_boundary_train']
    N = n_res + n_ic + n_bc

    torch.manual_seed(resample_seed)

    # Adaptive sampling config (read from per-problem config section)
    # merge_problem_features_to_toplevel() copies to top-level for compatibility
    problem_as = config.get(problem, {}).get('adaptive_sampling', {})
    as_cfg = problem_as if problem_as else config['adaptive_sampling']
    has_cache = cached_residuals is not None and len(cached_residuals) > 0
    as_enabled = as_cfg['enabled'] and has_cache
    as_ratio = as_cfg['adaptive_ratio']
    # phi config comes from the same per-problem adaptive_sampling section
    phi_cfg = {
        'phi': as_cfg['phi'],
        'phi_epsilon': as_cfg['phi_epsilon'],
        'phi_power': as_cfg['phi_power'],
    }

    x = torch.zeros(N, spatial_dim, device=device)
    t = torch.zeros(N, 1, device=device)
    h_gt = torch.zeros(N, output_dim, device=device)
    idx = 0

    # --- residual: mix of uniform + adaptive points ---
    if as_enabled:
        n_adaptive = int(n_res * as_ratio)
        n_uniform = n_res - n_adaptive
        # Uniform residual points
        for d in range(spatial_dim):
            lo, hi = spatial_domain[d]
            x[idx:idx + n_uniform, d] = torch.rand(n_uniform, device=device) * (hi - lo) + lo
        t[idx:idx + n_uniform, 0] = torch.rand(n_uniform, device=device) * (t_max - t_min) + t_min
        idx += n_uniform
        # Adaptive residual points
        x_adap, t_adap = _sample_adaptive_residual_points(
            cached_residuals, config, device, n_adaptive, phi_cfg, run_dir, epoch,
            causal_state=causal_state)
        x[idx:idx + n_adaptive] = x_adap
        t[idx:idx + n_adaptive] = t_adap
        idx += n_adaptive
        logger.info(f"  [Resample] n_residual={n_res} (uniform={n_uniform} + adaptive={n_adaptive})")
    else:
        # All uniform (default behavior, also used when no cached residuals)
        for d in range(spatial_dim):
            lo, hi = spatial_domain[d]
            x[idx:idx + n_res, d] = torch.rand(n_res, device=device) * (hi - lo) + lo
        t[idx:idx + n_res, 0] = torch.rand(n_res, device=device) * (t_max - t_min) + t_min
        idx += n_res

    # --- IC: random x at t_min, analytical h_gt ---
    for d in range(spatial_dim):
        lo, hi = spatial_domain[d]
        x[idx:idx + n_ic, d] = torch.rand(n_ic, device=device) * (hi - lo) + lo
    t[idx:idx + n_ic, 0] = t_min
    h_gt[idx:idx + n_ic] = _analytic_ic(problem, x[idx:idx + n_ic], pc)
    idx += n_ic

    # --- BC: boundary coordinates + analytical h_gt ---
    if spatial_dim != 1:
        raise ValueError(
            f"regenerate_training_data supports 1D spatial problems only "
            f"(got spatial_dim={spatial_dim}).")
    x_lo, x_hi = spatial_domain[0]
    n_left = n_bc // 2
    n_right = n_bc - n_left
    t_bc = torch.rand(max(n_left, n_right), device=device) * (t_max - t_min) + t_min
    x[idx:idx + n_left, 0] = x_lo
    t[idx:idx + n_left, 0] = t_bc[:n_left]
    idx += n_left
    x[idx:idx + n_right, 0] = x_hi
    t[idx:idx + n_right, 0] = t_bc[:n_right]
    idx += n_right

    bc_start = n_res + n_ic
    h_gt[bc_start:] = _analytic_bc(problem, x[bc_start:], t[bc_start:], pc)

    # --- masks ---
    mask_res = torch.zeros(N, dtype=torch.bool, device=device)
    mask_res[:n_res] = True
    mask_ic = torch.zeros(N, dtype=torch.bool, device=device)
    mask_ic[n_res:n_res + n_ic] = True
    mask_bc = torch.zeros(N, dtype=torch.bool, device=device)
    mask_bc[n_res + n_ic:] = True

    return {
        "x": x, "t": t, "h_gt": h_gt,
        "mask": {"residual": mask_res, "IC": mask_ic, "BC": mask_bc},
    }


def sample_residual_points(
    config: Dict,
    device: torch.device,
    n_res: int,
    cached_residuals: list = None,
    run_dir=None,
    epoch=None,
    causal_state: dict = None,
    collar_info: Dict = None,
):
    """Sample n_res residual (x, t) pairs.

    The total is split into up to three blocks (each ratio taken out of the
    same n_res budget): uniform over the domain, adaptive (residual-weighted,
    when adaptive_sampling is enabled and a cache is supplied), and collar
    (uniform within the >= 2-overlapping-supports region, when
    sampling.collar_data_ratio > 0 and ``collar_info`` is supplied).

    Returns:
        Tuple (x_res, t_res) each of shape (n_res, spatial_dim/1).
    """
    problem = config['problem']
    pc = config[problem]
    spatial_dim = pc['spatial_dim']
    spatial_domain = pc['spatial_domain']
    t_min, t_max = pc['temporal_domain']

    problem_as = config.get(problem, {}).get('adaptive_sampling', {})
    as_cfg = problem_as if problem_as else config['adaptive_sampling']
    has_cache = cached_residuals is not None and len(cached_residuals) > 0
    as_enabled = as_cfg['enabled'] and has_cache
    as_ratio = as_cfg['adaptive_ratio']
    phi_cfg = {
        'phi': as_cfg['phi'],
        'phi_epsilon': as_cfg['phi_epsilon'],
        'phi_power': as_cfg['phi_power'],
    }

    collar_ratio = (config.get('sampling', {}) or {}).get(
        'collar_data_ratio', 0.0) or 0.0
    collar_active = collar_info is not None and collar_ratio > 0

    n_adaptive = int(n_res * as_ratio) if as_enabled else 0
    n_collar = int(n_res * collar_ratio) if collar_active else 0
    if n_adaptive + n_collar > n_res:
        logger.warning(f"  [Sampling] adaptive_ratio + collar_data_ratio > 1; "
                       f"trimming collar share to fit n_res={n_res}.")
        n_collar = max(0, n_res - n_adaptive)
    n_uniform = n_res - n_adaptive - n_collar

    x_res = torch.zeros(n_res, spatial_dim, device=device)
    t_res = torch.zeros(n_res, 1, device=device)
    idx = 0

    for d in range(spatial_dim):
        lo, hi = spatial_domain[d]
        x_res[idx:idx + n_uniform, d] = (
            torch.rand(n_uniform, device=device) * (hi - lo) + lo
        )
    t_res[idx:idx + n_uniform, 0] = (
        torch.rand(n_uniform, device=device) * (t_max - t_min) + t_min
    )
    idx += n_uniform

    if n_adaptive > 0:
        x_adap, t_adap = _sample_adaptive_residual_points(
            cached_residuals, config, device, n_adaptive, phi_cfg,
            run_dir, epoch, causal_state=causal_state)
        x_res[idx:idx + n_adaptive] = x_adap
        t_res[idx:idx + n_adaptive] = t_adap
        idx += n_adaptive

    if n_collar > 0:
        x_col, t_col = _sample_collar_points(
            n_collar, collar_info, config, device)
        x_res[idx:idx + n_collar] = x_col
        t_res[idx:idx + n_collar] = t_col
        logger.info(f"  [Resample] Collar: {n_collar} points "
                    f"(uniform={n_uniform}, adaptive={n_adaptive})")
        if (collar_info.get('plot') and run_dir is not None
                and epoch is not None and spatial_dim == 1):
            _save_collar_sampling_plot(
                {
                    'uniform': (x_res[:n_uniform], t_res[:n_uniform]),
                    'adaptive': ((x_res[n_uniform:n_uniform + n_adaptive],
                                  t_res[n_uniform:n_uniform + n_adaptive])
                                 if n_adaptive > 0 else None),
                    'collar': (x_col, t_col),
                },
                collar_info, run_dir, epoch, config)

    return x_res, t_res


def resample_residual_inplace(
    train_data: Dict,
    config: Dict,
    device: torch.device,
    resample_seed: int = 0,
    cached_residuals: list = None,
    run_dir=None,
    epoch=None,
    causal_state: dict = None,
    collar_info: Dict = None,
) -> Dict:
    """Update only the residual rows in train_data with freshly sampled points.

    IC/BC coordinates and their h_gt values are left untouched.  This is the
    cheap per-epoch resample path; the full regenerate_training_data() is still
    used on initial build and after tree spawning.

    Returns train_data (modified in-place) for convenience.
    """
    res_mask = train_data['mask']['residual']
    n_res = int(res_mask.sum().item())
    torch.manual_seed(resample_seed)
    x_res, t_res = sample_residual_points(
        config, device, n_res, cached_residuals,
        run_dir, epoch, causal_state, collar_info=collar_info)
    train_data['x'][res_mask] = x_res
    train_data['t'][res_mask] = t_res
    return train_data


def load_dataset(
    path: str,
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """
    Load a dataset from disk.

    Args:
        path: Path to the .pt file
        device: Device to load tensors to (if None, keeps original device)

    Returns:
        Dictionary with dataset tensors
    """
    data = torch.load(path)

    if device is not None:
        # Move all tensors to specified device
        data['x'] = data['x'].to(device)
        data['t'] = data['t'].to(device)
        data['h_gt'] = data['h_gt'].to(device)
        data['mask']['residual'] = data['mask']['residual'].to(device)
        data['mask']['IC'] = data['mask']['IC'].to(device)
        data['mask']['BC'] = data['mask']['BC'].to(device)

    return data
