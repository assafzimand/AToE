"""Generate "perfect tree" visualizations and model-reconstruction
JSON files for every PDE problem.

For each problem, fits a full decision tree on ground-truth data
(from eval_data.pt), prunes it, and produces:

  For non-time-marching problems (3-panel PNG):
     - Original tree (before pruning)
     - After pruning (M regions)
     - Tree hierarchy diagram

  For time-marching problems (2-panel PNG):
     - All original trees concatenated (before pruning, all windows)
     - All accepted regions from all windows (after pruning)

  Plus a JSON file containing the full tree structure for reconstruction.
"""

import json
import sys
import importlib
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import deque
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive.region_detector import RegionDetector  # noqa: E402
from adaptive.visualization import prepare_ground_truth_grid  # noqa: E402
from utils.dataset_gen import calculate_dataset_sizes  # noqa: E402
from utils.plot_io import save_png  # noqa: E402
from trainer.time_marching import compute_m_per_window  # noqa: E402
from trainer.utils import native_ground_truth_grid  # noqa: E402


class _NumpySafeEncoder(json.JSONEncoder):
    """Handle numpy types that stdlib json can't serialize."""
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def load_config(plan_path: Path) -> dict:
    """Load experiments_plan.yaml and return the base_config."""
    with open(plan_path, 'r', encoding='utf-8') as f:
        plan = yaml.safe_load(f)
    return plan.get('base_config', {})


def get_problem_list(base_cfg: dict) -> list:
    """Return all problem names that have a sub-config with spatial_dim."""
    skip = {'sampling', 'adaptive_pinn'}
    problems = []
    for key, val in base_cfg.items():
        if isinstance(val, dict) and 'spatial_dim' in val and key not in skip:
            problems.append(key)
    return sorted(problems)


def ensure_eval_data(problem: str, base_cfg: dict) -> dict:
    """Load eval_data.pt, generating it if missing."""
    eval_path = Path('datasets') / problem / 'eval_data.pt'
    if eval_path.exists():
        return torch.load(eval_path, map_location='cpu')

    print(f"  Generating eval data for {problem}...")
    cfg = dict(base_cfg)
    cfg['problem'] = problem
    cfg['cuda'] = False
    sizes = calculate_dataset_sizes(cfg)
    solver = importlib.import_module(f'solvers.{problem}_solver')
    eval_data = solver.generate_dataset(
        n_residual=sizes['n_residual_eval'],
        n_ic=sizes['n_initial_eval'],
        n_bc=sizes['n_boundary_eval'],
        device=torch.device('cpu'),
        config=cfg,
    )
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(eval_data, eval_path)
    return eval_data


def build_domain_bounds(problem_cfg: dict) -> dict:
    """Build domain_bounds dict from problem config."""
    lower = [sd[0] for sd in problem_cfg['spatial_domain']]
    lower.append(problem_cfg['temporal_domain'][0])
    upper = [sd[1] for sd in problem_cfg['spatial_domain']]
    upper.append(problem_cfg['temporal_domain'][1])
    return {'lower': lower, 'upper': upper}


def extract_xy(eval_data: dict, output_dim: int):
    """Extract (X, y) arrays from eval data for tree fitting."""
    x = eval_data['x'].cpu().numpy() if isinstance(eval_data['x'], torch.Tensor) else eval_data['x']
    t = eval_data['t'].cpu().numpy() if isinstance(eval_data['t'], torch.Tensor) else eval_data['t']

    if x.ndim == 1:
        x = x[:, None]
    t = t.ravel()[:, None]
    X = np.hstack([x, t])

    for key in ('h_gt', 'h', 'u'):
        if key in eval_data:
            y = eval_data[key]
            break
    else:
        raise ValueError(f"No GT key found in eval_data. Keys: {list(eval_data.keys())}")

    y = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.ravel()
    return X, y


def build_symmetric_grid_data(eval_data, domain_bounds, resolution=200):
    """Interpolate the GT onto a symmetric regular grid for tree fitting.

    Random eval points put sklearn's split candidates (sample midpoints) at
    asymmetric, draw-dependent locations; a regular grid keeps candidate
    splits symmetric under the domain's reflections. Interpolates each
    output channel separately (unlike prepare_ground_truth_grid, which
    collapses multi-output to a norm for display).

    Returns (X_grid, y_grid) or (None, None) if the domain is not 2D.
    """
    from scipy.interpolate import griddata
    if len(domain_bounds['lower']) != 2:
        return None, None
    X_pts, y_pts = extract_xy(eval_data, output_dim=None)
    x_min, t_min = domain_bounds['lower']
    x_max, t_max = domain_bounds['upper']
    gx = np.linspace(x_min, x_max, resolution)
    gt = np.linspace(t_min, t_max, resolution)
    XX, TT = np.meshgrid(gx, gt, indexing='ij')
    X_grid = np.column_stack([XX.ravel(), TT.ravel()])

    y2 = y_pts if y_pts.ndim > 1 else y_pts[:, None]
    channels = []
    for c in range(y2.shape[1]):
        v = griddata(X_pts, y2[:, c], (XX, TT), method='linear')
        nan_mask = np.isnan(v)
        if nan_mask.any():
            v_nn = griddata(X_pts, y2[:, c], (XX, TT), method='nearest')
            v[nan_mask] = v_nn[nan_mask]
        channels.append(v.ravel())
    y_grid = np.column_stack(channels)
    if y_grid.shape[1] == 1:
        y_grid = y_grid.ravel()
    return X_grid, y_grid


def build_native_grid_data(problem, base_cfg):
    """Tree-fit data + heatmap grid from the solver's NATIVE solution.

    Same GT source as the trainer's rel-L2 reporting
    (trainer.utils.compute_native_grid_metrics): the exact solution on the
    solver's own grid, with no random eval sample and no interpolation.
    Sparse scattered eval points smear sharp features (e.g. thin KdV
    solitons), which distorts both the plots and the wavelet norms that
    drive the tree construction.

    Returns (X, y, (gt_grid, grid_x, grid_t)) or None if the solver has no
    cached native solution (callers fall back to eval-data interpolation).
    """
    cfg = dict(base_cfg)
    cfg['problem'] = problem
    res = native_ground_truth_grid(cfg)
    if res is None:
        return None
    gt_grid, grid_x, grid_t = res
    XX, TT = np.meshgrid(grid_x, grid_t, indexing='ij')
    X = np.column_stack([XX.ravel(), TT.ravel()])
    if gt_grid.ndim == 3:
        y = gt_grid.reshape(-1, gt_grid.shape[2])
    else:
        y = np.asarray(gt_grid).ravel()
    return X, y, (gt_grid, grid_x, grid_t)


def fit_and_get_all_nodes(
    X, y, max_depth, min_samples_leaf, M, variable_for_node_accept,
    domain_bounds,
    epsilon_node_acceptance=0.0,
):
    """Fit tree, prune, return visualization + reconstruction data.

    ``domain_bounds`` is the TRUE domain box ({'lower': [...], 'upper':
    [...]}); region boxes recurse sklearn thresholds down from it and are
    never inferred from the sampled points.

    Returns:
        node_dicts: list of dicts for ALL non-root nodes (for plots)
        accepted_ids: set of accepted tree node ids
        bfs_accepted: list of dicts for accepted nodes in BFS order,
            each with 'parent_tree_node_id' (nearest accepted
            ancestor, -1 for children of root). This is the data
            needed to reconstruct an adaptive model.
        children_left_arr: sklearn children_left array (to detect
            leaves of the pruned tree for AToELeaves)
    """
    detector = RegionDetector(
        n_estimators=1,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        domain_bounds=domain_bounds,
    )

    accepted_nodes, depth_stats = detector.fit_full_tree_and_prune(
        X, y,
        M=M,
        variable_for_node_accept=variable_for_node_accept,
        verbose=True,
        retain_siblings=True,
        epsilon_node_acceptance=epsilon_node_acceptance,
    )
    accepted_ids = {n.node_id for n, _ in accepted_nodes}

    tree = detector.rf.estimators_[0].tree_
    children_left = tree.children_left
    children_right = tree.children_right

    _node_depth = {0: 0}
    _parent_map = {}
    bfs = deque([0])
    while bfs:
        nid = bfs.popleft()
        for child in (children_left[nid], children_right[nid]):
            if child != -1:
                _parent_map[child] = nid
                _node_depth[child] = _node_depth[nid] + 1
                bfs.append(child)

    all_wn = detector.compute_wavelet_norms(X=X, y=y)

    node_dicts = []
    for nd in all_wn:
        if nd.node_id == 0:
            continue
        node_dicts.append({
            'node_id': nd.node_id,
            'parent_node_id': _parent_map.get(nd.node_id, -1),
            'wavelet_norm_squared': nd.wavelet_norm_squared,
            'new_wavelet_norm_squared': nd.new_wavelet_norm_squared,
            'smoothness_alpha': nd.smoothness_alpha,
            'smoothness_r2': nd.smoothness_r2,
            'smoothness_n_levels': nd.smoothness_n_levels,
            'n_samples': nd.n_samples,
            'is_leaf': bool(nd.is_leaf),
            'bounds_lower': nd.bounds_lower,
            'bounds_upper': nd.bounds_upper,
            'accepted': bool(nd.node_id in accepted_ids),
            'tree_depth': _node_depth.get(nd.node_id, -1),
        })

    # BFS-ordered accepted nodes with parent_tree_node_id
    # This mirrors what fit_full_tree_and_prune returns
    bfs_accepted = []
    for node_info, parent_tree_nid in accepted_nodes:
        is_pruned_leaf = (
            children_left[node_info.node_id] == -1
            or children_left[node_info.node_id] not in accepted_ids
        )
        bfs_accepted.append({
            'node_id': node_info.node_id,
            'parent_tree_node_id': parent_tree_nid,
            'bounds_lower': node_info.bounds_lower,
            'bounds_upper': node_info.bounds_upper,
            'wavelet_norm_squared': node_info.wavelet_norm_squared,
            'new_wavelet_norm_squared': node_info.new_wavelet_norm_squared,
            'smoothness_alpha': node_info.smoothness_alpha,
            'n_samples': node_info.n_samples,
            'tree_depth': _node_depth.get(
                node_info.node_id, -1),
            'is_leaf_in_pruned_tree': is_pruned_leaf,
        })

    return (
        node_dicts,
        accepted_ids,
        bfs_accepted,
        children_left,
    )


def _plot_regions_panel(ax, regions_dicts, domain_bounds, gt_grid, grid_x, grid_t, title):
    """Draw GT heatmap + region outlines on a given axes."""
    x_min, t_min = domain_bounds['lower'][:2]
    x_max, t_max = domain_bounds['upper'][:2]

    if gt_grid is not None and grid_x is not None and grid_t is not None:
        if gt_grid.ndim == 3:
            display = np.linalg.norm(gt_grid, axis=2)
        else:
            display = gt_grid
        T, X = np.meshgrid(grid_t, grid_x)
        im = ax.pcolormesh(X, T, display, shading='auto', cmap='viridis', alpha=0.7, zorder=0)
        plt.colorbar(im, ax=ax, shrink=0.7)

    for nd in regions_dicts:
        bl = nd['bounds_lower']
        bu = nd['bounds_upper']
        rx_min, rt_min = bl[0], bl[-1]
        rx_max, rt_max = bu[0], bu[-1]
        rect = patches.Rectangle(
            (rx_min, rt_min), rx_max - rx_min, rt_max - rt_min,
            linewidth=1.0, edgecolor='black', facecolor='none', zorder=10,
        )
        ax.add_patch(rect)

    pad = 0.03
    xr = x_max - x_min
    tr = t_max - t_min
    ax.set_xlim(x_min - pad * xr, x_max + pad * xr)
    ax.set_ylim(t_min - pad * tr, t_max + pad * tr)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title(title, fontsize=11)
    ax.set_aspect('auto')


def _plot_hierarchy_panel(ax, all_nodes, variable_for_node_accept):
    """Dendrogram-style tree hierarchy colored by the configured metric."""
    if not all_nodes:
        ax.set_title('Tree hierarchy')
        return

    children_map = {}
    node_map = {}
    for n in all_nodes:
        nid = n['node_id']
        pid = n.get('parent_node_id', -1)
        node_map[nid] = n
        children_map.setdefault(pid, []).append(n)

    root_children = children_map.get(0, [])
    if not root_children:
        root_children = [n for n in all_nodes if n.get('parent_node_id', -1) == -1]
    if not root_children:
        ax.set_title('Tree hierarchy')
        return

    leaf_counter = [0]
    positions = {}

    def _layout(nid):
        kids = children_map.get(nid, [])
        depth = node_map[nid]['tree_depth']
        if not kids:
            x = leaf_counter[0]
            leaf_counter[0] += 1
            positions[nid] = (x, depth)
            return x
        child_xs = [_layout(c['node_id']) for c in sorted(kids, key=lambda c: c['node_id'])]
        x = np.mean(child_xs)
        positions[nid] = (x, depth)
        return x

    rc_xs = [_layout(c['node_id']) for c in sorted(root_children, key=lambda c: c['node_id'])]
    root_x = np.mean(rc_xs)
    positions[0] = (root_x, 0)

    max_depth = max(n['tree_depth'] for n in all_nodes) if all_nodes else 1

    # Determine which metric to use for coloring and extract values
    if variable_for_node_accept == 'smoothness':
        metric_values = [n['smoothness_alpha'] for n in all_nodes
                        if n.get('smoothness_alpha') is not None]
        cmap = plt.get_cmap('RdYlGn')  # red=rough (low α), green=smooth (high α)
        metric_label = 'Smoothness α'
        def get_metric(node):
            return node.get('smoothness_alpha')
    elif variable_for_node_accept == 'norm':
        metric_values = [n['wavelet_norm_squared'] for n in all_nodes
                        if n.get('wavelet_norm_squared') is not None]
        cmap = plt.get_cmap('coolwarm')  # blue=low, red=high
        metric_label = 'Wavelet Norm²'
        def get_metric(node):
            return node.get('wavelet_norm_squared')
    elif variable_for_node_accept == 'new_norm':
        metric_values = [n['new_wavelet_norm_squared'] for n in all_nodes
                        if n.get('new_wavelet_norm_squared') is not None]
        cmap = plt.get_cmap('coolwarm')  # blue=low, red=high
        metric_label = 'New Norm²'
        def get_metric(node):
            return node.get('new_wavelet_norm_squared')
    else:
        metric_values = []
        cmap = plt.get_cmap('viridis')
        metric_label = 'Value'
        def get_metric(node):
            return 0.0

    # Build colormap normalization
    if metric_values:
        norm = plt.Normalize(vmin=min(metric_values), vmax=max(metric_values))
    else:
        norm = plt.Normalize(vmin=0.0, vmax=1.0)

    # Draw edges
    for n in all_nodes:
        nid = n['node_id']
        pid = n.get('parent_node_id', -1)
        if pid >= 0 and pid in positions and nid in positions:
            px, py = positions[pid]
            cx, cy = positions[nid]
            edge_alpha = 0.7 if n['accepted'] else 0.15
            ax.plot([px, cx], [-py, -cy], 'k-', alpha=edge_alpha, lw=0.6, zorder=1)

    ax.scatter(root_x, 0, c='white', s=60, edgecolors='black', linewidths=1.5, zorder=6, marker='s')

    # Draw nodes — gray when metric value is None
    for n in all_nodes:
        nid = n['node_id']
        if nid not in positions:
            continue
        x, y = positions[nid]
        metric_val = get_metric(n)
        if metric_val is None:
            color = ['#bbbbbb'] if n['accepted'] else ['#dddddd']
            ec = 'black' if n['accepted'] else 'gray'
            node_alpha = 1.0 if n['accepted'] else 0.3
        else:
            color = [cmap(norm(metric_val))]
            ec = 'black' if n['accepted'] else 'gray'
            node_alpha = 1.0 if n['accepted'] else 0.25

        size = 35 if n['accepted'] else 15
        lw = 0.8 if n['accepted'] else 0.3
        ax.scatter(x, -y, c=color, s=size, edgecolors=ec,
                   linewidths=lw, alpha=node_alpha,
                   zorder=5 if n['accepted'] else 4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=metric_label, shrink=0.7)

    ax.set_ylabel('Depth')
    ax.set_yticks([-d for d in range(max_depth + 1)])
    ax.set_yticklabels([str(d) for d in range(max_depth + 1)])
    ax.set_xticks([])
    
    ax.set_title('Tree hierarchy', fontsize=11)
    ax.grid(True, alpha=0.15, axis='y')


def build_problem_tree_data(
    problem, domain_bounds,
    max_depth, min_samples_leaf, M,
    bfs_accepted, node_dicts,
    epsilon_node_acceptance=0.0,
):
    """Build the dict for one problem's perfect tree."""
    n_leaves = sum(
        1 for n in bfs_accepted
        if n['is_leaf_in_pruned_tree']
    )
    return {
        'domain_bounds': domain_bounds,
        'tree_params': {
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'M': M,
            'epsilon_node_acceptance': epsilon_node_acceptance,
        },
        'summary': {
            'total_nodes': len(node_dicts),
            'accepted_nodes': len(bfs_accepted),
            'pruned_tree_leaves': n_leaves,
        },
        'accepted_nodes_bfs': bfs_accepted,
        'all_nodes': node_dicts,
    }


def process_problem_with_time_marching(
    problem: str, base_cfg: dict, output_dir: Path
):
    """Generate perfect trees for time-marching scenario.
    
    Creates one tree per window with window-specific M values,
    then combines all regions into a single 2-panel visualization.
    
    Returns the tree data dict for this problem, or None if skipped.
    """
    print(f"\n{'='*60}")
    print(f"  Problem: {problem} (TIME MARCHING)")
    print(f"{'='*60}")

    problem_cfg = base_cfg[problem]
    adaptive_cfg = base_cfg.get('adaptive_pinn', {})
    tm_cfg = problem_cfg.get('time_marching', {})

    max_depth = adaptive_cfg.get('tree_max_depth', 30)
    min_samples_leaf = adaptive_cfg.get('tree_min_samples_leaf', 10)
    global_M = adaptive_cfg.get('M_experts_num', 40)
    output_dim = problem_cfg.get('output_dim', 1)
    variable_for_node_accept = adaptive_cfg.get('variable_for_node_accept', 'norm')
    epsilon_node_acceptance = adaptive_cfg.get('epsilon_node_acceptance', 0.0)

    num_windows = tm_cfg.get('num_windows', 5)
    m_distribution = tm_cfg.get('m_distribution', 'equal')

    # Compute M per window
    m_per_window = compute_m_per_window(global_M, num_windows, m_distribution)

    print(f"  max_depth={max_depth}, min_samples_leaf={min_samples_leaf}, "
          f"epsilon_node_acceptance={epsilon_node_acceptance}")
    print(f"  num_windows={num_windows}, m_distribution={m_distribution}")
    print(f"  global_M={global_M}, M per window: {m_per_window}")

    domain_bounds = build_domain_bounds(problem_cfg)

    if len(domain_bounds['lower']) != 2:
        print(f"  Skipping {problem}: only 2D (x,t) domains supported.")
        return None

    # Prefer the solver's native solution grid (same GT source as the
    # trainer's rel-L2 reporting); fall back to interpolating the scattered
    # eval sample only if no native grid exists for this problem.
    native = build_native_grid_data(problem, base_cfg)
    if native is not None:
        X_full, y_full, native_heatmap = native
        print("  GT source: native solver grid")
    else:
        eval_data = ensure_eval_data(problem, base_cfg)
        X_full, y_full = build_symmetric_grid_data(eval_data, domain_bounds)
        native_heatmap = None
        print("  GT source: interpolated eval sample (no native grid)")
    print(f"  Full data: X={X_full.shape}, y={y_full.shape if hasattr(y_full, 'shape') else '?'}, "
          f"min_samples_leaf={min_samples_leaf}")

    # Compute window boundaries
    t_min, t_max = problem_cfg['temporal_domain']
    dt = (t_max - t_min) / num_windows
    
    # Collect nodes from all windows
    all_window_nodes = []  # All raw tree nodes from all windows
    all_accepted_nodes = []  # All accepted nodes from all windows
    all_bfs_accepted = []  # For JSON output
    
    for win_idx in range(num_windows):
        win_t_start = t_min + win_idx * dt
        win_t_end = t_min + (win_idx + 1) * dt
        win_M = m_per_window[win_idx]
        
        print(f"\n  Window {win_idx}: t in [{win_t_start:.4f}, {win_t_end:.4f}], M={win_M}")
        
        # Filter data to this window's temporal range
        t_col = X_full[:, -1]  # Last column is t
        mask = (t_col >= win_t_start) & (t_col < win_t_end)
        # Include endpoint for last window
        if win_idx == num_windows - 1:
            mask = (t_col >= win_t_start) & (t_col <= win_t_end)
        
        X_win = X_full[mask]
        y_win = y_full[mask] if y_full.ndim == 1 else y_full[mask]
        
        if len(X_win) < min_samples_leaf * 2:
            print(f"    Skipping window {win_idx}: too few samples ({len(X_win)})")
            continue
        
        print(f"    Window data: {len(X_win)} samples")
        
        # Fit tree for this window: the root box IS the exact window box
        # (domain in space, [win_t_start, win_t_end] in time) — bounds are
        # never derived from the sampled points.
        win_bounds = {
            'lower': list(domain_bounds['lower'][:-1]) + [win_t_start],
            'upper': list(domain_bounds['upper'][:-1]) + [win_t_end],
        }
        try:
            (node_dicts, accepted_ids,
             bfs_accepted, children_left) = fit_and_get_all_nodes(
                X_win, y_win, max_depth, min_samples_leaf, win_M, variable_for_node_accept,
                domain_bounds=win_bounds,
                epsilon_node_acceptance=epsilon_node_acceptance,
            )

            # Tag nodes with window index for later reference
            for nd in node_dicts:
                nd['window_idx'] = win_idx
            for nd in bfs_accepted:
                nd['window_idx'] = win_idx
            
            all_window_nodes.extend(node_dicts)
            all_accepted_nodes.extend([n for n in node_dicts if n['accepted']])
            all_bfs_accepted.extend(bfs_accepted)
            
            print(f"    Window {win_idx}: {len(node_dicts)} total nodes, "
                  f"{len([n for n in node_dicts if n['accepted']])} accepted")
                  
        except Exception as e:
            print(f"    Error in window {win_idx}: {e}")
            continue
    
    if not all_accepted_nodes:
        print(f"  No accepted nodes found across all windows!")
        return None
    
    n_total = len(all_window_nodes)
    n_accepted = len(all_accepted_nodes)
    print(f"\n  Combined: {n_total} total nodes, {n_accepted} accepted across {num_windows} windows")
    
    # Count pruned tree leaves (nodes that are leaves in their window's pruned tree)
    n_pruned_leaves = sum(
        1 for n in all_bfs_accepted
        if n.get('is_leaf_in_pruned_tree', False)
    )
    
    # Build tree data for JSON
    tree_data = {
        'domain_bounds': domain_bounds,
        'tree_params': {
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'global_M': global_M,
            'epsilon_node_acceptance': epsilon_node_acceptance,
            'num_windows': num_windows,
            'm_distribution': m_distribution,
            'm_per_window': m_per_window,
        },
        'summary': {
            'total_nodes': n_total,
            'accepted_nodes': n_accepted,
            'pruned_tree_leaves': n_pruned_leaves,
        },
        'accepted_nodes_bfs': all_bfs_accepted,
        'all_nodes': all_window_nodes,
    }
    
    # Generate 2-panel plot on FULL domain
    if native_heatmap is not None:
        gt_grid, grid_x, grid_t = native_heatmap
    else:
        gt_grid, grid_x, grid_t = prepare_ground_truth_grid(
            eval_data, domain_bounds, resolution=150)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    _plot_regions_panel(
        axes[0], all_window_nodes, domain_bounds,
        gt_grid, grid_x, grid_t,
        'Original trees')
    _plot_regions_panel(
        axes[1], all_accepted_nodes, domain_bounds,
        gt_grid, grid_x, grid_t,
        'After pruning')

    plt.tight_layout()

    out_path = save_png(
        output_dir / (
            f'perfect_tree_{problem}_W{num_windows}_{m_distribution}'
            f'_M{global_M}_acc{n_accepted}.png'),
        fig=fig)
    plt.close()
    print(f"  Saved plot: {out_path}")
    
    return tree_data


def process_problem(
    problem: str, base_cfg: dict, output_dir: Path
):
    """Generate the visualization + return tree data dict.

    Dispatches to time-marching handler if enabled, otherwise
    generates standard 3-panel image.
    
    Returns the tree data dict for this problem, or None if
    the problem was skipped.
    """
    problem_cfg = base_cfg[problem]
    
    # Check if time marching is enabled for this problem
    tm_cfg = problem_cfg.get('time_marching', {})
    if tm_cfg.get('enabled', False):
        return process_problem_with_time_marching(problem, base_cfg, output_dir)
    
    # Standard (non-time-marching) processing
    print(f"\n{'='*60}")
    print(f"  Problem: {problem}")
    print(f"{'='*60}")

    adaptive_cfg = base_cfg.get('adaptive_pinn', {})

    max_depth = adaptive_cfg.get('tree_max_depth', 30)
    min_samples_leaf = adaptive_cfg.get(
        'tree_min_samples_leaf', 10)
    M = adaptive_cfg.get('M_experts_num', 40)
    output_dim = problem_cfg.get('output_dim', 1)
    
    # Read variable_for_node_accept
    variable_for_node_accept = adaptive_cfg.get('variable_for_node_accept', 'norm')
    epsilon_node_acceptance = adaptive_cfg.get('epsilon_node_acceptance', 0.0)

    print(
        f"  max_depth={max_depth}, "
        f"min_samples_leaf={min_samples_leaf}, "
        f"M={M}, "
        f"variable_for_node_accept={variable_for_node_accept}, "
        f"epsilon_node_acceptance={epsilon_node_acceptance}")

    domain_bounds = build_domain_bounds(problem_cfg)

    if len(domain_bounds['lower']) != 2:
        print(
            f"  Skipping {problem}: "
            f"only 2D (x,t) domains supported.")
        return None

    # Prefer the solver's native solution grid (same GT source as the
    # trainer's rel-L2 reporting); fall back to interpolating the scattered
    # eval sample only if no native grid exists for this problem.
    native = build_native_grid_data(problem, base_cfg)
    if native is not None:
        X, y, native_heatmap = native
        print("  GT source: native solver grid")
    else:
        eval_data = ensure_eval_data(problem, base_cfg)
        X, y = build_symmetric_grid_data(eval_data, domain_bounds)
        native_heatmap = None
        print("  GT source: interpolated eval sample (no native grid)")
    print(f"  Data: X={X.shape}, y="
          f"{y.shape if hasattr(y, 'shape') else '?'}")

    (node_dicts, accepted_ids,
     bfs_accepted, children_left) = fit_and_get_all_nodes(
        X, y, max_depth, min_samples_leaf, M, variable_for_node_accept,
        domain_bounds=domain_bounds,
        epsilon_node_acceptance=epsilon_node_acceptance,
    )

    # -- Build tree data for unified JSON --
    tree_data = build_problem_tree_data(
        problem, domain_bounds,
        max_depth, min_samples_leaf, M,
        bfs_accepted, node_dicts,
        epsilon_node_acceptance=epsilon_node_acceptance,
    )

    # -- Generate 3-panel plot --
    if native_heatmap is not None:
        gt_grid, grid_x, grid_t = native_heatmap
    else:
        gt_grid, grid_x, grid_t = prepare_ground_truth_grid(
            eval_data, domain_bounds, resolution=150)

    all_region_dicts = list(node_dicts)
    accepted_region_dicts = [
        n for n in node_dicts if n['accepted']]

    n_total = len(all_region_dicts)
    n_accepted = len(accepted_region_dicts)
    print(f"  Nodes: {n_total} total, "
          f"{n_accepted} accepted (target M={M})")

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    _plot_regions_panel(
        axes[0], all_region_dicts, domain_bounds,
        gt_grid, grid_x, grid_t,
        'Original tree')
    _plot_regions_panel(
        axes[1], accepted_region_dicts, domain_bounds,
        gt_grid, grid_x, grid_t,
        'After pruning')
    _plot_hierarchy_panel(
        axes[2], node_dicts, variable_for_node_accept)

    plt.tight_layout()

    out_path = save_png(
        output_dir / (
            f'perfect_tree_{problem}_depth{max_depth}'
            f'_M{M}_acc{n_accepted}.png'),
        fig=fig)
    plt.close()
    print(f"  Saved plot: {out_path}")

    return tree_data


def main():
    plan_path = (
        Path(__file__).resolve().parent.parent
        / 'experiments_plan.yaml')
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(plan_path)
    problems = get_problem_list(base_cfg)

    print(f"Found {len(problems)} problems: {problems}")
    print(f"Output directory: {output_dir}")

    all_trees = {}
    for problem in problems:
        try:
            tree_data = process_problem(
                problem, base_cfg, output_dir)
            if tree_data is not None:
                all_trees[problem] = tree_data
        except Exception as e:
            print(f"\n  ERROR processing {problem}: {e}")
            import traceback
            traceback.print_exc()

    # Save unified JSON with all problems
    json_path = output_dir / 'perfect_trees.json'
    with open(json_path, 'w') as f:
        json.dump(all_trees, f, indent=2, cls=_NumpySafeEncoder)
    print(f"\nSaved unified JSON: {json_path}")
    print(f"  Problems included: {list(all_trees.keys())}")
    for p, d in all_trees.items():
        s = d['summary']
        print(f"    {p}: {s['accepted_nodes']} accepted "
              f"({s['pruned_tree_leaves']} leaves) "
              f"/ {s['total_nodes']} total")

    print(f"\nDone! Images + JSON in {output_dir}")


if __name__ == '__main__':
    main()
