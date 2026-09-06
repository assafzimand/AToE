"""Global top-M tree selection across time-marching windows.

Replaces the fixed per-window budget split (``m_distribution``: equal /
linear / quadratic → m_k per window, independent top-m_k per window) with
ONE budget the windows compete for:

  A. Per window: fit the full decision tree on the window's slice of the fit
     target (full-domain box, exactly like a single-domain fit; boxes are
     clipped to the window's t-range afterwards) and compute every node's
     wavelet norm.
  B. Pool the non-root nodes of ALL windows, rank by the acceptance metric
     (``variable_for_node_accept``) and take the top M globally, with the
     same epsilon tie rule at the cutoff as
     ``RegionDetector.fit_full_tree_and_prune``.
  C. Close each window's selection inside its own tree: all ancestors,
     then (``retain_siblings``) siblings iteratively. A window that received
     no top-M node keeps its ROOT: a single accepted leaf spanning the whole
     slice (the former M=0-slice convention), so every window always has at
     least one expert.

The per-window result dicts have the same shape as
``trainer.orchestrator._fit_tree_and_maps`` so the orchestrator consumes
them unchanged. Reference sweep of this selection on ground truth:
``scripts/sweep_perfect_trees_no_m_distribution.py``.
"""

import copy
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from adaptive.region_detector import RegionDetector, TreeNodeInfo
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Per-window node ids are offset so per-slice trees can be unioned without
# collisions (structure-only windowed tree). Same constant as before.
WINDOW_ID_OFFSET = 1_000_000


# ─────────────────────────── A: per-window fits ───────────────────────────

def fit_window_tree(
    X: np.ndarray, y: np.ndarray, max_depth: int, min_samples_leaf: int,
    domain_bounds: Dict[str, List[float]],
) -> Dict:
    """Fit one full tree on a window slice; return its node table + maps."""
    detector = RegionDetector(
        n_estimators=1, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, domain_bounds=domain_bounds,
    )
    detector.fit(X=X, y=y)
    tree = detector.rf.estimators_[0].tree_
    all_nodes = detector.compute_wavelet_norms(X=X, y=y)
    node_lookup = {n.node_id: n for n in all_nodes}

    children_left = np.asarray(tree.children_left)
    children_right = np.asarray(tree.children_right)
    parent_map, sibling_map, depth = {}, {}, {0: 0}
    q = deque([0])
    while q:
        nid = q.popleft()
        l, r = int(children_left[nid]), int(children_right[nid])
        for c in (l, r):
            if c != -1:
                parent_map[c] = nid
                depth[c] = depth[nid] + 1
                q.append(c)
        if l != -1 and r != -1:
            sibling_map[l] = r
            sibling_map[r] = l

    return {
        'node_lookup': node_lookup,
        'children_left': children_left,
        'children_right': children_right,
        'parent_map': parent_map,
        'sibling_map': sibling_map,
        'depth': depth,
        'node_count': int(tree.node_count),
        'n_samples': int(len(X)),
    }


def metric_value(node: TreeNodeInfo, variable: str) -> Optional[float]:
    """Acceptance metric of a node (same rules as fit_full_tree_and_prune)."""
    if variable == 'norm':
        return node.wavelet_norm_squared
    if variable == 'new_norm':
        return node.new_wavelet_norm_squared
    if variable == 'smoothness':
        if node.smoothness_r2 is None or node.smoothness_r2 < 0.5:
            return None
        return node.smoothness_alpha
    return None


# ───────────────────────── B: global top-M selection ──────────────────────

def select_global_top_m(
    windows: Sequence[Dict], M: int, variable: str, eps: float,
) -> Tuple[Dict[int, set], int, int, Optional[float], int]:
    """Pool every non-root node of every window and take the top M.

    Returns (top ids per window, M_actual, n_ties, cutoff metric, pool size).
    """
    pool = []
    for k, w in enumerate(windows):
        for nid, node in w['node_lookup'].items():
            if nid == 0:
                continue
            v = metric_value(node, variable)
            if v is not None:
                pool.append((k, nid, v))
    reverse = (variable != 'smoothness')
    pool.sort(key=lambda t: t[2], reverse=reverse)

    M_actual = min(M, len(pool))
    chosen = pool[:M_actual]
    cutoff = pool[M_actual - 1][2] if M_actual else None
    n_ties = 0
    if eps > 0 and 0 < M_actual < len(pool):
        tol = eps * abs(cutoff)
        for k, nid, v in pool[M_actual:]:
            gap = (cutoff - v) if reverse else (v - cutoff)
            if gap > tol:
                break
            chosen.append((k, nid, v))
            n_ties += 1

    top = {k: set() for k in range(len(windows))}
    for k, nid, _ in chosen:
        top[k].add(nid)
    return top, M_actual, n_ties, cutoff, len(pool)


# ───────────────────────── C: per-window closure ──────────────────────────

def close_window(w: Dict, top_ids: set, retain_siblings: bool = True) -> set:
    """Ancestors (always) + siblings (if retain_siblings), root excluded."""
    parent_map, sibling_map = w['parent_map'], w['sibling_map']
    accepted = set(top_ids)
    for nid in list(top_ids):
        cur = nid
        while cur in parent_map:
            cur = parent_map[cur]
            if cur == 0:
                break
            accepted.add(cur)
    if not retain_siblings:
        return accepted
    changed = True
    it = 0
    while changed and it < w['node_count']:
        changed = False
        it += 1
        for nid in list(accepted):
            sib = sibling_map.get(nid)
            if sib is not None and sib not in accepted:
                accepted.add(sib)
                changed = True
                cur = sib
                while cur in parent_map:
                    cur = parent_map[cur]
                    if cur == 0 or cur in accepted:
                        break
                    accepted.add(cur)
                    changed = True
    return accepted


def _clip_t(bl, bu, lo, hi, t_dim):
    bl, bu = list(bl), list(bu)
    bl[t_dim] = max(float(bl[t_dim]), lo)
    bu[t_dim] = min(float(bu[t_dim]), hi)
    return bl, bu


def build_window_result(
    w: Dict, accepted: set, top_ids: set, win_idx: int, lo: float, hi: float,
    domain_bounds: Dict[str, List[float]], t_dim: int, id_offset: int,
    variable: str,
) -> Dict:
    """Orchestrator-shaped result for one window (see _fit_tree_and_maps).

    Node boxes are clipped to the window's t-range. When ``accepted`` is
    empty the window root is kept as one whole-slice accepted leaf.
    """
    nl, cl, cr, depth = (w['node_lookup'], w['children_left'],
                         w['children_right'], w['depth'])

    def _off(nid: int) -> int:
        return nid + id_offset if nid >= 0 else -1

    diag_nodes = []
    for nid, nd in nl.items():
        if nid == 0:
            continue
        bl, bu = _clip_t(nd.bounds_lower, nd.bounds_upper, lo, hi, t_dim)
        diag_nodes.append({
            'node_id': _off(nid),
            'parent_node_id': _off(w['parent_map'].get(nid, -1)),
            'wavelet_norm_squared': nd.wavelet_norm_squared,
            'new_wavelet_norm_squared': nd.new_wavelet_norm_squared,
            'smoothness_alpha': nd.smoothness_alpha,
            'n_samples': nd.n_samples,
            'is_leaf': bool(nd.is_leaf),
            'bounds_lower': bl,
            'bounds_upper': bu,
            'tree_depth': depth.get(nid, -1),
            'from_top_M': bool(nid in top_ids),
            'window_idx': win_idx,
        })

    if not accepted:
        # Window root kept: one whole-slice accepted leaf. Its id sits past
        # every real node id of this window's tree so it can never collide
        # with a fitted node (the diag rows keep the real ids).
        root = nl[0]
        bl = list(domain_bounds['lower'])
        bu = list(domain_bounds['upper'])
        bl[t_dim], bu[t_dim] = lo, hi
        nid = id_offset + w['node_count']
        node = TreeNodeInfo(
            node_id=nid, tree_idx=0, is_leaf=True, n_samples=root.n_samples,
            bounds_lower=bl, bounds_upper=bu,
            prediction=np.asarray(root.prediction),
            parent_prediction=None,
            wavelet_norm_squared=root.wavelet_norm_squared,
            new_wavelet_norm_squared=root.new_wavelet_norm_squared,
        )
        return {
            'accepted_nodes': [(node, -1)],
            'children_left': {nid: -1},
            'parent_map': {nid: -1},
            'node_tree_depth': {nid: 1},
            'prune_depth_stats': {},
            'diag_nodes': diag_nodes,
            'node_count': w['node_count'],
            'top_M_count': 0,
            'root_kept': True,
        }

    accepted_bfs: List[Tuple[TreeNodeInfo, int]] = []
    q = deque([(0, -1)])
    while q:
        nid, anc = q.popleft()
        if nid != 0 and nid in accepted:
            # Copy: the fitted window (and its TreeNodeInfo objects) may be
            # reused for several budgets (reference sweep), so never mutate.
            nd = copy.copy(nl[nid])
            bl, bu = _clip_t(nd.bounds_lower, nd.bounds_upper, lo, hi, t_dim)
            nd.bounds_lower, nd.bounds_upper = bl, bu
            nd.node_id = _off(nid)
            accepted_bfs.append((nd, _off(anc)))
            next_anc = nid
        else:
            next_anc = anc
        for c in (int(cl[nid]), int(cr[nid])):
            if c != -1:
                q.append((c, next_anc))

    # Per-depth stats in the same shape as fit_full_tree_and_prune's.
    prune_depth_stats = {}
    for d in sorted({depth[n] for n in accepted}):
        at_d = [n for n in accepted if depth[n] == d]
        vals = [metric_value(nl[n], variable) for n in at_d]
        vals = [v for v in vals if v is not None]
        n_top = sum(1 for n in at_d if n in top_ids)
        if vals:
            prune_depth_stats[d] = {
                'n_nodes': len(at_d),
                'n_from_top_M': n_top,
                'n_added_for_closure': len(at_d) - n_top,
                'metric_min': float(min(vals)),
                'metric_max': float(max(vals)),
                'metric_median': float(np.median(vals)),
            }

    children_map = {_off(n): _off(int(cl[n])) for n in range(w['node_count'])}
    return {
        'accepted_nodes': accepted_bfs,
        'children_left': children_map,
        'parent_map': {_off(c): _off(p) for c, p in w['parent_map'].items()},
        'node_tree_depth': {_off(n): d for n, d in depth.items()},
        'prune_depth_stats': prune_depth_stats,
        'diag_nodes': diag_nodes,
        'node_count': w['node_count'],
        'top_M_count': len(top_ids),
        'root_kept': False,
    }


# ─────────────────────────────── driver ───────────────────────────────────

def build_global_windowed_trees(
    slices: Sequence[Tuple[float, float, np.ndarray, np.ndarray]],
    M: int,
    max_depth: int,
    min_samples_leaf: int,
    domain_bounds: Dict[str, List[float]],
    t_dim: int,
    variable: str = 'norm',
    eps: float = 0.0,
    retain_siblings: bool = True,
    id_offset_step: int = WINDOW_ID_OFFSET,
    verbose: bool = True,
) -> Tuple[List[Dict], Dict]:
    """Run A/B/C over ``slices`` = [(t_lo, t_hi, X, y), ...].

    Returns (per-window result dicts, summary). Window k's node ids are
    offset by ``(k + 1) * id_offset_step`` (0 disables offsetting).
    """
    windows = []
    for k, (lo, hi, X, y) in enumerate(slices):
        if verbose:
            logger.info(f"  [GlobalTree] window {k}: t in [{lo:.4f}, {hi:.4f}], "
                        f"{len(X)} points -> fitting full tree "
                        f"(max_depth={max_depth}, "
                        f"min_samples_leaf={min_samples_leaf})")
        w = fit_window_tree(X, y, max_depth, min_samples_leaf, domain_bounds)
        w['lo'], w['hi'] = float(lo), float(hi)
        if verbose:
            logger.info(f"  [GlobalTree] window {k}: {w['node_count']} nodes")
        windows.append(w)

    top, M_actual, n_ties, cutoff, pool_size = select_global_top_m(
        windows, M, variable, eps)
    per_window_top = [len(top[k]) for k in range(len(windows))]
    if verbose:
        logger.info(f"  [GlobalTree] pooled {pool_size} candidate nodes across "
                    f"{len(windows)} windows; top M={M_actual}"
                    f"{f' (+{n_ties} eps ties)' if n_ties else ''} by "
                    f"{variable} -> per window {per_window_top}; cutoff "
                    f"{variable}={cutoff if cutoff is None else f'{cutoff:.4g}'}")

    results = []
    for k, w in enumerate(windows):
        accepted = close_window(w, top[k], retain_siblings)
        res = build_window_result(
            w, accepted, top[k], k, w['lo'], w['hi'], domain_bounds, t_dim,
            id_offset=(k + 1) * id_offset_step if id_offset_step else 0,
            variable=variable)
        if verbose:
            logger.info(f"  [GlobalTree] window {k}: {len(top[k])} top-M -> "
                        f"{len(res['accepted_nodes'])} accepted after closure"
                        f"{' (ROOT KEPT: one whole-slice expert)' if res['root_kept'] else ''}")
        results.append(res)

    summary = {
        'selection': 'global_top_M',
        'global_M': int(M),
        'M_actual': int(M_actual),
        'n_ties': int(n_ties),
        'pool_size': int(pool_size),
        'cutoff_metric': cutoff,
        'variable_for_node_accept': variable,
        'm_per_window': per_window_top,
        'accepted_per_window': [len(r['accepted_nodes']) for r in results],
        'root_kept': [bool(r['root_kept']) for r in results],
        'node_count_per_window': [w['node_count'] for w in windows],
    }
    return results, summary
