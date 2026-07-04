"""Region detection using Random Forest geometric wavelets.

Implements the algorithm to detect high-variation regions for spawning expert PINNs:
1. Fit a Random Forest regressor to the current solution
2. Compute geometric wavelets at each tree node (Q_child - Q_parent for d-dim output)
3. Compute wavelet norm: ||Q_child - Q_parent||^2 * volume
4. Select / prune regions based on norm threshold
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from adaptive.indicators import RegionDescriptor
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TreeNodeInfo:
    """Information about a single tree node."""
    node_id: int
    tree_idx: int
    is_leaf: bool
    n_samples: int
    bounds_lower: List[float]
    bounds_upper: List[float]
    prediction: np.ndarray  # Q_Ω(x) - local mean value, shape (d,) for d-dim output
    parent_prediction: Optional[np.ndarray]  # Q_Ω_parent(x), shape (d,) or None
    wavelet_norm_squared: float = 0.0  # ||Q_child - Q_parent||^2 * volume
    new_wavelet_norm_squared: float = 0.0  # Sum of children's classic norms (internal) or temp tree result (leaf)
    # Local tree-Besov smoothness: slope of log(||ψ_ν||₂/|ν|^½) vs log(|ν|) over descendants
    # Larger α = smoother region; None = not enough descendants for reliable estimate
    smoothness_alpha: Optional[float] = None
    smoothness_r2: Optional[float] = None   # R² of the log-log regression
    smoothness_n_levels: int = 0              # number of descendants used in fit


class RegionDetector:
    """Detects refinement regions using Random Forest geometric wavelets.
    
    Algorithm:
    1. Fit RF to current PINN solution: f_RF(x,t) ≈ u(x,t)
    2. For each tree node, compute geometric wavelet: ψ = Q_child - Q_parent
    3. Compute wavelet norm: ||ψ||^2 * volume
    4. Return regions that pass the threshold
    
    Supports multi-dimensional output (e.g., Schrödinger's [u, v]).
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_leaf: int = 50,
        domain_bounds: Optional[Dict[str, List[float]]] = None
    ):
        """
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of each tree
            min_samples_leaf: Minimum samples required in a leaf node
            domain_bounds: {'lower': [x_min, t_min], 'upper': [x_max, t_max]}
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.domain_bounds = domain_bounds
        
        self.rf: Optional[RandomForestRegressor] = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> 'RegionDetector':
        """
        Fit Random Forest to the data.
        
        Args:
            X: (N, n_dims) array of coordinates [x, t] or [x, y, t]
            y: (N,) or (N, output_dim) array of solution values.
               Multi-output is supported (RF multi-output regression).
            **kwargs: Accepted for backward compatibility (loss_components,
                      residuals) but no longer used.
            
        Returns:
            self for chaining
        """
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.ravel()

        self.domain_bounds = {
            'lower': X.min(axis=0).tolist(),
            'upper': X.max(axis=0).tolist()
        }
        
        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=False,
            random_state=42,
            n_jobs=-1
        )
        self.rf.fit(X, y)
        
        return self
    
    def _get_parent_id(self, tree, node_id: int) -> Optional[int]:
        """Find the parent of a node by searching the tree."""
        if node_id == 0:
            return None  # Root has no parent
        
        children_left = tree.children_left
        children_right = tree.children_right
        
        for parent in range(tree.node_count):
            if children_left[parent] == node_id or children_right[parent] == node_id:
                return parent
        return None
    
    def _get_node_bounds(self, tree, node_id: int) -> Tuple[List[float], List[float]]:
        """
        Reconstruct the axis-aligned bounding box for a node.
        Trace from root to node, collecting split conditions.
        """
        n_features = tree.n_features
        bounds_lower = list(self.domain_bounds['lower'])
        bounds_upper = list(self.domain_bounds['upper'])
        
        if node_id == 0:
            return bounds_lower, bounds_upper
        
        # Build path from root to node
        path = []
        current = node_id
        while current != 0:
            parent = self._get_parent_id(tree, current)
            if parent is None:
                break
            went_left = (tree.children_left[parent] == current)
            path.append((parent, went_left))
            current = parent
        
        # Traverse path from root to node
        for parent, went_left in reversed(path):
            feature = tree.feature[parent]
            threshold = tree.threshold[parent]
            
            if went_left:
                # Left child: feature <= threshold
                bounds_upper[feature] = min(bounds_upper[feature], threshold)
            else:
                # Right child: feature > threshold
                bounds_lower[feature] = max(bounds_lower[feature], threshold)
        
        return bounds_lower, bounds_upper

    def _compute_smoothness_indices(
        self,
        nodes_by_id: Dict[int, TreeNodeInfo],
        children_left,
        children_right,
        min_levels: int = 4,
    ) -> None:
        """
        Populate smoothness_alpha/r2/n_levels in-place for each node.

        Local smoothness for node Omega is estimated from decay of subtree wavelet
        energy across relative tree levels.

        For each descendant nu of Omega:
            ||psi_nu||_2^2 = wavelet_norm_squared_nu

        For each relative level ell >= 1 below Omega:
            E_ell(Omega) = sum_{nu at relative level ell} ||psi_nu||_2^2

        We fit:
            log(E_ell) ~ a + b * ell

        and define:
            smoothness_alpha = -b / 2

        Interpretation:
            larger alpha   -> smoother region
            alpha near 0   -> weak decay / rough region
            alpha negative -> energy grows with scale refinement / very rough region

        If fewer than min_levels valid relative levels are available, alpha is None.
        """

        def _compute_depths() -> Dict[int, int]:
            depths = {}
            stack = [(0, 0)]
            while stack:
                nid, depth = stack.pop()
                if nid in depths:
                    continue
                depths[nid] = depth
                left, right = children_left[nid], children_right[nid]
                if left != -1:
                    stack.append((left, depth + 1))
                if right != -1:
                    stack.append((right, depth + 1))
            return depths

        def _collect_descendants(root_id: int) -> List[int]:
            result = []
            stack = [root_id]
            while stack:
                nid = stack.pop()
                for child in (children_left[nid], children_right[nid]):
                    if child != -1 and child in nodes_by_id:
                        result.append(child)
                        stack.append(child)
            return result

        node_depth = _compute_depths()

        for node_id, node_info in nodes_by_id.items():
            root_depth = node_depth.get(node_id, 0)
            desc_ids = _collect_descendants(node_id)

            # relative level -> total wavelet energy on that level
            level_to_energy: Dict[int, float] = {}

            for desc_id in desc_ids:
                desc = nodes_by_id[desc_id]

                if desc.n_samples < self.min_samples_leaf:
                    continue
                if desc.wavelet_norm_squared <= 0.0:
                    continue

                rel_level = node_depth.get(desc_id, root_depth) - root_depth
                if rel_level <= 0:
                    continue

                level_to_energy[rel_level] = (
                    level_to_energy.get(rel_level, 0.0) + float(desc.wavelet_norm_squared)
                )

            # Build regression points: x = relative level, y = log(total energy)
            xs = []
            ys = []

            for level in sorted(level_to_energy.keys()):
                energy = level_to_energy[level]
                if energy <= 0.0:
                    continue
                xs.append(float(level))
                ys.append(float(np.log(energy)))

            n_levels = len(xs)
            node_info.smoothness_n_levels = n_levels

            if n_levels < min_levels:
                node_info.smoothness_alpha = None
                node_info.smoothness_r2 = None
                continue

            x = np.asarray(xs, dtype=float)
            y = np.asarray(ys, dtype=float)

            # Need at least some level spread
            if float(np.max(x) - np.min(x)) < 1e-12:
                node_info.smoothness_alpha = None
                node_info.smoothness_r2 = None
                continue

            # Fit y = a + b*x
            A = np.vstack([x, np.ones_like(x)]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

            y_pred = slope * x + intercept
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else None

            # If log(E_l) ~ a - 2*alpha*l, then alpha = -slope / 2
            alpha = -0.5 * float(slope)

            node_info.smoothness_alpha = alpha
            node_info.smoothness_r2 = r2

            logger.info(
                f"    [Smoothness] Node {node_id}: "
                f"alpha={node_info.smoothness_alpha:.4f}, "
                f"r2={node_info.smoothness_r2:.4f}, "
                f"levels={n_levels}"
            )

    def compute_wavelet_norms(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> List[TreeNodeInfo]:
        """
        Compute geometric wavelet norms for all tree nodes (two-pass algorithm).

        Pass 1: Compute classic wavelet_norm_squared and smoothness_alpha for all nodes.
        Pass 2: Compute new_wavelet_norm_squared:
            - Internal nodes: sum of children's classic norms
            - Leaf nodes: if X, y provided, fit depth-1 tree, compute sum of hypothetical children's classic norms
        
        Uses RF's internal node values (tree.value) as Q_Ω for each node.
        Supports multi-dimensional output (d > 1): Q is d-dimensional vector.
        Classic wavelet norm = ||Q_child - Q_parent||^2 * Ω_volume
        
        Args:
            X: Optional (N, n_dims) sample coordinates for leaf new_norm computation
            y: Optional (N,) or (N, output_dim) target values for leaf new_norm computation
        
        Returns:
            List of TreeNodeInfo with both wavelet_norm_squared (classic) and new_wavelet_norm_squared
        """
        if self.rf is None:
            raise RuntimeError("Must call fit() before compute_wavelet_norms()")

        all_nodes = []

        for tree_idx, estimator in enumerate(self.rf.estimators_):
            tree = estimator.tree_
            tree_nodes: Dict[int, TreeNodeInfo] = {}

            # ══════════════════════════════════════════════════════════════
            # PASS 1: Compute classic wavelet_norm_squared and smoothness
            # ══════════════════════════════════════════════════════════════
            for node_id in range(tree.node_count):
                is_leaf = tree.children_left[node_id] == -1

                n_samples = int(tree.n_node_samples[node_id])

                if n_samples < self.min_samples_leaf:
                    continue

                # Get bounds
                bounds_lower, bounds_upper = self._get_node_bounds(tree, node_id)

                # Get Q_Ω from RF's internal node value
                # tree.value has shape (n_nodes, n_outputs, 1) for regressors
                # tree.value[node_id, :, 0] gives shape (n_outputs,) = (d,)
                prediction = tree.value[node_id, :, 0].copy()  # Shape (d,)

                # Get parent prediction Q_Ω_parent
                parent_id = self._get_parent_id(tree, node_id)
                parent_prediction = None
                if parent_id is not None:
                    parent_prediction = tree.value[parent_id, :, 0].copy()  # Shape (d,)

                wavelet_norm_squared = 0.0
                if parent_prediction is not None and n_samples > 0:
                    diff = prediction - parent_prediction
                    l2_norm_squared = float(np.sum(diff ** 2))
                    volume = float(np.prod(np.maximum(
                        np.asarray(bounds_upper, dtype=float) - np.asarray(bounds_lower, dtype=float),1e-12)))
                    wavelet_norm_squared = l2_norm_squared * volume

                node = TreeNodeInfo(
                    node_id=node_id,
                    tree_idx=tree_idx,
                    is_leaf=is_leaf,
                    n_samples=n_samples,
                    bounds_lower=bounds_lower,
                    bounds_upper=bounds_upper,
                    prediction=prediction,
                    parent_prediction=parent_prediction,
                    wavelet_norm_squared=wavelet_norm_squared,
                    new_wavelet_norm_squared=0.0,  # Will be computed in Pass 2
                )
                tree_nodes[node_id] = node

            self._compute_smoothness_indices(
                tree_nodes, tree.children_left, tree.children_right
            )

            # ══════════════════════════════════════════════════════════════
            # PASS 2: Compute new_wavelet_norm_squared
            # ══════════════════════════════════════════════════════════════
            # Get leaf assignments if we need them for leaf new_norm computation
            leaf_ids = None
            if X is not None and y is not None:
                leaf_ids = estimator.apply(X)

            for node_id, node in tree_nodes.items():
                is_leaf = node.is_leaf
                new_norm = 0.0

                if not is_leaf:
                    # ── Internal node: sum of children's classic norms ──
                    left_id = tree.children_left[node_id]
                    right_id = tree.children_right[node_id]
                    
                    if left_id in tree_nodes:
                        new_norm += tree_nodes[left_id].wavelet_norm_squared
                    if right_id in tree_nodes:
                        new_norm += tree_nodes[right_id].wavelet_norm_squared
                else:
                    # ── Leaf node: fit depth-1 temp tree if X, y available ──
                    if X is not None and y is not None and leaf_ids is not None:
                        mask = (leaf_ids == node_id)
                        X_leaf = X[mask]
                        y_leaf = y[mask]

                        if len(X_leaf) >= 2:
                            dt = DecisionTreeRegressor(
                                max_depth=1, min_samples_leaf=1, random_state=42
                            )
                            try:
                                dt.fit(X_leaf, y_leaf)
                                sub = dt.tree_
                                sub_l = sub.children_left[0]
                                sub_r = sub.children_right[0]
                                
                                if sub_l != -1:  # Successfully split
                                    # Compute classic norms for the two hypothetical children
                                    Q_parent = sub.value[0, :, 0]
                                    Q_left = sub.value[sub_l, :, 0]
                                    Q_right = sub.value[sub_r, :, 0]
                                    
                                    bounds_lower_l, bounds_upper_l = self._get_node_bounds(sub, sub_l)
                                    bounds_lower_r, bounds_upper_r = self._get_node_bounds(sub, sub_r)
                                    
                                    volume_l = float(np.prod(np.maximum(
                                        np.asarray(bounds_upper_l, dtype=float) - np.asarray(bounds_lower_l, dtype=float), 1e-12)))
                                    volume_r = float(np.prod(np.maximum(
                                        np.asarray(bounds_upper_r, dtype=float) - np.asarray(bounds_lower_r, dtype=float), 1e-12)))
                                    
                                    # Classic norm for left child
                                    classic_norm_left = float(np.sum((Q_left - Q_parent) ** 2)) * volume_l
                                    # Classic norm for right child
                                    classic_norm_right = float(np.sum((Q_right - Q_parent) ** 2)) * volume_r
                                    
                                    # New norm = sum of children's classic norms
                                    new_norm = classic_norm_left + classic_norm_right
                            except Exception:
                                pass  # Leave new_norm = 0.0

                # Update the node with new_norm
                node.new_wavelet_norm_squared = new_norm

            all_nodes.extend(tree_nodes.values())

        return all_nodes

    def fit_full_tree_and_prune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        M: int,
        variable_for_node_accept: str = 'norm',
        verbose: bool = True,
        retain_siblings: bool = True,
        **kwargs,
    ) -> Tuple[List[Tuple[TreeNodeInfo, int]], Dict]:
        """
        Fit a full decision tree on the entire domain and select top M nodes
        by the configured metric, ensuring valid binary tree structure.

        Selection criterion:
            Select top M nodes based on the configured metric:
            - 'norm': highest wavelet_norm_squared
            - 'new_norm': highest new_wavelet_norm_squared
            - 'smoothness': lowest smoothness_alpha (roughest regions)
            
        Then ensure valid structure by adding:
            - All ancestors of selected nodes (paths to root) — always
            - Siblings of any node in closure — only if retain_siblings=True

        Args:
            X: (N, n_dims) coordinates
            y: (N,) or (N, output_dim) predictions
            M: Number of top nodes to select
            variable_for_node_accept: 'norm' | 'new_norm' | 'smoothness'
            verbose: print diagnostics
            retain_siblings: If True (default), add siblings to ensure complete
                           binary tree (needed for ANT routing, AToE-Leaves tiling).
                           If False, keep only ancestors (AToE additive composition).
            **kwargs: ignored (backward compat)

        Returns:
            Tuple of:
            - List of (TreeNodeInfo, parent_tree_node_id) in BFS order.
              parent_tree_node_id is the tree node id of the nearest
              accepted ancestor, or -1 for children of root.
            - Dict of per-depth selection statistics.
        """
        from collections import deque

        old_n_estimators = self.n_estimators
        self.n_estimators = 1
        try:
            self.fit(X=X, y=y)
        finally:
            self.n_estimators = old_n_estimators

        tree = self.rf.estimators_[0].tree_
        all_nodes = self.compute_wavelet_norms(X=X, y=y)

        if not all_nodes:
            if verbose:
                logger.info("  [M-term Tree] No nodes in tree")
            return [], {}

        node_lookup = {node.node_id: node for node in all_nodes}

        # Build parent->children mapping and compute depth of each node
        children_left = tree.children_left
        children_right = tree.children_right
        node_depth = {}
        parent_map = {}
        sibling_map = {}  # node_id -> sibling_id
        queue = deque([(0, 0)])
        max_depth_seen = 0
        while queue:
            nid, depth = queue.popleft()
            node_depth[nid] = depth
            max_depth_seen = max(max_depth_seen, depth)
            l, r = children_left[nid], children_right[nid]
            if l != -1:
                parent_map[l] = nid
                queue.append((l, depth + 1))
            if r != -1:
                parent_map[r] = nid
                queue.append((r, depth + 1))
            # Map siblings to each other
            if l != -1 and r != -1:
                sibling_map[l] = r
                sibling_map[r] = l

        # Helper: extract the configured metric value from a node
        def _get_metric_value(nid: int) -> Optional[float]:
            if nid not in node_lookup:
                return None
            node = node_lookup[nid]
            if variable_for_node_accept == 'norm':
                return node.wavelet_norm_squared
            elif variable_for_node_accept == 'new_norm':
                return node.new_wavelet_norm_squared
            elif variable_for_node_accept == 'smoothness':
                # For smoothness, check reliability
                if node.smoothness_r2 is None or node.smoothness_r2 < 0.5:
                    return None  # unreliable smoothness estimate
                return node.smoothness_alpha
            return None

        # Get all nodes (excluding root) with valid metric values
        nodes_with_metrics = []
        for nid in range(1, tree.node_count):  # Skip root (0)
            metric_val = _get_metric_value(nid)
            if metric_val is not None and nid in node_lookup:
                nodes_with_metrics.append((nid, metric_val))

        # Sort nodes by metric (descending for norm/new_norm, ascending for smoothness)
        reverse_sort = (variable_for_node_accept != 'smoothness')
        nodes_with_metrics.sort(key=lambda x: x[1], reverse=reverse_sort)

        # Select top M nodes
        M_actual = min(M, len(nodes_with_metrics))
        top_M_nodes = {nid for nid, _ in nodes_with_metrics[:M_actual]}

        if verbose:
            if M_actual < M:
                logger.info(f"  [M-term Tree] Requested M={M}, but only {M_actual} nodes with valid metrics")
            else:
                logger.info(f"  [M-term Tree] Selected top M={M_actual} nodes by {variable_for_node_accept}")

        # Build closure: add ancestors (always) and siblings (conditional)
        accepted = set(top_M_nodes)
        
        # Add all ancestors of selected nodes (always needed for valid parent_idx linkage)
        for nid in list(top_M_nodes):
            cur = nid
            while cur in parent_map:
                cur = parent_map[cur]
                if cur == 0:  # Don't add root
                    break
                accepted.add(cur)

        # Ensure binary tree structure: iteratively add siblings
        # - retain_siblings=True: ANT (routing), AToE-Leaves (tiling) need complete binary tree
        # - retain_siblings=False: AToE additive composition only needs ancestors
        if retain_siblings:
            changed = True
            iterations = 0
            max_iterations = tree.node_count  # Safety limit
            while changed and iterations < max_iterations:
                changed = False
                iterations += 1
                for nid in list(accepted):
                    if nid in sibling_map:
                        sibling = sibling_map[nid]
                        if sibling not in accepted:
                            accepted.add(sibling)
                            changed = True
                            # Also add ancestors of newly added sibling
                            cur = sibling
                            while cur in parent_map:
                                cur = parent_map[cur]
                                if cur == 0 or cur in accepted:
                                    break
                                accepted.add(cur)
                                changed = True
        
        closure_type = "ancestors+siblings" if retain_siblings else "ancestors-only"
        if verbose:
            logger.info(f"  [M-term Tree] Closure: {closure_type} -> {len(accepted)} nodes")

        # Compute statistics by depth
        depth_stats = {}
        for depth in range(1, max_depth_seen + 1):
            nodes_at_depth = [nid for nid in accepted if node_depth.get(nid) == depth]
            if nodes_at_depth:
                metrics_at_depth = [_get_metric_value(nid) for nid in nodes_at_depth]
                metrics_at_depth = [m for m in metrics_at_depth if m is not None]
                top_m_at_depth = sum(1 for nid in nodes_at_depth if nid in top_M_nodes)
                if metrics_at_depth:
                    depth_stats[depth] = {
                        'n_nodes': len(nodes_at_depth),
                        'n_from_top_M': top_m_at_depth,
                        'n_added_for_closure': len(nodes_at_depth) - top_m_at_depth,
                        'metric_min': float(min(metrics_at_depth)),
                        'metric_max': float(max(metrics_at_depth)),
                        'metric_median': float(np.median(metrics_at_depth)),
                    }
        
        if verbose:
            logger.info(f"\n  [M-term Tree] Tree has {tree.node_count} nodes, "
                  f"max depth {max_depth_seen}")
            logger.info(f"  [M-term Tree] Top M={M_actual} selected, "
                  f"final accepted (with closure): {len(accepted)} nodes")
            logger.info(f"  [M-term Tree] Added {len(accepted) - M_actual} nodes for valid binary tree structure")
            logger.info(f"  [M-term Tree] Per-depth selection stats:")
            for d in sorted(depth_stats.keys()):
                s = depth_stats[d]
                logger.info(
                    f"    Depth {d:2d}: "
                    f"{s['n_nodes']:3d} nodes | "
                    f"{s['n_from_top_M']:2d} from top-M | "
                    f"{s['n_added_for_closure']:2d} added for closure | "
                    f"{variable_for_node_accept} [{s['metric_min']:.4f}, "
                    f"{s['metric_median']:.4f}, "
                    f"{s['metric_max']:.4f}]"
                )

        # Build result in BFS order with parent relationships
        result = []
        bfs = deque([(0, -1)])  # (node_id, nearest_accepted_ancestor)
        while bfs:
            nid, nearest_ancestor = bfs.popleft()
            if nid != 0 and nid in accepted and nid in node_lookup:
                result.append((node_lookup[nid], nearest_ancestor))
                if verbose:
                    node = node_lookup[nid]
                    anc_str = "Base" if nearest_ancestor == -1 else f"Node{nearest_ancestor}"
                    is_leaf_str = "leaf" if node.is_leaf else "internal"
                    metric_val = _get_metric_value(nid)
                    metric_str = (f"{metric_val:.4f}"
                                 if metric_val is not None else "None")
                    from_top_m = " [TOP-M]" if nid in top_M_nodes else ""
                    logger.info(f"    [M-term Tree] Node {nid} ({is_leaf_str}): "
                          f"ACCEPT (parent={anc_str}, "
                          f"{variable_for_node_accept}={metric_str}, "
                          f"samples={node.n_samples}){from_top_m}")
                next_ancestor = nid
            else:
                next_ancestor = nearest_ancestor

            l, r = children_left[nid], children_right[nid]
            if l != -1:
                bfs.append((l, next_ancestor))
            if r != -1:
                bfs.append((r, next_ancestor))

        if verbose:
            logger.info(f"  [M-term Tree] Result: {len(result)} accepted nodes")

        return result, depth_stats

    def _compute_outside_fraction(
        self, 
        node: TreeNodeInfo, 
        sibling_regions: List[RegionDescriptor]
    ) -> float:
        """
        Compute what fraction of a node's volume is OUTSIDE all sibling regions.
        
        Uses a simple approximation: compute total overlap with union of siblings.
        For non-overlapping siblings, this is exact. For overlapping siblings,
        this is a lower bound on the actual outside fraction.
        
        Args:
            node: The candidate node
            sibling_regions: List of same-depth regions (siblings)
            
        Returns:
            Fraction of node volume outside all siblings (0.0 to 1.0)
        """
        if not sibling_regions:
            return 1.0  # No siblings = 100% outside
        
        node_vol = 1.0
        for lo, hi in zip(node.bounds_lower, node.bounds_upper):
            node_vol *= (hi - lo)
        
        if node_vol <= 0:
            return 0.0
        
        # Compute total overlap volume with all siblings
        # Note: This may double-count if siblings overlap each other,
        # giving a conservative (lower) estimate of outside_fraction
        total_overlap = 0.0
        
        for region in sibling_regions:
            # Check if boxes overlap (any dimension must be disjoint for no overlap)
            overlaps = True
            for i in range(len(node.bounds_lower)):
                if (node.bounds_upper[i] <= region.bounds_lower[i] or
                    node.bounds_lower[i] >= region.bounds_upper[i]):
                    overlaps = False
                    break
            
            if overlaps:
                # Compute overlap volume
                overlap_lower = [max(node.bounds_lower[i], region.bounds_lower[i]) 
                                for i in range(len(node.bounds_lower))]
                overlap_upper = [min(node.bounds_upper[i], region.bounds_upper[i])
                                for i in range(len(node.bounds_upper))]
                
                overlap_vol = 1.0
                for lo, hi in zip(overlap_lower, overlap_upper):
                    overlap_vol *= max(0, hi - lo)
                
                total_overlap += overlap_vol
        
        # Cap overlap at node volume (in case of double-counting)
        total_overlap = min(total_overlap, node_vol)
        
        # Return fraction outside
        outside_fraction = (node_vol - total_overlap) / node_vol
        return outside_fraction
    
    def _check_overlap(self, node: TreeNodeInfo, existing_regions: List[RegionDescriptor]) -> bool:
        """Check if a node significantly overlaps with existing regions.
        
        DEPRECATED: Use _compute_outside_fraction for more precise control.
        """
        for region in existing_regions:
            # Check if boxes overlap (any dimension must be disjoint for no overlap)
            overlaps = True
            for i in range(len(node.bounds_lower)):
                if (node.bounds_upper[i] <= region.bounds_lower[i] or
                    node.bounds_lower[i] >= region.bounds_upper[i]):
                    overlaps = False
                    break
            
            if overlaps:
                # Compute overlap volume
                overlap_lower = [max(node.bounds_lower[i], region.bounds_lower[i]) 
                                for i in range(len(node.bounds_lower))]
                overlap_upper = [min(node.bounds_upper[i], region.bounds_upper[i])
                                for i in range(len(node.bounds_upper))]
                
                overlap_vol = 1.0
                for lo, hi in zip(overlap_lower, overlap_upper):
                    overlap_vol *= max(0, hi - lo)
                
                node_vol = 1.0
                for lo, hi in zip(node.bounds_lower, node.bounds_upper):
                    node_vol *= (hi - lo)
                
                # If overlap is more than 50% of node volume, consider it overlapping
                if node_vol > 0 and overlap_vol / node_vol > 0.5:
                    return True
        
        return False
    
    def detect(
        self,
        X: np.ndarray,
        y: np.ndarray,
        loss_components: Optional[Dict[str, np.ndarray]] = None,
        residuals: Optional[np.ndarray] = None,
        sibling_regions: Optional[List[RegionDescriptor]] = None,
        overlap_threshold: float = 0.5,
        wavelet_threshold: Optional[float] = None,
        spawn_epoch: int = 0,
        depth: int = 1,
        parent_idx: int = -1,
        verbose: bool = True
    ) -> Optional[RegionDescriptor]:
        """
        Convenience method to fit RF and detect refinement region in one call.
        
        Args:
            X: (N, n_dims) array of coordinates
            y: (N,) or (N, output_dim) array of solution values
            loss_components: Dict with 'residual', 'ic', 'bc' arrays and 'weights'
                (new approach using total loss)
            residuals: DEPRECATED - (N,) array of PDE residuals (for backward compatibility)
            sibling_regions: List of same-parent regions to check overlap against
            overlap_threshold: Accept region if more than this fraction is outside siblings
            wavelet_threshold: Minimum wavelet norm to spawn
            spawn_epoch: Current epoch for tracking
            depth: Depth level for the new expert (1 = child of base)
            parent_idx: Index of the parent expert (-1 for depth-1 experts)
            verbose: Print diagnostic information about why regions were rejected
            
        Returns:
            RegionDescriptor for the selected region, or None
        """
        self.fit(X, y, loss_components=loss_components, residuals=residuals)
        return self.select_refinement_region(
            sibling_regions=sibling_regions,
            overlap_threshold=overlap_threshold,
            wavelet_threshold=wavelet_threshold,
            spawn_epoch=spawn_epoch,
            depth=depth,
            parent_idx=parent_idx,
            verbose=verbose
        )
