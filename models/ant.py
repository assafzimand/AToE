"""Adaptive Neural Tree (ANT) model.

Implements a tree-structured PINN where each expert takes its
parent's last hidden layer activation as input rather than the
raw (x,t) coordinates. Only leaf nodes contribute to the final
solution via a partition of unity.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from collections import defaultdict
from torch.utils.hooks import RemovableHandle

from models.fc_model import FCNet
from models.network_factory import create_network
from adaptive.indicators import (
    RegionDescriptor,
    BatchedIndicators,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ANT(nn.Module):

    supports_decomposed = False

    def __init__(
        self,
        base_architecture: List[int],
        activation: str,
        config: Dict,
        adaptive_config: Dict,
    ):
        super().__init__()

        self.base_architecture = base_architecture
        self.activation = activation
        self.config = config
        self.adaptive_config = adaptive_config

        self.max_experts = adaptive_config['max_experts']
        self.blending_mode = adaptive_config['blending_mode']
        self.sigma_fraction = adaptive_config['sigma_fraction']
        self.base_weight = adaptive_config['base_weight']
        self.expert_type = adaptive_config['expert_type']
        if self.expert_type == 'piratenet':
            raise ValueError(
                "PirateNet is incompatible with ANT: ANT requires "
                "return_activation=True and a standard hidden-state interface "
                "that PirateNet does not provide. "
                "Use AToE or AToELeaves with expert_type='piratenet'."
            )

        raw_hidden = adaptive_config['ANT_default_hidden_layers']
        if isinstance(raw_hidden, list):
            if len(set(raw_hidden)) != 1:
                raise ValueError(
                    f"ANT_default_hidden_layers list must have "
                    f"uniform widths, got {raw_hidden}"
                )
            raw_hidden = raw_hidden[0]
        self.default_hidden_width = int(raw_hidden)

        raw_thresh = adaptive_config.get('ANT_threshold_architecture', None)  # Can be None
        if raw_thresh is not None:
            if isinstance(raw_thresh, list):
                if len(set(raw_thresh)) != 1:
                    raise ValueError(
                        f"ANT_threshold_architecture list must "
                        f"have uniform widths, got {raw_thresh}"
                    )
                raw_thresh = raw_thresh[0]
            self.ant_threshold_width = int(raw_thresh)
        else:
            self.ant_threshold_width = None

        problem = config['problem']
        problem_config = config[problem]
        self.output_dim = problem_config['output_dim']
        
        # Window configuration: smoothstep (compact) or sigmoid (legacy)
        self.window_type = problem_config.get('window_type', 'smoothstep')
        self.window_smoothness_order = problem_config.get('window_smoothness_order', 2)
        
        # Read variable_for_expert_size and corresponding threshold
        self.variable_for_expert_size = adaptive_config['variable_for_expert_size']
        if self.variable_for_expert_size == 'norm':
            self.expert_size_threshold = problem_config['wavelet_threshold']
        elif self.variable_for_expert_size == 'new_norm':
            self.expert_size_threshold = problem_config['new_norm_threshold']
        elif self.variable_for_expert_size == 'smoothness':
            self.expert_size_threshold = problem_config['tree_smoothness_threshold']
        else:
            self.expert_size_threshold = 1.0

        self.base_model = create_network(
            base_architecture, activation, config,
            is_base=True, expert_type=self.expert_type,
        )

        self.experts = nn.ModuleList()
        self.regions: List[RegionDescriptor] = []

        self.parent_indices: List[int] = []
        self.depths: List[int] = []
        self.leaf_status: List[bool] = []
        self.base_is_leaf: bool = True
        self.experts_by_depth: defaultdict = defaultdict(
            list
        )

        self.batched_indicators = BatchedIndicators(
            base_weight=self.base_weight
        )

        self.activations: Dict[str, torch.Tensor] = {}
        self.hook_handles: List[RemovableHandle] = []
        self._timer = None

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def get_highest_depth(self) -> int:
        if not self.depths:
            return 0
        return max(self.depths)

    def get_regions_at_depth(
        self, depth: int, before_epoch: int = None
    ) -> List[RegionDescriptor]:
        result = []
        for i, r in enumerate(self.regions):
            if self.depths[i] == depth:
                if (before_epoch is not None
                        and r.spawn_epoch >= before_epoch):
                    continue
                result.append(r)
        return result

    def get_experts_at_depth(
        self, depth: int
    ) -> List[tuple]:
        result = []
        for i, (expert, region) in enumerate(
            zip(self.experts, self.regions)
        ):
            if self.depths[i] == depth:
                result.append((expert, region))
        return result

    def count_experts_at_depth(self, depth: int) -> int:
        return sum(1 for d in self.depths if d == depth)

    def get_children_of_parent(
        self, parent_idx: int,
        before_epoch: int = None,
    ) -> List[RegionDescriptor]:
        result = []
        for i, r in enumerate(self.regions):
            if self.parent_indices[i] == parent_idx:
                if (before_epoch is not None
                        and r.spawn_epoch >= before_epoch):
                    continue
                result.append(r)
        return result

    def get_leaf_info(self):
        result = []
        if self.base_is_leaf:
            result.append((None, -1))
        for idx, is_leaf in enumerate(self.leaf_status):
            if is_leaf:
                result.append((self.regions[idx], idx))
        return result

    def sync_batched_indicators(self) -> None:
        if not self.regions:
            return
        device = next(self.base_model.parameters()).device
        self.batched_indicators.update(
            regions=self.regions,
            device=device,
            mode=self.blending_mode,
            sigma_fraction=self.sigma_fraction,
            window_type=self.window_type,
            window_smoothness_order=self.window_smoothness_order,
        )

    def spawn_expert(
        self, region: RegionDescriptor
    ) -> int:
        expert_idx = len(self.experts)
        parent_idx = region.parent_idx
        depth = region.depth

        if parent_idx == -1:
            parent_model = self.base_model
        else:
            parent_model = self.experts[parent_idx]

        parent_act_dim = parent_model.get_activation_dim()

        if self.ant_threshold_width is not None:
            # Get the metric value based on configured variable
            if self.variable_for_expert_size == 'norm':
                metric_value = region.wavelet_norm_squared
            elif self.variable_for_expert_size == 'new_norm':
                metric_value = region.new_wavelet_norm_squared
            elif self.variable_for_expert_size == 'smoothness':
                metric_value = region.smoothness_alpha if region.smoothness_alpha is not None else 0.0
            else:
                metric_value = region.wavelet_norm_squared
            
            ratio = max(
                metric_value / self.expert_size_threshold,
                1.0,
            )
            base_w = max(1, round(
                self.ant_threshold_width * ratio
            ))
        else:
            base_w = self.default_hidden_width

        if self.expert_type == 'resnet':
            hidden_layers = [base_w, base_w]
        else:
            hidden_layers = [base_w]

        architecture = (
            [parent_act_dim]
            + hidden_layers
            + [self.output_dim]
        )

        device = next(
            self.base_model.parameters()
        ).device
        expert = create_network(
            architecture, self.activation, self.config,
            is_base=False, expert_type=self.expert_type,
        )
        expert = expert.to(device)

        self.experts.append(expert)
        self.regions.append(region)
        self.parent_indices.append(parent_idx)
        self.depths.append(depth)
        self.leaf_status.append(True)
        self.experts_by_depth[depth].append(expert_idx)

        if parent_idx == -1:
            self.base_is_leaf = False
        else:
            self.leaf_status[parent_idx] = False

        self.sync_batched_indicators()

        parent_info = (
            "Base Model"
            if parent_idx == -1
            else f"E{parent_idx + 1}"
        )
        logger.info(
            f"  Spawned Expert {expert_idx + 1} "
            f"(depth={depth}, parent={parent_info}):"
        )
        logger.info(f"    Architecture: {architecture}")
        logger.info(
            f"    Region bounds: "
            f"{region.bounds_lower} -> "
            f"{region.bounds_upper}"
        )
        logger.info(
            f"    Wavelet norm: "
            f"{region.wavelet_norm_squared:.6f}"
        )
        logger.info(f"    Spawn epoch: {region.spawn_epoch}")

        return expert_idx

    def _collect_leaf_info(self):
        """Build ordered lists of leaf expert indices and their regions."""
        leaf_expert_indices = []
        leaf_regions = []
        if self.base_is_leaf:
            leaf_expert_indices.append(-1)
            leaf_regions.append(None)
        for idx, is_leaf in enumerate(self.leaf_status):
            if is_leaf:
                leaf_expert_indices.append(idx)
                leaf_regions.append(self.regions[idx])
        return leaf_expert_indices, leaf_regions

    def _compute_leaf_psi(self, inputs, leaf_expert_indices, leaf_regions):
        """Compute raw and normalized psi for leaves, with sparse filtering.

        Returns:
            psi_normalized: (N, num_active_leaves) normalized weights
            active_leaf_local: list of local indices into the leaf lists
                               that passed the activation threshold
        """
        N = inputs.size(0)
        device = inputs.device
        dtype = inputs.dtype
        num_leaves = len(leaf_regions)

        threshold = self.adaptive_config.get(
            'expert_activation_threshold', None
        )
        if threshold is not None:
            threshold = float(threshold)

        psi_raw = torch.empty(
            N, num_leaves, device=device, dtype=dtype
        )

        if self.regions:
            _, psi_experts = self.batched_indicators(inputs)
        else:
            psi_experts = None

        for i, region in enumerate(leaf_regions):
            if region is None:
                psi_raw[:, i] = self.base_weight
            else:
                region_idx = self.regions.index(region)
                psi_raw[:, i] = psi_experts[:, region_idx]

        if threshold is not None and num_leaves > 1:
            active_mask = psi_raw > threshold
            _ms = self.adaptive_config.get(
                'relevant_samples_to_activate_expert', 0)
            active_any = active_mask.sum(dim=0) > _ms
            active_leaf_local = torch.nonzero(
                active_any, as_tuple=True
            )[0].tolist()

            if len(active_leaf_local) == 0:
                active_leaf_local = list(range(num_leaves))
        else:
            active_leaf_local = list(range(num_leaves))

        psi_sum = psi_raw.sum(
            dim=1, keepdim=True
        )
        psi_normalized = psi_raw / psi_sum

        return psi_normalized, active_leaf_local

    def _build_path(self, eidx: int) -> list:
        """Return root-to-leaf path as list of expert indices (excludes -1 base)."""
        path = []
        cur = eidx
        while cur >= 0:
            path.append(cur)
            cur = self.parent_indices[cur]
        path.reverse()
        return path

    def _eval_active_paths(self, inputs, node_outputs, node_activations,
                           leaf_expert_indices, active_leaf_local):
        """Evaluate only the expert paths leading to active leaves."""
        for local_idx in active_leaf_local:
            eidx = leaf_expert_indices[local_idx]
            if eidx == -1:
                continue
            for node_idx in self._build_path(eidx):
                if node_idx in node_outputs:
                    continue
                pidx = self.parent_indices[node_idx]
                A_parent = node_activations[pidx]
                u_i, A_i = self.experts[node_idx](
                    A_parent, return_activation=True
                )
                node_outputs[node_idx] = u_i
                node_activations[node_idx] = A_i

    def forward_single_expert(self, expert_idx: int, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single expert, routed through frozen ancestors. No PoU.

        Returns the raw expert output u_j for use in per-expert split loss.
        Ancestors must already be frozen (requires_grad=False) by the caller.
        """
        u_base, A_base = self.base_model(inputs, return_activation=True)

        if expert_idx == -1:
            return u_base

        node_outputs = {-1: u_base}
        node_activations = {-1: A_base}

        path = self._build_path(expert_idx)
        for node_idx in path:
            pidx = self.parent_indices[node_idx]
            A_parent = node_activations[pidx]
            u_i, A_i = self.experts[node_idx](A_parent, return_activation=True)
            node_outputs[node_idx] = u_i
            node_activations[node_idx] = A_i

        return node_outputs[expert_idx]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if len(self.experts) == 0:
            return self.base_model(inputs)

        _t = self._timer

        leaf_expert_indices, leaf_regions = self._collect_leaf_info()

        if _t: _t.start('fwd.compute_masks')
        psi_normalized, active_leaf_local = self._compute_leaf_psi(
            inputs, leaf_expert_indices, leaf_regions
        )
        if _t: _t.stop('fwd.compute_masks')

        if _t: _t.start('fwd.sparse_eval')
        u_base, A_base = self.base_model(
            inputs, return_activation=True
        )
        node_outputs = {-1: u_base}
        node_activations = {-1: A_base}

        self._eval_active_paths(
            inputs, node_outputs, node_activations,
            leaf_expert_indices, active_leaf_local
        )
        if _t: _t.stop('fwd.sparse_eval')

        if _t: _t.start('fwd.blend')
        leaf_outputs = [
            node_outputs[leaf_expert_indices[i]]
            for i in active_leaf_local
        ]
        psi_active = psi_normalized[:, active_leaf_local]

        leaf_stack = torch.stack(leaf_outputs, dim=1)
        result = (
            psi_active.unsqueeze(-1) * leaf_stack
        ).sum(dim=1)
        if _t: _t.stop('fwd.blend')

        return result

    def forward_decomposed(
        self, inputs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        N = inputs.size(0)
        device = inputs.device
        result = {}

        if len(self.experts) == 0:
            u_base = self.base_model(inputs)
            result['base'] = u_base
            result['composed'] = u_base
            result['masks'] = {}
            result['weights_normalized'] = {
                'base': torch.ones(
                    N, 1,
                    device=device, dtype=inputs.dtype,
                )
            }
            return result

        leaf_expert_indices, leaf_regions = self._collect_leaf_info()
        psi_normalized, active_leaf_local = self._compute_leaf_psi(
            inputs, leaf_expert_indices, leaf_regions
        )

        u_base, A_base = self.base_model(
            inputs, return_activation=True
        )
        node_outputs = {-1: u_base}
        node_activations = {-1: A_base}
        result['base'] = u_base

        self._eval_active_paths(
            inputs, node_outputs, node_activations,
            leaf_expert_indices, active_leaf_local
        )
        for eidx in node_outputs:
            if eidx >= 0:
                result[f'expert_{eidx}'] = node_outputs[eidx]

        leaf_outputs = [
            node_outputs[leaf_expert_indices[i]]
            for i in active_leaf_local
        ]
        psi_active = psi_normalized[:, active_leaf_local]

        leaf_stack = torch.stack(leaf_outputs, dim=1)
        composed = (
            psi_active.unsqueeze(-1) * leaf_stack
        ).sum(dim=1)
        result['composed'] = composed

        masks = {}
        weights_normalized = {}
        if self.regions:
            _, psi_experts = self.batched_indicators(inputs)
        else:
            psi_experts = None

        for li, local_idx in enumerate(active_leaf_local):
            eidx = leaf_expert_indices[local_idx]
            if eidx == -1:
                masks['base'] = torch.full(
                    (N, 1), self.base_weight,
                    device=device, dtype=inputs.dtype,
                )
                weights_normalized['base'] = (
                    psi_active[:, li:li + 1]
                )
            else:
                masks[f'expert_{eidx}'] = (
                    psi_experts[:, eidx:eidx + 1]
                )
                weights_normalized[f'expert_{eidx}'] = (
                    psi_active[:, li:li + 1]
                )

        result['masks'] = masks
        result['weights_normalized'] = weights_normalized
        return result

    def reinitialize_base(self):
        """Reinitialize base model weights (fresh random init via reset_parameters)."""
        for module in self.base_model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        n_params = sum(p.numel() for p in self.base_model.parameters())
        logger.info(f"  [Reinit] Base model reinitialized ({n_params} params)")

    def get_domain_bounds(
        self,
    ) -> Dict[str, List[float]]:
        problem = self.config['problem']
        problem_config = self.config[problem]
        spatial_domain = problem_config['spatial_domain']
        temporal_domain = problem_config[
            'temporal_domain'
        ]

        if len(spatial_domain) == 1:
            return {
                'lower': [
                    spatial_domain[0][0],
                    temporal_domain[0],
                ],
                'upper': [
                    spatial_domain[0][1],
                    temporal_domain[1],
                ],
            }
        elif len(spatial_domain) == 2:
            return {
                'lower': [
                    spatial_domain[0][0],
                    spatial_domain[1][0],
                    temporal_domain[0],
                ],
                'upper': [
                    spatial_domain[0][1],
                    spatial_domain[1][1],
                    temporal_domain[1],
                ],
            }
        else:
            raise ValueError(
                f"Unsupported spatial dimension: "
                f"{len(spatial_domain)}"
            )

    def get_layer_names(self) -> List[str]:
        return self.base_model.get_layer_names()

    def register_ncc_hooks(
        self,
        layer_names: List[str],
        keep_gradients: bool = False,
    ) -> List[RemovableHandle]:
        self.remove_hooks()
        self.activations = {}
        handles = self.base_model.register_ncc_hooks(
            layer_names, keep_gradients
        )
        self.hook_handles = handles
        return handles

    def remove_hooks(self):
        self.base_model.remove_hooks()
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.activations = {}

    @property
    def activations(self) -> Dict[str, torch.Tensor]:
        return self.base_model.activations

    @activations.setter
    def activations(self, value):
        self._activations = value

    def get_mask_for_expert(
        self,
        inputs: torch.Tensor,
        expert_idx: int,
    ) -> torch.Tensor:
        if (expert_idx < 0
                or expert_idx >= len(self.regions)):
            return torch.ones(
                inputs.shape[0],
                dtype=torch.bool,
                device=inputs.device,
            )
        all_masks = (
            self.batched_indicators
            .compute_hard_masks_only(inputs)
        )
        if (all_masks.shape[1] == 0
                or expert_idx >= all_masks.shape[1]):
            return torch.ones(
                inputs.shape[0],
                dtype=torch.bool,
                device=inputs.device,
            )
        return all_masks[:, expert_idx].bool()

    def get_union_mask_at_depth(
        self,
        inputs: torch.Tensor,
        depth: int,
        before_epoch: int = None,
    ) -> torch.Tensor:
        N = inputs.shape[0]
        depth_indices = []
        for i, d in enumerate(self.depths):
            if d == depth:
                if (before_epoch is not None
                        and self.regions[i].spawn_epoch
                        >= before_epoch):
                    continue
                depth_indices.append(i)
        if not depth_indices:
            return torch.zeros(
                N, dtype=torch.bool,
                device=inputs.device,
            )
        all_masks = (
            self.batched_indicators
            .compute_hard_masks_only(inputs)
        )
        if all_masks.shape[1] == 0:
            return torch.zeros(
                N, dtype=torch.bool,
                device=inputs.device,
            )
        depth_masks = all_masks[:, depth_indices]
        return depth_masks.any(dim=1)

    def compute_children_coverage(
        self,
        inputs: torch.Tensor,
        parent_idx: int,
        before_epoch: int = None,
    ) -> float:
        all_masks = (
            self.batched_indicators
            .compute_hard_masks_only(inputs)
        )
        if parent_idx == -1:
            parent_mask = torch.ones(
                inputs.shape[0],
                dtype=torch.bool,
                device=inputs.device,
            )
        else:
            if parent_idx >= all_masks.shape[1]:
                return 0.0
            parent_mask = (
                all_masks[:, parent_idx].bool()
            )
        parent_count = parent_mask.sum().item()
        if parent_count == 0:
            return 0.0
        children_indices = []
        for i in range(len(self.parent_indices)):
            if self.parent_indices[i] == parent_idx:
                if (before_epoch is not None
                        and self.regions[i].spawn_epoch
                        >= before_epoch):
                    continue
                children_indices.append(i)
        if not children_indices:
            return 0.0
        children_masks = all_masks[:, children_indices]
        children_union = children_masks.any(dim=1)
        covered = (
            (parent_mask & children_union).sum().item()
        )
        return covered / parent_count

    def state_dict_extended(self) -> Dict:
        return {
            'base_model': self.base_model.state_dict(),
            'experts': [
                e.state_dict() for e in self.experts
            ],
            'expert_architectures': [
                e.layers for e in self.experts
            ],
            'regions': [
                r.to_dict() for r in self.regions
            ],
            'num_experts': len(self.experts),
            'base_architecture': self.base_architecture,
            'activation': self.activation,
            'adaptive_config': self.adaptive_config,
            'parent_indices': self.parent_indices,
            'depths': self.depths,
            'leaf_status': self.leaf_status,
            'base_is_leaf': self.base_is_leaf,
            'experts_by_depth': dict(
                self.experts_by_depth
            ),
        }

    def load_state_dict_extended(
        self, state_dict: Dict
    ):
        saved_base_arch = state_dict.get(
            'base_architecture'
        )
        saved_activation = state_dict.get(
            'activation', self.activation
        )
        saved_adaptive = state_dict.get(
            'adaptive_config', {}
        )
        saved_expert_type = saved_adaptive.get(
            'expert_type', 'mlp'
        )

        if saved_base_arch is None:
            saved_base_arch = (
                self._infer_architecture_from_state_dict(
                    state_dict['base_model']
                )
            )

        if saved_base_arch != self.base_architecture:
            logger.info(
                f"  Recreating base model: "
                f"{self.base_architecture} -> "
                f"{saved_base_arch}"
            )
            device = next(
                self.base_model.parameters()
            ).device
            self.base_model = create_network(
                saved_base_arch, saved_activation,
                self.config, is_base=True,
                expert_type=saved_expert_type,
            )
            self.base_model = self.base_model.to(device)
            self.base_architecture = saved_base_arch

        self.base_model.load_state_dict(
            state_dict['base_model']
        )

        self.experts = nn.ModuleList()
        self.regions = []
        self.parent_indices = state_dict.get(
            'parent_indices', []
        )
        self.depths = state_dict.get('depths', [])
        self.leaf_status = state_dict.get(
            'leaf_status', []
        )
        self.base_is_leaf = state_dict.get(
            'base_is_leaf', True
        )

        saved_by_depth = state_dict.get(
            'experts_by_depth', {}
        )
        self.experts_by_depth = defaultdict(list)
        for k, v in saved_by_depth.items():
            self.experts_by_depth[int(k)] = v

        device = next(
            self.base_model.parameters()
        ).device
        saved_expert_archs = state_dict.get(
            'expert_architectures', None
        )

        for i, (expert_state, region_dict) in enumerate(
            zip(
                state_dict['experts'],
                state_dict['regions'],
            )
        ):
            region = RegionDescriptor.from_dict(
                region_dict
            )
            if saved_expert_archs is not None:
                expert_arch = saved_expert_archs[i]
            else:
                expert_arch = (
                    self._infer_architecture_from_state_dict(
                        expert_state
                    )
                )
            expert = create_network(
                expert_arch, self.activation,
                self.config, is_base=False,
                expert_type=saved_expert_type,
            )
            expert.load_state_dict(expert_state)
            expert = expert.to(device)
            self.experts.append(expert)
            self.regions.append(region)

        self.sync_batched_indicators()

    @staticmethod
    def _infer_architecture_from_state_dict(
        state_dict: Dict,
    ) -> List[int]:
        architecture = []
        layer_idx = 1
        key = f'network.layer_{layer_idx}.weight'
        while key in state_dict:
            weight = state_dict[key]
            if layer_idx == 1:
                architecture.append(weight.shape[1])
            architecture.append(weight.shape[0])
            layer_idx += 1
            key = f'network.layer_{layer_idx}.weight'
        if not architecture:
            raise ValueError(
                "Could not infer architecture "
                "from state dict"
            )
        return architecture

    def debug_composition(self, sample_inputs: torch.Tensor) -> None:
        """Print detailed composition state with a sample input for debugging.
        
        Args:
            sample_inputs: (N, n_dims) sample coordinates for composition verification
        """
        logger.info(f"\n[DEBUG] ANT Composition State:")
        logger.info(f"  Num experts: {len(self.experts)}")
        logger.info(f"  base_is_leaf: {self.base_is_leaf}")
        logger.info(f"  leaf_status: {self.leaf_status}")
        logger.info(f"  parent_indices: {self.parent_indices}")
        logger.info(f"  depths: {self.depths}")
        
        with torch.no_grad():
            # get_leaf_info returns [(region, expert_idx), ...]
            leaf_info = self.get_leaf_info()
            leaf_regions = [r for r, _ in leaf_info]
            leaf_expert_indices = [idx for _, idx in leaf_info]
            num_leaves = len(leaf_expert_indices)
            
            logger.info(f"\n  Gathered {num_leaves} leaves: expert_indices={leaf_expert_indices}")
            
            if num_leaves == 1 and leaf_expert_indices[0] == -1:
                logger.info(f"  Mode: Base-only (no expert spawns yet)")
                return
            
            # Compute psi for leaves
            psi_normalized, active_leaf_local = self._compute_leaf_psi(
                sample_inputs, leaf_expert_indices, leaf_regions
            )
            
            logger.info(f"\n  Sample psi values for {num_leaves} leaves (N={sample_inputs.shape[0]} points):")
            for i, (exp_idx, region) in enumerate(zip(leaf_expert_indices, leaf_regions)):
                psi_i = psi_normalized[:, i] if i < psi_normalized.shape[1] else None
                if psi_i is not None:
                    if region is None:
                        bounds = "base (None)"
                    else:
                        bounds = f"{region.bounds_lower}->{region.bounds_upper}"
                    active_pct = (psi_i > 0.01).float().mean().item() * 100
                    logger.info(f"    psi_leaf[exp={exp_idx}] ({bounds}): "
                          f"min={psi_i.min():.4f}, max={psi_i.max():.4f}, "
                          f"mean={psi_i.mean():.4f}, active%={active_pct:.1f}%")
            
            # Check for normalization issues
            logger.info(f"\n  Normalization check:")
            psi_sum = psi_normalized.sum(dim=1)
            logger.info(f"    psi_normalized.sum: min={psi_sum.min():.6f}, max={psi_sum.max():.6f}")
            
            # Check for NaN
            nan_count = torch.isnan(psi_normalized).sum().item()
            if nan_count > 0:
                logger.info(f"\n  *** CRITICAL WARNING ***: {nan_count} NaN values in psi_normalized!")
            
            # Check raw psi sum before normalization
            if self.regions:
                _, psi_experts = self.batched_indicators(sample_inputs)
                for i, (exp_idx, region) in enumerate(zip(leaf_expert_indices, leaf_regions)):
                    if region is not None:
                        region_idx = self.regions.index(region)
                        raw_psi = psi_experts[:, region_idx]
                        logger.info(f"    raw_psi[exp={exp_idx}]: sum={raw_psi.sum():.4f}")
        
        logger.info("")

    def __repr__(self) -> str:
        base_str = " -> ".join(
            map(str, self.base_architecture)
        )
        lines = [
            "ANT(",
            f"  base: {base_str}",
            f"  activation: {self.activation}",
            f"  blending: {self.blending_mode}",
            f"  num_experts: "
            f"{len(self.experts)}/{self.max_experts}",
            f"  base_is_leaf: {self.base_is_leaf}",
        ]
        for i, region in enumerate(self.regions):
            arch_str = " -> ".join(
                map(str, self.experts[i].layers)
            )
            parent_info = (
                "Base"
                if self.parent_indices[i] == -1
                else f"E{self.parent_indices[i] + 1}"
            )
            leaf_tag = (
                " [leaf]" if self.leaf_status[i]
                else ""
            )
            lines.append(
                f"  expert_{i}: {arch_str}, "
                f"depth={self.depths[i]}, "
                f"parent={parent_info}, "
                f"region="
                f"{region.bounds_lower}->"
                f"{region.bounds_upper}"
                f"{leaf_tag}"
            )
        lines.append(")")
        return "\n".join(lines)
