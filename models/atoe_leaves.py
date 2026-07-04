"""Adaptive Tree-of-Experts PINN using only leaf experts (AToE-Leaves).

Only leaf experts (those with no children) participate in the solution.
When a parent gets children, the parent is removed from the leaf set and
its children are added. Children copy their parent's weights at spawn time.

Blending modes:
- Soft (default): partition of unity over leaves
    u(x,t) = Σ_{j ∈ leaves} ψ̃_j(x,t) · u_j(x,t)
    where ψ̃_j = ψ_j / Σ_{k ∈ leaves} ψ_k

- Hard: normalized hard masks (mean on shared faces)
    u(x,t) = Σ_{j ∈ leaves} (hard_j / Z) · u_j(x,t)
    where Z = Σ_{k ∈ leaves} hard_k (only leaves, NOT root)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Set

from models.fc_model import FCNet
from models.network_factory import create_network
from adaptive.indicators import (
    RegionDescriptor,
    BatchedIndicators
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


class AToELeaves(nn.Module):

    def __init__(
        self,
        base_architecture: List[int],
        activation: str,
        config: Dict,
        adaptive_config: Dict,
        experts_architecture: Optional[List[int]] = None,
    ):
        super().__init__()

        self.base_architecture = base_architecture
        self.activation = activation
        self.config = config
        self.adaptive_config = adaptive_config

        self.max_experts = adaptive_config['max_experts']
        self.sigma_fraction = adaptive_config['sigma_fraction']
        self.base_weight = adaptive_config['base_weight']
        self.base_everywhere = adaptive_config['base_everywhere']
        self.expert_type = adaptive_config['expert_type']
        
        # Blending mode: 'soft' (PoU) or 'hard' (step functions, mean on shared faces)
        self.blending_mode = adaptive_config.get('blending_mode', 'soft')

        self.atoe_threshold_capacity = adaptive_config.get(
            'AToE_threshold_capacity', None
        )
        problem = config['problem']
        problem_config = config[problem]
        self.input_dim = base_architecture[0]
        self.output_dim = problem_config['output_dim']
        
        # Window configuration: smoothstep (compact) or sigmoid (legacy)
        self.window_type = problem_config.get('window_type', 'smoothstep')
        self.window_smoothness_order = problem_config.get('window_smoothness_order', 2)
        if self.atoe_threshold_capacity is not None:
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

        self.leaf_indices: Set[int] = {-1}

        self.config_base_architecture = base_architecture
        self.experts_architecture = list(
            experts_architecture if experts_architecture is not None else base_architecture
        )

        self.base_model = create_network(
            base_architecture, activation, config,
            is_base=True, expert_type=self.expert_type
        )

        self.experts = nn.ModuleList()
        self.regions: List[RegionDescriptor] = []

        self.batched_indicators = BatchedIndicators(base_weight=self.base_weight)

        self._timer = None

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def get_leaf_info(self):
        """Return (region_or_None, expert_idx) for leaf nodes the trainer should try to split."""
        result = []
        if -1 in self.leaf_indices:
            result.append((None, -1))
        for i in sorted(self.leaf_indices):
            if i >= 0:
                result.append((self.regions[i], i))
        return result

    def forward_single_expert(self, expert_idx: int, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single leaf expert. No PoU.

        Returns the raw expert output u_j for use in per-expert split loss.
        AToE-Leaves experts take raw (x,t) coordinates directly.
        """
        if expert_idx == -1:
            return self.base_model(inputs)
        return self.experts[expert_idx](inputs)

    def get_regions_at_depth(self, depth: int, before_epoch: int = None) -> List[RegionDescriptor]:
        result = []
        for r in self.regions:
            if r.depth == depth:
                if before_epoch is not None and r.spawn_epoch >= before_epoch:
                    continue
                result.append(r)
        return result

    def get_experts_at_depth(self, depth: int) -> List[tuple]:
        result = []
        for expert, region in zip(self.experts, self.regions):
            if region.depth == depth:
                result.append((expert, region))
        return result

    def get_highest_depth(self) -> int:
        if not self.regions:
            return 0
        return max(r.depth for r in self.regions)

    def get_union_mask_at_depth(self, inputs: torch.Tensor, depth: int, before_epoch: int = None) -> torch.Tensor:
        N = inputs.shape[0]

        depth_indices = []
        for i, r in enumerate(self.regions):
            if r.depth == depth:
                if before_epoch is not None and r.spawn_epoch >= before_epoch:
                    continue
                depth_indices.append(i)

        if not depth_indices:
            return torch.zeros(N, dtype=torch.bool, device=inputs.device)

        all_masks = self.batched_indicators.compute_hard_masks_only(inputs)  # (N, K)

        if all_masks.shape[1] == 0:
            return torch.zeros(N, dtype=torch.bool, device=inputs.device)

        depth_masks = all_masks[:, depth_indices]  # (N, num_at_depth)
        return depth_masks.any(dim=1)  # (N,)

    def count_experts_at_depth(self, depth: int) -> int:
        return sum(1 for r in self.regions if r.depth == depth)

    def get_children_of_parent(self, parent_idx: int, before_epoch: int = None) -> List[RegionDescriptor]:
        result = []
        for r in self.regions:
            if r.parent_idx == parent_idx:
                if before_epoch is not None and r.spawn_epoch >= before_epoch:
                    continue
                result.append(r)
        return result

    def get_mask_for_expert(self, inputs: torch.Tensor, expert_idx: int) -> torch.Tensor:
        if expert_idx < 0 or expert_idx >= len(self.regions):
            return torch.ones(inputs.shape[0], dtype=torch.bool, device=inputs.device)

        all_masks = self.batched_indicators.compute_hard_masks_only(inputs)  # (N, K)

        if all_masks.shape[1] == 0 or expert_idx >= all_masks.shape[1]:
            return torch.ones(inputs.shape[0], dtype=torch.bool, device=inputs.device)

        return all_masks[:, expert_idx].bool()  # (N,)

    def compute_children_coverage(
        self,
        inputs: torch.Tensor,
        parent_idx: int,
        before_epoch: int = None
    ) -> float:
        all_masks = self.batched_indicators.compute_hard_masks_only(inputs)  # (N, K)

        if parent_idx == -1:
            parent_mask = torch.ones(inputs.shape[0], dtype=torch.bool, device=inputs.device)
        else:
            if parent_idx >= all_masks.shape[1]:
                return 0.0
            parent_mask = all_masks[:, parent_idx].bool()

        parent_count = parent_mask.sum().item()
        if parent_count == 0:
            return 0.0

        children_indices = []
        for i, r in enumerate(self.regions):
            if r.parent_idx == parent_idx:
                if before_epoch is not None and r.spawn_epoch >= before_epoch:
                    continue
                children_indices.append(i)

        if not children_indices:
            return 0.0

        children_masks = all_masks[:, children_indices]  # (N, num_children)
        children_union = children_masks.any(dim=1)  # (N,)

        covered = (parent_mask & children_union).sum().item()

        return covered / parent_count

    def get_expert_architecture(self, region: RegionDescriptor) -> List[int]:
        if self.atoe_threshold_capacity is None:
            return self.experts_architecture

        from models.architecture_bank import get_architecture_for_capacity
        
        # Get the metric value based on configured variable
        if self.variable_for_expert_size == 'norm':
            metric_value = region.wavelet_norm_squared
        elif self.variable_for_expert_size == 'new_norm':
            metric_value = region.new_wavelet_norm_squared
        elif self.variable_for_expert_size == 'smoothness':
            metric_value = region.smoothness_alpha if region.smoothness_alpha is not None else 0.0
        else:
            metric_value = region.wavelet_norm_squared
        
        ratio = max(metric_value / self.expert_size_threshold, 1.0)
        target_capacity = self.atoe_threshold_capacity * ratio
        return get_architecture_for_capacity(
            target_capacity, self.input_dim, self.output_dim
        )

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
            window_smoothness_order=self.window_smoothness_order
        )

    def spawn_expert(self, region: RegionDescriptor, copy_from_idx: Optional[int] = None) -> int:
        expert_idx = len(self.experts)
        architecture = self.get_expert_architecture(region)
        device = next(self.base_model.parameters()).device

        if copy_from_idx is not None and self.atoe_threshold_capacity is None:
            expert = create_network(
                architecture, self.activation, self.config,
                is_base=True, expert_type=self.expert_type
            )
            expert = expert.to(device)
            # Weight copy handled by apply_parent_copy_init in trainer.py
        else:
            expert = create_network(
                architecture, self.activation, self.config,
                is_base=True, expert_type=self.expert_type
            )
            expert = expert.to(device)

        self.experts.append(expert)
        self.regions.append(region)

        self.leaf_indices.add(expert_idx)
        self.leaf_indices.discard(region.parent_idx)

        parent_info = f"Base Model" if region.parent_idx == -1 else f"E{region.parent_idx + 1}"
        logger.info(f"  Spawned Expert {expert_idx + 1} (depth={region.depth}, parent={parent_info}):")
        logger.info(f"    Expert architecture: {architecture}")
        logger.info(f"    Region bounds: {region.bounds_lower} -> {region.bounds_upper}")
        logger.info(f"    Wavelet norm: {region.wavelet_norm_squared:.6f}")
        logger.info(f"    Spawn epoch: {region.spawn_epoch}")

        self.sync_batched_indicators()

        return expert_idx

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Base-only case (no experts spawned yet)
        if -1 in self.leaf_indices:
            return self.base_model(inputs)

        if self.blending_mode == 'hard':
            return self._forward_hard_only_leaves(inputs)
        return self._forward_soft_only_leaves(inputs)

    def _forward_soft_only_leaves(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Soft blending using only leaf experts.

        u(x,t) = Σ_{j ∈ leaves} ψ̃_j · u_j
        where ψ̃_j = ψ_j / Σ_{k ∈ leaves} ψ_k
        
        Note: base-only case is handled in forward() before calling this.
        """
        leaf_list = sorted(self.leaf_indices)
        _, psi_experts = self.batched_indicators(inputs)  # (N, K)
        psi_leaves = psi_experts[:, leaf_list]  # (N, L)
        psi_norm = psi_leaves / psi_leaves.sum(dim=1, keepdim=True)
        u_leaves = torch.stack([self.experts[i](inputs) for i in leaf_list], dim=1)
        return (psi_norm.unsqueeze(-1) * u_leaves).sum(dim=1)

    def _forward_hard_only_leaves(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Hard blending using only leaf experts (normalized hard masks = mean on shared faces).

        u(x,t) = Σ_{j ∈ leaves} (hard_j / Z) · u_j
        where Z = Σ_{k ∈ leaves} hard_k
        
        The normalization is over LEAVES ONLY (root is never in the denominator).
        In the interior of leaf j, only its mask is 1, so Z=1 and weight=1.
        On a face shared by two leaves, both masks are 1, so Z=2 and each gets weight=1/2.
        
        Note: base-only case is handled in forward() before calling this.
        """
        leaf_list = sorted(self.leaf_indices)
        hard_masks = self.batched_indicators.compute_hard_masks_only(inputs)  # (N, K)
        hard_leaves = hard_masks[:, leaf_list]  # (N, L)
        
        # Normalize: Z = sum of hard masks over leaves (NOT including root)
        Z = hard_leaves.sum(dim=1, keepdim=True)  # (N, 1)
        # Guard against Z=0 (shouldn't happen if leaves tile the domain)
        Z = Z.clamp(min=1e-8)
        hard_norm = hard_leaves / Z  # (N, L)
        
        u_leaves = torch.stack([self.experts[i](inputs) for i in leaf_list], dim=1)  # (N, L, out_dim)
        return (hard_norm.unsqueeze(-1) * u_leaves).sum(dim=1)

    def forward_decomposed(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning individual model contributions.

        Used by visualization and diagnostics (not by training losses).

        Returns:
            Dict with per-expert outputs, composed output, masks, and normalized weights.
            Supports both soft and hard blending modes.
        """
        result = {}
        N = inputs.shape[0]
        output_dim = self.base_model.layers[-1]
        device = inputs.device

        if -1 in self.leaf_indices:
            u_base = self.base_model(inputs)
            result['base'] = u_base
            result['composed'] = u_base
            result['masks'] = {}
            result['weights_normalized'] = {'base': torch.ones(N, 1, device=device)}
            result['blending_mode_info'] = 'base_only'
            return result

        leaf_list = sorted(self.leaf_indices)
        
        # Compute masks based on blending mode
        if self.blending_mode == 'hard':
            hard_masks = self.batched_indicators.compute_hard_masks_only(inputs)  # (N, K)
            psi_leaves = hard_masks[:, leaf_list]  # (N, L)
            psi_sum = psi_leaves.sum(dim=1, keepdim=True).clamp(min=1e-8)
            psi_norm = psi_leaves / psi_sum
            blending_info = 'hard_leaves'
        else:
            _, psi_experts = self.batched_indicators(inputs)  # (N, K)
            psi_leaves = psi_experts[:, leaf_list]  # (N, L)
            psi_sum = psi_leaves.sum(dim=1, keepdim=True)
            psi_norm = psi_leaves / psi_sum
            blending_info = 'soft_leaves'

        result['masks'] = {}
        result['weights_normalized'] = {}

        u_total = torch.zeros(N, output_dim, device=device, dtype=inputs.dtype)

        for local_idx, expert_idx in enumerate(leaf_list):
            u_k = self.experts[expert_idx](inputs)
            result[f'expert_{expert_idx}'] = u_k
            result['masks'][f'expert_{expert_idx}'] = psi_leaves[:, local_idx:local_idx+1]
            result['weights_normalized'][f'expert_{expert_idx}'] = psi_norm[:, local_idx:local_idx+1]
            u_total = u_total + psi_norm[:, local_idx:local_idx+1] * u_k

        result['composed'] = u_total
        result['blending_mode_info'] = blending_info
        return result

    def get_layer_names(self) -> List[str]:
        return self.base_model.get_layer_names()

    def get_domain_bounds(self) -> Dict[str, List[float]]:
        problem = self.config['problem']
        problem_config = self.config[problem]
        spatial_domain = problem_config['spatial_domain']
        temporal_domain = problem_config['temporal_domain']

        if len(spatial_domain) == 1:
            return {
                'lower': [spatial_domain[0][0], temporal_domain[0]],
                'upper': [spatial_domain[0][1], temporal_domain[1]]
            }
        elif len(spatial_domain) == 2:
            return {
                'lower': [spatial_domain[0][0], spatial_domain[1][0], temporal_domain[0]],
                'upper': [spatial_domain[0][1], spatial_domain[1][1], temporal_domain[1]]
            }
        else:
            raise ValueError(f"Unsupported spatial dimension: {len(spatial_domain)}")

    def state_dict_extended(self) -> Dict:
        return {
            'base_model': self.base_model.state_dict(),
            'experts': [expert.state_dict() for expert in self.experts],
            'expert_architectures': [e.layers for e in self.experts],
            'regions': [r.to_dict() for r in self.regions],
            'num_experts': len(self.experts),
            'base_architecture': self.base_architecture,
            'config_base_architecture': self.config_base_architecture,
            'experts_architecture': self.experts_architecture,
            'activation': self.activation,
            'adaptive_config': self.adaptive_config,
            'leaf_indices': sorted(self.leaf_indices),
        }

    def load_state_dict_extended(self, state_dict: Dict):
        saved_base_arch = state_dict.get('base_architecture')
        saved_activation = state_dict.get('activation', self.activation)
        saved_adaptive = state_dict.get('adaptive_config', {})
        saved_expert_type = saved_adaptive.get('expert_type', 'mlp')

        if saved_base_arch is None:
            saved_base_arch = self._infer_architecture_from_state_dict(state_dict['base_model'])

        if saved_base_arch != self.base_architecture:
            logger.info(f"  Recreating base model: {self.base_architecture} -> {saved_base_arch}")
            device = next(self.base_model.parameters()).device
            self.base_model = create_network(
                saved_base_arch, saved_activation, self.config,
                is_base=True, expert_type=saved_expert_type
            )
            self.base_model = self.base_model.to(device)
            self.base_architecture = saved_base_arch

        saved_experts_arch_cfg = state_dict.get('experts_architecture')
        if saved_experts_arch_cfg is not None:
            self.experts_architecture = list(saved_experts_arch_cfg)

        self.base_model.load_state_dict(state_dict['base_model'])

        self.experts = nn.ModuleList()
        self.regions = []
        saved_expert_archs = state_dict.get(
            'expert_architectures', None
        )

        for i, (expert_state, region_dict) in enumerate(zip(
            state_dict['experts'], state_dict['regions']
        )):
            region = RegionDescriptor.from_dict(region_dict)

            if saved_expert_archs is not None:
                expert_arch = saved_expert_archs[i]
            else:
                expert_arch = self._infer_architecture_from_state_dict(expert_state)

            expert = create_network(
                expert_arch, self.activation, self.config,
                is_base=True, expert_type=saved_expert_type
            )
            expert.load_state_dict(expert_state)

            device = next(self.base_model.parameters()).device
            expert = expert.to(device)

            self.experts.append(expert)
            self.regions.append(region)

        if 'leaf_indices' in state_dict:
            self.leaf_indices = set(state_dict['leaf_indices'])

        self.sync_batched_indicators()

    @staticmethod
    def _infer_architecture_from_state_dict(state_dict: Dict) -> List[int]:
        architecture = []
        layer_idx = 1
        while f'network.layer_{layer_idx}.weight' in state_dict:
            weight = state_dict[f'network.layer_{layer_idx}.weight']
            if layer_idx == 1:
                architecture.append(weight.shape[1])
            architecture.append(weight.shape[0])
            layer_idx += 1
        if not architecture:
            raise ValueError("Could not infer architecture from state dict")
        return architecture

    def debug_composition(self, sample_inputs: torch.Tensor) -> None:
        """Print detailed composition state with a sample input for debugging.
        
        Args:
            sample_inputs: (N, n_dims) sample coordinates for composition verification
        """
        logger.info(f"\n[DEBUG] AToELeaves Composition State:")
        logger.info(f"  Num experts: {len(self.experts)}")
        logger.info(f"  Leaf indices: {sorted(self.leaf_indices)}")
        logger.info(f"  Base in leaves: {-1 in self.leaf_indices}")
        
        if -1 in self.leaf_indices:
            logger.info(f"  Mode: Base-only (no expert spawns yet)")
            return
        
        with torch.no_grad():
            leaf_list = sorted(self.leaf_indices)
            _, psi_experts = self.batched_indicators(sample_inputs)
            psi_leaves = psi_experts[:, leaf_list]
            
            logger.info(f"\n  Sample psi values for {len(leaf_list)} leaves (N={sample_inputs.shape[0]} points):")
            for i, leaf_idx in enumerate(leaf_list):
                psi_i = psi_leaves[:, i]
                region = self.regions[leaf_idx] if leaf_idx < len(self.regions) else None
                bounds = f"{region.bounds_lower}->{region.bounds_upper}" if region else "?"
                active_pct = (psi_i > 0.01).float().mean().item() * 100
                logger.info(f"    psi_leaf[{leaf_idx}] ({bounds}): "
                      f"min={psi_i.min():.4f}, max={psi_i.max():.4f}, "
                      f"mean={psi_i.mean():.4f}, active%={active_pct:.1f}%")
            
            # Check normalization (this is the potential bug!)
            psi_sum = psi_leaves.sum(dim=1)
            zero_sum_points = (psi_sum < 1e-6).sum().item()
            
            logger.info(f"\n  Normalization check:")
            logger.info(f"    psi_sum: min={psi_sum.min():.6f}, max={psi_sum.max():.6f}, mean={psi_sum.mean():.4f}")
            logger.info(f"    Normalized weights sum: always 1.0 (by definition)")
            
            if zero_sum_points > 0:
                logger.info(f"\n  *** CRITICAL WARNING ***: {zero_sum_points} points have psi_sum < 1e-6!")
                logger.info(f"      This causes division by zero in normalization: psi / psi_sum")
                logger.info(f"      Result: NaN/Inf in model output -> rel-L2 explosion!")
                # Show coordinates of problematic points
                bad_mask = psi_sum < 1e-6
                bad_coords = sample_inputs[bad_mask][:5]  # First 5
                logger.info(f"      Sample bad coordinates: {bad_coords.tolist()}")
        
        logger.info("")

    def __repr__(self) -> str:
        base_str = " -> ".join(map(str, self.base_architecture))
        expert_archs = [
            " -> ".join(map(str, self.get_expert_architecture(i)))
            for i in range(len(self.experts))
        ]

        blending_str = f"{self.blending_mode} (leaves only)"

        repr_str = (
            f"AToELeaves(\n"
            f"  base: {base_str}\n"
            f"  activation: {self.activation}\n"
            f"  blending: {blending_str}\n"
            f"  num_experts: {len(self.experts)}/{self.max_experts}\n"
            f"  leaf_indices: {sorted(self.leaf_indices)}\n"
        )

        for i, (arch, region) in enumerate(zip(expert_archs, self.regions)):
            leaf_marker = " [LEAF]" if i in self.leaf_indices else ""
            repr_str += f"  expert_{i}: {arch}, region={region.bounds_lower}->{region.bounds_upper}{leaf_marker}\n"

        repr_str += ")"
        return repr_str
