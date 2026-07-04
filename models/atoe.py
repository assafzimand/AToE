"""Adaptive Tree of Experts (AToE) PINN with dynamic regional expert spawning.

Implements a composed PINN that combines:
- A global base model u_0(x,t) with constant indicator (active everywhere)
- Regional expert models u_i(x,t) with soft sigmoid indicators

The composed solution depends on the mode:

**Hard blending:**
    u(x,t) = u_0(x,t) + Σ 1_Ωi(x,t) · u_i(x,t)

**Soft blending (partition of unity):**
    u(x,t) = Σ_k ψ̃_k(x,t) · u_k(x,t)
    where ψ̃_k = ψ_k / Σ_j ψ_j (normalized, Σ ψ̃_k = 1)
    Base has constant ψ_0 = base_weight everywhere before normalization.

"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Set
from torch.utils.hooks import RemovableHandle

from models.fc_model import FCNet
from models.network_factory import create_network
from adaptive.indicators import (
    RegionDescriptor,
    BatchedIndicators
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


class BatchedModels:
    """Batched forward pass: loop over [base, *experts], stack results.

    Output tensor indices: [:, 0, :] = base, [:, 1:, :] = experts.
    GPU naturally parallelizes the sequential small model calls.
    """

    def __init__(self):
        self._models: List[nn.Module] = []

    def sync_from_models(self, base_model: nn.Module, experts: nn.ModuleList) -> None:
        """Register all models for batched forward. Call after spawning experts."""
        self._models = [base_model] + list(experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward all models and stack results.

        Args:
            x: Input tensor (N, input_dim)

        Returns:
            (N, K+1, output_dim) where K = number of experts
        """
        if not self._models:
            raise RuntimeError("No models registered. Call sync_from_models first.")
        if len(self._models) == 1:
            return self._models[0](x).unsqueeze(1)  # (N, 1, out_dim)
        return torch.stack([m(x) for m in self._models], dim=1)  # (N, K+1, out_dim)


class AToE(nn.Module):
    """Adaptive Tree of Experts PINN with dynamic regional expert spawning.

    Combines a base model with regional expert models using indicator functions.
    Experts are spawned during training based on wavelet-detected high-error regions.
    All experts always participate in the forward pass (no leaves concept).
    """

    supports_decomposed = True

    def __init__(
        self,
        base_architecture: List[int],
        activation: str,
        config: Dict,
        adaptive_config: Dict,
        experts_architecture: Optional[List[int]] = None,
    ):
        """
        Initialize AToE.

        Args:
            base_architecture: Layer sizes for base model [input_dim, ..., output_dim]
            activation: Activation function name
            config: Full configuration dictionary (for FCNet)
            adaptive_config: Adaptive PINN specific configuration
        """
        super().__init__()

        self.base_architecture = base_architecture
        self.activation = activation
        self.config = config
        self.adaptive_config = adaptive_config

        self.max_experts = adaptive_config['max_experts']
        self.blending_mode = adaptive_config['blending_mode']
        self.sigma_fraction = adaptive_config['sigma_fraction']
        self.base_weight = adaptive_config['base_weight']
        self.base_everywhere = adaptive_config['base_everywhere']
        self.expert_type = adaptive_config['expert_type']

        self.atoe_threshold_capacity = adaptive_config.get(
            'AToE_threshold_capacity', None
        )
        problem = config['problem']
        problem_config = config[problem]
        
        # Window configuration: smoothstep (compact) or sigmoid (legacy)
        self.window_type = problem_config.get('window_type', 'smoothstep')
        self.window_smoothness_order = problem_config.get('window_smoothness_order', 2)
        
        # Composition mode: 'additive' (per-level background) or 'pou' (global partition of unity)
        # AToE uses additive by default; ANT/AToE-Leaves use 'pou' (handled in their classes)
        self.composition_mode = problem_config.get('composition_mode', 'additive')
        
        self.input_dim = base_architecture[0]
        self.output_dim = problem_config['output_dim']
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

        self.leaf_indices: Set[int] = {-1}

        self.batched_indicators = BatchedIndicators(base_weight=self.base_weight)

        self.batched_models = BatchedModels()
        self.batched_models.sync_from_models(self.base_model, self.experts)

        self.activations: Dict[str, torch.Tensor] = {}
        self.hook_handles: List[RemovableHandle] = []

        self._timer = None
        
        # Cache for additive composition (updated by sync_batched_indicators)
        self._expert_depths: Optional[torch.Tensor] = None
        self._max_depth: int = 0
        
        # Additive mode toggle (from adaptive_pinn config)
        # When True: u = u_0 + Σ PoU_ℓ (all levels contribute, default AToE behavior)
        # When False: u = PoU_L (only active_max_depth level contributes)
        self.additive = adaptive_config.get('additive', True)
        
        # For non-additive mode: only experts at this depth are active in forward
        # Set by trainer after each level is trained; None means use all levels
        self.active_max_depth: Optional[int] = None

    @property
    def num_experts(self) -> int:
        """Number of spawned experts (not counting base model)."""
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

    def get_regions_at_depth(self, depth: int, before_epoch: int = None) -> List[RegionDescriptor]:
        """
        Get all regions at a specific depth level.

        Args:
            depth: Depth level (1 = children of base model)
            before_epoch: If provided, only include regions spawned before this epoch

        Returns:
            List of RegionDescriptors at that depth
        """
        result = []
        for r in self.regions:
            if r.depth == depth:
                if before_epoch is not None and r.spawn_epoch >= before_epoch:
                    continue
                result.append(r)
        return result

    def get_experts_at_depth(self, depth: int) -> List[tuple]:
        """
        Get all (expert, region) pairs at a specific depth level.

        Args:
            depth: Depth level (1 = children of base model)

        Returns:
            List of (expert, region) tuples at that depth
        """
        result = []
        for expert, region in zip(self.experts, self.regions):
            if region.depth == depth:
                result.append((expert, region))
        return result

    def get_highest_depth(self) -> int:
        """
        Get the highest depth level with at least one expert.

        Returns:
            Maximum depth (0 if no experts spawned yet)
        """
        if not self.regions:
            return 0
        return max(r.depth for r in self.regions)

    def get_union_mask_at_depth(self, inputs: torch.Tensor, depth: int, before_epoch: int = None) -> torch.Tensor:
        """
        Get a boolean mask for points inside ANY region at the given depth.

        Uses vectorized batched indicators for efficient GPU computation.

        Args:
            inputs: (N, n_dims) tensor of coordinates
            depth: Depth level to check
            before_epoch: If provided, only include regions spawned before this epoch

        Returns:
            (N,) boolean tensor - True if point is inside any depth-d region
        """
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
        """Count number of experts at a specific depth."""
        return sum(1 for r in self.regions if r.depth == depth)

    def get_children_of_parent(self, parent_idx: int, before_epoch: int = None) -> List[RegionDescriptor]:
        """
        Get all child regions of a specific parent.

        Args:
            parent_idx: Index of the parent expert (-1 for base model)
            before_epoch: If provided, only include regions spawned before this epoch

        Returns:
            List of RegionDescriptors that are children of the specified parent
        """
        result = []
        for r in self.regions:
            if r.parent_idx == parent_idx:
                if before_epoch is not None and r.spawn_epoch >= before_epoch:
                    continue
                result.append(r)
        return result

    def get_mask_for_expert(self, inputs: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """
        Get boolean mask for points inside a specific expert's region.

        Uses batched indicators for efficient GPU computation.

        Args:
            inputs: (N, n_dims) tensor of coordinates
            expert_idx: Index of the expert

        Returns:
            (N,) boolean tensor - True if point is inside the expert's region
        """
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
        """
        Compute what fraction of a parent's domain is covered by its children.

        Uses vectorized batched indicators for efficient GPU computation.

        Args:
            inputs: (N, n_dims) tensor of coordinates (eval points)
            parent_idx: Index of the parent expert (-1 for base model)
            before_epoch: If provided, only include children spawned before this epoch

        Returns:
            Coverage fraction (0.0 to 1.0)
        """
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
        """Get architecture for a new expert based on region metric."""
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
        """Synchronize batched indicators with current regions.

        Call this after spawning experts to update the batched tensors
        used for vectorized indicator computation.
        """
        if not self.regions:
            self._expert_depths = None
            self._max_depth = 0
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
        
        # Cache expert depths as tensor for efficient per-level grouping in additive composition
        self._expert_depths = torch.tensor(
            [r.depth for r in self.regions], dtype=torch.long, device=device
        )
        self._max_depth = max(r.depth for r in self.regions) if self.regions else 0
        
        # Log composition mode on first sync with experts
        if len(self.regions) > 0 and not getattr(self, '_logged_composition_mode', False):
            if self.composition_mode == 'additive':
                logger.info(f"  [AToE] Composition: additive per-level (base=1, per-level Z_ℓ = 1 + Σ_ℓ Ψ)")
            else:
                logger.info(f"  [AToE] Composition: partition of unity (global Z = Σ Ψ)")
            self._logged_composition_mode = True

    def sync_batched_models(self) -> None:
        """Synchronize batched models (base + experts) for O(1) forward pass.

        Call this after spawning experts or loading state dict to update
        the batched structure used for parallel computation.
        """
        self.batched_models.sync_from_models(self.base_model, self.experts)

    def reinitialize_base(self):
        """Reinitialize base model weights (fresh random init via reset_parameters).

        Preserves expert regions and weights. Used in 3-phase training when
        reinitialize_base_after_spawn=True to give the base a clean start for Phase 3.
        """
        for module in self.base_model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        n_params = sum(p.numel() for p in self.base_model.parameters())
        logger.info(f"  [Reinit] Base model reinitialized ({n_params} params)")
        self.batched_models.sync_from_models(self.base_model, self.experts)

    def spawn_expert(self, region: RegionDescriptor,
                     zero_init: bool = True) -> int:
        """
        Spawn a new expert PINN for the given region.

        Args:
            region: RegionDescriptor defining the expert's domain (includes depth)
            zero_init: If True, zero-initialize the final layer so the expert
                       initially contributes nothing (smooth integration during
                       on-the-fly spawning). If False, keep PyTorch default
                       random init (e.g. when reinitialize_base_after_spawn is set
                       and all M_term experts are created at once before Phase 3).

        Returns:
            Index of the new expert
        """
        expert_idx = len(self.experts)
        architecture = self.get_expert_architecture(region)
        device = next(self.base_model.parameters()).device

        expert = create_network(
            architecture, self.activation, self.config,
            is_base=True, expert_type=self.expert_type
        )
        expert = expert.to(device)
        if zero_init:
            if self.expert_type == 'resnet':
                nn.init.zeros_(expert.output_proj.weight)
                if expert.output_proj.bias is not None:
                    nn.init.zeros_(expert.output_proj.bias)
            else:
                layer_names = expert.get_layer_names()
                if layer_names:
                    final_layer = expert.network[layer_names[-1]]
                    nn.init.zeros_(final_layer.weight)
                    if final_layer.bias is not None:
                        nn.init.zeros_(final_layer.bias)

        self.experts.append(expert)
        self.regions.append(region)

        self.leaf_indices.add(expert_idx)
        self.leaf_indices.discard(region.parent_idx)

        parent_info = "Base Model" if region.parent_idx == -1 else f"E{region.parent_idx + 1}"
        logger.info(f"  Spawned Expert {expert_idx + 1} (depth={region.depth}, parent={parent_info}):")
        logger.info(f"    Expert architecture: {architecture}")
        logger.info(f"    Region bounds: {region.bounds_lower} -> {region.bounds_upper}")
        logger.info(f"    Wavelet norm: {region.wavelet_norm_squared:.6f}")
        logger.info(f"    Spawn epoch: {region.spawn_epoch}")

        self.sync_batched_indicators()
        self.sync_batched_models()

        return expert_idx

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Composed forward pass.

        For hard blending:
            u(x,t) = u_0(x,t) + Σ 1_Ωi(x,t) · u_i(x,t)

        For soft blending (partition of unity):
            u(x,t) = Σ_k ψ̃_k(x,t) · u_k(x,t)
            where ψ̃_k = ψ_k / Σ_j ψ_j (normalized weights, including base)

        Args:
            inputs: (N, n_dims) tensor of coordinates [x, t] or [x, y, t]

        Returns:
            u: (N, output_dim) composed solution
        """
        threshold = self.adaptive_config.get('expert_activation_threshold', None)
        if threshold is not None:
            threshold = float(threshold)

        # Non-additive mode: only use experts at active_max_depth (single level)
        if not self.additive and self.active_max_depth is not None:
            if self.blending_mode == 'hard':
                return self._forward_hard_single_level(inputs, self.active_max_depth)
            else:
                return self._forward_soft_additive_single_level(inputs, self.active_max_depth)
        
        if threshold is not None and len(self.experts) > 0:
            if self.blending_mode == 'hard':
                return self._forward_hard_sparse(inputs, threshold)
            else:
                return self._forward_soft_sparse(inputs, threshold)
        else:
            if self.blending_mode == 'hard':
                return self._forward_hard(inputs)
            elif self.composition_mode == 'additive' or self.additive:
                # AToE additive: u = u_0 + Σ w_i · u_i with per-level normalization
                return self._forward_soft_additive(inputs)
            else:
                # Legacy PoU: u = Σ ψ̃_k · u_k with global normalization
                return self._forward_soft(inputs)

    def _forward_hard(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Vectorized hard blending forward pass.

        u(x,t) = u_0(x,t) + Σ 1_Ωi(x,t) · u_i(x,t)

        Args:
            inputs: (N, n_dims) input coordinates
        """
        _t = self._timer

        if _t: _t.start('fwd.batched_models')
        u_all = self.batched_models.forward(inputs)  # (N, K+1, output_dim)
        if _t: _t.stop('fwd.batched_models')

        u_base = u_all[:, 0, :]  # (N, output_dim)

        if len(self.experts) == 0:
            return u_base

        u_experts = u_all[:, 1:, :]  # (N, K, output_dim)

        if _t: _t.start('fwd.compute_masks')
        masks = self.batched_indicators.compute_hard_masks_only(inputs)  # (N, K)
        if _t: _t.stop('fwd.compute_masks')

        if _t: _t.start('fwd.blend')
        weighted_experts = masks.unsqueeze(-1) * u_experts  # (N, K, out_dim)
        u_total = u_base + weighted_experts.sum(dim=1)  # (N, out_dim)
        if _t: _t.stop('fwd.blend')

        return u_total

    def _forward_soft(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Vectorized soft blending forward pass with partition-of-unity normalization.

        u(x,t) = Σ_k ψ̃_k(x,t) · u_k(x,t)
        where:
            - ψ_0 = uniform constant (base_weight) for base model
            - ψ_k = sigmoid-based bump function for expert k
            - ψ̃_k = ψ_k / Σ_j ψ_j (partition of unity: Σ ψ̃_k = 1)

        Args:
            inputs: (N, n_dims) input coordinates
        """
        _t = self._timer

        if _t: _t.start('fwd.compute_masks')
        psi_base, psi_experts = self.batched_indicators(inputs)
        if _t: _t.stop('fwd.compute_masks')

        if _t: _t.start('fwd.batched_models')
        u_all = self.batched_models.forward(inputs)  # (N, K+1, output_dim)
        if _t: _t.stop('fwd.batched_models')

        u_base = u_all[:, 0, :]  # (N, output_dim)

        if len(self.experts) == 0:
            return u_base

        u_experts = u_all[:, 1:, :]  # (N, K, output_dim)

        if _t: _t.start('fwd.blend')
        psi_sum = psi_base + psi_experts.sum(dim=1, keepdim=True)  # (N, 1)
        psi_base_norm = psi_base / psi_sum  # (N, 1)
        psi_experts_norm = psi_experts / psi_sum  # (N, K)

        weighted_experts = psi_experts_norm.unsqueeze(-1) * u_experts  # (N, K, out_dim)
        u_total = psi_base_norm * u_base + weighted_experts.sum(dim=1)  # (N, out_dim)
        if _t: _t.stop('fwd.blend')

        return u_total

    def _forward_soft_additive(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Additive composition with per-level PoU normalization (AToE).

        u(X) = u_0(X) + Σ_{ℓ=1..L} Σ_{i: level(i)=ℓ} w_i(X) · u_i(X)
        
        where:
            w_i(X) = Ψ_i(X) / Σ_{k: level(k)=ℓ(i)} Ψ_k(X)
        
        Key properties:
            - Base/root contributes with coefficient exactly 1 (not normalized)
            - Each level has its own denominator Z_ℓ = Σ_{level ℓ} Ψ (clean PoU)
            - With retain_siblings=True, each level forms a complete domain tiling
            - Composition grows incrementally as levels are spawned

        Args:
            inputs: (N, n_dims) input coordinates
        """
        _t = self._timer

        if _t: _t.start('fwd.compute_masks')
        _, psi_experts = self.batched_indicators(inputs)  # psi_base unused in additive
        if _t: _t.stop('fwd.compute_masks')

        if _t: _t.start('fwd.batched_models')
        u_all = self.batched_models.forward(inputs)  # (N, K+1, output_dim)
        if _t: _t.stop('fwd.batched_models')

        u_base = u_all[:, 0, :]  # (N, output_dim)

        if len(self.experts) == 0:
            return u_base

        u_experts = u_all[:, 1:, :]  # (N, K, output_dim)

        if _t: _t.start('fwd.blend')
        
        N = inputs.shape[0]
        K = len(self.experts)
        device = inputs.device
        
        # Compute per-level normalized weights (clean PoU: w_i = Ψ_i / Σ_{level} Ψ)
        psi_experts_norm = torch.zeros_like(psi_experts)  # (N, K)
        
        for depth in range(1, self._max_depth + 1):
            # Mask for experts at this depth
            depth_mask = (self._expert_depths == depth)  # (K,)
            if not depth_mask.any():
                continue
            
            # Sum of Ψ at this level (complete tiling with retain_siblings=True)
            psi_at_level = psi_experts[:, depth_mask]  # (N, num_at_depth)
            Z_level = psi_at_level.sum(dim=1, keepdim=True)  # (N, 1)
            
            # Normalized weights for this level
            psi_experts_norm[:, depth_mask] = psi_at_level / Z_level
        
        # Additive composition: u = u_0 + Σ w_i · u_i
        weighted_experts = psi_experts_norm.unsqueeze(-1) * u_experts  # (N, K, out_dim)
        u_total = u_base + weighted_experts.sum(dim=1)  # (N, out_dim)
        
        if _t: _t.stop('fwd.blend')

        return u_total

    def forward_single_expert(self, expert_idx: int, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through a single expert (raw output, no PoU blending).
        
        Used by split_loss for per-expert training.
        
        Args:
            expert_idx: Index of the expert in self.experts
            inputs: (N, n_dims) input coordinates
            
        Returns:
            (N, output_dim) raw expert output
        """
        return self.experts[expert_idx](inputs)

    def forward_frozen_composition(self, inputs: torch.Tensor, max_depth: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass through frozen composition up to a given depth.
        
        Computes u_0 + Σ_{ℓ=1..max_depth} PoU_ℓ(experts_at_ℓ) for additive AToE split_loss.
        When training level L, this gives the frozen background from levels 0..(L-1).
        
        Args:
            inputs: (N, n_dims) input coordinates
            max_depth: Maximum depth to include (None = use active_max_depth - 1)
            
        Returns:
            (N, output_dim) composed output up to max_depth
        """
        if max_depth is None:
            max_depth = (self.active_max_depth - 1) if self.active_max_depth else 0
        
        u_base = self.base_model(inputs)
        
        if max_depth <= 0 or len(self.experts) == 0:
            return u_base
        
        _, psi_experts = self.batched_indicators(inputs)
        
        # Evaluate all experts
        u_experts = torch.stack([exp(inputs) for exp in self.experts], dim=1)  # (N, K, out_dim)
        
        # Compute per-level PoU for levels 1..max_depth
        psi_experts_norm = torch.zeros_like(psi_experts)
        
        for depth in range(1, max_depth + 1):
            depth_mask = (self._expert_depths == depth)
            if not depth_mask.any():
                continue
            
            psi_at_level = psi_experts[:, depth_mask]
            Z_level = psi_at_level.sum(dim=1, keepdim=True)
            psi_experts_norm[:, depth_mask] = psi_at_level / Z_level
        
        weighted_experts = psi_experts_norm.unsqueeze(-1) * u_experts
        return u_base + weighted_experts.sum(dim=1)

    def _forward_soft_additive_single_level(self, inputs: torch.Tensor, target_depth: int) -> torch.Tensor:
        """
        Soft forward pass using only experts at a single depth level (non-additive mode).
        
        u(X) = Σ_{i: depth(i)=target_depth} [Ψ_i / Σ_{k: depth(k)=target_depth} Ψ_k] · u_i(X)
        
        Args:
            inputs: (N, n_dims) input coordinates
            target_depth: The depth level to use for composition
            
        Returns:
            (N, output_dim) composed output from single level
        """
        _, psi_experts = self.batched_indicators(inputs)
        
        # Get experts at target depth
        depth_mask = (self._expert_depths == target_depth)
        if not depth_mask.any():
            return self.base_model(inputs)
        
        # Evaluate only experts at target depth
        expert_indices = torch.nonzero(depth_mask, as_tuple=True)[0]
        u_experts = torch.stack([self.experts[i](inputs) for i in expert_indices], dim=1)
        
        # PoU over target depth only
        psi_at_level = psi_experts[:, depth_mask]
        Z_level = psi_at_level.sum(dim=1, keepdim=True)
        psi_norm = psi_at_level / Z_level
        
        weighted = psi_norm.unsqueeze(-1) * u_experts
        return weighted.sum(dim=1)

    def _forward_hard_single_level(self, inputs: torch.Tensor, target_depth: int) -> torch.Tensor:
        """
        Hard forward pass using only experts at a single depth level (non-additive mode).
        
        u(X) = Σ_{i: depth(i)=target_depth} 1_{Ω_i}(X) · u_i(X)
        
        Args:
            inputs: (N, n_dims) input coordinates
            target_depth: The depth level to use for composition
            
        Returns:
            (N, output_dim) composed output from single level with hard masks
        """
        masks, _ = self.batched_indicators(inputs)
        
        # Get experts at target depth
        depth_mask = (self._expert_depths == target_depth)
        if not depth_mask.any():
            return self.base_model(inputs)
        
        # Evaluate only experts at target depth
        expert_indices = torch.nonzero(depth_mask, as_tuple=True)[0]
        u_experts = torch.stack([self.experts[i](inputs) for i in expert_indices], dim=1)
        
        # Hard masks for level (no normalization)
        masks_at_level = masks[:, depth_mask]  # (N, num_at_depth)
        
        weighted = masks_at_level.unsqueeze(-1) * u_experts
        return weighted.sum(dim=1)

    def _forward_hard_sparse(self, inputs: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Sparse hard blending: only evaluate experts with mask == 1.

        u(x,t) = u_0(x,t) + Σ_{k: mask_k=1} u_k(x,t)

        Args:
            inputs: (N, n_dims) input coordinates
            threshold: Minimum mask value (typically 0 for hard masks)
        """
        _t = self._timer
        N = inputs.shape[0]
        output_dim = self.base_model.layers[-1]
        device = inputs.device

        if _t: _t.start('fwd.compute_masks')
        masks = self.batched_indicators.compute_hard_masks_only(inputs)  # (N, K)
        if _t: _t.stop('fwd.compute_masks')

        if _t: _t.start('fwd.sparse_selection')
        _ms = self.adaptive_config.get(
            'relevant_samples_to_activate_expert', 0)
        active_experts_any = (masks.sum(dim=0) > _ms)  # (K,)
        active_expert_indices = torch.nonzero(active_experts_any, as_tuple=True)[0]
        num_active = len(active_expert_indices)
        if _t: _t.stop('fwd.sparse_selection')

        if _t: _t.start('fwd.sparse_eval')
        u_base = self.base_model(inputs)  # (N, output_dim)

        if num_active == 0:
            if _t: _t.stop('fwd.sparse_eval')
            return u_base

        u_experts_sparse = torch.zeros(N, len(self.experts), output_dim, device=device)

        for expert_idx in active_expert_indices:
            expert_idx_item = expert_idx.item()
            u_experts_sparse[:, expert_idx_item, :] = self.experts[expert_idx_item](inputs)

        if _t: _t.stop('fwd.sparse_eval')

        if _t: _t.start('fwd.blend')
        weighted_experts = masks.unsqueeze(-1) * u_experts_sparse  # (N, K, out_dim)
        u_total = u_base + weighted_experts.sum(dim=1)  # (N, out_dim)
        if _t: _t.stop('fwd.blend')

        return u_total

    def _forward_soft_sparse(self, inputs: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Sparse soft blending: only evaluate experts with psi > threshold.

        Normalization uses the full set of experts (same as non-sparse).
        Inactive experts contribute 0 to the output (their u_k is not evaluated).

        Args:
            inputs: (N, n_dims) input coordinates
            threshold: Minimum psi value to evaluate expert (e.g., 1e-4)
        """
        _t = self._timer
        N = inputs.shape[0]
        output_dim = self.base_model.layers[-1]
        device = inputs.device

        if _t: _t.start('fwd.compute_masks')
        psi_base, psi_experts = self.batched_indicators(inputs)  # psi_base: (N, 1), psi_experts: (N, K)
        if _t: _t.stop('fwd.compute_masks')

        if _t: _t.start('fwd.sparse_selection')
        active_mask = psi_experts > threshold  # (N, K)
        _ms = self.adaptive_config.get(
            'relevant_samples_to_activate_expert', 0)
        active_experts_any = active_mask.sum(dim=0) > _ms  # (K,)
        active_expert_indices = torch.nonzero(active_experts_any, as_tuple=True)[0]
        num_active = len(active_expert_indices)
        if _t: _t.stop('fwd.sparse_selection')

        if _t: _t.start('fwd.sparse_eval')
        u_base = self.base_model(inputs)  # (N, output_dim)

        if num_active == 0:
            if _t: _t.stop('fwd.sparse_eval')
            return u_base

        u_experts_sparse = torch.zeros(N, len(self.experts), output_dim, device=device)

        for expert_idx in active_expert_indices:
            expert_idx_item = expert_idx.item()
            u_experts_sparse[:, expert_idx_item, :] = self.experts[expert_idx_item](inputs)

        if _t: _t.stop('fwd.sparse_eval')

        if _t: _t.start('fwd.blend')
        
        # Handle composition mode
        if self.composition_mode == 'additive' or self.additive:
            # Additive: base contributes at weight 1, experts get per-level clean PoU
            psi_experts_norm = torch.zeros_like(psi_experts)  # (N, K)
            
            for depth in range(1, self._max_depth + 1):
                depth_mask = (self._expert_depths == depth)  # (K,)
                if not depth_mask.any():
                    continue
                psi_at_level = psi_experts[:, depth_mask]  # (N, num_at_depth)
                Z_level = psi_at_level.sum(dim=1, keepdim=True)  # (N, 1) - clean PoU, no 1+
                psi_experts_norm[:, depth_mask] = psi_at_level / Z_level
            
            weighted_experts = psi_experts_norm.unsqueeze(-1) * u_experts_sparse  # (N, K, out_dim)
            u_total = u_base + weighted_experts.sum(dim=1)  # (N, out_dim)
        else:
            # Legacy PoU: global normalization
            psi_sum = psi_base + psi_experts.sum(dim=1, keepdim=True)  # (N, 1)
            psi_base_norm = psi_base / psi_sum  # (N, 1)
            psi_experts_norm = psi_experts / psi_sum  # (N, K)

            weighted_experts = psi_experts_norm.unsqueeze(-1) * u_experts_sparse  # (N, K, out_dim)
            u_total = psi_base_norm * u_base + weighted_experts.sum(dim=1)  # (N, out_dim)

        if _t: _t.stop('fwd.blend')

        return u_total

    def forward_for_pde_derivatives(self, inputs: torch.Tensor) -> dict:
        """
        Forward pass returning decomposed components for product-rule derivative computation.

        Instead of returning the composed scalar output, returns individual expert outputs
        and their normalized weights. The loss function uses these to compute PDE derivatives
        via the product rule, creating K small autograd graphs instead of one massive one.

        u(x,t) = Σ_k ψ̃_k(x,t) · u_k(x,t),  k includes base
        where ψ̃_k = ψ_k / Z, Z = Σ_j ψ_j

        Product rule gives: u_x = Σ_k (ψ̃_k_x · u_k + ψ̃_k · u_k_x), etc.

        Args:
            inputs: (N, n_dims) input coordinates. The x,t components must
                    have requires_grad=True (set by the loss function).

        Returns:
            dict with:
                - 'components': list of dicts, each with:
                    - 'u': (N, output_dim) model output, on autograd graph
                    - 'inputs': (N, D) per-expert input copy (for batched autograd)
                    - 'psi_norm': (N, 1) normalized weight, on autograd graph
                    - 'constant_psi': bool, True if psi_norm is constant (skip indicator derivatives)
                - 'composed': (N, output_dim) assembled output (uses detached psi for efficiency)
                - 'indicator_data': dict with bounds/sigma for analytical derivatives
        """
        _t = self._timer
        N = inputs.shape[0]
        output_dim = self.base_model.layers[-1]
        device = inputs.device

        threshold = self.adaptive_config.get('expert_activation_threshold', None)
        if threshold is not None:
            threshold = float(threshold)

        if _t: _t.start('fwd.compute_masks')
        psi_base, psi_experts = self.batched_indicators(inputs)  # (N, 1), (N, K)
        if _t: _t.stop('fwd.compute_masks')

        if _t: _t.start('fwd.sparse_selection')
        K = psi_experts.shape[1]

        if threshold is not None and K > 0:
            active_mask = psi_experts > threshold  # (N, K)
            _ms = self.adaptive_config.get(
                'relevant_samples_to_activate_expert', 0)
            active_experts_any = active_mask.sum(dim=0) > _ms  # (K,)
            active_expert_indices = torch.nonzero(active_experts_any, as_tuple=True)[0]
        elif K > 0:
            active_expert_indices = torch.arange(K, device=device)
        else:
            active_expert_indices = []

        if _t: _t.stop('fwd.sparse_selection')

        # Compute normalized weights based on composition mode
        if (self.composition_mode == 'additive' or self.additive) and K > 0:
            # Additive: base gets weight 1, experts get per-level clean PoU
            psi_norm_base = torch.ones(N, 1, device=device, dtype=inputs.dtype)
            psi_norm_experts = torch.zeros_like(psi_experts)  # (N, K)
            
            for depth in range(1, self._max_depth + 1):
                depth_mask = (self._expert_depths == depth)  # (K,)
                if not depth_mask.any():
                    continue
                psi_at_level = psi_experts[:, depth_mask]  # (N, num_at_depth)
                Z_level = psi_at_level.sum(dim=1, keepdim=True)  # (N, 1) - clean PoU, no 1+
                psi_norm_experts[:, depth_mask] = psi_at_level / Z_level
        else:
            # Legacy PoU: global normalization
            Z = psi_base + psi_experts.sum(dim=1, keepdim=True)  # (N, 1)
            psi_norm_base = psi_base / Z  # (N, 1)
            psi_norm_experts = psi_experts / Z  # (N, K)

        if _t: _t.start('fwd.sparse_eval')
        components = []

        inputs_base = inputs.detach().clone().requires_grad_(True)
        u_base = self.base_model(inputs_base)  # (N, output_dim)
        
        # For additive mode, base has constant weight = 1 (no gradient through weight)
        base_constant_psi = (self.composition_mode == 'additive' or self.additive)
        components.append({
            'u': u_base,
            'inputs': inputs_base,
            'psi_norm': psi_norm_base,
            'constant_psi': base_constant_psi,
        })

        for idx in active_expert_indices:
            k = idx.item() if torch.is_tensor(idx) else idx
            inputs_k = inputs.detach().clone().requires_grad_(True)
            u_k = self.experts[k](inputs_k)  # (N, output_dim)
            components.append({
                'u': u_k,
                'inputs': inputs_k,
                'psi_norm': psi_norm_experts[:, k:k+1],  # (N, 1)
                'constant_psi': False,
            })

        if _t: _t.stop('fwd.sparse_eval')

        composed = torch.zeros(N, output_dim, device=device, dtype=inputs.dtype)
        for c in components:
            composed = composed + c['psi_norm'].detach() * c['u']

        # LEGACY: indicator_data was for analytical derivatives (sigmoid window specific)
        # With smoothstep windows, all_delta replaces all_sigma but the analytical
        # derivative path is disabled anyway. Keep structure for backward compat.
        indicator_data = {
            'all_lower': self.batched_indicators.all_lower,   # (K, D) or None
            'all_upper': self.batched_indicators.all_upper,   # (K, D) or None
            'all_delta': self.batched_indicators.all_delta,   # (K, D) or None (was all_sigma)
            'psi_base': psi_base,                             # (N, 1)
            'psi_experts_filtered': psi_experts,              # (N, K) full psi for normalization
            'active_expert_indices': active_expert_indices,   # tensor of active indices
        }

        return {
            'components': components,
            'composed': composed,
            'indicator_data': indicator_data,
        }

    def forward_decomposed(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning individual model contributions.

        Useful for analysis and debugging.

        For hard blending: Uses efficient filtering (only compute for inside points)
        For soft blending: Returns unnormalized masks and normalized weights

        Args:
            inputs: (N, n_dims) tensor of coordinates

        Returns:
            Dict with:
                - 'base': base model output (N, output_dim)
                - 'expert_0', 'expert_1', ...: expert outputs (N, output_dim)
                - 'composed': final composed output (N, output_dim)
                - 'masks': dict of unnormalized masks per expert
                - 'weights_normalized': (soft only) dict of normalized weights per model
        """
        if self.blending_mode == 'hard':
            return self._forward_decomposed_hard(inputs)
        else:
            return self._forward_decomposed_soft(inputs)

    def _forward_decomposed_hard(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Vectorized hard blending decomposed forward pass with batched models."""
        result = {}

        u_all = self.batched_models.forward(inputs)  # (N, K+1, output_dim)
        result['base'] = u_all[:, 0, :]  # (N, output_dim)

        all_masks = self.batched_indicators.compute_hard_masks_only(inputs)

        result['masks'] = {}

        if len(self.experts) > 0:
            for i in range(len(self.experts)):
                result[f'expert_{i}'] = u_all[:, i+1, :]  # (N, out_dim)
                result['masks'][f'expert_{i}'] = all_masks[:, i:i+1]  # (N, 1)

            u_experts = u_all[:, 1:, :]  # (N, K, output_dim)
            weighted_experts = all_masks.unsqueeze(-1) * u_experts  # (N, K, out_dim)
            u_total = result['base'] + weighted_experts.sum(dim=1)  # (N, out_dim)
        else:
            u_total = result['base'].clone()

        result['composed'] = u_total
        return result

    def _forward_decomposed_soft(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Vectorized soft blending decomposed forward pass with batched models."""
        result = {}
        N = inputs.shape[0]
        device = inputs.device
        K = len(self.experts)

        psi_base, psi_experts = self.batched_indicators(inputs)  # (N, 1), (N, K)

        result['masks'] = {'base': psi_base}
        for i in range(psi_experts.shape[1]):
            result['masks'][f'expert_{i}'] = psi_experts[:, i:i+1]  # (N, 1)

        # Compute normalized weights based on composition mode
        if (self.composition_mode == 'additive' or self.additive) and K > 0:
            # Additive: base = 1, experts get per-level clean PoU
            psi_base_norm = torch.ones(N, 1, device=device, dtype=inputs.dtype)
            psi_experts_norm = torch.zeros_like(psi_experts)
            for depth in range(1, self._max_depth + 1):
                depth_mask = (self._expert_depths == depth)
                if not depth_mask.any():
                    continue
                psi_at_level = psi_experts[:, depth_mask]
                Z_level = psi_at_level.sum(dim=1, keepdim=True)  # clean PoU, no 1+
                psi_experts_norm[:, depth_mask] = psi_at_level / Z_level
            result['blending_mode_info'] = 'additive_per_level'
        else:
            # Legacy PoU: global normalization
            psi_sum = psi_base + psi_experts.sum(dim=1, keepdim=True)
            psi_base_norm = psi_base / psi_sum
            psi_experts_norm = psi_experts / psi_sum
            result['blending_mode_info'] = 'partition_of_unity'

        result['weights_normalized'] = {'base': psi_base_norm}
        for i in range(psi_experts_norm.shape[1]):
            result['weights_normalized'][f'expert_{i}'] = psi_experts_norm[:, i:i+1]

        u_all = self.batched_models.forward(inputs)  # (N, K+1, output_dim)
        u_base = u_all[:, 0, :]  # (N, output_dim)
        result['base'] = u_base

        if K > 0:
            u_experts = u_all[:, 1:, :]  # (N, K, output_dim)

            for i in range(K):
                result[f'expert_{i}'] = u_experts[:, i, :]

            weighted_experts = psi_experts_norm.unsqueeze(-1) * u_experts
            u_total = psi_base_norm * u_base + weighted_experts.sum(dim=1)
        else:
            u_total = u_base

        result['composed'] = u_total
        return result

    def get_layer_names(self) -> List[str]:
        """Get layer names from base model (for tracker compatibility)."""
        return self.base_model.get_layer_names()

    def register_ncc_hooks(
        self,
        layer_names: List[str],
        keep_gradients: bool = False
    ) -> List[RemovableHandle]:
        """
        Register hooks on base model layers for NCC/probe analysis.

        Args:
            layer_names: Layers to hook (from base model)
            keep_gradients: Whether to keep gradients (for derivatives tracking)

        Returns:
            List of RemovableHandle
        """
        self.remove_hooks()
        self.activations = {}

        handles = self.base_model.register_ncc_hooks(layer_names, keep_gradients)
        self.hook_handles = handles

        return handles

    def remove_hooks(self):
        """Remove all registered hooks."""
        self.base_model.remove_hooks()
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.activations = {}

    @property
    def activations(self) -> Dict[str, torch.Tensor]:
        """Get activations from base model (for tracker compatibility)."""
        return self.base_model.activations

    @activations.setter
    def activations(self, value):
        """Set activations (for initialization)."""
        self._activations = value

    def get_domain_bounds(self) -> Dict[str, List[float]]:
        """Get domain bounds from problem configuration."""
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
        """Get extended state dict including regions and indicators."""
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
        }

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

    def load_state_dict_extended(self, state_dict: Dict):
        """
        Load extended state dict including regions and indicators.

        Handles architecture mismatches by recreating models with the correct
        architecture from the checkpoint.

        Args:
            state_dict: Dict from state_dict_extended()
        """
        saved_base_arch = state_dict.get('base_architecture')
        saved_activation = state_dict.get('activation', self.activation)

        if saved_base_arch is None:
            saved_base_arch = self._infer_architecture_from_state_dict(state_dict['base_model'])

        saved_adaptive = state_dict.get('adaptive_config', {})
        saved_expert_type = saved_adaptive.get('expert_type', 'mlp')

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

        self.sync_batched_indicators()
        self.sync_batched_models()

    def debug_composition(self, sample_inputs: torch.Tensor) -> None:
        """Print detailed composition state with a sample input for debugging.
        
        Args:
            sample_inputs: (N, n_dims) sample coordinates for composition verification
        """
        logger.info(f"\n[DEBUG] AToE Composition State:")
        logger.info(f"  Blending mode: {self.blending_mode}")
        logger.info(f"  Window type: {self.window_type}")
        logger.info(f"  Num experts: {len(self.experts)}")
        logger.info(f"  Base weight: {self.base_weight}")
        
        with torch.no_grad():
            # Get indicators
            psi_base, psi_experts = self.batched_indicators(sample_inputs)
            
            logger.info(f"\n  Sample psi values (N={sample_inputs.shape[0]} points):")
            logger.info(f"    psi_base: min={psi_base.min():.4f}, max={psi_base.max():.4f}, mean={psi_base.mean():.4f}")
            
            if psi_experts.shape[1] > 0:
                for i in range(psi_experts.shape[1]):
                    psi_i = psi_experts[:, i]
                    depth = self._expert_depths[i].item() if hasattr(self, '_expert_depths') else '?'
                    region = self.regions[i] if i < len(self.regions) else None
                    bounds = f"{region.bounds_lower}->{region.bounds_upper}" if region else "?"
                    active_pct = (psi_i > 0.01).float().mean().item() * 100
                    logger.info(f"    psi_expert[{i}] (depth={depth}, {bounds}): "
                          f"min={psi_i.min():.4f}, max={psi_i.max():.4f}, "
                          f"mean={psi_i.mean():.4f}, active%={active_pct:.1f}%")
            
            # Verify normalization for additive mode
            if self.blending_mode == 'soft' and self.composition_mode == 'additive':
                logger.info(f"\n  Additive composition check:")
                for depth in range(1, getattr(self, '_max_depth', 1) + 1):
                    if hasattr(self, '_expert_depths'):
                        depth_mask = (self._expert_depths == depth)
                        if depth_mask.any():
                            psi_at_level = psi_experts[:, depth_mask]
                            Z_level = 1.0 + psi_at_level.sum(dim=1)
                            w_sum = psi_at_level.sum(dim=1) / Z_level
                            logger.info(f"    Level {depth}: Z=1+sum(psi)={Z_level.mean():.4f}, "
                                  f"w_sum={w_sum.mean():.4f} (should be <1)")
            
            # Check for potential issues
            total_psi = psi_base.sum(dim=1) + psi_experts.sum(dim=1)
            zero_psi_points = (total_psi < 1e-6).sum().item()
            if zero_psi_points > 0:
                logger.info(f"\n  WARNING: {zero_psi_points} points have near-zero total psi!")
        
        logger.info("")

    def __repr__(self) -> str:
        """String representation."""
        base_str = " -> ".join(map(str, self.base_architecture))
        expert_archs = [
            " -> ".join(map(str, self.get_expert_architecture(i)))
            for i in range(len(self.experts))
        ]

        repr_str = (
            f"AToE(\n"
            f"  base: {base_str}\n"
            f"  activation: {self.activation}\n"
            f"  blending: {self.blending_mode}\n"
            f"  num_experts: {len(self.experts)}/{self.max_experts}\n"
        )

        for i, (arch, region) in enumerate(zip(expert_archs, self.regions)):
            repr_str += f"  expert_{i}: {arch}, region={region.bounds_lower}->{region.bounds_upper}\n"

        repr_str += ")"
        return repr_str
