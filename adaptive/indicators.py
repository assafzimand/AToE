"""Indicator functions for adaptive expert regions.

Provides hard (step function) and soft (smooth sigmoid) indicator functions
for defining expert regions as axis-aligned bounding boxes.

Also provides BatchedIndicators for computing all indicator masks (base + K experts)
in a single vectorized GPU operation for maximum efficiency.

Window types:
- 'sigmoid' (legacy): product-of-sigmoids, non-compact support, C^∞ but never exactly 0/1
- 'smoothstep' (new): compact smoothstep windows, flat-top=1 inside, C^N ramps in collar,
   exact 0 outside. Requires window_smoothness_order N >= PDE spatial order.
"""

import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple


# -----------------------------------------------------------------------------
# Smoothstep polynomials S_N(t), t in [0,1], C^N continuity
# These are the Hermite interpolation polynomials with vanishing derivatives at 0 and 1.
# -----------------------------------------------------------------------------

def _smoothstep_1(t: torch.Tensor) -> torch.Tensor:
    """C^1 smoothstep: S_1(t) = 3t^2 - 2t^3"""
    return 3 * t**2 - 2 * t**3


def _smoothstep_2(t: torch.Tensor) -> torch.Tensor:
    """C^2 smoothstep: S_2(t) = 6t^5 - 15t^4 + 10t^3"""
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def _smoothstep_3(t: torch.Tensor) -> torch.Tensor:
    """C^3 smoothstep: S_3(t) = 35t^4 - 84t^5 + 70t^6 - 20t^7"""
    return 35 * t**4 - 84 * t**5 + 70 * t**6 - 20 * t**7


def _smoothstep_4(t: torch.Tensor) -> torch.Tensor:
    """C^4 smoothstep: S_4(t) = 126t^5 - 420t^6 + 540t^7 - 315t^8 + 70t^9"""
    return 126 * t**5 - 420 * t**6 + 540 * t**7 - 315 * t**8 + 70 * t**9


_SMOOTHSTEP_FNS = {1: _smoothstep_1, 2: _smoothstep_2, 3: _smoothstep_3, 4: _smoothstep_4}


def smoothstep_N(t: torch.Tensor, N: int) -> torch.Tensor:
    """
    Evaluate the C^N smoothstep polynomial S_N(t) for t in [0, 1].
    
    Properties: S_N(0)=0, S_N(1)=1, S_N^(k)(0)=S_N^(k)(1)=0 for k=1..N.
    
    Args:
        t: Input tensor (any shape), assumed to be in [0, 1]
        N: Smoothness order (1, 2, 3, or 4)
        
    Returns:
        S_N(t) with same shape as t
    """
    if N not in _SMOOTHSTEP_FNS:
        raise ValueError(f"Unsupported smoothstep order N={N}. Supported: {list(_SMOOTHSTEP_FNS.keys())}")
    return _SMOOTHSTEP_FNS[N](t)


def compact_ramp(s: torch.Tensor, N: int) -> torch.Tensor:
    """
    One-sided compact ramp ρ_N(s): exactly 0 for s<=0, S_N(s) for 0<s<1, exactly 1 for s>=1.
    
    This is the building block for compact smoothstep windows.
    
    Args:
        s: Input tensor (any shape)
        N: Smoothness order
        
    Returns:
        ρ_N(s) with same shape as s
    """
    # Clamp to [0, 1] so we get exact 0 outside and exact 1 inside
    s_clamped = torch.clamp(s, 0.0, 1.0)
    return smoothstep_N(s_clamped, N)


@dataclass
class RegionDescriptor:
    """Describes an axis-aligned rectangular region in the domain.
    
    Attributes:
        bounds_lower: Lower bounds for each dimension [x_min, t_min] or [x_min, y_min, t_min]
        bounds_upper: Upper bounds for each dimension [x_max, t_max] or [x_max, y_max, t_max]
        wavelet_norm_squared: L2 norm of the geometric wavelet (refinement priority)
        new_wavelet_norm_squared: New norm (sum of children's classic norms)
        spawn_epoch: Epoch at which this region's expert was spawned
        depth: Depth level in the expert tree (1 = child of base model)
        parent_idx: Index of the parent expert (-1 for depth-1, base model is parent)
    """
    bounds_lower: List[float]
    bounds_upper: List[float]
    wavelet_norm_squared: float = 0.0
    new_wavelet_norm_squared: float = 0.0
    spawn_epoch: int = 0
    depth: int = 1  # Depth level (1 = child of base)
    parent_idx: int = -1  # Parent expert index (-1 = base model)
    smoothness_alpha: Optional[float] = None  # local tree-Besov smoothness; larger = smoother
    
    @property
    def n_dims(self) -> int:
        """Number of dimensions in the region."""
        return len(self.bounds_lower)
    
    @property
    def volume(self) -> float:
        """Compute the volume (area in 2D) of the region."""
        vol = 1.0
        for lo, hi in zip(self.bounds_lower, self.bounds_upper):
            vol *= (hi - lo)
        return vol
    
    def contains(self, point: List[float]) -> bool:
        """Check if a point is inside the region."""
        for i, (lo, hi) in enumerate(zip(self.bounds_lower, self.bounds_upper)):
            if point[i] < lo or point[i] > hi:
                return False
        return True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'bounds_lower': self.bounds_lower,
            'bounds_upper': self.bounds_upper,
            'wavelet_norm_squared': self.wavelet_norm_squared,
            'new_wavelet_norm_squared': self.new_wavelet_norm_squared,
            'spawn_epoch': self.spawn_epoch,
            'depth': self.depth,
            'parent_idx': self.parent_idx,
            'smoothness_alpha': self.smoothness_alpha,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'RegionDescriptor':
        """Create from dictionary."""
        return cls(
            bounds_lower=d['bounds_lower'],
            bounds_upper=d['bounds_upper'],
            wavelet_norm_squared=d.get('wavelet_norm_squared', 0.0),
            new_wavelet_norm_squared=d.get('new_wavelet_norm_squared', 0.0),
            spawn_epoch=d.get('spawn_epoch', 0),
            depth=d.get('depth', 1),
            parent_idx=d.get('parent_idx', -1),
            smoothness_alpha=d.get('smoothness_alpha', None),
        )


class HardIndicator:
    """Hard (step function) indicator for axis-aligned box regions.
    
    Returns 1 if point is inside the region, 0 otherwise.
    Uses vectorized boolean masks for GPU efficiency.
    """
    
    def __init__(self, region: RegionDescriptor):
        """
        Args:
            region: RegionDescriptor defining the bounding box
        """
        self.region = region
        self._lower: Optional[torch.Tensor] = None
        self._upper: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None
    
    def _ensure_tensors(self, device: torch.device, dtype: torch.dtype):
        """Lazily create tensors on the correct device and dtype."""
        if self._device != device or (self._lower is not None and self._lower.dtype != dtype):
            self._lower = torch.tensor(self.region.bounds_lower, dtype=dtype, device=device)
            self._upper = torch.tensor(self.region.bounds_upper, dtype=dtype, device=device)
            self._device = device
    
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute indicator mask for batch of points.
        
        Args:
            inputs: (N, n_dims) tensor of coordinates [x, t] or [x, y, t]
            
        Returns:
            mask: (N, 1) float tensor - 1.0 if inside region, 0.0 otherwise
        """
        self._ensure_tensors(inputs.device, inputs.dtype)
        
        # Vectorized comparison: all dimensions must be within bounds
        # inputs >= lower: (N, n_dims) bool tensor
        # inputs <= upper: (N, n_dims) bool tensor
        inside = (inputs >= self._lower) & (inputs <= self._upper)  # (N, n_dims)
        
        # Point is inside if ALL dimensions are inside
        mask = inside.all(dim=1, keepdim=True)  # (N, 1)
        
        return mask.to(inputs.dtype)
    
    def get_bounds(self) -> tuple:
        """Return bounds as tuple of lists."""
        return (self.region.bounds_lower, self.region.bounds_upper)


class SoftIndicator:
    """Soft indicator for axis-aligned box regions.
    
    Supports two window types:
    - 'sigmoid' (legacy): product-of-sigmoids, C^∞ but non-compact (never exactly 0 or 1)
    - 'smoothstep' (new): compact flat-top window, C^N, exact 0 outside collar, exact 1 inside region
    
    The sigma/collar parameter is computed as a fraction of each dimension's region size,
    providing scale-invariant smoothness.
    """
    
    def __init__(
        self, 
        region: RegionDescriptor, 
        sigma_fraction: float = 0.2,
        window_type: str = 'smoothstep',
        window_smoothness_order: int = 2
    ):
        """
        Args:
            region: RegionDescriptor defining the bounding box
            sigma_fraction: Fraction of region size to use as collar/sigma per dimension.
                           For smoothstep: collar half-width δ = σ_frac × region_size.
                           For sigmoid: sigma for the sigmoid transitions.
            window_type: 'smoothstep' (compact, default) or 'sigmoid' (legacy)
            window_smoothness_order: For smoothstep, the C^N order (1, 2, 3, or 4).
                                    Rule: N >= PDE spatial order.
        """
        self.region = region
        self.sigma_fraction = sigma_fraction
        self.window_type = window_type
        self.window_smoothness_order = window_smoothness_order
        self._lower: Optional[torch.Tensor] = None
        self._upper: Optional[torch.Tensor] = None
        self._delta: Optional[torch.Tensor] = None  # Per-dimension collar width (smoothstep) or sigma (sigmoid)
        self._device: Optional[torch.device] = None
    
    def _ensure_tensors(self, device: torch.device, dtype: torch.dtype):
        """Lazily create tensors on the correct device and dtype."""
        if self._device != device or (self._lower is not None and self._lower.dtype != dtype):
            self._lower = torch.tensor(self.region.bounds_lower, dtype=dtype, device=device)
            self._upper = torch.tensor(self.region.bounds_upper, dtype=dtype, device=device)
            # Compute delta per dimension as fraction of region size
            region_sizes = self._upper - self._lower
            self._delta = self.sigma_fraction * region_sizes  # (n_dims,)
            # Ensure minimum delta to avoid numerical issues
            self._delta = torch.clamp(self._delta, min=1e-6)
            self._device = device
    
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute soft indicator mask for batch of points.
        
        Args:
            inputs: (N, n_dims) tensor of coordinates [x, t] or [x, y, t]
            
        Returns:
            mask: (N, 1) float tensor - smooth value in [0, 1]
        """
        self._ensure_tensors(inputs.device, inputs.dtype)
        
        if self.window_type == 'smoothstep':
            return self._compute_smoothstep_mask(inputs)
        else:
            return self._compute_sigmoid_mask(inputs)
    
    def _compute_sigmoid_mask(self, inputs: torch.Tensor) -> torch.Tensor:
        """Legacy sigmoid window (non-compact)."""
        # Distance from lower bound (positive = inside), scaled by per-dim sigma
        dist_lower = (inputs - self._lower) / self._delta  # Broadcasting (N, n_dims)
        # Distance from upper bound (positive = inside), scaled by per-dim sigma
        dist_upper = (self._upper - inputs) / self._delta  # Broadcasting (N, n_dims)
        
        # Sigmoid gives smooth 0→1 transition
        weight_lower = torch.sigmoid(dist_lower)  # (N, n_dims)
        weight_upper = torch.sigmoid(dist_upper)  # (N, n_dims)
        
        # Product over dimensions for intersection of half-spaces
        mask = (weight_lower * weight_upper).prod(dim=1, keepdim=True)  # (N, 1)
        return mask
    
    def _compute_smoothstep_mask(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compact smoothstep window: flat-top=1 inside, C^N ramps in collar, exact 0 outside."""
        N = self.window_smoothness_order
        delta = self._delta  # (n_dims,)
        
        # For each dimension j:
        # s_lo = (X_j - (a_j - δ_j)) / δ_j  →  0 at outer edge (a-δ), 1 at inner edge (a)
        # s_hi = ((b_j + δ_j) - X_j) / δ_j  →  1 at inner edge (b), 0 at outer edge (b+δ)
        # ω_j = ρ_N(s_lo) × ρ_N(s_hi)
        
        s_lo = (inputs - (self._lower - delta)) / delta  # (N_pts, n_dims)
        s_hi = ((self._upper + delta) - inputs) / delta  # (N_pts, n_dims)
        
        # Apply compact ramp to each side, then multiply
        ramp_lo = compact_ramp(s_lo, N)  # (N_pts, n_dims)
        ramp_hi = compact_ramp(s_hi, N)  # (N_pts, n_dims)
        omega_per_dim = ramp_lo * ramp_hi  # (N_pts, n_dims)
        
        # Tensor product over all dimensions
        mask = omega_per_dim.prod(dim=1, keepdim=True)  # (N_pts, 1)
        return mask
    
    def get_bounds(self) -> tuple:
        """Return bounds as tuple of lists."""
        return (self.region.bounds_lower, self.region.bounds_upper)


def create_indicator(
    region: RegionDescriptor, 
    mode: str = 'hard', 
    sigma_fraction: float = 0.2,
    window_type: str = 'smoothstep',
    window_smoothness_order: int = 2
):
    """Factory function to create an indicator of the specified type.
    
    Args:
        region: RegionDescriptor defining the bounding box
        mode: 'hard' or 'soft'
        sigma_fraction: For soft indicator, fraction of region size to use as collar/sigma.
        window_type: For soft mode, 'smoothstep' (compact, default) or 'sigmoid' (legacy)
        window_smoothness_order: For smoothstep, the C^N order (1, 2, 3, or 4)
        
    Returns:
        HardIndicator or SoftIndicator instance
    """
    if mode == 'hard':
        return HardIndicator(region)
    elif mode == 'soft':
        return SoftIndicator(
            region, 
            sigma_fraction=sigma_fraction,
            window_type=window_type,
            window_smoothness_order=window_smoothness_order
        )
    else:
        raise ValueError(f"Unknown indicator mode: {mode}. Use 'hard' or 'soft'.")


class BatchedIndicators:
    """Compute all indicator masks (base + K experts) in one vectorized GPU operation.
    
    This class provides significant speedup over calling individual indicators
    in a loop by batching all computations into single tensor operations.
    
    Supports:
    - hard: step function (1 inside, 0 outside)
    - soft + sigmoid: legacy product-of-sigmoids (non-compact)
    - soft + smoothstep: compact flat-top windows (C^N, exact 0 outside collar)
    """
    
    def __init__(self):
        self.all_lower: Optional[torch.Tensor] = None  # (K, D)
        self.all_upper: Optional[torch.Tensor] = None  # (K, D)
        self.all_delta: Optional[torch.Tensor] = None  # (K, D) collar/sigma for soft mode
        self.mode: str = 'hard'
        self.sigma_fraction: float = 0.2
        self.window_type: str = 'smoothstep'
        self.window_smoothness_order: int = 2
        self._num_experts: int = 0
        self._initialized: bool = False
    
    @property
    def num_experts(self) -> int:
        """Number of expert regions currently tracked."""
        return self._num_experts
    
    def update(
        self, 
        regions: List[RegionDescriptor], 
        device: torch.device, 
        mode: str = 'hard',
        sigma_fraction: float = 0.2,
        window_type: str = 'smoothstep',
        window_smoothness_order: int = 2
    ) -> None:
        """
        Update batched tensors from list of regions.
        
        Call this after spawning new experts to sync the batched state.
        
        Args:
            regions: List of RegionDescriptors for all experts
            device: Target device (cuda or cpu)
            mode: 'hard' or 'soft' blending mode
            sigma_fraction: For soft mode, fraction of region size to use as collar/sigma
            window_type: For soft mode, 'smoothstep' (compact) or 'sigmoid' (legacy)
            window_smoothness_order: For smoothstep, the C^N order (1, 2, 3, or 4)
        """
        self.mode = mode
        self.sigma_fraction = sigma_fraction
        self.window_type = window_type
        self.window_smoothness_order = window_smoothness_order
        self._num_experts = len(regions)
        
        if len(regions) == 0:
            self.all_lower = None
            self.all_upper = None
            self.all_delta = None
            self._initialized = False
            return
        
        # Stack all bounds into (K, D) tensors. Use the default dtype so
        # float64 runs keep the exact float64 bounds (a float32 round-trip
        # shifts region faces by ~1e-8*scale relative to the bounds used to
        # assign split-training points).
        _dtype = torch.get_default_dtype()
        self.all_lower = torch.stack([
            torch.tensor(r.bounds_lower, dtype=_dtype, device=device)
            for r in regions
        ])  # (K, D)

        self.all_upper = torch.stack([
            torch.tensor(r.bounds_upper, dtype=_dtype, device=device)
            for r in regions
        ])  # (K, D)
        
        # For soft mode, precompute collar/sigma per region per dimension
        if mode == 'soft':
            region_sizes = self.all_upper - self.all_lower
            self.all_delta = (sigma_fraction * region_sizes).clamp(min=1e-6)  # (K, D)
        else:
            self.all_delta = None
        
        self._initialized = True
        
        # Log window config on first update with experts
        if len(regions) > 0 and not getattr(self, '_logged_window_config', False):
            window_desc = f"smoothstep C^{window_smoothness_order}" if window_type == 'smoothstep' else "sigmoid (legacy)"
            print(f"  [BatchedIndicators] Window: {window_desc}, collar_fraction={sigma_fraction}")
            self._logged_window_config = True
    
    def __call__(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute base weight + all K expert masks in one vectorized operation.
        
        Args:
            inputs: (N, D) tensor of coordinates [x, t] or [x, y, t]
            
        Returns:
            Tuple of:
                - psi_base: (N, 1) base model weights (uniform)
                - psi_experts: (N, K) expert indicator masks
        """
        N_pts = inputs.shape[0]
        device = inputs.device
        dtype = inputs.dtype

        # Base weight is uniform across all points (leaves-only PoU discards it)
        psi_base = torch.ones(N_pts, 1, device=device, dtype=dtype)
        
        # Handle case with no experts
        if self.all_lower is None or self._num_experts == 0:
            return psi_base, torch.empty((N_pts, 0), device=device, dtype=dtype)
        
        # Ensure bounds are on correct device and dtype
        if self.all_lower.device != device or self.all_lower.dtype != dtype:
            self.all_lower = self.all_lower.to(device=device, dtype=dtype)
            self.all_upper = self.all_upper.to(device=device, dtype=dtype)
            if self.all_delta is not None:
                self.all_delta = self.all_delta.to(device=device, dtype=dtype)
        
        if self.mode == 'hard':
            psi_experts = self._compute_hard_masks(inputs)
        else:
            psi_experts = self._compute_soft_masks(inputs)
        
        return psi_base, psi_experts
    
    def _compute_hard_masks(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute all hard indicator masks in one vectorized operation.
        
        Args:
            inputs: (N, D) tensor of coordinates
            
        Returns:
            masks: (N, K) float tensor - 1.0 if inside, 0.0 if outside
        """
        # Expand for broadcasting: (N, 1, D) vs (1, K, D) -> (N, K, D)
        x = inputs.unsqueeze(1)  # (N, 1, D)
        lower = self.all_lower.unsqueeze(0)  # (1, K, D)
        upper = self.all_upper.unsqueeze(0)  # (1, K, D)
        
        # Check all dimensions at once
        inside = (x >= lower) & (x <= upper)  # (N, K, D)
        
        # Point is inside region if ALL dimensions are inside
        masks = inside.all(dim=2).float()  # (N, K)
        
        return masks
    
    def _compute_soft_masks(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute all soft indicator masks via broadcasting.
        
        Dispatches to sigmoid (legacy) or smoothstep (compact) based on window_type.

        Args:
            inputs: (N, D) tensor of coordinates

        Returns:
            masks: (N, K) float tensor - smooth values in [0, 1]
        """
        if self.window_type == 'smoothstep':
            return self._compute_smoothstep_masks(inputs)
        else:
            return self._compute_sigmoid_masks(inputs)
    
    def _compute_sigmoid_masks(self, inputs: torch.Tensor) -> torch.Tensor:
        """Legacy sigmoid window (non-compact, C^∞ but never exactly 0 or 1)."""
        # Broadcasting: (N, 1, D) vs (1, K, D) -> (N, K, D)
        x = inputs.unsqueeze(1)              # (N, 1, D)
        lower = self.all_lower.unsqueeze(0)  # (1, K, D)
        upper = self.all_upper.unsqueeze(0)  # (1, K, D)
        delta = self.all_delta.unsqueeze(0)  # (1, K, D)

        dist_lower = (x - lower) / delta    # (N, K, D)
        dist_upper = (upper - x) / delta    # (N, K, D)

        # Product over dimensions: point inside if all dims have high indicator value
        masks = (torch.sigmoid(dist_lower) * torch.sigmoid(dist_upper)).prod(dim=2)  # (N, K)
        return masks
    
    def _compute_smoothstep_masks(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compact smoothstep window: flat-top=1 inside, C^N ramps in collar, exact 0 outside.
        
        For each region i, dimension j:
          s_lo = (X_j - (a_{ij} - δ_{ij})) / δ_{ij}  →  0 at outer edge, 1 at inner edge
          s_hi = ((b_{ij} + δ_{ij}) - X_j) / δ_{ij}  →  1 at inner edge, 0 at outer edge
          ω_{ij} = ρ_N(s_lo) × ρ_N(s_hi)
          Ψ_i = ∏_j ω_{ij}
        """
        N = self.window_smoothness_order
        
        # Broadcasting: (N_pts, 1, D) vs (1, K, D) -> (N_pts, K, D)
        x = inputs.unsqueeze(1)              # (N_pts, 1, D)
        lower = self.all_lower.unsqueeze(0)  # (1, K, D)
        upper = self.all_upper.unsqueeze(0)  # (1, K, D)
        delta = self.all_delta.unsqueeze(0)  # (1, K, D)
        
        # Compute s_lo and s_hi for all points, all regions, all dims
        s_lo = (x - (lower - delta)) / delta  # (N_pts, K, D)
        s_hi = ((upper + delta) - x) / delta  # (N_pts, K, D)
        
        # Apply compact ramp: clamp to [0,1], then apply smoothstep
        ramp_lo = compact_ramp(s_lo, N)  # (N_pts, K, D)
        ramp_hi = compact_ramp(s_hi, N)  # (N_pts, K, D)
        
        # Per-dimension window: product of left and right ramps
        omega_per_dim = ramp_lo * ramp_hi  # (N_pts, K, D)
        
        # Tensor product over all dimensions → region indicator
        masks = omega_per_dim.prod(dim=2)  # (N_pts, K)
        return masks
    
    def compute_hard_masks_only(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute ONLY hard masks (no base), useful for coverage checks.
        
        Always uses hard indicator logic regardless of self.mode.
        
        Args:
            inputs: (N, D) tensor of coordinates
            
        Returns:
            masks: (N, K) float tensor - 1.0 if inside, 0.0 if outside
        """
        if self.all_lower is None or self._num_experts == 0:
            return torch.empty((inputs.shape[0], 0), device=inputs.device, dtype=inputs.dtype)
        
        # Ensure bounds are on correct device
        device = inputs.device
        if self.all_lower.device != device:
            self.all_lower = self.all_lower.to(device)
            self.all_upper = self.all_upper.to(device)
        
        return self._compute_hard_masks(inputs)
