"""
Time Marching Model Wrapper.

Wraps multiple window-specific models and routes queries to the appropriate
model based on the temporal coordinate.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Any


class TimeMarchingModel(nn.Module):
    """
    Wrapper that routes queries to the appropriate time window's model.
    
    For time marching, we train separate models for each window. This wrapper
    combines them into a single model that can be evaluated over the full
    temporal domain by routing each query point to the correct window model.
    
    The piecewise nature is transparent to evaluation code - it just calls
    forward() and gets the combined output.
    """
    
    def __init__(self, window_models: List[Tuple[Any, nn.Module]]):
        """
        Initialize the time marching model wrapper.
        
        Args:
            window_models: List of (TimeWindow, model) tuples, sorted by t_start
        """
        super().__init__()
        
        # Store windows and models
        self.window_info = []
        models_list = []
        
        for window, model in window_models:
            self.window_info.append({
                'idx': window.idx,
                't_start': window.t_start,
                't_end': window.t_end,
                'M': window.M,
            })
            models_list.append(model)
        
        # Store as ModuleList for proper parameter tracking
        self.models = nn.ModuleList(models_list)
        
        # Cache window boundaries for fast lookup
        self.t_starts = torch.tensor([w['t_start'] for w in self.window_info])
        self.t_ends = torch.tensor([w['t_end'] for w in self.window_info])
    
    @property
    def num_windows(self) -> int:
        """Number of time windows."""
        return len(self.models)
    
    @property
    def total_experts(self) -> int:
        """Total number of experts across all windows."""
        total = 0
        for model in self.models:
            if hasattr(model, 'num_experts'):
                total += model.num_experts
        return total
    
    def get_window_idx(self, t: torch.Tensor) -> torch.Tensor:
        """
        Return window index for each t value.
        
        Args:
            t: Tensor of shape (N, 1) or (N,) containing temporal coordinates
        
        Returns:
            Tensor of shape (N,) with integer window indices
        """
        t_flat = t.view(-1)
        device = t_flat.device
        
        # Move boundaries to same device
        t_starts = self.t_starts.to(device)
        t_ends = self.t_ends.to(device)
        
        window_idx = torch.zeros_like(t_flat, dtype=torch.long)
        
        for i in range(len(self.window_info)):
            mask = (t_flat >= t_starts[i]) & (t_flat < t_ends[i])
            window_idx[mask] = i
        
        # Handle t == t_max (assign to last window)
        window_idx[t_flat >= t_ends[-1]] = len(self.models) - 1
        
        return window_idx
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Route each input to correct window model and combine outputs.
        
        Args:
            inputs: Tensor of shape (N, d_in) where the last column is t
        
        Returns:
            Tensor of shape (N, d_out) with combined predictions
        """
        # Extract temporal coordinate (last column)
        t = inputs[:, -1:]
        window_idx = self.get_window_idx(t)
        
        # Initialize output by probing first model
        sample_out = self.models[0](inputs[:1])
        output = torch.zeros(
            inputs.shape[0], sample_out.shape[1],
            device=inputs.device, dtype=inputs.dtype
        )
        
        # Process each window's points
        for i, model in enumerate(self.models):
            mask = (window_idx == i)
            if mask.sum() > 0:
                output[mask] = model(inputs[mask])
        
        return output
    
    def get_layer_names(self) -> List[str]:
        """Get layer names from the first model (for NCC compatibility)."""
        if len(self.models) > 0 and hasattr(self.models[0], 'get_layer_names'):
            return self.models[0].get_layer_names()
        return []
    
    def get_window_model(self, window_idx: int) -> nn.Module:
        """Get the model for a specific window."""
        return self.models[window_idx]
    
    def get_window_info(self, window_idx: int) -> dict:
        """Get info about a specific window."""
        return self.window_info[window_idx]
    
    def __repr__(self) -> str:
        lines = [f"TimeMarchingModel(num_windows={self.num_windows})"]
        for i, (info, model) in enumerate(zip(self.window_info, self.models)):
            t_range = f"[{info['t_start']:.4f}, {info['t_end']:.4f}]"
            model_type = type(model).__name__
            experts = model.num_experts if hasattr(model, 'num_experts') else '?'
            lines.append(f"  Window {i}: t in {t_range}, M={info['M']}, {model_type}({experts} experts)")
        return "\n".join(lines)
