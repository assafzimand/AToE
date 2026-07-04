"""Adaptive Expert PINN module.

This module implements wavelet-based adaptive refinement for PINNs,
dynamically spawning regional expert networks during training.
"""

from adaptive.indicators import HardIndicator, SoftIndicator, RegionDescriptor
from adaptive.region_detector import RegionDetector
from adaptive.visualization import plot_expert_regions, plot_expert_regions_comparison

__all__ = [
    'HardIndicator',
    'SoftIndicator', 
    'RegionDescriptor',
    'RegionDetector',
    'plot_expert_regions',
    'plot_expert_regions_comparison',
]
