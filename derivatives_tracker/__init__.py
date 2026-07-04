"""Derivatives tracker module for layer-wise physics residual analysis."""

from .derivatives_core import (
    compute_layer_derivatives_via_probe,
    track_all_layers
)

from .derivatives_plotting import (
    plot_residual_summary,
    plot_term_magnitudes,
    plot_derivative_heatmaps,
    plot_residual_heatmaps,
    generate_all_derivative_plots
)

from .derivatives_runner import run_derivatives_tracker

__all__ = [
    'compute_layer_derivatives_via_probe',
    'track_all_layers',
    'plot_residual_summary',
    'plot_term_magnitudes',
    'plot_derivative_heatmaps',
    'plot_residual_heatmaps',
    'generate_all_derivative_plots',
    'run_derivatives_tracker'
]

