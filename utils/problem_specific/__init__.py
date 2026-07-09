"""
Problem-specific utilities loader.

Dynamically imports problem-specific visualization and evaluation utilities.
"""


def get_visualization_module(problem_name: str):
    """
    Dynamically import visualization functions for the problem.

    Args:
        problem_name: Name of the problem (e.g., 'schrodinger', 'burgers1d')

    Returns:
        Tuple (visualize_dataset, visualize_evaluation). Problems without a
        dedicated viz module get a no-op dataset visualizer and the generic
        prediction/error-map evaluator.
    """
    if problem_name == 'schrodinger':
        from .schrodinger_viz import visualize_dataset, visualize_evaluation
        return (visualize_dataset, visualize_evaluation)
    elif problem_name == 'burgers1d':
        from .burgers1d_viz import visualize_dataset, visualize_evaluation
        return (visualize_dataset, visualize_evaluation)
    else:
        # For problems without a dedicated viz module, use the generic evaluator.
        from .generic_viz import plot_predictions_and_error_maps

        def _generic_visualize_evaluation(model, save_dir, config):
            problem = config.get('problem', 'problem')
            plot_predictions_and_error_maps(
                model, save_dir, config,
                filename=f"pred_final_{problem}_relL2_{{relL2}}.png")

        def _noop(*args, **kwargs):
            pass

        return (_noop, _generic_visualize_evaluation)
