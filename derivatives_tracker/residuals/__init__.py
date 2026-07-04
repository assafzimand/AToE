"""
Residual computation registry for different PDE problems.

Each problem has its own residual module that defines:
- compute_residual_terms(): Calculate the PDE residual
- get_relevant_derivatives(): List which derivatives to plot/analyze
"""

from typing import Any, List


def get_residual_module(problem_name: str) -> Any:
    """
    Get the residual computation module for a given problem.
    
    Args:
        problem_name: Name of the problem (e.g., 'schrodinger', 'wave1d')
        
    Returns:
        Module containing compute_residual_terms() and get_relevant_derivatives()
        
    Example:
        >>> residual_module = get_residual_module('wave1d')
        >>> relevant_derivs = residual_module.get_relevant_derivatives()
        >>> # Returns: ['h_tt', 'h_xx']
    """
    if problem_name == 'schrodinger':
        from . import schrodinger_residual
        return schrodinger_residual
    elif problem_name == 'wave1d':
        from . import wave1d_residual
        return wave1d_residual
    elif problem_name == 'burgers1d':
        from . import burgers1d_residual
        return burgers1d_residual
    elif problem_name == 'burgers2d':
        from . import burgers2d_residual
        return burgers2d_residual
    else:
        raise ValueError(f"Unknown problem: {problem_name}. "
                       f"Available problems: schrodinger, wave1d, burgers1d, burgers2d")


def get_required_derivatives(problem_name: str) -> List[str]:
    """
    Get the list of required derivatives for a given problem.
    
    This is a convenience function that loads the residual module
    and returns its relevant derivatives list.
    
    Args:
        problem_name: Name of the problem (e.g., 'schrodinger', 'wave1d')
        
    Returns:
        List of derivative names needed for residual computation
        
    Example:
        >>> get_required_derivatives('wave1d')
        ['h_tt', 'h_xx']
        >>> get_required_derivatives('schrodinger')
        ['h', 'h_t', 'h_xx', 'nonlinear']
    """
    residual_module = get_residual_module(problem_name)
    return residual_module.get_relevant_derivatives()


__all__ = ['get_residual_module', 'get_required_derivatives']
