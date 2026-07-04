"""Training utility functions."""

import torch
from typing import Dict


def compute_relative_l2_error(
    h_pred: torch.Tensor,
    h_gt: torch.Tensor
) -> torch.Tensor:
    """
    Compute relative L2 error.

    Args:
        h_pred: Predicted values (N, output_dim)
        h_gt: Ground truth values (N, output_dim)

    Returns:
        Scalar relative L2 error: ||h_pred - h_gt||_2 / ||h_gt||_2
    """
    diff = h_pred - h_gt
    numerator = torch.norm(diff, p=2)
    denominator = torch.norm(h_gt, p=2) + 1e-10  # Avoid division by zero

    return numerator / denominator


def compute_infinity_norm_error(
    h_pred: torch.Tensor,
    h_gt: torch.Tensor
) -> torch.Tensor:
    """
    Compute infinity norm (max absolute) error.

    Args:
        h_pred: Predicted values (N, output_dim)
        h_gt: Ground truth values (N, output_dim)

    Returns:
        Scalar infinity norm error: max(|h_pred - h_gt|)
    """
    diff = h_pred - h_gt
    return torch.max(torch.abs(diff))
