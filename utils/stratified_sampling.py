"""Stratified sampling utilities for NCC dataset generation."""

import torch
from typing import Dict


def stratify_by_bins(
    large_data: Dict,
    bins: int,
    output_dim: int,
    target_size: int,
    device: torch.device
) -> Dict:
    """
    Uniform sampling - equal samples from each non-empty class.
    
    Algorithm:
    1. Bin all samples by ground truth (u, v, ...) -> assign to bins^output_dim classes
    2. Identify non-empty classes
    3. Sample equal number of samples from each non-empty class
    4. Return dataset with uniform class distribution
    
    Args:
        large_data: Large dataset dict with 'x', 't', 'h_gt', 'mask'
        bins: Number of bins per output dimension
        output_dim: Output dimensionality (e.g., 2 for complex field u+iv)
        target_size: Target total number of samples
        device: Device for computation
        
    Returns:
        Uniformly sampled dataset dict with same structure as input
    """
    # Import binning function from NCC
    from ncc.ncc_core import create_class_labels_from_regression
    
    print(f"  Stratifying {len(large_data['x'])} samples...")
    
    # Bin all samples (empty classes already filtered out)
    class_labels, class_map, bin_info = create_class_labels_from_regression(
        large_data['h_gt'], bins, device
    )
    num_classes = bin_info['num_classes']  # Number of non-empty classes after filtering
    
    print(f"  Using {num_classes} non-empty classes (out of {bin_info['original_num_classes']} total)")
    
    # Uniform sampling: equal samples per class (based on all non-empty classes)
    samples_per_class_value = target_size // num_classes
    
    # Filter classes that don't have 80% of required samples
    # Note: We keep samples_per_class fixed - total samples may be less than target
    min_samples_threshold = int(samples_per_class_value * 0.05)
    valid_classes = []
    
    for c in range(num_classes):
        class_mask = (class_labels == c)
        class_indices = torch.where(class_mask)[0]
        
        if len(class_indices) >= min_samples_threshold:
            valid_classes.append(c)
    
    num_valid_classes = len(valid_classes)
    
    print(f"  Filtered {num_classes - num_valid_classes} sparse classes (< {min_samples_threshold} samples)")
    print(f"  Using {num_valid_classes} classes with {samples_per_class_value} samples each")
    
    # Sample indices only from valid classes
    selected_indices = []
    
    for c in valid_classes:
        class_mask = (class_labels == c)
        class_indices = torch.where(class_mask)[0]
        
        # Sample from this valid class
        n_to_sample = samples_per_class_value
        
        if len(class_indices) >= n_to_sample:
            # Sample without replacement
            sampled = class_indices[torch.randperm(len(class_indices), device=device)[:n_to_sample]]
        else:
            # Sample with replacement if needed
            sampled = class_indices[torch.randint(0, len(class_indices), (n_to_sample,), device=device)]
        
        selected_indices.append(sampled)
    
    # Concatenate all selected indices
    selected_indices = torch.cat(selected_indices)
    
    print(f"  Uniform sampling complete:")
    print(f"    - Classes sampled: {num_valid_classes} (filtered from {num_classes})")
    print(f"    - Total samples: {len(selected_indices)}")
    print(f"    - Target size: {target_size}")
    print(f"    - Samples per class: {samples_per_class_value}")
    
    # Extract stratified subset
    return {
        'x': large_data['x'][selected_indices],
        't': large_data['t'][selected_indices],
        'h_gt': large_data['h_gt'][selected_indices],
        'mask': {
            'residual': large_data['mask']['residual'][selected_indices],
            'IC': large_data['mask']['IC'][selected_indices],
            'BC': large_data['mask']['BC'][selected_indices]
        }
    }

