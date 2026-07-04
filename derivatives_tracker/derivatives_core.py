"""Core derivatives computation via autograd and probes."""

import torch
from typing import Dict, Tuple
import numpy as np


def compute_ground_truth_derivatives(
    x: torch.Tensor,
    t: torch.Tensor,
    config: Dict
) -> Dict[str, np.ndarray]:
    """
    Compute ground truth derivatives using analytical solution and autograd.
    
    This computes h_gt and derivatives from the exact analytical solution.
    
    Args:
        x: Spatial coordinates (N, 1) - will be set to requires_grad
        t: Temporal coordinates (N, 1) - will be set to requires_grad
        config: Configuration dict (to get interpolator and problem name)
        
    Returns:
        Dictionary with h_gt and derivatives as numpy arrays
    """
    import importlib
    
    # Dynamic import of solver based on problem
    problem_name = config.get('problem', 'schrodinger')
    solver_module = importlib.import_module(f'solvers.{problem_name}_solver')
    
    # Get coordinates as numpy (detach first)
    x_np = x.detach().cpu().numpy()  # Keep shape for 2D spatial
    t_np = t.detach().cpu().numpy().flatten()
    
    # Get problem config
    output_dim = config[problem_name].get('output_dim', 2)
    
    result = {}
    
    # For problems with interpolators (schrodinger, burgers1d), get interpolator
    # For problems with analytical solutions (wave1d, burgers2d), skip interpolator
    interpolator = None
    h_values = None
    if problem_name in ['schrodinger', 'burgers1d']:
        interpolator = solver_module._get_interpolator(config)
        x_flat = x_np.flatten() if x_np.ndim > 1 else x_np
        h_values = interpolator(x_flat, t_np)  # (N,) - could be complex or real
    
    if problem_name == 'schrodinger':
        # Complex-valued solution
        x_flat = x_np.flatten()
        h_complex = h_values  # (N,) complex
        
        # Finite difference step sizes - must be appropriate for grid resolution
        # Schrodinger uses 512x201 grid, so dx~0.01, dt~0.005
        eps_x = 1e-3
        eps_t = 1e-3
        
        # Compute h_t using finite differences
        h_t_plus = interpolator(x_flat, t_np + eps_t)
        h_t_minus = interpolator(x_flat, t_np - eps_t)
        h_t_complex = (h_t_plus - h_t_minus) / (2 * eps_t)
        
        # Compute h_xx using finite differences
        h_x_plus = interpolator(x_flat + eps_x, t_np)
        h_x_minus = interpolator(x_flat - eps_x, t_np)
        h_xx_complex = (h_x_plus - 2*h_complex + h_x_minus) / (eps_x**2)
        
        # Convert to (N, 2) format [real, imag]
        result['h_gt'] = np.column_stack([h_complex.real, h_complex.imag])
        result['h_t_gt'] = np.column_stack([h_t_complex.real, h_t_complex.imag])
        result['h_xx_gt'] = np.column_stack([h_xx_complex.real, h_xx_complex.imag])
        
    elif problem_name == 'wave1d':
        # Real-valued solution - use analytical derivatives directly
        from solvers.wave1d_solver import (
            analytical_solution,
            analytical_derivative_t,
            analytical_derivative_tt,
            analytical_derivative_x,
            analytical_derivative_xx
        )
        
        x_flat = x_np.flatten()
        
        # Compute all derivatives analytically (no boundary issues!)
        h_real = analytical_solution(x_flat, t_np)  # (N,) real
        h_t_real = analytical_derivative_t(x_flat, t_np)  # (N,) real
        h_tt_real = analytical_derivative_tt(x_flat, t_np)  # (N,) real
        h_x_real = analytical_derivative_x(x_flat, t_np)  # (N,) real
        h_xx_real = analytical_derivative_xx(x_flat, t_np)  # (N,) real
        
        # Convert to (N, 1) format for output_dim=1
        result['h_gt'] = h_real.reshape(-1, 1)
        result['h_t_gt'] = h_t_real.reshape(-1, 1)
        result['h_tt_gt'] = h_tt_real.reshape(-1, 1)
        result['h_x_gt'] = h_x_real.reshape(-1, 1)
        result['h_xx_gt'] = h_xx_real.reshape(-1, 1)
    
    elif problem_name == 'burgers1d':
        # Real-valued solution - use finite differences on interpolator
        x_flat = x_np.flatten()
        h_real = h_values  # (N,) real
        
        # Finite difference step sizes - must be appropriate for grid resolution
        # Burgers uses 256x201 grid, so dx~0.0078, dt~0.005
        # Using eps ~ grid spacing / 2 for good accuracy without precision issues
        eps_x = 1e-3
        eps_t = 1e-3
        
        # Get domain bounds for boundary handling
        t_min, t_max = config[problem_name]['temporal_domain']
        
        # Compute h_t using finite differences with boundary handling
        # Use one-sided differences at temporal boundaries to avoid fill_value issues
        h_t_real = np.zeros_like(h_real)
        
        # Masks for different regions
        near_t_min = t_np < t_min + eps_t
        near_t_max = t_np > t_max - eps_t
        interior = ~near_t_min & ~near_t_max
        
        # Central difference for interior points
        if np.any(interior):
            h_t_plus_int = interpolator(x_flat[interior], t_np[interior] + eps_t)
            h_t_minus_int = interpolator(x_flat[interior], t_np[interior] - eps_t)
            h_t_real[interior] = (h_t_plus_int - h_t_minus_int) / (2 * eps_t)
        
        # Forward difference near t_min
        if np.any(near_t_min):
            h_0 = interpolator(x_flat[near_t_min], t_np[near_t_min])
            h_1 = interpolator(x_flat[near_t_min], t_np[near_t_min] + eps_t)
            h_2 = interpolator(x_flat[near_t_min], t_np[near_t_min] + 2*eps_t)
            # Second-order forward difference: (-3h0 + 4h1 - h2) / (2*eps)
            h_t_real[near_t_min] = (-3*h_0 + 4*h_1 - h_2) / (2 * eps_t)
        
        # Backward difference near t_max
        if np.any(near_t_max):
            h_0 = interpolator(x_flat[near_t_max], t_np[near_t_max])
            h_m1 = interpolator(x_flat[near_t_max], t_np[near_t_max] - eps_t)
            h_m2 = interpolator(x_flat[near_t_max], t_np[near_t_max] - 2*eps_t)
            # Second-order backward difference: (3h0 - 4h_{-1} + h_{-2}) / (2*eps)
            h_t_real[near_t_max] = (3*h_0 - 4*h_m1 + h_m2) / (2 * eps_t)
        
        # Compute h_x using finite differences
        h_x_plus = interpolator(x_flat + eps_x, t_np)
        h_x_minus = interpolator(x_flat - eps_x, t_np)
        h_x_real = (h_x_plus - h_x_minus) / (2 * eps_x)
        
        # Compute h_xx using finite differences
        h_xx_real = (h_x_plus - 2*h_real + h_x_minus) / (eps_x**2)
        
        # Convert to (N, 1) format for output_dim=1
        result['h_gt'] = h_real.reshape(-1, 1)
        result['h_t_gt'] = h_t_real.reshape(-1, 1)
        result['h_x_gt'] = h_x_real.reshape(-1, 1)
        result['h_xx_gt'] = h_xx_real.reshape(-1, 1)
    
    elif problem_name == 'burgers2d':
        # Real-valued solution - use analytical derivatives directly
        from solvers.burgers2d_solver import (
            analytical_solution,
            analytical_h_t,
            analytical_h_x0,
            analytical_h_x1,
            analytical_h_x0x0,
            analytical_h_x1x1
        )
        
        # For 2D spatial, x has shape (N, 2)
        x0_np = x_np[:, 0] if x_np.ndim > 1 else x_np
        x1_np = x_np[:, 1] if x_np.ndim > 1 else np.zeros_like(x_np)
        
        # Compute all derivatives analytically
        h_real = analytical_solution(x0_np, x1_np, t_np)  # (N,) real
        h_t_real = analytical_h_t(x0_np, x1_np, t_np)  # (N,) real
        h_x0_real = analytical_h_x0(x0_np, x1_np, t_np)  # (N,) real
        h_x1_real = analytical_h_x1(x0_np, x1_np, t_np)  # (N,) real
        h_x0x0_real = analytical_h_x0x0(x0_np, x1_np, t_np)  # (N,) real
        h_x1x1_real = analytical_h_x1x1(x0_np, x1_np, t_np)  # (N,) real
        
        # Convert to (N, 1) format for output_dim=1
        result['h_gt'] = h_real.reshape(-1, 1)
        result['h_t_gt'] = h_t_real.reshape(-1, 1)
        result['h_x0_gt'] = h_x0_real.reshape(-1, 1)
        result['h_x1_gt'] = h_x1_real.reshape(-1, 1)
        result['h_x0x0_gt'] = h_x0x0_real.reshape(-1, 1)
        result['h_x1x1_gt'] = h_x1x1_real.reshape(-1, 1)
        
    return result


def apply_linear_probe(embeddings: torch.Tensor, probe: torch.nn.Linear) -> torch.Tensor:
    """
    Apply a trained linear probe to embeddings.
    
    Args:
        embeddings: Layer activations (N, hidden_dim) WITH gradients
        probe: Trained linear probe (torch.nn.Linear)
        
    Returns:
        Predictions (N, 2) - treated as (u, v) for complex h = u + iv
    """
    return probe(embeddings)


def compute_layer_derivatives_via_probe(
    model: torch.nn.Module,
    layer_name: str,
    probe: torch.nn.Linear,
    x: torch.Tensor,
    t: torch.Tensor,
    config: Dict
) -> Dict[str, torch.Tensor]:
    """
    Compute derivatives of probe output w.r.t. inputs using autograd.
    
    This computes derivatives based on the problem requirements:
    - h = P_i(layer_i(...layer_1(x,t)))  where P_i is the probe
    - h_t, h_tt, h_x, h_xx as needed by the problem
    
    Args:
        model: Neural network model
        layer_name: Which layer to analyze (e.g., 'layer_1')
        probe: Trained linear probe for this layer
        x: Spatial coordinates (N, 1) - MUST have requires_grad=True
        t: Temporal coordinates (N, 1) - MUST have requires_grad=True
        config: Configuration dict (contains problem and output_dim)
        
    Returns:
        Dictionary with:
            'h': Probe predictions (N, output_dim)
            'h_t': Temporal derivative (N, output_dim) [if needed]
            'h_tt': Second temporal derivative (N, output_dim) [if needed]
            'h_x': Spatial derivative (N, output_dim) [if needed]
            'h_xx': Second spatial derivative (N, output_dim) [if needed]
    """
    from .residuals import get_required_derivatives
    
    # Ensure gradients can flow
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    
    # Register hooks with keep_gradients=True
    handles = model.register_ncc_hooks([layer_name], keep_gradients=True)
    
    # Forward pass through model to get layer activations (with gradients)
    inputs = torch.cat([x, t], dim=1)
    _ = model(inputs)
    
    # Get embeddings from hook (they have gradients!)
    embeddings = model.activations[layer_name]
    
    # Apply probe to get h
    h = apply_linear_probe(embeddings, probe)  # (N, output_dim)
    
    output_dim = h.shape[1]
    
    # Get required derivatives for this problem
    problem_name = config.get('problem', 'schrodinger')
    required_derivs = get_required_derivatives(problem_name)
    
    results = {'h': h}
    
    # Compute first temporal derivative if needed
    if 'h_t' in required_derivs or 'h_tt' in required_derivs:
        h_t = torch.zeros_like(h)
        for i in range(output_dim):
            grad_outputs = torch.ones_like(h[:, i])
            h_t[:, i:i+1] = torch.autograd.grad(
                outputs=h[:, i],
                inputs=t,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]
        results['h_t'] = h_t
    
    # Compute first spatial derivative if needed (1D spatial)
    if 'h_x' in required_derivs or 'h_xx' in required_derivs:
        h_x = torch.zeros_like(h)
        for i in range(output_dim):
            grad_outputs = torch.ones_like(h[:, i])
            h_x[:, i:i+1] = torch.autograd.grad(
                outputs=h[:, i],
                inputs=x,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]
        results['h_x'] = h_x
    
    # Compute spatial derivatives for 2D spatial problems
    if 'h_x0' in required_derivs or 'h_x0x0' in required_derivs:
        # x has shape (N, 2) for 2D spatial
        h_x_all = torch.zeros(h.shape[0], 2, device=h.device)
        for i in range(output_dim):
            grad_outputs = torch.ones_like(h[:, i])
            grad_x = torch.autograd.grad(
                outputs=h[:, i],
                inputs=x,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]  # (N, 2)
            h_x_all += grad_x
        h_x_all = h_x_all / max(output_dim, 1)  # Average if multi-output
        
        results['h_x0'] = h_x_all[:, 0:1]  # (N, 1)
        results['h_x1'] = h_x_all[:, 1:2]  # (N, 1)
    
    # Check if we need h_tt (affects retain_graph for h_xx)
    need_h_tt = 'h_tt' in required_derivs and 'h_t' in results
    
    # Compute second spatial derivative if needed (1D spatial)
    if 'h_xx' in required_derivs and 'h_x' in results:
        h_x = results['h_x']
        h_xx = torch.zeros_like(h)
        for i in range(output_dim):
            grad_outputs = torch.ones_like(h_x[:, i])
            # Keep graph if not last OR if we need h_tt later
            should_retain = (i < output_dim - 1) or need_h_tt
            h_xx[:, i:i+1] = torch.autograd.grad(
                outputs=h_x[:, i],
                inputs=x,
                grad_outputs=grad_outputs,
                retain_graph=should_retain,
                create_graph=False
            )[0]
        results['h_xx'] = h_xx
    
    # Compute second spatial derivatives for 2D spatial problems
    if 'h_x0x0' in required_derivs and 'h_x0' in results:
        h_x0 = results['h_x0']  # (N, 1)
        h_x1 = results['h_x1']  # (N, 1)
        
        # Compute d(h_x0)/dx0 and d(h_x1)/dx1
        grad_outputs = torch.ones_like(h_x0.squeeze())
        
        # h_x0x0
        h_x0_grad = torch.autograd.grad(
            outputs=h_x0.squeeze(),
            inputs=x,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False
        )[0]  # (N, 2)
        results['h_x0x0'] = h_x0_grad[:, 0:1]  # d(h_x0)/dx0, (N, 1)
        
        # h_x1x1
        grad_outputs = torch.ones_like(h_x1.squeeze())
        h_x1_grad = torch.autograd.grad(
            outputs=h_x1.squeeze(),
            inputs=x,
            grad_outputs=grad_outputs,
            retain_graph=need_h_tt,
            create_graph=False
        )[0]  # (N, 2)
        results['h_x1x1'] = h_x1_grad[:, 1:2]  # d(h_x1)/dx1, (N, 1)
    
    # Compute second temporal derivative if needed (for wave equation)
    if need_h_tt:
        h_t = results['h_t']
        h_tt = torch.zeros_like(h)
        for i in range(output_dim):
            grad_outputs = torch.ones_like(h_t[:, i])
            h_tt[:, i:i+1] = torch.autograd.grad(
                outputs=h_t[:, i],
                inputs=t,
                grad_outputs=grad_outputs,
                retain_graph=(i < output_dim - 1),
                create_graph=False
            )[0]
        results['h_tt'] = h_tt
    
    # Remove hooks and detach all results
    model.remove_hooks()
    
    # Detach all tensors in results
    for key in results:
        results[key] = results[key].detach()
    
    return results


def track_all_layers(
    model: torch.nn.Module,
    probes_dict: Dict[str, torch.nn.Linear],
    x: torch.Tensor,
    t: torch.Tensor,
    device: torch.device,
    config: Dict
) -> Dict[str, Dict]:
    """
    Track derivatives and residuals for all layers.
    
    Args:
        model: Trained neural network
        probes_dict: Dictionary mapping layer_name -> trained probe
        x: Spatial coordinates (N, 1)
        t: Temporal coordinates (N, 1)
        device: Device to run on
        config: Configuration dict (contains problem name for residual computation)
        
    Returns:
        Dictionary mapping layer_name -> {
            'h': (N, output_dim),
            'h_t': (N, output_dim),
            'h_x': (N, output_dim),
            'h_xx': (N, output_dim),
            'residual': (N, output_dim),
            'norms': {...}
        }
    """
    from .residuals import get_residual_module, get_required_derivatives
    
    model.eval()
    x = x.to(device)
    t = t.to(device)
    
    # Get problem-specific residual computer
    problem_name = config.get('problem', 'schrodinger')
    residual_module = get_residual_module(problem_name)
    compute_residual_fn = residual_module.compute_residual_terms
    required_derivs = get_required_derivatives(problem_name)
    
    results = {}
    
    for layer_name, probe in probes_dict.items():
        print(f"  Processing {layer_name}...")
        
        # Compute derivatives for this layer
        derivatives = compute_layer_derivatives_via_probe(
            model=model,
            layer_name=layer_name,
            probe=probe,
            x=x,
            t=t,
            config=config
        )
        
        # Extract derivatives (all problems have these)
        h = derivatives['h']
        h_x = derivatives.get('h_x')
        h_xx = derivatives.get('h_xx')
        h_t = derivatives.get('h_t')
        h_tt = derivatives.get('h_tt')  # Only for wave
        # 2D spatial derivatives
        h_x0 = derivatives.get('h_x0')
        h_x1 = derivatives.get('h_x1')
        h_x0x0 = derivatives.get('h_x0x0')
        h_x1x1 = derivatives.get('h_x1x1')
        
        # Compute residual terms using problem-specific function
        # Pass all available derivatives; function will use what it needs
        residual_terms = compute_residual_fn(
            h=h,
            h_t=h_t,
            h_tt=h_tt,
            h_x=h_x,
            h_xx=h_xx,
            h_x0=h_x0,
            h_x1=h_x1,
            h_x0x0=h_x0x0,
            h_x1x1=h_x1x1
        )
        
        # Compute L2 / L_inf norms across samples (mean per-sample norms)
        # For each term, compute ||term|| per sample, then take mean
        h_norms = torch.norm(h, dim=1)  # (N,)
        residual_norms = torch.norm(residual_terms['residual'], dim=1)
        residual_inf = torch.max(torch.abs(residual_terms['residual']), dim=1).values
        
        norms = {
            'h_norm': h_norms.mean().item(),
            'residual_norm': residual_norms.mean().item(),
            'residual_inf_norm': residual_inf.mean().item()
        }
        
        # Add optional derivative norms if they exist
        if h_t is not None:
            norms['h_t_norm'] = torch.norm(h_t, dim=1).mean().item()
        if h_x is not None:
            norms['h_x_norm'] = torch.norm(h_x, dim=1).mean().item()
        if h_xx is not None:
            norms['h_xx_norm'] = torch.norm(h_xx, dim=1).mean().item()
        if h_tt is not None:
            norms['h_tt_norm'] = torch.norm(h_tt, dim=1).mean().item()
        # 2D spatial derivative norms
        if h_x0 is not None:
            norms['h_x0_norm'] = torch.norm(h_x0, dim=1).mean().item()
        if h_x1 is not None:
            norms['h_x1_norm'] = torch.norm(h_x1, dim=1).mean().item()
        if h_x0x0 is not None:
            norms['h_x0x0_norm'] = torch.norm(h_x0x0, dim=1).mean().item()
        if h_x1x1 is not None:
            norms['h_x1x1_norm'] = torch.norm(h_x1x1, dim=1).mean().item()
        
        # Get problem-specific term metadata for dynamic norm computation
        term_metadata = {}
        if hasattr(residual_module, 'get_term_metadata'):
            term_metadata = residual_module.get_term_metadata()
        
        # Add norms for problem-specific terms (dynamically)
        for term_key in term_metadata:
            if term_key in residual_terms:
                term_tensor = residual_terms[term_key]
                if term_tensor.dim() == 1:
                    norms[f'{term_key}_norm'] = torch.abs(term_tensor).mean().item()
                else:
                    norms[f'{term_key}_norm'] = torch.norm(term_tensor, dim=1).mean().item()
        
        # Store all results for this layer
        layer_results = {
            'h': h.cpu().numpy(),
            'residual': residual_terms['residual'].cpu().numpy(),
            'norms': norms
        }
        
        # Add optional derivatives
        if h_t is not None:
            layer_results['h_t'] = h_t.cpu().numpy()
        if h_x is not None:
            layer_results['h_x'] = h_x.cpu().numpy()
        if h_xx is not None:
            layer_results['h_xx'] = h_xx.cpu().numpy()
        if h_tt is not None:
            layer_results['h_tt'] = h_tt.cpu().numpy()
        # 2D spatial derivatives
        if h_x0 is not None:
            layer_results['h_x0'] = h_x0.cpu().numpy()
        if h_x1 is not None:
            layer_results['h_x1'] = h_x1.cpu().numpy()
        if h_x0x0 is not None:
            layer_results['h_x0x0'] = h_x0x0.cpu().numpy()
        if h_x1x1 is not None:
            layer_results['h_x1x1'] = h_x1x1.cpu().numpy()
        
        # Add problem-specific terms (dynamically from term_metadata)
        for term_key in term_metadata:
            if term_key in residual_terms:
                term_tensor = residual_terms[term_key]
                if isinstance(term_tensor, torch.Tensor):
                    layer_results[term_key] = term_tensor.cpu().numpy()
        
        # Also add h_magnitude_sq if present (for compatibility)
        if 'h_magnitude_sq' in residual_terms:
            layer_results['h_magnitude_sq'] = residual_terms['h_magnitude_sq'].cpu().numpy()
        
        results[layer_name] = layer_results
        
        print(f"    Residual L2 norm: {norms['residual_norm']:.6f} | "
              f"L_inf: {norms['residual_inf_norm']:.6f}")
    
    return results

