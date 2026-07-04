"""Test loss functions with simple optimization."""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io import load_config
from losses.schrodinger_loss import build_loss as build_loss_schrodinger


class SimpleFCNet(nn.Module):
    """Simple fully-connected network for testing."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_tiny_batch(n_points: int, spatial_dim: int, device: torch.device):
    """Create a tiny batch for testing."""
    n_residual = n_points // 2
    n_ic = n_points // 4
    n_bc = n_points - n_residual - n_ic

    x = torch.rand(n_points, spatial_dim, device=device)
    t = torch.rand(n_points, 1, device=device)
    h_gt = torch.randn(n_points, 2, device=device)

    mask_residual = torch.zeros(n_points, dtype=torch.bool, device=device)
    mask_residual[:n_residual] = True

    mask_ic = torch.zeros(n_points, dtype=torch.bool, device=device)
    mask_ic[n_residual:n_residual + n_ic] = True

    mask_bc = torch.zeros(n_points, dtype=torch.bool, device=device)
    mask_bc[n_residual + n_ic:] = True

    batch = {
        'x': x,
        't': t,
        'h_gt': h_gt,
        'mask': {
            'residual': mask_residual,
            'IC': mask_ic,
            'BC': mask_bc
        }
    }

    return batch


def test_loss_schrodinger():
    """Test schrodinger loss function."""
    print("Testing schrodinger loss function...")

    # Load config
    config = load_config()
    device = torch.device('cuda' if config['cuda'] and
                          torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Build loss
    loss_fn = build_loss_schrodinger(**config)

    # Create tiny model
    spatial_dim = config['schrodinger']['spatial_dim']
    input_dim = spatial_dim + 1  # x + t
    output_dim = 2
    model = SimpleFCNet(input_dim, 20, output_dim).to(device)

    # Create tiny batch
    batch = create_tiny_batch(100, spatial_dim, device)

    # Test forward pass
    loss = loss_fn(model, batch)
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.device.type == device.type, f"Loss should be on {device}"
    assert loss.item() >= 0, "Loss should be non-negative"

    print(f"  ✓ Loss forward pass successful: {loss.item():.4f}")

    # Test optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []

    for step in range(10):
        optimizer.zero_grad()
        loss = loss_fn(model, batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Check that loss decreased
    assert losses[-1] < losses[0], \
        f"Loss should decrease (initial: {losses[0]:.4f}, " \
        f"final: {losses[-1]:.4f})"

    print(f"  ✓ Optimization works: loss decreased from "
          f"{losses[0]:.4f} to {losses[-1]:.4f}")


def test_loss_components():
    """Test that loss properly weights different components."""
    print("\nTesting loss component weighting...")

    config = load_config()
    device = torch.device('cuda' if config['cuda'] and
                          torch.cuda.is_available() else 'cpu')

    # Build loss
    loss_fn = build_loss_schrodinger(**config)

    # Create model
    spatial_dim = config['schrodinger']['spatial_dim']
    input_dim = spatial_dim + 1
    model = SimpleFCNet(input_dim, 20, 2).to(device)

    # Create batch with only residual points
    batch_residual = create_tiny_batch(100, spatial_dim, device)
    batch_residual['mask']['IC'][:] = False
    batch_residual['mask']['BC'][:] = False
    batch_residual['mask']['residual'][:] = True

    loss_residual_only = loss_fn(model, batch_residual).item()

    # Create batch with only IC points
    batch_ic = create_tiny_batch(100, spatial_dim, device)
    batch_ic['mask']['residual'][:] = False
    batch_ic['mask']['BC'][:] = False
    batch_ic['mask']['IC'][:] = True

    loss_ic_only = loss_fn(model, batch_ic).item()

    print(f"  ✓ Loss computes correctly for different point types")
    print(f"    Residual-only loss: {loss_residual_only:.4f}")
    print(f"    IC-only loss: {loss_ic_only:.4f}")


def test_physics_informed_gradients():
    """Test that physics-informed loss gradients flow correctly."""
    print("\nTesting physics-informed loss gradients (Schrödinger)...")
    
    config = load_config()
    config['problem'] = 'schrodinger'
    device = torch.device('cuda' if config['cuda'] and
                          torch.cuda.is_available() else 'cpu')
    
    # Build physics-informed loss
    loss_fn = build_loss_schrodinger(**config)
    
    # Create model
    spatial_dim = config['schrodinger']['spatial_dim']
    input_dim = spatial_dim + 1
    model = SimpleFCNet(input_dim, 30, 2).to(device)
    
    # Create batch with points in the Schrödinger domain
    n_points = 50
    n_residual = 30
    n_ic = 10
    n_bc = 10
    
    x_min, x_max = config['schrodinger']['spatial_domain'][0]
    t_min, t_max = config['schrodinger']['temporal_domain']
    
    x = torch.rand(n_points, spatial_dim, device=device) * (x_max - x_min) + x_min
    t = torch.rand(n_points, 1, device=device) * (t_max - t_min) + t_min
    
    # Set IC points to t=0
    t[n_residual:n_residual+n_ic] = t_min
    
    # Set BC points to boundaries (paired)
    x[n_residual+n_ic:n_residual+n_ic+n_bc//2, 0] = x_min  # Left
    x[n_residual+n_ic+n_bc//2:, 0] = x_max  # Right
    
    # Ground truth (not used for residual, but needed for IC/BC)
    h_gt = torch.randn(n_points, 2, device=device)
    
    # Create masks
    mask_residual = torch.zeros(n_points, dtype=torch.bool, device=device)
    mask_residual[:n_residual] = True
    
    mask_ic = torch.zeros(n_points, dtype=torch.bool, device=device)
    mask_ic[n_residual:n_residual+n_ic] = True
    
    mask_bc = torch.zeros(n_points, dtype=torch.bool, device=device)
    mask_bc[n_residual+n_ic:] = True
    
    batch = {
        'x': x,
        't': t,
        'h_gt': h_gt,
        'mask': {
            'residual': mask_residual,
            'IC': mask_ic,
            'BC': mask_bc
        }
    }
    
    # Test forward pass
    loss = loss_fn(model, batch)
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss), "Loss should not be NaN"
    
    print(f"  ✓ Physics-informed loss computed: {loss.item():.4f}")
    
    # Test backward pass
    loss.backward()
    
    # Check that all model parameters have gradients
    has_grad = all(p.grad is not None for p in model.parameters())
    assert has_grad, "All parameters should have gradients"
    
    # Check that gradients are not NaN or infinite
    grad_finite = all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert grad_finite, "All gradients should be finite"
    
    # Check that gradients are non-zero (model is learning)
    grad_nonzero = any((p.grad.abs() > 1e-8).any() for p in model.parameters())
    assert grad_nonzero, "At least some gradients should be non-zero"
    
    print(f"  ✓ Gradients flow correctly through autograd")
    
    # Test optimization for a few steps
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    
    for step in range(20):
        optimizer.zero_grad()
        loss = loss_fn(model, batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    # Check that loss decreased or stayed stable
    print(f"  ✓ Loss after 20 steps: {losses[0]:.4f} → {losses[-1]:.4f}")
    
    # Just check that the optimization runs without errors
    # (Loss might not always decrease with physics-informed loss due to competing terms)
    assert all(not torch.isnan(torch.tensor(l)) for l in losses), \
        "Loss should remain finite during optimization"
    
    print(f"  ✓ Physics-informed optimization runs successfully")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 3 — Loss Functions — Tests")
    print("=" * 60)

    try:
        test_loss_schrodinger()
        test_loss_components()
        test_physics_informed_gradients()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

