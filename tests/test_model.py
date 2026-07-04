"""Test FCNet model and hook functionality."""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io import load_config
from models.fc_model import FCNet


def test_model_construction():
    """Test model construction and architecture verification."""
    print("Testing model construction...")

    config = load_config()
    architecture = config['base_architecture']
    activation = config['activation']

    # Test successful construction
    model = FCNet(architecture, activation, config)

    print(f"  ✓ Model created successfully")
    print(f"  ✓ Architecture: {architecture}")
    print(f"  ✓ Activation: {activation}")
    print(f"  ✓ Layers: {model.get_layer_names()}")

    # Verify layer count
    expected_layers = len(architecture) - 1
    assert len(model.get_layer_names()) == expected_layers, \
        f"Expected {expected_layers} layers, got {len(model.get_layer_names())}"

    print(f"  ✓ Correct number of layers: {expected_layers}")


def test_model_forward():
    """Test forward pass with random inputs."""
    print("\nTesting forward pass...")

    config = load_config()
    device = torch.device('cuda' if config['cuda'] and
                          torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Build model
    architecture = config['base_architecture']
    activation = config['activation']
    model = FCNet(architecture, activation, config).to(device)

    # Create random input
    batch_size = 100
    input_dim = architecture[0]  # spatial_dim + 1
    output_dim = architecture[-1]

    x_input = torch.randn(batch_size, input_dim, device=device)

    # Forward pass
    output = model(x_input)

    # Check output shape
    assert output.shape == (batch_size, output_dim), \
        f"Expected output shape ({batch_size}, {output_dim}), " \
        f"got {output.shape}"

    # Check device
    assert output.device.type == device.type, \
        f"Output should be on {device}"

    # Check no NaN or Inf
    assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    print(f"  ✓ Forward pass successful")
    print(f"  ✓ Input shape: {x_input.shape}")
    print(f"  ✓ Output shape: {output.shape}")
    print(f"  ✓ Output device: {output.device}")


def test_model_hooks():
    """Test NCC hook registration and activation capture."""
    print("\nTesting NCC hooks...")

    config = load_config()
    device = torch.device('cuda' if config['cuda'] and
                          torch.cuda.is_available() else 'cpu')

    # Build model
    architecture = config['base_architecture']
    activation = config['activation']
    model = FCNet(architecture, activation, config).to(device)

    # Get layer names (all hidden layers)
    layer_names = model.get_layer_names()[:-1]  # Exclude output layer
    print(f"  Hooking layers: {layer_names}")

    # Register hooks
    handles = model.register_ncc_hooks(layer_names)

    assert len(handles) == len(layer_names), \
        "Number of handles should match number of layers"

    # Forward pass
    batch_size = 50
    input_dim = architecture[0]
    x_input = torch.randn(batch_size, input_dim, device=device)

    output = model(x_input)

    # Check activations were captured
    assert len(model.activations) == len(layer_names), \
        f"Expected {len(layer_names)} activations, " \
        f"got {len(model.activations)}"

    # Check activation shapes
    for i, layer_name in enumerate(layer_names):
        assert layer_name in model.activations, \
            f"Activation for {layer_name} not captured"

        activation_tensor = model.activations[layer_name]

        # Check shape
        expected_hidden_dim = architecture[i + 1]
        assert activation_tensor.shape == (batch_size, expected_hidden_dim), \
            f"Expected shape ({batch_size}, {expected_hidden_dim}), " \
            f"got {activation_tensor.shape}"

        # Check device
        assert activation_tensor.device.type == device.type, \
            f"Activation should be on {device}"

        print(f"  ✓ {layer_name}: shape {activation_tensor.shape}, "
              f"device {activation_tensor.device}")

    # Remove hooks
    model.remove_hooks()
    assert len(model.hook_handles) == 0, "Hooks should be removed"
    assert len(model.activations) == 0, "Activations should be cleared"

    print(f"  ✓ Hooks registered and removed successfully")


def test_architecture_verification():
    """Test that architecture verification catches mismatches."""
    print("\nTesting architecture verification...")

    config = load_config()

    # Get correct spatial dimension
    problem = config['problem']
    spatial_dim = config[problem]['spatial_dim']
    expected_input_dim = spatial_dim + 1

    # Test with wrong input dimension
    wrong_architecture = [expected_input_dim + 1, 50, 2]  # Wrong input dim

    try:
        model = FCNet(wrong_architecture, 'tanh', config)
        assert False, "Should have raised assertion error for wrong input dim"
    except AssertionError as e:
        print(f"  ✓ Correctly caught wrong input dimension")
        print(f"    Error message: {str(e)[:80]}...")

    # Test with correct dimension
    correct_architecture = [expected_input_dim, 50, 2]
    model = FCNet(correct_architecture, 'tanh', config)
    print(f"  ✓ Accepted correct input dimension: {expected_input_dim}")


def test_gradients_flow():
    """Test that gradients flow correctly through the model."""
    print("\nTesting gradient flow...")

    config = load_config()
    device = torch.device('cuda' if config['cuda'] and
                          torch.cuda.is_available() else 'cpu')

    # Build model
    architecture = config['base_architecture']
    activation = config['activation']
    model = FCNet(architecture, activation, config).to(device)

    # Forward pass
    batch_size = 32
    input_dim = architecture[0]
    x_input = torch.randn(batch_size, input_dim, device=device,
                          requires_grad=True)

    output = model(x_input)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients exist
    assert x_input.grad is not None, "Input gradients should exist"

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} should exist"
        assert torch.isfinite(param.grad).all(), \
            f"Gradient for {name} contains NaN or Inf"

    print(f"  ✓ Gradients flow correctly through all layers")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 4 — FC model + hooks — Tests")
    print("=" * 60)

    try:
        test_model_construction()
        test_model_forward()
        test_model_hooks()
        test_architecture_verification()
        test_gradients_flow()

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

