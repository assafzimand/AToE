"""
Test if ResNet can fit KS problem using supervised learning (MSE on ground truth).

This is a sanity check to see if the architecture is capable of representing
the solution when trained with direct supervision, without physics constraints.

Usage:
    python scripts/test_supervised_resnet.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import time

# Add parent directory to path to import from models/
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.network_factory import create_network


def _build_ks_pinn_config(
    fourier_enabled: bool = True,
    fourier_dim: int = 128,
    fourier_scale: float = 1.5,
    rwf: bool = True,
) -> dict:
    """Minimal config matching PINN (FF + RWF) so ResNetModel uses the same path."""
    return {
        'problem': 'ks',
        'ks': {
            'spatial_dim': 1,
            'output_dim': 1,
        },
        'fourier_features': {
            'enabled': fourier_enabled,
            'dim': fourier_dim,
            'scale': fourier_scale,
        },
        'rwf': rwf,
    }


def load_ks_data():
    """Load KS training and eval datasets."""
    dataset_dir = Path("datasets/ks")
    
    train_data = torch.load(dataset_dir / "training_data.pt")
    eval_data = torch.load(dataset_dir / "eval_data.pt")
    
    return train_data, eval_data


def compute_relative_l2_error(pred, target):
    """Compute relative L2 error: ||pred - target|| / ||target||"""
    return torch.norm(pred - target) / torch.norm(target)


def train_supervised(model, train_data, eval_data, device, epochs=5000, batch_size=4096, lr=0.001):
    """Train ResNet with supervised MSE loss."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Prepare data
    x_train = train_data['x'].to(device)
    t_train = train_data['t'].to(device)
    h_train = train_data['h_gt'].to(device)
    
    x_eval = eval_data['x'].to(device)
    t_eval = eval_data['t'].to(device)
    h_eval = eval_data['h_gt'].to(device)
    
    # Combine x, t as input
    inputs_train = torch.cat([x_train, t_train], dim=1)
    inputs_eval = torch.cat([x_eval, t_eval], dim=1)
    
    n_train = len(inputs_train)
    n_batches = (n_train + batch_size - 1) // batch_size
    
    # Metrics storage
    train_losses = []
    eval_losses = []
    eval_rel_l2 = []
    eval_epochs = []
    
    print(f"\n{'='*60}")
    print("Training Supervised ResNet on KS")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Train samples: {n_train}")
    print(f"Eval samples: {len(inputs_eval)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        # Shuffle training data
        perm = torch.randperm(n_train)
        inputs_shuffled = inputs_train[perm]
        targets_shuffled = h_train[perm]
        
        # Mini-batch training
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_train)
            
            batch_inputs = inputs_shuffled[start_idx:end_idx]
            batch_targets = targets_shuffled[start_idx:end_idx]
            
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= n_batches
        train_losses.append(epoch_loss)
        
        # Evaluate every 100 epochs
        if epoch % 100 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                eval_pred = model(inputs_eval)
                eval_loss = criterion(eval_pred, h_eval).item()
                rel_l2 = compute_relative_l2_error(eval_pred, h_eval).item()
            
            eval_losses.append(eval_loss)
            eval_rel_l2.append(rel_l2)
            eval_epochs.append(epoch)
            
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch}/{epochs}] ({elapsed:.1f}s) | "
                  f"Train Loss: {epoch_loss:.6f} | "
                  f"Eval Loss: {eval_loss:.6f} | "
                  f"Rel-L2: {rel_l2:.6f}")
    
    elapsed_total = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed in {elapsed_total:.1f}s")
    print(f"Best Rel-L2: {min(eval_rel_l2):.6f}")
    print(f"{'='*60}\n")
    
    return {
        'train_losses': train_losses,
        'eval_losses': eval_losses,
        'eval_rel_l2': eval_rel_l2,
        'eval_epochs': eval_epochs
    }


def plot_results(metrics, model, eval_data, device, output_dir):
    """Plot training curves and predictions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax = axes[0]
    ax.semilogy(range(1, len(metrics['train_losses']) + 1), metrics['train_losses'], 
                label='Train Loss', alpha=0.7)
    ax.semilogy(metrics['eval_epochs'], metrics['eval_losses'], 
                label='Eval Loss', marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training and Evaluation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rel-L2 curve
    ax = axes[1]
    ax.plot(metrics['eval_epochs'], metrics['eval_rel_l2'], 
            color='red', marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Relative L2 Error')
    ax.set_title('Evaluation Relative L2 Error')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {output_dir / 'training_curves.png'}")
    plt.close()
    
    # Plot predictions vs ground truth (heatmap on regular grid)
    # For KS: x in [0, 2π], t in [0, 1]
    x_eval = eval_data['x'].cpu().numpy()
    t_eval = eval_data['t'].cpu().numpy()
    h_eval = eval_data['h_gt'].cpu().numpy()
    
    model.eval()
    with torch.no_grad():
        inputs_eval = torch.cat([eval_data['x'], eval_data['t']], dim=1).to(device)
        h_pred = model(inputs_eval).cpu().numpy()
    
    # Create grid for visualization
    nx, nt = 100, 100
    x_grid = np.linspace(0, 2*np.pi, nx)
    t_grid = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x_grid, t_grid)
    
    # Interpolate predictions onto grid
    from scipy.interpolate import griddata
    points = np.column_stack([x_eval.flatten(), t_eval.flatten()])
    H_pred_grid = griddata(points, h_pred.flatten(), (X, T), method='linear')
    H_gt_grid = griddata(points, h_eval.flatten(), (X, T), method='linear')
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Ground truth
    im0 = axes[0].imshow(H_gt_grid, aspect='auto', origin='lower', 
                         extent=[0, 2*np.pi, 0, 1], cmap='viridis')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    axes[0].set_title('Ground Truth')
    plt.colorbar(im0, ax=axes[0])
    
    # Prediction
    im1 = axes[1].imshow(H_pred_grid, aspect='auto', origin='lower',
                         extent=[0, 2*np.pi, 0, 1], cmap='viridis')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')
    axes[1].set_title('Supervised ResNet Prediction')
    plt.colorbar(im1, ax=axes[1])
    
    # Error
    error = np.abs(H_pred_grid - H_gt_grid)
    im2 = axes[2].imshow(error, aspect='auto', origin='lower',
                         extent=[0, 2*np.pi, 0, 1], cmap='hot')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('t')
    axes[2].set_title('Pointwise Error')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'predictions.png', dpi=150, bbox_inches='tight')
    print(f"Predictions saved to {output_dir / 'predictions.png'}")
    plt.close()


def main():
    """Main entry point."""
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    architecture = [2, 256, 256, 256, 256, 256, 1]  # Same as PINN experiments
    activation = 'tanh'
    epochs = 5000
    batch_size = 4096
    lr = 0.001
    output_dir = Path("outputs/supervised_resnet_test")
    
    # Match current KS experiments: FF (dim=128, scale=1.5) + RWF
    fourier_dim = 128
    fourier_scale = 1.5
    use_ff = True
    use_rwf = True
    config = _build_ks_pinn_config(
        fourier_enabled=use_ff,
        fourier_dim=fourier_dim,
        fourier_scale=fourier_scale,
        rwf=use_rwf,
    )

    print("\n" + "="*60)
    print("Supervised ResNet Test for KS")
    print("="*60)
    print(f"Architecture: {architecture}")
    print(f"Activation: {activation}")
    print(f"Fourier features: {use_ff} (dim={fourier_dim}, scale={fourier_scale})")
    print(f"RWF: {use_rwf}")
    print("="*60)
    
    # Load data
    print("\nLoading KS datasets...")
    train_data, eval_data = load_ks_data()
    print(f"Train samples: {len(train_data['x'])}")
    print(f"Eval samples: {len(eval_data['x'])}")
    
    # Create model (create_network: layers, activation, config, expert_type)
    print("\nCreating ResNet model...")
    model = create_network(
        architecture,
        activation,
        config,
        is_base=True,
        expert_type='resnet',
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    
    # Train
    print("\nStarting training...")
    metrics = train_supervised(model, train_data, eval_data, device, 
                               epochs=epochs, batch_size=batch_size, lr=lr)
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(metrics, model, eval_data, device, output_dir)
    
    print("\n" + "="*60)
    print("Test complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
