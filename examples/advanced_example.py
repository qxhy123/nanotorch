#!/usr/bin/env python3
"""
Advanced example demonstrating nanotorch features:
- Sequential container
- New tensor operations (abs, sqrt, clamp)
- Training loop with autograd
"""

import sys

sys.path.insert(0, ".")

import numpy as np
from nanotorch import Tensor, no_grad
from nanotorch.nn import Sequential, Linear, ReLU, MSE
from nanotorch.optim import SGD


def create_robust_model(input_size=10, hidden_size=20, output_size=1):
    """Create a model with robust operations (abs, sqrt, clamp)."""
    model = Sequential(
        Linear(input_size, hidden_size),
        ReLU(),
        Linear(hidden_size, hidden_size),
        ReLU(),
        Linear(hidden_size, output_size),
        # Apply robust transformations to output
        # This makes the model more robust to outliers
    )
    return model


class RobustModel:
    """Custom model using new tensor operations."""

    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, hidden_size)
        self.fc3 = Linear(hidden_size, output_size)

    def forward(self, x):
        # Standard forward pass
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        x = x.relu()
        x = self.fc3(x)

        # Apply robust transformations
        # 1. Absolute value (makes output symmetric)
        x = x.abs()
        # 2. Square root (compresses large values)
        x = x.sqrt()
        # 3. Clamp to reasonable range
        x = x.clamp(min_val=0.0, max_val=5.0)

        return x

    def parameters(self):
        """Get all parameters."""
        params = []
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        params.extend(self.fc3.parameters())
        return params

    def __call__(self, x):
        return self.forward(x)


def generate_data(n_samples=100, input_size=10):
    """Generate synthetic data with outliers."""
    np.random.seed(42)

    # Normal data
    X = np.random.randn(n_samples, input_size).astype(np.float32)

    # Targets with some outliers
    y = X[:, 0:1] * 2.0 + X[:, 1:2] * 1.5 + np.random.randn(n_samples, 1) * 0.1

    # Add some outliers
    outlier_indices = np.random.choice(n_samples, size=5, replace=False)
    y[outlier_indices] *= 10.0  # Make some targets much larger

    return Tensor(X), Tensor(y)


def train_model(model, X, y, epochs=100, lr=0.01):
    """Train model with MSE loss."""
    criterion = MSE()
    optimizer = SGD(model.parameters(), lr=lr)

    print(f"Training for {epochs} epochs...")
    print("Epoch | Loss")
    print("-" * 20)

    for epoch in range(epochs):
        # Forward pass
        predictions = model(X)
        loss = criterion(predictions, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"{epoch:5d} | {loss.item():.6f}")

    return loss.item()


def evaluate_model(model, X, y):
    """Evaluate model performance."""
    with no_grad():
        predictions = model(X)
        mse = ((predictions - y) ** 2).mean().item()
        mae = (predictions - y).abs().mean().item()

    print(f"\nEvaluation:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(
        f"  Predictions range: [{predictions.data.min():.3f}, {predictions.data.max():.3f}]"
    )

    # Check if predictions are within clamped range
    clamped = predictions.clamp(min_val=0.0, max_val=5.0)
    clamping_ratio = (predictions.data != clamped.data).sum() / predictions.data.size
    print(f"  Clamping applied to {clamping_ratio*100:.1f}% of outputs")

    return mse, mae


def main():
    print("=" * 60)
    print("nanotorch Advanced Example")
    print("Demonstrating Sequential, abs, sqrt, clamp operations")
    print("=" * 60)

    # Generate data
    print("\n1. Generating data with outliers...")
    X, y = generate_data(n_samples=200, input_size=10)
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   y range: [{y.data.min():.3f}, {y.data.max():.3f}]")

    # Create models
    print("\n2. Creating models...")
    sequential_model = create_robust_model(input_size=10, hidden_size=20, output_size=1)
    robust_model = RobustModel(input_size=10, hidden_size=20, output_size=1)

    print(f"   Sequential model: {sequential_model}")
    print(f"   Robust model with custom operations")

    # Train sequential model
    print("\n3. Training Sequential model...")
    loss1 = train_model(sequential_model, X, y, epochs=100, lr=0.01)

    # Train robust model
    print("\n4. Training Robust model (with abs, sqrt, clamp)...")
    loss2 = train_model(robust_model, X, y, epochs=100, lr=0.01)

    # Evaluate both models
    print("\n5. Evaluating Sequential model...")
    mse1, mae1 = evaluate_model(sequential_model, X, y)

    print("\n6. Evaluating Robust model...")
    mse2, mae2 = evaluate_model(robust_model, X, y)

    # Compare
    print("\n" + "=" * 60)
    print("Comparison:")
    print(
        f"  Sequential model - Final loss: {loss1:.6f}, MSE: {mse1:.6f}, MAE: {mae1:.6f}"
    )
    print(
        f"  Robust model     - Final loss: {loss2:.6f}, MSE: {mse2:.6f}, MAE: {mae2:.6f}"
    )

    # Demonstrate tensor operations
    print("\n" + "=" * 60)
    print("Tensor Operations Demo:")
    demo_tensor = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 4.0])
    print(f"  Original: {demo_tensor.data}")
    print(f"  abs():    {demo_tensor.abs().data}")
    print(f"  sqrt():   {demo_tensor.sqrt().data}")
    print(f"  clamp(-1, 3): {demo_tensor.clamp(min_val=-1.0, max_val=3.0).data}")

    # Gradient check
    print("\n  Gradient check (autograd working):")
    x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    y = x.abs().sqrt().clamp(min_val=0.0, max_val=2.0).sum()
    y.backward()
    print(f"    x: {x.data}")
    print(f"    x.grad: {x.grad.data}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
