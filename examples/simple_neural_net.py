"""
Simple neural network example using nanotorch.

This example demonstrates:
1. Creating a simple MLP with nanotorch.nn
2. Training on synthetic data
3. Using autograd for backpropagation
4. Using SGD optimizer to update weights
"""

import numpy as np
import nanotorch as nt
from nanotorch.nn import Linear, ReLU, MSE
from nanotorch.optim import SGD


def generate_data(num_samples=100, input_dim=10, output_dim=2):
    """Generate synthetic training data."""
    np.random.seed(42)

    # True weight matrix and bias
    true_W = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.1
    true_b = np.random.randn(output_dim).astype(np.float32) * 0.1

    # Generate input data
    X = np.random.randn(num_samples, input_dim).astype(np.float32)

    # Generate target with some noise
    y = X @ true_W + true_b
    y += np.random.randn(*y.shape).astype(np.float32) * 0.01

    return X, y


class SimpleMLP(nt.nn.Module):
    """A simple multi-layer perceptron with one hidden layer."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = Linear(input_dim, hidden_dim)
        self.relu = ReLU()
        self.linear2 = Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def train():
    """Train the neural network."""
    # Hyperparameters
    input_dim = 10
    hidden_dim = 20
    output_dim = 2
    num_samples = 100
    learning_rate = 0.01
    num_epochs = 100
    batch_size = 16

    # Generate data
    X_np, y_np = generate_data(num_samples, input_dim, output_dim)

    # Convert to nanotorch tensors
    X = nt.Tensor(X_np)
    y = nt.Tensor(y_np)

    # Create model, loss function, and optimizer
    model = SimpleMLP(input_dim, hidden_dim, output_dim)
    criterion = MSE(reduction="mean")
    optimizer = SGD(model.parameters(), lr=learning_rate)

    print(
        f"Training model with {sum(p.data.size for p in model.parameters())} parameters"
    )
    print(f"Dataset: {num_samples} samples")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("-" * 50)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Simple mini-batch training (shuffle each epoch)
        indices = np.random.permutation(num_samples)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            # Get batch data
            X_batch = nt.Tensor(X_np[batch_indices])
            y_batch = nt.Tensor(y_np[batch_indices])

            # Forward pass
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] Loss: {avg_loss:.6f}")

    print("-" * 50)
    print("Training completed!")

    # Evaluate on training data
    with nt.no_grad():
        predictions = model(X)
        final_loss = criterion(predictions, y).item()
        print(f"Final training loss: {final_loss:.6f}")

        # Compute R^2 score
        ss_total = ((y_np - y_np.mean()) ** 2).sum()
        ss_residual = ((y_np - predictions.data) ** 2).sum()
        r2 = 1 - ss_residual / ss_total
        print(f"R^2 score: {r2:.4f}")

    return model


def gradient_check():
    """Perform a simple gradient check to verify autograd implementation."""
    print("\n" + "=" * 50)
    print("Gradient Check")
    print("=" * 50)

    # Create a simple linear model
    model = Linear(3, 2)
    criterion = MSE()

    # Create dummy data
    x = nt.Tensor(np.random.randn(4, 3).astype(np.float32), requires_grad=True)
    y = nt.Tensor(np.random.randn(4, 2).astype(np.float32))

    # Forward pass
    pred = model(x)
    loss = criterion(pred, y)

    # Backward pass
    loss.backward()

    # Check that gradients are not None
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"✓ {name}: gradient computed (shape {param.grad.shape})")
        else:
            print(f"✗ {name}: gradient is None")

    # Simple finite difference check for one parameter
    param = model.weight
    epsilon = 1e-4

    # Store original data
    original_data = param.data.copy()

    # Compute numerical gradient
    numerical_grad = np.zeros_like(param.data)
    it = np.nditer(param.data, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        original = param.data[idx]

        # f(x + epsilon)
        param.data[idx] = original + epsilon
        pred_plus = model(x)
        loss_plus = criterion(pred_plus, y).item()

        # f(x - epsilon)
        param.data[idx] = original - epsilon
        pred_minus = model(x)
        loss_minus = criterion(pred_minus, y).item()

        # Reset parameter
        param.data[idx] = original

        # Finite difference
        numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        it.iternext()

    # Compare with analytical gradient
    analytical_grad = param.grad.data
    diff = np.abs(analytical_grad - numerical_grad).max()
    relative_diff = np.abs(analytical_grad - numerical_grad) / (np.abs(numerical_grad) + 1e-8)
    max_relative = relative_diff.max()

    print(f"\nGradient check:")
    print(f"Max absolute difference: {diff:.2e}")
    print(f"Max relative difference: {max_relative:.2e}")
    
    if np.allclose(analytical_grad, numerical_grad, rtol=2e-2, atol=1e-3):
        print("✓ Gradient check passed!")
    else:
        print("✗ Gradient check failed - autograd may have issues")

    # Restore parameter
    param.data = original_data


if __name__ == "__main__":
    print("=" * 50)
    print("nanotorch - Simple Neural Network Example")
    print("=" * 50)

    # Train a simple model
    model = train()

    # Perform gradient check
    gradient_check()

    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("=" * 50)
