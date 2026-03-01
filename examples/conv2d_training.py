"""
Convolutional neural network example using nanotorch.

This example demonstrates:
1. Creating a simple CNN with Conv2D layer
2. Training on synthetic image data
3. Using autograd for backpropagation
4. Using SGD optimizer to update weights
"""

import numpy as np
import nanotorch as nt
from nanotorch.nn import Module, Conv2D, ReLU, Linear, MSE
from nanotorch.optim import SGD


def generate_image_data(
    num_samples=32, img_channels=1, img_height=8, img_width=8, num_classes=4
):
    """Generate synthetic image data for classification.

    Creates random images and labels using a random convolutional filter.
    """
    np.random.seed(42)

    # Create a random convolutional filter that we'll try to learn
    filter_size = 3
    true_filter = (
        np.random.randn(num_classes, img_channels, filter_size, filter_size).astype(
            np.float32
        )
        * 0.1
    )
    true_bias = np.random.randn(num_classes).astype(np.float32) * 0.1

    # Generate random input images
    X = np.random.randn(num_samples, img_channels, img_height, img_width).astype(
        np.float32
    )

    # Apply convolution to create target "features"
    # We'll simulate a simple classification task where the target is the
    # maximum activation across spatial dimensions after convolution
    y = np.zeros((num_samples, num_classes), dtype=np.float32)

    for i in range(num_samples):
        for c in range(num_classes):
            # Simple 2D convolution (valid mode)
            conv_result = np.zeros(
                (img_height - filter_size + 1, img_width - filter_size + 1)
            )
            for h in range(img_height - filter_size + 1):
                for w in range(img_width - filter_size + 1):
                    window = X[i, :, h : h + filter_size, w : w + filter_size]
                    conv_result[h, w] = np.sum(window * true_filter[c]) + true_bias[c]
            # Take max activation as target
            y[i, c] = conv_result.max()

    # Add small noise
    y += np.random.randn(*y.shape).astype(np.float32) * 0.01

    return X, y


class SimpleCNN(Module):
    """A simple convolutional neural network with one Conv2D layer."""

    def __init__(
        self, in_channels, out_channels, kernel_size, img_height, img_width, num_classes
    ):
        super().__init__()
        # Convert kernel_size to tuple if needed
        kernel_size_tuple = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.conv = Conv2D(in_channels, out_channels, kernel_size_tuple, padding=1)
        self.relu = ReLU()
        # Compute output size after convolution with padding=1, stride=1, dilation=1
        padding = 1
        stride = 1
        dilation = 1
        kernel_size_val = kernel_size_tuple[0]
        output_height = (
            img_height + 2 * padding - dilation * (kernel_size_val - 1) - 1
        ) // stride + 1
        output_width = (
            img_width + 2 * padding - dilation * (kernel_size_val - 1) - 1
        ) // stride + 1
        linear_input_features = out_channels * output_height * output_width
        self.linear = Linear(linear_input_features, num_classes)

    def forward(self, x):
        # x shape: (N, C, H, W)
        x = self.conv(x)  # (N, out_channels, H, W) with padding=1
        x = self.relu(x)
        # Flatten spatial dimensions
        N, C, H, W = x.shape
        x = x.reshape((N, C * H * W))
        x = self.linear(x)
        return x


def train():
    """Train the convolutional neural network."""
    # Hyperparameters
    img_channels = 1
    img_height = 8
    img_width = 8
    out_channels = 4
    kernel_size = 3
    num_classes = 4
    num_samples = 64
    learning_rate = 0.01
    num_epochs = 50
    batch_size = 8

    # Generate data
    X_np, y_np = generate_image_data(
        num_samples=num_samples,
        img_channels=img_channels,
        img_height=img_height,
        img_width=img_width,
        num_classes=num_classes,
    )

    # Convert to nanotorch tensors
    X = nt.Tensor(X_np)
    y = nt.Tensor(y_np)

    # Create model, loss function, and optimizer
    model = SimpleCNN(
        img_channels, out_channels, kernel_size, img_height, img_width, num_classes
    )
    criterion = MSE(reduction="mean")
    optimizer = SGD(model.parameters(), lr=learning_rate)

    print(
        f"Training CNN with {sum(p.data.size for p in model.parameters())} parameters"
    )
    print(f"Input shape: (N, {img_channels}, {img_height}, {img_width})")
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


def verify_conv2d_gradient():
    """Perform a simple gradient check for Conv2D layer."""
    print("\n" + "=" * 50)
    print("Conv2D Gradient Check")
    print("=" * 50)

    # Create a simple Conv2D layer
    conv = Conv2D(1, 2, kernel_size=(3, 3), padding=1)

    # Create dummy image data
    x = nt.Tensor(np.random.randn(2, 1, 5, 5).astype(np.float32), requires_grad=True)
    y = nt.Tensor(
        np.random.randn(2, 2, 5, 5).astype(np.float32)
    )  # same spatial size due to padding

    # Forward pass
    pred = conv(x)
    loss = pred.sum()  # simple loss

    # Backward pass
    loss.backward()

    # Check that gradients are not None
    for name, param in conv.named_parameters():
        if param.grad is not None:
            print(f"✓ {name}: gradient computed (shape {param.grad.shape})")
        else:
            print(f"✗ {name}: gradient is None")

    # Simple finite difference check for one parameter
    param = conv.weight
    epsilon = 1e-3

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
        pred_plus = conv(x)
        loss_plus = pred_plus.sum().item()

        # f(x - epsilon)
        param.data[idx] = original - epsilon
        pred_minus = conv(x)
        loss_minus = pred_minus.sum().item()

        # Reset parameter
        param.data[idx] = original

        # Finite difference
        numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        it.iternext()

    # Compare with analytical gradient
    assert param.grad is not None, "Gradient should be computed"
    analytical_grad = param.grad.data
    diff = np.abs(analytical_grad - numerical_grad).max()
    relative_diff = np.abs(analytical_grad - numerical_grad) / (np.abs(numerical_grad) + 1e-8)
    max_relative = relative_diff.max()

    print(f"\nGradient check:")
    print(f"Max absolute difference: {diff:.2e}")
    print(f"Max relative difference: {max_relative:.2e}")
    
    if np.allclose(analytical_grad, numerical_grad, rtol=2e-2, atol=1e-3):
        print("✓ Conv2D gradient check passed!")
    else:
        print("✗ Conv2D gradient check failed - check Conv2D implementation")

    # Restore parameter
    param.data = original_data


if __name__ == "__main__":
    print("=" * 50)
    print("nanotorch - Conv2D Neural Network Example")
    print("=" * 50)

    # Train a simple CNN
    model = train()

    # Perform gradient check for Conv2D
    verify_conv2d_gradient()

    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("=" * 50)
