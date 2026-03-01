"""
Conv3D (3D Convolution) example for nanotorch.

This example demonstrates:
1. Creating Conv3D layers with different kernel sizes
2. Forward pass with synthetic volumetric data
3. Output shape calculations for 3D convolutions
4. Gradient computation and verification
5. Simple training loop with 3D data
6. Comparison with Conv2D applied to slices
7. Educational explanation of 3D convolution concepts

Key concepts:
- 3D convolution slides a 3D kernel through volumetric data (D, H, W)
- Common applications: medical imaging (CT/MRI), video processing, scientific data
- Output size depends on kernel size, padding, stride, and dilation in 3D
"""

import numpy as np
import nanotorch as nt
from nanotorch.nn import Conv3D, Conv2D, ReLU, Linear, MSE
from nanotorch.optim import SGD


def create_volumetric_data(
    batch_size=2, channels=3, depth=8, height=8, width=8, pattern="random"
):
    """Create synthetic volumetric data for 3D convolution.

    Args:
        batch_size: Number of samples in batch
        channels: Number of channels (e.g., different modalities)
        depth: Depth dimension (number of slices/z-axis)
        height: Height dimension (y-axis)
        width: Width dimension (x-axis)
        pattern: Type of pattern ("random", "sphere", "cube", "slices")

    Returns:
        Tensor with shape (batch_size, channels, depth, height, width)
    """
    np.random.seed(42)

    if pattern == "random":
        data = np.random.randn(batch_size, channels, depth, height, width).astype(
            np.float32
        )

    elif pattern == "sphere":
        # Create a 3D sphere pattern
        data = np.zeros((batch_size, channels, depth, height, width), dtype=np.float32)
        center_d, center_h, center_w = depth // 2, height // 2, width // 2
        radius = min(depth, height, width) // 3

        z, y, x = np.mgrid[:depth, :height, :width]
        dist = np.sqrt(
            (z - center_d) ** 2 + (y - center_h) ** 2 + (x - center_w) ** 2
        )
        mask = dist <= radius

        for b in range(batch_size):
            for c in range(channels):
                # Add some noise to the sphere
                data[b, c] = np.where(
                    mask,
                    np.random.randn(depth, height, width) * 0.3 + 1.0,
                    np.random.randn(depth, height, width) * 0.1,
                )

    elif pattern == "cube":
        # Create a 3D cube pattern
        data = np.zeros((batch_size, channels, depth, height, width), dtype=np.float32)
        start_d, end_d = depth // 4, 3 * depth // 4
        start_h, end_h = height // 4, 3 * height // 4
        start_w, end_w = width // 4, 3 * width // 4

        for b in range(batch_size):
            for c in range(channels):
                data[b, c, start_d:end_d, start_h:end_h, start_w:end_w] = (
                    np.random.randn() * 0.2 + 1.0
                )
                data[b, c] += np.random.randn(depth, height, width) * 0.05

    elif pattern == "slices":
        # Create distinct patterns in different depth slices
        data = np.zeros((batch_size, channels, depth, height, width), dtype=np.float32)
        for b in range(batch_size):
            for c in range(channels):
                for d in range(depth):
                    # Each slice has a different random pattern
                    data[b, c, d] = np.random.randn(height, width) * 0.3
                    # Add a line that changes with depth
                    line_pos = int(d / depth * height)
                    data[b, c, d, line_pos, :] += 1.0

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return nt.Tensor(data, requires_grad=True)


def print_tensor_info(tensor, name="Tensor"):
    """Print shape and basic statistics of a tensor."""
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Mean: {tensor.data.mean():.6f}")
    print(f"  Std: {tensor.data.std():.6f}")
    print(f"  Min: {tensor.data.min():.6f}")
    print(f"  Max: {tensor.data.max():.6f}")


def demo_conv3d_forward_pass():
    """Demonstrate Conv3D forward pass with different configurations."""
    print("=" * 70)
    print("Conv3D Forward Pass Demonstration")
    print("=" * 70)

    # Create sample volumetric data
    x = create_volumetric_data(batch_size=2, channels=2, depth=8, height=8, width=8)
    print_tensor_info(x, "Input volumetric data")

    print("\n--- Testing different kernel sizes ---")

    # Test 1: Small kernel (3x3x3)
    conv_small = Conv3D(in_channels=2, out_channels=4, kernel_size=3, padding=1)
    output_small = conv_small(x)
    print_tensor_info(output_small, "Conv3D(kernel=3, padding=1) output")

    # Test 2: Larger kernel (5x5x5)
    conv_large = Conv3D(in_channels=2, out_channels=4, kernel_size=5, padding=2)
    output_large = conv_large(x)
    print_tensor_info(output_large, "Conv3D(kernel=5, padding=2) output")

    # Test 3: Non-cubic kernel (3x5x7)
    # Note: Conv3D currently only supports int padding (same for all dimensions)
    # Using padding=3 (maximum) to ensure all dimensions are adequately padded
    conv_noncubic = Conv3D(in_channels=2, out_channels=4, kernel_size=(3, 5, 7), padding=3)
    output_noncubic = conv_noncubic(x)
    print_tensor_info(output_noncubic, "Conv3D(kernel=(3,5,7), padding=3) output")

    # Test 4: With stride > 1
    conv_stride = Conv3D(in_channels=2, out_channels=4, kernel_size=3, padding=1, stride=2)
    output_stride = conv_stride(x)
    print_tensor_info(output_stride, "Conv3D(kernel=3, padding=1, stride=2) output")

    # Show layer parameters
    print("\n--- Layer Parameters ---")
    print(f"Conv3D (kernel=3) parameters:")
    for name, param in conv_small.named_parameters():
        print(f"  {name}: shape {param.shape}")

    return x, conv_small


def calculate_output_shape(input_shape, kernel_size, padding, stride, dilation=1):
    """Calculate output shape for 3D convolution.

    Formula: output = floor((input + 2*padding - dilation*(kernel-1) - 1) / stride) + 1
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    _, _, d_in, h_in, w_in = input_shape

    def calc_dim(in_dim, k, p, s, d):
        return (in_dim + 2 * p - d * (k - 1) - 1) // s + 1

    d_out = calc_dim(d_in, kernel_size[0], padding[0], stride[0], dilation[0])
    h_out = calc_dim(h_in, kernel_size[1], padding[1], stride[1], dilation[1])
    w_out = calc_dim(w_in, kernel_size[2], padding[2], stride[2], dilation[2])

    return d_out, h_out, w_out


def demo_output_shape_calculations():
    """Demonstrate and verify output shape calculations for Conv3D."""
    print("\n" + "=" * 70)
    print("Output Shape Calculations for Conv3D")
    print("=" * 70)

    input_shape = (2, 3, 16, 16, 16)  # (batch, channels, depth, height, width)
    print(f"\nInput shape: {input_shape}")

    configs = [
        {"kernel": 3, "padding": 1, "stride": 1, "desc": "Same size (kernel=3, pad=1)"},
        {"kernel": 5, "padding": 2, "stride": 1, "desc": "Same size (kernel=5, pad=2)"},
        {"kernel": 3, "padding": 0, "stride": 1, "desc": "No padding"},
        {"kernel": 3, "padding": 0, "stride": 2, "desc": "Downsampling (stride=2)"},
        # Note: Conv3D currently only supports int padding (same for all dimensions)
        {"kernel": (3, 5, 7), "padding": 3, "stride": 1, "desc": "Non-cubic kernel (padding=3)"},
    ]

    for config in configs:
        kernel = config["kernel"]
        padding = config["padding"]
        stride = config["stride"]
        desc = config["desc"]

        # Calculate expected output shape
        d_out, h_out, w_out = calculate_output_shape(
            input_shape, kernel, padding, stride
        )
        expected_shape = (input_shape[0], 4, d_out, h_out, w_out)

        # Verify with actual Conv3D layer
        conv = Conv3D(
            in_channels=3, out_channels=4, kernel_size=kernel, padding=padding, stride=stride
        )
        x = nt.Tensor(np.zeros(input_shape, dtype=np.float32))
        output = conv(x)
        actual_shape = output.shape

        print(f"\n{desc}:")
        print(f"  Config: kernel={kernel}, padding={padding}, stride={stride}")
        print(f"  Expected output shape: {expected_shape}")
        print(f"  Actual output shape:   {actual_shape}")
        print(f"  Match: {'✓' if expected_shape == actual_shape else '✗'}")


def demo_gradient_computation():
    """Verify gradient computation for Conv3D layer."""
    print("\n" + "=" * 70)
    print("Gradient Computation for Conv3D")
    print("=" * 70)

    # Create Conv3D layer and input
    conv = Conv3D(in_channels=2, out_channels=3, kernel_size=3, padding=1)
    x = create_volumetric_data(
        batch_size=2, channels=2, depth=6, height=6, width=6, pattern="random"
    )

    print_tensor_info(x, "Input")

    # Forward pass
    output = conv(x)
    print_tensor_info(output, "Output")

    # Create a simple loss (sum of output)
    loss = output.sum()
    print(f"\nLoss (sum of output): {loss.item():.6f}")

    # Backward pass
    loss.backward()

    # Check gradients
    print("\n--- Gradient Check ---")
    for name, param in conv.named_parameters():
        if param.grad is not None:
            grad_norm = np.sqrt((param.grad.data ** 2).sum())
            print(f"✓ {name}: gradient computed (shape {param.grad.shape}, norm={grad_norm:.6f})")
        else:
            print(f"✗ {name}: gradient is None")

    if x.grad is not None:
        grad_norm = np.sqrt((x.grad.data ** 2).sum())
        print(f"✓ Input gradient computed (shape {x.grad.shape}, norm={grad_norm:.6f})")
    else:
        print("✗ Input gradient is None")

    # Finite difference gradient check for weight
    print("\n--- Finite Difference Gradient Check ---")
    param = conv.weight
    epsilon = 1e-4
    original_data = param.data.copy()

    # Compute numerical gradient for a few elements (for efficiency)
    numerical_grad = np.zeros_like(param.data)
    # Conv3D weight shape: (out_channels, in_channels, kernel_d, kernel_h, kernel_w) = 5D
    indices = [(0, 0, 0, 0, 0), (0, 0, 1, 1, 1), (1, 1, 2, 2, 2)]  # Sample indices

    for idx in indices:
        if all(i < d for i, d in zip(idx, param.shape)):
            original = param.data[idx]

            # f(x + epsilon)
            param.data[idx] = original + epsilon
            output_plus = conv(x)
            loss_plus = output_plus.sum().item()

            # f(x - epsilon)
            param.data[idx] = original - epsilon
            output_minus = conv(x)
            loss_minus = output_minus.sum().item()

            # Reset parameter
            param.data[idx] = original

            # Finite difference
            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)

    # Compare with analytical gradient
    analytical_grad = param.grad.data
    checked_indices = [idx for idx in indices if all(i < d for i, d in zip(idx, param.shape))]

    max_diff = 0.0
    for idx in checked_indices:
        diff = abs(analytical_grad[idx] - numerical_grad[idx])
        max_diff = max(max_diff, diff)
        print(f"Index {idx}: analytical={analytical_grad[idx]:.6f}, numerical={numerical_grad[idx]:.6f}, diff={diff:.2e}")

    if max_diff < 1e-2:
        print(f"\n✓ Gradient check passed! Max difference: {max_diff:.2e}")
    else:
        print(f"\n✗ Gradient check failed! Max difference: {max_diff:.2e}")

    # Restore parameter
    param.data = original_data


def demo_training_loop():
    """Simple training demonstration with Conv3D layer."""
    print("\n" + "=" * 70)
    print("Training Demo with Conv3D Layer")
    print("=" * 70)

    # Generate synthetic volumetric data for classification
    np.random.seed(42)
    num_samples = 32
    batch_size = 8
    num_epochs = 20
    learning_rate = 0.01

    # Create input: small volumetric patches
    X_np = np.random.randn(num_samples, 2, 8, 8, 8).astype(np.float32)
    # Create target: simple pattern based on mean intensity (single value per sample)
    y_np = np.sign(X_np.mean(axis=(1, 2, 3, 4))).astype(np.float32)

    print(f"Training data shape: {X_np.shape}")
    print(f"Target shape: {y_np.shape}")
    print(f"Number of samples: {num_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {num_epochs}")

    # Simple model with Conv3D
    conv = Conv3D(in_channels=2, out_channels=4, kernel_size=3, padding=1)
    relu = ReLU()

    # Output after conv: (batch, 4, 8, 8, 8)
    # Flatten: (batch, 4 * 8 * 8 * 8) = (batch, 2048)
    linear = Linear(in_features=4 * 8 * 8 * 8, out_features=1)

    criterion = MSE()
    optimizer = SGD([*conv.parameters(), *linear.parameters()], lr=learning_rate)

    print(f"\nModel parameters: {sum(p.data.size for p in conv.parameters()) + sum(p.data.size for p in linear.parameters())}")

    # Training loop
    print("\n--- Training Loop ---")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle indices
        indices = np.random.permutation(num_samples)

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            # Get batch
            X_batch = nt.Tensor(X_np[batch_indices])
            y_batch = nt.Tensor(y_np[batch_indices].reshape(-1, 1))

            # Forward pass
            x = conv(X_batch)
            x = relu(x)
            # Flatten
            N, C, D, H, W = x.shape
            x = x.reshape((N, C * D * H * W))
            predictions = linear(x)

            loss = criterion(predictions, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] Loss: {avg_loss:.6f}")

    print("\n--- Training Evaluation ---")
    # Final prediction
    with nt.no_grad():
        x = conv(nt.Tensor(X_np))
        x = relu(x)
        N, C, D, H, W = x.shape
        x = x.reshape((N, C * D * H * W))
        final_predictions = linear(x)

        final_loss = criterion(final_predictions, nt.Tensor(y_np.reshape(-1, 1))).item()
        print(f"Final training loss: {final_loss:.6f}")

        # Calculate accuracy (sign prediction)
        predicted_labels = np.sign(final_predictions.data)
        accuracy = (predicted_labels.flatten() == y_np).mean() * 100
        print(f"Accuracy: {accuracy:.2f}%")


def compare_conv3d_with_conv2d():
    """Compare Conv3D with Conv2D applied to slices (educational)."""
    print("\n" + "=" * 70)
    print("Conv3D vs Conv2D Comparison")
    print("=" * 70)

    # Create volumetric data
    batch_size = 2
    channels = 2
    depth = 6
    height = 8
    width = 8

    x_3d = create_volumetric_data(
        batch_size=batch_size, channels=channels, depth=depth, height=height, width=width
    )
    print_tensor_info(x_3d, "Input volumetric data")

    # 3D convolution
    conv3d = Conv3D(in_channels=channels, out_channels=3, kernel_size=3, padding=1)
    output_3d = conv3d(x_3d)
    print_tensor_info(output_3d, "Conv3D output")

    # Apply Conv2D to each depth slice independently
    conv2d = Conv2D(in_channels=channels, out_channels=3, kernel_size=3, padding=1)

    # Reshape to apply Conv2D: (batch * depth, channels, height, width)
    N, C, D, H, W = x_3d.shape
    x_reshaped = x_3d.reshape((N * D, C, H, W))

    # Apply Conv2D
    output_2d_slices = conv2d(x_reshaped)

    # Reshape back: (batch, out_channels, depth, height, width)
    _, out_C, out_H, out_W = output_2d_slices.shape
    output_2d_reshaped = output_2d_slices.reshape((N, out_C, D, out_H, out_W))

    print_tensor_info(
        output_2d_reshaped, "Conv2D (applied slice-by-slice) output"
    )

    # Key difference explanation
    print("\n--- Key Differences ---")
    print("Conv3D:")
    print("  - Kernel is 3D: slides through (depth, height, width)")
    print("  - Captures spatial relationships in all three dimensions")
    print("  - Computationally expensive: O(k_d * k_h * k_w) operations per output")
    print("  - Applications: 3D object detection, medical imaging, video")

    print("\nConv2D on slices:")
    print("  - Kernel is 2D: slides through (height, width) only")
    print("  - Each depth slice processed independently")
    print("  - No temporal/spatial information across depth")
    print("  - Applications: Processing 2D slices from 3D data independently")

    # Visualize with a simple example
    print("\n--- Visual Example ---")
    print("Imagine a 3D cube with layers:")
    print("  Depth 0: [■■□□]")
    print("  Depth 1: [■■□□]")
    print("  Depth 2: [■■□□]")
    print("\nConv3D kernel=3x3x3 would see all 3 layers simultaneously")
    print("Conv2D kernel=3x3 sees each layer independently")


def demo_3d_patterns():
    """Demonstrate different 3D patterns and their Conv3D responses."""
    print("\n" + "=" * 70)
    print("3D Pattern Detection with Conv3D")
    print("=" * 70)

    patterns = ["random", "sphere", "cube", "slices"]

    for pattern in patterns:
        print(f"\n--- Pattern: {pattern.upper()} ---")

        # Create data with specific pattern
        x = create_volumetric_data(
            batch_size=1, channels=2, depth=8, height=8, width=8, pattern=pattern
        )

        # Apply Conv3D
        conv = Conv3D(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        output = conv(x)

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Input statistics: mean={x.data.mean():.3f}, std={x.data.std():.3f}")
        print(f"Output statistics: mean={output.data.mean():.3f}, std={output.data.std():.3f}")

        # Show response statistics per channel
        for c in range(4):
            channel_output = output.data[0, c]
            print(f"  Channel {c}: min={channel_output.min():.3f}, max={channel_output.max():.3f}")


def main():
    """Run all Conv3D demonstrations."""
    print("=" * 70)
    print("nanotorch - Conv3D (3D Convolution) Demonstration")
    print("=" * 70)
    print("\nThis example demonstrates 3D convolution for volumetric data processing.")
    print("Applications include:")
    print("  - Medical imaging (CT, MRI scans)")
    print("  - Video processing (temporal information)")
    print("  - Scientific data (3D simulations, volumetric measurements)")

    # 1. Forward pass demonstration
    x, conv = demo_conv3d_forward_pass()

    # 2. Output shape calculations
    demo_output_shape_calculations()

    # 3. Gradient computation
    demo_gradient_computation()

    # 4. Training loop
    demo_training_loop()

    # 5. Comparison with Conv2D
    compare_conv3d_with_conv2d()

    # 6. Pattern detection
    demo_3d_patterns()

    print("\n" + "=" * 70)
    print("Summary: Conv3D Key Concepts")
    print("=" * 70)
    print("1. Input shape: (batch, channels, depth, height, width)")
    print("2. Kernel slides through all 3 spatial dimensions")
    print("3. Output shape depends on kernel size, padding, stride, dilation")
    print("4. Computes gradients via backpropagation through time and space")
    print("5. Useful for data with 3D structure: medical, video, scientific")
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
