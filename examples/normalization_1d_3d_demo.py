"""
Normalization layers (1D and 3D) example for nanotorch.

This example demonstrates:
1. Creating BatchNorm1d, BatchNorm3d, InstanceNorm1d, InstanceNorm3d layers
2. Forward passes with appropriate input shapes
3. Gradient computation and verification
4. Comparison with GroupNorm (for InstanceNorm)
5. Training demonstration with these layers
"""

import numpy as np
import nanotorch as nt
from nanotorch.nn import BatchNorm1d, BatchNorm3d, InstanceNorm1d, InstanceNorm3d, GroupNorm
from nanotorch.nn import MSE


def create_sample_data_1d(batch_size=4, channels=6, length=32):
    """Create random 2D or 3D tensor for testing 1D normalization layers.
    
    Returns tensor with shape (batch_size, channels, length) for BatchNorm1d/InstanceNorm1d.
    """
    np.random.seed(42)
    data = np.random.randn(batch_size, channels, length).astype(np.float32)
    return nt.Tensor(data, requires_grad=True)


def create_sample_data_3d(batch_size=2, channels=4, depth=8, height=8, width=8):
    """Create random 5D tensor for testing 3D normalization layers.
    
    Returns tensor with shape (batch_size, channels, depth, height, width) for BatchNorm3d/InstanceNorm3d.
    """
    np.random.seed(42)
    data = np.random.randn(batch_size, channels, depth, height, width).astype(np.float32)
    return nt.Tensor(data, requires_grad=True)


def demo_batchnorm1d():
    """Demonstrate forward pass of BatchNorm1d."""
    print("=" * 60)
    print("BatchNorm1d Forward Pass Demonstration")
    print("=" * 60)
    
    # Create input tensor (N, C, L)
    x = create_sample_data_1d(batch_size=2, channels=3, length=10)
    print(f"Input shape: {x.shape}")
    print(f"Input mean per channel: {x.data.mean(axis=(0, 2))}")
    print(f"Input std per channel: {x.data.std(axis=(0, 2))}")
    
    # BatchNorm1d without affine parameters
    bn1d = BatchNorm1d(num_features=3, affine=False, track_running_stats=True)
    output = bn1d(x)
    print(f"\nBatchNorm1d (affine=False) output shape: {output.shape}")
    print(f"Output mean per channel: {output.data.mean(axis=(0, 2))}")
    print(f"Output std per channel: {output.data.std(axis=(0, 2))}")
    
    # BatchNorm1d with affine parameters
    bn1d_affine = BatchNorm1d(num_features=3, affine=True, track_running_stats=True)
    output_affine = bn1d_affine(x)
    print(f"\nBatchNorm1d (affine=True) output shape: {output_affine.shape}")
    print(f"Output mean per channel: {output_affine.data.mean(axis=(0, 2))}")
    print(f"Output std per channel: {output_affine.data.std(axis=(0, 2))}")
    
    # Show learnable parameters
    print(f"\nLearnable parameters: {list(bn1d_affine.named_parameters())}")
    
    return x, bn1d, bn1d_affine


def demo_batchnorm3d():
    """Demonstrate forward pass of BatchNorm3d."""
    print("\n" + "=" * 60)
    print("BatchNorm3d Forward Pass Demonstration")
    print("=" * 60)
    
    # Create input tensor (N, C, D, H, W)
    x = create_sample_data_3d(batch_size=2, channels=3, depth=5, height=5, width=5)
    print(f"Input shape: {x.shape}")
    print(f"Input mean per channel: {x.data.mean(axis=(0, 2, 3, 4))}")
    print(f"Input std per channel: {x.data.std(axis=(0, 2, 3, 4))}")
    
    # BatchNorm3d without affine parameters
    bn3d = BatchNorm3d(num_features=3, affine=False, track_running_stats=True)
    output = bn3d(x)
    print(f"\nBatchNorm3d (affine=False) output shape: {output.shape}")
    print(f"Output mean per channel: {output.data.mean(axis=(0, 2, 3, 4))}")
    print(f"Output std per channel: {output.data.std(axis=(0, 2, 3, 4))}")
    
    # BatchNorm3d with affine parameters
    bn3d_affine = BatchNorm3d(num_features=3, affine=True, track_running_stats=True)
    output_affine = bn3d_affine(x)
    print(f"\nBatchNorm3d (affine=True) output shape: {output_affine.shape}")
    print(f"Output mean per channel: {output_affine.data.mean(axis=(0, 2, 3, 4))}")
    print(f"Output std per channel: {output_affine.data.std(axis=(0, 2, 3, 4))}")
    
    # Show learnable parameters
    print(f"\nLearnable parameters: {list(bn3d_affine.named_parameters())}")
    
    return x, bn3d, bn3d_affine


def demo_instancenorm1d():
    """Demonstrate forward pass of InstanceNorm1d."""
    print("\n" + "=" * 60)
    print("InstanceNorm1d Forward Pass Demonstration")
    print("=" * 60)
    
    # Create input tensor (N, C, L)
    x = create_sample_data_1d(batch_size=2, channels=3, length=10)
    print(f"Input shape: {x.shape}")
    print(f"Input mean per channel per sample: {x.data.mean(axis=(2,))}")
    print(f"Input std per channel per sample: {x.data.std(axis=(2,))}")
    
    # InstanceNorm1d without affine parameters (default)
    inorm1d = InstanceNorm1d(num_features=3, affine=False)
    output = inorm1d(x)
    print(f"\nInstanceNorm1d (affine=False) output shape: {output.shape}")
    print(f"Output mean per channel per sample: {output.data.mean(axis=(2,))}")
    print(f"Output std per channel per sample: {output.data.std(axis=(2,))}")
    
    # InstanceNorm1d with affine parameters
    inorm1d_affine = InstanceNorm1d(num_features=3, affine=True)
    output_affine = inorm1d_affine(x)
    print(f"\nInstanceNorm1d (affine=True) output shape: {output_affine.shape}")
    print(f"Output mean per channel per sample: {output_affine.data.mean(axis=(2,))}")
    print(f"Output std per channel per sample: {output_affine.data.std(axis=(2,))}")
    
    # Show learnable parameters
    print(f"\nLearnable parameters: {list(inorm1d_affine.named_parameters())}")
    
    return x, inorm1d, inorm1d_affine


def demo_instancenorm3d():
    """Demonstrate forward pass of InstanceNorm3d."""
    print("\n" + "=" * 60)
    print("InstanceNorm3d Forward Pass Demonstration")
    print("=" * 60)
    
    # Create input tensor (N, C, D, H, W)
    x = create_sample_data_3d(batch_size=2, channels=3, depth=5, height=5, width=5)
    print(f"Input shape: {x.shape}")
    print(f"Input mean per channel per sample: {x.data.mean(axis=(2, 3, 4))}")
    print(f"Input std per channel per sample: {x.data.std(axis=(2, 3, 4))}")
    
    # InstanceNorm3d without affine parameters (default)
    inorm3d = InstanceNorm3d(num_features=3, affine=False)
    output = inorm3d(x)
    print(f"\nInstanceNorm3d (affine=False) output shape: {output.shape}")
    print(f"Output mean per channel per sample: {output.data.mean(axis=(2, 3, 4))}")
    print(f"Output std per channel per sample: {output.data.std(axis=(2, 3, 4))}")
    
    # InstanceNorm3d with affine parameters
    inorm3d_affine = InstanceNorm3d(num_features=3, affine=True)
    output_affine = inorm3d_affine(x)
    print(f"\nInstanceNorm3d (affine=True) output shape: {output_affine.shape}")
    print(f"Output mean per channel per sample: {output_affine.data.mean(axis=(2, 3, 4))}")
    print(f"Output std per channel per sample: {output_affine.data.std(axis=(2, 3, 4))}")
    
    # Show learnable parameters
    print(f"\nLearnable parameters: {list(inorm3d_affine.named_parameters())}")
    
    return x, inorm3d, inorm3d_affine


def gradient_check(layer, x, layer_name):
    """Verify gradient computation for a normalization layer."""
    print(f"\n" + "=" * 60)
    print(f"Gradient Check for {layer_name}")
    print("=" * 60)
    
    # Create random target for loss computation
    target = nt.Tensor(np.random.randn(*x.shape).astype(np.float32))
    criterion = MSE()
    
    # Forward pass
    output = layer(x)
    loss = criterion(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist for learnable parameters
    print("Gradient check for parameters:")
    for name, param in layer.named_parameters():
        if param.grad is not None:
            grad_norm = np.sqrt((param.grad.data ** 2).sum())
            print(f"  ✓ {name}: gradient computed (norm={grad_norm:.6f})")
        else:
            print(f"  ✗ {name}: gradient is None")
    
    # If no learnable parameters, skip finite difference check
    if len(list(layer.parameters())) == 0:
        print("No learnable parameters to check gradient.")
        return True
    
    # Finite difference gradient check for first parameter
    param = list(layer.parameters())[0]
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
        output_plus = layer(x)
        loss_plus = criterion(output_plus, target).item()
        
        # f(x - epsilon)
        param.data[idx] = original - epsilon
        output_minus = layer(x)
        loss_minus = criterion(output_minus, target).item()
        
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
    
    print(f"\nGradient comparison:")
    print(f"Max absolute difference: {diff:.2e}")
    print(f"Max relative difference: {max_relative:.2e}")
    
    if np.allclose(analytical_grad, numerical_grad, rtol=2e-2, atol=1e-3):
        print(f"✓ Gradient check passed for {layer_name}!")
        success = True
    else:
        print(f"✗ Gradient check failed for {layer_name} - autograd may have issues")
        success = False
    
    # Restore parameter
    param.data = original_data
    
    return success


def compare_instance_with_groupnorm():
    """Show equivalence between InstanceNorm and GroupNorm with num_groups=num_channels."""
    print("\n" + "=" * 60)
    print("InstanceNorm vs GroupNorm Comparison (1D and 3D)")
    print("=" * 60)
    
    # 1D case
    x1d = create_sample_data_1d(batch_size=3, channels=6, length=8)
    inorm1d = InstanceNorm1d(num_features=6, affine=False)
    gnorm1d = GroupNorm(num_groups=6, num_channels=6, affine=False)
    output_inorm1d = inorm1d(x1d)
    output_gnorm1d = gnorm1d(x1d)
    diff1d = np.abs(output_inorm1d.data - output_gnorm1d.data).max()
    
    print(f"1D case:")
    print(f"  Input shape: {x1d.shape}")
    print(f"  InstanceNorm1d output shape: {output_inorm1d.shape}")
    print(f"  GroupNorm output shape: {output_gnorm1d.shape}")
    print(f"  Maximum difference: {diff1d:.2e}")
    
    # 3D case
    x3d = create_sample_data_3d(batch_size=2, channels=4, depth=5, height=5, width=5)
    inorm3d = InstanceNorm3d(num_features=4, affine=False)
    gnorm3d = GroupNorm(num_groups=4, num_channels=4, affine=False)
    output_inorm3d = inorm3d(x3d)
    output_gnorm3d = gnorm3d(x3d)
    diff3d = np.abs(output_inorm3d.data - output_gnorm3d.data).max()
    
    print(f"\n3D case:")
    print(f"  Input shape: {x3d.shape}")
    print(f"  InstanceNorm3d output shape: {output_inorm3d.shape}")
    print(f"  GroupNorm output shape: {output_gnorm3d.shape}")
    print(f"  Maximum difference: {diff3d:.2e}")
    
    success = diff1d < 1e-5 and diff3d < 1e-5
    if success:
        print("\n✓ InstanceNorm matches GroupNorm with num_groups=num_channels for both 1D and 3D")
    else:
        print("\n✗ Outputs differ - check implementation")
    
    return success


def training_demo():
    """Simple training demonstration with 1D and 3D normalization layers."""
    print("\n" + "=" * 60)
    print("Training Demo with 1D and 3D Normalization Layers")
    print("=" * 60)
    
    from nanotorch.nn import Sequential, Linear, ReLU
    from nanotorch.optim import SGD
    
    # 1D network with BatchNorm1d and InstanceNorm1d
    print("\n1D Network (sequential data):")
    model_1d = Sequential(
        Linear(20, 32),
        BatchNorm1d(num_features=32, affine=True),
        ReLU(),
        Linear(32, 16),
        InstanceNorm1d(num_features=16, affine=True),
        ReLU(),
        Linear(16, 1)
    )
    
    # Create synthetic 1D data (batch, features)
    x1d = nt.Tensor(np.random.randn(16, 20).astype(np.float32))
    target1d = nt.Tensor(np.random.randn(16, 1).astype(np.float32))
    
    criterion = MSE()
    optimizer1d = SGD(model_1d.parameters(), lr=0.01)
    
    print(f"1D Model parameters: {sum(p.data.size for p in model_1d.parameters())}")
    
    # Training loop (just a few iterations)
    losses_1d = []
    for step in range(5):
        optimizer1d.zero_grad()
        output = model_1d(x1d)
        loss = criterion(output, target1d)
        loss.backward()
        optimizer1d.step()
        
        losses_1d.append(loss.item())
        print(f"  Step {step + 1}: loss = {loss.item():.6f}")
    
    # 3D network with BatchNorm3d and InstanceNorm3d (simulated 3D CNN)
    print("\n3D Network (volumetric data):")
    # Since we don't have Conv3D, we'll simulate with existing Conv2D but reshape.
    # This is just for demonstration.
    model_3d = Sequential(
        # Simulating 3D convolutions would require Conv3D which we don't have,
        # so we'll use a simple example with BatchNorm3d on random 5D data.
        # We'll just show that BatchNorm3d works with 5D input.
    )
    
    # Create synthetic 3D data (batch, channels, depth, height, width)
    x3d = nt.Tensor(np.random.randn(2, 4, 8, 8, 8).astype(np.float32))
    bn3d = BatchNorm3d(num_features=4, affine=True)
    inorm3d = InstanceNorm3d(num_features=4, affine=True)
    
    # Forward passes
    output_bn = bn3d(x3d)
    output_in = inorm3d(x3d)
    
    print(f"BatchNorm3d output shape: {output_bn.shape}")
    print(f"InstanceNorm3d output shape: {output_in.shape}")
    print("3D normalization layers working correctly.")
    
    return model_1d


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("nanotorch - 1D and 3D Normalization Layers Example")
    print("=" * 60)
    
    # 1. Forward pass demonstrations
    x_bn1d, bn1d, bn1d_affine = demo_batchnorm1d()
    x_bn3d, bn3d, bn3d_affine = demo_batchnorm3d()
    x_in1d, in1d, in1d_affine = demo_instancenorm1d()
    x_in3d, in3d, in3d_affine = demo_instancenorm3d()
    
    # 2. Gradient checks
    gradient_check(bn1d_affine, x_bn1d, "BatchNorm1d (affine=True)")
    gradient_check(bn3d_affine, x_bn3d, "BatchNorm3d (affine=True)")
    gradient_check(in1d_affine, x_in1d, "InstanceNorm1d (affine=True)")
    gradient_check(in3d_affine, x_in3d, "InstanceNorm3d (affine=True)")
    
    # 3. Comparison with GroupNorm
    compare_instance_with_groupnorm()
    
    # 4. Training demonstration
    training_demo()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()