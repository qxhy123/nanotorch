"""
Tests for InstanceNorm1d and InstanceNorm3d layers.
"""

import numpy as np
import nanotorch as nt
from nanotorch.nn import InstanceNorm1d, InstanceNorm3d, GroupNorm
from nanotorch import Tensor


def test_instancenorm1d_creation():
    """Test InstanceNorm1d initialization."""
    # Default parameters (affine=False, track_running_stats=False)
    inorm = InstanceNorm1d(6)  # 6 channels
    assert inorm.num_features == 6
    assert inorm.eps == 1e-5
    assert inorm.affine == False  # Default is False (different from BatchNorm!)
    assert inorm.track_running_stats == False  # Default is False
    assert inorm.training == True
    
    # Check parameters (should be None when affine=False)
    assert inorm.gamma is None
    assert inorm.beta is None
    
    # With affine parameters
    inorm2 = InstanceNorm1d(8, affine=True)
    assert inorm2.num_features == 8
    assert inorm2.affine == True
    assert inorm2.gamma is not None
    assert inorm2.beta is not None
    assert inorm2.gamma.shape == (8,)
    assert inorm2.beta.shape == (8,)
    assert inorm2.gamma.requires_grad == True
    assert inorm2.beta.requires_grad == True
    # gamma should be initialized to ones
    assert np.allclose(inorm2.gamma.data, 1.0)
    # beta should be initialized to zeros
    assert np.allclose(inorm2.beta.data, 0.0)
    
    # With track_running_stats=True (unusual but allowed)
    inorm3 = InstanceNorm1d(4, track_running_stats=True)
    assert inorm3.track_running_stats == True
    assert inorm3.running_mean is not None
    assert inorm3.running_var is not None
    assert inorm3.running_mean.shape == (4,)
    assert inorm3.running_var.shape == (4,)
    assert np.allclose(inorm3.running_mean.data, 0.0)
    assert np.allclose(inorm3.running_var.data, 1.0)
    
    # Custom epsilon
    inorm4 = InstanceNorm1d(4, eps=1e-3, affine=False)
    assert inorm4.num_features == 4
    assert inorm4.eps == 1e-3
    assert inorm4.affine == False
    
    print("✓ test_instancenorm1d_creation passed")


def test_instancenorm1d_forward_2d():
    """Test InstanceNorm1d forward pass with 2D input (C, L)."""
    inorm = InstanceNorm1d(6, affine=False)
    x = Tensor.randn((6, 10))  # (C, L)
    
    output = inorm(x)
    assert output.shape == x.shape
    
    # Verify mean ~0 and std ~1 per channel (since affine=False)
    output_np = output.data
    for c in range(6):
        channel_data = output_np[c]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        assert abs(mean) < 1e-4, f"Mean not zero: {mean}"
        assert abs(std - 1.0) < 1e-4, f"Std not 1: {std}"
    
    print("✓ test_instancenorm1d_forward_2d passed")


def test_instancenorm1d_forward_3d():
    """Test InstanceNorm1d forward pass with 3D input (N, C, L)."""
    inorm = InstanceNorm1d(6, affine=False)
    x = Tensor.randn((3, 6, 10))  # (N, C, L)
    
    output = inorm(x)
    assert output.shape == x.shape
    
    # Verify mean ~0 and std ~1 per channel per sample
    output_np = output.data
    for n in range(3):
        for c in range(6):
            channel_data = output_np[n, c]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            assert abs(mean) < 1e-4, f"Mean not zero: {mean}"
            assert abs(std - 1.0) < 1e-4, f"Std not 1: {std}"
    
    print("✓ test_instancenorm1d_forward_3d passed")


def test_instancenorm1d_affine_true():
    """Test InstanceNorm1d with affine parameters."""
    inorm = InstanceNorm1d(4, affine=True)
    x = Tensor.randn((2, 4, 8))
    
    output = inorm(x)
    assert output.shape == x.shape
    
    # With affine=True, output is not necessarily normalized to mean 0, std 1
    # but gradients should flow
    output_np = output.data
    # Ensure no NaNs
    assert not np.any(np.isnan(output_np))
    
    print("✓ test_instancenorm1d_affine_true passed")


def test_instancenorm1d_gradient_flow():
    """Test that gradients flow through InstanceNorm1d."""
    inorm = InstanceNorm1d(4, affine=True)
    x = Tensor.randn((2, 4, 5), requires_grad=True)
    
    # Forward pass
    output = inorm(x)
    
    # Create a loss and backward
    loss = output.sum()
    loss.backward()
    
    # Gradients should exist
    assert x.grad is not None
    assert x.grad.shape == x.shape
    
    # Affine parameter gradients should exist
    assert inorm.gamma.grad is not None
    assert inorm.beta.grad is not None
    assert inorm.gamma.grad.shape == inorm.gamma.shape
    assert inorm.beta.grad.shape == inorm.beta.shape
    
    print("✓ test_instancenorm1d_gradient_flow passed")


def test_instancenorm1d_gradient_correctness():
    """Test gradient correctness using finite differences."""
    np.random.seed(42)  # For reproducibility
    
    # Test with affine=False (simpler)
    inorm = InstanceNorm1d(3, affine=False, eps=1e-5)
    x = Tensor.randn((2, 3, 4), requires_grad=True)
    
    # Random loss weights to ensure non-zero gradient
    loss_weights = np.random.randn(2, 3, 4).astype(np.float32)
    
    # Forward pass with weighted sum (non-zero loss)
    output = inorm(x)
    loss = (output * Tensor(loss_weights)).sum()
    loss.backward()
    
    analytic_grad = x.grad.data.copy()
    
    # Compute numerical gradient using finite differences
    numerical_grad = np.zeros_like(x.data)
    eps = 1e-2  # Larger epsilon for float32 stability (matches GroupNorm test)
    
    it = np.nditer(x.data, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original = x.data[idx]
        
        # f(x + eps)
        x.data[idx] = original + eps
        output_plus = inorm(x)
        loss_plus = (output_plus * Tensor(loss_weights)).sum().item()
        
        # f(x - eps)
        x.data[idx] = original - eps
        output_minus = inorm(x)
        loss_minus = (output_minus * Tensor(loss_weights)).sum().item()
        
        # Reset
        x.data[idx] = original
        
        # Central difference
        numerical_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        
        it.iternext()
    
    # Compare gradients (allow larger tolerance for float32)
    max_diff = np.abs(analytic_grad - numerical_grad).max()
    assert max_diff < 1e-3, f"Gradient mismatch: max_diff={max_diff}"
    
    print("✓ test_instancenorm1d_gradient_correctness passed")


def test_instancenorm1d_invalid_input():
    """Test error handling for invalid inputs."""
    inorm = InstanceNorm1d(4)
    
    # Wrong number of dimensions (1D)
    try:
        x = Tensor.randn((4,))  # 1D
        inorm(x)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "expects 2D or 3D input" in str(e)
    
    # Wrong number of channels
    try:
        x = Tensor.randn((2, 5, 10))  # 5 channels, expected 4
        inorm(x)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "channels" in str(e).lower()
    
    print("✓ test_instancenorm1d_invalid_input passed")


def test_instancenorm3d_creation():
    """Test InstanceNorm3d initialization."""
    # Default parameters (affine=False, track_running_stats=False)
    inorm = InstanceNorm3d(6)  # 6 channels
    assert inorm.num_features == 6
    assert inorm.eps == 1e-5
    assert inorm.affine == False  # Default is False
    assert inorm.track_running_stats == False  # Default is False
    assert inorm.training == True
    
    # Check parameters (should be None when affine=False)
    assert inorm.gamma is None
    assert inorm.beta is None
    
    # With affine parameters
    inorm2 = InstanceNorm3d(8, affine=True)
    assert inorm2.num_features == 8
    assert inorm2.affine == True
    assert inorm2.gamma is not None
    assert inorm2.beta is not None
    assert inorm2.gamma.shape == (8,)
    assert inorm2.beta.shape == (8,)
    assert inorm2.gamma.requires_grad == True
    assert inorm2.beta.requires_grad == True
    # gamma should be initialized to ones
    assert np.allclose(inorm2.gamma.data, 1.0)
    # beta should be initialized to zeros
    assert np.allclose(inorm2.beta.data, 0.0)
    
    # With track_running_stats=True
    inorm3 = InstanceNorm3d(4, track_running_stats=True)
    assert inorm3.track_running_stats == True
    assert inorm3.running_mean is not None
    assert inorm3.running_var is not None
    assert inorm3.running_mean.shape == (4,)
    assert inorm3.running_var.shape == (4,)
    assert np.allclose(inorm3.running_mean.data, 0.0)
    assert np.allclose(inorm3.running_var.data, 1.0)
    
    # Custom epsilon
    inorm4 = InstanceNorm3d(4, eps=1e-3, affine=False)
    assert inorm4.num_features == 4
    assert inorm4.eps == 1e-3
    assert inorm4.affine == False
    
    print("✓ test_instancenorm3d_creation passed")


def test_instancenorm3d_forward_4d():
    """Test InstanceNorm3d forward pass with 4D input (C, D, H, W)."""
    inorm = InstanceNorm3d(6, affine=False)
    x = Tensor.randn((6, 5, 6, 7))  # (C, D, H, W)
    
    output = inorm(x)
    assert output.shape == x.shape
    
    # Verify mean ~0 and std ~1 per channel (since affine=False)
    output_np = output.data
    for c in range(6):
        channel_data = output_np[c]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        assert abs(mean) < 1e-4, f"Mean not zero: {mean}"
        assert abs(std - 1.0) < 1e-4, f"Std not 1: {std}"
    
    print("✓ test_instancenorm3d_forward_4d passed")


def test_instancenorm3d_forward_5d():
    """Test InstanceNorm3d forward pass with 5D input (N, C, D, H, W)."""
    inorm = InstanceNorm3d(6, affine=False)
    x = Tensor.randn((3, 6, 5, 6, 7))  # (N, C, D, H, W)
    
    output = inorm(x)
    assert output.shape == x.shape
    
    # Verify mean ~0 and std ~1 per channel per sample
    output_np = output.data
    for n in range(3):
        for c in range(6):
            channel_data = output_np[n, c]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            assert abs(mean) < 1e-4, f"Mean not zero: {mean}"
            assert abs(std - 1.0) < 1e-4, f"Std not 1: {std}"
    
    print("✓ test_instancenorm3d_forward_5d passed")


def test_instancenorm3d_affine_true():
    """Test InstanceNorm3d with affine parameters."""
    inorm = InstanceNorm3d(4, affine=True)
    x = Tensor.randn((2, 4, 3, 4, 5))
    
    output = inorm(x)
    assert output.shape == x.shape
    
    # With affine=True, output is not necessarily normalized to mean 0, std 1
    # but gradients should flow
    output_np = output.data
    assert not np.any(np.isnan(output_np))
    
    print("✓ test_instancenorm3d_affine_true passed")


def test_instancenorm3d_gradient_flow():
    """Test that gradients flow through InstanceNorm3d."""
    inorm = InstanceNorm3d(4, affine=True)
    x = Tensor.randn((2, 4, 3, 4, 5), requires_grad=True)
    
    # Forward pass
    output = inorm(x)
    
    # Create a loss and backward
    loss = output.sum()
    loss.backward()
    
    # Gradients should exist
    assert x.grad is not None
    assert x.grad.shape == x.shape
    
    # Affine parameter gradients should exist
    assert inorm.gamma.grad is not None
    assert inorm.beta.grad is not None
    assert inorm.gamma.grad.shape == inorm.gamma.shape
    assert inorm.beta.grad.shape == inorm.beta.shape
    
    print("✓ test_instancenorm3d_gradient_flow passed")


def test_instancenorm3d_gradient_correctness():
    """Test gradient correctness using finite differences."""
    np.random.seed(42)  # For reproducibility
    
    # Test with affine=False (simpler)
    inorm = InstanceNorm3d(2, affine=False, eps=1e-5)
    x = Tensor.randn((1, 2, 2, 3, 4), requires_grad=True)
    
    # Random loss weights to ensure non-zero gradient
    loss_weights = np.random.randn(1, 2, 2, 3, 4).astype(np.float32)
    
    # Forward pass with weighted sum (non-zero loss)
    output = inorm(x)
    loss = (output * Tensor(loss_weights)).sum()
    loss.backward()
    
    analytic_grad = x.grad.data.copy()
    
    # Compute numerical gradient using finite differences
    numerical_grad = np.zeros_like(x.data)
    eps = 1e-2  # Larger epsilon for float32 stability
    
    it = np.nditer(x.data, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original = x.data[idx]
        
        # f(x + eps)
        x.data[idx] = original + eps
        output_plus = inorm(x)
        loss_plus = (output_plus * Tensor(loss_weights)).sum().item()
        
        # f(x - eps)
        x.data[idx] = original - eps
        output_minus = inorm(x)
        loss_minus = (output_minus * Tensor(loss_weights)).sum().item()
        
        # Reset
        x.data[idx] = original
        
        # Central difference
        numerical_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        
        it.iternext()
    
    # Compare gradients (allow larger tolerance for float32)
    max_diff = np.abs(analytic_grad - numerical_grad).max()
    assert max_diff < 1e-3, f"Gradient mismatch: max_diff={max_diff}"
    
    print("✓ test_instancenorm3d_gradient_correctness passed")


def test_instancenorm3d_invalid_input():
    """Test error handling for invalid inputs."""
    inorm = InstanceNorm3d(4)
    
    # Wrong number of dimensions (3D)
    try:
        x = Tensor.randn((4, 3, 4))  # 3D
        inorm(x)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "expects 4D or 5D input" in str(e)
    
    # Wrong number of channels
    try:
        x = Tensor.randn((2, 5, 3, 4, 5))  # 5 channels, expected 4
        inorm(x)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "channels" in str(e).lower()
    
    print("✓ test_instancenorm3d_invalid_input passed")


def test_instancenorm_vs_groupnorm():
    """Test that InstanceNorm is equivalent to GroupNorm with num_groups=num_channels."""
    # Test for 1D
    inorm1d = InstanceNorm1d(6, affine=True, eps=1e-5)
    gnorm1d = GroupNorm(6, 6, affine=True, eps=1e-5)  # num_groups=num_channels
    
    # Use same random weights for fair comparison
    weight = Tensor.randn((6,), requires_grad=True)
    bias = Tensor.randn((6,), requires_grad=True)
    
    inorm1d.gamma = weight
    inorm1d.beta = bias
    gnorm1d.gamma = weight
    gnorm1d.beta = bias
    
    # Same input (3D)
    x1d = Tensor.randn((2, 6, 8))
    
    # Forward passes should give same result
    output_inorm = inorm1d(x1d)
    output_gnorm = gnorm1d(x1d)
    
    assert np.allclose(output_inorm.data, output_gnorm.data, rtol=1e-5, atol=1e-6), \
        "InstanceNorm1d should equal GroupNorm with num_groups=num_channels"
    
    # Test for 3D
    inorm3d = InstanceNorm3d(4, affine=True, eps=1e-5)
    gnorm3d = GroupNorm(4, 4, affine=True, eps=1e-5)
    
    weight2 = Tensor.randn((4,), requires_grad=True)
    bias2 = Tensor.randn((4,), requires_grad=True)
    
    inorm3d.gamma = weight2
    inorm3d.beta = bias2
    gnorm3d.gamma = weight2
    gnorm3d.beta = bias2
    
    x3d = Tensor.randn((2, 4, 3, 4, 5))
    
    output_inorm3d = inorm3d(x3d)
    output_gnorm3d = gnorm3d(x3d)
    
    assert np.allclose(output_inorm3d.data, output_gnorm3d.data, rtol=1e-5, atol=1e-6), \
        "InstanceNorm3d should equal GroupNorm with num_groups=num_channels"
    
    print("✓ test_instancenorm_vs_groupnorm passed")


def run_all_tests():
    """Run all instance norm 1D/3D tests."""
    test_instancenorm1d_creation()
    test_instancenorm1d_forward_2d()
    test_instancenorm1d_forward_3d()
    test_instancenorm1d_affine_true()
    test_instancenorm1d_gradient_flow()
    test_instancenorm1d_gradient_correctness()
    test_instancenorm1d_invalid_input()
    
    test_instancenorm3d_creation()
    test_instancenorm3d_forward_4d()
    test_instancenorm3d_forward_5d()
    test_instancenorm3d_affine_true()
    test_instancenorm3d_gradient_flow()
    test_instancenorm3d_gradient_correctness()
    test_instancenorm3d_invalid_input()
    
    test_instancenorm_vs_groupnorm()
    
    print("\n✅ All InstanceNorm1d/3D tests passed!")


if __name__ == "__main__":
    run_all_tests()