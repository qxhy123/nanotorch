"""
Tests for InstanceNorm layer.
"""

import numpy as np
import nanotorch as nt
from nanotorch.nn import InstanceNorm2d, instance_norm, GroupNorm
from nanotorch import Tensor


def test_instancenorm_creation():
    """Test InstanceNorm2d initialization."""
    # Default parameters (affine=False)
    inorm = InstanceNorm2d(6)  # 6 channels
    assert inorm.num_features == 6
    assert inorm.eps == 1e-5
    assert inorm.affine == False  # Default is False (different from BatchNorm!)
    assert inorm.training == True
    
    # Check parameters (should be None when affine=False)
    assert inorm.gamma is None
    assert inorm.beta is None
    
    # With affine parameters
    inorm2 = InstanceNorm2d(8, affine=True)
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
    
    # Custom epsilon
    inorm3 = InstanceNorm2d(4, eps=1e-3, affine=False)
    assert inorm3.num_features == 4
    assert inorm3.eps == 1e-3
    assert inorm3.affine == False
    
    print("✓ test_instancenorm_creation passed")





def test_instancenorm_forward_4d():
    """Test InstanceNorm2d forward pass with 4D input (N, C, H, W)."""
    # Standard 4D input
    inorm = InstanceNorm2d(6, affine=False)
    x = Tensor.randn((3, 6, 10, 10))  # (N, C, H, W)
    
    output = inorm(x)
    assert output.shape == x.shape
    
    # With affine parameters
    inorm2 = InstanceNorm2d(6, affine=True)
    output2 = inorm2(x)
    assert output2.shape == x.shape
    
    # Verify mean ~0 and std ~1 per channel per sample
    # Since affine=False, output should be normalized
    output_np = output.data
    for n in range(3):  # batch
        for c in range(6):  # channel
            channel_data = output_np[n, c]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            assert abs(mean) < 1e-5, f"Mean not zero: {mean}"
            assert abs(std - 1.0) < 1e-5, f"Std not 1: {std}"
    
    print("✓ test_instancenorm_forward_4d passed")


def test_instancenorm_gradient_flow():
    """Test that gradients flow through InstanceNorm."""
    inorm = InstanceNorm2d(4, affine=True)
    x = Tensor.randn((2, 4, 5, 5), requires_grad=True)
    
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
    
    print("✓ test_instancenorm_gradient_flow passed")


def test_instancenorm_gradient_correctness():
    """Test gradient correctness using finite differences."""
    np.random.seed(42)  # For reproducibility
    
    # Test with affine=False (simpler)
    inorm = InstanceNorm2d(3, affine=False, eps=1e-5)
    x = Tensor.randn((2, 3, 4, 4), requires_grad=True)
    
    # Random loss weights to ensure non-zero gradient
    loss_weights = np.random.randn(2, 3, 4, 4).astype(np.float32)
    
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
    
    print("✓ test_instancenorm_gradient_correctness passed")


def test_instancenorm_no_affine():
    """Test InstanceNorm without affine parameters (default)."""
    inorm = InstanceNorm2d(4, affine=False)  # Default
    x = Tensor.randn((2, 4, 6, 6))
    
    output = inorm(x)
    assert output.shape == x.shape
    
    # Without affine, operation should be deterministic
    # Running twice should give same result
    output2 = inorm(x)
    assert np.allclose(output.data, output2.data)
    
    # Parameters should be None
    assert inorm.gamma is None
    assert inorm.beta is None
    
    print("✓ test_instancenorm_no_affine passed")


def test_instancenorm_invalid_input():
    """Test error handling for invalid inputs."""
    inorm = InstanceNorm2d(4)
    
    # Wrong number of dimensions
    try:
        x = Tensor.randn((4,))  # 1D
        inorm(x)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "expects" in str(e) and "input" in str(e)
    
    # Wrong number of channels
    try:
        x = Tensor.randn((2, 5, 10, 10))  # 5 channels, expected 4
        inorm(x)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "channels" in str(e).lower()
    
    print("✓ test_instancenorm_invalid_input passed")


def test_instancenorm_vs_groupnorm():
    """Test that InstanceNorm is equivalent to GroupNorm with num_groups=num_channels."""
    # Create equivalent layers
    inorm = InstanceNorm2d(6, affine=True, eps=1e-5)
    gnorm = GroupNorm(6, 6, affine=True, eps=1e-5)  # num_groups=num_channels
    
    # Use same random weights for fair comparison
    weight = Tensor.randn((6,), requires_grad=True)
    bias = Tensor.randn((6,), requires_grad=True)
    
    inorm.gamma = weight
    inorm.beta = bias
    gnorm.gamma = weight
    gnorm.beta = bias
    
    # Same input
    x = Tensor.randn((2, 6, 8, 8))
    
    # Forward passes should give same result
    output_inorm = inorm(x)
    output_gnorm = gnorm(x)
    
    # Allow small numerical differences
    assert np.allclose(output_inorm.data, output_gnorm.data, rtol=1e-5, atol=1e-6), \
        "InstanceNorm should equal GroupNorm with num_groups=num_channels"
    
    print("✓ test_instancenorm_vs_groupnorm passed")


def test_functional_instance_norm():
    """Test functional instance_norm interface."""
    x = Tensor.randn((3, 4, 5, 5))
    
    # Without affine parameters
    output1 = instance_norm(x, eps=1e-5)
    assert output1.shape == x.shape
    
    # With affine parameters
    weight = Tensor.ones((4,), requires_grad=True)
    bias = Tensor.zeros((4,), requires_grad=True)
    output2 = instance_norm(x, weight=weight, bias=bias, eps=1e-5)
    assert output2.shape == x.shape
    
    # Test gradient flow
    x2 = Tensor.randn((2, 3, 4, 4), requires_grad=True)
    weight2 = Tensor.randn((3,), requires_grad=True)
    output3 = instance_norm(x2, weight=weight2)
    loss = output3.sum()
    loss.backward()
    
    assert x2.grad is not None
    assert weight2.grad is not None
    
    print("✓ test_functional_instance_norm passed")


if __name__ == "__main__":
    test_instancenorm_creation()
    test_instancenorm_forward_2d()
    test_instancenorm_forward_4d()
    test_instancenorm_gradient_flow()
    test_instancenorm_gradient_correctness()
    test_instancenorm_no_affine()
    test_instancenorm_invalid_input()
    test_instancenorm_vs_groupnorm()
    test_functional_instance_norm()
    print("\n✅ All InstanceNorm tests passed!")