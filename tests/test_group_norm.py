"""
Tests for GroupNorm layer.
"""

import numpy as np
import nanotorch as nt
from nanotorch.nn import GroupNorm, group_norm


def test_groupnorm_creation():
    """Test GroupNorm initialization."""
    # Default parameters
    gn = GroupNorm(2, 6)  # 6 channels, 2 groups
    assert gn.num_groups == 2
    assert gn.num_channels == 6
    assert gn.eps == 1e-5
    assert gn.affine == True
    assert gn.training == True
    
    # Check parameters
    if gn.affine:
        assert gn.gamma is not None
        assert gn.beta is not None
        assert gn.gamma.shape == (6,)
        assert gn.beta.shape == (6,)
        assert gn.gamma.requires_grad == True
        assert gn.beta.requires_grad == True
        # gamma should be initialized to ones
        assert np.allclose(gn.gamma.data, 1.0)
        # beta should be initialized to zeros
        assert np.allclose(gn.beta.data, 0.0)
    
    # Different group configuration
    gn2 = GroupNorm(3, 12)  # 12 channels, 3 groups
    assert gn2.num_groups == 3
    assert gn2.num_channels == 12
    assert gn2.affine == True
    
    # Custom parameters
    gn3 = GroupNorm(1, 8, eps=1e-3, affine=False)
    assert gn3.num_groups == 1
    assert gn3.num_channels == 8
    assert gn3.eps == 1e-3
    assert gn3.affine == False
    assert gn3.gamma is None
    assert gn3.beta is None
    
    print("✓ test_groupnorm_creation passed")


def test_groupnorm_forward_2d():
    """Test GroupNorm forward pass with 2D input (N, C)."""
    gn = GroupNorm(2, 6)
    x = nt.Tensor.randn((4, 6), requires_grad=True)
    
    # Forward pass
    out = gn(x)
    
    # Check output shape
    assert out.shape == x.shape
    
    # Check normalization per group
    # Reshape to (N, G, C/G) and check each group normalized
    N, C = x.shape
    G = gn.num_groups
    D = C // G
    
    x_data = x.data.reshape(N, G, D)
    out_data = out.data.reshape(N, G, D)
    
    for n in range(N):
        for g in range(G):
            group_mean = out_data[n, g].mean()
            group_std = np.sqrt(out_data[n, g].var(ddof=0))
            assert abs(group_mean) < 1e-5
            assert abs(group_std - 1.0) < 1e-3
    
    print("✓ test_groupnorm_forward_2d passed")


def test_groupnorm_forward_4d():
    """Test GroupNorm forward pass with 4D input (N, C, H, W)."""
    gn = GroupNorm(2, 8)
    x = nt.Tensor.randn((2, 8, 10, 10), requires_grad=True)
    
    # Forward pass
    out = gn(x)
    
    # Check output shape
    assert out.shape == x.shape
    
    # Check normalization per group across spatial dimensions
    N, C, H, W = x.shape
    G = gn.num_groups
    D = C // G
    
    x_data = x.data.reshape(N, G, D, H, W)
    out_data = out.data.reshape(N, G, D, H, W)
    
    for n in range(N):
        for g in range(G):
            group_mean = out_data[n, g].mean()
            group_std = np.sqrt(out_data[n, g].var(ddof=0))
            assert abs(group_mean) < 1e-5
            assert abs(group_std - 1.0) < 1e-3
    
    print("✓ test_groupnorm_forward_4d passed")


def test_groupnorm_gradient_flow():
    """Test that gradients flow through GroupNorm."""
    gn = GroupNorm(2, 6)
    x = nt.Tensor.randn((3, 6, 5, 5), requires_grad=True)
    
    # Forward pass
    out = gn(x)
    
    # Compute loss and backward
    loss = out.sum()
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None
    assert x.grad.shape == x.shape
    
    # Check parameter gradients if affine=True
    if gn.affine:
        assert gn.gamma is not None
        assert gn.beta is not None
        assert gn.gamma.grad is not None
        assert gn.beta.grad is not None
        assert gn.gamma.grad.shape == gn.gamma.shape
        assert gn.beta.grad.shape == gn.beta.shape
    
    print("✓ test_groupnorm_gradient_flow passed")


def test_groupnorm_gradient_correctness():
    """Test gradient correctness using finite differences."""
    # Use simpler case for finite differences
    gn = GroupNorm(2, 4, eps=1e-5, affine=True)
    x = nt.Tensor.randn((2, 4, 3, 3), requires_grad=True)
    
    # Forward pass
    out = gn(x)
    
    # Compute loss and backward
    loss = out.sum()
    loss.backward()
    
    # Finite difference check for input gradient
    eps = 1e-2  # Larger epsilon for float32
    x_np = x.data.copy()
    grad_numerical = np.zeros_like(x_np)
    
    # Compute numerical gradient
    it = np.nditer(x_np, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original = x_np[idx]
        
        # f(x + eps)
        x_np[idx] = original + eps
        x_plus = nt.Tensor(x_np.copy(), requires_grad=True)
        out_plus = gn(x_plus)
        loss_plus = out_plus.sum()
        
        # f(x - eps)
        x_np[idx] = original - eps
        x_minus = nt.Tensor(x_np.copy(), requires_grad=True)
        out_minus = gn(x_minus)
        loss_minus = out_minus.sum()
        
        # Numerical gradient
        grad_numerical[idx] = (loss_plus.data - loss_minus.data) / (2 * eps)
        
        # Restore original
        x_np[idx] = original
        it.iternext()
    
    # Compare analytical and numerical gradients
    assert x.grad is not None
    grad_analytic = x.grad.data
    diff = np.abs(grad_analytic - grad_numerical).max()
    assert diff < 1e-3, f"Gradient mismatch: max diff = {diff}"
    
    print("✓ test_groupnorm_gradient_correctness passed")


def test_groupnorm_no_affine():
    """Test GroupNorm without affine transformation."""
    gn = GroupNorm(2, 8, affine=False)
    x = nt.Tensor.randn((3, 8, 5, 5), requires_grad=True)
    
    # Forward pass
    out = gn(x)
    
    # Check output is normalized
    N, C, H, W = x.shape
    G = gn.num_groups
    D = C // G
    
    out_data = out.data.reshape(N, G, D, H, W)
    for n in range(N):
        for g in range(G):
            group_mean = out_data[n, g].mean()
            group_std = np.sqrt(out_data[n, g].var(ddof=0))
            assert abs(group_mean) < 1e-5
            assert abs(group_std - 1.0) < 1e-3
    
    # Gradient flow should still work
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    
    # No parameter gradients since affine=False
    assert gn.gamma is None
    assert gn.beta is None
    
    print("✓ test_groupnorm_no_affine passed")


def test_groupnorm_invalid_input():
    """Test GroupNorm with invalid input shapes."""
    # num_channels not divisible by num_groups
    try:
        gn = GroupNorm(3, 7)  # 7 not divisible by 3
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be divisible" in str(e)
    
    # Input with insufficient dimensions
    gn = GroupNorm(1, 4)
    x = nt.Tensor.randn((4,))  # 1D, needs at least 2D
    try:
        out = gn(x)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "expects at least 2D" in str(e)
    
    # Wrong number of channels
    gn = GroupNorm(2, 6)
    x = nt.Tensor.randn((2, 8, 5, 5))  # 8 channels, expected 6
    try:
        out = gn(x)
    except Exception:
        pass
    
    print("✓ test_groupnorm_invalid_input passed")


def test_groupnorm_special_cases():
    """Test GroupNorm special cases: G=1 (LayerNorm) and G=C (InstanceNorm)."""
    # G=1 should normalize across all channels (like LayerNorm)
    gn = GroupNorm(1, 8)
    x = nt.Tensor.randn((2, 8, 5, 5), requires_grad=True)
    out = gn(x)
    
    # Check each sample normalized across all channels and spatial
    for n in range(x.shape[0]):
        sample_mean = out.data[n].mean()
        sample_std = np.sqrt(out.data[n].var(ddof=0))
        assert abs(sample_mean) < 1e-5
        assert abs(sample_std - 1.0) < 1e-3
    
    # G=C should normalize per channel (like InstanceNorm)
    gn = GroupNorm(8, 8)  # 8 groups, 8 channels
    x = nt.Tensor.randn((2, 8, 5, 5), requires_grad=True)
    out = gn(x)
    
    # Check each channel normalized independently
    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            channel_mean = out.data[n, c].mean()
            channel_std = np.sqrt(out.data[n, c].var(ddof=0))
            assert abs(channel_mean) < 1e-5
            assert abs(channel_std - 1.0) < 1e-3
    
    print("✓ test_groupnorm_special_cases passed")


def test_functional_group_norm():
    """Test functional group_norm interface."""
    x = nt.Tensor.randn((2, 6, 4, 4), requires_grad=True)
    weight = nt.Tensor.ones((6,), requires_grad=True)
    bias = nt.Tensor.zeros((6,), requires_grad=True)
    
    # With weight and bias
    out = group_norm(x, num_groups=2, weight=weight, bias=bias)
    assert out.shape == x.shape
    
    # Check normalization
    out_data = out.data.reshape(2, 2, 3, 4, 4)
    for n in range(2):
        for g in range(2):
            group_mean = out_data[n, g].mean()
            group_std = np.sqrt(out_data[n, g].var(ddof=0))
            assert abs(group_mean) < 1e-5
            assert abs(group_std - 1.0) < 1e-3
    
    # Without weight and bias
    out2 = group_norm(x, num_groups=2, weight=None, bias=None)
    assert out2.shape == x.shape
    
    # Gradient flow
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None
    
    print("✓ test_functional_group_norm passed")


if __name__ == "__main__":
    test_groupnorm_creation()
    test_groupnorm_forward_2d()
    test_groupnorm_forward_4d()
    test_groupnorm_gradient_flow()
    test_groupnorm_gradient_correctness()
    test_groupnorm_no_affine()
    test_groupnorm_invalid_input()
    test_groupnorm_special_cases()
    test_functional_group_norm()
    print("\nAll GroupNorm tests passed!")