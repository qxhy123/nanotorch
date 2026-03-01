"""
Tests for LayerNorm layer.
"""

import numpy as np
import nanotorch as nt
from nanotorch.nn import LayerNorm, layer_norm


def test_layernorm_creation():
    """Test LayerNorm initialization."""
    # Default parameters with int normalized_shape
    ln = LayerNorm(64)
    assert ln.normalized_shape == (64,)
    assert ln.eps == 1e-5
    assert ln.elementwise_affine == True
    assert ln.training == True
    
    # Check parameters
    if ln.elementwise_affine:
        assert ln.gamma is not None
        assert ln.beta is not None
        assert ln.gamma.shape == (64,)
        assert ln.beta.shape == (64,)
        assert ln.gamma.requires_grad == True
        assert ln.beta.requires_grad == True
        # gamma should be initialized to ones
        assert np.allclose(ln.gamma.data, 1.0)
        # beta should be initialized to zeros
        assert np.allclose(ln.beta.data, 0.0)
    
    # Tuple normalized_shape
    ln2 = LayerNorm((10, 20))
    assert ln2.normalized_shape == (10, 20)
    assert ln2.elementwise_affine == True
    if ln2.elementwise_affine:
        assert ln2.gamma.shape == (10, 20)
        assert ln2.beta.shape == (10, 20)
    
    # Custom parameters
    ln3 = LayerNorm(32, eps=1e-3, elementwise_affine=False)
    assert ln3.normalized_shape == (32,)
    assert ln3.eps == 1e-3
    assert ln3.elementwise_affine == False
    assert ln3.gamma is None
    assert ln3.beta is None
    
    print("✓ test_layernorm_creation passed")


def test_layernorm_forward_1d():
    """Test forward pass with 1D normalized_shape."""
    # Input shape: (batch, features)
    ln = LayerNorm(64)
    x = nt.Tensor.randn((8, 64), requires_grad=False)
    
    # Forward pass
    out = ln(x)
    
    # Check output shape
    assert out.shape == x.shape
    
    # Check that output is approximately normalized (mean ~0, std ~1)
    # For each batch element, check last dimension is normalized
    for i in range(x.shape[0]):
        out_slice = out.data[i]
        out_mean = out_slice.mean()
        out_var = out_slice.var(ddof=0)
        out_std = np.sqrt(out_var)
        assert abs(out_mean) < 1e-5, f"Output mean {out_mean} not near 0"
        assert abs(out_std - 1.0) < 1e-5, f"Output std {out_std} not near 1"
    
    print("✓ test_layernorm_forward_1d passed")


def test_layernorm_forward_2d():
    """Test forward pass with 2D normalized_shape."""
    # Input shape: (batch, channels, height, width)
    # Normalize over last 2 dimensions (height, width)
    ln = LayerNorm((10, 10))
    x = nt.Tensor.randn((4, 3, 10, 10), requires_grad=False)
    
    # Forward pass
    out = ln(x)
    
    # Check output shape
    assert out.shape == x.shape
    
    # Check that output is normalized over last 2 dimensions
    # For each batch and channel, check (height, width) is normalized
    for b in range(x.shape[0]):
        for c in range(x.shape[1]):
            out_slice = out.data[b, c]
            out_mean = out_slice.mean()
            out_var = out_slice.var(ddof=0)
            out_std = np.sqrt(out_var)
            assert abs(out_mean) < 1e-5, f"Output mean {out_mean} not near 0"
            assert abs(out_std - 1.0) < 1e-5, f"Output std {out_std} not near 1"
    
    print("✓ test_layernorm_forward_2d passed")


def test_layernorm_gradient_flow():
    """Test gradient flow through LayerNorm."""
    ln = LayerNorm(32)
    x = nt.Tensor.randn((4, 32), requires_grad=True)
    
    # Forward pass
    out = ln(x)
    
    # Create a simple loss and backward
    loss = out.sum()
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None
    assert x.grad.shape == x.shape
    
    # Check parameter gradients if affine=True
    if ln.elementwise_affine:
        assert ln.gamma.grad is not None
        assert ln.beta.grad is not None
        assert ln.gamma.grad.shape == ln.gamma.shape
        assert ln.beta.grad.shape == ln.beta.shape
    
    print("✓ test_layernorm_gradient_flow passed")


def test_layernorm_gradient_correctness():
    """Test gradient correctness with finite differences."""
    # Use smaller tensor for faster computation
    ln = LayerNorm(8, eps=1e-5)
    x = nt.Tensor.randn((2, 8), requires_grad=True)
    
    # Forward pass
    out = ln(x)
    
    # Create a simple loss
    loss = (out ** 2).sum()
    
    # Compute analytical gradient
    loss.backward()
    analytical_grad = x.grad.data.copy()
    
    # Compute numerical gradient using finite differences
    numerical_grad = np.zeros_like(x.data)
    eps = 1e-2
    
    it = np.nditer(x.data, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original = x.data[idx]
        
        # f(x + eps)
        x.data[idx] = original + eps
        out_plus = ln(x)
        loss_plus = (out_plus ** 2).sum().item()
        
        # f(x - eps)
        x.data[idx] = original - eps
        out_minus = ln(x)
        loss_minus = (out_minus ** 2).sum().item()
        
        # Finite difference
        numerical_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        
        # Restore original value
        x.data[idx] = original
        
        it.iternext()
    
    # Compare gradients
    max_diff = np.abs(analytical_grad - numerical_grad).max()
    assert max_diff < 1e-3, f"Gradient mismatch: max_diff={max_diff}"
    
    print("✓ test_layernorm_gradient_correctness passed")


def test_layernorm_no_affine():
    """Test LayerNorm without affine transformation."""
    ln = LayerNorm(16, elementwise_affine=False)
    x = nt.Tensor.randn((4, 16), requires_grad=True)
    
    # Forward pass
    out = ln(x)
    
    # Check output is normalized
    for i in range(x.shape[0]):
        out_slice = out.data[i]
        out_mean = out_slice.mean()
        out_std = np.sqrt(out_slice.var(ddof=0))
        assert abs(out_mean) < 1e-5
        assert abs(out_std - 1.0) < 2e-5
    
    # Gradient flow should still work
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    
    # No parameter gradients since elementwise_affine=False
    assert ln.gamma is None
    assert ln.beta is None
    
    print("✓ test_layernorm_no_affine passed")


def test_layernorm_invalid_input():
    """Test LayerNorm with invalid input shapes."""
    ln = LayerNorm((10, 10))
    
    # Input with wrong trailing dimensions
    x = nt.Tensor.randn((4, 3, 8, 8), requires_grad=False)
    
    try:
        out = ln(x)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Expected input with trailing dimensions" in str(e)
    
    # Input with correct dimensions should work
    x2 = nt.Tensor.randn((4, 3, 10, 10), requires_grad=False)
    out = ln(x2)
    assert out.shape == x2.shape
    
    print("✓ test_layernorm_invalid_input passed")


def test_functional_layer_norm():
    """Test functional layer_norm interface."""
    # Test with weight and bias
    x = nt.Tensor.randn((4, 16), requires_grad=True)
    weight = nt.Tensor.ones(16, requires_grad=True)
    bias = nt.Tensor.zeros(16, requires_grad=True)
    
    out = layer_norm(x, 16, weight=weight, bias=bias)
    assert out.shape == x.shape
    
    # Check normalization
    for i in range(x.shape[0]):
        out_slice = out.data[i]
        out_mean = out_slice.mean()
        out_std = np.sqrt(out_slice.var(ddof=0))
        assert abs(out_mean) < 1e-5
        assert abs(out_std - 1.0) < 1e-5
    
    # Test without weight and bias
    out2 = layer_norm(x, 16, weight=None, bias=None)
    assert out2.shape == x.shape
    
    # Gradient flow
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None
    
    print("✓ test_functional_layer_norm passed")


def run_all_tests():
    """Run all LayerNorm tests."""
    test_layernorm_creation()
    test_layernorm_forward_1d()
    test_layernorm_forward_2d()
    test_layernorm_gradient_flow()
    test_layernorm_gradient_correctness()
    test_layernorm_no_affine()
    test_layernorm_invalid_input()
    test_functional_layer_norm()
    print("\n✅ All LayerNorm tests passed!")


if __name__ == "__main__":
    run_all_tests()