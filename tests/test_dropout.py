"""
Tests for Dropout layer.
"""

import numpy as np
import nanotorch as nt
from nanotorch.nn import Dropout
from nanotorch.utils import manual_seed


def test_dropout_creation():
    """Test Dropout initialization."""
    # Default parameters
    do = Dropout()
    assert do.p == 0.5
    assert do.inplace == False
    assert do.training == True

    # Custom parameters
    do2 = Dropout(p=0.3, inplace=True)
    assert do2.p == 0.3
    assert do2.inplace == True

    # Edge cases
    do3 = Dropout(p=0.0)
    assert do3.p == 0.0

    do4 = Dropout(p=1.0)
    assert do4.p == 1.0

    # Invalid probability should raise ValueError
    try:
        Dropout(p=-0.1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        Dropout(p=1.1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("✓ test_dropout_creation passed")


def test_dropout_forward_train():
    """Test forward pass in training mode."""
    manual_seed(42)  # For reproducibility

    # Create dropout with p=0.5
    do = Dropout(p=0.5)
    assert do.training == True

    # Input tensor
    x = nt.Tensor.ones((10, 20), requires_grad=False)

    # Forward pass
    out = do(x)

    # Check shape preserved
    assert out.shape == x.shape

    # Check that approximately half the elements are zero (due to p=0.5)
    # Allow some tolerance due to randomness
    out_data = out.data
    zero_count = np.sum(out_data == 0.0)
    total_elements = out_data.size
    zero_ratio = zero_count / total_elements

    # Expected zero ratio is p=0.5, allow +/- 0.1 tolerance
    assert abs(zero_ratio - 0.5) < 0.1, f"Zero ratio {zero_ratio} not near 0.5"

    # Check that non-zero elements are scaled by 1/(1-p) = 2.0
    non_zero_mask = out_data != 0.0
    if np.any(non_zero_mask):
        non_zero_values = out_data[non_zero_mask]
        # Should be exactly 2.0 (since input is all ones)
        assert np.allclose(
            non_zero_values, 2.0
        ), f"Non-zero values not scaled correctly: {non_zero_values[:5]}"

    print("✓ test_dropout_forward_train passed")


def test_dropout_forward_eval():
    """Test forward pass in evaluation mode."""
    do = Dropout(p=0.5)
    do.eval()
    assert do.training == False

    # Input tensor
    x = nt.Tensor.randn((5, 3, 7), requires_grad=False)
    x_data = x.data.copy()

    # Forward pass in eval mode
    out = do(x)

    # In eval mode, dropout should be identity
    assert out.shape == x.shape
    assert np.allclose(out.data, x_data)

    print("✓ test_dropout_forward_eval passed")


def test_dropout_gradient_flow():
    """Test gradient computation through Dropout."""
    manual_seed(123)

    do = Dropout(p=0.3)
    x = nt.Tensor.randn((4, 5), requires_grad=True)
    x_data = x.data.copy()

    # Forward pass
    out = do(x)

    # Create a loss (sum of outputs)
    loss = out.sum()

    # Backward pass
    loss.backward()

    # Check gradients
    assert x.grad is not None
    assert x.grad.shape == x.shape

    # Gradient should be scaled mask * grad_output (grad_output is 1 everywhere)
    # Since loss = sum(out), grad_output = 1 for each element
    # So gradient = mask * scale where mask is 0 or 1
    # We can verify that gradient values are either 0 or scale
    grad_data = x.grad.data
    scale = 1.0 / (1.0 - 0.3)  # scale = 1/(1-p)

    # Count zeros vs non-zeros
    zero_mask = grad_data == 0.0
    non_zero_mask = ~zero_mask

    # Non-zero gradients should be approximately scale (allow small floating error)
    if np.any(non_zero_mask):
        assert np.allclose(grad_data[non_zero_mask], scale, rtol=1e-5)

    # Ratio of zeros should be approximately p
    zero_ratio = np.mean(zero_mask)
    assert (
        abs(zero_ratio - 0.3) < 0.1
    ), f"Zero gradient ratio {zero_ratio} not near p=0.3"

    print("✓ test_dropout_gradient_flow passed")


def test_dropout_p_zero():
    """Test dropout with p=0 (no dropout)."""
    do = Dropout(p=0.0)
    x = nt.Tensor.randn((3, 4, 5), requires_grad=False)

    # In training mode
    out_train = do(x)
    assert np.allclose(out_train.data, x.data)

    # In eval mode
    do.eval()
    out_eval = do(x)
    assert np.allclose(out_eval.data, x.data)

    print("✓ test_dropout_p_zero passed")


def test_dropout_p_one():
    """Test dropout with p=1 (all zeros)."""
    manual_seed(456)

    do = Dropout(p=1.0)
    x = nt.Tensor.randn((2, 3), requires_grad=False)

    # In training mode, all outputs should be zero
    out = do(x)
    assert np.all(out.data == 0.0)

    # In eval mode, should be identity
    do.eval()
    out_eval = do(x)
    assert np.allclose(out_eval.data, x.data)

    print("✓ test_dropout_p_one passed")


def test_dropout_different_dimensions():
    """Test dropout with different input dimensions."""
    manual_seed(789)

    p = 0.2
    do = Dropout(p=p)

    # Test 2D input (batch, features)
    x2d = nt.Tensor.randn((10, 20), requires_grad=False)
    out2d = do(x2d)
    assert out2d.shape == x2d.shape

    # Test 3D input (batch, sequence, features)
    x3d = nt.Tensor.randn((4, 5, 6), requires_grad=False)
    out3d = do(x3d)
    assert out3d.shape == x3d.shape

    # Test 4D input (batch, channels, height, width)
    x4d = nt.Tensor.randn((2, 3, 8, 8), requires_grad=False)
    out4d = do(x4d)
    assert out4d.shape == x4d.shape

    print("✓ test_dropout_different_dimensions passed")


def test_dropout_functional():
    """Test functional dropout interface."""
    manual_seed(999)

    from nanotorch.nn.dropout import dropout

    x = nt.Tensor.randn((5, 5), requires_grad=False)

    # Training mode
    out_train = dropout(x, p=0.4, training=True)
    assert out_train.shape == x.shape

    # Evaluation mode
    out_eval = dropout(x, p=0.4, training=False)
    assert np.allclose(out_eval.data, x.data)

    print("✓ test_dropout_functional passed")


def run_all_tests():
    """Run all dropout tests."""
    test_dropout_creation()
    test_dropout_forward_train()
    test_dropout_forward_eval()
    test_dropout_gradient_flow()
    test_dropout_p_zero()
    test_dropout_p_one()
    test_dropout_different_dimensions()
    test_dropout_functional()
    print("\n✅ All Dropout tests passed!")


if __name__ == "__main__":
    run_all_tests()
