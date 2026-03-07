"""
Tests for numerical stability and validation features in nanotorch.

This module tests the numerical stability improvements and validation features
added to prevent silent failures.
"""

import numpy as np
import pytest
import nanotorch as nt


def test_division_by_zero_scalar():
    """Test that division by zero raises ValueError."""
    print("Testing division by zero (scalar)...")

    x = nt.Tensor(5.0, requires_grad=True)

    # Test division by zero scalar
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        result = x / 0

    with pytest.raises(ValueError, match="Cannot divide by zero"):
        result = x / 0.0

    print("  ✓ Division by zero scalar raises ValueError")


def test_division_by_zero_tensor():
    """Test that division by tensor containing zero raises ValueError."""
    print("Testing division by zero (tensor)...")

    x = nt.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = nt.Tensor([1.0, 0.0, 2.0], requires_grad=False)

    # Test division by tensor containing zero
    with pytest.raises(ValueError, match="Cannot divide by tensor containing zero values"):
        result = x / y

    print("  ✓ Division by tensor with zero raises ValueError")


def test_reverse_division_by_zero():
    """Test that reverse division by zero raises ValueError."""
    print("Testing reverse division by zero...")

    x = nt.Tensor([1.0, 0.0, 2.0], requires_grad=True)

    # Test reverse division by tensor containing zero
    with pytest.raises(ValueError, match="Cannot divide by tensor containing zero values"):
        result = 5.0 / x

    print("  ✓ Reverse division by zero raises ValueError")


def test_check_finite_nan():
    """Test check_finite method for NaN detection."""
    print("Testing check_finite for NaN...")

    # Test tensor without NaN
    x = nt.Tensor([1.0, 2.0, 3.0])
    assert x.check_finite(), "Tensor without NaN should pass finite check"
    print("  ✓ Finite tensor passes check")

    # Test tensor with NaN
    y = nt.Tensor([1.0, np.nan, 3.0])
    assert not y.check_finite(), "Tensor with NaN should fail finite check"
    print("  ✓ NaN tensor fails check")

    # Test with check_nan=False
    assert y.check_finite(check_nan=False, check_inf=True), "Should pass when NaN check disabled"
    print("  ✓ NaN check can be disabled")


def test_check_finite_inf():
    """Test check_finite method for Inf detection."""
    print("Testing check_finite for Inf...")

    # Test tensor with positive infinity
    x = nt.Tensor([1.0, np.inf, 3.0])
    assert not x.check_finite(), "Tensor with Inf should fail finite check"
    print("  ✓ Inf tensor fails check")

    # Test tensor with negative infinity
    y = nt.Tensor([1.0, -np.inf, 3.0])
    assert not y.check_finite(), "Tensor with -Inf should fail finite check"
    print("  ✓ -Inf tensor fails check")

    # Test with check_inf=False
    assert x.check_finite(check_nan=True, check_inf=False), "Should pass when Inf check disabled"
    print("  ✓ Inf check can be disabled")


def test_assert_finite():
    """Test assert_finite method."""
    print("Testing assert_finite...")

    # Test finite tensor (should not raise)
    x = nt.Tensor([1.0, 2.0, 3.0])
    x.assert_finite()
    print("  ✓ Finite tensor passes assertion")

    # Test NaN tensor (should raise)
    y = nt.Tensor([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="Tensor contains NaN or Inf values"):
        y.assert_finite()
    print("  ✓ NaN tensor raises ValueError")

    # Test Inf tensor (should raise)
    z = nt.Tensor([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="Tensor contains NaN or Inf values"):
        z.assert_finite()
    print("  ✓ Inf tensor raises ValueError")

    # Test custom error message
    with pytest.raises(ValueError, match="Custom error message"):
        y.assert_finite(msg="Custom error message")
    print("  ✓ Custom error message works")


def test_safe_operations():
    """Test that safe operations don't produce NaN/Inf unexpectedly."""
    print("Testing safe operations...")

    # Test safe division
    x = nt.Tensor([1.0, 2.0, 3.0])
    y = nt.Tensor([1.0, 2.0, 3.0])
    result = x / y
    assert result.check_finite(), "Safe division should produce finite values"
    print("  ✓ Safe division produces finite values")

    # Test log of positive numbers
    x = nt.Tensor([0.1, 1.0, 10.0])
    result = x.log()
    assert result.check_finite(), "Log of positive numbers should be finite"
    print("  ✓ Log of positive numbers is finite")

    # Test sqrt of positive numbers
    x = nt.Tensor([1.0, 4.0, 9.0])
    result = x.sqrt()
    assert result.check_finite(), "Sqrt of positive numbers should be finite"
    print("  ✓ Sqrt of positive numbers is finite")


def test_gradient_with_finite_check():
    """Test that gradients can be checked for finite values."""
    print("Testing gradient finite check...")

    x = nt.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x ** 2
    loss = y.sum()
    loss.backward()

    # Check gradient is finite
    assert x.grad.check_finite(), "Gradient should be finite for normal operations"
    print("  ✓ Normal gradient is finite")


if __name__ == "__main__":
    print("Running numerical stability tests...\n")

    # Run all tests
    test_division_by_zero_scalar()
    test_division_by_zero_tensor()
    test_reverse_division_by_zero()
    test_check_finite_nan()
    test_check_finite_inf()
    test_assert_finite()
    test_safe_operations()
    test_gradient_with_finite_check()

    print("\n" + "=" * 60)
    print("✓ All numerical stability tests passed!")
    print("=" * 60)
