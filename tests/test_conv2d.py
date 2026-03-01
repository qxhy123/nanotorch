"""
Tests for Conv2D layer.
"""

import numpy as np
import pytest
import nanotorch as nt
from nanotorch.nn import Conv2D
from nanotorch.autograd import Conv2DFunction
from nanotorch.utils import manual_seed


def test_conv2d_creation():
    """Test Conv2D initialization."""
    # Default parameters
    conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3)
    assert conv.in_channels == 3
    assert conv.out_channels == 16
    assert conv.kernel_size == (3, 3)
    assert conv.stride == 1
    assert conv.padding == 0
    assert conv.dilation == 1
    assert conv.bias is not None
    assert conv.weight.shape == (16, 3, 3, 3)
    assert conv.weight.requires_grad == True
    if conv.bias is not None:
        assert conv.bias.shape == (16, 1, 1)
        assert conv.bias.requires_grad == True

    # Custom parameters
    conv2 = Conv2D(
        in_channels=1,
        out_channels=8,
        kernel_size=(5, 7),
        stride=2,
        padding=1,
        dilation=2,
        bias=False,
    )
    assert conv2.in_channels == 1
    assert conv2.out_channels == 8
    assert conv2.kernel_size == (5, 7)
    assert conv2.stride == 2
    assert conv2.padding == 1
    assert conv2.dilation == 2
    assert conv2.bias is None
    assert conv2.weight.shape == (8, 1, 5, 7)

    print("✓ test_conv2d_creation passed")


def test_conv2d_forward():
    """Test forward pass of Conv2D."""
    manual_seed(42)

    # Create conv layer
    conv = Conv2D(in_channels=2, out_channels=4, kernel_size=3, padding=1)

    # Input tensor: batch=2, channels=2, height=5, width=5
    x = nt.Tensor.randn((2, 2, 5, 5), requires_grad=False)

    # Forward pass
    out = conv(x)

    # Check output shape
    assert out.shape == (2, 4, 5, 5)  # padding=1 preserves spatial dimensions

    # Check that output values are not all zeros
    assert not np.allclose(out.data, 0.0)

    # Test with stride=2
    conv2 = Conv2D(in_channels=2, out_channels=3, kernel_size=3, stride=2)
    out2 = conv2(x)
    expected_h = (5 - 3) // 2 + 1  # = 2
    expected_w = (5 - 3) // 2 + 1  # = 2
    assert out2.shape == (2, 3, expected_h, expected_w)

    print("✓ test_conv2d_forward passed")


def test_conv2d_gradient_flow():
    """Test gradient computation through Conv2D."""
    manual_seed(123)

    conv = Conv2D(in_channels=2, out_channels=3, kernel_size=3, padding=1)
    x = nt.Tensor.randn((1, 2, 4, 4), requires_grad=True)

    # Forward pass
    out = conv(x)

    # Create a loss (sum of outputs)
    loss = out.sum()

    # Backward pass
    loss.backward()

    # Check gradients
    assert x.grad is not None
    assert x.grad.shape == x.shape

    # Check weight gradients
    assert conv.weight.grad is not None
    assert conv.weight.grad.shape == conv.weight.shape

    # Check bias gradients
    if conv.bias is not None:
        assert conv.bias.grad is not None
        assert conv.bias.grad.shape == conv.bias.shape

    # Verify gradients are not all zero (some learning should happen)
    assert not np.allclose(x.grad.data, 0.0)
    assert not np.allclose(conv.weight.grad.data, 0.0)

    print("✓ test_conv2d_gradient_flow passed")


def test_conv2d_no_bias():
    """Test Conv2D without bias."""
    conv = Conv2D(in_channels=1, out_channels=2, kernel_size=3, bias=False)
    assert conv.bias is None

    x = nt.Tensor.randn((1, 1, 6, 6), requires_grad=False)
    out = conv(x)
    assert out.shape == (1, 2, 4, 4)

    # Ensure no bias parameter exists
    assert "bias" not in conv._parameters or conv._parameters["bias"] is None

    print("✓ test_conv2d_no_bias passed")


def test_conv2d_padding():
    """Test Conv2D with different padding values."""
    # Input size 5x5, kernel 3x3, padding 0 -> output 3x3
    conv1 = Conv2D(in_channels=1, out_channels=1, kernel_size=3, padding=0)
    x = nt.Tensor.randn((1, 1, 5, 5), requires_grad=False)
    out1 = conv1(x)
    assert out1.shape == (1, 1, 3, 3)

    # padding 1 -> output 5x5 (preserve size)
    conv2 = Conv2D(in_channels=1, out_channels=1, kernel_size=3, padding=1)
    out2 = conv2(x)
    assert out2.shape == (1, 1, 5, 5)

    # padding 2 -> output 7x7 (increase size)
    conv3 = Conv2D(in_channels=1, out_channels=1, kernel_size=3, padding=2)
    out3 = conv3(x)
    assert out3.shape == (1, 1, 7, 7)

    print("✓ test_conv2d_padding passed")


def test_conv2d_gradient_correctness():
    """Test Conv2D gradient correctness with finite differences."""
    np.random.seed(42)
    
    # Test a representative subset of configurations
    configs = [
        {'stride': 1, 'padding': 0, 'dilation': 1, 'bias': True},
        {'stride': 2, 'padding': 0, 'dilation': 1, 'bias': False},
        {'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True},
        {'stride': 1, 'padding': 0, 'dilation': 2, 'bias': False},
    ]
    
    N, C_in, C_out = 2, 3, 4
    H, W = 8, 8
    kernel_size = (3, 3)
    
    for config in configs:
        stride = config['stride']
        padding = config['padding']
        dilation = config['dilation']
        bias = config['bias']
        
        K_H, K_W = kernel_size
        H_out = (H + 2 * padding - dilation * (K_H - 1) - 1) // stride + 1
        W_out = (W + 2 * padding - dilation * (K_W - 1) - 1) // stride + 1
        
        if H_out <= 0 or W_out <= 0:
            continue
        
        # Create random tensors
        input_np = np.random.randn(N, C_in, H, W).astype(np.float32) * 0.1
        weight_np = np.random.randn(C_out, C_in, K_H, K_W).astype(np.float32) * 0.1
        bias_np = np.random.randn(C_out, 1, 1).astype(np.float32) * 0.1 if bias else None
        grad_output_np = np.random.randn(N, C_out, H_out, W_out).astype(np.float32) * 0.1
        
        input_t = nt.Tensor(input_np, requires_grad=True)
        weight_t = nt.Tensor(weight_np, requires_grad=True)
        bias_t = nt.Tensor(bias_np, requires_grad=True) if bias_np is not None else None
        
        # Forward pass
        output = Conv2DFunction.apply(
            input_t, weight_t, bias_t, stride, padding, dilation
        )
        grad_output_t = nt.Tensor(grad_output_np, requires_grad=False)
        
        # Backward pass
        output.backward(grad_output_t)
        
        # Numeric gradient for input
        eps = 1e-3
        numeric_grad_x = np.zeros_like(input_np)
        for idx in np.ndindex(input_np.shape):
            x_plus = input_np.copy()
            x_minus = input_np.copy()
            x_plus[idx] += eps
            x_minus[idx] -= eps
            
            x_plus_t = nt.Tensor(x_plus, requires_grad=False)
            x_minus_t = nt.Tensor(x_minus, requires_grad=False)
            
            output_plus = Conv2DFunction.apply(
                x_plus_t, weight_t, bias_t, stride, padding, dilation
            )
            output_minus = Conv2DFunction.apply(
                x_minus_t, weight_t, bias_t, stride, padding, dilation
            )
            
            loss_plus = np.sum(output_plus.data.astype(np.float64) * grad_output_np.astype(np.float64))
            loss_minus = np.sum(output_minus.data.astype(np.float64) * grad_output_np.astype(np.float64))
            numeric_grad_x[idx] = (loss_plus - loss_minus) / (2 * eps)
        
        # Numeric gradient for weight
        numeric_grad_w = np.zeros_like(weight_np)
        weight_np_copy = weight_np.copy()
        for idx in np.ndindex(weight_np.shape):
            w_plus = weight_np_copy.copy()
            w_minus = weight_np_copy.copy()
            w_plus[idx] += eps
            w_minus[idx] -= eps
            
            weight_plus_t = nt.Tensor(w_plus, requires_grad=False)
            weight_minus_t = nt.Tensor(w_minus, requires_grad=False)
            
            output_plus = Conv2DFunction.apply(
                input_t, weight_plus_t, bias_t, stride, padding, dilation
            )
            output_minus = Conv2DFunction.apply(
                input_t, weight_minus_t, bias_t, stride, padding, dilation
            )
            
            loss_plus = np.sum(output_plus.data.astype(np.float64) * grad_output_np.astype(np.float64))
            loss_minus = np.sum(output_minus.data.astype(np.float64) * grad_output_np.astype(np.float64))
            numeric_grad_w[idx] = (loss_plus - loss_minus) / (2 * eps)
        
        # Numeric gradient for bias if present
        if bias_np is not None:
            numeric_grad_b = np.zeros_like(bias_np)
            bias_np_copy = bias_np.copy()
            for idx in np.ndindex(bias_np.shape):
                b_plus = bias_np_copy.copy()
                b_minus = bias_np_copy.copy()
                b_plus[idx] += eps
                b_minus[idx] -= eps
                
                bias_plus_t = nt.Tensor(b_plus, requires_grad=False)
                bias_minus_t = nt.Tensor(b_minus, requires_grad=False)
                
                output_plus = Conv2DFunction.apply(
                    input_t, weight_t, bias_plus_t, stride, padding, dilation
                )
                output_minus = Conv2DFunction.apply(
                    input_t, weight_t, bias_minus_t, stride, padding, dilation
                )
                
                loss_plus = np.sum(output_plus.data.astype(np.float64) * grad_output_np.astype(np.float64))
                loss_minus = np.sum(output_minus.data.astype(np.float64) * grad_output_np.astype(np.float64))
                numeric_grad_b[idx] = (loss_plus - loss_minus) / (2 * eps)
        
        # Compare gradients with relaxed thresholds due to float32 precision
        def relative_error(a, b):
            return np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-8)
        
        assert input_t.grad is not None, "input gradient should exist"
        assert weight_t.grad is not None, "weight gradient should exist"
        error_x = relative_error(input_t.grad.data, numeric_grad_x).max()
        error_w = relative_error(weight_t.grad.data, numeric_grad_w).max()
        
        # Thresholds from final_gradient_check.py
        threshold_x = 1e-2
        threshold_w = 3e-2
        
        assert error_x < threshold_x, (
            f"Input gradient error too high: {error_x:.2e} > {threshold_x:.0e} "
            f"for config {config}"
        )
        assert error_w < threshold_w, (
            f"Weight gradient error too high: {error_w:.2e} > {threshold_w:.0e} "
            f"for config {config}"
        )
        
        if bias_np is not None:
            assert bias_t is not None, "bias_t should exist when bias_np is not None"
            assert bias_t.grad is not None, "bias gradient should exist"
            error_b = relative_error(bias_t.grad.data, numeric_grad_b).max()
            threshold_b = 1e-4
            assert error_b < threshold_b, (
                f"Bias gradient error too high: {error_b:.2e} > {threshold_b:.0e} "
                f"for config {config}"
            )
    
    print("✓ test_conv2d_gradient_correctness passed")


def run_all_tests():
    """Run all Conv2D tests."""
    test_conv2d_creation()
    test_conv2d_forward()
    test_conv2d_gradient_flow()
    test_conv2d_gradient_correctness()
    test_conv2d_no_bias()
    test_conv2d_padding()
    print("\n✅ All Conv2D tests passed!")


if __name__ == "__main__":
    run_all_tests()
