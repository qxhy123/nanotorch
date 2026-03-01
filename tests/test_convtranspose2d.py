"""
Tests for ConvTranspose2D layer.
"""

import numpy as np
import pytest
import nanotorch as nt
from nanotorch.nn import ConvTranspose2D
from nanotorch.autograd import ConvTranspose2DFunction
from nanotorch.utils import manual_seed


def test_convtranspose2d_creation():
    """Test ConvTranspose2D initialization."""
    manual_seed(42)

    # Default parameters
    conv = ConvTranspose2D(in_channels=3, out_channels=16, kernel_size=3)
    assert conv.in_channels == 3
    assert conv.out_channels == 16
    assert conv.kernel_size == (3, 3)
    assert conv.stride == 1
    assert conv.padding == 0
    assert conv.output_padding == 0
    assert conv.dilation == 1
    assert conv.bias is not None
    # Weight shape for ConvTranspose2D: (in_channels, out_channels, K_H, K_W)
    assert conv.weight.shape == (3, 16, 3, 3)
    assert conv.weight.requires_grad == True
    if conv.bias is not None:
        assert conv.bias.shape == (16, 1, 1)
        assert conv.bias.requires_grad == True

    # Custom parameters
    conv2 = ConvTranspose2D(
        in_channels=1,
        out_channels=8,
        kernel_size=(5, 7),
        stride=2,
        padding=1,
        output_padding=1,
        dilation=2,
        bias=False,
    )
    assert conv2.in_channels == 1
    assert conv2.out_channels == 8
    assert conv2.kernel_size == (5, 7)
    assert conv2.stride == 2
    assert conv2.padding == 1
    assert conv2.output_padding == 1
    assert conv2.dilation == 2
    assert conv2.bias is None
    # Weight shape: (in_channels, out_channels // groups, K_H, K_W)
    # groups=1 by default
    assert conv2.weight.shape == (1, 8, 5, 7)

    print("✓ test_convtranspose2d_creation passed")


def test_convtranspose2d_forward():
    """Test forward pass of ConvTranspose2D."""
    manual_seed(42)

    # Create conv transpose layer
    conv = ConvTranspose2D(in_channels=2, out_channels=4, kernel_size=3, stride=2, padding=1)

    # Input tensor: batch=2, channels=2, height=5, width=5
    x = nt.Tensor.randn((2, 2, 5, 5), requires_grad=False)

    # Forward pass
    out = conv(x)

    # Check output shape using formula
    # H_out = (H_in - 1) * stride - 2 * padding + dilation * (K_H - 1) + output_padding + 1
    H_in, W_in = 5, 5
    stride = 2
    padding = 1
    dilation = 1
    output_padding = 0
    K_H, K_W = 3, 3
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (K_H - 1) + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (K_W - 1) + output_padding + 1
    assert H_out == 9, f"Expected H_out=9, got {H_out}"
    assert W_out == 9, f"Expected W_out=9, got {W_out}"
    assert out.shape == (2, 4, H_out, W_out)

    # Check that output values are not all zeros
    assert not np.allclose(out.data, 0.0)

    # Test with stride=1, padding=0 (simple upsampling)
    conv2 = ConvTranspose2D(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=0)
    out2 = conv2(x)
    expected_h = (5 - 1) * 1 - 2 * 0 + 1 * (3 - 1) + 0 + 1  # = 5 - 1 + 3 = 7
    expected_w = (5 - 1) * 1 - 2 * 0 + 1 * (3 - 1) + 0 + 1  # = 7
    assert out2.shape == (2, 3, expected_h, expected_w)

    print("✓ test_convtranspose2d_forward passed")


def test_convtranspose2d_gradient_flow():
    """Test gradient computation through ConvTranspose2D."""
    manual_seed(123)

    conv = ConvTranspose2D(in_channels=2, out_channels=3, kernel_size=3, padding=1)
    x = nt.Tensor.randn((1, 2, 4, 4), requires_grad=True)

    out = conv(x)
    loss = out.sum()
    loss.backward()

    print("✓ test_convtranspose2d_gradient_flow passed")


def test_convtranspose2d_no_bias():
    """Test ConvTranspose2D without bias."""
    conv = ConvTranspose2D(in_channels=1, out_channels=2, kernel_size=3, bias=False)
    assert conv.bias is None

    x = nt.Tensor.randn((1, 1, 6, 6), requires_grad=False)
    out = conv(x)
    
    # Compute expected output shape
    H_in, W_in = 6, 6
    stride = 1
    padding = 0
    dilation = 1
    output_padding = 0
    K_H, K_W = 3, 3
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (K_H - 1) + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (K_W - 1) + output_padding + 1
    assert out.shape == (1, 2, H_out, W_out)

    # Ensure no bias parameter exists
    assert "bias" not in conv._parameters or conv._parameters["bias"] is None

    print("✓ test_convtranspose2d_no_bias passed")


def test_convtranspose2d_padding():
    """Test ConvTranspose2D with different padding values."""
    # Input size 5x5, kernel 3x3, stride=1
    # padding 0 -> output 7x7
    conv1 = ConvTranspose2D(in_channels=1, out_channels=1, kernel_size=3, padding=0)
    x = nt.Tensor.randn((1, 1, 5, 5), requires_grad=False)
    out1 = conv1(x)
    assert out1.shape == (1, 1, 7, 7)

    # padding 1 -> output 5x5 (preserve size with stride=1)
    conv2 = ConvTranspose2D(in_channels=1, out_channels=1, kernel_size=3, padding=1)
    out2 = conv2(x)
    assert out2.shape == (1, 1, 5, 5)

    # padding 2 -> output 3x3 (reduce size)
    conv3 = ConvTranspose2D(in_channels=1, out_channels=1, kernel_size=3, padding=2)
    out3 = conv3(x)
    assert out3.shape == (1, 1, 3, 3)

    print("✓ test_convtranspose2d_padding passed")


def test_convtranspose2d_output_padding():
    """Test ConvTranspose2D with output_padding."""
    # Input 4x4, kernel=3, stride=2, padding=1, output_padding=1
    # Without output_padding: H_out = (4-1)*2 - 2*1 + (3-1) + 1 = 3*2 -2 +2 +1 = 6-2+2+1=7
    # With output_padding=1: H_out = 7 + 1 = 8
    conv = ConvTranspose2D(
        in_channels=1, 
        out_channels=1, 
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1
    )
    x = nt.Tensor.randn((1, 1, 4, 4), requires_grad=False)
    out = conv(x)
    assert out.shape == (1, 1, 8, 8)

    # Test that output_padding < stride (PyTorch constraint)
    # This should work since output_padding=1 < stride=2
    print("✓ test_convtranspose2d_output_padding passed")


def test_convtranspose2d_gradient_correctness():
    """Test ConvTranspose2D gradient correctness with finite differences."""
    np.random.seed(42)
    
    # Test a representative subset of configurations
    configs = [
        {'stride': 1, 'padding': 0, 'dilation': 1, 'bias': True, 'output_padding': 0},
        {'stride': 2, 'padding': 1, 'dilation': 1, 'bias': False, 'output_padding': 1},
        {'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True, 'output_padding': 0},
        {'stride': 1, 'padding': 0, 'dilation': 2, 'bias': False, 'output_padding': 0},
    ]
    
    N, C_in, C_out = 2, 3, 4
    H_in, W_in = 8, 8
    kernel_size = (3, 3)
    
    for config in configs:
        stride = config['stride']
        padding = config['padding']
        dilation = config['dilation']
        bias = config['bias']
        output_padding = config['output_padding']
        
        K_H, K_W = kernel_size
        
        # Compute output size using ConvTranspose2D formula
        def output_size(input_size, kernel_size, stride, padding, output_padding, dilation):
            kernel_size_dilated = dilation * (kernel_size - 1) + 1
            return (input_size - 1) * stride - 2 * padding + kernel_size_dilated + output_padding
        
        H_out = output_size(H_in, K_H, stride, padding, output_padding, dilation)
        W_out = output_size(W_in, K_W, stride, padding, output_padding, dilation)
        
        if H_out <= 0 or W_out <= 0:
            continue
        
        # Create random tensors
        input_np = np.random.randn(N, C_in, H_in, W_in).astype(np.float32) * 0.1
        # Weight shape for ConvTranspose2D: (C_in, C_out, K_H, K_W)
        weight_np = np.random.randn(C_in, C_out, K_H, K_W).astype(np.float32) * 0.1
        bias_np = np.random.randn(C_out, 1, 1).astype(np.float32) * 0.1 if bias else None
        grad_output_np = np.random.randn(N, C_out, H_out, W_out).astype(np.float32) * 0.1
        
        input_t = nt.Tensor(input_np, requires_grad=True)
        weight_t = nt.Tensor(weight_np, requires_grad=True)
        bias_t = nt.Tensor(bias_np, requires_grad=True) if bias_np is not None else None
        
        # Forward pass
        output = ConvTranspose2DFunction.apply(
            input_t, weight_t, bias_t, stride, padding, output_padding, dilation
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
            
            output_plus = ConvTranspose2DFunction.apply(
                x_plus_t, weight_t, bias_t, stride, padding, output_padding, dilation
            )
            output_minus = ConvTranspose2DFunction.apply(
                x_minus_t, weight_t, bias_t, stride, padding, output_padding, dilation
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
            
            output_plus = ConvTranspose2DFunction.apply(
                input_t, weight_plus_t, bias_t, stride, padding, output_padding, dilation
            )
            output_minus = ConvTranspose2DFunction.apply(
                input_t, weight_minus_t, bias_t, stride, padding, output_padding, dilation
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
                
                output_plus = ConvTranspose2DFunction.apply(
                    input_t, weight_t, bias_plus_t, stride, padding, output_padding, dilation
                )
                output_minus = ConvTranspose2DFunction.apply(
                    input_t, weight_t, bias_minus_t, stride, padding, output_padding, dilation
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
        
        # Use same thresholds as Conv2D for consistency
        threshold_x = 1e-2  # Input gradient
        threshold_w = 3e-2  # Weight gradient
        
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
    
    print("✓ test_convtranspose2d_gradient_correctness passed")


def run_all_tests():
    """Run all ConvTranspose2D tests."""
    test_convtranspose2d_creation()
    test_convtranspose2d_forward()
    test_convtranspose2d_gradient_flow()
    test_convtranspose2d_no_bias()
    test_convtranspose2d_padding()
    test_convtranspose2d_output_padding()
    test_convtranspose2d_gradient_correctness()
    print("\n✅ All ConvTranspose2D tests passed!")


if __name__ == "__main__":
    run_all_tests()