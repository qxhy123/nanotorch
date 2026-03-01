"""
Tests for Conv3D layer.
"""

import numpy as np
import pytest
import nanotorch as nt
from nanotorch.nn import Conv3D
from nanotorch.autograd import Conv3DFunction
from nanotorch.utils import manual_seed


def test_conv3d_creation():
    """Test Conv3D initialization."""
    # Default parameters
    conv = Conv3D(in_channels=3, out_channels=16, kernel_size=3)
    assert conv.in_channels == 3
    assert conv.out_channels == 16
    assert conv.kernel_size == (3, 3, 3)
    assert conv.stride == (1, 1, 1)
    assert conv.padding == (0, 0, 0)
    assert conv.dilation == (1, 1, 1)
    assert conv.bias is not None
    assert conv.weight.shape == (16, 3, 3, 3, 3)
    assert conv.weight.requires_grad == True
    if conv.bias is not None:
        assert conv.bias.shape == (16, 1, 1, 1)
        assert conv.bias.requires_grad == True

    # Custom parameters with int values (should be converted to tuples)
    conv2 = Conv3D(
        in_channels=1,
        out_channels=8,
        kernel_size=(5, 5, 7),
        stride=2,
        padding=1,
        dilation=2,
        bias=False,
    )
    assert conv2.in_channels == 1
    assert conv2.out_channels == 8
    assert conv2.kernel_size == (5, 5, 7)
    assert conv2.stride == (2, 2, 2)
    assert conv2.padding == (1, 1, 1)
    assert conv2.dilation == (2, 2, 2)
    assert conv2.bias is None
    assert conv2.weight.shape == (8, 1, 5, 5, 7)

    # Custom parameters with tuple values
    conv3 = Conv3D(
        in_channels=2,
        out_channels=4,
        kernel_size=(3, 3, 3),
        stride=(2, 1, 1),
        padding=(1, 2, 1),
        dilation=(1, 2, 1),
        bias=True,
    )
    assert conv3.stride == (2, 1, 1)
    assert conv3.padding == (1, 2, 1)
    assert conv3.dilation == (1, 2, 1)

    print("✓ test_conv3d_creation passed")


def test_conv3d_forward():
    """Test forward pass of Conv3D."""
    manual_seed(42)

    # Create conv layer
    conv = Conv3D(in_channels=2, out_channels=4, kernel_size=3, padding=1)

    # Input tensor: batch=2, channels=2, depth=5, height=5, width=5
    x = nt.Tensor.randn((2, 2, 5, 5, 5), requires_grad=False)

    # Forward pass
    out = conv(x)

    # Check output shape
    assert out.shape == (2, 4, 5, 5, 5)  # padding=1 preserves spatial dimensions

    # Check that output values are not all zeros
    assert not np.allclose(out.data, 0.0)

    # Test with stride=2
    conv2 = Conv3D(in_channels=2, out_channels=3, kernel_size=3, stride=2)
    out2 = conv2(x)
    expected_d = (5 - 3) // 2 + 1  # = 2
    expected_h = (5 - 3) // 2 + 1  # = 2
    expected_w = (5 - 3) // 2 + 1  # = 2
    assert out2.shape == (2, 3, expected_d, expected_h, expected_w)

    print("✓ test_conv3d_forward passed")


def test_conv3d_gradient_flow():
    """Test gradient computation through Conv3D."""
    manual_seed(123)

    conv = Conv3D(in_channels=2, out_channels=3, kernel_size=3, padding=1)
    x = nt.Tensor.randn((1, 2, 4, 4, 4), requires_grad=True)

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

    print("✓ test_conv3d_gradient_flow passed")


def test_conv3d_no_bias():
    """Test Conv3D without bias."""
    conv = Conv3D(in_channels=1, out_channels=2, kernel_size=3, bias=False)
    assert conv.bias is None

    x = nt.Tensor.randn((1, 1, 6, 6, 6), requires_grad=False)
    out = conv(x)
    assert out.shape == (1, 2, 4, 4, 4)

    # Ensure no bias parameter exists
    assert "bias" not in conv._parameters or conv._parameters["bias"] is None

    print("✓ test_conv3d_no_bias passed")


def test_conv3d_padding():
    """Test Conv3D with different padding values."""
    # Input size 5x5x5, kernel 3x3x3, padding 0 -> output 3x3x3
    conv1 = Conv3D(in_channels=1, out_channels=1, kernel_size=3, padding=0)
    x = nt.Tensor.randn((1, 1, 5, 5, 5), requires_grad=False)
    out1 = conv1(x)
    assert out1.shape == (1, 1, 3, 3, 3)

    # padding 1 -> output 5x5x5 (preserve size)
    conv2 = Conv3D(in_channels=1, out_channels=1, kernel_size=3, padding=1)
    out2 = conv2(x)
    assert out2.shape == (1, 1, 5, 5, 5)

    # padding 2 -> output 7x7x7 (increase size)
    conv3 = Conv3D(in_channels=1, out_channels=1, kernel_size=3, padding=2)
    out3 = conv3(x)
    assert out3.shape == (1, 1, 7, 7, 7)

    print("✓ test_conv3d_padding passed")


def test_conv3d_gradient_correctness():
    """Test Conv3D gradient correctness with finite differences."""
    np.random.seed(42)

    # Test a representative subset of configurations
    configs = [
        {'stride': 1, 'padding': 0, 'dilation': 1, 'bias': True},
        {'stride': 2, 'padding': 0, 'dilation': 1, 'bias': False},
        {'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True},
        {'stride': 1, 'padding': 0, 'dilation': 2, 'bias': False},
    ]

    N, C_in, C_out = 2, 3, 4
    D, H, W = 8, 8, 8
    kernel_size = (3, 3, 3)

    for config in configs:
        stride_int = config['stride']
        padding_int = config['padding']
        dilation_int = config['dilation']
        bias = config['bias']

        # Convert to tuples for Conv3DFunction
        stride = (stride_int, stride_int, stride_int)
        padding = (padding_int, padding_int, padding_int)
        dilation = (dilation_int, dilation_int, dilation_int)

        K_D, K_H, K_W = kernel_size
        D_out = (D + 2 * padding_int - dilation_int * (K_D - 1) - 1) // stride_int + 1
        H_out = (H + 2 * padding_int - dilation_int * (K_H - 1) - 1) // stride_int + 1
        W_out = (W + 2 * padding_int - dilation_int * (K_W - 1) - 1) // stride_int + 1

        if D_out <= 0 or H_out <= 0 or W_out <= 0:
            continue

        # Create random tensors
        input_np = np.random.randn(N, C_in, D, H, W).astype(np.float32) * 0.1
        weight_np = np.random.randn(C_out, C_in, K_D, K_H, K_W).astype(np.float32) * 0.1
        bias_np = np.random.randn(C_out, 1, 1, 1).astype(np.float32) * 0.1 if bias else None
        grad_output_np = np.random.randn(N, C_out, D_out, H_out, W_out).astype(np.float32) * 0.1

        input_t = nt.Tensor(input_np, requires_grad=True)
        weight_t = nt.Tensor(weight_np, requires_grad=True)
        bias_t = nt.Tensor(bias_np, requires_grad=True) if bias_np is not None else None

        # Forward pass
        output = Conv3DFunction.apply(
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

            output_plus = Conv3DFunction.apply(
                x_plus_t, weight_t, bias_t, stride, padding, dilation
            )
            output_minus = Conv3DFunction.apply(
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

            output_plus = Conv3DFunction.apply(
                input_t, weight_plus_t, bias_t, stride, padding, dilation
            )
            output_minus = Conv3DFunction.apply(
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

                output_plus = Conv3DFunction.apply(
                    input_t, weight_t, bias_plus_t, stride, padding, dilation
                )
                output_minus = Conv3DFunction.apply(
                    input_t, weight_t, bias_minus_t, stride, padding, dilation
                )

                loss_plus = np.sum(output_plus.data.astype(np.float64) * grad_output_np.astype(np.float64))
                loss_minus = np.sum(output_minus.data.astype(np.float64) * grad_output_np.astype(np.float64))
                numeric_grad_b[idx] = (loss_plus - loss_minus) / (2 * eps)

        # Compare gradients with relaxed thresholds due to float32 precision
        # Note: Conv3D has higher numerical errors than Conv2D due to additional depth dimension
        def relative_error(a, b):
            return np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-8)

        assert input_t.grad is not None, "input gradient should exist"
        assert weight_t.grad is not None, "weight gradient should exist"
        error_x = relative_error(input_t.grad.data, numeric_grad_x).max()
        error_w = relative_error(weight_t.grad.data, numeric_grad_w).max()

        # Thresholds adjusted for Conv3D - higher than Conv2D due to additional dimension
        # High relative errors typically occur on elements with very small gradient magnitudes
        threshold_x = 5e-2  # Relaxed from 1e-2 (Conv2D) for Conv3D
        threshold_w = 5e-2  # Relaxed from 3e-2 (Conv2D) for Conv3D

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
            threshold_b = 1e-4  # Same as Conv2D for bias
            assert error_b < threshold_b, (
                f"Bias gradient error too high: {error_b:.2e} > {threshold_b:.0e} "
                f"for config {config}"
            )

    print("✓ test_conv3d_gradient_correctness passed")


def test_conv3d_tuple_parameters():
    """Test Conv3D with tuple parameters for stride, padding, and dilation."""
    np.random.seed(42)

    N, C_in, C_out = 2, 3, 4
    D, H, W = 8, 10, 12
    kernel_size = (3, 5, 7)

    # Test with tuple parameters
    conv = Conv3D(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=kernel_size,
        stride=(2, 1, 3),
        padding=(1, 2, 3),
        dilation=(1, 2, 1),
        bias=True,
    )

    # Check that parameters are stored as tuples
    assert conv.stride == (2, 1, 3)
    assert conv.padding == (1, 2, 3)
    assert conv.dilation == (1, 2, 1)

    # Create input and run forward pass
    x = nt.Tensor(np.random.randn(N, C_in, D, H, W).astype(np.float32), requires_grad=True)
    output = conv(x)

    # Calculate expected output size
    K_D, K_H, K_W = kernel_size
    stride_d, stride_h, stride_w = conv.stride
    padding_d, padding_h, padding_w = conv.padding
    dilation_d, dilation_h, dilation_w = conv.dilation

    D_out = (D + 2 * padding_d - dilation_d * (K_D - 1) - 1) // stride_d + 1
    H_out = (H + 2 * padding_h - dilation_h * (K_H - 1) - 1) // stride_h + 1
    W_out = (W + 2 * padding_w - dilation_w * (K_W - 1) - 1) // stride_w + 1

    assert output.shape == (N, C_out, D_out, H_out, W_out), (
        f"Expected shape {(N, C_out, D_out, H_out, W_out)}, got {output.shape}"
    )

    # Test backward pass
    output.backward()
    assert x.grad is not None
    assert conv.weight.grad is not None
    if conv.bias is not None:
        assert conv.bias.grad is not None

    print("✓ test_conv3d_tuple_parameters passed")


def test_conv3d_mixed_parameters():
    """Test Conv3D with mixed int and tuple parameters."""
    # Test that int parameters are converted to tuples
    conv = Conv3D(
        in_channels=2,
        out_channels=4,
        kernel_size=3,  # int
        stride=2,  # int
        padding=1,  # int
        dilation=1,  # int
    )

    assert conv.kernel_size == (3, 3, 3)
    assert conv.stride == (2, 2, 2)
    assert conv.padding == (1, 1, 1)
    assert conv.dilation == (1, 1, 1)

    # Test with mixed: some tuple, some int
    conv2 = Conv3D(
        in_channels=2,
        out_channels=4,
        kernel_size=(3, 3, 3),  # tuple
        stride=2,  # int
        padding=(1, 2, 1),  # tuple
        dilation=1,  # int
    )

    assert conv2.stride == (2, 2, 2)
    assert conv2.padding == (1, 2, 1)
    assert conv2.dilation == (1, 1, 1)

    print("✓ test_conv3d_mixed_parameters passed")


def run_all_tests():
    """Run all Conv3D tests."""
    test_conv3d_creation()
    test_conv3d_forward()
    test_conv3d_gradient_flow()
    test_conv3d_gradient_correctness()
    test_conv3d_no_bias()
    test_conv3d_padding()
    test_conv3d_tuple_parameters()
    test_conv3d_mixed_parameters()
    print("\n✅ All Conv3D tests passed!")


if __name__ == "__main__":
    run_all_tests()
