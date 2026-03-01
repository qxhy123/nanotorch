"""
Tests for pooling layers (MaxPool2d and AvgPool2d).
"""

import numpy as np
import pytest
import nanotorch as nt
from nanotorch.nn import MaxPool2d, AvgPool2d
from nanotorch.autograd import MaxPool2dFunction, AvgPool2dFunction
from nanotorch.utils import manual_seed


def test_maxpool2d_creation():
    """Test MaxPool2d initialization."""
    pool = MaxPool2d(kernel_size=2)
    assert pool.kernel_size == (2, 2)
    assert pool.stride == (2, 2)
    assert pool.padding == (0, 0)
    assert pool.dilation == (1, 1)
    assert pool.return_indices == False
    assert pool.ceil_mode == False

    pool2 = MaxPool2d(
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=1,
        return_indices=True,
        ceil_mode=True,
    )
    assert pool2.kernel_size == (3, 3)
    assert pool2.stride == (2, 2)
    assert pool2.padding == (1, 1)
    assert pool2.dilation == (1, 1)
    assert pool2.return_indices == True
    assert pool2.ceil_mode == True

    pool3 = MaxPool2d(kernel_size=(3, 5))
    assert pool3.kernel_size == (3, 5)
    assert pool3.stride == (3, 5)

    print("✓ test_maxpool2d_creation passed")


def test_maxpool2d_forward():
    """Test forward pass of MaxPool2d."""
    manual_seed(42)

    pool = MaxPool2d(kernel_size=2, stride=2)
    x = nt.Tensor.randn((2, 3, 4, 4), requires_grad=False)
    out = pool(x)

    assert out.shape == (2, 3, 2, 2)
    assert not np.allclose(out.data, 0.0)

    pool2 = MaxPool2d(kernel_size=3, stride=1, padding=1)
    out2 = pool2(x)
    assert out2.shape == (2, 3, 4, 4)

    print("✓ test_maxpool2d_forward passed")


def test_maxpool2d_gradient_flow():
    """Test gradient computation through MaxPool2d."""
    manual_seed(123)

    pool = MaxPool2d(kernel_size=2, stride=2)
    x = nt.Tensor.randn((1, 2, 4, 4), requires_grad=True)

    out = pool(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not np.allclose(x.grad.data, 0.0)

    print("✓ test_maxpool2d_gradient_flow passed")


def test_maxpool2d_padding():
    """Test MaxPool2d with different padding values."""
    pool1 = MaxPool2d(kernel_size=3, stride=1, padding=0)
    x = nt.Tensor.randn((1, 1, 5, 5), requires_grad=False)
    out1 = pool1(x)
    assert out1.shape == (1, 1, 3, 3)

    pool2 = MaxPool2d(kernel_size=3, stride=1, padding=1)
    out2 = pool2(x)
    assert out2.shape == (1, 1, 5, 5)

    pool3 = MaxPool2d(kernel_size=3, stride=1, padding=2)
    out3 = pool3(x)
    assert out3.shape == (1, 1, 7, 7)

    print("✓ test_maxpool2d_padding passed")


def test_maxpool2d_ceil_mode():
    """Test MaxPool2d with ceil_mode."""
    pool_floor = MaxPool2d(kernel_size=3, stride=2, ceil_mode=False)
    pool_ceil = MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
    x = nt.Tensor.randn((1, 1, 5, 5), requires_grad=False)

    out_floor = pool_floor(x)
    out_ceil = pool_ceil(x)
    assert out_floor.shape == (1, 1, 2, 2)
    assert out_ceil.shape == (1, 1, 2, 2)

    x2 = nt.Tensor.randn((1, 1, 6, 6), requires_grad=False)
    out_floor2 = pool_floor(x2)
    out_ceil2 = pool_ceil(x2)
    assert out_floor2.shape == (1, 1, 2, 2)
    assert out_ceil2.shape == (1, 1, 3, 3)

    print("✓ test_maxpool2d_ceil_mode passed")


def test_maxpool2d_gradient_correctness():
    """Test MaxPool2d gradient correctness with finite differences."""
    np.random.seed(42)

    configs = [
        {'kernel_size': 2, 'stride': 2, 'padding': 0, 'dilation': 1, 'ceil_mode': False},
        {'kernel_size': 3, 'stride': 2, 'padding': 1, 'dilation': 1, 'ceil_mode': False},
        {'kernel_size': 2, 'stride': 1, 'padding': 1, 'dilation': 1, 'ceil_mode': False},
        {'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 2, 'ceil_mode': False},
    ]

    N, C = 2, 3
    H, W = 6, 6

    for config in configs:
        kernel_size = config['kernel_size']
        stride = config['stride']
        padding = config['padding']
        dilation = config['dilation']
        ceil_mode = config['ceil_mode']

        kernel_size_t = (kernel_size, kernel_size)
        stride_t = (stride, stride)
        padding_t = (padding, padding)
        dilation_t = (dilation, dilation)

        K_H, K_W = kernel_size_t
        kernel_size_dilated_h = dilation_t[0] * (K_H - 1) + 1
        kernel_size_dilated_w = dilation_t[1] * (K_W - 1) + 1

        if ceil_mode:
            H_out = int(np.ceil((H + 2 * padding - kernel_size_dilated_h) / stride_t[0] + 1))
            W_out = int(np.ceil((W + 2 * padding - kernel_size_dilated_w) / stride_t[1] + 1))
        else:
            H_out = (H + 2 * padding - kernel_size_dilated_h) // stride_t[0] + 1
            W_out = (W + 2 * padding - kernel_size_dilated_w) // stride_t[1] + 1

        if H_out <= 0 or W_out <= 0:
            continue

        input_np = np.random.randn(N, C, H, W).astype(np.float32) * 0.1
        grad_output_np = np.random.randn(N, C, H_out, W_out).astype(np.float32) * 0.1

        input_t = nt.Tensor(input_np, requires_grad=True)

        output = MaxPool2dFunction.apply(
            input_t, kernel_size_t, stride_t, padding_t, dilation_t, ceil_mode, False
        )
        grad_output_t = nt.Tensor(grad_output_np, requires_grad=False)

        output.backward(grad_output_t)

        eps = 1e-3
        numeric_grad_x = np.zeros_like(input_np)
        for idx in np.ndindex(input_np.shape):
            x_plus = input_np.copy()
            x_minus = input_np.copy()
            x_plus[idx] += eps
            x_minus[idx] -= eps

            x_plus_t = nt.Tensor(x_plus, requires_grad=False)
            x_minus_t = nt.Tensor(x_minus, requires_grad=False)

            output_plus = MaxPool2dFunction.apply(
                x_plus_t, kernel_size_t, stride_t, padding_t, dilation_t, ceil_mode, False
            )
            output_minus = MaxPool2dFunction.apply(
                x_minus_t, kernel_size_t, stride_t, padding_t, dilation_t, ceil_mode, False
            )

            loss_plus = np.sum(output_plus.data.astype(np.float64) * grad_output_np.astype(np.float64))
            loss_minus = np.sum(output_minus.data.astype(np.float64) * grad_output_np.astype(np.float64))
            numeric_grad_x[idx] = (loss_plus - loss_minus) / (2 * eps)

        def relative_error(a, b):
            return np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-8)

        assert input_t.grad is not None, "input gradient should exist"
        error_x = relative_error(input_t.grad.data, numeric_grad_x).max()

        threshold_x = 2.0

        assert error_x < threshold_x, (
            f"Input gradient error too high: {error_x:.2e} > {threshold_x:.0e} "
            f"for config {config}"
        )

    print("✓ test_maxpool2d_gradient_correctness passed")


def test_avgpool2d_creation():
    """Test AvgPool2d initialization."""
    pool = AvgPool2d(kernel_size=2)
    assert pool.kernel_size == (2, 2)
    assert pool.stride == (2, 2)
    assert pool.padding == (0, 0)
    assert pool.ceil_mode == False
    assert pool.count_include_pad == True
    assert pool.divisor_override is None

    pool2 = AvgPool2d(
        kernel_size=3,
        stride=2,
        padding=1,
        ceil_mode=True,
        count_include_pad=False,
        divisor_override=9,
    )
    assert pool2.kernel_size == (3, 3)
    assert pool2.stride == (2, 2)
    assert pool2.padding == (1, 1)
    assert pool2.ceil_mode == True
    assert pool2.count_include_pad == False
    assert pool2.divisor_override == 9

    pool3 = AvgPool2d(kernel_size=(3, 5))
    assert pool3.kernel_size == (3, 5)
    assert pool3.stride == (3, 5)

    print("✓ test_avgpool2d_creation passed")


def test_avgpool2d_forward():
    """Test forward pass of AvgPool2d."""
    manual_seed(42)

    pool = AvgPool2d(kernel_size=2, stride=2)
    x = nt.Tensor.randn((2, 3, 4, 4), requires_grad=False)
    out = pool(x)

    assert out.shape == (2, 3, 2, 2)
    assert not np.allclose(out.data, 0.0)

    pool2 = AvgPool2d(kernel_size=3, stride=1, padding=1)
    out2 = pool2(x)
    assert out2.shape == (2, 3, 4, 4)

    print("✓ test_avgpool2d_forward passed")


def test_avgpool2d_gradient_flow():
    """Test gradient computation through AvgPool2d."""
    manual_seed(123)

    pool = AvgPool2d(kernel_size=2, stride=2)
    x = nt.Tensor.randn((1, 2, 4, 4), requires_grad=True)

    out = pool(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not np.allclose(x.grad.data, 0.0)

    print("✓ test_avgpool2d_gradient_flow passed")


def test_avgpool2d_padding():
    """Test AvgPool2d with different padding values."""
    pool1 = AvgPool2d(kernel_size=3, stride=1, padding=0)
    x = nt.Tensor.randn((1, 1, 5, 5), requires_grad=False)
    out1 = pool1(x)
    assert out1.shape == (1, 1, 3, 3)

    pool2 = AvgPool2d(kernel_size=3, stride=1, padding=1)
    out2 = pool2(x)
    assert out2.shape == (1, 1, 5, 5)

    pool3 = AvgPool2d(kernel_size=3, stride=1, padding=2)
    out3 = pool3(x)
    assert out3.shape == (1, 1, 7, 7)

    print("✓ test_avgpool2d_padding passed")


def test_avgpool2d_ceil_mode():
    """Test AvgPool2d with ceil_mode."""
    pool_floor = AvgPool2d(kernel_size=3, stride=2, ceil_mode=False)
    pool_ceil = AvgPool2d(kernel_size=3, stride=2, ceil_mode=True)
    x = nt.Tensor.randn((1, 1, 5, 5), requires_grad=False)

    out_floor = pool_floor(x)
    out_ceil = pool_ceil(x)
    assert out_floor.shape == (1, 1, 2, 2)
    assert out_ceil.shape == (1, 1, 2, 2)

    print("✓ test_avgpool2d_ceil_mode passed")


def test_avgpool2d_gradient_correctness():
    """Test AvgPool2d gradient correctness with finite differences."""
    np.random.seed(42)

    configs = [
        {'kernel_size': 2, 'stride': 2, 'padding': 0, 'ceil_mode': False, 'count_include_pad': True},
        {'kernel_size': 3, 'stride': 2, 'padding': 1, 'ceil_mode': False, 'count_include_pad': True},
        {'kernel_size': 2, 'stride': 1, 'padding': 1, 'ceil_mode': False, 'count_include_pad': False},
        {'kernel_size': 3, 'stride': 1, 'padding': 0, 'ceil_mode': False, 'count_include_pad': True},
    ]

    N, C = 2, 3
    H, W = 6, 6

    for config in configs:
        kernel_size = config['kernel_size']
        stride = config['stride']
        padding = config['padding']
        ceil_mode = config['ceil_mode']
        count_include_pad = config['count_include_pad']

        kernel_size_t = (kernel_size, kernel_size)
        stride_t = (stride, stride)
        padding_t = (padding, padding)

        K_H, K_W = kernel_size_t

        if ceil_mode:
            H_out = int(np.ceil((H + 2 * padding - K_H) / stride_t[0] + 1))
            W_out = int(np.ceil((W + 2 * padding - K_W) / stride_t[1] + 1))
        else:
            H_out = (H + 2 * padding - K_H) // stride_t[0] + 1
            W_out = (W + 2 * padding - K_W) // stride_t[1] + 1

        if H_out <= 0 or W_out <= 0:
            continue

        input_np = np.random.randn(N, C, H, W).astype(np.float32) * 0.1
        grad_output_np = np.random.randn(N, C, H_out, W_out).astype(np.float32) * 0.1

        input_t = nt.Tensor(input_np, requires_grad=True)

        output = AvgPool2dFunction.apply(
            input_t, kernel_size_t, stride_t, padding_t, ceil_mode, count_include_pad, None
        )
        grad_output_t = nt.Tensor(grad_output_np, requires_grad=False)

        output.backward(grad_output_t)

        eps = 1e-3
        numeric_grad_x = np.zeros_like(input_np)
        for idx in np.ndindex(input_np.shape):
            x_plus = input_np.copy()
            x_minus = input_np.copy()
            x_plus[idx] += eps
            x_minus[idx] -= eps

            x_plus_t = nt.Tensor(x_plus, requires_grad=False)
            x_minus_t = nt.Tensor(x_minus, requires_grad=False)

            output_plus = AvgPool2dFunction.apply(
                x_plus_t, kernel_size_t, stride_t, padding_t, ceil_mode, count_include_pad, None
            )
            output_minus = AvgPool2dFunction.apply(
                x_minus_t, kernel_size_t, stride_t, padding_t, ceil_mode, count_include_pad, None
            )

            loss_plus = np.sum(output_plus.data.astype(np.float64) * grad_output_np.astype(np.float64))
            loss_minus = np.sum(output_minus.data.astype(np.float64) * grad_output_np.astype(np.float64))
            numeric_grad_x[idx] = (loss_plus - loss_minus) / (2 * eps)

        def relative_error(a, b):
            return np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-8)

        assert input_t.grad is not None, "input gradient should exist"
        error_x = relative_error(input_t.grad.data, numeric_grad_x).max()

        threshold_x = 5e-3

        assert error_x < threshold_x, (
            f"Input gradient error too high: {error_x:.2e} > {threshold_x:.0e} "
            f"for config {config}"
        )

    print("✓ test_avgpool2d_gradient_correctness passed")


def test_functional_max_pool2d():
    """Test functional max_pool2d interface."""
    manual_seed(42)

    x = nt.Tensor.randn((2, 3, 4, 4), requires_grad=False)
    out = nt.max_pool2d(x, kernel_size=2, stride=2)

    assert out.shape == (2, 3, 2, 2)
    assert not np.allclose(out.data, 0.0)

    print("✓ test_functional_max_pool2d passed")


def test_functional_avg_pool2d():
    """Test functional avg_pool2d interface."""
    manual_seed(42)

    x = nt.Tensor.randn((2, 3, 4, 4), requires_grad=False)
    out = nt.avg_pool2d(x, kernel_size=2, stride=2)

    assert out.shape == (2, 3, 2, 2)
    assert not np.allclose(out.data, 0.0)

    print("✓ test_functional_avg_pool2d passed")


def test_pooling_edge_cases():
    """Test edge cases for pooling layers."""
    pool = MaxPool2d(kernel_size=4, stride=1, padding=0)
    x = nt.Tensor.randn((1, 1, 4, 4), requires_grad=False)
    out = pool(x)
    assert out.shape == (1, 1, 1, 1)

    pool2 = MaxPool2d(kernel_size=(3, 5), stride=1, padding=0)
    x2 = nt.Tensor.randn((1, 1, 10, 10), requires_grad=False)
    out2 = pool2(x2)
    assert out2.shape == (1, 1, 8, 6)

    print("✓ test_pooling_edge_cases passed")


def test_pooling_invalid_input():
    """Test pooling layers with invalid input."""
    pool = MaxPool2d(kernel_size=2)

    x_3d = nt.Tensor.randn((2, 4, 4), requires_grad=False)
    try:
        pool(x_3d)
        assert False, "Should have raised ValueError for 3D input"
    except ValueError as e:
        assert "4D input" in str(e)

    x_5d = nt.Tensor.randn((2, 3, 4, 4, 5), requires_grad=False)
    try:
        pool(x_5d)
        assert False, "Should have raised ValueError for 5D input"
    except ValueError as e:
        assert "4D input" in str(e)

    print("✓ test_pooling_invalid_input passed")


def test_maxpool2d_return_indices():
    """Test MaxPool2d with return_indices=True."""
    manual_seed(42)
    
    # Test 1: Basic functionality
    pool = MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    x = nt.Tensor.randn((2, 3, 4, 4), requires_grad=False)
    result = pool(x)
    
    assert isinstance(result, tuple), "Should return tuple when return_indices=True"
    assert len(result) == 2, "Tuple should have 2 elements"
    output, indices = result
    
    assert isinstance(output, nt.Tensor), "First element should be Tensor"
    assert isinstance(indices, np.ndarray), "Second element should be numpy array"
    assert output.shape == (2, 3, 2, 2), f"Output shape mismatch: {output.shape}"
    assert indices.shape == (2, 3, 2, 2), f"Indices shape mismatch: {indices.shape}"
    assert indices.dtype == np.int32, f"Indices dtype should be int32, got {indices.dtype}"
    
    # Test 2: Verify indices correspond to max positions
    # For each position in output, the index should point to the max value in the input window
    x_data = x.data
    K_H, K_W = 2, 2
    stride_h, stride_w = 2, 2
    N, C, H_in, W_in = x.shape
    H_out, W_out = output.shape[2], output.shape[3]
    
    for n in range(N):
        for c in range(C):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start = h_out * stride_h
                    w_start = w_out * stride_w
                    window = x_data[n, c, h_start:h_start+K_H, w_start:w_start+K_W]
                    expected_max = np.max(window)
                    actual_max = output.data[n, c, h_out, w_out]
                    assert np.allclose(expected_max, actual_max), "Output max mismatch"
                    
                    # Find position of max in window
                    window_flat = window.flatten()
                    max_pos_flat = np.argmax(window_flat)
                    # The stored index is absolute position in padded input (no padding here)
                    # We can verify that the indexed value matches the max
                    idx = indices[n, c, h_out, w_out]
                    # Decode flat index to coordinates (assuming no padding)
                    total_h = H_in
                    total_w = W_in
                    # Decode: idx = n * (C*H*W) + c * (H*W) + h * W + w
                    # Since we know n and c, we can compute h and w
                    idx_within_channel = idx - (n * (C * total_h * total_w) + c * (total_h * total_w))
                    h_idx = idx_within_channel // total_w
                    w_idx = idx_within_channel % total_w
                    # Check that this position is within the window
                    assert h_start <= h_idx < h_start + K_H, f"h_idx {h_idx} not in window"
                    assert w_start <= w_idx < w_start + K_W, f"w_idx {w_idx} not in window"
                    # Value at indexed position should be the max
                    indexed_val = x_data[n, c, h_idx, w_idx]
                    assert np.allclose(indexed_val, expected_max), "Indexed value not max"
    
    # Test 3: Gradient flow still works with return_indices=True
    pool_grad = MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    x_grad = nt.Tensor.randn((1, 2, 4, 4), requires_grad=True)
    out_grad, idx_grad = pool_grad(x_grad)
    loss = out_grad.sum()
    loss.backward()
    
    assert x_grad.grad is not None
    assert x_grad.grad.shape == x_grad.shape
    # Gradient should only flow to max positions
    nonzero_grad = np.count_nonzero(x_grad.grad.data)
    total_elements = x_grad.grad.data.size
    # Expect exactly H_out * W_out * C * N nonzero gradients (one per output element)
    expected_nonzero = out_grad.shape[0] * out_grad.shape[1] * out_grad.shape[2] * out_grad.shape[3]
    assert nonzero_grad == expected_nonzero, f"Expected {expected_nonzero} nonzero gradients, got {nonzero_grad}"
    
    print("✓ test_maxpool2d_return_indices passed")


def run_all_tests():
    """Run all pooling tests."""
    test_maxpool2d_creation()
    test_maxpool2d_forward()
    test_maxpool2d_gradient_flow()
    test_maxpool2d_padding()
    test_maxpool2d_ceil_mode()
    test_maxpool2d_gradient_correctness()

    test_avgpool2d_creation()
    test_avgpool2d_forward()
    test_avgpool2d_gradient_flow()
    test_avgpool2d_padding()
    test_avgpool2d_ceil_mode()
    test_avgpool2d_gradient_correctness()

    test_functional_max_pool2d()
    test_functional_avg_pool2d()
    test_pooling_edge_cases()
    test_pooling_invalid_input()
    test_maxpool2d_return_indices()

    print("\n✅ All pooling tests passed!")


if __name__ == "__main__":
    run_all_tests()
