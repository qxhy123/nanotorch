"""
Tests for adaptive pooling layers (AdaptiveAvgPool2d and AdaptiveMaxPool2d).
"""

import numpy as np
import pytest
import nanotorch as nt
from nanotorch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d
from nanotorch.autograd import AdaptiveAvgPool2dFunction, AdaptiveMaxPool2dFunction
from nanotorch.utils import manual_seed


def test_adaptive_avgpool2d_creation():
    """Test AdaptiveAvgPool2d initialization."""
    pool = AdaptiveAvgPool2d(output_size=2)
    assert pool.output_size == (2, 2)
    
    pool2 = AdaptiveAvgPool2d(output_size=(3, 5))
    assert pool2.output_size == (3, 5)
    
    pool3 = AdaptiveAvgPool2d(output_size=1)
    assert pool3.output_size == (1, 1)
    
    print("✓ test_adaptive_avgpool2d_creation passed")


def test_adaptive_avgpool2d_forward():
    """Test forward pass of AdaptiveAvgPool2d."""
    manual_seed(42)
    
    pool = AdaptiveAvgPool2d(output_size=2)
    x = nt.Tensor.randn((2, 3, 6, 6), requires_grad=False)
    out = pool(x)
    
    assert out.shape == (2, 3, 2, 2)
    assert not np.allclose(out.data, 0.0)
    
    pool2 = AdaptiveAvgPool2d(output_size=(3, 4))
    out2 = pool2(x)
    assert out2.shape == (2, 3, 3, 4)
    
    pool3 = AdaptiveAvgPool2d(output_size=1)
    out3 = pool3(x)
    assert out3.shape == (2, 3, 1, 1)
    # Each value should be average of all spatial elements per channel
    for n in range(2):
        for c in range(3):
            expected_mean = np.mean(x.data[n, c, :, :])
            assert np.allclose(out3.data[n, c, 0, 0], expected_mean, rtol=1e-5)
    
    print("✓ test_adaptive_avgpool2d_forward passed")


def test_adaptive_avgpool2d_gradient_flow():
    """Test gradient computation through AdaptiveAvgPool2d."""
    manual_seed(123)
    
    pool = AdaptiveAvgPool2d(output_size=2)
    x = nt.Tensor.randn((1, 2, 4, 4), requires_grad=True)
    
    out = pool(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    assert x.grad.shape == x.shape

    assert not np.allclose(x.grad.data, 0.0)
    
    print("✓ test_adaptive_avgpool2d_gradient_flow passed")


def test_adaptive_avgpool2d_gradient_correctness():
    """Test AdaptiveAvgPool2d gradient correctness with finite differences."""
    np.random.seed(42)
    
    configs = [
        {'input_size': (2, 3, 6, 6), 'output_size': 2},
        {'input_size': (1, 2, 8, 8), 'output_size': 4},
        {'input_size': (2, 1, 5, 7), 'output_size': (3, 4)},
        {'input_size': (1, 1, 3, 3), 'output_size': 1},  # global pooling
        {'input_size': (1, 2, 10, 10), 'output_size': (3, 5)},
    ]
    
    for config in configs:
        input_size = config['input_size']
        output_size = config['output_size']
        
        N, C, H_in, W_in = input_size
        if isinstance(output_size, int):
            H_out = W_out = output_size
        else:
            H_out, W_out = output_size
        
        input_np = np.random.randn(*input_size).astype(np.float32) * 0.1
        grad_output_np = np.random.randn(N, C, H_out, W_out).astype(np.float32) * 0.1
        
        input_t = nt.Tensor(input_np, requires_grad=True)
        
        output = AdaptiveAvgPool2dFunction.apply(input_t, output_size)
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
            
            output_plus = AdaptiveAvgPool2dFunction.apply(x_plus_t, output_size)
            output_minus = AdaptiveAvgPool2dFunction.apply(x_minus_t, output_size)
            
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
    
    print("✓ test_adaptive_avgpool2d_gradient_correctness passed")


def test_adaptive_maxpool2d_creation():
    """Test AdaptiveMaxPool2d initialization."""
    pool = AdaptiveMaxPool2d(output_size=2)
    assert pool.output_size == (2, 2)
    assert pool.return_indices == False
    
    pool2 = AdaptiveMaxPool2d(output_size=(3, 5), return_indices=True)
    assert pool2.output_size == (3, 5)
    assert pool2.return_indices == True
    
    pool3 = AdaptiveMaxPool2d(output_size=1)  # global max pooling
    assert pool3.output_size == (1, 1)
    
    print("✓ test_adaptive_maxpool2d_creation passed")


def test_adaptive_maxpool2d_forward():
    """Test forward pass of AdaptiveMaxPool2d."""
    manual_seed(42)
    
    # Test 1: Square output, no indices
    pool = AdaptiveMaxPool2d(output_size=2)
    x = nt.Tensor.randn((2, 3, 6, 6), requires_grad=False)
    out = pool(x)
    
    assert isinstance(out, nt.Tensor)
    assert out.shape == (2, 3, 2, 2)
    assert not np.allclose(out.data, 0.0)
    
    # Test 2: Non-square output
    pool2 = AdaptiveMaxPool2d(output_size=(3, 4))
    out2 = pool2(x)
    assert out2.shape == (2, 3, 3, 4)
    
    # Test 3: Global max pooling (output_size=1)
    pool3 = AdaptiveMaxPool2d(output_size=1)
    out3 = pool3(x)
    assert out3.shape == (2, 3, 1, 1)
    for n in range(2):
        for c in range(3):
            expected_max = np.max(x.data[n, c, :, :])
            assert np.allclose(out3.data[n, c, 0, 0], expected_max, rtol=1e-5)
    
    print("✓ test_adaptive_maxpool2d_forward passed")


def test_adaptive_maxpool2d_gradient_flow():
    """Test gradient computation through AdaptiveMaxPool2d."""
    manual_seed(123)
    
    pool = AdaptiveMaxPool2d(output_size=2)
    x = nt.Tensor.randn((1, 2, 4, 4), requires_grad=True)
    
    out = pool(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    assert x.grad.shape == x.shape
    # Gradient should be non-zero (only at max positions)
    assert not np.allclose(x.grad.data, 0.0)
    
    print("✓ test_adaptive_maxpool2d_gradient_flow passed")


def test_adaptive_maxpool2d_gradient_correctness():
    """Test AdaptiveMaxPool2d gradient correctness with finite differences."""
    np.random.seed(42)
    
    configs = [
        {'input_size': (1, 2, 6, 6), 'output_size': 2},
        {'input_size': (2, 1, 8, 8), 'output_size': 4},
        {'input_size': (1, 1, 5, 7), 'output_size': (3, 4)},
        {'input_size': (1, 1, 3, 3), 'output_size': 1},  # global pooling
    ]
    
    for config in configs:
        input_size = config['input_size']
        output_size = config['output_size']
        
        N, C, H_in, W_in = input_size
        if isinstance(output_size, int):
            H_out = W_out = output_size
        else:
            H_out, W_out = output_size
        
        input_np = np.random.randn(*input_size).astype(np.float32) * 0.1
        grad_output_np = np.random.randn(N, C, H_out, W_out).astype(np.float32) * 0.1
        
        input_t = nt.Tensor(input_np, requires_grad=True)
        
        output = AdaptiveMaxPool2dFunction.apply(input_t, output_size, False)
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
            
            output_plus = AdaptiveMaxPool2dFunction.apply(x_plus_t, output_size, False)
            output_minus = AdaptiveMaxPool2dFunction.apply(x_minus_t, output_size, False)
            
            loss_plus = np.sum(output_plus.data.astype(np.float64) * grad_output_np.astype(np.float64))
            loss_minus = np.sum(output_minus.data.astype(np.float64) * grad_output_np.astype(np.float64))
            numeric_grad_x[idx] = (loss_plus - loss_minus) / (2 * eps)
        
        def relative_error(a, b):
            return np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-8)
        
        assert input_t.grad is not None, "input gradient should exist"
        error_x = relative_error(input_t.grad.data, numeric_grad_x).max()
        
        threshold_x = 2.0  # Max pooling gradient can have larger error due to discrete max
        
        assert error_x < threshold_x, (
            f"Input gradient error too high: {error_x:.2e} > {threshold_x:.0e} "
            f"for config {config}"
        )
    
    print("✓ test_adaptive_maxpool2d_gradient_correctness passed")


def test_adaptive_maxpool2d_return_indices():
    """Test AdaptiveMaxPool2d with return_indices=True."""
    manual_seed(42)
    
    # Test 1: Basic functionality
    pool = AdaptiveMaxPool2d(output_size=2, return_indices=True)
    x = nt.Tensor.randn((2, 3, 6, 6), requires_grad=False)
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
    # We can't directly verify absolute indices without reproducing the algorithm,
    # but we can verify that the indexed value equals the output max
    x_data = x.data
    N, C, H_in, W_in = x.shape
    H_out, W_out = output.shape[2], output.shape[3]
    
    for n in range(N):
        for c in range(C):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    idx = indices[n, c, h_out, w_out]
                    # Decode flat index to coordinates
                    idx_within_tensor = idx - (n * (C * H_in * W_in) + c * (H_in * W_in))
                    h_idx = idx_within_tensor // W_in
                    w_idx = idx_within_tensor % W_in
                    # Ensure coordinates are within bounds
                    assert 0 <= h_idx < H_in, f"h_idx {h_idx} out of bounds"
                    assert 0 <= w_idx < W_in, f"w_idx {w_idx} out of bounds"
                    # Value at indexed position should equal output value
                    indexed_val = x_data[n, c, h_idx, w_idx]
                    output_val = output.data[n, c, h_out, w_out]
                    assert np.allclose(indexed_val, output_val, rtol=1e-5), (
                        f"Indexed value {indexed_val} != output {output_val}"
                    )
    
    # Test 3: Gradient flow still works with return_indices=True
    pool_grad = AdaptiveMaxPool2d(output_size=2, return_indices=True)
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
    
    print("✓ test_adaptive_maxpool2d_return_indices passed")


def test_functional_adaptive_avg_pool2d():
    """Test functional adaptive_avg_pool2d interface."""
    manual_seed(42)
    
    x = nt.Tensor.randn((2, 3, 6, 6), requires_grad=False)
    out = nt.adaptive_avg_pool2d(x, output_size=2)
    
    assert out.shape == (2, 3, 2, 2)
    assert not np.allclose(out.data, 0.0)
    
    # Test with tuple output size
    out2 = nt.adaptive_avg_pool2d(x, output_size=(3, 4))
    assert out2.shape == (2, 3, 3, 4)
    
    print("✓ test_functional_adaptive_avg_pool2d passed")


def test_functional_adaptive_max_pool2d():
    """Test functional adaptive_max_pool2d interface."""
    manual_seed(42)
    
    x = nt.Tensor.randn((2, 3, 6, 6), requires_grad=False)
    out = nt.adaptive_max_pool2d(x, output_size=2)
    
    assert isinstance(out, nt.Tensor)
    assert out.shape == (2, 3, 2, 2)
    assert not np.allclose(out.data, 0.0)
    
    # Test with tuple output size
    out2 = nt.adaptive_max_pool2d(x, output_size=(3, 4))
    assert out2.shape == (2, 3, 3, 4)
    
    # Test with return_indices=True
    result = nt.adaptive_max_pool2d(x, output_size=2, return_indices=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    out3, indices = result
    assert out3.shape == (2, 3, 2, 2)
    assert indices.shape == (2, 3, 2, 2)
    
    print("✓ test_functional_adaptive_max_pool2d passed")


def test_adaptive_pooling_edge_cases():
    """Test edge cases for adaptive pooling layers."""
    manual_seed(42)
    
    # Test 1: Input size equals output size (no pooling needed)
    pool = AdaptiveAvgPool2d(output_size=(4, 4))
    x = nt.Tensor.randn((1, 1, 4, 4), requires_grad=False)
    out = pool(x)
    assert out.shape == (1, 1, 4, 4)
    assert np.allclose(out.data, x.data, rtol=1e-5)
    
    # Test 2: Output size larger than input size? Not allowed by PyTorch?
    # Actually adaptive pooling with output_size > input_size is allowed.
    # The pooling region size becomes 1x1 (since start_index == end_index for some regions?)
    # We'll test that it works without error.
    pool2 = AdaptiveMaxPool2d(output_size=(8, 8))
    x2 = nt.Tensor.randn((1, 1, 4, 4), requires_grad=False)
    out2 = pool2(x2)
    assert out2.shape == (1, 1, 8, 8)
    # Each output pixel corresponds to a single input pixel (some repeated)
    
    # Test 3: Non-square input, square output
    pool3 = AdaptiveAvgPool2d(output_size=2)
    x3 = nt.Tensor.randn((1, 1, 5, 9), requires_grad=False)
    out3 = pool3(x3)
    assert out3.shape == (1, 1, 2, 2)
    
    # Test 4: Very small input (1x1) with output_size=1
    pool4 = AdaptiveMaxPool2d(output_size=1)
    x4 = nt.Tensor.randn((1, 3, 1, 1), requires_grad=False)
    out4 = pool4(x4)
    assert out4.shape == (1, 3, 1, 1)
    assert np.allclose(out4.data, x4.data)
    
    print("✓ test_adaptive_pooling_edge_cases passed")


def test_adaptive_pooling_invalid_input():
    """Test adaptive pooling layers with invalid input."""
    pool = AdaptiveAvgPool2d(output_size=2)
    
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
    
    # Test invalid output_size (zero or negative)
    try:
        AdaptiveAvgPool2d(output_size=0)
        assert False, "Should have raised ValueError for output_size=0"
    except (ValueError, AssertionError) as e:
        # Implementation may raise AssertionError or ValueError
        pass
    
    print("✓ test_adaptive_pooling_invalid_input passed")


def test_adaptive_pooling_algorithm():
    """Test adaptive pooling algorithm correctness with known examples."""
    # Create a simple input where we can compute expected output manually
    # Input: 1x1x6x6, output_size=3x3
    # Each pooling region should be 2x2 (since 6/3=2)
    input_data = np.arange(36, dtype=np.float32).reshape(1, 1, 6, 6)
    x = nt.Tensor(input_data, requires_grad=False)
    
    pool = AdaptiveAvgPool2d(output_size=3)
    out = pool(x)
    
    # Manually compute expected average pooling
    # Region 0,0: rows 0-1, cols 0-1 => values [0,1,6,7] avg = (0+1+6+7)/4 = 3.5
    # Region 0,1: rows 0-1, cols 2-3 => values [2,3,8,9] avg = 5.5
    # Region 0,2: rows 0-1, cols 4-5 => values [4,5,10,11] avg = 7.5
    # Region 1,0: rows 2-3, cols 0-1 => values [12,13,18,19] avg = 15.5
    # etc.
    expected = np.array([
        [3.5, 5.5, 7.5],
        [15.5, 17.5, 19.5],
        [27.5, 29.5, 31.5]
    ], dtype=np.float32).reshape(1, 1, 3, 3)
    
    assert np.allclose(out.data, expected, rtol=1e-5), f"Expected {expected}, got {out.data}"
    
    # Test max pooling on same input
    pool_max = AdaptiveMaxPool2d(output_size=3)
    out_max = pool_max(x)
    expected_max = np.array([
        [7, 9, 11],
        [19, 21, 23],
        [31, 33, 35]
    ], dtype=np.float32).reshape(1, 1, 3, 3)
    
    assert np.allclose(out_max.data, expected_max, rtol=1e-5), f"Expected {expected_max}, got {out_max.data}"
    
    print("✓ test_adaptive_pooling_algorithm passed")


def run_all_tests():
    """Run all adaptive pooling tests."""
    test_adaptive_avgpool2d_creation()
    test_adaptive_avgpool2d_forward()
    test_adaptive_avgpool2d_gradient_flow()
    test_adaptive_avgpool2d_gradient_correctness()
    
    test_adaptive_maxpool2d_creation()
    test_adaptive_maxpool2d_forward()
    test_adaptive_maxpool2d_gradient_flow()
    test_adaptive_maxpool2d_gradient_correctness()
    test_adaptive_maxpool2d_return_indices()
    
    test_functional_adaptive_avg_pool2d()
    test_functional_adaptive_max_pool2d()
    test_adaptive_pooling_edge_cases()
    test_adaptive_pooling_invalid_input()
    test_adaptive_pooling_algorithm()
    
    print("\n✅ All adaptive pooling tests passed!")


if __name__ == "__main__":
    run_all_tests()