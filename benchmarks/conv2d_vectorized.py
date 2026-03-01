"""Test vectorized Conv2D implementation using sliding_window_view."""
import numpy as np
import time
from nanotorch import Tensor
from nanotorch.autograd import Conv2DFunction

def conv2d_vectorized_forward(input_data, weight_data, bias_data=None, stride=1, padding=0, dilation=1):
    """Vectorized forward pass using sliding_window_view."""
    N, C_in, H_in, W_in = input_data.shape
    C_out, C_in_w, K_H, K_W = weight_data.shape
    assert C_in == C_in_w
    
    # Calculate output dimensions
    kernel_size_dilated_h = dilation * (K_H - 1) + 1
    kernel_size_dilated_w = dilation * (K_W - 1) + 1
    H_out = (H_in + 2 * padding - kernel_size_dilated_h) // stride + 1
    W_out = (W_in + 2 * padding - kernel_size_dilated_w) // stride + 1
    
    # Pad input if needed
    if padding > 0:
        padded = np.pad(input_data, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
    else:
        padded = input_data
    
    # Use sliding_window_view to extract windows
    # Note: sliding_window_view is available in numpy >= 1.20
    from numpy.lib.stride_tricks import sliding_window_view
    # Shape (N, C_in, H_out, W_out, K_H, K_W)
    windows = sliding_window_view(padded, (K_H, K_W), axis=(-2, -1))
    # Subsampling with stride
    windows = windows[:, :, ::stride, ::stride, :, :]
    # Now windows shape (N, C_in, H_out, W_out, K_H, K_W)
    # Reshape to (N, H_out, W_out, C_in, K_H, K_W) then flatten last three dimensions
    windows_flat = windows.transpose(0, 2, 3, 1, 4, 5).reshape(N * H_out * W_out, C_in * K_H * K_W)
    
    # Reshape weight to (C_out, C_in * K_H * K_W)
    weight_flat = weight_data.reshape(C_out, -1)
    
    # Matrix multiplication
    output_flat = windows_flat @ weight_flat.T  # (N*H_out*W_out, C_out)
    output = output_flat.T.reshape(C_out, N, H_out, W_out).transpose(1, 0, 2, 3)
    
    # Add bias
    if bias_data is not None:
        output += bias_data.reshape(1, C_out, 1, 1)
    
    return output

def test_correctness():
    np.random.seed(42)
    N, C_in, H, W = 2, 3, 8, 8
    C_out, K_H, K_W = 4, 3, 3
    stride = 1
    padding = 1
    
    input_np = np.random.randn(N, C_in, H, W).astype(np.float32)
    weight_np = np.random.randn(C_out, C_in, K_H, K_W).astype(np.float32)
    bias_np = np.random.randn(C_out, 1, 1).astype(np.float32)
    
    # Original Conv2DFunction forward
    input_tensor = Tensor(input_np, requires_grad=False)
    weight_tensor = Tensor(weight_np, requires_grad=False)
    bias_tensor = Tensor(bias_np, requires_grad=False)
    
    output_tensor = Conv2DFunction.apply(input_tensor, weight_tensor, bias_tensor, stride, padding, 1)
    output_original = output_tensor.data
    
    # Vectorized forward
    output_vectorized = conv2d_vectorized_forward(input_np, weight_np, bias_np, stride, padding, 1)
    
    diff = np.abs(output_original - output_vectorized).max()
    print(f"Max difference: {diff}")
    assert diff < 1e-4, f"Difference too large: {diff}"
    print("Correctness test passed.")
    
    # Performance comparison
    import time
    repeats = 50
    # Warmup
    for _ in range(10):
        _ = Conv2DFunction.apply(input_tensor, weight_tensor, bias_tensor, stride, padding, 1)
        _ = conv2d_vectorized_forward(input_np, weight_np, bias_np, stride, padding, 1)
    
    # Time original
    start = time.perf_counter()
    for _ in range(repeats):
        _ = Conv2DFunction.apply(input_tensor, weight_tensor, bias_tensor, stride, padding, 1)
    t_original = (time.perf_counter() - start) / repeats * 1000
    
    # Time vectorized
    start = time.perf_counter()
    for _ in range(repeats):
        _ = conv2d_vectorized_forward(input_np, weight_np, bias_np, stride, padding, 1)
    t_vectorized = (time.perf_counter() - start) / repeats * 1000
    
    print(f"Original forward: {t_original:.3f} ms")
    print(f"Vectorized forward: {t_vectorized:.3f} ms")
    print(f"Speedup: {t_original/t_vectorized:.2f}x")

if __name__ == "__main__":
    test_correctness()