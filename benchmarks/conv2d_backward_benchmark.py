"""Benchmark Conv2D backward pass performance with vectorized input gradient."""
import numpy as np
import time
from nanotorch import Tensor
from nanotorch.autograd import Conv2DFunction

def benchmark_conv2d_backward(N, C_in, H_in, W_in, C_out, K_H, K_W, stride, padding, dilation, repeats=10):
    H_out = (H_in + 2 * padding - dilation * (K_H - 1) - 1) // stride + 1
    W_out = (W_in + 2 * padding - dilation * (K_W - 1) - 1) // stride + 1
    
    # Create tensors with requires_grad
    input_tensor = Tensor.randn((N, C_in, H_in, W_in), requires_grad=True)
    weight_tensor = Tensor.randn((C_out, C_in, K_H, K_W), requires_grad=True)
    bias_tensor = Tensor.randn((C_out, 1, 1), requires_grad=True)
    
    # Forward pass
    output = Conv2DFunction.apply(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation)
    
    # Create gradient for output
    grad_output = Tensor.ones_like(output)
    
    # Warmup
    for _ in range(5):
        input_tensor.zero_grad()
        weight_tensor.zero_grad()
        bias_tensor.zero_grad()
        output.backward(grad_output)
    
    # Benchmark backward pass
    start = time.perf_counter()
    for _ in range(repeats):
        input_tensor.zero_grad()
        weight_tensor.zero_grad()
        bias_tensor.zero_grad()
        output.backward(grad_output)
    elapsed = (time.perf_counter() - start) / repeats * 1000
    
    return elapsed

def main():
    print("Conv2D Backward Pass Benchmark")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        ("Small (dilation=1)", 2, 3, 8, 8, 4, 3, 3, 1, 0, 1),
        ("Medium (dilation=1)", 4, 16, 32, 32, 32, 3, 3, 1, 1, 1),
        ("Large (dilation=1)", 8, 64, 64, 64, 128, 3, 3, 1, 1, 1),
        ("Stride 2 (dilation=1)", 4, 16, 32, 32, 32, 3, 3, 2, 0, 1),
        ("Dilation 2 (non-vectorized)", 4, 16, 32, 32, 32, 3, 3, 1, 0, 2),
    ]
    
    for name, *params in test_cases:
        N, C_in, H_in, W_in, C_out, K_H, K_W, stride, padding, dilation = params
        time_ms = benchmark_conv2d_backward(N, C_in, H_in, W_in, C_out, K_H, K_W, stride, padding, dilation, repeats=5)
        print(f"{name:30} {time_ms:7.2f} ms")
        
        # Calculate operations count for context
        H_out = (H_in + 2 * padding - dilation * (K_H - 1) - 1) // stride + 1
        W_out = (W_in + 2 * padding - dilation * (K_W - 1) - 1) // stride + 1
        ops = N * C_out * H_out * W_out * C_in * K_H * K_W * 2  # Approximate FLOPs
        print(f"  Output shape: ({N}, {C_out}, {H_out}, {W_out})")
        print(f"  Approx ops: {ops / 1e6:.1f} MFLOPs")
        print()

if __name__ == "__main__":
    main()