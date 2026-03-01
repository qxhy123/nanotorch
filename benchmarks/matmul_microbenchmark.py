#!/usr/bin/env python3
"""Microbenchmark matrix multiplications in forward vs backward pass."""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanotorch.tensor import Tensor

def warmup_cpu():
    """Warm up CPU by doing some heavy computation."""
    print("Warming up CPU...")
    a = np.random.randn(1000, 1000).astype(np.float32)
    b = np.random.randn(1000, 1000).astype(np.float32)
    for _ in range(5):
        _ = a @ b
    print("CPU warmed up")

def benchmark_forward_vs_backward_operations(size_a=(256, 512), size_b=(512, 256), repeats=100):
    """Benchmark the actual matrix multiplications in forward vs backward."""
    print(f"\n=== Forward vs Backward MatMul Operations (A={size_a}, B={size_b}) ===")
    
    np_a = np.random.randn(*size_a).astype(np.float32)
    np_b = np.random.randn(*size_b).astype(np.float32)
    
    # Create tensors to get gradient shape
    a = Tensor(np_a, requires_grad=True)
    b = Tensor(np_b, requires_grad=True)
    result = a @ b
    
    # Compute gradient (ones)
    result.backward()
    
    # Get numpy arrays
    grad = result.grad.data  # shape (256, 256)
    a_data = a.data
    b_data = b.data
    
    print(f"Shapes: a={a_data.shape}, b={b_data.shape}, grad={grad.shape}")
    print(f"Forward: a @ b = {a_data.shape} @ {b_data.shape} = {result.shape}")
    print(f"Backward grad_a: grad @ b.T = {grad.shape} @ {b_data.T.shape} = {grad.shape} @ {b_data.T.shape}")
    print(f"Backward grad_b: a.T @ grad = {a_data.T.shape} @ {grad.shape} = {a_data.T.shape} @ {grad.shape}")
    
    # Benchmark forward operation
    forward_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = a_data @ b_data
        forward_times.append(time.perf_counter() - start)
    
    # Benchmark backward operation 1: grad @ b.T
    backward1_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = grad @ b_data.T
        backward1_times.append(time.perf_counter() - start)
    
    # Benchmark backward operation 2: a.T @ grad
    backward2_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = a_data.T @ grad
        backward2_times.append(time.perf_counter() - start)
    
    # Benchmark combined backward (both operations)
    backward_combined_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = grad @ b_data.T
        _ = a_data.T @ grad
        backward_combined_times.append(time.perf_counter() - start)
    
    # Calculate statistics
    def stats(times):
        times = np.array(times)
        median = np.median(times)
        std = np.std(times)
        min_val = np.min(times)
        max_val = np.max(times)
        return median, std, min_val, max_val
    
    f_median, f_std, f_min, f_max = stats(forward_times)
    b1_median, b1_std, b1_min, b1_max = stats(backward1_times)
    b2_median, b2_std, b2_min, b2_max = stats(backward2_times)
    bc_median, bc_std, bc_min, bc_max = stats(backward_combined_times)
    
    print("\n--- Results ---")
    print(f"Forward (a @ b):")
    print(f"  Median: {f_median*1000:.3f} ms, Std: {f_std*1000:.3f} ms")
    print(f"  Min: {f_min*1000:.3f} ms, Max: {f_max*1000:.3f} ms")
    
    print(f"\nBackward 1 (grad @ b.T):")
    print(f"  Median: {b1_median*1000:.3f} ms, Std: {b1_std*1000:.3f} ms")
    print(f"  Min: {b1_min*1000:.3f} ms, Max: {b1_max*1000:.3f} ms")
    print(f"  Relative to forward: {b1_median/f_median:.2f}x")
    
    print(f"\nBackward 2 (a.T @ grad):")
    print(f"  Median: {b2_median*1000:.3f} ms, Std: {b2_std*1000:.3f} ms")
    print(f"  Min: {b2_min*1000:.3f} ms, Max: {b2_max*1000:.3f} ms")
    print(f"  Relative to forward: {b2_median/f_median:.2f}x")
    
    print(f"\nBackward combined (both):")
    print(f"  Median: {bc_median*1000:.3f} ms, Std: {bc_std*1000:.3f} ms")
    print(f"  Min: {bc_min*1000:.3f} ms, Max: {bc_max*1000:.3f} ms")
    print(f"  Relative to forward: {bc_median/f_median:.2f}x")
    print(f"  Expected if independent: {(b1_median + b2_median)/f_median:.2f}x")
    
    # Check if transposes affect performance
    print(f"\n--- Transpose Analysis ---")
    # Precompute transposes
    b_T = b_data.T
    a_T = a_data.T
    
    # Benchmark with precomputed transpose
    b_T_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = grad @ b_T
        b_T_times.append(time.perf_counter() - start)
    
    a_T_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = a_T @ grad
        a_T_times.append(time.perf_counter() - start)
    
    b_T_median = np.median(b_T_times)
    a_T_median = np.median(a_T_times)
    
    print(f"With precomputed transpose:")
    print(f"  grad @ b.T: {b_T_median*1000:.3f} ms (vs {b1_median*1000:.3f} ms, diff: {(b_T_median - b1_median)/b1_median*100:+.1f}%)")
    print(f"  a.T @ grad: {a_T_median*1000:.3f} ms (vs {b2_median*1000:.3f} ms, diff: {(a_T_median - b2_median)/b2_median*100:+.1f}%)")
    
    return {
        "forward_median_ms": f_median * 1000,
        "backward1_median_ms": b1_median * 1000,
        "backward2_median_ms": b2_median * 1000,
        "backward_combined_median_ms": bc_median * 1000,
        "relative_backward1": b1_median / f_median,
        "relative_backward2": b2_median / f_median,
        "relative_combined": bc_median / f_median,
    }

def benchmark_different_sizes():
    """Benchmark with different matrix sizes."""
    sizes = [
        ((256, 512), (512, 256)),  # Medium
        ((100, 50), (50, 30)),     # Small
        ((10, 5), (5, 2)),         # Very small
        ((512, 1024), (1024, 512)), # Large
    ]
    
    results = []
    for size_a, size_b in sizes:
        warmup_cpu()
        res = benchmark_forward_vs_backward_operations(size_a, size_b, repeats=50)
        results.append((size_a, size_b, res))
    
    print("\n=== Size Comparison Summary ===")
    for size_a, size_b, res in results:
        print(f"{size_a} @ {size_b}:")
        print(f"  Forward: {res['forward_median_ms']:.3f} ms")
        print(f"  Backward combined: {res['backward_combined_median_ms']:.3f} ms ({res['relative_combined']:.2f}x forward)")
        print()

if __name__ == "__main__":
    print("Matrix Multiplication Microbenchmark")
    print("=" * 60)
    warmup_cpu()
    benchmark_forward_vs_backward_operations()
    benchmark_different_sizes()