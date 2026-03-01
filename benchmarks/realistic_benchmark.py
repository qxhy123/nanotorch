#!/usr/bin/env python3
"""Realistic benchmark reusing tensors with zero_grad."""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanotorch.tensor import Tensor

def realistic_matmul_benchmark(size_a=(256, 512), size_b=(512, 256), repeats=100):
    """Benchmark with tensor reuse (like real training loop)."""
    print(f"\n=== Realistic MatMul Benchmark (A={size_a}, B={size_b}) ===")
    
    np_a = np.random.randn(*size_a).astype(np.float32)
    np_b = np.random.randn(*size_b).astype(np.float32)
    
    # Create tensors once (like model parameters and input)
    a = Tensor(np_a, requires_grad=True)
    b = Tensor(np_b, requires_grad=True)
    
    # Warm up
    for _ in range(10):
        result = a @ b
        result.backward()
        a.zero_grad()
        b.zero_grad()
    
    # Benchmark forward+backward with zero_grad
    forward_times = []
    backward_times = []
    total_times = []
    
    for i in range(repeats):
        start = time.perf_counter()
        result = a @ b
        forward_end = time.perf_counter()
        result.backward()
        backward_end = time.perf_counter()
        forward_times.append(forward_end - start)
        backward_times.append(backward_end - forward_end)
        total_times.append(backward_end - start)
        
        # Zero gradients for next iteration (like optimizer.step())
        a.zero_grad()
        b.zero_grad()
        
        if i % 20 == 0:
            print(f"  Iteration {i}: forward {forward_times[-1]*1000:.3f} ms, backward {backward_times[-1]*1000:.3f} ms")
    
    # Compare with NumPy baseline
    np_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = np_a @ np_b
        np_times.append(time.perf_counter() - start)
    
    np_median = np.median(np_times)
    forward_median = np.median(forward_times)
    backward_median = np.median(backward_times)
    total_median = np.median(total_times)
    
    print(f"\n--- Results ---")
    print(f"NumPy forward median: {np_median*1000:.3f} ms")
    print(f"nanotorch forward median: {forward_median*1000:.3f} ms")
    print(f"nanotorch backward median: {backward_median*1000:.3f} ms")
    print(f"nanotorch total median: {total_median*1000:.3f} ms")
    print(f"Overhead forward: {forward_median/np_median:.2f}x")
    print(f"Overhead total: {total_median/np_median:.2f}x")
    print(f"Gradient overhead factor: {backward_median/forward_median:.2f}x")
    
    # Also benchmark forward without gradient tracking
    a_nograd = Tensor(np_a, requires_grad=False)
    b_nograd = Tensor(np_b, requires_grad=False)
    nograd_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = a_nograd @ b_nograd
        nograd_times.append(time.perf_counter() - start)
    
    nograd_median = np.median(nograd_times)
    print(f"\nForward without gradient: {nograd_median*1000:.3f} ms")
    print(f"Overhead vs NumPy: {nograd_median/np_median:.2f}x")
    print(f"Gradient tracking overhead: {forward_median/nograd_median:.2f}x")
    
    return {
        "size_a": size_a,
        "size_b": size_b,
        "numpy_forward_ms": np_median * 1000,
        "nt_forward_ms": forward_median * 1000,
        "nt_backward_ms": backward_median * 1000,
        "nt_total_ms": total_median * 1000,
        "overhead_forward": forward_median / np_median,
        "overhead_total": total_median / np_median,
        "gradient_overhead_factor": backward_median / forward_median,
    }

def realistic_small_tensor_benchmark():
    """Benchmark small tensors with reuse."""
    print("\n=== Realistic Small Tensor Benchmark ===")
    
    np_a = np.random.randn(10, 5).astype(np.float32)
    np_b = np.random.randn(5, 2).astype(np.float32)
    
    a = Tensor(np_a, requires_grad=True)
    b = Tensor(np_b, requires_grad=True)
    
    # Warm up
    for _ in range(10):
        result = a @ b
        result.backward()
        a.zero_grad()
        b.zero_grad()
    
    repeats = 500
    total_times = []
    for i in range(repeats):
        start = time.perf_counter()
        result = a @ b
        result.backward()
        total_times.append(time.perf_counter() - start)
        a.zero_grad()
        b.zero_grad()
    
    np_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = np_a @ np_b
        np_times.append(time.perf_counter() - start)
    
    np_median = np.median(np_times)
    total_median = np.median(total_times)
    
    print(f"Small tensor (10,5) @ (5,2):")
    print(f"  NumPy forward: {np_median*1000:.3f} ms")
    print(f"  nanotorch total: {total_median*1000:.3f} ms")
    print(f"  Overhead: {total_median/np_median:.2f}x")
    
    return total_median / np_median

if __name__ == "__main__":
    print("Realistic Training Loop Benchmarks")
    print("=" * 60)
    
    # Warm up CPU
    print("Warming up CPU...")
    warmup = np.random.randn(1000, 1000).astype(np.float32)
    for _ in range(5):
        _ = warmup @ warmup
    
    results = []
    results.append(realistic_matmul_benchmark())
    results.append(realistic_matmul_benchmark(size_a=(100, 50), size_b=(50, 30)))
    results.append(realistic_matmul_benchmark(size_a=(10, 5), size_b=(5, 2)))
    
    print("\n=== Summary ===")
    for res in results:
        print(f"{res['size_a']} @ {res['size_b']}: "
              f"forward overhead {res['overhead_forward']:.2f}x, "
              f"total overhead {res['overhead_total']:.2f}x, "
              f"gradient factor {res['gradient_overhead_factor']:.2f}x")