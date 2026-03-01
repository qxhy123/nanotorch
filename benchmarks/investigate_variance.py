#!/usr/bin/env python3
"""Investigate performance variance in gradient benchmarks."""

import numpy as np
import time
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanotorch.tensor import Tensor

def benchmark_with_gc_control(size_a=(256, 512), size_b=(512, 256), repeats=30):
    """Benchmark with garbage collection control."""
    print(f"\n=== Benchmark with GC control (A={size_a}, B={size_b}) ===")
    
    np_a = np.random.randn(*size_a).astype(np.float32)
    np_b = np.random.randn(*size_b).astype(np.float32)
    
    # Warm up
    print("Warming up...")
    for _ in range(10):
        a = Tensor(np_a, requires_grad=True)
        b = Tensor(np_b, requires_grad=True)
        result = a @ b
        result.backward()
    
    # Disable GC
    print("Running with GC disabled...")
    gc.disable()
    try:
        times = []
        for i in range(repeats):
            a = Tensor(np_a, requires_grad=True)
            b = Tensor(np_b, requires_grad=True)
            start = time.perf_counter()
            result = a @ b
            result.backward()
            times.append(time.perf_counter() - start)
            if i % 10 == 0:
                print(f"  Iteration {i}: {times[-1]*1000:.3f} ms")
    finally:
        gc.enable()
    
    median_time = np.median(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"Median: {median_time*1000:.3f} ms")
    print(f"Std: {std_time*1000:.3f} ms")
    print(f"Min: {min_time*1000:.3f} ms")
    print(f"Max: {max_time*1000:.3f} ms")
    print(f"Range: {max_time/min_time:.2f}x")
    
    return times

def benchmark_numpy_matmul(size_a=(256, 512), size_b=(512, 256), repeats=30):
    """Benchmark NumPy matmul for comparison."""
    print(f"\n=== NumPy matmul benchmark (A={size_a}, B={size_b}) ===")
    
    np_a = np.random.randn(*size_a).astype(np.float32)
    np_b = np.random.randn(*size_b).astype(np.float32)
    
    # Warm up
    for _ in range(10):
        _ = np_a @ np_b
    
    times = []
    for i in range(repeats):
        start = time.perf_counter()
        _ = np_a @ np_b
        times.append(time.perf_counter() - start)
    
    median_time = np.median(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"Median: {median_time*1000:.3f} ms")
    print(f"Std: {std_time*1000:.3f} ms")
    print(f"Min: {min_time*1000:.3f} ms")
    print(f"Max: {max_time*1000:.3f} ms")
    print(f"Range: {max_time/min_time:.2f}x")
    
    return times

def benchmark_gradient_computation_only(size_a=(256, 512), size_b=(512, 256), repeats=30):
    """Benchmark only gradient computation (reusing forward result)."""
    print(f"\n=== Gradient-only benchmark (A={size_a}, B={size_b}) ===")
    
    np_a = np.random.randn(*size_a).astype(np.float32)
    np_b = np.random.randn(*size_b).astype(np.float32)
    
    # Create tensors once
    a = Tensor(np_a, requires_grad=True)
    b = Tensor(np_b, requires_grad=True)
    result = a @ b
    
    # Warm up
    for _ in range(10):
        result.backward()
        a.zero_grad()
        b.zero_grad()
    
    times = []
    for i in range(repeats):
        start = time.perf_counter()
        result.backward()
        times.append(time.perf_counter() - start)
        a.zero_grad()
        b.zero_grad()
        if i % 10 == 0:
            print(f"  Iteration {i}: {times[-1]*1000:.3f} ms")
    
    median_time = np.median(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"Median: {median_time*1000:.3f} ms")
    print(f"Std: {std_time*1000:.3f} ms")
    print(f"Min: {min_time*1000:.3f} ms")
    print(f"Max: {max_time*1000:.3f} ms")
    print(f"Range: {max_time/min_time:.2f}x")
    
    return times

def analyze_variance():
    """Run all variance analyses."""
    print("Investigating performance variance in nanotorch gradients")
    print("=" * 60)
    
    # Benchmark NumPy baseline
    numpy_times = benchmark_numpy_matmul()
    
    # Benchmark nanotorch with GC disabled
    nt_times = benchmark_with_gc_control()
    
    # Benchmark gradient-only
    grad_times = benchmark_gradient_computation_only()
    
    print("\n=== Variance Analysis Summary ===")
    print(f"NumPy matmul: {np.std(numpy_times)/np.median(numpy_times)*100:.1f}% relative std")
    print(f"nanotorch total: {np.std(nt_times)/np.median(nt_times)*100:.1f}% relative std")
    print(f"Gradient-only: {np.std(grad_times)/np.median(grad_times)*100:.1f}% relative std")
    
    # Check for outliers
    def detect_outliers(times):
        median = np.median(times)
        mad = np.median(np.abs(times - median))
        if mad == 0:
            return []
        # Use modified Z-score > 3.5 as outlier
        modified_z_scores = 0.6745 * (times - median) / mad
        return np.where(np.abs(modified_z_scores) > 3.5)[0]
    
    numpy_outliers = detect_outliers(numpy_times)
    nt_outliers = detect_outliers(nt_times)
    grad_outliers = detect_outliers(grad_times)
    
    print(f"NumPy outliers: {len(numpy_outliers)}/{len(numpy_times)}")
    print(f"nanotorch outliers: {len(nt_outliers)}/{len(nt_times)}")
    print(f"Gradient-only outliers: {len(grad_outliers)}/{len(grad_times)}")
    
    return {
        "numpy_times": numpy_times,
        "nt_times": nt_times,
        "grad_times": grad_times,
        "numpy_outliers": numpy_outliers,
        "nt_outliers": nt_outliers,
        "grad_outliers": grad_outliers,
    }

if __name__ == "__main__":
    analyze_variance()