#!/usr/bin/env python3
"""
Fixed matmul benchmark script with better methodology.

Run with: python benchmarks/benchmark_matmul_fixed.py --repeats 30
"""

import numpy as np
import time
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import nanotorch
sys.path.insert(0, str(Path(__file__).parent.parent))
from nanotorch.tensor import Tensor


def benchmark_matmul(size_a=(256, 512), size_b=(512, 256), repeats=30):
    """Benchmark matrix multiplication with improved methodology."""
    
    print(f"\n=== Matrix Multiplication Benchmark (A={size_a}, B={size_b}) ===")
    print(f"Iterations: {repeats}")
    print(f"Operations: {size_a[0] * size_a[1] * size_b[1]:,} FLOPs")
    
    # ========== NumPy Benchmark ==========
    # Fresh data for NumPy
    np_a = np.random.randn(*size_a).astype(np.float32)
    np_b = np.random.randn(*size_b).astype(np.float32)
    
    # Extended warm-up (15 operations to stabilize BLAS and CPU)
    for _ in range(15):
        _ = np_a @ np_b
    
    # NumPy with sum() (fair comparison)
    np_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = np_a @ np_b
        _ = result.sum()  # Force computation
        np_times.append(time.perf_counter() - start)
    
    # ========== nanotorch Benchmark (no grad) ==========
    # Fresh data for nanotorch
    nt_a = Tensor(np.random.randn(*size_a).astype(np.float32), requires_grad=False)
    nt_b = Tensor(np.random.randn(*size_b).astype(np.float32), requires_grad=False)
    
    # Extended warm-up
    for _ in range(15):
        result = nt_a @ nt_b
        _ = result.data.sum()
    
    nt_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = nt_a @ nt_b
        _ = result.data.sum()
        nt_times.append(time.perf_counter() - start)
    
    # ========== nanotorch Benchmark (with grad) ==========
    # Fresh data with grad tracking
    nt_a_grad = Tensor(np.random.randn(*size_a).astype(np.float32), requires_grad=True)
    nt_b_grad = Tensor(np.random.randn(*size_b).astype(np.float32), requires_grad=True)
    
    # Extended warm-up with grad
    for _ in range(15):
        result = nt_a_grad @ nt_b_grad
        _ = result.data.sum()
    
    nt_grad_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = nt_a_grad @ nt_b_grad
        _ = result.data.sum()
        nt_grad_times.append(time.perf_counter() - start)
    
    # ========== Results ==========
    print("\n--- NumPy (with sum) ---")
    print(f"  Median: {np.median(np_times)*1000:.3f} ms")
    print(f"  Mean:   {np.mean(np_times)*1000:.3f} ms")
    print(f"  Std:    {np.std(np_times)*1000:.3f} ms")
    print(f"  Min:    {np.min(np_times)*1000:.3f} ms")
    print(f"  Max:    {np.max(np_times)*1000:.3f} ms")
    
    print("\n--- nanotorch (no grad) ---")
    print(f"  Median: {np.median(nt_times)*1000:.3f} ms")
    print(f"  Mean:   {np.mean(nt_times)*1000:.3f} ms")
    print(f"  Std:    {np.std(nt_times)*1000:.3f} ms")
    print(f"  Min:    {np.min(nt_times)*1000:.3f} ms")
    print(f"  Max:    {np.max(nt_times)*1000:.3f} ms")
    
    print("\n--- nanotorch (with grad) ---")
    print(f"  Median: {np.median(nt_grad_times)*1000:.3f} ms")
    print(f"  Mean:   {np.mean(nt_grad_times)*1000:.3f} ms")
    print(f"  Std:    {np.std(nt_grad_times)*1000:.3f} ms")
    print(f"  Min:    {np.min(nt_grad_times)*1000:.3f} ms")
    print(f"  Max:    {np.max(nt_grad_times)*1000:.3f} ms")
    
    print("\n--- Overhead Comparison (median) ---")
    overhead_no_grad = np.median(nt_times) / np.median(np_times)
    overhead_grad = np.median(nt_grad_times) / np.median(np_times)
    print(f"  nanotorch (no grad): {overhead_no_grad:.2f}x overhead")
    print(f"  nanotorch (with grad): {overhead_grad:.2f}x overhead")
    
    # Interpretation
    print("\n--- Interpretation ---")
    if overhead_no_grad < 0.95:
        print("  ⚠ Note: nanotorch appears faster - likely measurement noise")
    elif overhead_no_grad > 1.5:
        print("  ⚠ Note: High overhead - may be system variance")
    else:
        print("  ✓ Note: Realistic overhead (negligible to ~50%)")
    
    print("\n" + "="*60)
    
    return {
        "operation": "matmul",
        "size_a": size_a,
        "size_b": size_b,
        "numpy_median_ms": np.median(np_times) * 1000,
        "nanotorch_median_ms": np.median(nt_times) * 1000,
        "nanotorch_grad_median_ms": np.median(nt_grad_times) * 1000,
        "overhead_no_grad": overhead_no_grad,
        "overhead_grad": overhead_grad,
        "repeats": repeats,
    }


def main():
    parser = argparse.ArgumentParser(description="Fixed matmul benchmark")
    parser.add_argument("--repeats", type=int, default=30, help="Number of repetitions")
    parser.add_argument("--size-a", type=int, nargs=2, default=[256, 512], help="Matrix A size (rows cols)")
    parser.add_argument("--size-b", type=int, nargs=2, default=[512, 256], help="Matrix B size (rows cols)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Fixed MatMul Benchmark for nanotorch")
    print("="*60)
    
    benchmark_matmul(
        size_a=tuple(args.size_a),
        size_b=tuple(args.size_b),
        repeats=args.repeats
    )


if __name__ == "__main__":
    main()
