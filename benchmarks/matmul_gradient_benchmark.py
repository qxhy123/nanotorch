#!/usr/bin/env python3
"""
Benchmark for matmul gradient performance in nanotorch.
Measures forward + backward time for matrix multiplication.
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanotorch.tensor import Tensor


def benchmark_matmul_gradient(size_a=(256, 512), size_b=(512, 256), repeats=10):
    """Benchmark matrix multiplication with gradient computation."""
    print(f"\n=== MatMul Gradient Benchmark (A={size_a}, B={size_b}) ===")

    np_a = np.random.randn(*size_a).astype(np.float32)
    np_b = np.random.randn(*size_b).astype(np.float32)

    times = []

    for _ in range(repeats):
        a = Tensor(np_a, requires_grad=True)
        b = Tensor(np_b, requires_grad=True)

        start = time.perf_counter()

        c = a @ b
        c.sum().backward()

        times.append(time.perf_counter() - start)

        a.zero_grad()
        b.zero_grad()

    print(f"Total time (forward + backward): {np.median(times)*1000:.3f} ms (median)")
    print(f"Throughput: {1/np.median(times):.2f} passes/sec")

    return {
        "size_a": size_a,
        "size_b": size_b,
        "median_time_ms": np.median(times) * 1000,
        "throughput_per_sec": 1 / np.median(times),
    }


def benchmark_matmul_gradient_various_sizes():
    """Benchmark matmul gradient for various matrix sizes."""
    sizes = [
        ((32, 64), (64, 32)),
        ((128, 256), (256, 128)),
        ((256, 512), (512, 256)),
        ((512, 1024), (1024, 512)),
    ]

    results = []

    for size_a, size_b in sizes:
        result = benchmark_matmul_gradient(size_a, size_b, repeats=5)
        results.append(result)

    print("\n" + "=" * 60)
    print("MATMUL GRADIENT PERFORMANCE SUMMARY")
    print("=" * 60)

    for result in results:
        print(
            f"Size {result['size_a']} x {result['size_b']}: "
            f"{result['median_time_ms']:.2f} ms, "
            f"{result['throughput_per_sec']:.2f} passes/sec"
        )

    return results


def benchmark_gradient_computation_overhead():
    """Compare gradient computation overhead vs forward pass."""
    print("\n=== Gradient Computation Overhead ===")

    size_a = (256, 512)
    size_b = (512, 256)

    np_a = np.random.randn(*size_a).astype(np.float32)
    np_b = np.random.randn(*size_b).astype(np.float32)

    forward_times = []
    for _ in range(10):
        a = Tensor(np_a, requires_grad=False)
        b = Tensor(np_b, requires_grad=False)
        start = time.perf_counter()
        c = a @ b
        forward_times.append(time.perf_counter() - start)

    grad_times = []
    for _ in range(10):
        a = Tensor(np_a, requires_grad=True)
        b = Tensor(np_b, requires_grad=True)
        start = time.perf_counter()
        c = a @ b
        c.sum().backward()
        grad_times.append(time.perf_counter() - start)

    forward_median = np.median(forward_times)
    grad_median = np.median(grad_times)

    print(f"Forward only: {forward_median*1000:.3f} ms")
    print(f"Forward + backward: {grad_median*1000:.3f} ms")
    print(f"Overhead factor: {grad_median/forward_median:.2f}x")
    print(f"Gradient computation: {(grad_median - forward_median)*1000:.3f} ms")

    return {
        "forward_median_ms": forward_median * 1000,
        "grad_median_ms": grad_median * 1000,
        "overhead_factor": grad_median / forward_median,
        "gradient_time_ms": (grad_median - forward_median) * 1000,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark nanotorch matmul gradient performance"
    )
    parser.add_argument(
        "--test",
        choices=["single", "sizes", "overhead", "all"],
        default="all",
        help="Test to run",
    )
    parser.add_argument("--repeats", type=int, default=10, help="Number of repetitions")

    args = parser.parse_args()

    print("=" * 60)
    print("nanotorch MatMul Gradient Benchmark")
    print("=" * 60)

    if args.test in ["single", "all"]:
        benchmark_matmul_gradient(repeats=args.repeats)

    if args.test in ["sizes", "all"]:
        benchmark_matmul_gradient_various_sizes()

    if args.test in ["overhead", "all"]:
        benchmark_gradient_computation_overhead()

    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
