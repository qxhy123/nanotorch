#!/usr/bin/env python3
"""
Tensor operations benchmark for nanotorch.

Compares performance of nanotorch operations with equivalent NumPy operations.
Helps identify performance bottlenecks and optimization opportunities.
"""

import numpy as np
import time
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import nanotorch
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanotorch.tensor import Tensor
from nanotorch import no_grad


def benchmark_add(size=(1024, 1024), repeats=10):
    """Benchmark element-wise addition."""
    print(f"\n=== Addition Benchmark (size={size}) ===")

    # Create random data
    np_a = np.random.randn(*size).astype(np.float32)
    np_b = np.random.randn(*size).astype(np.float32)

    # NumPy benchmark
    np_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = np_a + np_b
        np_times.append(time.perf_counter() - start)

    # nanotorch benchmark (without gradient tracking)
    nt_a = Tensor(np_a, requires_grad=False)
    nt_b = Tensor(np_b, requires_grad=False)

    nt_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = nt_a + nt_b
        nt_times.append(time.perf_counter() - start)

    # nanotorch benchmark (with gradient tracking)
    nt_a_grad = Tensor(np_a, requires_grad=True)
    nt_b_grad = Tensor(np_b, requires_grad=True)

    nt_grad_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = nt_a_grad + nt_b_grad
        nt_grad_times.append(time.perf_counter() - start)

    # Print results
    print(f"NumPy:          {np.median(np_times)*1000:.3f} ms (median)")
    print(f"nanotorch (no grad): {np.median(nt_times)*1000:.3f} ms (median)")
    print(f"nanotorch (grad):    {np.median(nt_grad_times)*1000:.3f} ms (median)")
    print(f"Overhead (no grad):  {np.median(nt_times)/np.median(np_times):.2f}x")
    print(f"Overhead (grad):     {np.median(nt_grad_times)/np.median(np_times):.2f}x")

    return {
        "operation": "add",
        "size": size,
        "numpy_median_ms": np.median(np_times) * 1000,
        "nanotorch_median_ms": np.median(nt_times) * 1000,
        "nanotorch_grad_median_ms": np.median(nt_grad_times) * 1000,
        "overhead_no_grad": np.median(nt_times) / np.median(np_times),
        "overhead_grad": np.median(nt_grad_times) / np.median(np_times),
    }


def benchmark_matmul(size_a=(256, 512), size_b=(512, 256), repeats=30):
    """Benchmark matrix multiplication."""
    print(f"\n=== Matrix Multiplication Benchmark (A={size_a}, B={size_b}) ===")

    # Create random data
    np_a = np.random.randn(*size_a).astype(np.float32)
    np_b = np.random.randn(*size_b).astype(np.float32)

    # Warm up BLAS (important for consistent measurements)
    _ = np_a @ np_b

    # NumPy benchmark

    np_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = np_a @ np_b
        np_times.append(time.perf_counter() - start)

    # NumPy benchmark (with sum() for fair comparison)
    np_times_full = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = np_a @ np_b
        _ = result.sum()
        np_times_full.append(time.perf_counter() - start)
    np_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = np_a @ np_b
        np_times.append(time.perf_counter() - start)

    # nanotorch benchmark (without gradient tracking)
    nt_a = Tensor(np_a, requires_grad=False)
    nt_b = Tensor(np_b, requires_grad=False)
    
    # Warm up nanotorch (ensures BLAS warm and any Python overhead)
    _ = nt_a @ nt_b

    nt_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = nt_a @ nt_b
        # Ensure computation by accessing all data
        _ = result.data.sum()
        nt_times.append(time.perf_counter() - start)

    # nanotorch benchmark (with gradient tracking)
    nt_a_grad = Tensor(np_a, requires_grad=True)
    nt_b_grad = Tensor(np_b, requires_grad=True)
    
    # Warm up gradient tracking
    _ = nt_a_grad @ nt_b_grad

    nt_grad_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = nt_a_grad @ nt_b_grad
        # Ensure computation by accessing all data
        _ = result.data.sum()
        nt_grad_times.append(time.perf_counter() - start)

    # Print results
    print(f"NumPy (matmul only): {np.median(np_times)*1000:.3f} ms (median)")
    print(f"NumPy (with sum):    {np.median(np_times_full)*1000:.3f} ms (median)")
    print(f"nanotorch (no grad): {np.median(nt_times)*1000:.3f} ms (median)")
    print(f"nanotorch (grad):    {np.median(nt_grad_times)*1000:.3f} ms (median)")
    print(f"Overhead (no grad):  {np.median(nt_times)/np.median(np_times_full):.2f}x")
    print(f"Overhead (grad):     {np.median(nt_grad_times)/np.median(np_times_full):.2f}x")

    return {
        "operation": "matmul",
        "size_a": size_a,
        "size_b": size_b,
        "numpy_median_ms": np.median(np_times_full) * 1000,
        "nanotorch_median_ms": np.median(nt_times) * 1000,
        "nanotorch_grad_median_ms": np.median(nt_grad_times) * 1000,
        "overhead_no_grad": np.median(nt_times) / np.median(np_times_full),
        "overhead_grad": np.median(nt_grad_times) / np.median(np_times_full),
    }


def benchmark_elementwise_ops(size=(1024, 1024), repeats=10):
    """Benchmark various element-wise operations."""
    print(f"\n=== Element-wise Operations Benchmark (size={size}) ===")

    # Create random data
    np_a = np.random.randn(*size).astype(np.float32)
    np_b = np.random.randn(*size).astype(np.float32)
    # Positive data for log operation (avoid log(0) and log(negative))
    np_a_pos = np.random.rand(*size).astype(np.float32) * 2 + 0.1

    operations = [
        ("multiply", lambda x, y: x * y),
        ("divide", lambda x, y: x / y),
        ("relu", lambda x, y: np.maximum(x, 0)),  # using x only
        ("sigmoid", lambda x, y: 1 / (1 + np.exp(-np.clip(x, -15, 15)))),
        ("exp", lambda x, y: np.exp(x)),
        ("log", lambda x, y: np.log(x)),
    ]

    results = []

    for op_name, np_op in operations:
        print(f"\n--- {op_name} ---")

        # Select appropriate data for operation
        current_np_a = np_a_pos if op_name == "log" else np_a
        
        # NumPy benchmark
        np_times = []
        for _ in range(repeats):
            start = time.perf_counter()
            if op_name in ["relu", "sigmoid", "exp", "log"]:
                result = np_op(current_np_a, None)
            else:
                result = np_op(current_np_a, np_b)
            np_times.append(time.perf_counter() - start)

        # nanotorch benchmark (without gradient tracking)
        # nanotorch benchmark (without gradient tracking)
        # Use appropriate data for current operation
        current_data = np_a_pos if op_name == "log" else np_a
        nt_a = Tensor(current_data, requires_grad=False)

        nt_times = []
        for _ in range(repeats):
            start = time.perf_counter()
            if op_name == "multiply":
                nt_b = Tensor(np_b, requires_grad=False)
                result = nt_a * nt_b
            elif op_name == "divide":
                nt_b = Tensor(np_b, requires_grad=False)
                result = nt_a / nt_b
            elif op_name == "relu":
                result = nt_a.relu()
            elif op_name == "sigmoid":
                result = nt_a.sigmoid()
            elif op_name == "exp":
                result = nt_a.exp()
            elif op_name == "log":
                result = nt_a.log()
            nt_times.append(time.perf_counter() - start)

        # Print results
        print(f"NumPy:          {np.median(np_times)*1000:.3f} ms (median)")
        print(f"nanotorch (no grad): {np.median(nt_times)*1000:.3f} ms (median)")
        print(f"Overhead:           {np.median(nt_times)/np.median(np_times):.2f}x")

        results.append(
            {
                "operation": op_name,
                "size": size,
                "numpy_median_ms": np.median(np_times) * 1000,
                "nanotorch_median_ms": np.median(nt_times) * 1000,
                "overhead": np.median(nt_times) / np.median(np_times),
            }
        )

    return results


def benchmark_reduction_ops(size=(1024, 1024), repeats=10):
    """Benchmark reduction operations (sum, mean)."""
    print(f"\n=== Reduction Operations Benchmark (size={size}) ===")

    # Create random data
    np_a = np.random.randn(*size).astype(np.float32)

    operations = [
        ("sum", lambda x: x.sum()),
        ("mean", lambda x: x.mean()),
        ("sum_axis0", lambda x: x.sum(axis=0)),
        ("mean_axis1", lambda x: x.mean(axis=1)),
    ]

    results = []

    for op_name, np_op in operations:
        print(f"\n--- {op_name} ---")

        # NumPy benchmark
        np_times = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = np_op(np_a)
            np_times.append(time.perf_counter() - start)

        # nanotorch benchmark (without gradient tracking)
        nt_a = Tensor(np_a, requires_grad=False)

        nt_times = []
        for _ in range(repeats):
            start = time.perf_counter()
            if op_name == "sum":
                result = nt_a.sum()
            elif op_name == "mean":
                result = nt_a.mean()
            elif op_name == "sum_axis0":
                result = nt_a.sum(axis=0)
            elif op_name == "mean_axis1":
                result = nt_a.mean(axis=1)
            nt_times.append(time.perf_counter() - start)

        # Print results
        print(f"NumPy:          {np.median(np_times)*1000:.3f} ms (median)")
        print(f"nanotorch (no grad): {np.median(nt_times)*1000:.3f} ms (median)")
        print(f"Overhead:           {np.median(nt_times)/np.median(np_times):.2f}x")

        results.append(
            {
                "operation": op_name,
                "size": size,
                "numpy_median_ms": np.median(np_times) * 1000,
                "nanotorch_median_ms": np.median(nt_times) * 1000,
                "overhead": np.median(nt_times) / np.median(np_times),
            }
        )

    return results


def benchmark_backward_pass(size=(100, 50), repeats=5):
    """Benchmark backward pass computation."""
    print(f"\n=== Backward Pass Benchmark (size={size}) ===")

    # Create a simple computational graph: y = sum(W * x + b)
    np_W = np.random.randn(*size).astype(np.float32)
    np_x = np.random.randn(size[1], 1).astype(np.float32)
    np_b = np.random.randn(size[0], 1).astype(np.float32)

    # nanotorch benchmark
    times = []

    for _ in range(repeats):
        # Create tensors with gradient tracking
        W = Tensor(np_W, requires_grad=True)
        x = Tensor(np_x, requires_grad=True)
        b = Tensor(np_b, requires_grad=True)

        start = time.perf_counter()

        # Forward pass
        y = (W @ x + b).relu().sum()

        # Backward pass
        y.backward()

        times.append(time.perf_counter() - start)

    # Print results
    print(f"Total time (forward + backward): {np.median(times)*1000:.3f} ms (median)")
    print(f"Throughput: {1/np.median(times):.2f} passes/sec")

    return {
        "operation": "backward_pass",
        "size": size,
        "median_time_ms": np.median(times) * 1000,
        "throughput_per_sec": 1 / np.median(times),
    }


def benchmark_gather(input_shape=(1024, 512), index_shape=(256, 512), dim=0, repeats=10):
    """Benchmark gather operation along specified dimension."""
    print(f"\n=== Gather Benchmark (input_shape={input_shape}, index_shape={index_shape}, dim={dim}) ===")

    # Create random input data
    np_input = np.random.randn(*input_shape).astype(np.float32)
    # Create random indices within valid range for the gather dimension
    np_indices = np.random.randint(0, input_shape[dim], size=index_shape).astype(np.int64)
    
    # NumPy benchmark using take_along_axis
    np_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = np.take_along_axis(np_input, np_indices, axis=dim)
        np_times.append(time.perf_counter() - start)
    
    # nanotorch benchmark (without gradient tracking)
    nt_input = Tensor(np_input, requires_grad=False)
    nt_indices = Tensor(np_indices.astype(np.float32), requires_grad=False)  # Tensor expects float32
    
    nt_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = nt_input.gather(dim, nt_indices)
        nt_times.append(time.perf_counter() - start)
    
    # nanotorch benchmark (with gradient tracking)
    nt_input_grad = Tensor(np_input, requires_grad=True)
    nt_indices_grad = Tensor(np_indices.astype(np.float32), requires_grad=False)  # indices don't need grad
    
    nt_grad_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = nt_input_grad.gather(dim, nt_indices_grad)
        nt_grad_times.append(time.perf_counter() - start)
    
    # Print results
    print(f"NumPy:          {np.median(np_times)*1000:.3f} ms (median)")
    print(f"nanotorch (no grad): {np.median(nt_times)*1000:.3f} ms (median)")
    print(f"nanotorch (grad):    {np.median(nt_grad_times)*1000:.3f} ms (median)")
    print(f"Overhead (no grad):  {np.median(nt_times)/np.median(np_times):.2f}x")
    print(f"Overhead (grad):     {np.median(nt_grad_times)/np.median(np_times):.2f}x")
    
    return {
        "operation": "gather",
        "input_shape": input_shape,
        "index_shape": index_shape,
        "dim": dim,
        "numpy_median_ms": np.median(np_times) * 1000,
        "nanotorch_median_ms": np.median(nt_times) * 1000,
        "nanotorch_grad_median_ms": np.median(nt_grad_times) * 1000,
        "overhead_no_grad": np.median(nt_times) / np.median(np_times),
        "overhead_grad": np.median(nt_grad_times) / np.median(np_times),
    }


def benchmark_scatter(input_shape=(1024, 512), index_shape=(256, 512), src_shape=(256, 512), dim=0, repeats=10):
    """Benchmark scatter operation along specified dimension."""
    print(f"\n=== Scatter Benchmark (input_shape={input_shape}, index_shape={index_shape}, src_shape={src_shape}, dim={dim}) ===")

    # Create random input data
    np_input = np.random.randn(*input_shape).astype(np.float32)
    # Create random indices within valid range for the scatter dimension
    np_indices = np.random.randint(0, input_shape[dim], size=index_shape).astype(np.int64)
    # Create random source data
    np_src = np.random.randn(*src_shape).astype(np.float32)
    
    # NumPy benchmark using put_along_axis (modifies array in-place, so we copy each iteration)
    np_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        arr = np_input.copy()
        np.put_along_axis(arr, np_indices, np_src, axis=dim)
        np_times.append(time.perf_counter() - start)
    
    # nanotorch benchmark (without gradient tracking)
    nt_input = Tensor(np_input, requires_grad=False)
    nt_indices = Tensor(np_indices.astype(np.float32), requires_grad=False)
    nt_src = Tensor(np_src, requires_grad=False)
    
    nt_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = nt_input.scatter(dim, nt_indices, nt_src)
        nt_times.append(time.perf_counter() - start)
    
    # nanotorch benchmark (with gradient tracking)
    nt_input_grad = Tensor(np_input, requires_grad=True)
    nt_indices_grad = Tensor(np_indices.astype(np.float32), requires_grad=False)
    nt_src_grad = Tensor(np_src, requires_grad=True)
    
    nt_grad_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = nt_input_grad.scatter(dim, nt_indices_grad, nt_src_grad)
        nt_grad_times.append(time.perf_counter() - start)
    
    # Print results
    print(f"NumPy:          {np.median(np_times)*1000:.3f} ms (median)")
    print(f"nanotorch (no grad): {np.median(nt_times)*1000:.3f} ms (median)")
    print(f"nanotorch (grad):    {np.median(nt_grad_times)*1000:.3f} ms (median)")
    print(f"Overhead (no grad):  {np.median(nt_times)/np.median(np_times):.2f}x")
    print(f"Overhead (grad):     {np.median(nt_grad_times)/np.median(np_times):.2f}x")
    
    return {
        "operation": "scatter",
        "input_shape": input_shape,
        "index_shape": index_shape,
        "src_shape": src_shape,
        "dim": dim,
        "numpy_median_ms": np.median(np_times) * 1000,
        "nanotorch_median_ms": np.median(nt_times) * 1000,
        "nanotorch_grad_median_ms": np.median(nt_grad_times) * 1000,
        "overhead_no_grad": np.median(nt_times) / np.median(np_times),
        "overhead_grad": np.median(nt_grad_times) / np.median(np_times),
    }


def benchmark_memory_usage():
    """Simple memory usage benchmark."""
    print("\n=== Memory Usage Benchmark ===")

    import psutil
    import os

    process = psutil.Process(os.getpid())

    sizes = [(100, 100), (500, 500), (1000, 1000)]

    results = []

    for size in sizes:
        print(f"\n--- Tensor size {size} ---")

        # Measure memory before
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Create large tensor
        tensor = Tensor.randn(size)

        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB

        memory_used = memory_after - memory_before
        expected_memory = size[0] * size[1] * 4 / 1024 / 1024  # float32 = 4 bytes

        print(f"Memory used: {memory_used:.2f} MB")
        print(f"Expected:    {expected_memory:.2f} MB")
        print(f"Ratio:       {memory_used/expected_memory:.2f}x")

        results.append(
            {
                "size": size,
                "memory_used_mb": memory_used,
                "expected_memory_mb": expected_memory,
                "ratio": memory_used / expected_memory,
            }
        )

        # Clean up
        del tensor

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark nanotorch tensor operations"
    )
    parser.add_argument(
        "--op",
        choices=[
            "all",
            "add",
            "matmul",
            "elementwise",
            "reduction",
            "backward",
            "memory",
            "gather",
            "scatter",
        ],
        default="all",
        help="Operation to benchmark",
    )
    parser.add_argument("--repeats", type=int, default=10, help="Number of repetitions")
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[1024, 1024],
        help="Tensor size for elementwise operations (rows cols)",
    )
    parser.add_argument(
        "--matmul-size",
        type=int,
        nargs=4,
        default=[256, 512, 512, 256],
        help="Matrix multiplication sizes (a_rows a_cols b_rows b_cols)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("nanotorch Tensor Operations Benchmark")
    print("=" * 60)

    all_results = {}

    # Run selected benchmarks
    if args.op in ["all", "add"]:
        all_results["add"] = benchmark_add(size=tuple(args.size), repeats=args.repeats)

    if args.op in ["all", "matmul"]:
        all_results["matmul"] = benchmark_matmul(
            size_a=(args.matmul_size[0], args.matmul_size[1]),
            size_b=(args.matmul_size[2], args.matmul_size[3]),
            repeats=args.repeats,
        )

    if args.op in ["all", "elementwise"]:
        all_results["elementwise"] = benchmark_elementwise_ops(
            size=tuple(args.size), repeats=args.repeats
        )

    if args.op in ["all", "reduction"]:
        all_results["reduction"] = benchmark_reduction_ops(
            size=tuple(args.size), repeats=args.repeats
        )

    if args.op in ["all", "backward"]:
        all_results["backward"] = benchmark_backward_pass(
            size=(100, 50), repeats=min(args.repeats, 5)
        )

    if args.op in ["all", "memory"]:
        try:
            import psutil

            all_results["memory"] = benchmark_memory_usage()
        except ImportError:
            print("\npsutil not installed. Skipping memory benchmark.")
            print("Install with: pip install psutil")

    if args.op in ["all", "gather"]:
        all_results["gather"] = benchmark_gather(repeats=args.repeats)

    if args.op in ["all", "scatter"]:
        all_results["scatter"] = benchmark_scatter(repeats=args.repeats)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if "add" in all_results:
        result = all_results["add"]
        print(
            f"Addition: {result['overhead_no_grad']:.2f}x overhead (no grad), {result['overhead_grad']:.2f}x (grad)"
        )

    if "matmul" in all_results:
        result = all_results["matmul"]
        print(
            f"MatMul:   {result['overhead_no_grad']:.2f}x overhead (no grad), {result['overhead_grad']:.2f}x (grad)"
        )

    if "elementwise" in all_results:
        results = all_results["elementwise"]
        for r in results:
            print(f"{r['operation']:10s}: {r['overhead']:.2f}x overhead")

    if "reduction" in all_results:
        results = all_results["reduction"]
        for r in results:
            print(f"{r['operation']:10s}: {r['overhead']:.2f}x overhead")

    if "gather" in all_results:
        result = all_results["gather"]
        print(
            f"Gather:   {result['overhead_no_grad']:.2f}x overhead (no grad), {result['overhead_grad']:.2f}x (grad)"
        )

    if "scatter" in all_results:
        result = all_results["scatter"]
        print(
            f"Scatter:  {result['overhead_no_grad']:.2f}x overhead (no grad), {result['overhead_grad']:.2f}x (grad)"
        )

    if "backward" in all_results:
        result = all_results["backward"]
        print(f"Backward: {result['throughput_per_sec']:.2f} passes/sec")

    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
