#!/usr/bin/env python3
"""
Memory usage benchmark for nanotorch.

Measures memory consumption of tensors, computational graphs, and training loops.
Helps identify memory leaks and optimize memory usage.
"""

import numpy as np
import sys
import gc
import argparse
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple, Any  # noqa: F401

# Add parent directory to path to import nanotorch
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanotorch.tensor import Tensor  # noqa: E402
from nanotorch.nn import Linear, ReLU, MSE, Module  # noqa: E402
from nanotorch.optim import SGD  # noqa: E402


def estimate_tensor_memory(tensor: "Tensor") -> int:
    """Estimate total memory used by a Tensor and its components in bytes."""
    total = sys.getsizeof(tensor)

    total += sys.getsizeof(tensor.data)
    total += tensor.data.nbytes

    if tensor.grad is not None:
        total += estimate_tensor_memory(tensor.grad)

    if tensor._op is not None:
        total += sys.getsizeof(tensor._op)

    total += sys.getsizeof(tensor._parents)

    if tensor._ctx is not None:
        total += sys.getsizeof(tensor._ctx)
        if isinstance(tensor._ctx, dict):
            for k, v in tensor._ctx.items():
                total += sys.getsizeof(k) + sys.getsizeof(v)

    return total


def estimate_ndarray_memory(arr: np.ndarray) -> int:
    """Estimate total memory used by a numpy array in bytes."""
    total = sys.getsizeof(arr)
    total += arr.nbytes
    return total


def measure_memory_tracemalloc(create_fn, cleanup_fn=None) -> int:
    """Measure memory allocation using tracemalloc (bytes)."""
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    obj = create_fn()

    snapshot2 = tracemalloc.take_snapshot()
    stats = snapshot2.compare_to(snapshot1, "lineno")
    allocated = sum(stat.size for stat in stats)

    if cleanup_fn:
        cleanup_fn(obj)
    else:
        del obj

    tracemalloc.stop()
    return allocated


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        # Fallback to using garbage collector stats
        gc.collect()
        import resource

        return (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        )  # MB (Linux/Unix)


def benchmark_tensor_memory() -> List[Dict[str, Any]]:
    """Benchmark memory usage of tensor creation."""
    print("=== Tensor Memory Benchmark ===")

    sizes = [
        (100, 100),  # 10,000 elements
        (200, 200),  # 40,000 elements
        (500, 500),  # 250,000 elements
        (1000, 1000),  # 1,000,000 elements
    ]

    results = []

    for size in sizes:
        print(f"\n--- Tensor size {size} ---")
        num_elements = size[0] * size[1]
        data_bytes = num_elements * 4  # float32

        # Create tensor with gradient tracking (typical training scenario)
        def create_tensor():
            return Tensor.randn(size, requires_grad=True)

        def create_numpy():
            return np.random.randn(*size).astype(np.float32)

        # Measure with tracemalloc (accurate for Python heap allocations)
        tensor_tracemalloc = measure_memory_tracemalloc(create_tensor)
        numpy_tracemalloc = measure_memory_tracemalloc(create_numpy)

        # Measure with psutil RSS (system memory, coarse but includes all allocations)
        gc.collect()
        memory_before = get_memory_usage()
        tensor = create_tensor()
        gc.collect()
        memory_after = get_memory_usage()
        tensor_rss = (memory_after - memory_before) * 1024 * 1024  # bytes

        del tensor
        gc.collect()

        # Estimate theoretical memory usage
        tensor_obj = create_tensor()
        numpy_obj = create_numpy()

        tensor_estimated = estimate_tensor_memory(tensor_obj)
        numpy_estimated = estimate_ndarray_memory(numpy_obj)

        # Calculate overhead ratios
        tracemalloc_ratio = (
            tensor_tracemalloc / numpy_tracemalloc
            if numpy_tracemalloc > 0
            else float("inf")
        )
        estimated_ratio = tensor_estimated / numpy_estimated
        rss_ratio = (
            tensor_rss / numpy_tracemalloc if numpy_tracemalloc > 0 else float("inf")
        )

        # Print results
        print(f"Elements:           {num_elements:,}")
        print(
            f"Data size:          {data_bytes:,} bytes ({data_bytes/1024/1024:.2f} MB)"
        )
        print()
        print(
            f"NumPy (tracemalloc): {numpy_tracemalloc:,} bytes ({numpy_tracemalloc/1024:.1f} KB)"  # noqa: E501
        )
        print(
            f"Tensor (tracemalloc): {tensor_tracemalloc:,} bytes ({tensor_tracemalloc/1024:.1f} KB)"  # noqa: E501
        )
        print(f"  -> Overhead: {tracemalloc_ratio:.2f}x")
        print()
        print(
            f"NumPy (estimated):   {numpy_estimated:,} bytes ({numpy_estimated/1024:.1f} KB)"  # noqa: E501
        )
        print(
            f"Tensor (estimated):  {tensor_estimated:,} bytes ({tensor_estimated/1024:.1f} KB)"  # noqa: E501
        )
        print(f"  -> Overhead: {estimated_ratio:.2f}x")
        print()
        print(
            f"Tensor (RSS):        {tensor_rss:,.0f} bytes ({tensor_rss/1024/1024:.2f} MB)"  # noqa: E501
        )
        print(f"  -> RSS ratio: {rss_ratio:.2f}x")

        # Component breakdown for educational insight
        print("\nTensor memory breakdown:")
        print(f"  Tensor object: {sys.getsizeof(tensor_obj):,} bytes")
        print(f"  data object:   {sys.getsizeof(tensor_obj.data):,} bytes")
        print(f"  data.nbytes:   {tensor_obj.data.nbytes:,} bytes")
        if tensor_obj.grad is not None:
            grad_size = estimate_tensor_memory(tensor_obj.grad)
            print(f"  grad tensor:   {grad_size:,} bytes")

        results.append(
            {
                "size": size,
                "elements": num_elements,
                "tensor_tracemalloc_bytes": tensor_tracemalloc,
                "numpy_tracemalloc_bytes": numpy_tracemalloc,
                "tracemalloc_ratio": tracemalloc_ratio,
                "tensor_estimated_bytes": tensor_estimated,
                "numpy_estimated_bytes": numpy_estimated,
                "estimated_ratio": estimated_ratio,
                "tensor_rss_bytes": tensor_rss,
                "rss_ratio": rss_ratio,
            }
        )

        # Clean up
        del tensor_obj, numpy_obj
        gc.collect()

    return results


def benchmark_gradient_memory() -> List[Dict[str, Any]]:
    """Benchmark memory overhead of gradient tracking."""
    print("\n=== Gradient Memory Overhead ===")

    sizes = [
        (100, 100),  # 10,000 elements
        (500, 500),  # 250,000 elements
        (1000, 100),  # 100,000 elements (rectangular)
    ]

    results = []

    for size in sizes:
        print(f"\n--- Tensor size {size} ---")
        num_elements = size[0] * size[1]
        data_bytes = num_elements * 4

        # Create functions for measurement
        def create_no_grad():
            return Tensor.randn(size, requires_grad=False)

        def create_with_grad():
            return Tensor.randn(size, requires_grad=True)

        # Measure with tracemalloc (most accurate for Python allocations)
        no_grad_tracemalloc = measure_memory_tracemalloc(create_no_grad)
        with_grad_tracemalloc = measure_memory_tracemalloc(create_with_grad)

        # Calculate overhead
        overhead_bytes = with_grad_tracemalloc - no_grad_tracemalloc
        overhead_ratio = with_grad_tracemalloc / no_grad_tracemalloc

        # Estimate theoretical memory
        tensor_no_grad = create_no_grad()
        tensor_with_grad = create_with_grad()

        no_grad_estimated = estimate_tensor_memory(tensor_no_grad)
        with_grad_estimated = estimate_tensor_memory(tensor_with_grad)
        estimated_overhead = with_grad_estimated - no_grad_estimated
        estimated_ratio = with_grad_estimated / no_grad_estimated

        # Print results
        print(f"Elements: {num_elements:,}")
        print(f"Data size: {data_bytes:,} bytes ({data_bytes/1024/1024:.2f} MB)")
        print()
        print(
            f"No gradient (tracemalloc):  {no_grad_tracemalloc:,} bytes ({no_grad_tracemalloc/1024:.1f} KB)"  # noqa: E501
        )
        print(
            f"With gradient (tracemalloc): {with_grad_tracemalloc:,} bytes ({with_grad_tracemalloc/1024:.1f} KB)"  # noqa: E501
        )
        print(f"  -> Overhead: {overhead_bytes:,} bytes ({overhead_ratio:.2f}x)")
        print()
        print(
            f"No gradient (estimated):    {no_grad_estimated:,} bytes ({no_grad_estimated/1024:.1f} KB)"  # noqa: E501
        )
        print(
            f"With gradient (estimated):  {with_grad_estimated:,} bytes ({with_grad_estimated/1024:.1f} KB)"  # noqa: E501
        )
        print(f"  -> Overhead: {estimated_overhead:,} bytes ({estimated_ratio:.2f}x)")

        # Component breakdown for educational insight
        print("\nMemory breakdown (with gradient):")
        print(f"  Tensor object: {sys.getsizeof(tensor_with_grad):,} bytes")
        print(f"  data object:   {sys.getsizeof(tensor_with_grad.data):,} bytes")
        print(f"  data.nbytes:   {tensor_with_grad.data.nbytes:,} bytes")
        if tensor_with_grad.grad is not None:
            grad_size = estimate_tensor_memory(tensor_with_grad.grad)
            print(f"  grad tensor:   {grad_size:,} bytes")
            print(f"    -> grad data: {tensor_with_grad.grad.data.nbytes:,} bytes")

        print("\nMemory breakdown (no gradient):")
        print(f"  Tensor object: {sys.getsizeof(tensor_no_grad):,} bytes")
        print(f"  data object:   {sys.getsizeof(tensor_no_grad.data):,} bytes")
        print(f"  data.nbytes:   {tensor_no_grad.data.nbytes:,} bytes")

        results.append(
            {
                "size": size,
                "elements": num_elements,
                "no_grad_tracemalloc_bytes": no_grad_tracemalloc,
                "with_grad_tracemalloc_bytes": with_grad_tracemalloc,
                "overhead_bytes": overhead_bytes,
                "overhead_ratio": overhead_ratio,
                "no_grad_estimated_bytes": no_grad_estimated,
                "with_grad_estimated_bytes": with_grad_estimated,
                "estimated_overhead_bytes": estimated_overhead,
                "estimated_ratio": estimated_ratio,
            }
        )

        # Clean up
        del tensor_no_grad, tensor_with_grad
        gc.collect()

    return results


def benchmark_computational_graph() -> List[Dict[str, Any]]:
    """Benchmark memory usage of computational graphs."""
    print("\n=== Computational Graph Memory ===")

    # Test different graph depths
    depths = [1, 5, 10, 20, 50]

    results = []

    for depth in depths:
        print(f"\n--- Graph depth {depth} ---")

        # Create computational graph using tracemalloc for accurate measurement
        def create_graph():
            x = Tensor([1.0], requires_grad=True)
            current = x
            for i in range(depth):
                current = current * 2.0
            y = current
            y.backward()  # Build gradient graph
            return x, y, current

        # Measure memory with tracemalloc
        graph_memory = measure_memory_tracemalloc(create_graph)

        # Also measure with RSS for comparison
        gc.collect()
        memory_before = get_memory_usage()
        x, y, current = create_graph()
        gc.collect()
        memory_after = get_memory_usage()
        rss_memory = (memory_after - memory_before) * 1024 * 1024  # bytes

        # Estimate memory per tensor
        tensor_sample = Tensor([1.0], requires_grad=True)
        tensor_memory_est = estimate_tensor_memory(tensor_sample)
        del tensor_sample

        nodes = depth + 1  # x + depth operations
        memory_per_node_tracemalloc = graph_memory / nodes if nodes > 0 else 0
        memory_per_node_rss = rss_memory / nodes if nodes > 0 else 0

        print(f"Nodes in graph: {nodes}")
        print(
            f"Tracemalloc memory: {graph_memory:,} bytes ({graph_memory/1024:.1f} KB)"
        )
        print(
            f"  -> Per node: {memory_per_node_tracemalloc:,.0f} bytes ({memory_per_node_tracemalloc/1024:.2f} KB)"  # noqa: E501
        )
        print(f"RSS memory: {rss_memory:,.0f} bytes ({rss_memory/1024/1024:.2f} MB)")
        print(
            f"  -> Per node: {memory_per_node_rss:,.0f} bytes ({memory_per_node_rss/1024:.2f} KB)"  # noqa: E501
        )
        print(
            f"Estimated single tensor: {tensor_memory_est:,} bytes ({tensor_memory_est/1024:.1f} KB)"  # noqa: E501
        )

        # Show that computational graph overhead is primarily tensor objects
        print("\nNote: Computational graph memory is dominated by Tensor objects,")
        print(
            f"      each requiring ~{tensor_memory_est:,.0f} bytes for scalar tensors."
        )
        print("      Graph edges (_parents, _op) add minimal additional memory.")

        results.append(
            {
                "depth": depth,
                "nodes": nodes,
                "tracemalloc_bytes": graph_memory,
                "rss_bytes": rss_memory,
                "memory_per_node_tracemalloc_bytes": memory_per_node_tracemalloc,
                "memory_per_node_rss_bytes": memory_per_node_rss,
                "estimated_tensor_bytes": tensor_memory_est,
            }
        )

        # Clean up
        del x, y, current
        gc.collect()

    return results


def benchmark_training_loop() -> Dict[str, Any]:
    """Benchmark memory usage during a training loop."""
    print("\n=== Training Loop Memory ===")

    # Create a simple neural network
    batch_size = 32
    input_size = 100
    hidden_size = 50
    output_size = 10

    # Define a simple neural network
    class SimpleNet(Module):
        def __init__(self):
            super().__init__()
            self.fc1 = Linear(input_size, hidden_size)
            self.relu = ReLU()
            self.fc2 = Linear(hidden_size, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    model = SimpleNet()

    criterion = MSE()
    optimizer = SGD(model.parameters(), lr=0.01)

    # Generate random data
    inputs = Tensor.randn((batch_size, input_size))
    targets = Tensor.randn((batch_size, output_size))

    # Measure memory before training
    gc.collect()
    memory_before = get_memory_usage()

    # Run a few training steps
    num_steps = 10
    for step in range(num_steps):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if step % 2 == 0:
            gc.collect()
            memory_current = get_memory_usage()
            print(f"  Step {step}: Memory = {memory_current:.2f} MB")

    # Measure memory after training
    gc.collect()
    memory_after = get_memory_usage()

    memory_used = memory_after - memory_before

    print("\nTraining memory usage:")
    print(f"Before: {memory_before:.2f} MB")
    print(f"After:  {memory_after:.2f} MB")
    print(f"Used:   {memory_used:.2f} MB")
    print(f"Steps:  {num_steps}")

    # Check for memory leaks by running garbage collection
    del model, inputs, targets
    gc.collect()
    memory_final = get_memory_usage()

    leak_mb = memory_final - memory_before
    print(
        f"Leak:   {leak_mb:.2f} MB ({leak_mb/memory_used*100:.1f}% of training memory)"
    )

    return {
        "memory_before_mb": memory_before,
        "memory_after_mb": memory_after,
        "memory_used_mb": memory_used,
        "num_steps": num_steps,
        "memory_leak_mb": leak_mb,
        "memory_leak_percent": leak_mb / memory_used * 100 if memory_used > 0 else 0,
    }


def benchmark_memory_leak() -> Dict[str, Any]:
    """Test for memory leaks in repeated operations."""
    print("\n=== Memory Leak Test ===")

    # Test repeated tensor operations
    num_iterations = 100

    gc.collect()
    memory_before = get_memory_usage()

    for i in range(num_iterations):
        # Create and use tensors
        a = Tensor.randn((100, 100), requires_grad=True)
        b = Tensor.randn((100, 100), requires_grad=True)

        # Perform operations
        c = a * b
        d = c.sum()

        # Backward pass
        d.backward()

        # Delete references (should allow garbage collection)
        del a, b, c, d

        # Print progress
        if i % 20 == 0:
            gc.collect()
            memory_current = get_memory_usage()
            print(f"  Iteration {i}: Memory = {memory_current:.2f} MB")

    # Force garbage collection
    gc.collect()
    memory_after = get_memory_usage()

    memory_leak = memory_after - memory_before

    print("\nMemory leak test results:")
    print(f"Before: {memory_before:.2f} MB")
    print(f"After:  {memory_after:.2f} MB")
    print(f"Leak:   {memory_leak:.2f} MB")
    print(f"Iterations: {num_iterations}")

    if memory_leak > 1.0:  # More than 1 MB leak
        print("⚠️  WARNING: Possible memory leak detected!")
    elif memory_leak > 0.1:  # More than 0.1 MB leak
        print("⚠️  NOTE: Small memory increase detected (may be normal)")
    else:
        print("✓ No significant memory leak detected")

    return {
        "memory_before_mb": memory_before,
        "memory_after_mb": memory_after,
        "memory_leak_mb": memory_leak,
        "iterations": num_iterations,
        "has_leak": memory_leak > 1.0,
    }


def benchmark_operation_memory() -> List[Dict[str, Any]]:
    """Benchmark memory usage of tensor operations."""
    print("\n=== Operation Memory Benchmark ===")
    
    import gc
    import sys
    import numpy as np
    from nanotorch.tensor import Tensor
    
    results = []
    

    
    # ========== Matrix Multiplication ==========
    print("\n--- Matrix Multiplication ---")
    size_a = (256, 512)
    size_b = (512, 256)
    
    np_a = np.random.randn(*size_a).astype(np.float32)
    np_b = np.random.randn(*size_b).astype(np.float32)
    
    def numpy_matmul():
        return np_a @ np_b
    
    numpy_memory = measure_memory_tracemalloc(numpy_matmul)
    
    nt_a = Tensor(np_a, requires_grad=False)
    nt_b = Tensor(np_b, requires_grad=False)
    
    def nanotorch_matmul_no_grad():
        return nt_a @ nt_b
    
    nt_no_grad_memory = measure_memory_tracemalloc(nanotorch_matmul_no_grad)
    
    nt_a_grad = Tensor(np_a, requires_grad=True)
    nt_b_grad = Tensor(np_b, requires_grad=True)
    
    def nanotorch_matmul_grad():
        return nt_a_grad @ nt_b_grad
    
    nt_grad_memory = measure_memory_tracemalloc(nanotorch_matmul_grad)
    
    overhead_no_grad = nt_no_grad_memory / numpy_memory if numpy_memory > 0 else float('inf')
    overhead_grad = nt_grad_memory / numpy_memory if numpy_memory > 0 else float('inf')
    
    print(f"NumPy:           {numpy_memory:,} bytes")
    print(f"nanotorch (no grad): {nt_no_grad_memory:,} bytes")
    print(f"nanotorch (grad):    {nt_grad_memory:,} bytes")
    print(f"Overhead (no grad):  {overhead_no_grad:.2f}x")
    print(f"Overhead (grad):     {overhead_grad:.2f}x")
    
    results.append({
        "operation": "matmul",
        "size_a": size_a,
        "size_b": size_b,
        "numpy_memory_bytes": numpy_memory,
        "nanotorch_no_grad_memory_bytes": nt_no_grad_memory,
        "nanotorch_grad_memory_bytes": nt_grad_memory,
        "overhead_no_grad": overhead_no_grad,
        "overhead_grad": overhead_grad,
    })
    
    # ========== Gather Operation ==========
    print("\n--- Gather Operation ---")
    input_shape = (1024, 512)
    index_shape = (256, 512)
    dim = 0
    
    np_input = np.random.randn(*input_shape).astype(np.float32)
    np_indices = np.random.randint(0, input_shape[dim], size=index_shape).astype(np.int64)
    
    def numpy_gather():
        return np.take_along_axis(np_input, np_indices, axis=dim)
    
    numpy_memory = measure_memory_tracemalloc(numpy_gather)
    
    nt_input = Tensor(np_input, requires_grad=False)
    nt_indices = Tensor(np_indices.astype(np.float32), requires_grad=False)
    
    def nanotorch_gather_no_grad():
        return nt_input.gather(dim, nt_indices)
    
    nt_no_grad_memory = measure_memory_tracemalloc(nanotorch_gather_no_grad)
    
    nt_input_grad = Tensor(np_input, requires_grad=True)
    nt_indices_grad = Tensor(np_indices.astype(np.float32), requires_grad=False)
    
    def nanotorch_gather_grad():
        return nt_input_grad.gather(dim, nt_indices_grad)
    
    nt_grad_memory = measure_memory_tracemalloc(nanotorch_gather_grad)
    
    overhead_no_grad = nt_no_grad_memory / numpy_memory if numpy_memory > 0 else float('inf')
    overhead_grad = nt_grad_memory / numpy_memory if numpy_memory > 0 else float('inf')
    
    print(f"NumPy:           {numpy_memory:,} bytes")
    print(f"nanotorch (no grad): {nt_no_grad_memory:,} bytes")
    print(f"nanotorch (grad):    {nt_grad_memory:,} bytes")
    print(f"Overhead (no grad):  {overhead_no_grad:.2f}x")
    print(f"Overhead (grad):     {overhead_grad:.2f}x")
    
    results.append({
        "operation": "gather",
        "input_shape": input_shape,
        "index_shape": index_shape,
        "dim": dim,
        "numpy_memory_bytes": numpy_memory,
        "nanotorch_no_grad_memory_bytes": nt_no_grad_memory,
        "nanotorch_grad_memory_bytes": nt_grad_memory,
        "overhead_no_grad": overhead_no_grad,
        "overhead_grad": overhead_grad,
    })
    
    # ========== Scatter Operation ==========
    print("\n--- Scatter Operation ---")
    input_shape = (1024, 512)
    index_shape = (256, 512)
    src_shape = (256, 512)
    dim = 0
    
    np_input = np.random.randn(*input_shape).astype(np.float32)
    np_indices = np.random.randint(0, input_shape[dim], size=index_shape).astype(np.int64)
    np_src = np.random.randn(*src_shape).astype(np.float32)
    
    def numpy_scatter():
        arr = np_input.copy()
        np.put_along_axis(arr, np_indices, np_src, axis=dim)
        return arr
    
    numpy_memory = measure_memory_tracemalloc(numpy_scatter)
    
    nt_input = Tensor(np_input, requires_grad=False)
    nt_indices = Tensor(np_indices.astype(np.float32), requires_grad=False)
    nt_src = Tensor(np_src, requires_grad=False)
    
    def nanotorch_scatter_no_grad():
        return nt_input.scatter(dim, nt_indices, nt_src)
    
    nt_no_grad_memory = measure_memory_tracemalloc(nanotorch_scatter_no_grad)
    
    nt_input_grad = Tensor(np_input, requires_grad=True)
    nt_indices_grad = Tensor(np_indices.astype(np.float32), requires_grad=False)
    nt_src_grad = Tensor(np_src, requires_grad=True)
    
    def nanotorch_scatter_grad():
        return nt_input_grad.scatter(dim, nt_indices_grad, nt_src_grad)
    
    nt_grad_memory = measure_memory_tracemalloc(nanotorch_scatter_grad)
    
    overhead_no_grad = nt_no_grad_memory / numpy_memory if numpy_memory > 0 else float('inf')
    overhead_grad = nt_grad_memory / numpy_memory if numpy_memory > 0 else float('inf')
    
    print(f"NumPy:           {numpy_memory:,} bytes")
    print(f"nanotorch (no grad): {nt_no_grad_memory:,} bytes")
    print(f"nanotorch (grad):    {nt_grad_memory:,} bytes")
    print(f"Overhead (no grad):  {overhead_no_grad:.2f}x")
    print(f"Overhead (grad):     {overhead_grad:.2f}x")
    
    results.append({
        "operation": "scatter",
        "input_shape": input_shape,
        "index_shape": index_shape,
        "src_shape": src_shape,
        "dim": dim,
        "numpy_memory_bytes": numpy_memory,
        "nanotorch_no_grad_memory_bytes": nt_no_grad_memory,
        "nanotorch_grad_memory_bytes": nt_grad_memory,
        "overhead_no_grad": overhead_no_grad,
        "overhead_grad": overhead_grad,
    })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark nanotorch memory usage")
    parser.add_argument(
        "--test",
        choices=["all", "tensor", "gradient", "graph", "training", "leak", "operations"],
        default="all",
        help="Specific test to run",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("nanotorch Memory Usage Benchmark")
    print("=" * 60)

    all_results = {}

    # Check for psutil
    try:
        import psutil  # noqa: F401
    except ImportError:
        print("\n⚠️  psutil not installed. Memory measurements will be less accurate.")
        print("Install with: pip install psutil")
        print("Using fallback memory measurement...\n")

    # Run selected benchmarks
    if args.test in ["all", "tensor"]:
        all_results["tensor"] = benchmark_tensor_memory()

    if args.test in ["all", "gradient"]:
        all_results["gradient"] = benchmark_gradient_memory()

    if args.test in ["all", "graph"]:
        all_results["graph"] = benchmark_computational_graph()

    if args.test in ["all", "training"]:
        all_results["training"] = benchmark_training_loop()

    if args.test in ["all", "leak"]:
        all_results["leak"] = benchmark_memory_leak()
    
    if args.test in ["all", "operations"]:
        all_results["operations"] = benchmark_operation_memory()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if "tensor" in all_results:
        results = all_results["tensor"]
        print("\nTensor Memory (tracemalloc overhead vs NumPy):")
        for r in results:
            print(f"  Size {r['size']}: {r['tracemalloc_ratio']:.2f}x overhead")

    if "gradient" in all_results:
        results = all_results["gradient"]
        print("\nGradient Overhead (with grad vs no grad):")
        for r in results:
            print(f"  Size {r['size']}: {r['overhead_ratio']:.2f}x overhead")

    if "graph" in all_results:
        results = all_results["graph"]
        print("\nComputational Graph (tracemalloc per node):")
        for r in results:
            per_node_kb = r["memory_per_node_tracemalloc_bytes"] / 1024
            print(f"  Depth {r['depth']}: {per_node_kb:.1f} KB per node")

    if "training" in all_results:
        result = all_results["training"]
        print("\nTraining Loop:")
        print(f"  Memory used: {result['memory_used_mb']:.2f} MB")
        print(
            f"  Memory leak: {result['memory_leak_mb']:.2f} MB ({result['memory_leak_percent']:.1f}%)"  # noqa: E501
        )

    if "leak" in all_results:
        result = all_results["leak"]
        print("\nMemory Leak Test:")
        print(
            f"  Leak: {result['memory_leak_mb']:.2f} MB over {result['iterations']} iterations"  # noqa: E501
        )
        if result["has_leak"]:
            print("  ⚠️  WARNING: Possible memory leak!")
        else:
            print("  ✓ No significant leak detected")

    if "operations" in all_results:
        results = all_results["operations"]
        print("\nOperation Memory Overhead (nanotorch vs NumPy):")
        for r in results:
            print(f"  {r['operation']:10s}: {r['overhead_no_grad']:.2f}x (no grad), {r['overhead_grad']:.2f}x (grad)")
    
    print("\nMemory benchmark completed!")
    print("\nRecommendations:")
    print("1. Use 'with no_grad():' for inference to save memory")
    print("2. Manually delete tensors when no longer needed")
    print("3. Call gc.collect() periodically in long-running scripts")
    print("4. Use smaller batch sizes if memory is limited")


if __name__ == "__main__":
    main()
