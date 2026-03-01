#!/usr/bin/env python3
"""Profile small tensor operations to identify bottlenecks."""

import numpy as np
import time
import cProfile
import pstats
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanotorch.tensor import Tensor


def profile_small_matmul():
    """Profile small matrix multiplication with gradient computation."""
    print("Profiling small tensor matmul (10,5) @ (5,2)")
    
    np_a = np.random.randn(10, 5).astype(np.float32)
    np_b = np.random.randn(5, 2).astype(np.float32)
    
    a = Tensor(np_a, requires_grad=True)
    b = Tensor(np_b, requires_grad=True)
    
    def run_operation():
        result = a @ b
        result.backward()
        a.zero_grad()
        b.zero_grad()
    
    # Run once to ensure JIT compilation if any
    run_operation()
    
    # Profile
    pr = cProfile.Profile()
    pr.enable()
    
    for _ in range(1000):
        run_operation()
    
    pr.disable()
    
    # Print profiling results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())


def time_components():
    """Time individual components of small tensor operations."""
    print("\nTiming individual components:")
    
    np_a = np.random.randn(10, 5).astype(np.float32)
    np_b = np.random.randn(5, 2).astype(np.float32)
    
    # Component 1: Tensor creation
    start = time.perf_counter()
    for _ in range(1000):
        a = Tensor(np_a, requires_grad=True)
        b = Tensor(np_b, requires_grad=True)
    tensor_creation_time = time.perf_counter() - start
    print(f"Tensor creation: {tensor_creation_time/1000*1e6:.2f} µs per pair")
    
    # Component 2: Forward pass (matmul)
    a = Tensor(np_a, requires_grad=True)
    b = Tensor(np_b, requires_grad=True)
    start = time.perf_counter()
    for _ in range(1000):
        result = a @ b
    forward_time = time.perf_counter() - start
    print(f"Forward matmul: {forward_time/1000*1e6:.2f} µs per operation")
    
    # Component 3: Backward pass
    a = Tensor(np_a, requires_grad=True)
    b = Tensor(np_b, requires_grad=True)
    result = a @ b
    start = time.perf_counter()
    for _ in range(1000):
        result.backward()
        a.zero_grad()
        b.zero_grad()
    backward_time = time.perf_counter() - start
    print(f"Backward + zero_grad: {backward_time/1000*1e6:.2f} µs per operation")
    
    # Component 4: Gradient accumulation
    parent = Tensor(np_a, requires_grad=True)
    grad_data = np.random.randn(10, 5).astype(np.float32)
    start = time.perf_counter()
    for _ in range(1000):
        Tensor._accumulate_grad(parent, grad_data)
        parent.zero_grad()
    accumulate_time = time.perf_counter() - start
    print(f"Gradient accumulation: {accumulate_time/1000*1e6:.2f} µs per call")
    
    # Component 5: NumPy baseline
    start = time.perf_counter()
    for _ in range(1000):
        _ = np_a @ np_b
    numpy_time = time.perf_counter() - start
    print(f"NumPy matmul: {numpy_time/1000*1e6:.2f} µs per operation")
    
    print(f"\nForward overhead: {forward_time/numpy_time:.2f}x")
    print(f"Backward overhead: {backward_time/numpy_time:.2f}x")
    print(f"Total overhead: {(forward_time + backward_time)/numpy_time:.2f}x")


def analyze_gradient_accumulation():
    """Analyze gradient accumulation performance for small tensors."""
    print("\nAnalyzing gradient accumulation for small tensors:")
    
    shapes = [(10, 5), (5, 2), (10, 2), (1, 1), (10, 1), (1, 5)]
    
    for shape in shapes:
        np_data = np.random.randn(*shape).astype(np.float32)
        parent = Tensor(np_data, requires_grad=True)
        grad_data = np.random.randn(*shape).astype(np.float32)
        
        # Time accumulation
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            Tensor._accumulate_grad(parent, grad_data)
            times.append(time.perf_counter() - start)
            parent.zero_grad()
        
        median_time = np.median(times) * 1e6
        print(f"Shape {shape}: {median_time:.2f} µs per accumulation")


def profile_memory_allocation():
    """Profile memory allocation overhead."""
    print("\nProfiling memory allocation:")
    
    import gc
    
    # Disable GC
    gc.disable()
    
    sizes = [10, 50, 100, 500]
    
    for size in sizes:
        np_data = np.random.randn(size, size).astype(np.float32)
        
        # Time Tensor creation
        start = time.perf_counter()
        for _ in range(100):
            t = Tensor(np_data, requires_grad=True)
        tensor_time = time.perf_counter() - start
        
        # Time gradient allocation
        t = Tensor(np_data, requires_grad=True)
        start = time.perf_counter()
        for _ in range(100):
            t.zero_grad()
        grad_time = time.perf_counter() - start
        
        print(f"Size {size}x{size}: Tensor creation {tensor_time/100*1e6:.2f} µs, "
              f"gradient allocation {grad_time/100*1e6:.2f} µs")
    
    gc.enable()


def main():
    """Main profiling routine."""
    print("=" * 60)
    print("Small Tensor Performance Profiling")
    print("=" * 60)
    
    # Warm up
    print("Warming up...")
    warmup = np.random.randn(100, 100).astype(np.float32)
    for _ in range(5):
        _ = warmup @ warmup
    
    # Run profiling
    profile_small_matmul()
    time_components()
    analyze_gradient_accumulation()
    profile_memory_allocation()
    
    print("\n" + "=" * 60)
    print("Profiling complete")
    print("=" * 60)


if __name__ == "__main__":
    main()