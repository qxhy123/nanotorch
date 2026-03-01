#!/usr/bin/env python3
"""Profile backward pass performance for nanotorch."""

import numpy as np
import time
import cProfile
import pstats
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanotorch.tensor import Tensor

def profile_matmul_backward(size_a=(256, 512), size_b=(512, 256)):
    """Profile matrix multiplication backward pass."""
    print(f"Profiling matmul backward: A={size_a}, B={size_b}")
    
    np_a = np.random.randn(*size_a).astype(np.float32)
    np_b = np.random.randn(*size_b).astype(np.float32)
    
    a = Tensor(np_a, requires_grad=True)
    b = Tensor(np_b, requires_grad=True)
    
    pr = cProfile.Profile()
    pr.enable()
    
    for _ in range(10):
        result = a @ b
        result.backward()
        a.zero_grad()
        b.zero_grad()
    
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    print("=== Top 20 functions by cumulative time ===")
    print(s.getvalue())
    
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats('time')
    ps2.print_stats(20)
    
    print("\n=== Top 20 functions by internal time ===")
    print(s2.getvalue())
    
    pr2 = cProfile.Profile()
    pr2.enable()
    
    result = a @ b
    pr2.disable()
    
    s3 = io.StringIO()
    ps3 = pstats.Stats(pr2, stream=s3).sort_stats('time')
    ps3.print_stats(10)
    
    print("\n=== Forward pass only (top 10) ===")
    print(s3.getvalue())
    
    return pr

def profile_small_tensor_backward():
    """Profile backward pass for small tensors."""
    print("\n=== Profiling small tensor backward pass ===")
    
    np_a = np.random.randn(10, 5).astype(np.float32)
    np_b = np.random.randn(5, 2).astype(np.float32)
    
    a = Tensor(np_a, requires_grad=True)
    b = Tensor(np_b, requires_grad=True)
    
    pr = cProfile.Profile()
    pr.enable()
    
    for _ in range(100):
        result = a @ b
        result.backward()
        a.zero_grad()
        b.zero_grad()
    
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('time')
    ps.print_stats(15)
    
    print("Top 15 functions for small tensor backward:")
    print(s.getvalue())

if __name__ == "__main__":
    print("Profiling nanotorch backward pass performance")
    print("=" * 60)
    
    profile_matmul_backward()
    profile_small_tensor_backward()