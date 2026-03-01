import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanotorch.tensor import Tensor

def benchmark_small_matmul_gradient(size_a=(100, 50), size_b=(50, 30), repeats=100):
    print(f"\n=== Small MatMul Gradient Benchmark (A={size_a}, B={size_b}) ===")
    
    np_a = np.random.randn(*size_a).astype(np.float32)
    np_b = np.random.randn(*size_b).astype(np.float32)
    
    np_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = np_a @ np_b
        np_times.append(time.perf_counter() - start)
    
    nt_forward_times = []
    nt_backward_times = []
    nt_total_times = []
    
    for _ in range(repeats):
        a = Tensor(np_a, requires_grad=True)
        b = Tensor(np_b, requires_grad=True)
        
        start = time.perf_counter()
        result = a @ b
        forward_end = time.perf_counter()
        
        result.backward()
        backward_end = time.perf_counter()
        
        nt_forward_times.append(forward_end - start)
        nt_backward_times.append(backward_end - forward_end)
        nt_total_times.append(backward_end - start)
    
    print(f"NumPy forward: {np.median(np_times)*1000:.3f} ms")
    print(f"nanotorch forward: {np.median(nt_forward_times)*1000:.3f} ms")
    print(f"nanotorch backward: {np.median(nt_backward_times)*1000:.3f} ms")
    print(f"nanotorch total: {np.median(nt_total_times)*1000:.3f} ms")
    print(f"Overhead forward: {np.median(nt_forward_times)/np.median(np_times):.2f}x")
    print(f"Overhead total (forward+backward): {np.median(nt_total_times)/np.median(np_times):.2f}x")
    print(f"Gradient accumulation overhead: {np.median(nt_backward_times)/np.median(np_times):.2f}x")
    
    return {
        "size_a": size_a,
        "size_b": size_b,
        "numpy_forward_ms": np.median(np_times) * 1000,
        "nt_forward_ms": np.median(nt_forward_times) * 1000,
        "nt_backward_ms": np.median(nt_backward_times) * 1000,
        "nt_total_ms": np.median(nt_total_times) * 1000,
        "overhead_forward": np.median(nt_forward_times) / np.median(np_times),
        "overhead_total": np.median(nt_total_times) / np.median(np_times),
    }

def benchmark_gradient_accumulation_loop(size=(10, 10), repeats=1000):
    print(f"\n=== Gradient Accumulation Loop (size={size}) ===")
    
    np_data = np.random.randn(*size).astype(np.float32)
    
    tensor = Tensor(np_data, requires_grad=True)
    
    start = time.perf_counter()
    for _ in range(repeats):
        result = tensor.sum()
        result.backward()
        tensor.zero_grad()
    end = time.perf_counter()
    
    total_time = end - start
    print(f"Total time for {repeats} accumulations: {total_time*1000:.3f} ms")
    print(f"Time per accumulation: {total_time/repeats*1e6:.3f} µs")
    
    return {
        "size": size,
        "repeats": repeats,
        "total_time_ms": total_time * 1000,
        "time_per_accumulation_us": total_time / repeats * 1e6,
    }

def benchmark_matmul_gradient_overhead(size_a=(256, 512), size_b=(512, 256), repeats=30):
    print(f"\n=== MatMul Gradient Overhead Benchmark (A={size_a}, B={size_b}) ===")
    
    np_a = np.random.randn(*size_a).astype(np.float32)
    np_b = np.random.randn(*size_b).astype(np.float32)
    

    np_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = np_a @ np_b
        np_times.append(time.perf_counter() - start)
    

    nt_no_grad_times = []
    for _ in range(repeats):
        a = Tensor(np_a, requires_grad=False)
        b = Tensor(np_b, requires_grad=False)
        start = time.perf_counter()
        result = a @ b
        nt_no_grad_times.append(time.perf_counter() - start)
    

    nt_grad_times = []
    for _ in range(repeats):
        a = Tensor(np_a, requires_grad=True)
        b = Tensor(np_b, requires_grad=True)
        start = time.perf_counter()
        result = a @ b
        result.backward()
        nt_grad_times.append(time.perf_counter() - start)
    
    np_median = np.median(np_times)
    nt_no_grad_median = np.median(nt_no_grad_times)
    nt_grad_median = np.median(nt_grad_times)
    
    print(f"NumPy forward: {np_median*1000:.3f} ms")
    print(f"nanotorch forward (no grad): {nt_no_grad_median*1000:.3f} ms")
    print(f"nanotorch forward+backward (grad): {nt_grad_median*1000:.3f} ms")
    print(f"Overhead (no grad): {nt_no_grad_median/np_median:.2f}x")
    print(f"Overhead (grad): {nt_grad_median/np_median:.2f}x")
    print(f"Gradient computation overhead: {nt_grad_median/nt_no_grad_median:.2f}x")
    
    return {
        "size_a": size_a,
        "size_b": size_b,
        "numpy_forward_ms": np_median * 1000,
        "nt_no_grad_ms": nt_no_grad_median * 1000,
        "nt_grad_ms": nt_grad_median * 1000,
        "overhead_no_grad": nt_no_grad_median / np_median,
        "overhead_grad": nt_grad_median / np_median,
        "gradient_overhead_factor": nt_grad_median / nt_no_grad_median,
    }

if __name__ == "__main__":
    print("Gradient Accumulation Benchmarks")
    print("=" * 50)
    
    results = []
    
    results.append(benchmark_small_matmul_gradient(size_a=(100, 50), size_b=(50, 30)))
    results.append(benchmark_small_matmul_gradient(size_a=(10, 5), size_b=(5, 2)))
    results.append(benchmark_gradient_accumulation_loop(size=(10, 10), repeats=500))
    results.append(benchmark_matmul_gradient_overhead())
    
    print("\n=== Summary ===")
    for res in results:
        if "size_a" in res:
            if "gradient_overhead_factor" in res:
                print(f"MatMul Gradient Overhead {res['size_a']} @ {res['size_b']}: "
                      f"no grad overhead {res['overhead_no_grad']:.2f}x, "
                      f"grad overhead {res['overhead_grad']:.2f}x, "
                      f"gradient factor {res['gradient_overhead_factor']:.2f}x")
            else:
                print(f"MatMul {res['size_a']} @ {res['size_b']}: "
                      f"forward overhead {res['overhead_forward']:.2f}x, "
                      f"total overhead {res['overhead_total']:.2f}x")
        else:
            print(f"Accumulation loop {res['size']}: "
                  f"{res['time_per_accumulation_us']:.3f} µs per accumulation")