#!/usr/bin/env python3
"""Robust benchmarking with CPU warm-up, outlier detection, and process control."""

import numpy as np
import time
import gc
import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanotorch.tensor import Tensor


class RobustBenchmark:
    """Robust benchmarking with statistical stability measures."""
    
    def __init__(self, name: str = "benchmark"):
        self.name = name
        self.results: Dict[str, Any] = {}
        
    def cpu_warmup(self, duration_seconds: float = 2.0) -> None:
        """Warm up CPU with progressively larger matrix multiplications.
        
        Uses a series of matrix multiplications of increasing size to
        ensure CPU reaches stable operating temperature and frequency.
        """
        print(f"Warming up CPU for {duration_seconds:.1f} seconds...")
        start_time = time.perf_counter()
        
        sizes = [10, 50, 100, 200, 400, 800]
        elapsed = 0.0
        
        while elapsed < duration_seconds:
            for size in sizes:
                if elapsed >= duration_seconds:
                    break
                
                a = np.random.randn(size, size).astype(np.float32)
                b = np.random.randn(size, size).astype(np.float32)
                
                iter_start = time.perf_counter()
                _ = a @ b
                iter_elapsed = time.perf_counter() - iter_start
                elapsed += iter_elapsed
                
                if iter_elapsed > 0.1:
                    sizes = [s // 2 for s in sizes if s > 10]
                elif iter_elapsed < 0.001:
                    sizes = [s * 2 for s in sizes]
        
        print(f"CPU warmup completed in {time.perf_counter() - start_time:.2f} seconds")
    
    def disable_gc(self) -> None:
        """Disable garbage collection during benchmark."""
        self.gc_was_enabled = gc.isenabled()
        gc.disable()
    
    def enable_gc(self) -> None:
        """Re-enable garbage collection if it was enabled."""
        if getattr(self, 'gc_was_enabled', True):
            gc.enable()
    
    def set_process_priority(self, niceness: int = 10) -> bool:
        """Set process priority (niceness) to reduce scheduling noise.
        
        Args:
            niceness: Niceness value (higher = lower priority, default 10).
                      Only works on Unix-like systems.
        
        Returns:
            True if priority was set successfully, False otherwise.
        """
        if platform.system() not in ['Linux', 'Darwin']:
            print(f"Warning: Cannot set process priority on {platform.system()}")
            return False
        
        try:
            current_nice = os.nice(0)
            os.nice(niceness - current_nice)
            print(f"Set process niceness from {current_nice} to {niceness}")
            return True
        except (AttributeError, OSError) as e:
            print(f"Warning: Could not set process priority: {e}")
            return False
    
    def measure_function(
        self,
        func: Callable[[], Any],
        min_iterations: int = 10,
        target_duration: float = 1.0,
        max_iterations: int = 1000,
        use_median: bool = True,
        trim_outliers: bool = True,
    ) -> Dict[str, float]:
        """Measure execution time of a function with statistical robustness.
        
        Args:
            func: Function to benchmark (no arguments).
            min_iterations: Minimum number of iterations to run.
            target_duration: Target total duration in seconds.
            max_iterations: Maximum number of iterations.
            use_median: Whether to use median (True) or mean (False).
            trim_outliers: Whether to trim outliers using MAD.
        
        Returns:
            Dictionary with timing statistics.
        """
        times = []
        
        for _ in range(min(5, min_iterations // 2)):
            func()
        
        start_time = time.perf_counter()
        for i in range(max_iterations):
            iteration_start = time.perf_counter()
            func()
            iteration_time = time.perf_counter() - iteration_start
            times.append(iteration_time)
            
            elapsed = time.perf_counter() - start_time
            if i + 1 >= min_iterations and elapsed >= target_duration:
                break
        
        times_array = np.array(times)
        
        if trim_outliers and len(times_array) >= 10:
            median = np.median(times_array)
            mad = np.median(np.abs(times_array - median))
            
            if mad > 0:
                modified_z_scores = 0.6745 * (times_array - median) / mad
                inliers = np.abs(modified_z_scores) <= 3.5
                
                if np.sum(inliers) >= min_iterations // 2:
                    times_array = times_array[inliers]
                    print(f"  Trimmed {len(times) - len(times_array)} outliers")
        
        if use_median:
            central_tendency = np.median(times_array)
            measure_name = "median"
        else:
            central_tendency = np.mean(times_array)
            measure_name = "mean"
        
        stats = {
            f"{measure_name}_time_ms": central_tendency * 1000,
            "min_time_ms": np.min(times_array) * 1000,
            "max_time_ms": np.max(times_array) * 1000,
            "std_time_ms": np.std(times_array) * 1000,
            "relative_std_percent": (np.std(times_array) / central_tendency * 100) 
                                    if central_tendency > 0 else 0.0,
            "iterations": len(times_array),
            "total_duration_ms": np.sum(times_array) * 1000,
        }
        
        print(f"  {measure_name.capitalize()}: {stats[f'{measure_name}_time_ms']:.3f} ms")
        print(f"  Range: {stats['min_time_ms']:.3f} - {stats['max_time_ms']:.3f} ms")
        print(f"  Relative std: {stats['relative_std_percent']:.1f}%")
        
        return stats
    
    def benchmark_matmul(
        self,
        size_a: Tuple[int, int] = (256, 512),
        size_b: Tuple[int, int] = (512, 256),
        requires_grad: bool = True,
    ) -> Dict[str, float]:
        """Benchmark matrix multiplication with gradient computation."""
        print(f"\n=== Benchmark: MatMul {size_a} @ {size_b} (grad={requires_grad}) ===")
        
        # Prepare data
        np_a = np.random.randn(*size_a).astype(np.float32)
        np_b = np.random.randn(*size_b).astype(np.float32)
        
        # Create tensors once (like in training loop)
        a = Tensor(np_a, requires_grad=requires_grad)
        b = Tensor(np_b, requires_grad=requires_grad)
        
        def forward_backward():
            """Single forward+backward pass."""
            result = a @ b
            if requires_grad:
                result.backward()
                a.zero_grad()
                b.zero_grad()
        
        # Benchmark
        stats = self.measure_function(
            forward_backward,
            min_iterations=20,
            target_duration=2.0,
            max_iterations=200,
        )
        
        # Compare with NumPy baseline
        def numpy_matmul():
            _ = np_a @ np_b
        
        np_stats = self.measure_function(
            numpy_matmul,
            min_iterations=50,
            target_duration=1.0,
            max_iterations=500,
        )
        
        # Compute overhead
        overhead = stats["median_time_ms"] / np_stats["median_time_ms"] if requires_grad else stats["median_time_ms"] / np_stats["median_time_ms"]
        
        print(f"\n--- Comparison ---")
        print(f"NumPy median: {np_stats['median_time_ms']:.3f} ms")
        print(f"nanotorch median: {stats['median_time_ms']:.3f} ms")
        print(f"Overhead: {overhead:.2f}x")
        
        # Store results
        self.results[f"matmul_{size_a}_{size_b}_grad_{requires_grad}"] = {
            "nanotorch": stats,
            "numpy": np_stats,
            "overhead": overhead,
        }
        
        return stats
    
    def benchmark_small_tensor(self) -> Dict[str, float]:
        """Benchmark small tensor operations."""
        print("\n=== Benchmark: Small Tensor Operations ===")
        
        sizes = [(10, 5), (5, 2)]
        np_a = np.random.randn(*sizes[0]).astype(np.float32)
        np_b = np.random.randn(*sizes[1]).astype(np.float32)
        
        # With gradient tracking
        a = Tensor(np_a, requires_grad=True)
        b = Tensor(np_b, requires_grad=True)
        
        def small_matmul_with_grad():
            result = a @ b
            result.backward()
            a.zero_grad()
            b.zero_grad()
        
        stats_with_grad = self.measure_function(
            small_matmul_with_grad,
            min_iterations=100,
            target_duration=1.0,
            max_iterations=1000,
        )
        
        # Without gradient tracking
        a_nograd = Tensor(np_a, requires_grad=False)
        b_nograd = Tensor(np_b, requires_grad=False)
        
        def small_matmul_no_grad():
            _ = a_nograd @ b_nograd
        
        stats_no_grad = self.measure_function(
            small_matmul_no_grad,
            min_iterations=200,
            target_duration=1.0,
            max_iterations=2000,
        )
        
        # NumPy baseline
        def numpy_small_matmul():
            _ = np_a @ np_b
        
        np_stats = self.measure_function(
            numpy_small_matmul,
            min_iterations=300,
            target_duration=1.0,
            max_iterations=3000,
        )
        
        print(f"\n--- Small Tensor ({sizes[0]} @ {sizes[1]}) ---")
        print(f"NumPy median: {np_stats['median_time_ms']:.3f} ms")
        print(f"nanotorch (no grad): {stats_no_grad['median_time_ms']:.3f} ms")
        print(f"nanotorch (with grad): {stats_with_grad['median_time_ms']:.3f} ms")
        print(f"Overhead no grad: {stats_no_grad['median_time_ms'] / np_stats['median_time_ms']:.2f}x")
        print(f"Overhead with grad: {stats_with_grad['median_time_ms'] / np_stats['median_time_ms']:.2f}x")
        print(f"Gradient tracking overhead: {stats_with_grad['median_time_ms'] / stats_no_grad['median_time_ms']:.2f}x")
        
        # Store results
        self.results["small_tensor"] = {
            "with_grad": stats_with_grad,
            "no_grad": stats_no_grad,
            "numpy": np_stats,
            "overhead_no_grad": stats_no_grad["median_time_ms"] / np_stats["median_time_ms"],
            "overhead_with_grad": stats_with_grad["median_time_ms"] / np_stats["median_time_ms"],
            "gradient_overhead": stats_with_grad["median_time_ms"] / stats_no_grad["median_time_ms"],
        }
        
        return stats_with_grad
    
    def benchmark_gradient_accumulation(self) -> Dict[str, float]:
        """Benchmark gradient accumulation performance."""
        print("\n=== Benchmark: Gradient Accumulation ===")
        
        # Create a medium-sized tensor
        size = (100, 100)
        np_data = np.random.randn(*size).astype(np.float32)
        
        # Create parent tensor
        parent = Tensor(np_data, requires_grad=True)
        
        # Create gradient contributions of various sizes/shapes
        grad_shapes = [
            size,  # Exact match
            (1, 100),  # Broadcasting
            (100, 1),  # Broadcasting
            (1, 1),  # Scalar-like
        ]
        
        results = {}
        
        for grad_shape in grad_shapes:
            print(f"\n  Gradient shape: {grad_shape}")
            grad_data = np.random.randn(*grad_shape).astype(np.float32)
            
            def accumulate_grad():
                Tensor._accumulate_grad(parent, grad_data)
                parent.zero_grad()
            
            stats = self.measure_function(
                accumulate_grad,
                min_iterations=50,
                target_duration=1.0,
                max_iterations=500,
            )
            
            results[str(grad_shape)] = stats
        
        self.results["gradient_accumulation"] = results
        return results
    
    def run_all(self) -> Dict[str, Any]:
        """Run all benchmarks with proper setup."""
        print("=" * 60)
        print("Robust nanotorch Benchmark Suite")
        print("=" * 60)
        
        try:
            self.cpu_warmup(duration_seconds=1.0)
            self.set_process_priority(niceness=10)
            
            self.disable_gc()
            try:
                self.benchmark_matmul(size_a=(256, 512), size_b=(512, 256))
                self.benchmark_small_tensor()
                self.benchmark_matmul(size_a=(100, 50), size_b=(50, 30))
            finally:
                self.enable_gc()
            
            # Print summary
            self.print_summary()
            
            return self.results
            
        except Exception as e:
            print(f"Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        if not self.results:
            print("No results to display")
            return
        
        # Extract key metrics
        summary_data = []
        
        for key, value in self.results.items():
            if key.startswith("matmul"):
                if "overhead" in value:
                    summary_data.append({
                        "Benchmark": key,
                        "NumPy (ms)": f"{value['numpy']['median_time_ms']:.3f}",
                        "nanotorch (ms)": f"{value['nanotorch']['median_time_ms']:.3f}",
                        "Overhead": f"{value['overhead']:.2f}x",
                        "Std %": f"{value['nanotorch']['relative_std_percent']:.1f}",
                    })
            elif key == "small_tensor":
                summary_data.append({
                    "Benchmark": "small_tensor",
                    "NumPy (ms)": f"{value['numpy']['median_time_ms']:.3f}",
                    "nanotorch (ms)": f"{value['with_grad']['median_time_ms']:.3f}",
                    "Overhead": f"{value['overhead_with_grad']:.2f}x",
                    "Std %": f"{value['with_grad']['relative_std_percent']:.1f}",
                })
        
        # Print table
        if summary_data:
            headers = ["Benchmark", "NumPy (ms)", "nanotorch (ms)", "Overhead", "Std %"]
            col_widths = [max(len(h), max(len(str(d[h])) for d in summary_data)) for h in headers]
            
            # Header
            header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
            print(header_row)
            print("-" * len(header_row))
            
            # Data rows
            for row in summary_data:
                print(" | ".join(str(row[h]).ljust(col_widths[i]) for i, h in enumerate(headers)))
        
        # Print key findings
        print("\n--- KEY FINDINGS ---")
        
        if "small_tensor" in self.results:
            small = self.results["small_tensor"]
            print(f"Small tensor overhead: {small['overhead_with_grad']:.2f}x")
            print(f"Gradient tracking overhead: {small['gradient_overhead']:.2f}x")
        
        # Check for high variability
        high_variability = []
        for key, value in self.results.items():
            if isinstance(value, dict) and "nanotorch" in value:
                if value["nanotorch"]["relative_std_percent"] > 20:
                    high_variability.append((key, value["nanotorch"]["relative_std_percent"]))
        
        if high_variability:
            print(f"\n⚠️  High variability detected:")
            for bench, std in high_variability:
                print(f"  {bench}: {std:.1f}% relative std")
            print("  Consider increasing benchmark duration or investigating system noise.")


def main():
    """Main entry point."""
    benchmark = RobustBenchmark("nanotorch_performance")
    results = benchmark.run_all()
    
    # Save results to file
    import json
    output_file = Path(__file__).parent / "robust_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())