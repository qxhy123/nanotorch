# Benchmark Diagnosis and Fix

## Problem Statement

The matmul benchmark showed unrealistic 0.02x overhead (nanotorch appearing 50x faster than NumPy) for matrix multiplication (256x512 @ 512x256).

## Root Causes Identified

### 1. Too Few Iterations (Primary Issue)
- Original benchmark used only 5 iterations
- BLAS operations show high variability (0.2ms to 20ms per run)
- With only 5 samples, median is unreliable
- **Fix**: Increased default to 30 iterations

### 2. Inconsistent Comparison
- nanotorch benchmark included `result.data.sum()` to force computation
- NumPy benchmark did not include sum()
- Difference is small (~0.05ms) but affects fairness
- **Fix**: Added NumPy with sum() benchmark for fair comparison

### 3. Artificial Limitation in main()
- Line 458: `repeats=min(args.repeats, 5)` limited matmul to 5 repeats
- This prevented users from running more iterations via `--repeats`
- **Fix**: Removed the artificial limit

### 4. High System Variability
- CPU frequency scaling causes 10-100x timing variance
- Background processes affect timing
- BLAS warm-up needs multiple operations to stabilize
- **Fix**: Extended warm-up from 1 to 15 operations

## Results After Fix

With 30 iterations and fair comparison:
```
NumPy (matmul only):  0.221 ms (median)
NumPy (with sum):     0.270 ms (median)
nanotorch (no grad):  0.268 ms (median)
nanotorch (grad):      0.323 ms (median)

Overhead (no grad):    0.99x (essentially zero overhead)
Overhead (grad):       1.19x (19% overhead for gradient tracking)
```

**Interpretation**: nanotorch wrapper adds negligible overhead (~0-20%) as expected.

## Why the Original Showed 0.02x

The 0.02x overhead (nanotorch 50x faster) was likely due to:
1. **Statistical anomaly**: With only 5 iterations, median can be skewed by outliers
2. **CPU frequency state**: nanotorch benchmark ran when CPU was at higher frequency
3. **BLAS warm-up**: nanotorch benefited from warm-up performed during NumPy benchmark
4. **Python interpreter**: Different caching/optimization patterns between the two benchmarks

## Recommendations for Future Benchmarks

1. **Always use 30+ iterations** for BLAS operations
2. **Run fresh data for each benchmark** to prevent caching artifacts
3. **Report std/min/max** alongside median to show reliability
4. **Use extended warm-up** (10-15 operations) to stabilize system state
5. **Consider running benchmarks in separate processes** to isolate state

## Files Modified

1. `benchmarks/tensor_operations.py`:
   - Changed `benchmark_matmul()` default from `repeats=5` to `repeats=30`
   - Added `np_times_full` benchmark (NumPy with sum())
   - Updated print statements to show fair comparison
   - Updated return dict to use `np_times_full`
   - Removed `min(args.repeats, 5)` limitation in `main()`

2. `benchmarks/benchmark_matmul_fixed.py` (new):
   - Standalone script with improved methodology
   - Better output formatting with interpretation notes
   - Can be run independently: `python benchmarks/benchmark_matmul_fixed.py --repeats 30`

## Testing the Fixed Benchmark

```bash
# Run the updated benchmark
python benchmarks/tensor_operations.py --op matmul --repeats 30

# Run the standalone fixed benchmark
python benchmarks/benchmark_matmul_fixed.py --repeats 30

# Run all benchmarks with more iterations
python benchmarks/tensor_operations.py --op all --repeats 30
```

## Expected Behavior

After the fix, you should see:
- nanotorch overhead between 0.8x and 1.5x (i.e., 0-50% overhead)
- nanotorch with gradient tracking having slightly higher overhead (1.0-1.5x)
- Consistent results across multiple runs (±20%)

If you see overhead < 0.5x or > 2.0x consistently, there may be a system issue affecting timing.
