#!/usr/bin/env python3
"""
Gather and scatter operations example using nanotorch.

This example demonstrates:
1. Gather operation: selecting elements from a tensor using indices
2. Scatter operation: scattering values into a tensor at specified indices
3. Gradient computation for both operations
4. Comparison with NumPy equivalents
5. Practical use case: top-k selection and embedding updates
"""

import numpy as np
from nanotorch import Tensor


def demonstrate_gather():
    """Demonstrate gather operation with gradient tracking."""
    print("=" * 60)
    print("Gather Operation Demonstration")
    print("=" * 60)
    
    # Create input tensor: 3x4 matrix
    # [[ 0,  1,  2,  3],
    #  [ 4,  5,  6,  7],
    #  [ 8,  9, 10, 11]]
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    x = Tensor(data, requires_grad=True)
    print(f"Input tensor (shape {x.shape}):")
    print(x.data)
    
    # Create indices for gathering along dimension 0 (rows)
    # indices shape must match input shape except at gather dimension
    # Here we gather from rows: indices[i,j] specifies which row's element to take
    indices = np.array([[0, 1, 0, 1],  # take row 0,1,0,1 for columns 0-3
                        [2, 1, 2, 1],  # take row 2,1,2,1
                        [0, 2, 0, 2]], dtype=np.int64)
    idx = Tensor(indices, requires_grad=False)  # indices don't require gradients
    
    # Gather along dimension 0
    result = x.gather(dim=0, index=idx)
    print(f"\nIndices for gather along dim=0 (shape {idx.shape}):")
    print(indices)
    print(f"\nGathered result (shape {result.shape}):")
    print(result.data)
    
    # Verify with NumPy equivalent
    np_result = np.take_along_axis(data, indices, axis=0)
    print(f"\nNumPy equivalent (np.take_along_axis):")
    print(np_result)
    print(f"Match with nanotorch: {np.allclose(result.data, np_result)}")
    
    # Compute gradient
    loss = result.sum()  # Simple loss: sum of gathered elements
    loss.backward()
    print(f"\nGradient of input tensor w.r.t. loss (sum of gathered elements):")
    print(x.grad)
    print("\nInterpretation: Each input element's gradient equals how many times")
    print("it was gathered. For example, element (0,0) appears once in indices[0,0]")
    print(f"so gradient is 1. Element (1,1) appears twice (indices[0,1] and indices[1,3])")
    print(f"so gradient is 2.")
    
    # Another example: gather along dimension 1 (columns)
    print("\n" + "-" * 40)
    print("Gather along dimension 1 (columns)")
    x2 = Tensor(np.arange(12, dtype=np.float32).reshape(3, 4), requires_grad=True)
    indices_col = np.array([[0, 2],  # for row 0, take columns 0 and 2
                           [1, 3],   # for row 1, take columns 1 and 3
                           [2, 0]], dtype=np.int64)  # shape (3, 2)
    idx_col = Tensor(indices_col, requires_grad=False)
    result_col = x2.gather(dim=1, index=idx_col)
    print(f"Input shape: {x2.shape}")
    print(f"Indices shape: {idx_col.shape}")
    print(f"Result shape: {result_col.shape}")
    print("Result:")
    print(result_col.data)
    
    # Gradient check
    loss_col = result_col.sum()
    loss_col.backward()
    print(f"\nGradient of input (column gather):")
    print(x2.grad)
    
    return x, result


def demonstrate_scatter():
    """Demonstrate scatter operation with gradient tracking."""
    print("\n" + "=" * 60)
    print("Scatter Operation Demonstration")
    print("=" * 60)
    
    # Create input tensor (will be modified in-place in scatter)
    data = np.zeros((3, 4), dtype=np.float32)
    x = Tensor(data, requires_grad=True)
    print(f"Input tensor (shape {x.shape}):")
    print(x.data)
    
    # Create source values to scatter
    src_data = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    src = Tensor(src_data, requires_grad=True)
    print(f"\nSource tensor (shape {src.shape}):")
    print(src.data)
    
    # Create indices for scattering along dimension 0 (rows)
    # Each src element will be placed at (index[i,j], j) in output
    indices = np.array([[0, 1, 0, 1],
                        [2, 1, 2, 1],
                        [0, 2, 0, 2]], dtype=np.int64)
    idx = Tensor(indices, requires_grad=False)
    
    # Scatter src into x along dimension 0
    result = x.scatter(dim=0, index=idx, src=src)
    print(f"\nIndices for scatter along dim=0 (shape {idx.shape}):")
    print(indices)
    print(f"\nScattered result (shape {result.shape}):")
    print(result.data)
    
    # Verify with NumPy equivalent
    np_result = np.zeros((3, 4), dtype=np.float32)
    np.put_along_axis(np_result, indices, src_data, axis=0)
    print(f"\nNumPy equivalent (np.put_along_axis):")
    print(np_result)
    print(f"Match with nanotorch: {np.allclose(result.data, np_result)}")
    
    # Compute gradient
    loss = result.sum()
    loss.backward()
    print(f"\nGradient of source tensor w.r.t. loss (sum of scattered result):")
    print(src.grad)
    print("\nInterpretation: Each source element's gradient is 1 because it")
    print("contributes directly to the sum through its scattered position.")
    print(f"\nGradient of input tensor (zeros):")
    print(x.grad)
    print("Input tensor gradient is zero because scattering overwrites input values.")
    
    # Scatter along dimension 1 (columns)
    print("\n" + "-" * 40)
    print("Scatter along dimension 1 (columns)")
    x2 = Tensor(np.zeros((3, 4), dtype=np.float32), requires_grad=True)
    src2 = Tensor(np.arange(1, 7, dtype=np.float32).reshape(3, 2), requires_grad=True)
    indices_col = np.array([[0, 2],
                           [1, 3],
                           [2, 0]], dtype=np.int64)
    idx_col = Tensor(indices_col, requires_grad=False)
    result_col = x2.scatter(dim=1, index=idx_col, src=src2)
    print(f"Input shape: {x2.shape}")
    print(f"Source shape: {src2.shape}")
    print(f"Indices shape: {idx_col.shape}")
    print(f"Result shape: {result_col.shape}")
    print("Result:")
    print(result_col.data)
    
    # Gradient check
    loss_col = result_col.sum()
    loss_col.backward()
    print(f"\nGradient of source (column scatter):")
    print(src2.grad)
    
    return x, src, result


def practical_use_case():
    """Practical use case: top-k selection and embedding updates."""
    print("\n" + "=" * 60)
    print("Practical Use Case: Top-k Selection and Embedding Updates")
    print("=" * 60)
    
    # Example 1: Select top-k values from each row
    print("\n1. Selecting top-2 values from each row of a matrix")
    scores = Tensor(np.random.randn(5, 10), requires_grad=True)
    print(f"Scores shape: {scores.shape}")
    
    # Get indices of top-2 values per row
    scores_np = scores.data
    topk_indices = np.argsort(scores_np, axis=1)[:, -2:]  # shape (5, 2)
    topk_indices = np.flip(topk_indices, axis=1)  # descending order
    idx = Tensor(topk_indices, requires_grad=False)
    
    # Gather top-k values
    topk_values = scores.gather(dim=1, index=idx)
    print(f"Top-2 indices per row:")
    print(topk_indices)
    print(f"Top-2 values per row:")
    print(topk_values.data)
    
    # Example 2: Update embedding matrix with scattered gradients
    print("\n2. Updating embedding matrix with scattered gradients")
    # Simulate word embeddings
    vocab_size = 10
    embedding_dim = 4
    embeddings = Tensor(np.random.randn(vocab_size, embedding_dim), requires_grad=True)
    
    # Simulate batch of word indices
    batch_indices = np.array([2, 5, 2, 7], dtype=np.int64)  # shape (4,)
    # Expand indices for scattering
    expanded_indices = batch_indices.reshape(-1, 1).repeat(embedding_dim, axis=1)
    idx_emb = Tensor(expanded_indices, requires_grad=False)
    
    # Simulate gradient updates for these embeddings
    grad_updates = Tensor(np.ones((len(batch_indices), embedding_dim)), requires_grad=True)
    
    # Scatter updates into a zero tensor
    grad_accumulator = Tensor.zeros((vocab_size, embedding_dim), requires_grad=False)
    updated = grad_accumulator.scatter(dim=0, index=idx_emb, src=grad_updates)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Batch indices: {batch_indices}")
    print(f"Gradient updates shape: {grad_updates.shape}")
    print(f"Accumulated gradients (sum per embedding):")
    print(updated.data)
    
    # Note: In practice, you'd subtract these gradients from embeddings
    print("\nThis demonstrates how scatter can be used to accumulate gradients")
    print("for embedding layers where multiple batch elements reference the same embedding.")
    
    return scores, embeddings


def compare_performance():
    """Quick performance comparison with NumPy."""
    print("\n" + "=" * 60)
    print("Performance Comparison (small tensors)")
    print("=" * 60)
    
    import time
    
    # Gather performance
    data = np.random.randn(100, 50).astype(np.float32)
    indices = np.random.randint(0, 100, size=(30, 50), dtype=np.int64)
    
    # NumPy
    start = time.perf_counter()
    for _ in range(100):
        np.take_along_axis(data, indices, axis=0)
    np_time = (time.perf_counter() - start) / 100
    
    # nanotorch (no grad)
    x = Tensor(data, requires_grad=False)
    idx = Tensor(indices, requires_grad=False)
    start = time.perf_counter()
    for _ in range(100):
        x.gather(dim=0, index=idx)
    nt_time = (time.perf_counter() - start) / 100
    
    print(f"Gather operation (100 repetitions):")
    print(f"  NumPy: {np_time*1000:.3f} ms per call")
    print(f"  nanotorch (no grad): {nt_time*1000:.3f} ms per call")
    print(f"  Overhead: {nt_time/np_time:.2f}x")
    
    # Scatter performance (small)
    src = np.random.randn(30, 50).astype(np.float32)
    
    # NumPy
    start = time.perf_counter()
    for _ in range(100):
        result = np.zeros_like(data)
        np.put_along_axis(result, indices, src, axis=0)
    np_time_scatter = (time.perf_counter() - start) / 100
    
    # nanotorch
    x = Tensor(np.zeros_like(data), requires_grad=False)
    idx = Tensor(indices, requires_grad=False)
    s = Tensor(src, requires_grad=False)
    start = time.perf_counter()
    for _ in range(100):
        x.scatter(dim=0, index=idx, src=s)
    nt_time_scatter = (time.perf_counter() - start) / 100
    
    print(f"\nScatter operation (100 repetitions):")
    print(f"  NumPy: {np_time_scatter*1000:.3f} ms per call")
    print(f"  nanotorch (no grad): {nt_time_scatter*1000:.3f} ms per call")
    print(f"  Overhead: {nt_time_scatter/np_time_scatter:.2f}x")
    
    print("\nNote: For more accurate benchmarks, run benchmarks/tensor_operations.py")
    print("with --op gather or --op scatter")


def main():
    """Run all demonstrations."""
    print(__doc__)
    
    # Demonstrate gather
    x_gather, result_gather = demonstrate_gather()
    
    # Demonstrate scatter  
    x_scatter, src_scatter, result_scatter = demonstrate_scatter()
    
    # Practical use case
    scores, embeddings = practical_use_case()
    
    # Performance comparison
    compare_performance()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("Gather and scatter operations are essential for:")
    print("1. Index-based selection (e.g., top-k, batch selection)")
    print("2. Sparse updates (e.g., embedding gradients)")
    print("3. Advanced indexing patterns")
    print("\nKey points:")
    print("- Indices do not require gradients (same as PyTorch)")
    print("- Gather gradient: input gradient accumulates based on index usage")
    print("- Scatter gradient: source gradient propagates to output positions")
    print("- NumPy equivalents: take_along_axis (gather), put_along_axis (scatter)")
    print("\nRun `python benchmarks/tensor_operations.py --op gather --repeats 10`")
    print("for comprehensive performance benchmarks.")


if __name__ == "__main__":
    main()