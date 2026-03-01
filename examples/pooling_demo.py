#!/usr/bin/env python3
"""
Demonstration of 2D pooling layers (MaxPool2d and AvgPool2d) in nanotorch.

This script shows how to use the pooling layers, compute gradients,
and verify output shapes match the expected formulas.
"""

import numpy as np
from nanotorch import Tensor
from nanotorch.nn import MaxPool2d, AvgPool2d, Sequential, AdaptiveAvgPool2d, AdaptiveMaxPool2d
from nanotorch.utils import manual_seed


def demo_maxpool2d():
    """Demonstrate MaxPool2d functionality."""
    print("=" * 60)
    print("MaxPool2d Demonstration")
    print("=" * 60)
    
    # Set seed for reproducibility
    manual_seed(42)
    
    # Create input tensor: batch=2, channels=3, height=8, width=8
    x = Tensor.randn((2, 3, 8, 8), requires_grad=True)
    print(f"Input shape: {x.shape}")
    print(f"Input data range: [{x.data.min():.3f}, {x.data.max():.3f}]")
    
    # Test 1: Basic max pooling with kernel_size=2, stride=2
    pool1 = MaxPool2d(kernel_size=2, stride=2)
    out1 = pool1(x)
    print(f"\n1. Basic MaxPool2d(kernel_size=2, stride=2):")
    print(f"   Output shape: {out1.shape} (expected: (2, 3, 4, 4))")
    print(f"   Output range: [{out1.data.min():.3f}, {out1.data.max():.3f}]")
    
    # Test 2: Max pooling with padding
    pool2 = MaxPool2d(kernel_size=3, stride=2, padding=1)
    out2 = pool2(x)
    print(f"\n2. MaxPool2d(kernel_size=3, stride=2, padding=1):")
    print(f"   Output shape: {out2.shape} (expected: (2, 3, 4, 4))")
    
    # Test 3: Max pooling with ceil_mode=True
    pool3 = MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
    out3 = pool3(x)
    print(f"\n3. MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True):")
    print(f"   Output shape: {out3.shape} (expected: (2, 3, 4, 4) with this input)")
    
    # Test 4: Gradient computation
    print(f"\n4. Gradient computation test:")
    # Create a simple loss (sum of outputs)
    loss = out1.sum()
    print(f"   Loss (sum of outputs): {loss.item():.3f}")
    
    # Zero gradients (if any) and compute backward
    if x.grad is not None:
        x.zero_grad()
    loss.backward()
    
    if x.grad is not None:
        grad_norm = np.sqrt((x.grad.data ** 2).sum())
        print(f"   Gradient norm w.r.t input: {grad_norm:.3f}")
        print(f"   Gradient shape: {x.grad.shape}")
        # Check that gradient only flows to max positions
        nonzero_grad = np.count_nonzero(x.grad.data)
        total_elements = x.grad.data.size
        print(f"   Non-zero gradient elements: {nonzero_grad}/{total_elements} "
              f"({100*nonzero_grad/total_elements:.1f}%)")
    
    # Test 5: Return indices (now fully supported)
    pool5 = MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    out5 = pool5(x)
    print(f"\n5. MaxPool2d with return_indices=True:")
    print(f"   Output type: {type(out5)}")
    if isinstance(out5, tuple):
        output_tensor, indices = out5
        print(f"   Output tensor shape: {output_tensor.shape}")
        print(f"   Indices shape: {indices.shape}")
        print(f"   Indices dtype: {indices.dtype}")
        # Verify indices correctness
        # For each output position, check that index points to max in window
        correct = 0
        total = 0
        for n in range(output_tensor.shape[0]):
            for c in range(output_tensor.shape[1]):
                for h in range(output_tensor.shape[2]):
                    for w in range(output_tensor.shape[3]):
                        h_start = h * 2
                        w_start = w * 2
                        window = x.data[n, c, h_start:h_start+2, w_start:w_start+2]
                        max_val = np.max(window)
                        idx = indices[n, c, h, w]
                        # Decode index to position
                        H_in, W_in = x.shape[2], x.shape[3]
                        idx_within = idx - (n * (x.shape[1] * H_in * W_in) + c * (H_in * W_in))
                        h_idx = idx_within // W_in
                        w_idx = idx_within % W_in
                        if h_start <= h_idx < h_start+2 and w_start <= w_idx < w_start+2:
                            if np.allclose(x.data[n, c, h_idx, w_idx], max_val):
                                correct += 1
                        total += 1
        print(f"   Indices correctness: {correct}/{total} ({100*correct/total:.1f}%)")
    else:
        print(f"   Note: return_indices=True but got single tensor")
    
    print("\nMaxPool2d demo completed.\n")


def demo_avgpool2d():
    """Demonstrate AvgPool2d functionality."""
    print("=" * 60)
    print("AvgPool2d Demonstration")
    print("=" * 60)
    
    manual_seed(123)
    
    x = Tensor.randn((2, 3, 8, 8), requires_grad=True)
    print(f"Input shape: {x.shape}")
    
    # Test 1: Basic average pooling
    pool1 = AvgPool2d(kernel_size=2, stride=2)
    out1 = pool1(x)
    print(f"\n1. Basic AvgPool2d(kernel_size=2, stride=2):")
    print(f"   Output shape: {out1.shape} (expected: (2, 3, 4, 4))")
    print(f"   Output range: [{out1.data.min():.3f}, {out1.data.max():.3f}]")
    
    # Test 2: Average pooling with padding
    pool2 = AvgPool2d(kernel_size=3, stride=2, padding=1)
    out2 = pool2(x)
    print(f"\n2. AvgPool2d(kernel_size=3, stride=2, padding=1):")
    print(f"   Output shape: {out2.shape} (expected: (2, 3, 4, 4))")
    
    # Test 3: Gradient computation
    print(f"\n3. Gradient computation test:")
    loss = out1.sum()
    print(f"   Loss (sum of outputs): {loss.item():.3f}")
    
    if x.grad is not None:
        x.zero_grad()
    loss.backward()
    
    if x.grad is not None:
        grad_norm = np.sqrt((x.grad.data ** 2).sum())
        print(f"   Gradient norm w.r.t input: {grad_norm:.3f}")
        print(f"   Gradient shape: {x.grad.shape}")
        # For average pooling, all input positions in window get gradient
        nonzero_grad = np.count_nonzero(x.grad.data)
        total_elements = x.grad.data.size
        print(f"   Non-zero gradient elements: {nonzero_grad}/{total_elements} "
              f"({100*nonzero_grad/total_elements:.1f}%)")
    
    # Test 4: count_include_pad=False
    pool4 = AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
    out4 = pool4(x)
    print(f"\n4. AvgPool2d with count_include_pad=False:")
    print(f"   Output shape: {out4.shape}")
    # Compare with count_include_pad=True
    diff = np.abs(out4.data - out2.data).max()
    print(f"   Max difference from count_include_pad=True: {diff:.6f}")
    
    print("\nAvgPool2d demo completed.\n")


def demo_adaptive_pooling():
    """Demonstrate adaptive pooling layers."""
    print("=" * 60)
    print("Adaptive Pooling Demonstration")
    print("=" * 60)
    
    manual_seed(456)
    
    x = Tensor.randn((2, 3, 8, 8), requires_grad=True)
    print(f"Input shape: {x.shape}")
    
    # Adaptive average pooling
    print(f"\n1. AdaptiveAvgPool2d(output_size=4):")
    pool1 = AdaptiveAvgPool2d(output_size=4)
    out1 = pool1(x)
    print(f"   Output shape: {out1.shape} (expected: (2, 3, 4, 4))")
    
    # Global average pooling (output_size=1)
    print(f"\n2. AdaptiveAvgPool2d(output_size=1) - Global average pooling:")
    pool2 = AdaptiveAvgPool2d(output_size=1)
    out2 = pool2(x)
    print(f"   Output shape: {out2.shape}")
    # Verify each output is mean of corresponding channel spatial dims
    for n in range(2):
        for c in range(3):
            expected = np.mean(x.data[n, c, :, :])
            actual = out2.data[n, c, 0, 0]
            assert np.allclose(actual, expected, rtol=1e-5), f"Global avg mismatch: {actual} vs {expected}"
    print("   Verified: each output equals spatial mean of input channel")
    
    # Adaptive max pooling with return_indices
    print(f"\n3. AdaptiveMaxPool2d(output_size=(3, 5), return_indices=True):")
    pool3 = AdaptiveMaxPool2d(output_size=(3, 5), return_indices=True)
    out3 = pool3(x)
    if isinstance(out3, tuple):
        output_tensor, indices = out3
        print(f"   Output tensor shape: {output_tensor.shape}")
        print(f"   Indices shape: {indices.shape}")
        print(f"   Indices dtype: {indices.dtype}")
    
    # Gradient computation for adaptive avg pooling
    print(f"\n4. Gradient computation for AdaptiveAvgPool2d:")
    loss = out1.sum()
    loss.backward()
    if x.grad is not None:
        grad_norm = np.sqrt((x.grad.data ** 2).sum())
        print(f"   Gradient norm w.r.t input: {grad_norm:.3f}")
        nonzero = np.count_nonzero(x.grad.data)
        total = x.grad.data.size
        print(f"   Non-zero gradient elements: {nonzero}/{total} ({100*nonzero/total:.1f}%)")
    
    print("\nAdaptive pooling demo completed.\n")


def demo_pooling_in_network():
    """Show pooling layers in a simple neural network."""
    print("=" * 60)
    print("Pooling in a Neural Network")
    print("=" * 60)
    
    manual_seed(999)
    
    # Create a simple CNN with pooling
    model = Sequential(
        # Conv layer would go here, but we'll just use pooling for demo
        MaxPool2d(kernel_size=2, stride=2),
        AvgPool2d(kernel_size=2, stride=2)
    )
    
    print(f"Model architecture:\n{model}")
    
    # Create dummy input
    x = Tensor.randn((4, 3, 16, 16), requires_grad=True)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    out = model(x)
    print(f"Output shape after two pooling layers: {out.shape}")
    print(f"Output range: [{out.data.min():.3f}, {out.data.max():.3f}]")
    
    # Compute loss and gradients
    loss = out.sum()
    loss.backward()
    
    print(f"\nLoss: {loss.item():.3f}")
    if x.grad is not None:
        print(f"Input gradient norm: {np.sqrt((x.grad.data ** 2).sum()):.3f}")
    
    print("\nNetwork demo completed.\n")


def output_shape_formula():
    """Demonstrate output shape formula for pooling layers."""
    print("=" * 60)
    print("Output Shape Formula Verification")
    print("=" * 60)
    
    # Formula from PyTorch documentation:
    # H_out = floor((H_in + 2*padding - kernel_size) / stride + 1) for ceil_mode=False
    # H_out = ceil((H_in + 2*padding - kernel_size) / stride + 1) for ceil_mode=True
    
    test_cases = [
        {"H_in": 8, "kernel": 2, "stride": 2, "padding": 0, "ceil": False, "expected": 4},
        {"H_in": 7, "kernel": 2, "stride": 2, "padding": 0, "ceil": False, "expected": 3},
        {"H_in": 7, "kernel": 2, "stride": 2, "padding": 0, "ceil": True, "expected": 4},
        {"H_in": 8, "kernel": 3, "stride": 2, "padding": 1, "ceil": False, "expected": 4},
        {"H_in": 8, "kernel": 3, "stride": 2, "padding": 1, "ceil": True, "expected": 4},
        {"H_in": 5, "kernel": 3, "stride": 2, "padding": 1, "ceil": False, "expected": 3},
        {"H_in": 5, "kernel": 3, "stride": 2, "padding": 1, "ceil": True, "expected": 3},
    ]
    
    print("Testing output shape formula with various inputs:\n")
    print("H_in | kernel | stride | pad | ceil | Expected | Actual | Match")
    print("-" * 65)
    
    for case in test_cases:
        H_in = case["H_in"]
        kernel = case["kernel"]
        stride = case["stride"]
        padding = case["padding"]
        ceil_mode = case["ceil"]
        expected = case["expected"]
        
        # Create input tensor
        x = Tensor.randn((1, 1, H_in, H_in), requires_grad=False)
        
        # Create pooling layer
        if ceil_mode:
            pool = MaxPool2d(kernel_size=kernel, stride=stride, padding=padding, ceil_mode=True)
        else:
            pool = MaxPool2d(kernel_size=kernel, stride=stride, padding=padding, ceil_mode=False)
        
        out = pool(x)
        actual = out.shape[2]  # height
        
        match = "✓" if actual == expected else "✗"
        print(f"{H_in:4} | {kernel:6} | {stride:6} | {padding:3} | {str(ceil_mode):4} | {expected:8} | {actual:6} | {match}")
    
    print("\nFormula verification completed.\n")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("nanotorch 2D Pooling Layers Demonstration")
    print("=" * 70 + "\n")
    
    demo_maxpool2d()
    demo_avgpool2d()
    demo_adaptive_pooling()
    demo_pooling_in_network()
    output_shape_formula()
    
    print("=" * 70)
    print("All demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()