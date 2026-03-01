"""
Autograd demonstration for nanotorch.

This example demonstrates:
1. Computational graph building and visualization
2. Gradient checking with finite differences
3. Custom autograd functions
4. Advanced gradient flow scenarios
"""

import numpy as np
import nanotorch as nt
from nanotorch.autograd import Function
from nanotorch.tensor import no_grad


def demonstrate_computational_graph():
    """Show how tensors build computational graphs."""
    print("=" * 50)
    print("1. Computational Graph Demonstration")
    print("=" * 50)
    
    # Create leaf tensors
    x = nt.Tensor([2.0], requires_grad=True)
    y = nt.Tensor([3.0], requires_grad=True)
    
    print(f"Leaf tensors:")
    print(f"  x = {x.data}, requires_grad={x.requires_grad}")
    print(f"  y = {y.data}, requires_grad={y.requires_grad}")
    
    # Build computation
    z = x * y  # Multiplication
    w = z + y  # Addition
    result = w ** 2  # Power
    
    print(f"\nComputation: result = ((x * y) + y) ** 2")
    print(f"  z = x * y = {z.data}")
    print(f"  w = z + y = {w.data}")
    print(f"  result = w ** 2 = {result.data}")
    
    # Show graph structure
    print(f"\nComputational graph:")
    print(f"  result._op: {result._op}")
    print(f"  result._parents: {len(result._parents)} parents")
    
    # Backward pass
    result.backward()
    
    print(f"\nGradients after backward:")
    print(f"  x.grad = {x.grad}")
    print(f"  y.grad = {y.grad}")
    
    # Verify gradients mathematically
    # Let f(x,y) = ((x*y) + y)^2
    # df/dx = 2*((x*y)+y) * y
    # df/dy = 2*((x*y)+y) * (x + 1)
    expected_grad_x = 2 * ((2*3) + 3) * 3  # 2*(6+3)*3 = 2*9*3 = 54
    expected_grad_y = 2 * ((2*3) + 3) * (2 + 1)  # 2*9*3 = 54
    print(f"\nExpected gradients (manual calculation):")
    print(f"  df/dx = {expected_grad_x}")
    print(f"  df/dy = {expected_grad_y}")
    
    if abs(x.grad.data[0] - expected_grad_x) < 1e-6:
        print("✓ x.grad matches expected value")
    else:
        print(f"✗ x.grad mismatch: {x.grad.data[0]} vs {expected_grad_x}")

    if abs(y.grad.data[0] - expected_grad_y) < 1e-6:
        print("✓ y.grad matches expected value")
    else:
        print(f"✗ y.grad mismatch: {y.grad.data[0]} vs {expected_grad_y}")


def gradient_checking():
    """Compare autograd gradients with finite differences."""
    print("\n" + "=" * 50)
    print("2. Gradient Checking with Finite Differences")
    print("=" * 50)
    
    # Define a simple function: f(x) = x^3 + 2x^2 + 3x + 4
    def func(x):
        return x ** 3 + 2 * x ** 2 + 3 * x + 4
    
    # Create tensor
    x = nt.Tensor([1.5], requires_grad=True)
    
    # Forward pass with autograd
    y = func(x)
    y.backward()
    autograd_grad = x.grad.data[0]
    
    # Finite difference gradient
    epsilon = 1e-5
    x_plus = nt.Tensor([1.5 + epsilon])
    x_minus = nt.Tensor([1.5 - epsilon])
    y_plus = func(x_plus).data[0]
    y_minus = func(x_minus).data[0]
    finite_diff_grad = (y_plus - y_minus) / (2 * epsilon)
    
    print(f"Function: f(x) = x^3 + 2x^2 + 3x + 4")
    print(f"Point: x = {x.data[0]}")
    print(f"\nGradients:")
    print(f"  Autograd:    {autograd_grad:.8f}")
    print(f"  Finite diff: {finite_diff_grad:.8f}")
    print(f"  Difference:  {abs(autograd_grad - finite_diff_grad):.2e}")
    
    if abs(autograd_grad - finite_diff_grad) < 1e-6:
        print("✓ Gradient check passed!")
    else:
        print("✗ Gradient check failed!")


class CustomSigmoid(Function):
    """Custom sigmoid activation function implemented as an autograd Function."""
    
    @staticmethod
    def forward(ctx, x):
        """Forward pass: sigmoid(x) = 1 / (1 + exp(-x))."""
        # Compute sigmoid
        sigmoid = 1.0 / (1.0 + np.exp(-x.data))
        result = nt.Tensor(sigmoid, _op=CustomSigmoid, _parents=(x,))
        
        # Save for backward pass
        ctx.saved_tensors = [result]
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: derivative of sigmoid = sigmoid * (1 - sigmoid)."""
        sigmoid = ctx.saved_tensors[0]
        # grad_input = grad_output * sigmoid * (1 - sigmoid)
        grad_data = grad_output.data * sigmoid.data * (1.0 - sigmoid.data)
        
        # Return gradient for the input tensor
        # Since sigmoid has one parent, we return one gradient
        return nt.Tensor(grad_data, requires_grad=False)


def custom_sigmoid(x):
    """Functional wrapper for custom sigmoid."""
    return CustomSigmoid.apply(x)


def demonstrate_custom_function():
    """Show how to create and use custom autograd functions."""
    print("\n" + "=" * 50)
    print("3. Custom Autograd Function: Sigmoid")
    print("=" * 50)
    
    # Create input tensor
    x = nt.Tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
    
    # Apply custom sigmoid
    y = custom_sigmoid(x)
    
    print(f"Input x: {x.data}")
    print(f"Sigmoid(x): {y.data}")
    
    # Compare with numpy sigmoid
    expected = 1.0 / (1.0 + np.exp(-x.data))
    print(f"Expected (numpy): {expected}")
    
    if np.allclose(y.data, expected, rtol=1e-6):
        print("✓ Forward pass matches numpy sigmoid")
    else:
        print("✗ Forward pass mismatch")
    
    # Test backward pass
    y.backward(np.ones_like(y.data))
    
    # Compute expected gradient: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    expected_grad = expected * (1.0 - expected)
    print(f"\nGradients:")
    print(f"  Autograd: {x.grad}")
    print(f"  Expected: {expected_grad}")
    
    if np.allclose(x.grad.data, expected_grad, rtol=1e-6):
        print("✓ Backward pass matches expected gradient")
    else:
        print("✗ Backward pass mismatch")


def demonstrate_gradient_accumulation():
    """Show how gradients accumulate across multiple backward passes."""
    print("\n" + "=" * 50)
    print("4. Gradient Accumulation")
    print("=" * 50)
    
    # Create a tensor
    x = nt.Tensor([3.0], requires_grad=True)
    
    # Multiple forward-backward passes
    gradients = []
    for i in range(3):
        y = x * (i + 1)  # Different operations
        y.backward()
        gradients.append(x.grad.data.copy())
        print(f"Pass {i+1}: y = x * {i+1}, x.grad = {x.grad}")
    
    print(f"\nFinal x.grad: {x.grad}")
    print("Note: Gradients accumulate across backward calls (x.grad += new_grad)")
    
    # Reset gradients
    x.zero_grad()
    print(f"After zero_grad(): x.grad = {x.grad}")


def demonstrate_no_grad_context():
    """Show how no_grad() context manager disables gradient tracking."""
    print("\n" + "=" * 50)
    print("5. no_grad() Context Manager")
    print("=" * 50)
    
    # Create tensor with gradient tracking
    x = nt.Tensor([2.0], requires_grad=True)
    
    # Operation inside no_grad()
    with no_grad():
        y = x * 3
        z = y ** 2
    
    print(f"Operations inside no_grad():")
    print(f"  x = {x.data}, requires_grad={x.requires_grad}")
    print(f"  y = {y.data}, requires_grad={y.requires_grad}")
    print(f"  z = {z.data}, requires_grad={z.requires_grad}")
    print(f"  y._op: {y._op}")
    print(f"  z._op: {z._op}")
    
    # Try backward (should fail or have no effect)
    try:
        z.backward()
        print(f"  x.grad after backward: {x.grad}")
    except Exception as e:
        print(f"  Backward failed (expected): {e}")


def demonstrate_higher_order_gradients():
    """Show second-order gradients (gradients of gradients)."""
    print("\n" + "=" * 50)
    print("6. Higher-Order Gradients")
    print("=" * 50)
    
    # Create tensor
    x = nt.Tensor([2.0], requires_grad=True)
    
    # First forward pass
    y = x ** 3  # y = x^3
    y.backward()
    first_grad = x.grad.data.copy()
    print(f"First derivative (dy/dx): {first_grad}")
    
    # Clear gradient for second pass
    x.zero_grad()
    
    # Compute gradient of gradient (second derivative)
    # d²y/dx² = d/dx (3x^2) = 6x
    # We can compute by taking gradient of first_grad w.r.t x
    # But nanotorch may not support higher-order gradients directly
    # For demonstration, we'll compute manually
    expected_second_grad = 6 * x.data[0]
    print(f"Expected second derivative (d²y/dx²): {expected_second_grad}")
    print("\nNote: Higher-order gradients require retain_grad or double backward,")
    print("      which may not be fully implemented in nanotorch.")


if __name__ == "__main__":
    print("=" * 50)
    print("nanotorch - Autograd Demonstration")
    print("=" * 50)
    
    # Run demonstrations
    demonstrate_computational_graph()
    gradient_checking()
    demonstrate_custom_function()
    demonstrate_gradient_accumulation()
    demonstrate_no_grad_context()
    demonstrate_higher_order_gradients()
    
    print("\n" + "=" * 50)
    print("All demonstrations completed!")
    print("=" * 50)