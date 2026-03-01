"""
Comprehensive tests for automatic differentiation in nanotorch.

This module tests gradient correctness for all tensor operations using
finite difference gradient checking.
"""

import numpy as np
import nanotorch as nt


def finite_difference_gradient(func, tensor, eps=1e-5):
    """Compute gradient using finite differences.

    Args:
        func: Callable that returns a scalar loss given tensor data
        tensor: Tensor to compute gradient for
        eps: Perturbation size for finite differences

    Returns:
        numpy array with numerical gradient
    """
    numerical_grad = np.zeros_like(tensor.data)

    # Iterate over all elements of the tensor
    it = np.nditer(tensor.data, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        original = tensor.data[idx].copy()

        # Compute f(x + eps)
        tensor.data[idx] = original + eps
        loss_plus = func()

        # Compute f(x - eps)
        tensor.data[idx] = original - eps
        loss_minus = func()

        # Central difference
        numerical_grad[idx] = (loss_plus - loss_minus) / (2 * eps)

        # Restore original value
        tensor.data[idx] = original

        it.iternext()

    return numerical_grad


def gradient_check(
    operation_func, inputs, output_idx=0, eps=1e-3, rtol=2e-2, atol=1e-3
):
    """Check gradient correctness for an operation.

    Args:
        operation_func: Function that takes inputs and returns output tensor(s)
        inputs: List of input tensors (all require_grad=True)
        output_idx: Which output to check gradients against (default 0)
        eps: Finite difference epsilon
        rtol: Relative tolerance for gradient comparison
        atol: Absolute tolerance for gradient comparison

    Returns:
        bool: True if gradients match within tolerance
        dict: Detailed results for each input
    """
    # Run operation to get output
    output = operation_func(*inputs)
    if isinstance(output, tuple):
        output_tensor = output[output_idx]
    else:
        output_tensor = output

    # Create a scalar loss (sum of output)
    loss = output_tensor.sum()

    # Compute gradients via autograd
    for tensor in inputs:
        tensor.zero_grad()
    loss.backward()

    results = {}
    all_pass = True

    for i, tensor in enumerate(inputs):
        if not tensor.requires_grad:
            continue

        # Define function for finite differences
        def closure():
            # Re-run operation with current tensor data
            return operation_func(*inputs).sum().data.item()

        # Save original data and gradient
        original_data = tensor.data.copy()

        # Compute numerical gradient
        numerical_grad = finite_difference_gradient(closure, tensor, eps=eps)

        # Get analytical gradient
        analytical_grad = (
            tensor.grad.data if tensor.grad is not None else np.zeros_like(tensor.data)
        )

        # Compare gradients
        diff = np.abs(analytical_grad - numerical_grad)
        max_diff = np.max(diff)
        rel_diff = diff / (np.abs(numerical_grad) + 1e-8)
        max_rel_diff = np.max(rel_diff)

        # Check if gradients match within tolerance
        match = np.allclose(analytical_grad, numerical_grad, rtol=rtol, atol=atol)

        results[f"input_{i}"] = {
            "shape": tensor.shape,
            "max_diff": max_diff,
            "max_rel_diff": max_rel_diff,
            "match": match,
            "analytical_grad_norm": np.linalg.norm(analytical_grad.ravel()),
            "numerical_grad_norm": np.linalg.norm(numerical_grad.ravel()),
        }

        if not match:
            all_pass = False

        # Restore original data
        tensor.data[...] = original_data

    return all_pass, results


def test_gradient_addition():
    """Test gradient for element-wise addition."""
    print("Testing gradient for addition...")

    # Test 1: Simple scalar addition
    a = nt.Tensor(3.0, requires_grad=True)
    b = nt.Tensor(4.0, requires_grad=True)

    def add_func(x, y):
        return x + y

    pass_check, results = gradient_check(add_func, [a, b])
    assert pass_check, f"Addition gradient failed: {results}"
    print("  ✓ Scalar addition")

    # Test 2: Vector addition
    a = nt.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = nt.Tensor([4.0, 5.0, 6.0], requires_grad=True)

    pass_check, results = gradient_check(add_func, [a, b])
    assert pass_check, f"Vector addition gradient failed: {results}"
    print("  ✓ Vector addition")

    # Test 3: Broadcasting addition
    a = nt.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # 2x2
    b = nt.Tensor([1.0, 2.0], requires_grad=True)  # 2

    pass_check, results = gradient_check(add_func, [a, b])
    assert pass_check, f"Broadcast addition gradient failed: {results}"
    print("  ✓ Broadcasting addition")

    print("✓ All addition gradient tests passed")


def test_gradient_multiplication():
    """Test gradient for element-wise multiplication."""
    print("Testing gradient for multiplication...")

    # Test 1: Simple scalar multiplication
    a = nt.Tensor(3.0, requires_grad=True)
    b = nt.Tensor(4.0, requires_grad=True)

    def mul_func(x, y):
        return x * y

    pass_check, results = gradient_check(mul_func, [a, b])
    assert pass_check, f"Multiplication gradient failed: {results}"
    print("  ✓ Scalar multiplication")

    # Test 2: Vector multiplication
    a = nt.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = nt.Tensor([4.0, 5.0, 6.0], requires_grad=True)

    pass_check, results = gradient_check(mul_func, [a, b])
    assert pass_check, f"Vector multiplication gradient failed: {results}"
    print("  ✓ Vector multiplication")

    # Test 3: Broadcasting multiplication
    a = nt.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = nt.Tensor([0.5, 2.0], requires_grad=True)

    pass_check, results = gradient_check(mul_func, [a, b])
    assert pass_check, f"Broadcast multiplication gradient failed: {results}"
    print("  ✓ Broadcasting multiplication")

    print("✓ All multiplication gradient tests passed")


def test_gradient_matmul():
    """Test gradient for matrix multiplication."""
    print("Testing gradient for matrix multiplication...")

    # Test 1: 2x2 @ 2x2
    a = nt.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = nt.Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    def matmul_func(x, y):
        return x @ y

    pass_check, results = gradient_check(matmul_func, [a, b])
    assert pass_check, f"Matmul gradient failed: {results}"
    print("  ✓ 2x2 matrix multiplication")

    # Test 2: 3x2 @ 2x4
    a = nt.Tensor.randn((3, 2), requires_grad=True)
    b = nt.Tensor.randn((2, 4), requires_grad=True)

    pass_check, results = gradient_check(matmul_func, [a, b])
    assert pass_check, f"Matmul gradient failed: {results}"
    print("  ✓ 3x2 @ 2x4 matrix multiplication")

    print("✓ All matmul gradient tests passed")


def test_gradient_division():
    """Test gradient for element-wise division."""
    print("Testing gradient for division...")

    # Test 1: Simple scalar division
    a = nt.Tensor(6.0, requires_grad=True)
    b = nt.Tensor(2.0, requires_grad=True)

    def div_func(x, y):
        return x / y

    pass_check, results = gradient_check(div_func, [a, b])
    assert pass_check, f"Division gradient failed: {results}"
    print("  ✓ Scalar division")

    # Test 2: Vector division
    a = nt.Tensor([1.0, 4.0, 9.0], requires_grad=True)
    b = nt.Tensor([1.0, 2.0, 3.0], requires_grad=True)

    pass_check, results = gradient_check(div_func, [a, b])
    assert pass_check, f"Vector division gradient failed: {results}"
    print("  ✓ Vector division")

    print("✓ All division gradient tests passed")


def test_gradient_power():
    """Test gradient for element-wise power."""
    print("Testing gradient for power...")

    # Test 1: Square (x^2)
    a = nt.Tensor([1.0, 2.0, 3.0], requires_grad=True)

    def pow_func(x):
        return x**2

    pass_check, results = gradient_check(pow_func, [a])
    assert pass_check, f"Power gradient failed: {results}"
    print("  ✓ x^2")

    # Test 2: Cube (x^3)
    def pow3_func(x):
        return x**3

    pass_check, results = gradient_check(pow3_func, [a])
    assert pass_check, f"Power gradient failed: {results}"
    print("  ✓ x^3")

    # Test 3: Square root (x^0.5)
    a = nt.Tensor([1.0, 4.0, 9.0], requires_grad=True)

    def sqrt_func(x):
        return x**0.5

    pass_check, results = gradient_check(sqrt_func, [a])
    assert pass_check, f"Square root gradient failed: {results}"
    print("  ✓ x^0.5")

    print("✓ All power gradient tests passed")


def test_gradient_activations():
    """Test gradient for activation functions."""
    print("Testing gradient for activation functions...")

    # Test 1: ReLU
    a = nt.Tensor([-2.0, -1.0, 0.5, 1.0, 2.0], requires_grad=True)

    def relu_func(x):
        return x.relu()

    pass_check, results = gradient_check(relu_func, [a])
    assert pass_check, f"ReLU gradient failed: {results}"
    print("  ✓ ReLU")

    # Test 2: Sigmoid
    def sigmoid_func(x):
        return x.sigmoid()

    pass_check, results = gradient_check(sigmoid_func, [a])
    assert pass_check, f"Sigmoid gradient failed: {results}"
    print("  ✓ Sigmoid")

    # Test 3: Tanh
    def tanh_func(x):
        return x.tanh()

    pass_check, results = gradient_check(tanh_func, [a])
    assert pass_check, f"Tanh gradient failed: {results}"
    print("  ✓ Tanh")

    # Test 4: Exp
    a = nt.Tensor([-1.0, 0.0, 1.0], requires_grad=True)

    def exp_func(x):
        return x.exp()

    pass_check, results = gradient_check(exp_func, [a])
    assert pass_check, f"Exp gradient failed: {results}"
    print("  ✓ Exp")

    # Test 5: Log
    a = nt.Tensor([0.1, 1.0, 10.0], requires_grad=True)

    def log_func(x):
        return x.log()

    pass_check, results = gradient_check(log_func, [a])
    assert pass_check, f"Log gradient failed: {results}"
    print("  ✓ Log")

    print("✓ All activation gradient tests passed")


def test_gradient_softmax():
    """Test gradient for softmax and log-softmax."""
    print("Testing gradient for softmax and log-softmax...")

    # Test 1: Softmax 1D
    a = nt.Tensor([1.0, 2.0, 3.0], requires_grad=True)

    def softmax_func_1d(x):
        return x.softmax(dim=-1)

    pass_check, results = gradient_check(softmax_func_1d, [a])
    assert pass_check, f"Softmax 1D gradient failed: {results}"
    print("  ✓ Softmax 1D")

    # Test 2: Softmax 2D along dim=0
    a = nt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)

    def softmax_func_dim0(x):
        return x.softmax(dim=0)

    pass_check, results = gradient_check(softmax_func_dim0, [a])
    assert pass_check, f"Softmax dim=0 gradient failed: {results}"
    print("  ✓ Softmax dim=0")

    # Test 3: Softmax 2D along dim=1
    def softmax_func_dim1(x):
        return x.softmax(dim=1)

    pass_check, results = gradient_check(softmax_func_dim1, [a])
    assert pass_check, f"Softmax dim=1 gradient failed: {results}"
    print("  ✓ Softmax dim=1")

    # Test 4: LogSoftmax 1D
    a = nt.Tensor([1.0, 2.0, 3.0], requires_grad=True)

    def log_softmax_func_1d(x):
        return x.log_softmax(dim=-1)

    pass_check, results = gradient_check(log_softmax_func_1d, [a])
    assert pass_check, f"LogSoftmax 1D gradient failed: {results}"
    print("  ✓ LogSoftmax 1D")

    # Test 5: LogSoftmax 2D dim=1
    a = nt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)

    def log_softmax_func_dim1(x):
        return x.log_softmax(dim=1)

    pass_check, results = gradient_check(log_softmax_func_dim1, [a])
    assert pass_check, f"LogSoftmax dim=1 gradient failed: {results}"
    print("  ✓ LogSoftmax dim=1")

    print("✓ All softmax/log-softmax gradient tests passed")


def test_gradient_shape_operations():
    """Test gradient for shape operations."""
    print("Testing gradient for shape operations...")

    # Test 1: Reshape
    a = nt.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    def reshape_func(x):
        return x.reshape((4,))

    pass_check, results = gradient_check(reshape_func, [a])
    assert pass_check, f"Reshape gradient failed: {results}"
    print("  ✓ Reshape")

    # Test 2: Sum (full reduction)
    def sum_func(x):
        return x.sum()

    pass_check, results = gradient_check(sum_func, [a])
    assert pass_check, f"Sum gradient failed: {results}"
    print("  ✓ Sum (full)")

    # Test 3: Sum along axis
    def sum_axis_func(x):
        return x.sum(axis=0)

    pass_check, results = gradient_check(sum_axis_func, [a])
    assert pass_check, f"Sum axis gradient failed: {results}"
    print("  ✓ Sum (axis=0)")

    # Test 4: Mean (full reduction)
    def mean_func(x):
        return x.mean()

    pass_check, results = gradient_check(mean_func, [a])
    assert pass_check, f"Mean gradient failed: {results}"
    print("  ✓ Mean (full)")

    # Test 5: Mean along axis
    def mean_axis_func(x):
        return x.mean(axis=1)

    pass_check, results = gradient_check(mean_axis_func, [a])
    assert pass_check, f"Mean axis gradient failed: {results}"
    print("  ✓ Mean (axis=1)")

    # Test 6: Squeeze
    a = nt.Tensor([[[1.0]], [[2.0]]], requires_grad=True)  # shape (2, 1, 1)

    def squeeze_func(x):
        return x.squeeze()

    pass_check, results = gradient_check(squeeze_func, [a])
    assert pass_check, f"Squeeze gradient failed: {results}"
    print("  ✓ Squeeze")

    # Test 7: Transpose
    a = nt.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    def transpose_func(x):
        return x.T

    pass_check, results = gradient_check(transpose_func, [a])
    assert pass_check, f"Transpose gradient failed: {results}"
    print("  ✓ Transpose")

    print("✓ All shape operation gradient tests passed")


def test_gradient_subtraction():
    """Test gradient for subtraction (implemented as addition with negation)."""
    print("Testing gradient for subtraction...")

    # Test 1: Simple scalar subtraction
    a = nt.Tensor(5.0, requires_grad=True)
    b = nt.Tensor(3.0, requires_grad=True)

    def sub_func(x, y):
        return x - y

    pass_check, results = gradient_check(sub_func, [a, b])
    assert pass_check, f"Subtraction gradient failed: {results}"
    print("  ✓ Scalar subtraction")

    # Test 2: Vector subtraction
    a = nt.Tensor([5.0, 6.0, 7.0], requires_grad=True)
    b = nt.Tensor([1.0, 2.0, 3.0], requires_grad=True)

    pass_check, results = gradient_check(sub_func, [a, b])
    assert pass_check, f"Vector subtraction gradient failed: {results}"
    print("  ✓ Vector subtraction")

    print("✓ All subtraction gradient tests passed")


def test_gradient_negation():
    """Test gradient for negation."""
    print("Testing gradient for negation...")

    a = nt.Tensor([1.0, -2.0, 3.0], requires_grad=True)

    def neg_func(x):
        return -x

    pass_check, results = gradient_check(neg_func, [a])
    assert pass_check, f"Negation gradient failed: {results}"
    print("  ✓ Negation")

    print("✓ All negation gradient tests passed")


def test_gradient_chained_operations():
    """Test gradient for chained operations."""
    print("Testing gradient for chained operations...")

    # Test: f(x, y) = (x + y) * (x - y)
    x = nt.Tensor(3.0, requires_grad=True)
    y = nt.Tensor(2.0, requires_grad=True)

    def chained_func(a, b):
        return (a + b) * (a - b)

    pass_check, results = gradient_check(chained_func, [x, y])
    assert pass_check, f"Chained operations gradient failed: {results}"
    print("  ✓ (x+y)*(x-y)")

    # Test: f(x) = exp(log(x^2 + 1))
    x = nt.Tensor([1.0, 2.0, 3.0], requires_grad=True)

    def complex_func(a):
        return (a**2 + 1).log().exp()

    pass_check, results = gradient_check(complex_func, [x])
    assert pass_check, f"Complex chained operations gradient failed: {results}"
    print("  ✓ exp(log(x^2+1))")

    print("✓ All chained operations gradient tests passed")


def test_no_grad_context():
    """Test that no_grad context manager disables gradient tracking."""
    print("Testing no_grad context manager...")

    x = nt.Tensor(3.0, requires_grad=True)
    y = nt.Tensor(4.0, requires_grad=True)

    with nt.no_grad():
        z = x + y
        assert (
            not z.requires_grad
        ), "Tensor created in no_grad context should not require grad"

    # Outside context, gradient tracking should work
    w = x * y
    assert w.requires_grad, "Tensor created outside no_grad should require grad"

    print("✓ no_grad context manager works correctly")


if __name__ == "__main__":
    # Run all gradient tests
    test_gradient_addition()
    test_gradient_subtraction()
    test_gradient_negation()
    test_gradient_multiplication()
    test_gradient_division()
    test_gradient_power()
    test_gradient_matmul()
    test_gradient_activations()
    test_gradient_shape_operations()
    test_gradient_chained_operations()
    test_no_grad_context()

    print("\n" + "=" * 60)
    print("✓ All autograd tests passed!")
    print("=" * 60)
