"""
Tests for optimizers.
"""

import numpy as np
import nanotorch as nt
from nanotorch.nn import Linear
from nanotorch.optim import SGD, Adam, AdamW, RMSprop, Adagrad
from nanotorch.utils import manual_seed


def test_sgd_creation():
    """Test SGD optimizer initialization."""
    # Create a simple model
    model = Linear(3, 2)
    params = list(model.parameters())

    # Default SGD
    optimizer = SGD(params, lr=0.01)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[0]["momentum"] == 0
    assert optimizer.param_groups[0]["weight_decay"] == 0
    assert optimizer.param_groups[0]["nesterov"] == False

    # SGD with momentum and weight decay
    optimizer2 = SGD(params, lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    assert optimizer2.param_groups[0]["lr"] == 0.1
    assert optimizer2.param_groups[0]["momentum"] == 0.9
    assert optimizer2.param_groups[0]["weight_decay"] == 1e-4
    assert optimizer2.param_groups[0]["nesterov"] == True

    # Invalid learning rate should raise ValueError
    try:
        SGD(params, lr=-0.1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("✓ test_sgd_creation passed")


def test_sgd_step():
    """Test SGD optimization step."""
    manual_seed(42)

    # Create a simple linear layer
    linear = Linear(2, 1)
    initial_weight = linear.weight.data.copy()
    initial_bias = linear.bias.data.copy() if linear.bias is not None else None

    # Create optimizer
    optimizer = SGD(linear.parameters(), lr=0.1)

    # Create dummy gradient (simulate loss backward)
    # For testing, we manually set gradients
    linear.weight.grad = nt.Tensor.ones_like(linear.weight)
    if linear.bias is not None:
        linear.bias.grad = nt.Tensor.ones_like(linear.bias)

    # Take optimization step
    optimizer.step()

    # Check that parameters have been updated (decreased by lr * grad)
    # Since grad is ones, weight should become weight - 0.1 * 1
    expected_weight = initial_weight - 0.1
    assert np.allclose(linear.weight.data, expected_weight)

    if linear.bias is not None:
        expected_bias = initial_bias - 0.1
        assert np.allclose(linear.bias.data, expected_bias)

    # Test zero_grad
    optimizer.zero_grad()
    assert np.allclose(linear.weight.grad.data, 0.0)
    if linear.bias is not None:
        assert np.allclose(linear.bias.grad.data, 0.0)

    print("✓ test_sgd_step passed")


def test_sgd_momentum():
    """Test SGD with momentum."""
    manual_seed(123)

    # Single parameter tensor
    param = nt.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    initial_data = param.data.copy()

    optimizer = SGD([param], lr=0.1, momentum=0.9)

    # First step
    param.grad = nt.Tensor([0.5, -0.5, 1.0])
    optimizer.step()

    # velocity = 0.9 * 0 + (1-0) * grad = grad
    # param = param - lr * velocity = [1.0, 2.0, 3.0] - 0.1 * [0.5, -0.5, 1.0]
    expected = initial_data - 0.1 * np.array([0.5, -0.5, 1.0])
    assert np.allclose(param.data, expected)

    # Second step with same gradient
    optimizer.zero_grad()
    param.grad = nt.Tensor([0.5, -0.5, 1.0])
    optimizer.step()

    # velocity = 0.9 * previous_velocity + grad = 0.9 * [0.5, -0.5, 1.0] + [0.5, -0.5, 1.0]
    # = [0.95, -0.95, 1.9]
    # param = previous_param - 0.1 * velocity
    previous_param = expected
    velocity = 0.9 * np.array([0.5, -0.5, 1.0]) + np.array([0.5, -0.5, 1.0])
    expected2 = previous_param - 0.1 * velocity
    assert np.allclose(param.data, expected2)

    print("✓ test_sgd_momentum passed")


def test_sgd_weight_decay():
    """Test SGD with weight decay."""
    manual_seed(456)

    param = nt.Tensor([2.0], requires_grad=True)
    optimizer = SGD([param], lr=0.1, weight_decay=0.5)

    # Set gradient
    param.grad = nt.Tensor([1.0])

    # Weight decay adds weight_decay * param to gradient
    # So effective grad = 1.0 + 0.5 * 2.0 = 2.0
    # param = 2.0 - 0.1 * 2.0 = 1.8
    optimizer.step()

    assert np.allclose(param.data, 1.8)

    print("✓ test_sgd_weight_decay passed")


def test_adam_creation():
    """Test Adam optimizer initialization."""
    model = Linear(5, 3)
    params = list(model.parameters())

    # Default Adam
    optimizer = Adam(params)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 0.001
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)
    assert optimizer.param_groups[0]["eps"] == 1e-8
    assert optimizer.param_groups[0]["weight_decay"] == 0
    assert optimizer.param_groups[0]["amsgrad"] == False

    # Custom Adam
    optimizer2 = Adam(
        params, lr=0.01, betas=(0.8, 0.888), eps=1e-7, weight_decay=1e-3, amsgrad=True
    )
    assert optimizer2.param_groups[0]["lr"] == 0.01
    assert optimizer2.param_groups[0]["betas"] == (0.8, 0.888)
    assert optimizer2.param_groups[0]["eps"] == 1e-7
    assert optimizer2.param_groups[0]["weight_decay"] == 1e-3
    assert optimizer2.param_groups[0]["amsgrad"] == True

    # Invalid beta should raise ValueError
    try:
        Adam(params, betas=(1.1, 0.999))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("✓ test_adam_creation passed")


def test_adam_step():
    """Test Adam optimization step."""
    manual_seed(789)

    param = nt.Tensor([2.0, -1.0], requires_grad=True)
    initial_data = param.data.copy()

    optimizer = Adam([param], lr=0.001)

    # Set gradient
    param.grad = nt.Tensor([0.5, -0.2])

    # Take a step
    optimizer.step()

    # Check that parameter changed (not equal to initial)
    assert not np.allclose(param.data, initial_data)

    # Check that step counter increased
    param_state = optimizer.state[param]
    assert param_state["step"] == 1

    # Take another step
    optimizer.zero_grad()
    param.grad = nt.Tensor([0.3, 0.1])
    optimizer.step()

    assert param_state["step"] == 2

    print("✓ test_adam_step passed")


def test_adam_weight_decay():
    """Test Adam with weight decay."""
    manual_seed(999)

    param = nt.Tensor([3.0], requires_grad=True)
    optimizer = Adam([param], lr=0.001, weight_decay=0.1)

    param.grad = nt.Tensor([1.0])
    optimizer.step()

    # Weight decay should affect gradient
    # We won't compute exact Adam update, just ensure parameter changed
    assert param.data != 3.0

    print("✓ test_adam_weight_decay passed")


def test_adamw_creation():
    """Test AdamW optimizer initialization."""
    model = Linear(5, 3)
    params = list(model.parameters())

    # Default AdamW (weight_decay=1e-2 by default, not 0!)
    optimizer = AdamW(params)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 0.001
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)
    assert optimizer.param_groups[0]["eps"] == 1e-8
    assert optimizer.param_groups[0]["weight_decay"] == 1e-2  # Different from Adam!
    assert optimizer.param_groups[0]["amsgrad"] == False

    # Custom AdamW
    optimizer2 = AdamW(
        params, lr=0.01, betas=(0.8, 0.888), eps=1e-7, weight_decay=1e-3, amsgrad=True
    )
    assert optimizer2.param_groups[0]["lr"] == 0.01
    assert optimizer2.param_groups[0]["betas"] == (0.8, 0.888)
    assert optimizer2.param_groups[0]["eps"] == 1e-7
    assert optimizer2.param_groups[0]["weight_decay"] == 1e-3
    assert optimizer2.param_groups[0]["amsgrad"] == True

    # Invalid beta should raise ValueError
    try:
        AdamW(params, betas=(1.1, 0.999))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("✓ test_adamw_creation passed")


def test_adamw_step():
    """Test AdamW optimization step."""
    manual_seed(789)

    param = nt.Tensor([2.0, -1.0], requires_grad=True)
    initial_data = param.data.copy()

    optimizer = AdamW([param], lr=0.001)

    # Set gradient
    param.grad = nt.Tensor([0.5, -0.2])

    # Take a step
    optimizer.step()

    # Check that parameter changed (not equal to initial)
    assert not np.allclose(param.data, initial_data)

    # Check that step counter increased
    param_state = optimizer.state[param]
    assert param_state["step"] == 1

    # Take another step
    optimizer.zero_grad()
    param.grad = nt.Tensor([0.3, 0.1])
    optimizer.step()

    assert param_state["step"] == 2

    print("✓ test_adamw_step passed")


def test_adamw_weight_decay():
    """Test AdamW with decoupled weight decay."""
    manual_seed(999)

    param = nt.Tensor([3.0], requires_grad=True)
    optimizer = AdamW([param], lr=0.001, weight_decay=0.1)

    param.grad = nt.Tensor([1.0])
    optimizer.step()

    # Weight decay should affect parameter directly (decoupled)
    # We won't compute exact AdamW update, just ensure parameter changed
    assert param.data != 3.0

    print("✓ test_adamw_weight_decay passed")


def test_adamw_vs_adam_weight_decay():
    """Test that AdamW weight decay is decoupled (different from Adam)."""
    manual_seed(1234)
    
    # Same parameters and gradients for both optimizers
    param_adam = nt.Tensor([2.0], requires_grad=True)
    param_adamw = nt.Tensor([2.0], requires_grad=True)
    
    # Same initial gradient
    grad = nt.Tensor([0.5])
    
    # Create optimizers with same hyperparameters
    adam_opt = Adam([param_adam], lr=0.01, weight_decay=0.1)
    adamw_opt = AdamW([param_adamw], lr=0.01, weight_decay=0.1)
    
    # Set same gradients
    param_adam.grad = grad
    param_adamw.grad = grad
    
    # Take a step with each
    adam_opt.step()
    adamw_opt.step()
    
    # They should produce DIFFERENT results because weight decay is applied differently
    # (coupled vs decoupled)
    assert not np.allclose(param_adam.data, param_adamw.data), \
        "Adam and AdamW should produce different results with weight decay"
    
    print("✓ test_adamw_vs_adam_weight_decay passed")


def test_rmsprop_creation():
    """Test RMSprop optimizer initialization."""
    # Create a simple model
    model = Linear(3, 2)
    params = list(model.parameters())

    # Default RMSprop
    optimizer = RMSprop(params, lr=0.01)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[0]["alpha"] == 0.99
    assert optimizer.param_groups[0]["eps"] == 1e-8
    assert optimizer.param_groups[0]["weight_decay"] == 0
    assert optimizer.param_groups[0]["momentum"] == 0
    assert optimizer.param_groups[0]["centered"] == False

    # RMSprop with custom parameters
    optimizer2 = RMSprop(params, lr=0.1, alpha=0.9, eps=1e-6, weight_decay=1e-4, momentum=0.9, centered=True)
    assert optimizer2.param_groups[0]["lr"] == 0.1
    assert optimizer2.param_groups[0]["alpha"] == 0.9
    assert optimizer2.param_groups[0]["eps"] == 1e-6
    assert optimizer2.param_groups[0]["weight_decay"] == 1e-4
    assert optimizer2.param_groups[0]["momentum"] == 0.9
    assert optimizer2.param_groups[0]["centered"] == True

    # Invalid learning rate should raise ValueError
    try:
        RMSprop(params, lr=-0.1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("✓ test_rmsprop_creation passed")


def test_rmsprop_step():
    """Test RMSprop optimization step."""
    manual_seed(42)

    # Single parameter tensor
    param = nt.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    initial_data = param.data.copy()

    optimizer = RMSprop([param], lr=0.1)

    # Set gradient
    param.grad = nt.Tensor([0.5, -0.5, 1.0])
    optimizer.step()

    # Check that parameter updated (direction should be opposite gradient)
    # RMSprop scales gradient by running average; with first step, square_avg = (1-alpha)*grad^2
    # We'll just ensure parameter changed
    assert not np.allclose(param.data, initial_data)

    # Test zero_grad
    optimizer.zero_grad()
    assert np.allclose(param.grad.data, 0.0)

    print("✓ test_rmsprop_step passed")


def test_adagrad_creation():
    """Test Adagrad optimizer initialization."""
    model = Linear(3, 2)
    params = list(model.parameters())

    optimizer = Adagrad(params, lr=0.01)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[0]["lr_decay"] == 0
    assert optimizer.param_groups[0]["eps"] == 1e-10
    assert optimizer.param_groups[0]["weight_decay"] == 0

    optimizer2 = Adagrad(params, lr=0.1, lr_decay=1e-5, eps=1e-8, weight_decay=1e-4)
    assert optimizer2.param_groups[0]["lr"] == 0.1
    assert optimizer2.param_groups[0]["lr_decay"] == 1e-5
    assert optimizer2.param_groups[0]["eps"] == 1e-8
    assert optimizer2.param_groups[0]["weight_decay"] == 1e-4

    print("✓ test_adagrad_creation passed")


def test_adagrad_step():
    """Test Adagrad optimization step."""
    manual_seed(42)

    param = nt.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    initial_data = param.data.copy()

    optimizer = Adagrad([param], lr=0.1)

    param.grad = nt.Tensor([0.5, -0.5, 1.0])
    optimizer.step()

    # Parameter should change
    assert not np.allclose(param.data, initial_data)

    optimizer.zero_grad()
    assert np.allclose(param.grad.data, 0.0)

    print("✓ test_adagrad_step passed")


def run_all_tests():
    """Run all optimizer tests."""
    test_sgd_creation()
    test_sgd_step()
    test_sgd_momentum()
    test_sgd_weight_decay()
    test_adam_creation()
    test_adam_step()
    test_adam_weight_decay()
    test_adamw_creation()
    test_adamw_step()
    test_adamw_weight_decay()
    test_adamw_vs_adam_weight_decay()
    print("\n✅ All optimizer tests passed!")


if __name__ == "__main__":
    run_all_tests()
