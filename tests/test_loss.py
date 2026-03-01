"""
Tests for loss functions in nanotorch.nn.loss.
"""

import numpy as np
import pytest
import nanotorch as nt
from nanotorch.nn.loss import MSE, CrossEntropyLoss, mse_loss, cross_entropy_loss


class TestMSE:
    """Test Mean Squared Error loss."""
    
    def test_forward_mean_reduction(self):
        """Test MSE forward pass with mean reduction (default)."""
        loss_fn = MSE(reduction="mean")
        pred = nt.Tensor([[1.0, 2.0], [3.0, 4.0]])
        target = nt.Tensor([[0.0, 1.0], [2.0, 3.0]])
        loss = loss_fn(pred, target)
        expected = ((1.0**2 + 1.0**2 + 1.0**2 + 1.0**2) / 4.0)
        assert np.allclose(loss.item(), expected)
    
    def test_forward_sum_reduction(self):
        """Test MSE forward pass with sum reduction."""
        loss_fn = MSE(reduction="sum")
        pred = nt.Tensor([[1.0, 2.0], [3.0, 4.0]])
        target = nt.Tensor([[0.0, 1.0], [2.0, 3.0]])
        loss = loss_fn(pred, target)
        expected = 1.0**2 + 1.0**2 + 1.0**2 + 1.0**2
        assert np.allclose(loss.item(), expected)
    
    def test_forward_none_reduction(self):
        """Test MSE forward pass with no reduction."""
        loss_fn = MSE(reduction="none")
        pred = nt.Tensor([[1.0, 2.0], [3.0, 4.0]])
        target = nt.Tensor([[0.0, 1.0], [2.0, 3.0]])
        loss = loss_fn(pred, target)
        assert loss.shape == pred.shape
        expected = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert np.allclose(loss.data, expected)
    
    def test_shape_mismatch_raises(self):
        """Test that MSE raises error on shape mismatch."""
        loss_fn = MSE()
        pred = nt.Tensor.randn((2, 3))
        target = nt.Tensor.randn((2, 4))
        with pytest.raises(RuntimeError):
            loss_fn(pred, target)
    
    def test_gradient_mean_reduction(self):
        """Test gradient correctness for MSE with mean reduction."""
        np.random.seed(42)
        pred = nt.Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        target = nt.Tensor(np.random.randn(3, 4).astype(np.float32))
        loss_fn = MSE(reduction="mean")
        
        # Forward
        loss = loss_fn(pred, target)
        # Backward
        loss.backward()
        
        # Compute gradient numerically
        epsilon = 1e-4
        numerical_grad = np.zeros_like(pred.data)
        it = np.nditer(pred.data, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            original = pred.data[idx]
            
            pred.data[idx] = original + epsilon
            loss_plus = loss_fn(pred, target).item()
            
            pred.data[idx] = original - epsilon
            loss_minus = loss_fn(pred, target).item()
            
            pred.data[idx] = original
            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            it.iternext()
        
        # Compare gradients
        analytical_grad = pred.grad.data
        assert np.allclose(analytical_grad, numerical_grad, rtol=2e-2, atol=1e-3)
    
    def test_gradient_sum_reduction(self):
        """Test gradient correctness for MSE with sum reduction."""
        np.random.seed(42)
        pred = nt.Tensor(np.random.randn(2, 3).astype(np.float32), requires_grad=True)
        target = nt.Tensor(np.random.randn(2, 3).astype(np.float32))
        loss_fn = MSE(reduction="sum")
        
        loss = loss_fn(pred, target)
        loss.backward()
        
        # Numerical gradient
        epsilon = 1e-4
        numerical_grad = np.zeros_like(pred.data)
        it = np.nditer(pred.data, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            original = pred.data[idx]
            
            pred.data[idx] = original + epsilon
            loss_plus = loss_fn(pred, target).item()
            
            pred.data[idx] = original - epsilon
            loss_minus = loss_fn(pred, target).item()
            
            pred.data[idx] = original
            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            it.iternext()
        
        analytical_grad = pred.grad.data
        assert np.allclose(analytical_grad, numerical_grad, rtol=2e-2, atol=1e-3)
    
    def test_gradient_none_reduction(self):
        """Test gradient correctness for MSE with no reduction."""
        np.random.seed(42)
        pred = nt.Tensor(np.random.randn(2, 2).astype(np.float32), requires_grad=True)
        target = nt.Tensor(np.random.randn(2, 2).astype(np.float32))
        loss_fn = MSE(reduction="none")
        
        loss = loss_fn(pred, target)
        # loss is a tensor of shape (2,2). Need to sum to get scalar for backward
        loss_sum = loss.sum()
        loss_sum.backward()
        
        # Numerical gradient of sum of loss elements
        epsilon = 1e-4
        numerical_grad = np.zeros_like(pred.data)
        it = np.nditer(pred.data, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            original = pred.data[idx]
            
            pred.data[idx] = original + epsilon
            loss_plus = loss_fn(pred, target).sum().item()
            
            pred.data[idx] = original - epsilon
            loss_minus = loss_fn(pred, target).sum().item()
            
            pred.data[idx] = original
            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            it.iternext()
        
        analytical_grad = pred.grad.data
        assert np.allclose(analytical_grad, numerical_grad, rtol=2e-2, atol=1e-3)
    
    def test_functional_mse_loss(self):
        """Test functional MSE loss interface."""
        pred = nt.Tensor([[1.0, 2.0], [3.0, 4.0]])
        target = nt.Tensor([[0.0, 1.0], [2.0, 3.0]])
        loss = mse_loss(pred, target, reduction="mean")
        expected = ((1.0**2 + 1.0**2 + 1.0**2 + 1.0**2) / 4.0)
        assert np.allclose(loss.item(), expected)


class TestCrossEntropyLoss:
    """Test Cross-Entropy loss."""
    
    def test_forward_class_indices(self):
        """Test CE loss with class indices target."""
        loss_fn = CrossEntropyLoss(reduction="mean")
        logits = nt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        target = nt.Tensor([0, 2])  # class indices
        loss = loss_fn(logits, target)
        # Compute expected loss manually
        # softmax of first row: [0.09003057, 0.24472847, 0.66524096]
        # -log(0.09003057) = 2.407605
        # softmax of second row: [0.09003057, 0.24472847, 0.66524096] (same scale)
        # -log(0.66524096) = 0.407605
        # mean = (2.407605 + 0.407605) / 2 = 1.407605
        expected = 1.407605
        assert np.allclose(loss.item(), expected, rtol=1e-5)
    
    def test_forward_class_probabilities(self):
        """Test CE loss with class probabilities target (soft labels)."""
        loss_fn = CrossEntropyLoss(reduction="mean")
        logits = nt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        target = nt.Tensor([[0.1, 0.3, 0.6], [0.4, 0.4, 0.2]])
        loss = loss_fn(logits, target)
        # Not computing exact expected value; just ensure it runs
        assert loss.item() > 0
    
    def test_shape_validation(self):
        """Test that CE loss validates shapes."""
        loss_fn = CrossEntropyLoss()
        # Batch size mismatch
        logits = nt.Tensor.randn((3, 5))
        target = nt.Tensor([0, 1])  # only 2 samples
        with pytest.raises(RuntimeError):
            loss_fn(logits, target)
        # Invalid target dimension
        target_3d = nt.Tensor.randn((3, 5, 2))
        with pytest.raises(RuntimeError):
            loss_fn(logits, target_3d)
    
    def test_gradient_class_indices(self):
        """Test gradient correctness for CE loss with class indices."""
        np.random.seed(42)
        logits = nt.Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        target = nt.Tensor(np.random.randint(0, 4, size=(3,)).astype(np.int32))
        loss_fn = CrossEntropyLoss(reduction="mean")
        
        loss = loss_fn(logits, target)
        loss.backward()
        
        # Numerical gradient
        epsilon = 1e-4
        numerical_grad = np.zeros_like(logits.data)
        it = np.nditer(logits.data, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            original = logits.data[idx]
            
            logits.data[idx] = original + epsilon
            loss_plus = loss_fn(logits, target).item()
            
            logits.data[idx] = original - epsilon
            loss_minus = loss_fn(logits, target).item()
            
            logits.data[idx] = original
            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            it.iternext()
        
        analytical_grad = logits.grad.data
        # Use relaxed tolerance due to numerical instability in softmax
        assert np.allclose(analytical_grad, numerical_grad, rtol=2e-2, atol=1e-3)
    
    def test_gradient_class_probabilities(self):
        """Test gradient correctness for CE loss with class probabilities."""
        np.random.seed(42)
        logits = nt.Tensor(np.random.randn(2, 3).astype(np.float32), requires_grad=True)
        target = nt.Tensor(np.random.rand(2, 3).astype(np.float32))
        # Normalize rows to sum to 1
        target.data = target.data / target.data.sum(axis=1, keepdims=True)
        loss_fn = CrossEntropyLoss(reduction="mean")
        
        loss = loss_fn(logits, target)
        loss.backward()
        
        epsilon = 1e-4
        numerical_grad = np.zeros_like(logits.data)
        it = np.nditer(logits.data, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            original = logits.data[idx]
            
            logits.data[idx] = original + epsilon
            loss_plus = loss_fn(logits, target).item()
            
            logits.data[idx] = original - epsilon
            loss_minus = loss_fn(logits, target).item()
            
            logits.data[idx] = original
            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            it.iternext()
        
        analytical_grad = logits.grad.data
        assert np.allclose(analytical_grad, numerical_grad, rtol=2e-2, atol=1e-3)
    
    def test_functional_cross_entropy_loss(self):
        """Test functional cross-entropy loss interface."""
        logits = nt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        target = nt.Tensor([0, 2])
        loss = cross_entropy_loss(logits, target, reduction="mean")
        expected = 1.407605
        assert np.allclose(loss.item(), expected, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])