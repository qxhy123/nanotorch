"""
Loss functions for neural networks.

This module provides loss functions similar to PyTorch's nn module.
"""

import numpy as np
from numpy.typing import NDArray
import nanotorch.tensor as T
from nanotorch.nn.module import Module


class MSE(Module):
    """Mean Squared Error loss.

    Measures the element-wise mean squared error between input and target.

    Formula: MSE(x, y) = mean((x - y)^2)
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Initialize MSE loss.

        Args:
            reduction: Specifies the reduction to apply to the output:
                'mean': returns the mean loss (default)
                'sum': returns the summed loss
                'none': returns element-wise loss
        """
        super().__init__()
        self.reduction = reduction

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )

    def forward(self, input: T.Tensor, target: T.Tensor) -> T.Tensor:
        """Forward pass of MSE loss.

        Args:
            input: Predicted tensor.
            target: Ground truth tensor.

        Returns:
            Loss tensor.

        Raises:
            RuntimeError: If input and target shapes don't match.
        """
        if input.shape != target.shape:
            raise RuntimeError(
                f"MSE loss requires input and target to have same shape, "
                f"got input shape {input.shape} and target shape {target.shape}"
            )

        # Compute element-wise squared error
        loss = (input - target) ** 2

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

    def extra_repr(self) -> str:
        """Extra representation string for the module."""
        return f"reduction='{self.reduction}'"

    def __repr__(self) -> str:
        """String representation of the MSE loss."""
        return f"MSELoss({self.extra_repr()})"


class CrossEntropyLoss(Module):
    """Cross-entropy loss for classification tasks.

    Combines LogSoftmax and NLLLoss in one class.

    Formula: CE(x, target) = -log(softmax(x)[target])
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Initialize CrossEntropyLoss.

        Args:
            reduction: Specifies the reduction to apply to the output:
                'mean': returns the mean loss (default)
                'sum': returns the summed loss
                'none': returns element-wise loss
        """
        super().__init__()
        self.reduction = reduction

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )

    def forward(self, input: T.Tensor, target: T.Tensor) -> T.Tensor:
        """Forward pass of cross-entropy loss.

        Args:
            input: Predicted logits tensor of shape (batch_size, num_classes).
            target: Ground truth class indices tensor of shape (batch_size,)
                   or class probabilities tensor of shape (batch_size, num_classes).

        Returns:
            Loss tensor.

        Raises:
            RuntimeError: If input and target shapes are incompatible.
        """
        # Handle different target formats
        if target.ndim == 1:
            # Target contains class indices
            return self._forward_class_indices(input, target)
        elif target.ndim == 2:
            # Target contains class probabilities (one-hot or soft labels)
            return self._forward_class_probabilities(input, target)
        else:
            raise RuntimeError(
                f"Target must be 1D (class indices) or 2D (class probabilities), "
                f"got {target.ndim}D tensor"
            )

    def _forward_class_indices(self, input: T.Tensor, target: T.Tensor) -> T.Tensor:
        """Forward pass for class indices target."""
        batch_size, num_classes = input.shape

        if target.shape[0] != batch_size:
            raise RuntimeError(
                f"Batch size mismatch: input has {batch_size} samples, "
                f"target has {target.shape[0]} samples"
            )

        # Convert class indices to one-hot encoding
        # target.data is numpy array of shape (batch_size,)
        target_np: NDArray[np.int_] = target.data.astype(int)
        one_hot: NDArray[np.float32] = np.zeros(
            (batch_size, num_classes), dtype=np.float32
        )
        one_hot[np.arange(batch_size), target_np] = 1.0
        one_hot_tensor = T.Tensor(one_hot, requires_grad=False)

        # Use the class probabilities forward pass (which handles soft labels)
        return self._forward_class_probabilities(input, one_hot_tensor)

    def _forward_class_probabilities(
        self, input: T.Tensor, target: T.Tensor
    ) -> T.Tensor:
        """Forward pass for class probabilities target."""
        if input.shape != target.shape:
            raise RuntimeError(
                f"Shape mismatch for class probabilities: input shape {input.shape}, "
                f"target shape {target.shape}"
            )

        # Cross-entropy with soft labels: -sum(target * log(softmax(input)))
        # Compute log softmax
        x_max = input.max(axis=1, keepdims=True)
        x_exp = (input - x_max).exp()
        x_exp_sum = x_exp.sum(axis=1, keepdims=True)
        log_probs = input - x_max - x_exp_sum.log()

        # Element-wise multiplication: target * log_probs
        weighted_log_probs = target * log_probs

        # Sum over classes and negate
        loss = -weighted_log_probs.sum(axis=1)

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

    def extra_repr(self) -> str:
        """Extra representation string for the module."""
        return f"reduction='{self.reduction}'"

    def __repr__(self) -> str:
        """String representation of the CrossEntropyLoss."""
        return f"CrossEntropyLoss({self.extra_repr()})"


# Functional versions of loss functions
def mse_loss(input: T.Tensor, target: T.Tensor, reduction: str = "mean") -> T.Tensor:
    """Functional interface for MSE loss."""
    return MSE(reduction=reduction)(input, target)


def cross_entropy_loss(
    input: T.Tensor, target: T.Tensor, reduction: str = "mean"
) -> T.Tensor:
    """Functional interface for cross-entropy loss."""
    return CrossEntropyLoss(reduction=reduction)(input, target)


class L1Loss(Module):
    """Mean Absolute Error loss.

    Formula: L1Loss(x, y) = mean(|x - y|)
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )
        self.reduction = reduction

    def forward(self, input: T.Tensor, target: T.Tensor) -> T.Tensor:
        if input.shape != target.shape:
            raise RuntimeError(
                f"L1 loss requires input and target to have same shape, "
                f"got input shape {input.shape} and target shape {target.shape}"
            )

        loss = (input - target).abs()

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"

    def __repr__(self) -> str:
        return f"L1Loss({self.extra_repr()})"


class SmoothL1Loss(Module):
    """Smooth L1 loss (Huber loss).

    Formula:
        loss(x, y) = 0.5 * (x - y)^2 if |x - y| < beta else beta * (|x - y| - 0.5 * beta)
    """

    def __init__(self, reduction: str = "mean", beta: float = 1.0) -> None:
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )
        self.reduction = reduction
        self.beta = beta

    def forward(self, input: T.Tensor, target: T.Tensor) -> T.Tensor:
        if input.shape != target.shape:
            raise RuntimeError(
                f"SmoothL1 loss requires input and target to have same shape, "
                f"got input shape {input.shape} and target shape {target.shape}"
            )

        diff = (input - target).abs()
        beta = self.beta
        
        # Create condition tensor: 1.0 where |x-y| < beta, else 0.0
        cond_data = (diff.data < beta).astype(np.float32)
        cond = T.Tensor(cond_data, requires_grad=False)
        
        # Quadratic loss when |x-y| < beta, linear loss otherwise
        loss = cond * (0.5 * diff * diff / beta) + (T.Tensor(np.ones_like(cond_data)) - cond) * (diff - 0.5 * beta)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}', beta={self.beta}"

    def __repr__(self) -> str:
        return f"SmoothL1Loss({self.extra_repr()})"


class BCELoss(Module):
    """Binary Cross Entropy loss.

    Formula: BCELoss(x, y) = -[y * log(x) + (1 - y) * log(1 - x)]
    
    Note: Input should be probabilities (after sigmoid), use BCEWithLogitsLoss for logits.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )
        self.reduction = reduction

    def forward(self, input: T.Tensor, target: T.Tensor) -> T.Tensor:
        if input.shape != target.shape:
            raise RuntimeError(
                f"BCE loss requires input and target to have same shape, "
                f"got input shape {input.shape} and target shape {target.shape}"
            )

        eps = 1e-7
        input_clipped = input.clamp(eps, 1 - eps)
        loss = -(target * input_clipped.log() + (T.Tensor(np.ones_like(target.data)) - target) * (T.Tensor(np.ones_like(input_clipped.data)) - input_clipped).log())

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"

    def __repr__(self) -> str:
        return f"BCELoss({self.extra_repr()})"


class BCEWithLogitsLoss(Module):
    """Binary Cross Entropy loss with logits.

    Combines sigmoid and BCE loss for numerical stability.
    
    Formula: BCEWithLogitsLoss(x, y) = max(x, 0) - x * y + log(1 + exp(-|x|))
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )
        self.reduction = reduction

    def forward(self, input: T.Tensor, target: T.Tensor) -> T.Tensor:
        if input.shape != target.shape:
            raise RuntimeError(
                f"BCEWithLogitsLoss requires input and target to have same shape, "
                f"got input shape {input.shape} and target shape {target.shape}"
            )

        # log(sigmoid(x)) = -softplus(-x) = -log(1 + exp(-x))
        # log(1 - sigmoid(x)) = -softplus(x) = -log(1 + exp(x))
        # BCE = -[y * log(sigmoid(x)) + (1-y) * log(1 - sigmoid(x))]
        #     = y * softplus(-x) + (1-y) * softplus(x)
        #     = (1-y) * log(1 + exp(x)) + y * log(1 + exp(-x))
        
        # Numerically stable: max(x, 0) - x * z + log(1 + exp(-|x|))
        max_val = input.relu()
        loss = max_val - input * target + (-input.abs()).softplus()

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"

    def __repr__(self) -> str:
        return f"BCEWithLogitsLoss({self.extra_repr()})"


class NLLLoss(Module):
    """Negative Log Likelihood loss.

    Used with LogSoftmax for classification.
    
    Formula: NLLLoss(x, target) = -x[range(batch), target]
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )
        self.reduction = reduction

    def forward(self, input: T.Tensor, target: T.Tensor) -> T.Tensor:
        if target.ndim != 1:
            raise RuntimeError(
                f"NLLLoss expects target to be 1D (class indices), got {target.ndim}D"
            )

        batch_size, num_classes = input.shape
        if target.shape[0] != batch_size:
            raise RuntimeError(
                f"Batch size mismatch: input has {batch_size} samples, "
                f"target has {target.shape[0]} samples"
            )

        # Gather log probabilities at target indices
        target_np = target.data.astype(int)
        
        # Create one-hot encoding for gathering
        one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
        one_hot[np.arange(batch_size), target_np] = 1.0
        
        # Negated log probabilities at target positions
        loss = -(input * T.Tensor(one_hot, requires_grad=False)).sum(axis=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"

    def __repr__(self) -> str:
        return f"NLLLoss({self.extra_repr()})"


# Functional versions of new loss functions
def l1_loss(input: T.Tensor, target: T.Tensor, reduction: str = "mean") -> T.Tensor:
    """Functional interface for L1 loss."""
    return L1Loss(reduction=reduction)(input, target)


def smooth_l1_loss(input: T.Tensor, target: T.Tensor, reduction: str = "mean", beta: float = 1.0) -> T.Tensor:
    """Functional interface for SmoothL1 loss."""
    return SmoothL1Loss(reduction=reduction, beta=beta)(input, target)


def bce_loss(input: T.Tensor, target: T.Tensor, reduction: str = "mean") -> T.Tensor:
    """Functional interface for BCE loss."""
    return BCELoss(reduction=reduction)(input, target)


def bce_with_logits_loss(input: T.Tensor, target: T.Tensor, reduction: str = "mean") -> T.Tensor:
    """Functional interface for BCEWithLogits loss."""
    return BCEWithLogitsLoss(reduction=reduction)(input, target)


def nll_loss(input: T.Tensor, target: T.Tensor, reduction: str = "mean") -> T.Tensor:
    """Functional interface for NLL loss."""
    return NLLLoss(reduction=reduction)(input, target)
