# Tutorial 15: Advanced Topics

## Table of Contents

1. [Overview](#overview)
2. [Gradient Clipping](#gradient-clipping)
3. [Learning Rate Warmup](#learning-rate-warmup)
4. [Model Serialization](#model-serialization)
5. [Gradient Checking](#gradient-checking)
6. [Training Tips](#training-tips)
7. [Debugging Tips](#debugging-tips)
8. [Summary](#summary)

---

## Overview

This tutorial covers advanced features and practical tips in nanotorch:

- **Gradient Clipping**: Prevent gradient explosion
- **Learning Rate Warmup**: Stabilize early training
- **Model Serialization**: Save and load models
- **Gradient Checking**: Verify autodiff implementation
- **Training/Debugging Tips**: Best practices

---

## Gradient Clipping

### Why Gradient Clipping is Needed

During training, gradients can become very large (gradient explosion), causing:
- Parameter updates that are too large, destroying learned features
- Loss becoming NaN/Inf
- Unstable training

### Implementation

```python
# nanotorch/utils.py

from typing import Iterable
import numpy as np
from nanotorch.tensor import Tensor

def clip_grad_norm_(
    parameters: Iterable[Tensor],
    max_norm: float,
    norm_type: float = 2.0
) -> float:
    """Clip gradients by norm.
    
    If gradient norm exceeds max_norm, scales gradients proportionally.
    
    Args:
        parameters: Parameter iterator
        max_norm: Maximum norm
        norm_type: Norm type (1, 2, or inf)
    
    Returns:
        Total gradient norm before clipping
    """
    grads = []
    for param in parameters:
        if param.grad is not None:
            grads.append(param.grad.flatten())
    
    if len(grads) == 0:
        return 0.0
    
    total_grad = np.concatenate(grads)
    
    if norm_type == float('inf'):
        total_norm = np.abs(total_grad).max()
    else:
        total_norm = np.linalg.norm(total_grad, ord=norm_type)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        for param in parameters:
            if param.grad is not None:
                param.grad = param.grad * clip_coef
    
    return float(total_norm)


def clip_grad_value_(
    parameters: Iterable[Tensor],
    clip_value: float
) -> None:
    """Clip gradients by value."""
    for param in parameters:
        if param.grad is not None:
            param.grad = np.clip(param.grad, -clip_value, clip_value)


def get_grad_norm_(
    parameters: Iterable[Tensor],
    norm_type: float = 2.0
) -> float:
    """Get gradient norm (without clipping)."""
    grads = []
    for param in parameters:
        if param.grad is not None:
            grads.append(param.grad.flatten())
    
    if len(grads) == 0:
        return 0.0
    
    total_grad = np.concatenate(grads)
    
    if norm_type == float('inf'):
        return float(np.abs(total_grad).max())
    else:
        return float(np.linalg.norm(total_grad, ord=norm_type))
```

### Usage Example

```python
from nanotorch.utils import clip_grad_norm_, get_grad_norm_

# Training loop
for x, y in dataloader:
    optimizer.zero_grad()
    
    output = model(Tensor(x))
    loss = criterion(output, Tensor(y))
    loss.backward()
    
    # Gradient clipping
    grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"Gradient norm: {grad_norm:.4f}")
    
    optimizer.step()
```

---

## Learning Rate Warmup

### Why Warmup is Needed

In early training:
- Model parameters are randomly initialized
- Large learning rates may cause unstable training
- Warmup gradually increases learning rate, stabilizing early training

### Warmup Schedulers in nanotorch

```python
# nanotorch/optim/lr_scheduler.py

class LinearWarmup:
    """Linear warmup scheduler."""
    
    def __init__(self, optimizer, warmup_epochs: int, start_lr: float = 0.0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.target_lr = optimizer.lr
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            alpha = self.current_epoch / self.warmup_epochs
            lr = self.start_lr + alpha * (self.target_lr - self.start_lr)
            self.optimizer.lr = lr
        self.current_epoch += 1


class CosineWarmupScheduler:
    """Cosine warmup + cosine decay."""
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self.current_epoch = 0
    
    def step(self):
        import math
        if self.current_epoch < self.warmup_epochs:
            alpha = self.current_epoch / self.warmup_epochs
            lr = self.min_lr + alpha * (self.base_lr - self.min_lr)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        self.optimizer.lr = lr
        self.current_epoch += 1
```

### Usage Example

```python
from nanotorch.optim import AdamW, CosineWarmupScheduler

optimizer = AdamW(model.parameters(), lr=1e-3)
scheduler = CosineWarmupScheduler(
    optimizer,
    warmup_epochs=5,
    max_epochs=100,
    min_lr=1e-6
)

for epoch in range(100):
    train_one_epoch(model, dataloader, optimizer)
    scheduler.step()
    print(f"Epoch {epoch}, LR: {optimizer.lr:.6f}")
```

---

## Model Serialization

### Save and Load Models

```python
# Save model
state_dict = model.state_dict()
np.savez('model.npz', **state_dict)

# Load model
state_dict = dict(np.load('model.npz'))
model.load_state_dict(state_dict)
```

### Save Training Checkpoint

```python
def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save complete training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }
    np.savez(path, **checkpoint)

def load_checkpoint(model, path):
    """Load training checkpoint."""
    checkpoint = dict(np.load(path, allow_pickle=True))
    model.load_state_dict(checkpoint['model_state_dict'].item())
    return checkpoint['epoch'], checkpoint['loss']
```

---

## Gradient Checking

### Principle

Use finite differences to verify autodiff correctness:

```
Numerical gradient ≈ (f(x + eps) - f(x - eps)) / (2 * eps)
```

### Implementation

```python
def gradient_check(
    func: callable,
    inputs: list,
    eps: float = 1e-5,
    atol: float = 1e-4,
    rtol: float = 1e-3
) -> bool:
    """Verify gradients using finite differences."""
    # Compute analytical gradients
    loss = func(inputs)
    loss.backward()
    
    analytic_grads = [inp.grad.copy() if inp.grad is not None else None for inp in inputs]
    
    # Compute numerical gradients
    for i, inp in enumerate(inputs):
        if analytic_grads[i] is None:
            continue
        
        numerical_grad = np.zeros_like(inp.data)
        it = np.nditer(inp.data, flags=['multi_index'], op_flags=['readwrite'])
        
        while not it.finished:
            idx = it.multi_index
            original = inp.data[idx]
            
            # f(x + eps)
            inp.data[idx] = original + eps
            loss_plus = func([Tensor(inp.data) if j == i else inputs[j] for j in range(len(inputs))])
            
            # f(x - eps)
            inp.data[idx] = original - eps
            loss_minus = func([Tensor(inp.data) if j == i else inputs[j] for j in range(len(inputs))])
            
            # Numerical gradient
            numerical_grad[idx] = (loss_plus.item() - loss_minus.item()) / (2 * eps)
            
            # Restore original value
            inp.data[idx] = original
            it.iternext()
        
        # Compare gradients
        diff = np.abs(analytic_grads[i] - numerical_grad)
        max_diff = diff.max()
        max_grad = max(np.abs(analytic_grads[i]).max(), np.abs(numerical_grad).max())
        
        if max_diff > atol and max_diff / (max_grad + 1e-8) > rtol:
            print(f"Gradient check failed for input {i}!")
            return False
    
    print("Gradient check passed!")
    return True
```

### Usage Example

```python
from nanotorch import Tensor
from nanotorch.nn import Linear, MSELoss
from nanotorch.utils import gradient_check

linear = Linear(10, 5)

def compute_loss(inputs):
    x, y = inputs
    output = linear(x)
    loss = MSELoss()(output, y)
    return loss

x = Tensor.randn((4, 10), requires_grad=True)
y = Tensor.randn((4, 5))

gradient_check(compute_loss, [x, y])
```

---

## Training Tips

### Early Stopping

```python
class EarlyStopping:
    """Early stopping mechanism."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# Usage
early_stop = EarlyStopping(patience=15)

for epoch in range(epochs):
    train_loss = train(model, dataloader, optimizer)
    val_loss = validate(model, val_loader)
    
    if early_stop(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

### Exponential Moving Average (EMA)

```python
class EMA:
    """Exponential Moving Average of parameters."""
    
    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.copy()
    
    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name] = (
                self.decay * self.shadow[name] + (1 - self.decay) * param.data
            )
    
    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            param.data = self.shadow[name].copy()

# Usage
ema = EMA(model, decay=0.999)

for epoch in range(epochs):
    train(...)
    ema.update(model)

# Use EMA parameters for inference
ema.apply_shadow(model)
evaluate(model)
```

---

## Debugging Tips

### Check Gradient Flow

```python
def check_gradients(model):
    """Check if gradients are normal."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = np.linalg.norm(param.grad)
            has_nan = np.isnan(param.grad).any()
            has_inf = np.isinf(param.grad).any()
            
            print(f"{name}: grad_norm={grad_norm:.6f}, nan={has_nan}, inf={has_inf}")
```

### Check Model Output

```python
def check_output(output, name="output"):
    """Check if output is normal."""
    data = output.data if hasattr(output, 'data') else output
    
    print(f"{name}:")
    print(f"  shape: {data.shape}")
    print(f"  mean: {data.mean():.6f}")
    print(f"  std: {data.std():.6f}")
    print(f"  min: {data.min():.6f}")
    print(f"  max: {data.max():.6f}")
    print(f"  has_nan: {np.isnan(data).any()}")
    print(f"  has_inf: {np.isinf(data).any()}")
```

### Visualize Computation Graph

```python
def print_computation_graph(tensor, indent=0):
    """Print computation graph."""
    prefix = "  " * indent
    print(f"{prefix}Tensor(shape={tensor.shape}, requires_grad={tensor.requires_grad})")
    
    if hasattr(tensor, '_op') and tensor._op is not None:
        print(f"{prefix}  op: {tensor._op}")
    
    if hasattr(tensor, '_parents'):
        for parent in tensor._parents:
            print_computation_graph(parent, indent + 2)
```

---

## Summary

This tutorial introduced advanced features and practical tips in nanotorch:

| Feature | Function |
|---------|----------|
| **clip_grad_norm_** | Clip gradients by norm |
| **clip_grad_value_** | Clip gradients by value |
| **LinearWarmup** | Linear warmup |
| **CosineWarmupScheduler** | Cosine warmup + decay |
| **gradient_check** | Verify gradient correctness |
| **EarlyStopping** | Early stopping mechanism |
| **EMA** | Exponential moving average |

### Training Best Practices

1. **Gradient Clipping**: Essential for RNN and Transformer training
2. **Learning Rate Warmup**: Key for large model training stability
3. **Early Stopping**: Prevent overfitting
4. **EMA**: Improve model generalization
5. **Gradient Checking**: Verify correctness when implementing new operations

### Debugging Tips

1. Check gradient norms and NaN/Inf
2. Monitor activation distributions
3. Overfit small dataset first
4. Gradually increase model complexity

---

**Congratulations!** You've completed the nanotorch tutorial series!

Through these tutorials, you've learned:
- Core principles of Tensor and automatic differentiation
- How to implement various neural network layers
- Data loading and augmentation
- Optimizers and schedulers
- Training and debugging techniques

Now you can:
1. Read nanotorch source code for deeper understanding
2. Try implementing more features (distributed training, mixed precision)
3. Apply this knowledge to actual PyTorch projects

**References**:
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)
