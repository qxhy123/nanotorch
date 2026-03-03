# Chapter 6: Optimizers

## In the Fog, How to Find the Valley...

You're trapped on a misty mountain, visibility less than three meters.

You can't see the full picture, don't know which direction the valley is. You can only sense one thing: **the slope beneath your feet**.

The slope tells you: "This way is the fastest upward." Then the opposite direction is the fastest way down.

So you take a step. Sense the slope again. Take another step. Step by step... until the mist clears, and you find yourself at the foot of the mountain.

```
The Philosophy of Gradient Descent:

  I don't know what the mountain looks like
  But I know the slope beneath my feet
  Walk in the opposite direction of the slope
  Someday, I'll reach the valley
```

**The optimizer is that "descent strategy."** It tells you how to walk each step—how big the steps should be, whether to remember previous directions, what to do when encountering flat ground.

Some optimizers are careful,步步为营 (SGD). Some optimizers are clever, adaptively adjusting step size (Adam). But their core philosophy is the same:

**Follow the gradient, walk downhill.**

---

## 6.1 Gradient Descent Basics

### Core Formula

```
θ_new = θ_old - lr × ∂L/∂θ

Parameter   Parameter   Learning rate   Gradient
   ↓           ↓            ↓            ↓
 Where to go  Now      How big step   Which way up
```

### Life Analogy

```
Going downhill = Gradient descent

1. Look at slope beneath feet (gradient)
2. Take a step in the opposite direction of slope
3. Repeat until flat ground (loss minimum)

Learning rate = Step size
  Too small → Walk too slowly, takes forever
  Too large → Might step over the valley, or even walk back up
```

```
θ_new = θ_old - lr * ∂L/∂θ
```

- `θ`: Model parameters
- `lr`: Learning rate
- `∂L/∂θ`: Gradient

## 6.2 Optimizer Base Class

```python
# optimizer.py
from typing import List, Dict, Any
from nanotorch.tensor import Tensor

class Optimizer:
    """Base class for all optimizers"""
    
    def __init__(self, params: List[Tensor], lr: float = 0.01):
        self.params = list(params)
        self.lr = lr
        self.param_groups: List[Dict[str, Any]] = []
        
        # Default parameter group
        self.param_groups.append({
            'params': self.params,
            'lr': lr
        })
    
    def zero_grad(self) -> None:
        """Zero out gradients for all parameters"""
        for param in self.params:
            if param.grad is not None:
                param.grad = None
    
    def step(self) -> None:
        """Perform one parameter update step"""
        raise NotImplementedError("Subclasses must implement step()")
```

## 6.3 SGD Implementation

```python
class SGD(Optimizer):
    """Stochastic Gradient Descent
    
    Supports momentum and Nesterov acceleration
    
    v = momentum * v + gradient
    θ = θ - lr * v
    """
    
    def __init__(
        self, 
        params: List[Tensor], 
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
        # Momentum cache
        self.velocities = [np.zeros_like(p.data) for p in self.params]
    
    def step(self) -> None:
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # Weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Momentum update
            if self.momentum != 0:
                v = self.velocities[i]
                v = self.momentum * v + grad
                
                if self.nesterov:
                    grad = grad + self.momentum * v
                else:
                    grad = v
                
                self.velocities[i] = v
            
            # Parameter update
            param.data = param.data - self.lr * grad
```

### The Role of Momentum

```
Without momentum: Update direction may change significantly each step
With momentum: Accumulates historical gradients, smoothing update direction
```

## 6.4 Adam Implementation

```python
class Adam(Optimizer):
    """Adam: Adaptive Moment Estimation
    
    m = β1 * m + (1 - β1) * grad     # First moment estimate
    v = β2 * v + (1 - β2) * grad²    # Second moment estimate
    
    m_hat = m / (1 - β1^t)           # Bias correction
    v_hat = v / (1 - β2^t)
    
    θ = θ - lr * m_hat / (√v_hat + ε)
    """
    
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Moment estimate caches
        self.m = [np.zeros_like(p.data) for p in self.params]  # First moment
        self.v = [np.zeros_like(p.data) for p in self.params]  # Second moment
        self.t = 0  # Time step
    
    def step(self) -> None:
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # Weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Update first moment
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update second moment
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Parameter update
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

### Why is Adam Good?

1. **Adaptive Learning Rate**: Each parameter has a different learning rate
2. **Momentum**: Accelerates convergence
3. **Bias Correction**: Addresses initialization bias

## 6.5 Usage Example

```python
from nanotorch import Tensor
from nanotorch.nn import Sequential, Linear, ReLU
from nanotorch.nn.loss import CrossEntropyLoss
from nanotorch.optim import SGD, Adam

# Model
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

# Optimizer
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

# Training loop
for epoch in range(100):
    # Forward pass
    X, y = get_batch()
    logits = model(X)
    loss = criterion(logits, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Parameter update
    optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## 6.6 Learning Rate Scheduler

```python
class StepLR:
    """Every step_size epochs, multiply learning rate by gamma"""
    
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0
    
    def step(self) -> None:
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma


# Usage
optimizer = SGD(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    train(...)
    scheduler.step()  # Update learning rate
```

## 6.7 Optimizer Comparison

| Optimizer | Characteristics | Use Case |
|-----------|-----------------|----------|
| SGD | Simple, requires tuning | Convex optimization |
| SGD+Momentum | Accelerates convergence | Deep networks |
| Adam | Adaptive learning rate | General default choice |
| AdamW | Decoupled weight decay | Transformers |
| RMSprop | Adaptive learning rate | RNNs |

## 6.8 Exercises

1. **Implement AdamW**: Adam + decoupled weight decay

2. **Implement RMSprop**: `v = α*v + (1-α)*grad²`

3. **Implement CosineAnnealingLR** scheduler

## Next Chapter

In the next chapter, we will integrate all components and implement a complete **training loop**.

→ [Chapter 7: Training Loop](07-training.md)
