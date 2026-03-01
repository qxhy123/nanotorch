# Chapter 5: Loss Functions

## The Compass of Learning...

Imagine hiking through a dense fog.

You know where you want to go—the summit. But you can't see it. All you have is a sense of whether you're getting closer or drifting away.

This is the role of the loss function in deep learning.

**The loss function tells the model how wrong it is.** Not just "wrong" or "right"—but *how* wrong, and in *what direction* the answer lies. It converts the gap between prediction and truth into a single number, a guiding star that the optimizer follows downhill.

```
The Journey of Learning:

  Model makes prediction    → "This is a cat" (90% confidence)
  Reality answers           → "Actually, it's a dog"
  Loss function measures    → "You were very wrong. Loss = 2.3"

  Gradient points the way   → "Adjust these weights to reduce error"
  Optimizer takes a step    → Weights shift, ever so slightly

  Next time, perhaps        → "This might be a dog?" (60% confidence)
  Loss decreases            → 2.3 → 0.8 → 0.3...

Like a hiker following a compass,
the model follows the gradient of loss,
always moving toward lower ground,
always seeking the valley where predictions match reality.
```

**Without a loss function, there is no destination.** The model would make predictions, but never know if they were good or bad. It would wander aimlessly, never improving.

Different tasks need different loss functions. Regression needs MSE. Classification needs Cross-Entropy. Each is a different way of measuring "wrongness," each suited to a different kind of problem.

In this chapter, we'll implement these loss functions and understand the mathematics behind them—why they work, when to use which, and how gradients flow through them.

---

## 5.1 The Role of Loss Functions

```
Prediction y_pred → Loss Function → Scalar L
                       ↑
                   True y_true
```

Training objective: **Minimize L**

## 5.2 MSE (Mean Squared Error)

```python
class MSE(Module):
    """Mean Squared Error: L = mean((y_pred - y_true)²)"""
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        diff = y_pred - y_true
        return (diff * diff).mean()
```

**Derivative**:
```
∂L/∂y_pred = 2 * (y_pred - y_true) / n
```

**Use Case**: Regression problems

## 5.3 CrossEntropyLoss

The most commonly used classification loss:

```python
class CrossEntropyLoss(Module):
    """Cross Entropy Loss (with built-in Softmax)
    
    L = -sum(y_true * log(softmax(y_pred)))
    
    For one-hot labels: L = -log(softmax(y_pred)[correct_class])
    """
    
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        # logits: (batch, num_classes)
        # target: (batch,) - class indices
        
        # Numerically stable softmax
        shifted = logits - logits.max(dim=1, keepdims=True)
        exp_x = shifted.exp()
        softmax = exp_x / exp_x.sum(dim=1, keepdims=True)
        
        # Extract correct class probabilities
        batch_size = logits.shape[0]
        correct_probs = softmax[np.arange(batch_size), target.data.astype(int)]
        
        # Negative log likelihood
        loss = -correct_probs.log().mean()
        
        return loss
```

**Mathematical Derivation**:

Let $p = \text{softmax}(\text{logits})$, $y = \text{target}$ (one-hot)

$$L = -\sum_i y_i \cdot \log(p_i)$$

$$\frac{\partial L}{\partial \text{logits}_i} = p_i - y_i$$

**Simplified Form**: Gradient = `softmax(logits) - one_hot(target)`

## 5.4 Complete CrossEntropyLoss Implementation

```python
class CrossEntropyLoss(Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        # Numerically stable log_softmax
        shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
        log_softmax = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        
        # Extract correct class log probabilities
        batch_size = logits.shape[0]
        target_indices = target.data.astype(np.int64)
        correct_log_probs = log_softmax[np.arange(batch_size), target_indices]
        
        # Compute loss
        if self.reduction == 'mean':
            loss = -np.mean(correct_log_probs)
        elif self.reduction == 'sum':
            loss = -np.sum(correct_log_probs)
        else:
            loss = -correct_log_probs
        
        # Create output tensor and set up backpropagation
        out = Tensor(loss, _children=(logits,), _op='cross_entropy')
        out.requires_grad = logits.requires_grad
        
        def _backward():
            if logits.requires_grad:
                # Gradient = softmax - one_hot
                softmax = np.exp(log_softmax)
                grad = softmax.copy()
                grad[np.arange(batch_size), target_indices] -= 1
                grad /= batch_size  # Because of mean reduction
                
                if logits.grad is None:
                    logits.grad = grad * out.grad
                else:
                    logits.grad += grad * out.grad
        
        out._backward = _backward
        return out
```

## 5.5 BCELoss (Binary Cross Entropy)

```python
class BCELoss(Module):
    """Binary Cross Entropy: L = -[y*log(p) + (1-y)*log(1-p)]"""
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        eps = 1e-7
        y_pred = y_pred.clip(eps, 1 - eps)
        
        loss = -(y_true * y_pred.log() + (1 - y_true) * (1 - y_pred).log())
        return loss.mean()
```

## 5.6 Usage Example

```python
from nanotorch import Tensor
from nanotorch.nn import Sequential, Linear, ReLU
from nanotorch.nn.loss import CrossEntropyLoss

# Model
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

# Loss function
criterion = CrossEntropyLoss()

# Simulated data
X = Tensor.randn((32, 784))
y = Tensor(np.random.randint(0, 10, 32))

# Forward pass
logits = model(X)
loss = criterion(logits, y)

# Backward pass
loss.backward()

print(f"Loss: {loss.item():.4f}")
```

## 5.7 Loss Function Comparison

| Loss Function | Formula | Use Case |
|---------------|---------|----------|
| MSE | mean((y-ŷ)²) | Regression |
| MAE/L1 | mean(\|y-ŷ\|) | Regression (robust) |
| CrossEntropy | -Σy*log(p) | Multi-class classification |
| BCE | -[y*log(p)+(1-y)*log(1-p)] | Binary classification |
| BCEWithLogits | BCE(sigmoid(x), y) | Binary classification (numerically stable) |

## 5.8 Exercises

1. **Implement L1Loss**: `mean(|y_pred - y_true|)`

2. **Implement SmoothL1Loss** (Huber Loss)

3. **Implement NLLLoss**: Negative log likelihood (without Softmax)

## Next Chapter

In the next chapter, we will implement **optimizers** to update parameters using gradients.

→ [Chapter 6: Optimizers](06-optimizer.md)
