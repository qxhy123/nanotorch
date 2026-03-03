# Chapter 4: Activation Functions

## If the World Had Only Straight Lines...

Imagine a world composed only of straight lines.

Mountains? A section of slope. Waves? A few broken lines. Faces? Countless small triangles pieced together.

This isn't impossible, but too clumsy. You need countless straight lines to barely approximate a curve.

Neural networks face the same awkward situation. Without activation functions, no matter how many layers you stack, it's essentially still drawing straight lines.

```
y = W2(W1x + b1) + b2
  = W2W1x + W2b1 + b2
  = W'x + b'           ← Still a straight line

A hundred layers = one layer, what a tragedy
```

Activation functions are the **magic that makes straight lines curve**. Between each layer, they insert non-linearity, giving the network the ability to fit any shape.

```
ReLU: Cut off negative numbers, keep only positives
Sigmoid: Compress everything between 0 and 1
Tanh: Undulate between -1 and 1

Straight lines → Broken lines → Curves → Everything
```

**Activation functions are the soul of neural networks.** Without them, a network is just a pile of linear transformations, no matter how deep, it cannot learn the complex world.

---

## 4.1 Why Do We Need Activation Functions?

### Problem: No Activation Function = Can Only Draw Straight Lines

```
Two-layer linear network:
y = W2 @ (W1 @ x + b1) + b2
  = (W2 @ W1) @ x + (W2 @ b1 + b2)
  = W @ x + b                    ← Still linear!

No matter how many layers you stack, ultimately equivalent to one layer!
```

### Solution: Activation Functions Introduce "Curvature"

```
With activation function:
y = W2 @ relu(W1 @ x + b1) + b2

relu() breaks the linear relationship!
Now the network can fit arbitrarily complex curves.
```

A multi-layer network without activation functions is equivalent to a single layer:

```
y = W2 @ (W1 @ x) = (W2 @ W1) @ x = W @ x
```

Activation functions break this linear relationship, allowing the network to approximate any function.

## 4.2 Implementing Activation Function Modules

```python
# activation.py
from nanotorch.nn.module import Module
from nanotorch.tensor import Tensor
import numpy as np

class ReLU(Module):
    """ReLU: f(x) = max(0, x)"""
    
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
    
    def extra_repr(self) -> str:
        return ""


class Sigmoid(Module):
    """Sigmoid: f(x) = 1 / (1 + exp(-x))"""
    
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    """Tanh: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
    
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Softmax(Module):
    """Softmax: f(x_i) = exp(x_i) / sum(exp(x_j))"""
    
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(dim=self.dim)
```

## 4.3 ReLU in Detail

```
ReLU(x) = max(0, x)

When x > 0: output x, gradient 1
When x ≤ 0: output 0, gradient 0
```

**Advantages**:
- Simple computation
- Mitigates vanishing gradients
- Sparse activation

**Disadvantages**:
- Dying ReLU problem (gradient always 0)

```python
# Add to Tensor class
def relu(self):
    out = Tensor(
        np.maximum(0, self.data),
        _children=(self,),
        _op='relu'
    )
    
    def _backward():
        if self.requires_grad:
            self.grad = (self.grad if self.grad is not None else 0) + \
                        (self.data > 0).astype(np.float32) * out.grad
    
    out._backward = _backward
    out.requires_grad = self.requires_grad
    return out
```

## 4.4 LeakyReLU

Solves the dying ReLU problem:

```python
class LeakyReLU(Module):
    """LeakyReLU: f(x) = x if x > 0 else α*x"""
    
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: Tensor) -> Tensor:
        return x.leaky_relu(self.negative_slope)


# Tensor method
def leaky_relu(self, negative_slope=0.01):
    out = Tensor(
        np.where(self.data > 0, self.data, self.data * negative_slope),
        _children=(self,),
        _op='leaky_relu'
    )
    
    def _backward():
        if self.requires_grad:
            grad = np.where(self.data > 0, 1, negative_slope)
            self.grad = (self.grad if self.grad is not None else 0) + grad * out.grad
    
    out._backward = _backward
    return out
```

## 4.5 GELU

Commonly used in Transformers:

```python
class GELU(Module):
    """Gaussian Error Linear Unit"""
    
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


# GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
def gelu(self):
    # Approximate implementation
    cdf = 0.5 * (1 + np.tanh(
        np.sqrt(2 / np.pi) * (self.data + 0.044715 * self.data ** 3)
    ))
    out = Tensor(self.data * cdf, _children=(self,), _op='gelu')
    
    def _backward():
        if self.requires_grad:
            x = self.data
            grad = cdf + x * 0.5 * (1 - np.tanh(...) ** 2) * ...  # Full derivative
            self.grad = (self.grad if self.grad is not None else 0) + grad * out.grad
    
    out._backward = _backward
    return out
```

## 4.6 Softmax in Detail

```python
def softmax(self, dim=-1):
    # Numerical stability: subtract max value
    shifted = self.data - np.max(self.data, axis=dim, keepdims=True)
    exp_x = np.exp(shifted)
    out = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
    
    result = Tensor(out, _children=(self,), _op='softmax')
    
    def _backward():
        if self.requires_grad:
            # Softmax Jacobian: ∂softmax_i/∂x_j = softmax_i * (δ_ij - softmax_j)
            # Simplified: grad_input = softmax * (grad_output - sum(grad_output * softmax))
            sum_grad = np.sum(result.grad * out, axis=dim, keepdims=True)
            grad_input = out * (result.grad - sum_grad)
            
            if self.grad is None:
                self.grad = grad_input
            else:
                self.grad += grad_input
    
    result._backward = _backward
    result.requires_grad = self.requires_grad
    return result
```

### Softmax Derivative Derivation

Let $s = \text{softmax}(x)$, $L$ be the loss

$$\frac{\partial L}{\partial x_i} = \sum_j \left(\frac{\partial L}{\partial s_j} \cdot \frac{\partial s_j}{\partial x_i}\right)$$

$$\frac{\partial s_j}{\partial x_i} = s_j \cdot (\delta_{ij} - s_i)$$

When $i = j$: $\frac{\partial s_i}{\partial x_i} = s_i \cdot (1 - s_i)$

When $i \neq j$: $\frac{\partial s_j}{\partial x_i} = -s_i \cdot s_j$

## 4.7 Usage Example

```python
from nanotorch import Tensor
from nanotorch.nn import Sequential, Linear, ReLU, Softmax

model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10),
    Softmax(dim=-1)
)

x = Tensor.randn((32, 784))
probs = model(x)

print(f"Output shape: {probs.shape}")  # (32, 10)
print(f"Sum: {probs.sum(dim=1).data}")  # [1, 1, 1, ...]
```

## 4.8 Activation Function Comparison

| Function | Formula | Range | Gradient Range | Common Use |
|----------|---------|-------|----------------|------------|
| ReLU | max(0, x) | $[0, \infty)$ | {0, 1} | Hidden layers |
| LeakyReLU | max(αx, x) | $(-\infty, \infty)$ | {α, 1} | Hidden layers |
| Sigmoid | $1/(1+e^{-x})$ | (0, 1) | (0, 0.25) | Binary classification output |
| Tanh | $(e^x-e^{-x})/(e^x+e^{-x})$ | (-1, 1) | (0, 1) | RNN |
| Softmax | $e^{x_i}/\sum e^{x_j}$ | (0, 1) | Complex | Multi-class output |
| GELU | $x \cdot \Phi(x)$ | $(-\infty, \infty)$ | Complex | Transformer |

## 4.9 Exercises

1. **Implement ELU**: `f(x) = x if x > 0 else α(e^x - 1)`

2. **Implement Swish/SiLU**: `f(x) = x * sigmoid(x)`

3. **Implement PReLU**: Learnable `α` parameter

4. **Plot activation function curves**

## Next Chapter

In the next chapter, we will implement **loss functions** to define training objectives.

→ [Chapter 5: Loss Functions](05-loss.md)
