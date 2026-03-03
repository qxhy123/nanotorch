# Chapter 2: Automatic Differentiation (Autograd)

## Have You Ever Been Lost in the Mountains?

You set off in the morning, the sun shining brightly. In the afternoon, you find yourself surrounded by mountains, unfamiliar scenery all around.

It's getting dark, and you must get down quickly. The question is—**which direction leads to the foot of the mountain fastest?**

You look around and notice the slope beneath your feet seems to point northeast. So you take a step. Look around again, adjust direction. Step by step... until you see the lights of the valley below.

This is **gradient descent**: at each step, you move in the steepest downward direction.

```
Mountain height = Loss function
Your position   = Model parameters
Downhill direction = Opposite of gradient
Mountain foot   = Optimal solution
```

But there's a problem: in the "ten-thousand-dimensional mountain" of neural networks, manually calculating gradients is nearly impossible. A million parameters means a million directions.

**Automatic differentiation** is your compass. It remembers how you arrived at each step, then tells you precisely: for each direction, what is the slope.

---

## 2.1 Why Do We Need Automatic Differentiation?

### Problem: Training Neural Networks Requires Computing Gradients

The neural network training process:

```
1. Forward pass:  Input → Network → Prediction
2. Compute loss:  Prediction vs Ground truth → Loss
3. Backward pass: Loss → Gradients (tell each parameter how to adjust)
4. Update parameters: Parameters - Learning rate × Gradient
```

**Step 3 requires computing gradients**, the problem is: the network may have millions of parameters, manual differentiation is impossible!

### Comparison of Three Differentiation Methods

| Method | Principle | Pros | Cons |
|--------|-----------|------|------|
| **Manual differentiation** | Human calculates formula | Exact | Too slow, error-prone |
| **Numerical differentiation** | `(f(x+h)-f(x-h))/2h` | Simple | Inexact, slow |
| **Automatic differentiation** | Computational graph + Chain rule | Exact, fast | Complex to implement |

### Numerical Differentiation Example

```python
def numerical_grad(f, x, eps=1e-5):
    """Numerical differentiation: approximate derivative with small difference"""
    return (f(x + eps) - f(x - eps)) / (2 * eps)

# Test: f(x) = x², derivative at x=3
f = lambda x: x ** 2
print(numerical_grad(f, 3))  # ≈ 6.0 (exact value is 6)

# Problem: if function is complex, need to compute f() many times
```

### Automatic Differentiation: What We'll Implement

```python
x = Tensor([3.0], requires_grad=True)
y = x ** 2           # y = x²
y.backward()         # Automatically compute gradient
print(x.grad)        # [6.] ← Exact answer!
```

**One sentence summary**: Automatic differentiation = Computer computes derivatives for you, you only need to write forward computation.

---

## 2.2 Computational Graph: The Map of Automatic Differentiation

### What is a Computational Graph?

**Computational graph = Drawing the computation process as a graph**

Consider the function `f(x) = x² + 2x + 1`, we need to find `df/dx`:

**Manual differentiation**: `df/dx = 2x + 2`

**Numerical differentiation** (imprecise):
```python
def numerical_grad(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps)) / (2 * eps)
```

**Automatic differentiation** (exact, general):
```python
x = Tensor([3.0], requires_grad=True)
y = x**2 + 2*x + 1
y.backward()
print(x.grad)  # [8.] = 2*3 + 2
```

## 2.2 Computational Graph

Automatic differentiation is based on **computational graphs**:

```
x = 3
  │
  ├──→ (x²) ──┐
  │           │
  └──→ (2x) ──┼──→ (+) ──→ (+1) ──→ y = 16
              │
              └─────────────────────→ (+)
```

Forward propagation: from input to output
Backward propagation: from output to input, passing gradients

## 2.3 Chain Rule

The core of backpropagation is the **chain rule**:

$$
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}
$$

Example: `y = (x + 1)²`
- `g = x + 1`, `y = g²`
- `dy/dg = 2g`
- `dg/dx = 1`
- `dy/dx = dy/dg * dg/dx = 2g * 1 = 2(x+1)`

## 2.4 Simplest Implementation

Let's add gradient tracking to Tensor:

```python
# tensor_v3.py
class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        
        # For building computational graph
        self._prev = set(_children)  # Parent nodes
        self._op = _op               # Operation that created this tensor
        self._backward = lambda: None  # Backward propagation function
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op='+')
        
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad  # d(a+b)/da = 1
            
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += out.grad  # d(a+b)/db = 1
        
        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op='*')
        
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += other.data * out.grad  # d(a*b)/da = b
            
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += self.data * out.grad  # d(a*b)/db = a
        
        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out
```

### Gradient Formula Derivation

**Addition**: `y = a + b`
- `dy/da = 1`
- `dy/db = 1`
- So `a.grad += out.grad * 1`

**Multiplication**: `y = a * b`
- `dy/da = b`
- `dy/db = a`
- So `a.grad += out.grad * b`

## 2.5 Backward Propagation

Now implement the `backward()` method:

```python
def backward(self):
    # Build topological sort
    topo = []
    visited = set()
    
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    
    build_topo(self)
    
    # Initialize output gradient to 1
    self.grad = np.ones_like(self.data)
    
    # Backward propagation (from end to beginning)
    for node in reversed(topo):
        node._backward()


# Test
x = Tensor([2.0], requires_grad=True)
y = x * x + 2 * x + 1  # x² + 2x + 1
y.backward()
print(x.grad)  # [6.] = 2*2 + 2 = 6
```

### Topological Sort

Why do we need topological sorting?

```
Computational graph:
  a ──→ c
  │     │
  └──→ b ──→ d
```

Backward propagation must go **from back to front**:
1. d → c, b
2. c → a
3. b → a

Topological sort ensures: parent nodes are processed after child nodes.

## 2.6 Backward Propagation for More Operations

```python
def __neg__(self):
    return self * (-1)

def __sub__(self, other):
    return self + (-other)

def __pow__(self, n):
    out = Tensor(self.data ** n, _children=(self,), _op=f'**{n}')
    
    def _backward():
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            # d(x^n)/dx = n * x^(n-1)
            self.grad += n * (self.data ** (n-1)) * out.grad
    
    out._backward = _backward
    out.requires_grad = self.requires_grad
    return out

def relu(self):
    out = Tensor(np.maximum(0, self.data), _children=(self,), _op='relu')
    
    def _backward():
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            # d(relu(x))/dx = 1 if x > 0 else 0
            self.grad += (self.data > 0).astype(np.float32) * out.grad
    
    out._backward = _backward
    out.requires_grad = self.requires_grad
    return out
```

### Common Derivative Formulas

| Function f(x) | Derivative f'(x) |
|---------------|------------------|
| `x + c` | `1` |
| `x * c` | `c` |
| `x²` | `2x` |
| `x^n` | `nx^(n-1)` |
| `e^x` | `e^x` |
| `ln(x)` | `1/x` |
| `sin(x)` | `cos(x)` |
| `cos(x)` | `-sin(x)` |
| `ReLU(x)` | `1 if x>0 else 0` |
| `Sigmoid(x)` | `σ(x)(1-σ(x))` |
| `Tanh(x)` | `1-tanh²(x)` |

## 2.7 Matrix Operation Gradients

Matrix multiplication gradients are the most complex:

```python
def matmul(self, other):
    out = Tensor(
        np.matmul(self.data, other.data),
        _children=(self, other),
        _op='@'
    )
    
    def _backward():
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            # d(A @ B) / dA = grad @ B.T
            self.grad += np.matmul(out.grad, other.data.T)
        
        if other.requires_grad:
            if other.grad is None:
                other.grad = np.zeros_like(other.data)
            # d(A @ B) / dB = A.T @ grad
            other.grad += np.matmul(self.data.T, out.grad)
    
    out._backward = _backward
    out.requires_grad = self.requires_grad or other.requires_grad
    return out
```

### Matrix Multiplication Gradient Derivation

Let `Y = A @ B`, where `A: (M, K)`, `B: (K, N)`, `Y: (M, N)`

For `A[i,j]`:
$$
\frac{\partial L}{\partial A[i,j]} = \sum_k \left( \frac{\partial L}{\partial Y[i,k]} \cdot \frac{\partial Y[i,k]}{\partial A[i,j]} \right) = \sum_k \text{grad}[i,k] \cdot B[j,k] = (\text{grad} @ B^T)[i,j]
$$

So `A.grad = grad @ B.T`

Similarly `B.grad = A.T @ grad`

## 2.8 Broadcasting Gradient Handling

When operations involve broadcasting, gradients need to be summed:

```python
def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data, _children=(self, other), _op='+')
    
    def _backward():
        if self.requires_grad:
            grad = out.grad
            # Handle broadcasting: if self was broadcast, need to sum
            # e.g., (3,) + (2,3) -> self's gradient needs sum along axis=0
            self.grad = self.grad + grad if self.grad is not None else grad
        
        if other.requires_grad:
            grad = out.grad
            # Similarly handle broadcasting
            other.grad = other.grad + grad if other.grad is not None else grad
    
    out._backward = _backward
    return out
```

### Broadcasting Gradient Example

```python
# A: (2, 3), b: (3,)
# C = A + b, b is broadcast to (2, 3)
# C.grad: (2, 3)
# b.grad: needs sum along axis=0, becomes (3,)
```

## 2.9 Complete Implementation

```python
# autograd.py
class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
    
    def backward(self, gradient=None):
        if gradient is None:
            gradient = np.ones_like(self.data)
        
        self.grad = gradient
        
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self):
        self.grad = None
```

## 2.10 Verifying Gradient Correctness

Verify using numerical gradients:

```python
def numerical_gradient(f, x, eps=1e-5):
    """Compute numerical gradient"""
    grad = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=['multi_index'])
    
    while not it.finished:
        idx = it.multi_index
        old_val = x.data[idx]
        
        x.data[idx] = old_val + eps
        fxh = f().data
        
        x.data[idx] = old_val - eps
        fxl = f().data
        
        x.data[idx] = old_val
        grad[idx] = (fxh - fxl) / (2 * eps)
        
        it.iternext()
    
    return grad


def check_gradient():
    x = Tensor([2.0, 3.0, 4.0], requires_grad=True)
    
    def f():
        return (x * x).sum()
    
    # Analytical gradient
    y = f()
    y.backward()
    analytical = x.grad.copy()
    
    # Numerical gradient
    x.grad = None
    numerical = numerical_gradient(f, x)
    
    # Compare
    diff = np.abs(analytical - numerical).max()
    print(f"Analytical: {analytical}")
    print(f"Numerical: {numerical}")
    print(f"Max diff: {diff}")
    
    assert diff < 1e-5, "Gradient check failed!"
    print("✓ Gradient check passed!")


check_gradient()
```

## 2.11 Exercises

1. **Implement `exp()` backward propagation**

2. **Implement `log()` backward propagation**

3. **Implement `sum(dim)` backward propagation** (Hint: gradient needs to expand back to original shape)

4. **Implement `softmax()` backward propagation** (Jacobian matrix)

5. **Challenge**: Implement `conv2d` backward propagation

## 2.12 Debugging Tips

```python
# Print computational graph
def print_graph(tensor, indent=0):
    print("  " * indent + f"{tensor._op or 'input'} {tensor.shape}")
    for child in tensor._prev:
        print_graph(child, indent + 1)

# Check gradient flow
def check_grad_flow(tensor):
    for node in tensor._prev:
        if node.requires_grad and node.grad is None:
            print(f"⚠ No gradient at {node._op}")
        check_grad_flow(node)
```

## Next Chapter

Now Tensor can automatically compute gradients! In the next chapter, we will implement **Neural Network Modules**, building trainable models.

→ [Chapter 3: Module Base Class](03-nn-module_EN.md)
