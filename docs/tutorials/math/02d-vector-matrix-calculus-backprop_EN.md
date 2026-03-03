# Chapter 2(d): Vector Matrix Calculus and Backpropagation

Vector matrix calculus is the core mathematical tool for deep learning. Neural network parameters are typically in matrix form, and understanding how to differentiate with respect to matrices is crucial for implementing backpropagation. This chapter systematically introduces matrix derivative formulas and the implementation of the backpropagation algorithm.

---

## 🎯 Life Analogy: Backpropagation is a "Responsibility Chain"

Imagine a **bubble tea shop** with a chain of workers:

```
Order → Cashier → Barista → Cup Sealer → Customer

If the customer complains "too sweet", who's responsible?
- Work backwards from the customer
- Each person: "How much did MY action affect the sweetness?"
- Pass this info to the previous person

This is BACKPROPAGATION!
```

### The Chain of Responsibility

```
Output Layer    Hidden Layer    Input Layer
    ●  ──────────→  ●  ──────────→  ●
    │               │               │
 "I need to     "Pass error     "I adjust
  adjust by       backward"       inputs"
  this much"
       ↑               ↑               ↑
   Gradient       Chain Rule      Gradient
   flows backward through layers  flows to start
```

### 📝 Step-by-Step Backpropagation Example

**Simple network**: $y = w \cdot x$, loss $L = (y - target)^2$

Given: $x = 2$, $w = 3$, $target = 10$

**Forward pass**:
$$y = 3 \times 2 = 6$$
$$L = (6 - 10)^2 = 16$$

**Backward pass** (chain rule):

**Step 1**: $\frac{\partial L}{\partial y} = 2(y - target) = 2(6 - 10) = -8$

**Step 2**: $\frac{\partial y}{\partial w} = x = 2$

**Step 3**: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w} = -8 \times 2 = -16$

**Weight update** (learning rate = 0.01):
$$w_{new} = w - 0.01 \times (-16) = 3 + 0.16 = 3.16$$

### 📖 Plain English Translation

| Math Term | Plain English |
|-----------|---------------|
| Gradient | Direction of steepest increase |
| Backpropagation | Passing error backward through the network |
| Chain rule | Propagating derivatives through composite functions |
| Jacobian | All partial derivatives arranged in a matrix |

---

## Table of Contents

1. [Vector Derivative Basics](#vector-derivative-basics)
2. [Jacobian Matrix](#jacobian-matrix)
3. [Matrix Derivative Formulas](#matrix-derivative-formulas)
4. [Gradient Computation in Deep Learning](#gradient-computation-in-deep-learning)
5. [Backpropagation Algorithm](#backpropagation-algorithm)
6. [Gradient Numerical Verification](#gradient-numerical-verification)
7. [Summary](#summary)

---

## Vector Derivative Basics

### Layout Conventions

There are two common layout conventions for matrix derivatives:

| Layout | Shape of $\frac{\partial y}{\partial \mathbf{x}}$ |
|------|-----------------------------------------------|
| Numerator layout | $1 \times n$ (row vector) |
| Denominator layout | $n \times 1$ (column vector, **we use this**) |

### Scalar to Vector Derivative

Let $f: \mathbb{R}^n \to \mathbb{R}$, $\mathbf{x} \in \mathbb{R}^n$:

$$
\frac{\partial f}{\partial \mathbf{x}} = \nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix} \in \mathbb{R}^n
$$

### Vector to Vector Derivative

Let $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$:

$$
\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \mathbf{J} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix} \in \mathbb{R}^{m \times n}
$$

### Scalar to Matrix Derivative

Let $f: \mathbb{R}^{m \times n} \to \mathbb{R}$, $\mathbf{X} \in \mathbb{R}^{m \times n}$:

$$
\frac{\partial f}{\partial \mathbf{X}} = \begin{bmatrix}
\frac{\partial f}{\partial X_{11}} & \cdots & \frac{\partial f}{\partial X_{1n}} \\
\vdots & \ddots & \vdots \\
\frac{\partial f}{\partial X_{m1}} & \cdots & \frac{\partial f}{\partial X_{mn}}
\end{bmatrix} \in \mathbb{R}^{m \times n}
$$

```python
import numpy as np

def gradient_scalar_to_vector(f, x, h=1e-5):
    """Gradient of scalar with respect to vector"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2*h)
    return grad

def jacobian_vector_to_vector(f, x, h=1e-5):
    """Jacobian matrix of vector with respect to vector"""
    y = f(x)
    m, n = len(y), len(x)
    J = np.zeros((m, n))

    for j in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += h
        x_minus[j] -= h
        J[:, j] = (f(x_plus) - f(x_minus)) / (2*h)

    return J

# Example
# f(x) = ||x||^2 = sum(x_i^2)
f_scalar = lambda x: np.sum(x**2)
x = np.array([1.0, 2.0, 3.0])
grad = gradient_scalar_to_vector(f_scalar, x)
print(f"∇(||x||²) = {grad}")  # Should be [2, 4, 6]

# f(x) = [x_1 + x_2, x_1^2, x_2^3]
f_vector = lambda x: np.array([x[0] + x[1], x[0]**2, x[1]**3])
J = jacobian_vector_to_vector(f_vector, np.array([1.0, 2.0]))
print(f"Jacobian matrix:\n{J}")
# [[1, 1], [2, 0], [0, 12]]
```

---

## Jacobian Matrix

### Definition

The **Jacobian matrix** of vector-valued function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$:

$$
\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}_{m \times n}
$$

### Matrix Form of Chain Rule

For $\mathbf{z} = \mathbf{f}(\mathbf{y})$, $\mathbf{y} = \mathbf{g}(\mathbf{x})$:

$$
\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \mathbf{J}_f \cdot \mathbf{J}_g
$$

### Jacobian Matrices of Common Functions

**Linear transformation**: $\mathbf{f}(\mathbf{x}) = \mathbf{A}\mathbf{x}$

$$
\mathbf{J} = \mathbf{A}
$$

**Element-wise function**: $\mathbf{f}(\mathbf{x}) = [\sigma(x_1), \ldots, \sigma(x_n)]^\top$

$$
\mathbf{J} = \text{diag}(\sigma'(x_1), \ldots, \sigma'(x_n))
$$

**Softmax**: $f_i(\mathbf{x}) = \frac{e^{x_i}}{\sum_j e^{x_j}}$

$$
\frac{\partial f_i}{\partial x_j} = f_i(\delta_{ij} - f_j)
$$

$$
\mathbf{J} = \text{diag}(\mathbf{f}) - \mathbf{f}\mathbf{f}^\top
$$

```python
import numpy as np

def softmax(x):
    """Numerically stable softmax"""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)

def softmax_jacobian(x):
    """Jacobian matrix of Softmax"""
    s = softmax(x)
    return np.diag(s) - np.outer(s, s)

# Example
x = np.array([1.0, 2.0, 3.0])
s = softmax(x)
J = softmax_jacobian(x)

print(f"Softmax(x) = {s}")
print(f"Softmax Jacobian matrix:\n{J}")

# Verify: Sum of each row of Jacobian matrix is 0
print(f"Row sums: {J.sum(axis=1)}")  # Close to [0, 0, 0]
```

---

## Matrix Derivative Formulas

### Basic Formulas

**Linear function**: $f(\mathbf{x}) = \mathbf{a}^\top \mathbf{x}$

$$
\frac{\partial f}{\partial \mathbf{x}} = \mathbf{a}
$$

**Quadratic form**: $f(\mathbf{x}) = \mathbf{x}^\top \mathbf{A} \mathbf{x}$

$$
\frac{\partial f}{\partial \mathbf{x}} = (\mathbf{A} + \mathbf{A}^\top) \mathbf{x}
$$

If $\mathbf{A}$ is symmetric:

$$
\frac{\partial f}{\partial \mathbf{x}} = 2\mathbf{A}\mathbf{x}
$$

### Matrix Derivatives

**Trace of matrix multiplication**: $f(\mathbf{X}) = \text{tr}(\mathbf{A}\mathbf{X})$

$$
\frac{\partial f}{\partial \mathbf{X}} = \mathbf{A}^\top
$$

**Trace of quadratic form**: $f(\mathbf{X}) = \text{tr}(\mathbf{X}^\top \mathbf{A} \mathbf{X})$

$$
\frac{\partial f}{\partial \mathbf{X}} = (\mathbf{A} + \mathbf{A}^\top)\mathbf{X}
$$

**Determinant**: $f(\mathbf{X}) = \det(\mathbf{X})$

$$
\frac{\partial f}{\partial \mathbf{X}} = \det(\mathbf{X}) \cdot (\mathbf{X}^{-1})^\top
$$

**Matrix multiplication**: $f(\mathbf{X}) = \mathbf{a}^\top \mathbf{X} \mathbf{b}$

$$
\frac{\partial f}{\partial \mathbf{X}} = \mathbf{a} \mathbf{b}^\top
$$

### Matrix Derivatives Commonly Used in Deep Learning

**Linear layer**: $\mathbf{Y} = \mathbf{X}\mathbf{W}$, given $\frac{\partial L}{\partial \mathbf{Y}}$

$$
\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^\top \frac{\partial L}{\partial \mathbf{Y}}
$$

$$
\frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Y}} \mathbf{W}^\top
$$

```python
import numpy as np

def verify_matrix_derivatives():
    """Verify matrix derivative formulas"""

    # 1. Verify derivative of tr(AX) with respect to X = A^T
    np.random.seed(42)
    A = np.random.randn(3, 4)
    X = np.random.randn(4, 3)

    def f1(X):
        return np.trace(A @ X)

    # Numerical gradient
    h = 1e-5
    grad_numeric = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_plus = X.copy()
            X_minus = X.copy()
            X_plus[i, j] += h
            X_minus[i, j] -= h
            grad_numeric[i, j] = (f1(X_plus) - f1(X_minus)) / (2*h)

    grad_analytic = A.T
    print(f"Derivative of tr(AX) with respect to X:")
    print(f"  Numerical gradient norm error: {np.linalg.norm(grad_numeric - grad_analytic):.2e}")

    # 2. Verify derivative of a^T X b with respect to X = a b^T
    a = np.random.randn(3)
    b = np.random.randn(4)
    X = np.random.randn(3, 4)

    def f2(X):
        return a @ X @ b

    grad_numeric = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_plus = X.copy()
            X_minus = X.copy()
            X_plus[i, j] += h
            X_minus[i, j] -= h
            grad_numeric[i, j] = (f2(X_plus) - f2(X_minus)) / (2*h)

    grad_analytic = np.outer(a, b)
    print(f"\nDerivative of a^T X b with respect to X:")
    print(f"  Numerical gradient norm error: {np.linalg.norm(grad_numeric - grad_analytic):.2e}")

verify_matrix_derivatives()
```

---

## Gradient Computation in Deep Learning

### Gradients of Linear Layers

**Forward propagation**:

$$
\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{b}
$$

Where:
- $\mathbf{X} \in \mathbb{R}^{B \times D_{in}}$: Input
- $\mathbf{W} \in \mathbb{R}^{D_{in} \times D_{out}}$: Weights
- $\mathbf{b} \in \mathbb{R}^{D_{out}}$: Bias
- $\mathbf{Y} \in \mathbb{R}^{B \times D_{out}}$: Output

**Backpropagation** (given $\frac{\partial L}{\partial \mathbf{Y}}$):

$$
\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^\top \frac{\partial L}{\partial \mathbf{Y}}
$$

$$
\frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Y}} \mathbf{W}^\top
$$

$$
\frac{\partial L}{\partial \mathbf{b}} = \sum_{batch} \frac{\partial L}{\partial \mathbf{Y}}
$$

```python
class LinearLayer:
    """Linear layer implementation"""

    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)

        # Cache for backpropagation
        self.x = None

        # Gradients
        self.dW = None
        self.db = None

    def forward(self, x):
        """Forward propagation: Y = XW + b"""
        self.x = x  # Cache input
        return x @ self.W + self.b

    def backward(self, dL_dY):
        """Backpropagation"""
        # Gradient with respect to W: X^T @ dL/dY
        self.dW = self.x.T @ dL_dY

        # Gradient with respect to b: sum over batch
        self.db = np.sum(dL_dY, axis=0)

        # Gradient with respect to X: dL/dY @ W^T
        dL_dX = dL_dY @ self.W.T

        return dL_dX

# Test
linear = LinearLayer(10, 5)
x = np.random.randn(32, 10)  # batch of 32

# Forward
y = linear.forward(x)
print(f"Output shape: {y.shape}")  # (32, 5)

# Backward
dL_dY = np.random.randn(32, 5)
dL_dX = linear.backward(dL_dY)
print(f"dW shape: {linear.dW.shape}")  # (10, 5)
print(f"db shape: {linear.db.shape}")  # (5,)
print(f"dL/dX shape: {dL_dX.shape}")   # (32, 10)
```

### Gradients of Activation Functions

**Sigmoid**: $\sigma(x) = \frac{1}{1+e^{-x}}$

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_backward(dL_dY, Y):
    """Sigmoid backpropagation"""
    # Y = sigmoid(x), dL/dx = dL/dY * sigmoid'(x) = dL/dY * Y * (1-Y)
    return dL_dY * Y * (1 - Y)
```

**ReLU**: $\text{ReLU}(x) = \max(0, x)$

$$
\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

```python
def relu(x):
    return np.maximum(0, x)

def relu_backward(dL_dY, X):
    """ReLU backpropagation"""
    return dL_dY * (X > 0).astype(float)
```

**Softmax + Cross-entropy combination**

Computing the gradient of Softmax alone is complex, but when combined with cross-entropy it becomes very simple:

$$
\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i
$$

**Derivation of Softmax + Cross-entropy gradient**:

Let the Softmax output be $\hat{y}_i = \frac{e^{z_i}}{\sum_k e^{z_k}}$, and the cross-entropy loss be $L = -\sum_j y_j \log \hat{y}_j$.

**Step 1**: Compute $\frac{\partial \hat{y}_j}{\partial z_i}$ (Jacobian matrix of Softmax).

**When $i = j$**:

$$\frac{\partial \hat{y}_i}{\partial z_i} = \frac{e^{z_i} \sum_k e^{z_k} - e^{z_i} \cdot e^{z_i}}{(\sum_k e^{z_k})^2} = \hat{y}_i(1 - \hat{y}_i)$$

**When $i \neq j$**:

$$\frac{\partial \hat{y}_j}{\partial z_i} = \frac{0 - e^{z_j} \cdot e^{z_i}}{(\sum_k e^{z_k})^2} = -\hat{y}_j \hat{y}_i$$

Combined:

$$\frac{\partial \hat{y}_j}{\partial z_i} = \hat{y}_j(\delta_{ij} - \hat{y}_i)$$

Where $\delta_{ij}$ is the Kronecker delta (1 when $i=j$, otherwise 0).

**Step 2**: Use the chain rule to compute $\frac{\partial L}{\partial z_i}$.

$$\frac{\partial L}{\partial z_i} = \sum_j \frac{\partial L}{\partial \hat{y}_j} \cdot \frac{\partial \hat{y}_j}{\partial z_i}$$

Where:

$$\frac{\partial L}{\partial \hat{y}_j} = -\frac{y_j}{\hat{y}_j}$$

**Step 3**: Substitute and compute.

$$\frac{\partial L}{\partial z_i} = \sum_j \left(-\frac{y_j}{\hat{y}_j}\right) \cdot \hat{y}_j(\delta_{ij} - \hat{y}_i)$$

$$= -\sum_j y_j(\delta_{ij} - \hat{y}_i)$$

$$= -\sum_j y_j \delta_{ij} + \sum_j y_j \hat{y}_i$$

**Step 4**: Simplify.

First term: $\sum_j y_j \delta_{ij} = y_i$ (only contributes non-zero when $j=i$)

Second term: $\sum_j y_j \hat{y}_i = \hat{y}_i \sum_j y_j = \hat{y}_i \cdot 1 = \hat{y}_i$ (since $y$ is one-hot, sum is 1)

Therefore:

$$\frac{\partial L}{\partial z_i} = -y_i + \hat{y}_i = \hat{y}_i - y_i$$

$$\boxed{\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y}}$$

**Intuitive understanding**:
- $\hat{y}_i$ is the predicted probability by the model
- $y_i$ is the true label (one-hot)
- Gradient = prediction error, very concise!

```python
def softmax(x):
    """Numerically stable softmax"""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cross_entropy_loss(predictions, targets):
    """
    Cross-entropy loss
    predictions: softmax output (batch, num_classes)
    targets: one-hot encoding (batch, num_classes)
    """
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.mean(np.sum(targets * np.log(predictions), axis=1))

def softmax_cross_entropy_backward(predictions, targets):
    """
    Combined gradient of Softmax + Cross-entropy

    The gradient after combination is very simple: dL/dz = predictions - targets
    """
    return (predictions - targets) / predictions.shape[0]

# Example
batch_size, num_classes = 4, 10
logits = np.random.randn(batch_size, num_classes)
targets_onehot = np.zeros((batch_size, num_classes))
targets_onehot[np.arange(batch_size), np.random.randint(0, num_classes, batch_size)] = 1

# Forward
probs = softmax(logits)
loss = cross_entropy_loss(probs, targets_onehot)

# Backward
grad = softmax_cross_entropy_backward(probs, targets_onehot)

print(f"Loss: {loss:.4f}")
print(f"Gradient shape: {grad.shape}")
```

### Gradients of Loss Functions

**MSE loss**: $L = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2$

$$
\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)
$$

```python
def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def mse_loss_backward(predictions, targets):
    return 2 * (predictions - targets) / predictions.size
```

**Cross-entropy loss**: $L = -\sum_i y_i \log \hat{y}_i$

$$
\frac{\partial L}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i}
$$

---

## Backpropagation Algorithm

### Computation Graph

Backpropagation is essentially applying the chain rule in reverse along the **computation graph**.

```
Input x
    ↓
  Linear layer1 (W1, b1)
    ↓
  Activation function1
    ↓
  Linear layer2 (W2, b2)
    ↓
  Activation function2
    ↓
  Loss function
```

### Backpropagation Steps

1. **Forward propagation**: Compute and cache intermediate results
2. **Compute loss**: Get scalar loss value
3. **Backpropagation**: Starting from loss, compute gradients layer by layer
4. **Update parameters**: Use gradients to update weights

### Complete Implementation

```python
import numpy as np

class NeuralNetwork:
    """Simple feedforward neural network"""

    def __init__(self, layer_sizes):
        """
        layer_sizes: List of layer sizes, e.g., [784, 256, 128, 10]
        """
        self.weights = []
        self.biases = []

        # Xavier initialization
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * \
                np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

        # Cache for backpropagation
        self.activations = []
        self.z_values = []

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, x):
        """Forward propagation"""
        self.activations = [x]
        self.z_values = []

        current = x
        for i in range(len(self.weights)):
            z = current @ self.weights[i] + self.biases[i]
            self.z_values.append(z)

            # Last layer uses softmax, others use ReLU
            if i == len(self.weights) - 1:
                current = self.softmax(z)
            else:
                current = self.relu(z)

            self.activations.append(current)

        return current

    def compute_loss(self, predictions, targets):
        """Cross-entropy loss"""
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.mean(np.sum(targets * np.log(predictions), axis=1))

    def backward(self, targets):
        """Backpropagation"""
        num_layers = len(self.weights)
        batch_size = targets.shape[0]

        # Initialize gradient storage
        dW = [None] * num_layers
        db = [None] * num_layers

        # Output layer gradient (softmax + cross-entropy combination)
        delta = (self.activations[-1] - targets) / batch_size

        # Propagate gradient from back to front
        for i in range(num_layers - 1, -1, -1):
            # Compute gradients with respect to weights and biases
            dW[i] = self.activations[i].T @ delta
            db[i] = np.sum(delta, axis=0, keepdims=True)

            if i > 0:
                # Propagate to previous layer
                delta = (delta @ self.weights[i].T) * self.relu_derivative(self.z_values[i-1])

        return dW, db

    def update_parameters(self, dW, db, learning_rate):
        """Update parameters"""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dW[i]
            self.biases[i] -= learning_rate * db[i]

    def train_step(self, x, targets, learning_rate):
        """Single training step"""
        # Forward propagation
        predictions = self.forward(x)

        # Compute loss
        loss = self.compute_loss(predictions, targets)

        # Backpropagation
        dW, db = self.backward(targets)

        # Update parameters
        self.update_parameters(dW, db, learning_rate)

        return loss

# Training example
np.random.seed(42)

# Create network
nn = NeuralNetwork([784, 256, 128, 10])

# Simulate data
batch_size = 32
x = np.random.randn(batch_size, 784)
y = np.random.randint(0, 10, batch_size)
targets = np.zeros((batch_size, 10))
targets[np.arange(batch_size), y] = 1

# Training
for epoch in range(10):
    loss = nn.train_step(x, targets, learning_rate=0.01)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
```

---

## Gradient Numerical Verification

### Importance

When implementing backpropagation, it's easy to make mistakes. Numerical gradient verification is a key step to ensure correctness.

### Numerical Gradient

Using central differences:

$$
\frac{\partial f}{\partial x_i} \approx \frac{f(x_i + h) - f(x_i - h)}{2h}
$$

### Verification Function

```python
import numpy as np

def numerical_gradient(f, x, h=1e-5):
    """
    Compute numerical gradient

    Args:
        f: Scalar function
        x: Parameters (can be vector, matrix, or arbitrary shape)
        h: Difference step size

    Returns:
        Gradient with same shape as x
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]

        x[idx] = old_value + h
        fx_plus = f()

        x[idx] = old_value - h
        fx_minus = f()

        grad[idx] = (fx_plus - fx_minus) / (2 * h)
        x[idx] = old_value

        it.iternext()

    return grad

def check_gradient(analytic_grad, numeric_grad, name="gradient", threshold=1e-6):
    """
    Compare analytical gradient and numerical gradient
    """
    diff = np.abs(analytic_grad - numeric_grad)
    rel_error = diff / (np.abs(analytic_grad) + np.abs(numeric_grad) + 1e-8)
    max_error = np.max(rel_error)

    status = "✓ passed" if max_error < threshold else "✗ failed"
    print(f"{name}: max relative error = {max_error:.2e} {status}")

    return max_error < threshold

# Example: Verify neural network gradients
def verify_neural_network_gradients():
    np.random.seed(42)

    # Small network for easier verification
    nn = NeuralNetwork([4, 3, 2])

    # Small batch
    x = np.random.randn(2, 4)
    y = np.array([0, 1])
    targets = np.zeros((2, 2))
    targets[np.arange(2), y] = 1

    # Forward propagation
    predictions = nn.forward(x)

    # Analytical gradients
    dW, db = nn.backward(targets)

    # Numerically verify each weight matrix
    for layer_idx in range(len(nn.weights)):
        # Verify W
        def loss_fn_W():
            preds = nn.forward(x)
            return nn.compute_loss(preds, targets)

        numeric_dW = numerical_gradient(loss_fn_W, nn.weights[layer_idx].copy())
        check_gradient(dW[layer_idx], numeric_dW, f"W{layer_idx+1}")

        # Verify b
        def loss_fn_b():
            preds = nn.forward(x)
            return nn.compute_loss(preds, targets)

        numeric_db = numerical_gradient(loss_fn_b, nn.biases[layer_idx].copy())
        check_gradient(db[layer_idx], numeric_db, f"b{layer_idx+1}")

verify_neural_network_gradients()
```

---

## Applications in Deep Learning

### 1. Automatic Differentiation System

The core of deep learning frameworks is automatic differentiation (autograd), which automatically computes gradients:

```python
import numpy as np

class Tensor:
    """Simplified automatic differentiation tensor"""

    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad else out.grad
            if other.requires_grad:
                other.grad = other.grad + out.grad if other.grad else out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op='*')

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + other.data * out.grad if self.grad else other.data * out.grad
            if other.requires_grad:
                other.grad = other.grad + self.data * out.grad if other.grad else self.data * out.grad
        out._backward = _backward
        return out

    def matmul(self, other):
        """Matrix multiplication"""
        out = Tensor(self.data @ other.data, _children=(self, other), _op='@')

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad @ other.data.T if self.grad else out.grad @ other.data.T
            if other.requires_grad:
                other.grad = other.grad + self.data.T @ out.grad if other.grad else self.data.T @ out.grad
        out._backward = _backward
        return out

    def backward(self):
        """Backpropagation"""
        # Build computation graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Backpropagation
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

# Usage example
x = Tensor([[1, 2], [3, 4]], requires_grad=True)
w = Tensor([[0.5, 0.5], [0.5, 0.5]], requires_grad=True)
y = x.matmul(w)
loss = y.mul(y)  # loss = (X @ W)^2
loss.backward()

print(f"x.grad:\n{x.grad}")
print(f"w.grad:\n{w.grad}")
```

### 2. Gradient Flow in Transformer

```python
def attention_gradients_demo():
    """Demonstrate gradient flow in attention mechanism"""
    np.random.seed(42)

    # Parameters
    batch_size, seq_len, d_k = 2, 4, 8

    # Input
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)

    # Forward propagation
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)

    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Output
    output = attention_weights @ V

    # Assume gradient from subsequent layers
    d_output = np.random.randn(*output.shape)

    # Backpropagation
    # dV
    dV = attention_weights.transpose(0, 2, 1) @ d_output

    # d_attention_weights
    d_attention_weights = d_output @ V.transpose(0, 2, 1)

    # d_scores (Softmax backward)
    # ds = da * s - s * (da * s).sum()
    ds = d_attention_weights * attention_weights - attention_weights * (d_attention_weights * attention_weights).sum(axis=-1, keepdims=True)

    # dQ, dK
    dQ = ds @ K / np.sqrt(d_k)
    dK = ds.transpose(0, 2, 1) @ Q / np.sqrt(d_k)

    print(f"Attention output shape: {output.shape}")
    print(f"dQ shape: {dQ.shape}")
    print(f"dK shape: {dK.shape}")
    print(f"dV shape: {dV.shape}")

    return dQ, dK, dV

dQ, dK, dV = attention_gradients_demo()
```

### 3. Gradient Checkpointing

```python
def gradient_checkpointing_demo():
    """Demonstrate gradient checkpointing technique"""

    class CheckpointedFunction:
        """Simplified gradient checkpointing implementation"""

        def __init__(self, fn, *args):
            self.fn = fn
            self.args = args
            self.saved_tensors = None

        def forward(self):
            # Don't save intermediate activations, only save inputs
            self.saved_tensors = [arg.copy() for arg in self.args]
            return self.fn(*self.args)

        def backward(self, grad_output):
            # Recompute forward pass to get intermediate results
            with np.enable_grad():  # Pseudo-code, actual frameworks track gradients
                output = self.fn(*self.saved_tensors)
                # Compute gradients...
            return grad_output

    print("Gradient checkpointing trades computation time for memory")
    print("Commonly used for training large Transformer models")
```

### 4. Gradient Handling in Mixed Precision Training

```python
def mixed_precision_gradients():
    """Gradient scaling in mixed precision training"""

    # Simulate FP16 forward propagation
    def forward_fp16(x, w):
        x_fp16 = x.astype(np.float16)
        w_fp16 = w.astype(np.float16)
        return (x_fp16 @ w_fp16).astype(np.float32)

    # Gradient scaling
    class GradScaler:
        def __init__(self, init_scale=2**16):
            self.scale = init_scale

        def scale_loss(self, loss):
            return loss * self.scale

        def unscale_grad(self, grad):
            return grad / self.scale

        def update(self, has_inf_or_nan):
            if has_inf_or_nan:
                self.scale *= 0.5
            else:
                self.scale *= 2.0

    scaler = GradScaler()
    print(f"Initial scaling factor: {scaler.scale}")
    print("Mixed precision training accelerates computation with FP16 while maintaining FP32 precision")
```

### 5. Gradient Flow in ResNet Residual Connections

Residual Networks (ResNet) solve the vanishing gradient problem in deep networks through **skip connections**.

#### Residual Block Structure

Forward propagation of a standard residual block:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

Where:
- $\mathbf{x}$ is the input
- $\mathcal{F}(\mathbf{x}, \{W_i\})$ is the residual mapping (typically contains 2-3 convolutional layers)
- $\mathbf{y}$ is the output

#### Complete Derivation of Residual Connection Gradients

**Goal**: Derive how gradient $\frac{\partial L}{\partial \mathbf{x}}$ flows through the residual connection.

**Step 1**: Write forward propagation.

$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

**Step 2**: Apply chain rule.

$$\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$$

**Step 3**: Compute $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$.

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial}{\partial \mathbf{x}}(\mathcal{F}(\mathbf{x}) + \mathbf{x}) = \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I}$$

Where $\mathbf{I}$ is the identity matrix.

**Step 4**: Obtain final gradient formula.

$$\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \left(\frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I}\right)$$

$$\boxed{\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \cdot \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \frac{\partial L}{\partial \mathbf{y}}}$$

#### Gradient Propagation in Deep Networks

For a residual network with $L$ layers, let the output of layer $l$ be $\mathbf{x}_{l+1} = \mathcal{F}_l(\mathbf{x}_l) + \mathbf{x}_l$.

**Gradient from layer $L$ to layer $l$**:

$$\frac{\partial L}{\partial \mathbf{x}_l} = \frac{\partial L}{\partial \mathbf{x}_L} \prod_{i=l}^{L-1} \left(\frac{\partial \mathcal{F}_i}{\partial \mathbf{x}_i} + \mathbf{I}\right)$$

**Key observation**:
- Even if $\frac{\partial \mathcal{F}_i}{\partial \mathbf{x}_i}$ is small (vanishing gradient), the $\mathbf{I}$ term guarantees at least $\frac{\partial L}{\partial \mathbf{x}_L}$ can be directly passed through
- This is equivalent to providing a "highway" for gradients

#### Comparison with Ordinary Networks

**Ordinary deep networks**:

$$\frac{\partial L}{\partial \mathbf{x}_l} = \frac{\partial L}{\partial \mathbf{x}_L} \prod_{i=l}^{L-1} \frac{\partial \mathcal{F}_i}{\partial \mathbf{x}_i}$$

When the network is deep, if $\|\frac{\partial \mathcal{F}_i}{\partial \mathbf{x}_i}\| < 1$, gradients decay exponentially.

**Residual networks**:

$$\frac{\partial L}{\partial \mathbf{x}_l} = \frac{\partial L}{\partial \mathbf{x}_L} \prod_{i=l}^{L-1} \left(\mathbf{I} + \frac{\partial \mathcal{F}_i}{\partial \mathbf{x}_i}\right)$$

Due to the existence of the $\mathbf{I}$ term, even if the gradient of some $\mathcal{F}_i$ is small, the overall gradient can still propagate effectively.

#### Identity Mapping Condition

He et al. (2016) proved that when residual blocks satisfy the "identity mapping" condition, gradient propagation is optimal:

$$\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}_l(\mathbf{x}_l)$$

**Derivation**: Assume $\mathcal{F}_l(\mathbf{x}_l) \approx 0$ (initialized close to zero), then:

$$\mathbf{x}_{l+1} \approx \mathbf{x}_l$$

This means signals can be passed directly during initial training, and gradients can also propagate smoothly.

#### Code Implementation

```python
import numpy as np

class ResidualBlock:
    """Residual block implementation"""

    def __init__(self, in_channels, out_channels, stride=1):
        # Main path: two 3x3 convolutions
        self.conv1 = np.random.randn(3, 3, in_channels, out_channels) * 0.01
        self.conv2 = np.random.randn(3, 3, out_channels, out_channels) * 0.01

        # If dimensions don't match, need 1x1 convolution for adjustment
        self.shortcut_conv = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut_conv = np.random.randn(1, 1, in_channels, out_channels) * 0.01

        self.cache = None

    def forward(self, x):
        """
        Forward propagation: y = F(x) + x (or shortcut)
        """
        identity = x

        # Main path F(x)
        out = self._conv2d(x, self.conv1)
        out = self._relu(out)
        out = self._conv2d(out, self.conv2)

        # Skip connection
        if self.shortcut_conv is not None:
            identity = self._conv2d(x, self.shortcut_conv)

        # Residual connection: This is the key!
        out = out + identity

        self.cache = (x, identity, out)
        return self._relu(out)

    def backward(self, grad_output):
        """
        Backpropagation demonstrating gradient flow

        Key: Gradient flows back through two paths
        dL/dx = dL/dy * (dF/dx + I)
        """
        x, identity, pre_relu = self.cache

        # ReLU gradient
        grad = grad_output * (pre_relu > 0)

        # Gradient bifurcation in residual connection
        # One path: through main path
        grad_main = self._conv2d_backward(grad, self.conv2)

        # Another path: direct transmission (skip connection)
        # This is the "gradient highway"!
        grad_skip = grad if self.shortcut_conv is None else self._conv2d_backward(grad, self.shortcut_conv)

        # Merge gradients from both paths
        grad_input = grad_main + grad_skip

        return grad_input

    def _conv2d(self, x, w):
        """Simplified convolution operation"""
        return x  # Actual implementation omitted

    def _conv2d_backward(self, grad, w):
        """Simplified convolution backpropagation"""
        return grad  # Actual implementation omitted

    def _relu(self, x):
        return np.maximum(0, x)


def demonstrate_gradient_flow():
    """Demonstrate how residual connections maintain gradient flow"""

    # Simulate deep network
    num_layers = 100

    # Ordinary network: gradients decay layer by layer
    grad_normal = 1.0
    avg_grad_magnitude = 0.9  # Assume each layer's gradient shrinks by 10%

    for i in range(num_layers):
        grad_normal *= avg_grad_magnitude

    print(f"Gradient after {num_layers} layers in ordinary network: {grad_normal:.6e}")

    # Residual network: gradients are maintained
    grad_residual = 1.0

    for i in range(num_layers):
        # Even if F's gradient is small, the I term ensures gradients don't vanish
        # Assume dF/dx ≈ 0.1, plus I term
        grad_residual *= (0.1 + 1.0)  # 1.1 > 1

    print(f"Gradient after {num_layers} layers in residual network: {grad_residual:.6e}")

    print("\nKey conclusions:")
    print("1. Ordinary network gradients decay exponentially, almost no gradient in deep layers")
    print("2. Residual network gradients have 'identity mapping' protection, deep layers can still learn effectively")

demonstrate_gradient_flow()
```

#### Output Example

```
Gradient after 100 layers in ordinary network: 2.656140e-05
Gradient after 100 layers in residual network: 1.378061e+04

Key conclusions:
1. Ordinary network gradients decay exponentially, almost no gradient in deep layers
2. Residual network gradients have 'identity mapping' protection, deep layers can still learn effectively
```

#### Gradient Characteristics of ResNet Variants

| Variant | Skip connection method | Gradient characteristics |
|------|-------------|----------|
| ResNet-v1 | $\mathbf{x} + \mathcal{F}(\mathbf{x})$ | Basic residual connection |
| ResNet-v2 | Pre-activation | Better gradient flow |
| DenseNet | Dense connections | Gradients reach from multiple layers directly |
| Wide ResNet | Widen residual blocks | More gradient paths |

#### Mathematical Proof: Why Residual Networks Are Easier to Optimize

**Theorem**: For any small $\epsilon > 0$, there exists a residual network such that $\|\frac{\partial L}{\partial \mathbf{x}_l} - \frac{\partial L}{\partial \mathbf{x}_L}\| < \epsilon$.

**Proof idea**:

Let $\mathcal{F}_i(\mathbf{x}) \equiv 0$ (identity mapping), then:

$$\mathbf{x}_{l+1} = \mathbf{x}_l + 0 = \mathbf{x}_l$$

$$\frac{\partial \mathbf{x}_{l+1}}{\partial \mathbf{x}_l} = \mathbf{I}$$

Therefore:

$$\frac{\partial L}{\partial \mathbf{x}_l} = \frac{\partial L}{\partial \mathbf{x}_L} \cdot \mathbf{I}^{L-l} = \frac{\partial L}{\partial \mathbf{x}_L}$$

This shows that the residual structure **at least** won't cause gradients to vanish, which ordinary networks cannot achieve.

### 6. Actual Framework Comparison

| Feature | PyTorch | TensorFlow | JAX |
|------|---------|------------|-----|
| Automatic differentiation | torch.autograd | tf.GradientTape | jax.grad |
| Computation graph | Dynamic graph | Static/dynamic | Function transformation |
| Gradient accumulation | Supported | Supported | Supported |
| Mixed precision | torch.cuda.amp | tf.mixed_precision | jax.default_matmul_precision |

---

## Summary

This chapter introduced vector matrix calculus and the backpropagation algorithm:

| Concept | Formula/Definition | Application |
|------|----------|------|
| Jacobian matrix | $\mathbf{J}_{ij} = \frac{\partial f_i}{\partial x_j}$ | Derivative of vector-valued functions |
| Linear layer gradient | $\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^\top \frac{\partial L}{\partial \mathbf{Y}}$ | Backpropagation |
| Softmax gradient | $\frac{\partial f_i}{\partial x_j} = f_i(\delta_{ij} - f_j)$ | Classification output |
| Softmax+Cross-entropy | $\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$ | Concise gradient |

### Key Formulas

**Matrix derivatives**:
- $\frac{\partial}{\partial \mathbf{X}} \text{tr}(\mathbf{A}\mathbf{X}) = \mathbf{A}^\top$
- $\frac{\partial}{\partial \mathbf{X}} \mathbf{a}^\top \mathbf{X} \mathbf{b} = \mathbf{a}\mathbf{b}^\top$
- $\frac{\partial}{\partial \mathbf{X}} \det(\mathbf{X}) = \det(\mathbf{X})(\mathbf{X}^{-1})^\top$

**Backpropagation core**:
- Forward propagation: cache intermediate results
- Backpropagation: apply chain rule layer by layer
- Numerical verification: ensure gradients are correct

---

**Previous section**: [Chapter 2(c): Higher-order Derivatives and Taylor Expansion](02c-higher-derivatives-taylor_EN.md)

**Next chapter**: [Chapter 3: Probability Theory](03-probability.md) - Learn about probability distributions, conditional probability, Bayes' theorem, and other concepts.

**Return to**: [Mathematics Fundamentals Tutorial Index](../math-fundamentals.md)
