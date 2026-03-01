# Getting Started with nanotorch

nanotorch is a minimal PyTorch implementation from scratch, designed for educational purposes. This guide will help you install nanotorch and start using it for basic deep learning tasks.

## Table of Contents
- [Installation](#installation)
- [Basic Tensor Operations](#basic-tensor-operations)
- [Automatic Differentiation](#automatic-differentiation)
- [Building Neural Networks](#building-neural-networks)
- [Training a Model](#training-a-model)
- [Saving and Loading Models](#saving-and-loading-models)
- [Next Steps](#next-steps)

## Installation

### Prerequisites
- Python 3.8 or higher
- NumPy

### Installation Steps

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd nanotorch
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy
   ```

3. **Install nanotorch in development mode**:
   ```bash
   pip install -e .
   ```

   Or, if you prefer to use uv (recommended for this project):
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv sync
   ```

4. **Verify installation**:
   ```python
   import nanotorch
   print(nanotorch.__version__)  # Should print "0.1.0"
   ```

## Basic Tensor Operations

### Creating Tensors

```python
import numpy as np
from nanotorch import Tensor

# Create from Python lists
t1 = Tensor([1, 2, 3, 4])

# Create from NumPy arrays
t2 = Tensor(np.array([1.0, 2.0, 3.0]))

# Create scalar tensor
t3 = Tensor(5.0)

# Use factory methods
zeros = Tensor.zeros((3, 4))      # 3x4 tensor of zeros
ones = Tensor.ones((2, 3))        # 2x3 tensor of ones
rand = Tensor.rand((2, 2))        # Uniform random [0, 1)
randn = Tensor.randn((3, 3))      # Standard normal
eye = Tensor.eye(4)               # 4x4 identity matrix
arange = Tensor.arange(10)        # [0, 1, ..., 9]
```

### Tensor Properties

```python
t = Tensor.randn((3, 4))

print(t.shape)           # (3, 4)
print(t.ndim)            # 2
print(t.dtype)           # float32
print(t.requires_grad)   # False (by default)
```

### Basic Operations

```python
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

# Arithmetic operations
c = a + b      # Element-wise addition
d = a - b      # Element-wise subtraction
e = a * b      # Element-wise multiplication
f = a / b      # Element-wise division
g = a ** 2     # Element-wise power

# Matrix operations (for 2D tensors)
x = Tensor.randn((3, 4))
y = Tensor.randn((4, 5))
z = x @ y      # Matrix multiplication
w = x.T        # Transpose

# Reduction operations
total = a.sum()        # Sum of all elements
mean = a.mean()        # Mean of all elements
max_val = a.max()      # Maximum value
min_val = a.min()      # Minimum value

# Shape operations
reshaped = x.reshape((2, 6))  # Reshape to 2x6
squeezed = Tensor([[[1], [2], [3]]]).squeeze()  # Remove size-1 dimensions

# Activation functions
relu_out = a.relu()      # Rectified Linear Unit
sigmoid_out = a.sigmoid()  # Sigmoid
tanh_out = a.tanh()      # Hyperbolic tangent
```

## Automatic Differentiation

### Gradient Tracking

```python
# Enable gradient tracking
x = Tensor([2.0], requires_grad=True)
y = Tensor([3.0], requires_grad=True)

# Perform operations
z = x * y + x ** 2

# Compute gradients
z.backward()

print(f"dz/dx = {x.grad}")  # dz/dx = y + 2x = 3 + 4 = 7
print(f"dz/dy = {y.grad}")  # dz/dy = x = 2
```

### Using `no_grad()` Context

```python
from nanotorch import no_grad

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

# Operations inside no_grad() won't track gradients
with no_grad():
    y = x * 2  # No gradient tracking

# Operations outside track gradients
z = x * 3      # Tracks gradients
z.backward()
```

### Gradient Checking

You can verify gradient correctness using finite differences:

```python
def gradient_check(func, tensor, eps=1e-5):
    """Compare autograd gradient with finite differences."""
    analytic_grad = tensor.grad.data
    
    # Compute numerical gradient
    numerical_grad = np.zeros_like(tensor.data)
    it = np.nditer(tensor.data, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        original = tensor.data[idx]
        
        # f(x + eps)
        tensor.data[idx] = original + eps
        loss_plus = func()
        
        # f(x - eps)
        tensor.data[idx] = original - eps
        loss_minus = func()
        
        # Central difference
        numerical_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        tensor.data[idx] = original
        it.iternext()
    
    diff = np.abs(analytic_grad - numerical_grad).max()
    return diff < 1e-7, diff

# Example usage
x = Tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()

# Define function for gradient checking
def f():
    return (x ** 2 + 3 * x + 1).item()

is_correct, max_diff = gradient_check(f, x)
print(f"Gradient correct: {is_correct}, max difference: {max_diff}")
```

## Building Neural Networks

### Creating a Simple Network

```python
from nanotorch.nn import Module, Linear, ReLU, Sequential

# Method 1: Define a custom module
class SimpleNet(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Method 2: Use Sequential
model = Sequential(
    Linear(10, 20),
    ReLU(),
    Linear(20, 5),
    ReLU(),
    Linear(5, 1)
)

# Create input and run forward pass
input_tensor = Tensor.randn((32, 10))  # Batch of 32 samples
output = model(input_tensor)
print(f"Output shape: {output.shape}")  # (32, 1)
```

### Accessing Parameters

```python
# Get all parameters
params = list(model.parameters())
print(f"Number of parameters: {len(params)}")

# Count total parameters
total_params = sum(p.data.size for p in params)
print(f"Total parameters: {total_params}")

# Print parameter shapes
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

### Training and Evaluation Modes

```python
from nanotorch.nn import Dropout

model = Sequential(
    Linear(10, 20),
    Dropout(p=0.5),  # Dropout layer
    ReLU(),
    Linear(20, 1)
)

# Set to training mode (dropout active)
model.train()
output_train = model(input_tensor)

# Set to evaluation mode (dropout inactive)
model.eval()
output_eval = model(input_tensor)
```

## Training a Model

### Complete Training Example

```python
import numpy as np
from nanotorch import Tensor
from nanotorch.nn import Sequential, Linear, ReLU, MSE
from nanotorch.optim import SGD

# 1. Create synthetic data
np.random.seed(42)
X = np.random.randn(100, 10).astype(np.float32)
y = np.random.randn(100, 1).astype(np.float32)

# Convert to tensors
X_tensor = Tensor(X)
y_tensor = Tensor(y)

# 2. Create model
model = Sequential(
    Linear(10, 20),
    ReLU(),
    Linear(20, 10),
    ReLU(),
    Linear(10, 1)
)

# 3. Define loss function and optimizer
criterion = MSE()
optimizer = SGD(model.parameters(), lr=0.01)

# 4. Training loop
num_epochs = 100
batch_size = 32
num_samples = X.shape[0]

for epoch in range(num_epochs):
    # Shuffle data
    indices = np.random.permutation(num_samples)
    X_shuffled = X_tensor.data[indices]
    y_shuffled = y_tensor.data[indices]
    
    epoch_loss = 0
    num_batches = 0
    
    # Mini-batch training
    for i in range(0, num_samples, batch_size):
        # Get batch
        X_batch = Tensor(X_shuffled[i:i+batch_size])
        y_batch = Tensor(y_shuffled[i:i+batch_size])
        
        # Forward pass
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    # Print progress
    if epoch % 10 == 0:
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch:3d}, Loss: {avg_loss:.4f}")

print("Training completed!")
```

### Using Different Optimizers

```python
from nanotorch.optim import Adam

# SGD with momentum
sgd_optimizer = SGD(
    params=model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001
)

# Adam optimizer
adam_optimizer = Adam(
    params=model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0
)
```

### Monitoring Training

```python
# Track training history
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = train_one_epoch(model, train_data, criterion, optimizer)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    with no_grad():
        val_loss = evaluate(model, val_data, criterion)
    val_losses.append(val_loss)
    
    # Early stopping
    if len(val_losses) > 10 and val_loss > min(val_losses[-10:]):
        print("Early stopping triggered")
        break
```

## Saving and Loading Models

### Saving Model State

```python
from nanotorch.utils import save, save_state_dict

# Save entire model
save(model, "model.pth")

# Save only parameters
save_state_dict(model.state_dict(), "model_state.pth")

# Save with metadata
import pickle
metadata = {
    "epoch": 100,
    "loss": 0.05,
    "accuracy": 0.92
}
with open("model_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
```

### Loading Model State

```python
from nanotorch.utils import load, load_state_dict

# Load entire model
loaded_model = load("model.pth")

# Load parameters into existing model
state_dict = load_state_dict("model_state.pth")
model.load_state_dict(state_dict)

# Load with strict=False to ignore missing keys
model.load_state_dict(state_dict, strict=False)
```

### Resuming Training

```python
def load_checkpoint(path, model, optimizer):
    """Load checkpoint and resume training."""
    checkpoint = load_state_dict(path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    
    print(f"Resumed from epoch {epoch}, loss {loss:.4f}")
    return epoch, loss

# Save checkpoint
checkpoint = {
    "epoch": epoch,
    "loss": loss,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict()
}
save_state_dict(checkpoint, "checkpoint.pth")
```

## Next Steps

### Explore Examples

Check out the `examples/` directory for more comprehensive examples:

```bash
python examples/simple_neural_net.py
python examples/conv2d_training.py
```

### Run Benchmarks

Test performance with the benchmark scripts:

```bash
python benchmarks/tensor_operations.py --op matmul
python benchmarks/memory_usage.py --test all
```

### Read Documentation

- [API Reference](api.md) - Complete API documentation
- [Design Documentation](design.md) - Architecture and implementation details

### Extend nanotorch

Consider implementing:
1. New activation functions
2. Additional loss functions
3. More optimizer algorithms
4. GPU support (using CuPy)

### Troubleshooting

Common issues and solutions:

1. **ImportError**: Make sure you're in the correct directory and have installed dependencies.
2. **Shape mismatch**: Check tensor dimensions with `.shape`.
3. **Memory issues**: Use smaller batch sizes or enable gradient checkpointing.
4. **Numerical instability**: Add small epsilon to divisions and logs.

### Getting Help

If you encounter issues:
1. Check the existing tests for usage examples
2. Review the source code (it's meant to be readable!)
3. Compare with PyTorch documentation for similar functionality

## Conclusion

You've now learned how to:
- Install and import nanotorch
- Perform basic tensor operations
- Use automatic differentiation
- Build and train neural networks
- Save and load models

nanotorch is designed to be educational—feel free to explore the source code to understand how each component works. Happy learning!