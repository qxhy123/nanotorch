# nanotorch Design Documentation

This document describes the architecture and design decisions behind nanotorch, a minimal PyTorch implementation from scratch.

## Table of Contents
- [Project Goals](#project-goals)
- [Architecture Overview](#architecture-overview)
- [Tensor Implementation](#tensor-implementation)
- [Automatic Differentiation](#automatic-differentiation)
- [Neural Network Modules](#neural-network-modules)
- [Optimizers](#optimizers)
- [Performance Considerations](#performance-considerations)
- [Future Extensions](#future-extensions)

## Project Goals

nanotorch was created with the following goals:

1. **Educational Value**: Provide a clear, readable implementation that teaches how PyTorch works internally.
2. **Minimal Dependencies**: Use only NumPy as an external dependency.
3. **API Compatibility**: Follow PyTorch-like APIs where it enhances educational value.
4. **Complete Core Functionality**: Implement tensors, autograd, nn modules, and optimizers.
5. **Test-Driven Development**: Maintain comprehensive test coverage for all core functionality.

## Architecture Overview

nanotorch is organized into several modules:

```
nanotorch/
├── tensor.py          # Core Tensor class with operations
├── autograd.py        # Automatic differentiation engine
├── nn/                # Neural network modules
│   ├── module.py      # Base Module class
│   ├── linear.py      # Linear layer
│   ├── activation.py  # Activation functions
│   ├── loss.py        # Loss functions
│   ├── conv.py        # Convolutional layers
│   ├── batchnorm.py   # Batch normalization
│   └── dropout.py     # Dropout regularization
├── optim/             # Optimizers
│   ├── optimizer.py   # Base Optimizer class
│   ├── sgd.py         # SGD optimizer
│   └── adam.py        # Adam optimizer
└── utils.py           # Utility functions
```

### Core Design Principles

1. **Explicit over Implicit**: Clear computational graph building and traversal.
2. **Pythonic API**: Follow Python conventions and PyTorch patterns.
3. **Numerical Stability**: Handle edge cases like division by zero, NaN propagation.
4. **Memory Efficiency**: Minimize unnecessary tensor copies.

## Tensor Implementation

### Data Storage

Tensors store data as NumPy arrays with `float32` dtype:

```python
class Tensor:
    def __init__(self, data, requires_grad=False, _op=None, _parents=()):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._op = _op          # Operation that created this tensor
        self._parents = _parents  # Parent tensors in computational graph
```

### Computational Graph

Each tensor maintains:
- `_op`: String identifier of the operation that created it
- `_parents`: Tuple of parent tensors in the computational graph
- `_ctx`: Additional context needed for gradient computation

### Operations

Operations are implemented as methods on the `Tensor` class. Each operation:
1. Computes the forward pass using NumPy
2. Creates a new tensor with `_op` and `_parents` set
3. Sets `requires_grad` if any parent requires gradients

Example of addition operation:

```python
def __add__(self, other):
    if not isinstance(other, Tensor):
        other = Tensor(other, requires_grad=False)
    
    result_data = self.data + other.data
    return Tensor(
        result_data,
        requires_grad=self.requires_grad or other.requires_grad,
        _op="add",
        _parents=(self, other),
    )
```

## Automatic Differentiation

### Reverse-Mode Autodiff

nanotorch implements reverse-mode automatic differentiation (backpropagation):

1. **Forward Pass**: Operations build a computational graph.
2. **Backward Pass**: Gradients are propagated from output to inputs using the chain rule.

### Gradient Computation

The `backward()` method:
1. Builds a topological ordering of the computational graph using DFS
2. Initializes the gradient of the output tensor (default: ones)
3. Propagates gradients backwards through the graph

```python
def backward(self, gradient=None):
    # Build topological order
    topo = []
    visited = set()
    
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for parent in v._parents:
                if isinstance(parent, Tensor):
                    build_topo(parent)
            topo.append(v)
    
    build_topo(self)
    
    # Initialize gradient
    if gradient is None:
        gradient = Tensor(np.ones_like(self.data), requires_grad=False)
    
    # Propagate gradients backwards
    for v in reversed(topo):
        if v._op is not None and v.requires_grad and v.grad is not None:
            self._backward_operation(v, v._op, v._parents)
```

### Operation-Specific Gradients

Each operation has a gradient defined in `_backward_operation`. For example, addition:

```python
if op == "add":
    # Gradient for addition: dL/da = dL/dout, dL/db = dL/dout
    if len(parents) == 2:
        a, b = parents
        accumulate_grad(a, tensor.grad)
        accumulate_grad(b, tensor.grad)
```

### Broadcasting Support

The autograd system handles NumPy-style broadcasting by summing gradients over broadcast dimensions:

```python
def reduce_gradient(grad_contrib, target_shape):
    """Reduce gradient to target shape by summing over broadcast dimensions."""
    if grad_contrib.shape == target_shape:
        return grad_contrib
    
    # Identify axes where broadcasting occurred
    axes_to_sum = []
    # ... compute axes ...
    
    if axes_to_sum:
        reduced_data = grad_contrib.data.sum(axis=tuple(axes_to_sum), keepdims=False)
        return Tensor(reduced_data.reshape(target_shape), requires_grad=False)
```

## Neural Network Modules

### Module Base Class

The `Module` class provides:
- Parameter management via `register_parameter()`
- Training/evaluation modes
- State dict serialization
- Forward pass abstraction

```python
class Module:
    def __init__(self):
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self.training = True
    
    def forward(self, x):
        raise NotImplementedError
    
    def __call__(self, x):
        return self.forward(x)
```

### Parameter Management

Parameters are tensors with `requires_grad=True` that are registered with the module:

```python
def register_parameter(self, name, param):
    if not isinstance(param, Tensor):
        raise TypeError(f"Parameter must be Tensor, got {type(param)}")
    self._parameters[name] = param
```

### Sequential Container

The `Sequential` class allows stacking modules:

```python
class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            self.add_module(str(i), module)
    
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x
```

## Optimizers

### Optimizer Base Class

The `Optimizer` class manages parameter updates:

```python
class Optimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.zero_grad()
    
    def step(self):
        raise NotImplementedError
```

### SGD Implementation

Stochastic Gradient Descent with momentum:

```python
class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, weight_decay=0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.params]
    
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.data
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Apply momentum
            if self.momentum != 0:
                self.velocity[i] = self.momentum * self.velocity[i] + grad
                grad = self.velocity[i]
            
            # Update parameter
            param.data -= self.lr * grad
```

### Adam Implementation

Adam optimizer with bias correction:

```python
class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0
    
    def step(self):
        self.t += 1
        beta1, beta2 = self.betas
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.data
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Update biased first moment estimate
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
            
            # Update biased second moment estimate
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad * grad)
            
            # Compute bias-corrected moment estimates
            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)
            
            # Update parameter
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

## Performance Considerations

### Memory Usage

1. **Tensor Overhead**: Each tensor stores data, gradient, and graph metadata.
2. **Gradient Accumulation**: Gradients are accumulated across backward passes.
3. **Graph Retention**: The computational graph is retained until tensors are deleted.

### Optimization Opportunities

1. **Operation Fusion**: Combine multiple operations to reduce memory traffic.
2. **In-place Operations**: Use in-place operations where safe to reduce memory allocation.
3. **Gradient Checkpointing**: Store only some activations and recompute others.

### Current Limitations

1. **No GPU Support**: All operations run on CPU using NumPy.
2. **Basic Autograd**: Gradient computation may not handle all edge cases.
3. **Limited Operations**: Compared to PyTorch, only core operations are implemented.

## Future Extensions

### Planned Features

1. **GPU Support**: Add CUDA backend using CuPy or similar.
2. **More Operations**: Implement more PyTorch operations (e.g., conv2d backward, pooling).
3. **Distributed Training**: Basic multi-GPU support.
4. **JIT Compilation**: Simple just-in-time compilation for performance.
5. **ONNX Export**: Export models to ONNX format.

### Research Directions

1. **Sparse Tensors**: Support for sparse tensor operations.
2. **Quantization**: Post-training quantization for inference.
3. **Pruning**: Network pruning for model compression.
4. **AutoDiff Improvements**: More efficient autograd implementations.

## Testing Strategy

nanotorch uses a comprehensive test suite:

1. **Unit Tests**: Test individual components in isolation.
2. **Integration Tests**: Test interactions between components.
3. **Gradient Checking**: Compare autograd gradients with finite differences.
4. **Performance Tests**: Benchmark against NumPy and PyTorch.

## Contributing

When contributing to nanotorch:

1. **Follow Existing Patterns**: Maintain consistency with the codebase.
2. **Add Tests**: All new features must include tests.
3. **Update Documentation**: Keep API and design docs up to date.
4. **Performance Considerations**: Profile new features for performance impact.

## References

- PyTorch Documentation: https://pytorch.org/docs/
- "Automatic Differentiation in Machine Learning: a Survey" (Baydin et al., 2018)
- "Deep Learning" (Goodfellow, Bengio, Courville, 2016)
- NumPy Documentation: https://numpy.org/doc/

## License

nanotorch is released under the MIT License. See LICENSE file for details.