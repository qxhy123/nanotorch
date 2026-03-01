# Chapter 3: Module Base Class and Parameter Management

Module is the base class for all neural network layers. In this chapter, we implement parameter management and module composition.

## 3.1 Why Do We Need Module?

Building networks directly with Tensors is tedious:

```python
# Without Module
W1 = Tensor.randn((784, 256), requires_grad=True)
b1 = Tensor.zeros((256,), requires_grad=True)
W2 = Tensor.randn((256, 10), requires_grad=True)
b2 = Tensor.zeros((10,), requires_grad=True)

def forward(x):
    h = x @ W1 + b1
    h = relu(h)
    return h @ W2 + b2

# Manually collect parameters
params = [W1, b1, W2, b2]
for p in params:
    p.grad = None
```

Using Module is more elegant:

```python
# With Module
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

# Automatic parameter management
for p in model.parameters():
    p.grad = None
```

## 3.2 Module Base Class Implementation

```python
# module.py
from collections import OrderedDict
from typing import Dict, Iterator, Optional, Any

class Module:
    """Base class for all neural network modules"""
    
    def __init__(self):
        self._parameters: Dict[str, Tensor] = OrderedDict()
        self._modules: Dict[str, 'Module'] = OrderedDict()
        self._buffers: Dict[str, Tensor] = OrderedDict()
        self.training = True
    
    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError("Subclasses must implement forward()")
    
    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)
    
    def register_parameter(self, name: str, param: Tensor) -> None:
        """Register a parameter"""
        if param is not None:
            self._parameters[name] = param
    
    def register_module(self, name: str, module: 'Module') -> None:
        """Register a submodule"""
        self._modules[name] = module
    
    def add_module(self, name: str, module: 'Module') -> None:
        """Add a submodule"""
        self.register_module(name, module)
    
    def parameters(self, recurse: bool = True) -> Iterator[Tensor]:
        """Return all parameters"""
        for param in self._parameters.values():
            yield param
        
        if recurse:
            for module in self._modules.values():
                yield from module.parameters()
    
    def named_parameters(self, recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        """Return all parameters with names"""
        for name, param in self._parameters.items():
            yield name, param
        
        if recurse:
            for module_name, module in self._modules.items():
                for param_name, param in module.named_parameters():
                    yield f"{module_name}.{param_name}", param
    
    def modules(self) -> Iterator['Module']:
        """Return all submodules"""
        for module in self._modules.values():
            yield module
            yield from module.modules()
    
    def train(self, mode: bool = True) -> 'Module':
        """Set training mode"""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self) -> 'Module':
        """Set evaluation mode"""
        return self.train(False)
    
    def zero_grad(self) -> None:
        """Zero out all parameter gradients"""
        for param in self.parameters():
            if param.grad is not None:
                param.grad = None
```

## 3.3 Implementing Linear Layer

```python
# linear.py
import math

class Linear(Module):
    """Fully connected layer: y = x @ W.T + b"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights (Kaiming/He initialization)
        scale = math.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(out_features, in_features) * scale,
            requires_grad=True
        )
        self.register_parameter('weight', self.weight)
        
        if bias:
            self.bias = Tensor(
                np.zeros(out_features),
                requires_grad=True
            )
            self.register_parameter('bias', self.bias)
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        # y = x @ W.T + b
        output = x @ self.weight.T
        if self.bias is not None:
            output = output + self.bias
        return output
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
```

### Linear Layer Dimensions

```
Input x: (batch, in_features)
Weight W: (out_features, in_features)
Bias b: (out_features,)
Output y: (batch, out_features)

y = x @ W.T + b
  = (batch, in_features) @ (in_features, out_features) + (out_features,)
  = (batch, out_features) + (out_features,)
  = (batch, out_features)
```

## 3.4 Implementing Sequential Container

```python
class Sequential(Module):
    """Sequential container, executes layers in order"""
    
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            self.add_module(str(i), layer)
    
    def forward(self, x: Tensor) -> Tensor:
        for layer in self._modules.values():
            x = layer(x)
        return x
```

## 3.5 Usage Example

```python
from nanotorch import Tensor
from nanotorch.nn import Module, Linear, Sequential
from nanotorch.nn import ReLU

# Define model
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10)
)

# View parameters
print(f"Total parameters: {sum(p.data.size for p in model.parameters())}")

# View parameter names
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Forward propagation
x = Tensor.randn((32, 784))
output = model(x)
print(f"Output shape: {output.shape}")  # (32, 10)

# Zero gradients
model.zero_grad()

# Training/evaluation mode
model.train()   # Training mode
model.eval()    # Evaluation mode
```

## 3.6 Implementing State Dict

State dict is used for saving and loading models:

```python
class Module:
    # ... previous code ...
    
    def state_dict(self) -> Dict[str, np.ndarray]:
        """Get model state"""
        state = OrderedDict()
        
        # Save parameters
        for name, param in self.named_parameters():
            state[name] = param.data.copy()
        
        # Save buffers (like BatchNorm's running_mean)
        for name, buffer in self.named_buffers():
            state[name] = buffer.data.copy()
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, np.ndarray], strict: bool = True):
        """Load model state"""
        own_state = self.state_dict()
        
        for name, param in own_state.items():
            if name in state_dict:
                param.data = state_dict[name].copy()
            elif strict:
                raise KeyError(f"Missing key: {name}")
        
        if strict:
            missing = set(state_dict.keys()) - set(own_state.keys())
            if missing:
                raise KeyError(f"Unexpected keys: {missing}")
```

## 3.7 Saving and Loading Models

```python
import numpy as np

# Save model
state = model.state_dict()
np.savez('model.npz', **state)

# Load model
state = dict(np.load('model.npz'))
model.load_state_dict(state)
```

## 3.8 Implementing Nested Modules

```python
class MLP(Module):
    """Custom MLP module"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # Register submodules
        self.layer1 = Linear(input_dim, hidden_dim)
        self.layer2 = Linear(hidden_dim, output_dim)
        self.activation = ReLU()
        
        # Register to _modules
        self.register_module('layer1', self.layer1)
        self.register_module('layer2', self.layer2)
        self.register_module('activation', self.activation)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Usage
mlp = MLP(784, 256, 10)
x = Tensor.randn((32, 784))
output = mlp(x)
```

## 3.9 Module Traversal

```python
# Iterate all modules
for module in model.modules():
    print(module.__class__.__name__)

# Iterate all parameters
total_params = 0
for name, param in model.named_parameters():
    n_params = param.data.size
    total_params += n_params
    print(f"{name}: {param.shape}, {n_params} params")

print(f"Total: {total_params} parameters")
```

## 3.10 Complete Code

```python
# module.py
from collections import OrderedDict
from typing import Dict, Iterator, Optional, Tuple, Any
import numpy as np

class Module:
    def __init__(self):
        self._parameters: Dict[str, 'Tensor'] = OrderedDict()
        self._modules: Dict[str, 'Module'] = OrderedDict()
        self._buffers: Dict[str, 'Tensor'] = OrderedDict()
        self.training = True
    
    def forward(self, *args, **kwargs) -> 'Tensor':
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs) -> 'Tensor':
        return self.forward(*args, **kwargs)
    
    def register_parameter(self, name: str, param: 'Tensor') -> None:
        if param is not None:
            self._parameters[name] = param
    
    def register_buffer(self, name: str, tensor: 'Tensor') -> None:
        self._buffers[name] = tensor
    
    def add_module(self, name: str, module: 'Module') -> None:
        self._modules[name] = module
    
    def parameters(self, recurse: bool = True) -> Iterator['Tensor']:
        for p in self._parameters.values():
            yield p
        if recurse:
            for m in self._modules.values():
                yield from m.parameters()
    
    def named_parameters(self, recurse: bool = True) -> Iterator[Tuple[str, 'Tensor]]:
        for name, p in self._parameters.items():
            yield name, p
        if recurse:
            for module_name, m in self._modules.items():
                for param_name, p in m.named_parameters():
                    yield f"{module_name}.{param_name}", p
    
    def train(self, mode: bool = True) -> 'Module':
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self
    
    def eval(self) -> 'Module':
        return self.train(False)
    
    def zero_grad(self) -> None:
        for p in self.parameters():
            if p.grad is not None:
                p.grad = None
    
    def state_dict(self) -> Dict[str, np.ndarray]:
        state = OrderedDict()
        for name, p in self.named_parameters():
            state[name] = p.data.copy()
        for name, b in self.named_buffers():
            state[name] = b.data.copy()
        return state
    
    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        for name, p in self.named_parameters():
            if name in state_dict:
                p.data = state_dict[name].copy()


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            self.add_module(str(i), layer)
    
    def forward(self, x: 'Tensor') -> 'Tensor':
        for layer in self._modules.values():
            x = layer(x)
        return x
```

## 3.11 Exercises

1. **Implement `Module.__repr__()`**: Print module structure

2. **Implement `Module.apply(fn)`**: Apply function to each submodule

3. **Implement `children()`**: Return only direct submodules

4. **Implement `named_children()`**: Return direct submodules with names

5. **Challenge**: Implement `named_buffers()` method

## Next Chapter

Now we have a module system! In the next chapter, we will implement **activation functions**, enabling neural networks to learn non-linear relationships.

→ [Chapter 4: Activation Functions](04-activation_EN.md)
