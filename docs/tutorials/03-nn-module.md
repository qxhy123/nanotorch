# 第三章：Module 基类与参数管理

Module 是所有神经网络层的基类。本章我们实现参数管理和模块组合。

## 3.1 为什么需要 Module？

直接用 Tensor 构建网络很繁琐：

```python
# 不使用 Module
W1 = Tensor.randn((784, 256), requires_grad=True)
b1 = Tensor.zeros((256,), requires_grad=True)
W2 = Tensor.randn((256, 10), requires_grad=True)
b2 = Tensor.zeros((10,), requires_grad=True)

def forward(x):
    h = x @ W1 + b1
    h = relu(h)
    return h @ W2 + b2

# 手动收集参数
params = [W1, b1, W2, b2]
for p in params:
    p.grad = None
```

使用 Module 更优雅：

```python
# 使用 Module
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

# 自动管理参数
for p in model.parameters():
    p.grad = None
```

## 3.2 Module 基类实现

```python
# module.py
from collections import OrderedDict
from typing import Dict, Iterator, Optional, Any

class Module:
    """所有神经网络模块的基类"""
    
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
        """注册一个参数"""
        if param is not None:
            self._parameters[name] = param
    
    def register_module(self, name: str, module: 'Module') -> None:
        """注册一个子模块"""
        self._modules[name] = module
    
    def add_module(self, name: str, module: 'Module') -> None:
        """添加子模块"""
        self.register_module(name, module)
    
    def parameters(self, recurse: bool = True) -> Iterator[Tensor]:
        """返回所有参数"""
        for param in self._parameters.values():
            yield param
        
        if recurse:
            for module in self._modules.values():
                yield from module.parameters()
    
    def named_parameters(self, recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        """返回所有参数（带名称）"""
        for name, param in self._parameters.items():
            yield name, param
        
        if recurse:
            for module_name, module in self._modules.items():
                for param_name, param in module.named_parameters():
                    yield f"{module_name}.{param_name}", param
    
    def modules(self) -> Iterator['Module']:
        """返回所有子模块"""
        for module in self._modules.values():
            yield module
            yield from module.modules()
    
    def train(self, mode: bool = True) -> 'Module':
        """设置训练模式"""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self) -> 'Module':
        """设置评估模式"""
        return self.train(False)
    
    def zero_grad(self) -> None:
        """清零所有参数的梯度"""
        for param in self.parameters():
            if param.grad is not None:
                param.grad = None
```

## 3.3 实现 Linear 层

```python
# linear.py
import math

class Linear(Module):
    """全连接层: y = x @ W.T + b"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 初始化权重 (Kaiming/He 初始化)
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

### Linear 层的维度

```
输入 x: (batch, in_features)
权重 W: (out_features, in_features)
偏置 b: (out_features,)
输出 y: (batch, out_features)

y = x @ W.T + b
  = (batch, in_features) @ (in_features, out_features) + (out_features,)
  = (batch, out_features) + (out_features,)
  = (batch, out_features)
```

## 3.4 实现 Sequential 容器

```python
class Sequential(Module):
    """顺序容器，按顺序执行多个层"""
    
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            self.add_module(str(i), layer)
    
    def forward(self, x: Tensor) -> Tensor:
        for layer in self._modules.values():
            x = layer(x)
        return x
```

## 3.5 使用示例

```python
from nanotorch import Tensor
from nanotorch.nn import Module, Linear, Sequential
from nanotorch.nn import ReLU

# 定义模型
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10)
)

# 查看参数
print(f"Total parameters: {sum(p.data.size for p in model.parameters())}")

# 查看参数名称
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# 前向传播
x = Tensor.randn((32, 784))
output = model(x)
print(f"Output shape: {output.shape}")  # (32, 10)

# 清零梯度
model.zero_grad()

# 训练/评估模式
model.train()   # 训练模式
model.eval()    # 评估模式
```

## 3.6 实现状态字典

状态字典用于保存和加载模型：

```python
class Module:
    # ... 前面的代码 ...
    
    def state_dict(self) -> Dict[str, np.ndarray]:
        """获取模型状态"""
        state = OrderedDict()
        
        # 保存参数
        for name, param in self.named_parameters():
            state[name] = param.data.copy()
        
        # 保存缓冲区（如 BatchNorm 的 running_mean）
        for name, buffer in self.named_buffers():
            state[name] = buffer.data.copy()
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, np.ndarray], strict: bool = True):
        """加载模型状态"""
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

## 3.7 保存和加载模型

```python
import numpy as np

# 保存模型
state = model.state_dict()
np.savez('model.npz', **state)

# 加载模型
state = dict(np.load('model.npz'))
model.load_state_dict(state)
```

## 3.8 实现嵌套模块

```python
class MLP(Module):
    """自定义 MLP 模块"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # 注册子模块
        self.layer1 = Linear(input_dim, hidden_dim)
        self.layer2 = Linear(hidden_dim, output_dim)
        self.activation = ReLU()
        
        # 注册到 _modules
        self.register_module('layer1', self.layer1)
        self.register_module('layer2', self.layer2)
        self.register_module('activation', self.activation)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# 使用
mlp = MLP(784, 256, 10)
x = Tensor.randn((32, 784))
output = mlp(x)
```

## 3.9 模块遍历

```python
# 遍历所有模块
for module in model.modules():
    print(module.__class__.__name__)

# 遍历所有参数
total_params = 0
for name, param in model.named_parameters():
    n_params = param.data.size
    total_params += n_params
    print(f"{name}: {param.shape}, {n_params} params")

print(f"Total: {total_params} parameters")
```

## 3.10 完整代码

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
    
    def named_parameters(self, recurse: bool = True) -> Iterator[Tuple[str, 'Tensor']]:
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

## 3.11 练习

1. **实现 `Module.__repr__()`**：打印模块结构

2. **实现 `Module.apply(fn)`**：对每个子模块应用函数

3. **实现 `children()`**：只返回直接子模块

4. **实现 `named_children()`**：返回直接子模块及名称

5. **挑战**：实现 `named_buffers()` 方法

## 下一章

现在我们有了模块系统！下一章，我们将实现**激活函数**，让神经网络能够学习非线性关系。

→ [第四章：激活函数](04-activation.md)
