# 第三章：Module 基类与参数管理

## 小时候，你一定玩过积木...

那些五颜六色的小方块，每一块都平平无奇。但当你把它们拼在一起——一座城堡、一辆赛车、一只恐龙，就从无到有地诞生了。

乐高积木的神奇之处在于：**任意两块，都能无缝衔接**。上面有凸起，下面有凹槽，统一的接口让一切组合成为可能。

神经网络也是如此。

一个 Linear 层，平平无奇。一个 ReLU 激活，也很简单。但当你把它们串起来——784→256→128→10——突然间，这堆数字电路学会了认出手写数字，识别人脸，甚至理解语言。

```
积木的哲学：

  简单单元 → 无限组合 → 无穷可能

  Linear + ReLU + Linear = 分类器
  Conv + Pool + Conv + Pool = 视觉系统
  Embedding + Attention + FFN = 语言模型
```

**Module，就是神经网络世界的"乐高接口"。**

---

## 3.1 为什么需要 Module？

### 问题：直接用 Tensor 太麻烦

```python
# 不使用 Module —— 像用散乱的积木
W1 = Tensor.randn((784, 256), requires_grad=True)  # 第一层权重
b1 = Tensor.zeros((256,), requires_grad=True)      # 第一层偏置
W2 = Tensor.randn((256, 10), requires_grad=True)  # 第二层权重
b2 = Tensor.zeros((10,), requires_grad=True)      # 第二层偏置

def forward(x):
    h = x @ W1 + b1
    h = relu(h)
    return h @ W2 + b2

# 问题1：怎么获取所有参数？
params = [W1, b1, W2, b2]  # 手动收集，容易漏！

# 问题2：怎么清零梯度？
for p in params:
    p.grad = None  # 每次都要写循环

# 问题3：怎么保存模型？
# 要手动保存每个参数...

# 问题4：如果网络有100层呢？
# 💀 噩梦...
```

### 解决：用 Module 统一管理

```python
# 使用 Module —— 像用乐高套件
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

# 自动收集所有参数
for p in model.parameters():
    p.grad = None  # 一行搞定！

# 保存模型
state = model.state_dict()
np.savez('model.npz', **state)
```

**一句话总结**：Module = 参数自动管理 + 层自动组合 + 状态自动保存。

---

## 3.2 Module 基类：乐高的"凹凸接口"

### 设计思路

Module 需要解决三个问题：

```
1. 存储什么？ → 参数（权重、偏置）+ 子模块
2. 怎么访问？ → parameters() 迭代器
3. 怎么组合？ → 嵌套的树形结构
```

```
Module 的树形结构：

              Model (Module)
              /      \
         layer1    layer2
         /    \        \
      weight  bias    weight
        ↓      ↓        ↓
      Tensor Tensor   Tensor

parameters() 会递归遍历这棵树，收集所有 Tensor
```

### 实现

```python
# module.py
from collections import OrderedDict
from typing import Dict, Iterator, Tuple

class Module:
    """
    所有神经网络模块的基类

    类比：就像乐高积木的通用接口
    - 每个模块可以包含参数（权重）
    - 每个模块可以包含子模块（层）
    - 自动管理所有参数
    """

    def __init__(self):
        # 存储参数：{名字: Tensor}
        self._parameters: Dict[str, Tensor] = OrderedDict()
        # 存储子模块：{名字: Module}
        self._modules: Dict[str, Module] = OrderedDict()
        # 存储非参数张量（如 BatchNorm 的 running_mean）
        self._buffers: Dict[str, Tensor] = OrderedDict()
        # 训练/评估模式
        self.training = True

    def forward(self, *args, **kwargs) -> Tensor:
        """
        前向传播：子类必须实现

        类比：乐高积木的"功能"
        - Linear: y = x @ W + b
        - ReLU: y = max(0, x)
        - Conv2d: 卷积运算
        """
        raise NotImplementedError("子类必须实现 forward()")

    def __call__(self, *args, **kwargs) -> Tensor:
        """
        让模块可以像函数一样调用

        model(x) 实际上调用 model.forward(x)
        """
        return self.forward(*args, **kwargs)

    def register_parameter(self, name: str, param: Tensor) -> None:
        """注册一个参数"""
        if param is not None:
            self._parameters[name] = param

    def add_module(self, name: str, module: 'Module') -> None:
        """添加子模块"""
        self._modules[name] = module

    def parameters(self, recurse: bool = True) -> Iterator[Tensor]:
        """
        返回所有参数（递归）

        类比：打开所有乐高盒子，找出里面的积木

        用法：
            for p in model.parameters():
                p.grad = None
        """
        # 自己的参数
        for param in self._parameters.values():
            yield param

        # 递归获取子模块的参数
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

    def train(self, mode: bool = True) -> 'Module':
        """
        设置训练模式

        影响的行为：
        - Dropout: 训练时随机丢弃，评估时不丢弃
        - BatchNorm: 训练时更新统计量，评估时使用固定统计量
        """
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

---

## 3.3 实现 Linear 层：最基础的积木

### 为什么叫 "Linear"（线性）？

```
线性变换：y = Wx + b

"线性"的含义：
1. 输入翻倍 → 输出翻倍
2. 叠加性：f(a+b) = f(a) + f(b)

非线性的例子：
- ReLU: max(0, x) —— 截断了负数，不是线性的
- Sigmoid: 1/(1+e^-x) —— 有上限，不是线性的
```

### 实现

```python
# linear.py
import math
import numpy as np

class Linear(Module):
    """
    全连接层（线性层）

    数学：y = x @ W.T + b

    类比：一个"变换器"
    - 输入 784 维 → 输出 256 维
    - 就像把一张 28×28 的图片压缩成 256 个特征
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        参数:
            in_features: 输入特征数
            out_features: 输出特征数
            bias: 是否使用偏置
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 权重初始化（Kaiming/He 初始化）
        # 为什么这样初始化？为了保持前向/反向传播时方差稳定
        scale = math.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(out_features, in_features) * scale,
            requires_grad=True
        )
        self.register_parameter('weight', self.weight)

        # 偏置初始化为 0
        if bias:
            self.bias = Tensor(
                np.zeros(out_features, dtype=np.float32),
                requires_grad=True
            )
            self.register_parameter('bias', self.bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播

        x: (batch_size, in_features)
        W: (out_features, in_features)
        b: (out_features,)

        输出: (batch_size, out_features)
        """
        # y = x @ W.T + b
        output = x @ self.weight.T
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        """打印信息"""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


# ===== 测试 =====
layer = Linear(784, 256)
print(layer)  # Linear(in_features=784, out_features=256, bias=True)

x = Tensor.randn((32, 784))  # 32 个样本，每个 784 维
y = layer(x)
print(f"输入形状: {x.shape}")   # (32, 784)
print(f"输出形状: {y.shape}")   # (32, 256)
print(f"权重形状: {layer.weight.shape}")  # (256, 784)
```

### 维度可视化

```
Linear(784, 256) 的维度变化：

输入 x:          权重 W:           输出 y:
(32, 784)   @   (256, 784).T  +   (256,)
   ↓               ↓               ↓
(32, 784)   @   (784, 256)   +   (256,)
   ↓               ↓               ↓
         (32, 256)  +  广播  =  (32, 256)

记忆：
  输入最后一维 = 权重最后一维（in_features）
  输出最后一维 = 权重第一维（out_features）
```

---

## 3.4 实现 Sequential：按顺序拼积木

### 概念

```
Sequential = 把多个层串起来

数据流向：
x → Layer1 → Layer2 → Layer3 → output

类比：流水线
  原料 → 加工1 → 加工2 → 加工3 → 成品
```

### 实现

```python
class Sequential(Module):
    """
    顺序容器：按顺序执行多个层

    类比：乐高积木一条龙
    """

    def __init__(self, *layers):
        """
        参数:
            *layers: 要串联的层
        """
        super().__init__()
        for i, layer in enumerate(layers):
            self.add_module(str(i), layer)

    def forward(self, x: Tensor) -> Tensor:
        """按顺序执行每一层"""
        for layer in self._modules.values():
            x = layer(x)
        return x

    def __getitem__(self, idx: int) -> Module:
        """通过索引获取层"""
        return list(self._modules.values())[idx]

    def __len__(self) -> int:
        """返回层数"""
        return len(self._modules)


# ===== 测试 =====
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10)
)

x = Tensor.randn((32, 784))
output = model(x)
print(f"输出形状: {output.shape}")  # (32, 10)
print(f"模型层数: {len(model)}")    # 5
```

---

## 3.5 使用 Module 构建模型

### 统计参数数量

```python
def count_parameters(model: Module) -> int:
    """统计模型参数数量"""
    return sum(p.data.size for p in model.parameters())


model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

print(f"总参数: {count_parameters(model):,}")
# Linear(784,256): 784*256 + 256 = 200,960
# Linear(256,10):  256*10 + 10 = 2,570
# 总计: 203,530
```

### 查看模型结构

```python
def print_model(model: Module, indent: int = 0) -> None:
    """打印模型结构"""
    prefix = "  " * indent
    for name, module in model._modules.items():
        print(f"{prefix}{name}: {module.__class__.__name__}")
        if module._modules:
            print_model(module, indent + 1)
        else:
            for param_name, param in module._parameters.items():
                print(f"{prefix}  {param_name}: {param.shape}")


print_model(model)
# 0: Linear
#   weight: (256, 784)
#   bias: (256,)
# 1: ReLU
# 2: Linear
#   weight: (10, 256)
#   bias: (10,)
```

### 自定义模块

```python
class MLP(Module):
    """
    自定义多层感知机

    展示如何组合多个层
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        # 创建子模块
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, output_dim)
        self.activation = ReLU()

        # 注册子模块（这样 parameters() 才能找到它们）
        self.add_module('fc1', self.fc1)
        self.add_module('fc2', self.fc2)
        self.add_module('activation', self.activation)

    def forward(self, x: Tensor) -> Tensor:
        """前向传播：定义数据如何流动"""
        x = self.fc1(x)          # 第一层
        x = self.activation(x)   # 激活
        x = self.fc2(x)          # 第二层
        return x


# 使用
mlp = MLP(784, 256, 10)
x = Tensor.randn((32, 784))
output = mlp(x)
print(f"输出形状: {output.shape}")  # (32, 10)
```

---

## 3.6 保存和加载模型

### state_dict：模型的状态快照

```python
class Module:
    # ... 前面的代码 ...

    def state_dict(self) -> Dict[str, np.ndarray]:
        """
        获取模型状态

        类比：给模型拍个"快照"
        返回：{参数名: 参数值} 的字典
        """
        state = OrderedDict()

        # 收集所有参数
        for name, param in self.named_parameters():
            state[name] = param.data.copy()

        # 收集缓冲区（如 BatchNorm 的 running_mean）
        for name, buffer in self._buffers.items():
            state[name] = buffer.data.copy()

        return state

    def load_state_dict(self, state_dict: Dict[str, np.ndarray], strict: bool = True):
        """
        加载模型状态

        类比：从"快照"恢复模型
        """
        own_state = self.state_dict()

        for name, param_data in state_dict.items():
            if name in own_state:
                # 找到对应的参数，赋值
                own_state[name][:] = param_data
            elif strict:
                raise KeyError(f"Unexpected key: {name}")

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if missing:
                raise KeyError(f"Missing keys: {missing}")
```

### 保存/加载示例

```python
import numpy as np

# 训练好的模型
model = MLP(784, 256, 10)

# ===== 保存 =====
state = model.state_dict()
np.savez('model.npz', **state)
print("模型已保存到 model.npz")

# 查看保存的内容
for name, data in state.items():
    print(f"  {name}: {data.shape}")
# fc1.weight: (256, 784)
# fc1.bias: (256,)
# fc2.weight: (10, 256)
# fc2.bias: (10,)

# ===== 加载 =====
new_model = MLP(784, 256, 10)
loaded = dict(np.load('model.npz'))
new_model.load_state_dict(loaded)
print("模型已从 model.npz 加载")
```

---

## 3.7 训练/评估模式

### 为什么需要不同模式？

```
训练模式 vs 评估模式的区别：

Dropout:
  训练：随机丢弃 50% 神经元
  评估：不丢弃（全部参与）

BatchNorm:
  训练：用当前批次计算均值/方差，并更新 running_mean
  评估：使用训练时的 running_mean（固定）
```

### 使用

```python
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Dropout(0.5),  # 训练时丢弃50%
    Linear(256, 10)
)

# 训练时
model.train()
for x, y in train_loader:
    output = model(x)  # Dropout 会随机丢弃
    loss = loss_fn(output, y)
    ...

# 评估时
model.eval()
for x, y in test_loader:
    output = model(x)  # Dropout 不丢弃
    ...
```

---

## 3.8 常见陷阱

### 陷阱1：忘记注册子模块

```python
class BadModel(Module):
    def __init__(self):
        super().__init__()
        self.layer = Linear(10, 10)  # 忘记 add_module！

    def forward(self, x):
        return self.layer(x)

model = BadModel()
print(list(model.parameters()))  # [] 空的！子模块参数没被追踪

# 正确做法
class GoodModel(Module):
    def __init__(self):
        super().__init__()
        self.layer = Linear(10, 10)
        self.add_module('layer', self.layer)  # 注册！
```

### 陷阱2：参数没设置 requires_grad

```python
class MyLayer(Module):
    def __init__(self):
        super().__init__()
        self.weight = Tensor(np.randn(10, 10))  # 忘了 requires_grad=True！
        self.register_parameter('weight', self.weight)

layer = MyLayer()
for p in layer.parameters():
    print(p.requires_grad)  # False！无法训练

# 正确做法
self.weight = Tensor(np.randn(10, 10), requires_grad=True)
```

### 陷阱3：多次 backward 没清零

```python
model = Linear(10, 10)
x = Tensor.randn((32, 10))

for epoch in range(3):
    y = model(x)
    loss = y.sum()
    loss.backward()
    # 忘了 model.zero_grad()！

# 梯度会累积：第一次 1x，第二次 2x，第三次 3x
# 正确做法：每次 backward 前清零
model.zero_grad()
loss.backward()
```

---

## 3.9 完整代码

```python
# module.py
from collections import OrderedDict
from typing import Dict, Iterator, Tuple
import numpy as np

class Module:
    """所有神经网络模块的基类"""

    def __init__(self):
        self._parameters: Dict[str, Tensor] = OrderedDict()
        self._modules: Dict[str, Module] = OrderedDict()
        self._buffers: Dict[str, Tensor] = OrderedDict()
        self.training = True

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def register_parameter(self, name: str, param: Tensor) -> None:
        if param is not None:
            self._parameters[name] = param

    def register_buffer(self, name: str, tensor: Tensor) -> None:
        self._buffers[name] = tensor

    def add_module(self, name: str, module: Module) -> None:
        self._modules[name] = module

    def parameters(self, recurse: bool = True) -> Iterator[Tensor]:
        for p in self._parameters.values():
            yield p
        if recurse:
            for m in self._modules.values():
                yield from m.parameters()

    def named_parameters(self, recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        for name, p in self._parameters.items():
            yield name, p
        if recurse:
            for m_name, m in self._modules.items():
                for p_name, p in m.named_parameters():
                    yield f"{m_name}.{p_name}", p

    def train(self, mode: bool = True) -> Module:
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    def zero_grad(self) -> None:
        for p in self.parameters():
            if p.grad is not None:
                p.grad = None

    def state_dict(self) -> Dict[str, np.ndarray]:
        state = OrderedDict()
        for name, p in self.named_parameters():
            state[name] = p.data.copy()
        return state

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        for name, p in self.named_parameters():
            if name in state_dict:
                p.data = state_dict[name].copy()


class Sequential(Module):
    """顺序容器"""

    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            self.add_module(str(i), layer)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._modules.values():
            x = layer(x)
        return x
```

---

## 3.10 练习

### 基础练习

1. **实现 `Module.__repr__()`**：打印模块结构

2. **实现 `children()`**：只返回直接子模块（不递归）

3. **实现 `named_children()`**：返回直接子模块及名称

### 进阶练习

4. **实现 `Module.apply(fn)`**：对每个子模块应用函数
   ```python
   # 用法：初始化所有权重
   def init_weights(m):
       if isinstance(m, Linear):
           m.weight.data = np.randn(*m.weight.shape) * 0.01

   model.apply(init_weights)
   ```

5. **实现 `named_buffers()`**：返回所有缓冲区

### 挑战

6. **实现 `ModuleList`**：像列表一样存储模块
7. **实现 `ModuleDict`**：像字典一样存储模块

---

## 一句话总结

| 概念 | 一句话 |
|------|--------|
| Module | 神经网络的乐高积木接口 |
| parameters() | 自动收集所有可训练参数 |
| Sequential | 按顺序串联多个层 |
| state_dict | 模型参数的快照，用于保存/加载 |
| train()/eval() | 切换训练/评估模式 |

---

## 下一章

现在我们有了模块系统！

但是，只有线性层的网络只能学习线性关系。为了让神经网络能够学习复杂的非线性关系，我们需要**激活函数**。

→ [第四章：激活函数](04-activation.md)

```python
# 预告：下一章你将实现
class ReLU(Module):
    def forward(self, x):
        return x.relu()  # max(0, x)

# 激活函数让网络能够"弯曲"决策边界
```
