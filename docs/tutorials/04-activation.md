# 第四章：激活函数

激活函数为神经网络引入非线性，使其能够学习复杂的模式。

## 4.1 为什么需要激活函数？

没有激活函数的多层网络等价于单层：

```
y = W2 @ (W1 @ x) = (W2 @ W1) @ x = W @ x
```

激活函数打破这种线性关系，让网络能够拟合任意函数。

## 4.2 实现激活函数模块

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

## 4.3 ReLU 详解

```
ReLU(x) = max(0, x)

当 x > 0: 输出 x, 梯度 1
当 x ≤ 0: 输出 0, 梯度 0
```

**优点**：
- 计算简单
- 缓解梯度消失
- 稀疏激活

**缺点**：
- 死亡 ReLU 问题（梯度永远为 0）

```python
# Tensor 类中添加
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

解决死亡 ReLU 问题：

```python
class LeakyReLU(Module):
    """LeakyReLU: f(x) = x if x > 0 else α*x"""
    
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: Tensor) -> Tensor:
        return x.leaky_relu(self.negative_slope)


# Tensor 方法
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

Transformer 常用：

```python
class GELU(Module):
    """Gaussian Error Linear Unit"""
    
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


# GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
def gelu(self):
    # 近似实现
    cdf = 0.5 * (1 + np.tanh(
        np.sqrt(2 / np.pi) * (self.data + 0.044715 * self.data ** 3)
    ))
    out = Tensor(self.data * cdf, _children=(self,), _op='gelu')
    
    def _backward():
        if self.requires_grad:
            x = self.data
            grad = cdf + x * 0.5 * (1 - np.tanh(...) ** 2) * ...  # 完整导数
            self.grad = (self.grad if self.grad is not None else 0) + grad * out.grad
    
    out._backward = _backward
    return out
```

## 4.6 Softmax 详解

```python
def softmax(self, dim=-1):
    # 数值稳定性：减去最大值
    shifted = self.data - np.max(self.data, axis=dim, keepdims=True)
    exp_x = np.exp(shifted)
    out = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
    
    result = Tensor(out, _children=(self,), _op='softmax')
    
    def _backward():
        if self.requires_grad:
            # Softmax Jacobian: ∂softmax_i/∂x_j = softmax_i * (δ_ij - softmax_j)
            # 简化为: grad_input = softmax * (grad_output - sum(grad_output * softmax))
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

### Softmax 导数推导

设 `s = softmax(x)`，`L` 是损失

```
∂L/∂x_i = Σ_j (∂L/∂s_j * ∂s_j/∂x_i)

∂s_j/∂x_i = s_j * (δ_ij - s_i)

当 i = j: ∂s_i/∂x_i = s_i * (1 - s_i)
当 i ≠ j: ∂s_j/∂x_i = -s_i * s_j
```

## 4.7 使用示例

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

## 4.8 激活函数对比

| 函数 | 公式 | 范围 | 梯度范围 | 常用场景 |
|------|------|------|----------|----------|
| ReLU | max(0, x) | [0, ∞) | {0, 1} | 隐藏层 |
| LeakyReLU | max(αx, x) | (-∞, ∞) | {α, 1} | 隐藏层 |
| Sigmoid | 1/(1+e⁻ˣ) | (0, 1) | (0, 0.25) | 二分类输出 |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | (0, 1) | RNN |
| Softmax | eˣᵢ/Σeˣⱼ | (0, 1) | 复杂 | 多分类输出 |
| GELU | x*Φ(x) | (-∞, ∞) | 复杂 | Transformer |

## 4.9 练习

1. **实现 ELU**：`f(x) = x if x > 0 else α(eˣ - 1)`

2. **实现 Swish/SiLU**：`f(x) = x * sigmoid(x)`

3. **实现 PReLU**：可学习的 `α` 参数

4. **画出各激活函数曲线**

## 下一章

下一章，我们将实现**损失函数**，定义训练目标。

→ [第五章：损失函数](05-loss.md)
