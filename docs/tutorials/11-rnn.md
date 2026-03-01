# 教程 11：循环神经网络 (RNN/LSTM/GRU)

## 目录

1. [概述](#概述)
2. [RNN 基础](#rnn-基础)
3. [LSTM 详解](#lstm-详解)
4. [GRU 详解](#gru-详解)
5. [Cell 版本 vs 完整版本](#cell-版本-vs-完整版本)
6. [双向 RNN](#双向-rnn)
7. [使用示例](#使用示例)
8. [总结](#总结)

---

## 概述

循环神经网络（Recurrent Neural Network, RNN）是处理**序列数据**的核心架构。与传统的前馈网络不同，RNN 具有**记忆**能力，能够捕捉序列中的时序依赖关系。

nanotorch 实现了三种主要的循环神经网络：
- **RNN**：基础循环神经网络
- **LSTM**：长短期记忆网络
- **GRU**：门控循环单元

---

## RNN 基础

### 核心思想

RNN 通过隐藏状态（hidden state）在时间步之间传递信息：

```
时刻 t 的计算:
h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
y_t = h_t  (对于基础 RNN)
```

### 信息流

```
输入序列:  x_1, x_2, x_3, ..., x_T
            ↓    ↓    ↓         ↓
隐藏状态:  h_1→ h_2→ h_3→ ...→ h_T
            ↓    ↓    ↓         ↓
输出:      y_1, y_2, y_3, ..., y_T
```

### RNNCell 实现

```python
# nanotorch/nn/rnn.py

class RNNCell(Module):
    """单步 RNN 单元。
    
    h' = tanh(W_{ih} @ x + b_{ih} + W_{hh} @ h + b_{hh})
    
    Args:
        input_size: 输入特征数
        hidden_size: 隐藏状态大小
        bias: 是否使用偏置
        nonlinearity: 激活函数 ('tanh' 或 'relu')
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        
        # 输入到隐藏层的权重
        self.weight_ih = Tensor(
            np.random.randn(input_size, hidden_size).astype(np.float32)
            * math.sqrt(1.0 / hidden_size),
            requires_grad=True,
        )
        
        # 隐藏层到隐藏层的权重
        self.weight_hh = Tensor(
            np.random.randn(hidden_size, hidden_size).astype(np.float32)
            * math.sqrt(1.0 / hidden_size),
            requires_grad=True,
        )
        
        # 偏置
        self.bias_ih = None
        self.bias_hh = None
        if bias:
            self.bias_ih = Tensor(np.zeros(hidden_size), requires_grad=True)
            self.bias_hh = Tensor(np.zeros(hidden_size), requires_grad=True)

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        """单步前向传播。
        
        Args:
            x: 当前时刻输入 (batch, input_size)
            h: 上一时刻隐藏状态 (batch, hidden_size)
        
        Returns:
            新的隐藏状态 (batch, hidden_size)
        """
        if h is None:
            h = Tensor(np.zeros((x.shape[0], self.hidden_size)))
        
        # 计算输入部分
        ih = x.matmul(self.weight_ih)
        if self.bias_ih is not None:
            ih = ih + self.bias_ih
        
        # 计算隐藏部分
        hh = h.matmul(self.weight_hh)
        if self.bias_hh is not None:
            hh = hh + self.bias_hh
        
        # 合并并应用激活函数
        out = ih + hh
        if self.nonlinearity == "tanh":
            out = out.tanh()
        elif self.nonlinearity == "relu":
            out = out.relu()
        
        return out
```

### 完整 RNN 实现

```python
class RNN(RNNBase):
    """多层 RNN 网络。
    
    Args:
        input_size: 输入特征数
        hidden_size: 隐藏状态大小
        num_layers: 层数
        nonlinearity: 'tanh' 或 'relu'
        bias: 是否使用偏置
        batch_first: 输入是否为 (batch, seq, feature) 格式
        dropout: 层间 dropout 概率
        bidirectional: 是否双向
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__(
            mode="RNN",
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            nonlinearity=nonlinearity,
        )
```

### RNN 的问题：梯度消失

基础 RNN 在长序列上存在**梯度消失**问题：

```
梯度通过时间步传播:
∂L/∂h_t = ∂L/∂h_T × ∏(∂h_i/∂h_{i-1})

当序列很长时，连乘的梯度会指数级衰减或爆炸。
```

---

## LSTM 详解

### 核心创新

LSTM（Long Short-Term Memory）通过**门控机制**解决梯度消失问题：
- **遗忘门（Forget Gate）**：决定丢弃哪些信息
- **输入门（Input Gate）**：决定更新哪些信息
- **输出门（Output Gate）**：决定输出哪些信息
- **细胞状态（Cell State）**：长期记忆

### LSTM 公式

```
f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)     遗忘门
i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)     输入门
o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)     输出门
g_t = tanh(W_g @ [h_{t-1}, x_t] + b_g)  候选值

c_t = f_t * c_{t-1} + i_t * g_t         更新细胞状态
h_t = o_t * tanh(c_t)                   更新隐藏状态
```

### LSTMCell 实现

```python
class LSTMCell(Module):
    """LSTM 单元。
    
    Args:
        input_size: 输入特征数
        hidden_size: 隐藏状态大小
        bias: 是否使用偏置
    
    Shape:
        - Input: (batch, input_size)
        - Hidden: ((batch, hidden_size), (batch, hidden_size))  # (h, c)
        - Output: ((batch, hidden_size), (batch, hidden_size))  # (h', c')
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 四个门的参数合并在一起 (4 * hidden_size)
        gate_size = 4 * hidden_size
        
        self.weight_ih = Tensor(
            np.random.randn(input_size, gate_size).astype(np.float32)
            * math.sqrt(1.0 / hidden_size),
            requires_grad=True,
        )
        self.weight_hh = Tensor(
            np.random.randn(hidden_size, gate_size).astype(np.float32)
            * math.sqrt(1.0 / hidden_size),
            requires_grad=True,
        )
        
        self.bias_ih = None
        self.bias_hh = None
        if bias:
            self.bias_ih = Tensor(np.zeros(gate_size), requires_grad=True)
            self.bias_hh = Tensor(np.zeros(gate_size), requires_grad=True)

    def forward(
        self,
        x: Tensor,
        state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """单步前向传播。
        
        Args:
            x: 当前输入 (batch, input_size)
            state: (h, c) 元组
        
        Returns:
            (h_new, c_new) 元组
        """
        if state is None:
            h = Tensor(np.zeros((x.shape[0], self.hidden_size)))
            c = Tensor(np.zeros((x.shape[0], self.hidden_size)))
        else:
            h, c = state
        
        # 计算所有门
        gates = x.matmul(self.weight_ih) + h.matmul(self.weight_hh)
        if self.bias_ih is not None:
            gates = gates + self.bias_ih
        if self.bias_hh is not None:
            gates = gates + self.bias_hh
        
        # 分割四个门
        i = gates[:, :self.hidden_size].sigmoid()           # 输入门
        f = gates[:, self.hidden_size:2*self.hidden_size].sigmoid()  # 遗忘门
        g = gates[:, 2*self.hidden_size:3*self.hidden_size].tanh()    # 候选值
        o = gates[:, 3*self.hidden_size:].sigmoid()         # 输出门
        
        # 更新状态
        c_new = f * c + i * g
        h_new = o * c_new.tanh()
        
        return h_new, c_new
```

### 完整 LSTM

```python
from nanotorch.nn import LSTM

lstm = LSTM(
    input_size=64,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
    bidirectional=True
)

x = Tensor.randn((32, 10, 64))  # (batch, seq_len, input_size)
output, (h_n, c_n) = lstm(x)

print(output.shape)  # (32, 10, 256)  # 256 = 128 * 2 (bidirectional)
print(h_n.shape)     # (4, 32, 128)   # 4 = 2 layers * 2 directions
```

---

## GRU 详解

### 核心思想

GRU（Gated Recurrent Unit）是 LSTM 的简化版本，只有两个门：
- **重置门（Reset Gate）**：控制如何组合新输入和之前的记忆
- **更新门（Update Gate）**：控制保留多少之前的隐藏状态

### GRU 公式

```
r_t = σ(W_r @ [h_{t-1}, x_t])      重置门
z_t = σ(W_z @ [h_{t-1}, x_t])      更新门
n_t = tanh(W_n @ [r_t * h_{t-1}, x_t])  候选隐藏状态

h_t = (1 - z_t) * n_t + z_t * h_{t-1}   新隐藏状态
```

### GRUCell 实现

```python
class GRUCell(Module):
    """GRU 单元。
    
    Shape:
        - Input: (batch, input_size)
        - Hidden: (batch, hidden_size)
        - Output: (batch, hidden_size)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 三个门的参数 (3 * hidden_size)
        self.weight_ih = Tensor(
            np.random.randn(input_size, 3 * hidden_size).astype(np.float32)
            * math.sqrt(1.0 / hidden_size),
            requires_grad=True,
        )
        self.weight_hh = Tensor(
            np.random.randn(hidden_size, 3 * hidden_size).astype(np.float32)
            * math.sqrt(1.0 / hidden_size),
            requires_grad=True,
        )
        
        # bias 初始化...

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        if h is None:
            h = Tensor(np.zeros((x.shape[0], self.hidden_size)))
        
        # 计算门
        gi = x.matmul(self.weight_ih)
        gh = h.matmul(self.weight_hh)
        # ... 加 bias
        
        # 分割三个部分
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        
        # 计算门
        r = (i_r + h_r).sigmoid()  # 重置门
        z = (i_z + h_z).sigmoid()  # 更新门
        n = (i_n + r * h_n).tanh() # 候选隐藏状态
        
        # 更新隐藏状态
        h_new = (1 - z) * n + z * h
        
        return h_new
```

### LSTM vs GRU

| 特性 | LSTM | GRU |
|------|------|-----|
| 门数量 | 3 | 2 |
| 状态数量 | 2 (h, c) | 1 (h) |
| 参数量 | 更多 | 更少 |
| 计算速度 | 较慢 | 较快 |
| 表达能力 | 较强 | 相似 |
| 训练难度 | 较难 | 较易 |

---

## Cell 版本 vs 完整版本

### Cell 版本

用于**手动控制**每个时间步：

```python
from nanotorch.nn import LSTMCell

cell = LSTMCell(input_size=64, hidden_size=128)

h = None
c = None
outputs = []

for t in range(seq_len):
    x_t = x[:, t, :]  # 当前时刻输入
    h, c = cell(x_t, (h, c))
    outputs.append(h)

output = Tensor(np.stack([o.data for o in outputs], axis=1))
```

### 完整版本

自动处理整个序列：

```python
from nanotorch.nn import LSTM

lstm = LSTM(input_size=64, hidden_size=128, batch_first=True)

# 一次性处理整个序列
output, (h_n, c_n) = lstm(x)
```

---

## 双向 RNN

### 原理

双向 RNN 同时从前向后和从后向前处理序列：

```
前向:  h_1 → h_2 → h_3 → ... → h_T
后向:  h'_1 ← h'_2 ← h'_3 ← ... ← h'_T

输出: [h_1, h'_1], [h_2, h'_2], ..., [h_T, h'_T]
```

### 使用

```python
from nanotorch.nn import LSTM

# 双向 LSTM
bi_lstm = LSTM(
    input_size=64,
    hidden_size=128,
    bidirectional=True,  # 开启双向
    batch_first=True
)

x = Tensor.randn((32, 10, 64))
output, (h_n, c_n) = bi_lstm(x)

print(output.shape)  # (32, 10, 256)  # 256 = 128 * 2 directions
```

---

## 使用示例

### 文本分类

```python
from nanotorch import Tensor
from nanotorch.nn import Embedding, LSTM, Linear

class TextClassifier:
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
    
    def __call__(self, x):
        # x: (batch, seq_len) 整数索引
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        output, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        # h_n shape: (num_layers * 2, batch, hidden_dim)
        # 取最后一层的双向输出
        h_forward = h_n[-2]  # 前向最后层
        h_backward = h_n[-1]  # 后向最后层
        h_concat = Tensor(np.concatenate([h_forward.data, h_backward.data], axis=-1))
        
        return self.fc(h_concat)
    
    def parameters(self):
        return self.embedding.parameters() + self.lstm.parameters() + self.fc.parameters()

# 使用
model = TextClassifier(vocab_size=10000, embed_dim=128, hidden_dim=256, num_classes=5)
x = Tensor(np.random.randint(0, 10000, (32, 50)))  # (batch, seq_len)
logits = model(x)
```

### 序列标注（NER）

```python
class SequenceTagger:
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.lstm = LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = Linear(hidden_dim * 2, num_tags)
    
    def __call__(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)  # output: (batch, seq_len, hidden_dim * 2)
        
        # 对每个时间步进行分类
        batch, seq_len, _ = output.shape
        output = output.reshape(batch * seq_len, -1)
        logits = self.fc(output)
        return logits.reshape(batch, seq_len, -1)
```

### 语言模型

```python
class LanguageModel:
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.lstm = LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = Linear(hidden_dim, vocab_size)
    
    def __call__(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        logits = self.fc(output)
        return logits
    
    def generate(self, start_token, max_len=100):
        """自回归生成文本"""
        tokens = [start_token]
        h = None
        c = None
        
        for _ in range(max_len):
            x = Tensor([[tokens[-1]]])
            x = self.embedding(x)
            output, (h, c) = self.lstm(x, (h, c) if h else None)
            logits = self.fc(output[:, -1, :])
            next_token = np.argmax(logits.data, axis=-1)[0]
            tokens.append(next_token)
        
        return tokens
```

---

## 总结

本教程介绍了 nanotorch 中的循环神经网络实现：

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| **RNN** | 简单，但梯度消失 | 短序列 |
| **LSTM** | 三个门，长期记忆 | 长序列，复杂任务 |
| **GRU** | 两个门，更简单 | 长序列，效率优先 |

### 关键要点

1. **Cell 版本** 适合需要精细控制的场景
2. **完整版本** 适合直接处理整个序列
3. **双向 RNN** 可以同时利用前后文信息
4. **LSTM/GRU** 通过门控机制解决梯度消失

### 下一步

在 [教程 12：Transformer](12-transformer.md) 中，我们将学习如何实现 Transformer 架构，这是现代 NLP 的基础。

---

**参考资源**：
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Learning Phrase Representations using RNN Encoder-Decoder (GRU)](https://arxiv.org/abs/1406.1078)
