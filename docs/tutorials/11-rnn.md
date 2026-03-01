# 第十一章：循环神经网络

## 你在看一部悬疑电影...

第一幕，主角捡到一把钥匙。你没多想。

第二幕，主角发现一个上锁的箱子。你开始期待。

第三幕，主角用钥匙打开箱子，里面是——你屏住呼吸。

你之所以紧张，是因为你**记住了第一幕的钥匙**。如果每一幕都是独立的，你根本不会在意箱子能不能打开。

**记忆，是理解序列的关键。**

但普通的神经网络没有记忆。它像金鱼，只活在当下。给它一篇文章，它看完第一个字，忘了第一个字；看完第二字，忘了第二个字。

RNN（循环神经网络）给了网络一份"记忆"。

```
普通网络：
  看到"我" → 忘了
  看到"爱" → 忘了
  看到"你" → 输出"你是什么？"

RNN：
  看到"我" → 记住：主语是我
  看到"爱" → 记住：主语爱我
  看到"你" → 理解：我爱你的完整句子
```

**RNN，让机器拥有了时间感。** 它不再只活在当下，而是能理解过去、把握现在、预见未来。

---

## 11.1 为什么需要 RNN？

### 问题：普通网络没有记忆

```
普通网络处理句子：

"我 爱 北京 天安门"
 ↓   ↓    ↓     ↓
独立处理每个词

问题：
  - 不知道"爱"的主语是"我"
  - 不知道"天安门"在北京
  - 每个词都是孤立的
```

### 解决：引入隐藏状态

```
RNN 处理句子：

时刻1:  "我" + 隐藏状态0 → 输出1 → 隐藏状态1
时刻2: "爱" + 隐藏状态1 → 输出2 → 隐藏状态2
时刻3: "北京" + 隐藏状态2 → 输出3 → 隐藏状态3
...

隐藏状态 = 记忆，传递之前的信息
```

---

## 11.2 RNN 基础

### 核心公式

```
h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)
y_t = W_hy @ h_t + b_y

解释：
  x_t = 当前输入
  h_{t-1} = 上一个时刻的隐藏状态（记忆）
  h_t = 当前隐藏状态（更新后的记忆）
  y_t = 当前输出

tanh：把值压缩到 (-1, 1)，防止数值爆炸
```

### 图解 RNN

```
时间展开：

       x_1      x_2      x_3
        ↓        ↓        ↓
     ┌─────┐  ┌─────┐  ┌─────┐
h_0→│ RNN │→│ RNN │→│ RNN │→ h_3
     └─────┘  └─────┘  └─────┘
        ↓        ↓        ↓
       y_1      y_2      y_3

每个 RNN 单元共享同一组权重！
```

### RNNCell 实现

```python
class RNNCell(Module):
    """
    单步 RNN 单元

    类比：
      - 输入：当前看到的画面
      - 隐藏状态：之前的记忆
      - 输出：更新后的记忆
    """

    def __init__(
        self,
        input_size: int,    # 输入维度
        hidden_size: int,   # 隐藏状态维度
        bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 输入到隐藏层的权重
        self.weight_ih = Tensor(
            np.random.randn(input_size, hidden_size).astype(np.float32) * 0.1,
            requires_grad=True,
        )

        # 隐藏层到隐藏层的权重（记忆传递）
        self.weight_hh = Tensor(
            np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.1,
            requires_grad=True,
        )

        if bias:
            self.bias_ih = Tensor(np.zeros(hidden_size), requires_grad=True)
            self.bias_hh = Tensor(np.zeros(hidden_size), requires_grad=True)

    def forward(self, x: Tensor, h: Tensor = None) -> Tensor:
        """
        单步前向传播

        Args:
            x: 当前输入 (batch, input_size)
            h: 上一个隐藏状态 (batch, hidden_size)

        Returns:
            新的隐藏状态
        """
        if h is None:
            h = Tensor(np.zeros((x.shape[0], self.hidden_size)))

        # 计算新隐藏状态
        ih = x.matmul(self.weight_ih)
        hh = h.matmul(self.weight_hh)

        if hasattr(self, 'bias_ih'):
            ih = ih + self.bias_ih
            hh = hh + self.bias_hh

        # tanh 激活
        h_new = (ih + hh).tanh()

        return h_new
```

### 使用

```python
# 创建 RNN 单元
cell = RNNCell(input_size=64, hidden_size=128)

# 手动循环处理序列
h = None
outputs = []

for t in range(seq_len):
    x_t = x[:, t, :]  # 当前时刻输入
    h = cell(x_t, h)   # 更新隐藏状态
    outputs.append(h)

# outputs 是所有时刻的隐藏状态
```

---

## 11.3 RNN 的问题：梯度消失

### 问题

```
RNN 的梯度传播：

∂L/∂h_1 = ∂L/∂h_T × ∂h_T/∂h_{T-1} × ... × ∂h_2/∂h_1

问题：连续乘很多次！

如果每个梯度 < 1：
  0.9 × 0.9 × 0.9 × ... × 0.9 (100次) ≈ 0.00003

梯度消失 → 早期的信息学不到
```

### 生活类比

```
传话游戏：

原话： "明天下午三点在老地方见面"
第1人： "明天三点老地方见"
第2人： "明天三点见面"
第3人： "明天见面"
...
第10人："？？？"（忘了）

信息在传递中逐渐丢失
```

---

## 11.4 LSTM：长期记忆

### 核心思想

```
LSTM = Long Short-Term Memory

引入"细胞状态" c_t 作为长期记忆
用"门"控制信息的流动

三个门：
  - 遗忘门：决定忘记哪些旧信息
  - 输入门：决定记住哪些新信息
  - 输出门：决定输出哪些信息
```

### 图解 LSTM

```
                c_{t-1} ─────────────────────→ c_t
                   │                    ↑
                   │    ┌─────────────┐ │
                   └──→ │   遗忘门    │─┘
                        │   输入门    │──→ h_t
            h_{t-1} ──→ │   输出门    │──→ 输出
                ↑       └─────────────┘
                │             ↑
            x_t ─────────────┘
```

### LSTM 公式

```
遗忘门：f_t = sigmoid(W_f @ [h_{t-1}, x_t])     "要忘记多少"
输入门：i_t = sigmoid(W_i @ [h_{t-1}, x_t])     "要记住多少"
候选值：g_t = tanh(W_g @ [h_{t-1}, x_t])        "新信息"
输出门：o_t = sigmoid(W_o @ [h_{t-1}, x_t])     "要输出多少"

细胞状态：c_t = f_t * c_{t-1} + i_t * g_t      "更新长期记忆"
隐藏状态：h_t = o_t * tanh(c_t)                "更新短期记忆"
```

### LSTMCell 实现

```python
class LSTMCell(Module):
    """
    LSTM 单元

    类比：
      - c (细胞状态) = 长期记忆（记很久以前的事）
      - h (隐藏状态) = 短期记忆（记最近的事）
      - 门 = 大脑的控制机制
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # 四个门合并在一起计算（效率更高）
        gate_size = 4 * hidden_size

        self.weight_ih = Tensor(
            np.random.randn(input_size, gate_size).astype(np.float32) * 0.1,
            requires_grad=True,
        )
        self.weight_hh = Tensor(
            np.random.randn(hidden_size, gate_size).astype(np.float32) * 0.1,
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(gate_size), requires_grad=True)

    def forward(self, x: Tensor, state: tuple = None):
        """
        Args:
            x: 输入 (batch, input_size)
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
        gates = x.matmul(self.weight_ih) + h.matmul(self.weight_hh) + self.bias

        # 分割四个门
        i = gates[:, :self.hidden_size].sigmoid()                    # 输入门
        f = gates[:, self.hidden_size:2*self.hidden_size].sigmoid()  # 遗忘门
        g = gates[:, 2*self.hidden_size:3*self.hidden_size].tanh()   # 候选值
        o = gates[:, 3*self.hidden_size:].sigmoid()                  # 输出门

        # 更新状态
        c_new = f * c + i * g        # 长期记忆：选择性遗忘 + 选择性记住
        h_new = o * c_new.tanh()     # 短期记忆：选择性输出

        return h_new, c_new
```

### 为什么 LSTM 不容易梯度消失？

```
细胞状态的更新：
  c_t = f_t * c_{t-1} + i_t * g_t

梯度传播：
  ∂c_t/∂c_{t-1} = f_t

如果 f_t ≈ 1，梯度可以直接传递！
遗忘门学会了什么时候该保留信息。
```

---

## 11.5 GRU：简化版 LSTM

### 核心思想

```
GRU = Gated Recurrent Unit

LSTM 的简化版本：
  - 只有 2 个门（LSTM 有 3 个）
  - 只有 1 个状态（LSTM 有 h 和 c）
  - 参数更少，训练更快
```

### GRU 公式

```
重置门：r_t = sigmoid(W_r @ [h_{t-1}, x_t])    "要重置多少"
更新门：z_t = sigmoid(W_z @ [h_{t-1}, x_t])    "要更新多少"
候选值：n_t = tanh(W_n @ [r_t * h_{t-1}, x_t]) "新信息"

新状态：h_t = (1-z_t) * n_t + z_t * h_{t-1}
```

### GRUCell 实现

```python
class GRUCell(Module):
    """
    GRU 单元

    比 LSTM 简单，但效果差不多
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # 3 个门
        gate_size = 3 * hidden_size

        self.weight_ih = Tensor(
            np.random.randn(input_size, gate_size).astype(np.float32) * 0.1,
            requires_grad=True,
        )
        self.weight_hh = Tensor(
            np.random.randn(hidden_size, gate_size).astype(np.float32) * 0.1,
            requires_grad=True,
        )

    def forward(self, x: Tensor, h: Tensor = None) -> Tensor:
        if h is None:
            h = Tensor(np.zeros((x.shape[0], self.hidden_size)))

        gates = x.matmul(self.weight_ih) + h.matmul(self.weight_hh)

        # 分割
        r = gates[:, :self.hidden_size].sigmoid()                     # 重置门
        z = gates[:, self.hidden_size:2*self.hidden_size].sigmoid()   # 更新门
        n = gates[:, 2*self.hidden_size:].tanh()                       # 候选值

        # 更新
        h_new = (1 - z) * n + z * h

        return h_new
```

---

## 11.6 LSTM vs GRU

| 特性 | LSTM | GRU |
|------|------|-----|
| 门数量 | 3个 | 2个 |
| 状态数量 | 2个 (h, c) | 1个 (h) |
| 参数量 | 多 | 少 30% |
| 计算速度 | 较慢 | 较快 |
| 表达能力 | 略强 | 相近 |
| 训练难度 | 较难 | 较易 |

```
选择建议：
  - 追求效果：LSTM
  - 追求速度：GRU
  - 不确定：先试 GRU
```

---

## 11.7 完整 RNN 层

### 封装循环

```python
class LSTM(Module):
    """
    完整的 LSTM 层

    自动处理整个序列，不用手动循环
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = False,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        # 创建多层 LSTM
        self.cells = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.cells.append(LSTMCell(in_size, hidden_size))

    def forward(self, x: Tensor, state: tuple = None):
        """
        Args:
            x: 输入序列
               batch_first=False: (seq_len, batch, input_size)
               batch_first=True:  (batch, seq_len, input_size)
            state: 初始状态 (h_0, c_0)

        Returns:
            output: 所有时刻的输出
            (h_n, c_n): 最后时刻的状态
        """
        if self.batch_first:
            # 转换为 (seq_len, batch, input_size)
            x = x.transpose(0, 1)

        seq_len, batch, _ = x.shape

        # 初始化状态
        if state is None:
            h = [Tensor(np.zeros((batch, self.hidden_size))) for _ in range(self.num_layers)]
            c = [Tensor(np.zeros((batch, self.hidden_size))) for _ in range(self.num_layers)]
        else:
            h, c = state

        outputs = []

        # 逐时刻处理
        for t in range(seq_len):
            x_t = x[t]

            # 逐层处理
            for layer, cell in enumerate(self.cells):
                h[layer], c[layer] = cell(x_t, (h[layer], c[layer]))
                x_t = h[layer]

            outputs.append(h[-1])

        output = Tensor(np.stack([o.data for o in outputs], axis=0))

        return output, (h, c)
```

### 使用

```python
# 创建 LSTM
lstm = LSTM(
    input_size=64,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
    bidirectional=False
)

# 输入
x = Tensor.randn((32, 10, 64))  # (batch, seq_len, input_size)

# 前向传播
output, (h_n, c_n) = lstm(x)

print(output.shape)  # (32, 10, 128)
```

---

## 11.8 双向 RNN

### 原理

```
普通 RNN：只看过去
  我 → 爱 → 你 → ！

双向 RNN：同时看过去和未来
  前向：我 → 爱 → 你 → ！
  后向：我 ← 爱 ← 你 ← ！

合并：[前向h_t, 后向h_t]
```

### 使用场景

```
适合：需要完整上下文的任务
  - 机器翻译
  - 命名实体识别
  - 情感分析

不适合：实时生成任务
  - 实时语音识别（不能看未来）
  - 文本生成
```

### 使用

```python
# 双向 LSTM
bi_lstm = LSTM(
    input_size=64,
    hidden_size=128,
    bidirectional=True,  # 开启双向
    batch_first=True
)

x = Tensor.randn((32, 10, 64))
output, (h_n, c_n) = bi_lstm(x)

print(output.shape)  # (32, 10, 256)  # 256 = 128 * 2 方向
```

---

## 11.9 应用示例

### 文本分类

```python
class TextClassifier:
    """
    用 LSTM 做文本分类

    结构：Embedding → LSTM → 取最后隐藏状态 → 分类
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.lstm = LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = Linear(hidden_dim, num_classes)

    def __call__(self, x):
        # x: (batch, seq_len) 词索引
        x = self.embedding(x)        # (batch, seq_len, embed_dim)
        output, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]              # 最后一层的隐藏状态
        return self.fc(h_last)

    def parameters(self):
        return self.embedding.parameters() + self.lstm.parameters() + self.fc.parameters()
```

### 语言模型

```python
class LanguageModel:
    """
    用 LSTM 做语言模型

    预测下一个词
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.lstm = LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = Linear(hidden_dim, vocab_size)

    def __call__(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        logits = self.fc(output)  # 每个位置都预测下一个词
        return logits

    def generate(self, start_token, max_len=50):
        """生成文本"""
        tokens = [start_token]
        h, c = None, None

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

## 11.10 常见陷阱

### 陷阱1：忘记传递隐藏状态

```python
# 错误：每个时刻都重置
for t in range(seq_len):
    h = cell(x[t], None)  # 每次都重新初始化！

# 正确：传递隐藏状态
h = None
for t in range(seq_len):
    h = cell(x[t], h)
```

### 陷阱2：batch_first 混淆

```python
# LSTM 默认 batch_first=False
lstm = LSTM(64, 128)
x = Tensor.randn((32, 10, 64))  # (batch, seq, feature)

# 会出错！应该先转置或设置 batch_first=True
lstm = LSTM(64, 128, batch_first=True)
```

### 陷阱3：梯度裁剪

```python
# RNN 容易梯度爆炸，需要裁剪
from nanotorch.utils import clip_grad_norm_

loss.backward()
clip_grad_norm_(model.parameters(), max_norm=1.0)  # 重要！
optimizer.step()
```

---

## 11.11 一句话总结

| 概念 | 一句话 |
|------|--------|
| RNN | 有记忆的神经网络，记住历史信息 |
| 隐藏状态 | 传递信息的载体 |
| LSTM | 用门控解决长期记忆 |
| GRU | LSTM 的简化版，更快 |
| 双向 | 同时看过去和未来 |

---

## 下一章

现在我们学会了处理序列的 RNN！

下一章，我们将学习**Transformer** —— 现代NLP的基石，ChatGPT的核心。

→ [第十二章：Transformer](12-transformer.md)

```python
# 预告：下一章你将学到
attention = MultiheadAttention(embed_dim=512, num_heads=8)
# 用注意力机制替代循环
# 并行处理整个序列
```
