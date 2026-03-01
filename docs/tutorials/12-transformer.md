# 教程 12：Transformer 架构

## 目录

1. [概述](#概述)
2. [注意力机制](#注意力机制)
3. [多头注意力](#多头注意力)
4. [Transformer Encoder Layer](#transformer-encoder-layer)
5. [Transformer Decoder Layer](#transformer-decoder-layer)
6. [位置编码](#位置编码)
7. [完整 Transformer](#完整-transformer)
8. [使用示例](#使用示例)
9. [总结](#总结)

---

## 概述

Transformer 是 2017 年 Google 提出的革命性架构，它完全基于**注意力机制**，摒弃了传统的循环和卷积结构。

核心优势：
- **并行计算**：不像 RNN 需要顺序处理
- **长距离依赖**：注意力机制直接连接任意位置
- **可扩展性**：容易扩展到大规模模型

nanotorch 实现的组件：
- `MultiheadAttention`：多头自注意力
- `TransformerEncoderLayer`：编码器层
- `TransformerDecoderLayer`：解码器层
- `TransformerEncoder`：编码器堆叠
- `TransformerDecoder`：解码器堆叠

---

## 注意力机制

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

其中:
- $Q$ (Query): 查询矩阵 (batch, seq_len, d_k)
- $K$ (Key): 键矩阵 (batch, seq_len, d_k)
- $V$ (Value): 值矩阵 (batch, seq_len, d_v)
- $d_k$: Key 的维度

### 直观理解

```
注意力机制 = "查询"与"键"的相似度，加权"值"

问题: "苹果是什么颜色的？"
      Q: 当前词的查询向量
      K: 其他词的键向量（表示"我是关于什么的"）
      V: 其他词的值向量（表示"我包含什么信息"）

计算过程:
1. Q @ K^T: 计算每个词与当前词的相关性
2. softmax: 归一化为概率分布
3. @ V: 根据相关性加权求和
```

### 实现

```python
def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, 
    mask: Optional[Tensor] = None,
    dropout: Optional[float] = None
) -> Tuple[Tensor, Tensor]:
    """缩放点积注意力。
    
    Args:
        q: Query (batch, heads, seq_len, d_k)
        k: Key (batch, heads, seq_len, d_k)
        v: Value (batch, heads, seq_len, d_v)
        mask: 可选的掩码
        dropout: dropout 概率
    
    Returns:
        (output, attention_weights)
    """
    d_k = q.shape[-1]
    
    # 计算注意力分数
    scores = q.matmul(k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 应用掩码（用于 decoder 的因果注意力）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax 归一化
    attn_weights = scores.softmax(dim=-1)
    
    # 可选的 dropout
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    
    # 加权求和
    output = attn_weights.matmul(v)
    
    return output, attn_weights
```

---

## 多头注意力

### 原理

多头注意力将 Q、K、V 投影到多个子空间，分别计算注意力，然后合并结果：

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

其中:
head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
```

### 好处

1. **多头** 可以同时关注不同位置的不同表示子空间
2. **并行** 多个头可以同时计算
3. **丰富** 每个头可以学习不同类型的依赖关系

### 实现

```python
# nanotorch/nn/attention.py

class MultiheadAttention(Module):
    """多头注意力层。
    
    Args:
        embed_dim: 模型维度
        num_heads: 头数
        dropout: dropout 概率
        bias: 是否使用偏置
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"
        
        # Q, K, V 投影
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        
        # 输出投影
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = dropout

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """前向传播。
        
        Args:
            query: (batch, seq_len, embed_dim)
            key: (batch, seq_len, embed_dim)
            value: (batch, seq_len, embed_dim)
            attn_mask: 可选的注意力掩码
        
        Returns:
            (output, attention_weights)
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # 线性投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 重塑为多头形式: (batch, seq_len, num_heads, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, key.shape[1], self.num_heads, self.head_dim)
        v = v.reshape(batch_size, value.shape[1], self.num_heads, self.head_dim)
        
        # 转置为: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力
        attn_output, attn_weights = scaled_dot_product_attention(
            q, k, v, attn_mask, self.dropout
        )
        
        # 合并多头: (batch, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.embed_dim
        )
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output, attn_weights
```

### 使用示例

```python
from nanotorch.nn import MultiheadAttention

# 创建多头注意力层
mha = MultiheadAttention(embed_dim=512, num_heads=8)

# 自注意力（query = key = value）
x = Tensor.randn((32, 100, 512))  # (batch, seq_len, embed_dim)
output, weights = mha(x, x, x)

print(output.shape)   # (32, 100, 512)
print(weights.shape)  # (32, 8, 100, 100)  # (batch, heads, seq, seq)
```

---

## Transformer Encoder Layer

### 结构

```
输入 x
   │
   ├──────────────────┐
   ↓                  │
MultiheadAttention    │
   ↓                  │
Dropout               │
   ↓                  │
Add (残差连接) ←──────┘
   ↓
LayerNorm
   │
   ├──────────────────┐
   ↓                  │
FeedForward           │
   ↓                  │
Dropout               │
   ↓                  │
Add (残差连接) ←──────┘
   ↓
LayerNorm
   │
输出
```

### 实现

```python
# nanotorch/nn/transformer.py

class TransformerEncoderLayer(Module):
    """Transformer 编码器层。
    
    Args:
        d_model: 模型维度
        nhead: 注意力头数
        dim_feedforward: 前馈网络维度
        dropout: dropout 概率
        activation: 激活函数 ('relu' 或 'gelu')
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        
        # 自注意力
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前馈网络
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        # 归一化
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Dropout
        self.dropout = Dropout(dropout)
        
        # 激活函数
        self.activation = ReLU() if activation == "relu" else GELU()

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """前向传播。
        
        Args:
            src: 输入 (seq_len, batch, d_model) 或 (batch, seq_len, d_model)
            src_mask: 可选的源序列掩码
        
        Returns:
            输出张量
        """
        # 自注意力 + 残差 + LayerNorm
        src2, _ = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # 前馈网络 + 残差 + LayerNorm
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        
        return src
```

### 编码器堆叠

```python
class TransformerEncoder(Module):
    """Transformer 编码器（多层堆叠）。"""

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = [encoder_layer for _ in range(num_layers)]
        self.num_layers = num_layers
        self.norm = LayerNorm(encoder_layer.d_model)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, mask)
        return self.norm(output)
```

---

## Transformer Decoder Layer

### 结构

Decoder 比 Encoder 多了一个**交叉注意力**（Cross-Attention）：

```
输入 tgt (目标序列)
   │
   ├──────────────────┐
   ↓                  │
Masked Self-Attention │  (带因果掩码)
   ↓                  │
Add ←─────────────────┘
   ↓
LayerNorm
   │
   ├──────────────────┐
   ↓                  │
Cross-Attention       │  (query=tgt, key=value=memory)
   ↓                  │
Add ←─────────────────┘
   ↓
LayerNorm
   │
   ├──────────────────┐
   ↓                  │
FeedForward           │
   ↓                  │
Add ←─────────────────┘
   ↓
LayerNorm
```

### 实现

```python
class TransformerDecoderLayer(Module):
    """Transformer 解码器层。"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        
        # 自注意力（带掩码）
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 交叉注意力
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前馈网络
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        # 归一化
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        
        # Dropout
        self.dropout = Dropout(dropout)
        
        self.activation = ReLU() if activation == "relu" else GELU()

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """前向传播。
        
        Args:
            tgt: 目标序列
            memory: 编码器输出
            tgt_mask: 目标序列因果掩码
            memory_mask: 编码器输出掩码
        """
        # 自注意力
        tgt2, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        
        # 交叉注意力
        tgt2, _ = self.multihead_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        # 前馈网络
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt
```

---

## 位置编码

### 为什么需要位置编码？

Transformer 没有循环结构，无法自动感知位置信息。需要显式添加位置编码。

### 正弦位置编码

```python
class PositionalEncoding(Module):
    """正弦位置编码。
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = Dropout(dropout)
        
        # 计算位置编码
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = Tensor(pe[np.newaxis, :, :])  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """添加位置编码。
        
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)
```

### 可学习位置编码

```python
class LearnablePositionalEncoding(Module):
    """可学习的位置编码。"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = Tensor(
            np.random.randn(1, max_len, d_model).astype(np.float32) * 0.02,
            requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, :x.shape[1], :]
```

---

## 完整 Transformer

### Seq2Seq 模型

```python
class Transformer(Module):
    """完整的 Transformer 模型。"""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # 编码
        memory = self.encoder(src, src_mask)
        
        # 解码
        output = self.decoder(tgt, memory, tgt_mask)
        
        return output
```

### 生成因果掩码

```python
def generate_square_subsequent_mask(sz: int) -> Tensor:
    """生成用于 decoder 的因果掩码。
    
    防止位置 i 看到位置 i 之后的信息。
    
    Example for sz=4:
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
    """
    mask = np.triu(np.ones((sz, sz)), k=1) == 0
    return Tensor(mask.astype(np.float32))
```

---

## 使用示例

### 文本分类（仅 Encoder）

```python
from nanotorch import Tensor
from nanotorch.nn import (
    Embedding, TransformerEncoderLayer, TransformerEncoder,
    Linear, LayerNorm, PositionalEncoding
)

class TransformerClassifier:
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes):
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        
        self.fc = Linear(d_model, num_classes)
    
    def __call__(self, x):
        # x: (batch, seq_len) 词索引
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        
        # 使用 [CLS] 位置或平均池化
        x = x.mean(dim=1)  # (batch, d_model)
        return self.fc(x)
```

### 机器翻译（Encoder-Decoder）

```python
class TranslationModel:
    def __init__(self, src_vocab, tgt_vocab, d_model=512, nhead=8, num_layers=6):
        self.src_embedding = Embedding(src_vocab, d_model)
        self.tgt_embedding = Embedding(tgt_vocab, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )
        
        self.fc_out = Linear(d_model, tgt_vocab)
    
    def encode(self, src):
        src = self.src_embedding(src)
        src = self.pos_encoder(src)
        return self.transformer.encoder(src)
    
    def decode(self, tgt, memory):
        tgt = self.tgt_embedding(tgt)
        tgt = self.pos_encoder(tgt)
        tgt_mask = generate_square_subsequent_mask(tgt.shape[1])
        return self.transformer.decoder(tgt, memory, tgt_mask)
    
    def __call__(self, src, tgt):
        memory = self.encode(src)
        output = self.decode(tgt, memory)
        return self.fc_out(output)
```

### GPT 风格（仅 Decoder）

```python
class GPTModel:
    """GPT 风格的自回归语言模型。"""
    
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_model * 4)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        
        self.lm_head = Linear(d_model, vocab_size)
    
    def __call__(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # 因果掩码
        mask = generate_square_subsequent_mask(x.shape[1])
        
        # 对于 decoder-only，memory 就是自己
        output = self.decoder(x, x, tgt_mask=mask)
        
        return self.lm_head(output)
```

---

## 总结

本教程介绍了 Transformer 架构的核心组件：

| 组件 | 作用 |
|------|------|
| **MultiheadAttention** | 并行计算多个注意力 |
| **TransformerEncoder** | 编码输入序列 |
| **TransformerDecoder** | 解码生成目标序列 |
| **PositionalEncoding** | 注入位置信息 |

### 关键要点

1. **注意力机制** 通过 Q、K、V 计算相关性
2. **多头注意力** 允许模型关注不同子空间
3. **残差连接 + LayerNorm** 稳定深层网络训练
4. **因果掩码** 防止 decoder 看到未来信息

### 下一步

在 [教程 13：数据加载](13-dataloader.md) 中，我们将学习如何实现 DataLoader 和 Dataset，用于高效地加载和处理训练数据。

---

**参考资源**：
- [Attention Is All You Need (原始论文)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Transformers from Scratch](https://e2eml.school/transformers.html)
