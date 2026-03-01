# 第十二章：Transformer

## 2017年，一场革命悄然发生...

在那之前，自然语言处理被"循环"统治。RNN、LSTM、GRU——它们像一个勤奋的读者，从左到右，一字一句地读过去。

这种方法有个致命的弱点：**无法并行**。读完最后一个字，才能理解整句话。训练时间漫长，长句子更是噩梦。

然后，一篇论文横空出世：《Attention Is All You Need》。

它说：为什么非要从左到右？为什么不能一眼看完全文？为什么不能让每个字都直接"看到"其他所有字？

这就是**自注意力**。

```
RNN 的方式：
  "我" → "爱" → "你"
  串行阅读，像翻书

Transformer 的方式：
  "我" ←→ "爱" ←→ "你"
  每个词都能直接看到其他词
  就像把书摊开，一眼看全
```

**Transformer，用"注意力"取代了"循环"。** 它让机器学会了：在理解一句话时，不是逐字阅读，而是同时关注所有词，找出它们之间的关系。

从此，大模型时代开启。

---

## 12.1 注意力机制

### 生活类比

```
你在看一张照片：

  👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦
  一群人在聚会

你的注意力：
  - 找朋友时：主要看"人脸"
  - 数人数时：看"人头"
  - 看背景时：看"环境"

同一个场景，你"关注"的点不同
→ 这就是注意力
```

### 核心思想

```
注意力机制：查询(Q) × 键(K) × 值(V)

类比图书馆：
  Q (Query) = 你要找什么（查询）
  K (Key)   = 每本书的标签（键）
  V (Value) = 书的内容（值）

过程：
  1. 用 Q 和所有 K 比较（相关性）
  2. 得到注意力权重（哪些书相关）
  3. 用权重组合 V（得到相关内容）
```

### 自注意力

```
自注意力：Q、K、V 都来自同一个输入

句子："我 爱 北京 天安门"

Q = "爱" 的表示（我想找什么）
K = 所有词的表示（每个词的标签）
V = 所有词的表示（每个词的内容）

计算："爱" 和每个词的相关性
  - 和 "我" 相关：0.7（主语）
  - 和 "爱" 相关：0.1（自己）
  - 和 "北京" 相关：0.15（宾语）
  - 和 "天安门" 相关：0.05（修饰）

用这些权重组合所有词 → "爱" 的新表示
```

---

## 12.2 注意力计算

### 公式

```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

步骤分解：
  1. Q @ K^T     ：计算相关性得分
  2. / √d_k      ：缩放（防止梯度消失）
  3. softmax     ：归一化为概率
  4. @ V         ：加权求和
```

### 图解

```
输入序列：[词1, 词2, 词3]

Q = [q1]    K^T = [k1, k2, k3]    V = [v1]
    [q2]                           [v2]
    [q3]                           [v3]

第1步：Q @ K^T = 相关性矩阵
        词1   词2   词3
词1 [ 0.9   0.3   0.1 ]    词1 和词1 最相关
词2 [ 0.2   0.8   0.4 ]
词3 [ 0.1   0.3   0.9 ]

第2步：softmax = 注意力权重
        词1   词2   词3
词1 [ 0.70  0.20  0.10 ]
词2 [ 0.15  0.60  0.25 ]
词3 [ 0.10  0.25  0.65 ]

第3步：@ V = 加权组合
输出 = [0.7*v1 + 0.2*v2 + 0.1*v3]
       [0.15*v1 + 0.6*v2 + 0.25*v3]
       [0.1*v1 + 0.25*v2 + 0.65*v3]
```

### 实现

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    缩放点积注意力

    类比：
      q = 查询（我要找什么）
      k = 键（每本书的标签）
      v = 值（书的内容）
    """
    d_k = q.shape[-1]

    # 1. 计算相关性得分
    scores = q.matmul(k.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. 应用掩码（可选）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 3. softmax 归一化
    attn_weights = scores.softmax(dim=-1)

    # 4. 加权求和
    output = attn_weights.matmul(v)

    return output, attn_weights
```

---

## 12.3 多头注意力

### 为什么需要多头？

```
单头注意力：
  只能学习一种"相关性"

多头注意力：
  可以学习多种"相关性"

例子：
  头1：关注语法关系（主语-谓语）
  头2：关注语义关系（同义词）
  头3：关注位置关系（相邻词）
  ...

每个头独立学习，最后合并
```

### 图解

```
输入 X (batch, seq_len, d_model)
         │
         ├──→ W_Q → Q ──┐
         ├──→ W_K → K ──┼──→ 头1 注意力 ──┐
         └──→ W_V → V ──┘                  │
                                            │
         ┌─────────────────────────────────┘
         │
         ├──→ W_Q → Q ──┐
         ├──→ W_K → K ──┼──→ 头2 注意力 ──┼──→ Concat ──→ W_O ──→ 输出
         └──→ W_V → V ──┘                  │
                                            │
         ┌─────────────────────────────────┘
         │
         ...（更多头）
```

### 实现

```python
class MultiheadAttention(Module):
    """
    多头注意力

    类比：多个专家从不同角度看问题
    """

    def __init__(
        self,
        embed_dim: int,      # 模型维度
        num_heads: int,      # 头数
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim

        # Q, K, V 投影
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)

        # 输出投影
        self.out_proj = Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        # 1. 线性投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2. 重塑为多头 (batch, seq, heads, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, key.shape[1], self.num_heads, self.head_dim)
        v = v.reshape(batch_size, value.shape[1], self.num_heads, self.head_dim)

        # 3. 转置 (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 4. 计算注意力
        attn_output, attn_weights = scaled_dot_product_attention(
            q, k, v, attn_mask
        )

        # 5. 合并多头
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.embed_dim
        )

        # 6. 输出投影
        output = self.out_proj(attn_output)

        return output, attn_weights
```

### 使用

```python
# 创建多头注意力
mha = MultiheadAttention(embed_dim=512, num_heads=8)

# 自注意力（Q = K = V）
x = Tensor.randn((32, 100, 512))  # (batch, seq_len, embed_dim)
output, weights = mha(x, x, x)

print(output.shape)    # (32, 100, 512)
print(weights.shape)   # (32, 8, 100, 100)  注意力权重
```

---

## 12.4 Transformer Encoder

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
Add ←─────────────────┘ (残差连接)
   ↓
LayerNorm
   │
   ├──────────────────┐
   ↓                  │
FeedForward           │
   ↓                  │
Dropout               │
   ↓                  │
Add ←─────────────────┘ (残差连接)
   ↓
LayerNorm
   │
输出
```

### 实现

```python
class TransformerEncoderLayer(Module):
    """
    Transformer 编码器层

    结构：注意力 → 残差+归一化 → 前馈 → 残差+归一化
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 自注意力
        self.self_attn = MultiheadAttention(d_model, nhead, dropout)

        # 前馈网络
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)

        # 归一化
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # Dropout
        self.dropout = Dropout(dropout)

    def forward(self, src, src_mask=None):
        # 1. 自注意力 + 残差 + LayerNorm
        src2, _ = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout(src2)  # 残差连接
        src = self.norm1(src)

        # 2. 前馈网络 + 残差 + LayerNorm
        src2 = self.linear2(self.dropout(self.linear1(src).relu()))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src
```

### 残差连接

```
为什么需要残差连接？

问题：深层网络难训练
  输入 → 层1 → 层2 → ... → 层100 → 输出
  梯度要传100层，容易消失

解决：残差连接
  输入 → 层 → + → 输出
         ↑_____|
  梯度可以直接"跳过"某些层

类比：
  没有：每层都要完美
  有：每层只需要学习"残差"（和输入的差异）
```

---

## 12.5 Transformer Decoder

### 和 Encoder 的区别

```
Decoder 多了一个"交叉注意力"：

Encoder 输出 ────→ 交叉注意力的 K, V
                     ↓
目标序列 ──→ 自注意力 → 交叉注意力 → 前馈 → 输出
              (有掩码)
```

### 因果掩码

```
Decoder 的自注意力需要"因果掩码"：

为什么？
  - 解码时，不能看到未来的词
  - 翻译时，不能看到还没翻译的部分

掩码示例（4个位置）：
  [[1, 0, 0, 0],   位置1：只能看位置1
   [1, 1, 0, 0],   位置2：能看位置1,2
   [1, 1, 1, 0],   位置3：能看位置1,2,3
   [1, 1, 1, 1]]   位置4：能看位置1,2,3,4

0 的位置会被设为 -inf，softmax 后变成 0
```

### 生成掩码

```python
def generate_square_subsequent_mask(sz: int):
    """
    生成因果掩码

    防止看到未来的信息
    """
    mask = np.triu(np.ones((sz, sz)), k=1) == 0
    # 上三角为 False，下三角为 True
    return Tensor(mask.astype(np.float32))
```

---

## 12.6 位置编码

### 为什么需要？

```
Transformer 没有"循环"，所有位置同时处理

问题：
  "我爱你" 和 "你爱我" 处理方式一样
  但意思完全不同！

解决：
  给每个位置加一个"位置标签"
  让模型知道"我是第几个词"
```

### 正弦位置编码

```python
class PositionalEncoding(Module):
    """
    正弦位置编码

    用不同频率的正弦波编码位置

    位置0：sin(0), cos(0), sin(0), cos(0), ...
    位置1：sin(1), cos(1), sin(0.01), cos(0.01), ...
    位置2：sin(2), cos(2), sin(0.02), cos(0.02), ...
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = Dropout(dropout)

        # 计算位置编码
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = np.cos(position * div_term)  # 奇数位置

        self.pe = Tensor(pe[np.newaxis, :, :])  # (1, max_len, d_model)

    def forward(self, x: Tensor):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)
```

---

## 12.7 完整 Transformer

### 架构图

```
┌─────────────────────────────────────────┐
│              Transformer                │
├─────────────────────────────────────────┤
│                                         │
│  源语言 ──→ Embedding ──→ + 位置编码    │
│                    ↓                    │
│            ┌──────────────┐             │
│            │   Encoder    │ × N 层      │
│            └──────────────┘             │
│                    ↓                    │
│               Memory                    │
│                    ↓                    │
│  目标语言 ──→ Embedding ──→ + 位置编码  │
│                    ↓                    │
│            ┌──────────────┐             │
│            │   Decoder    │ × N 层      │
│            └──────────────┘             │
│                    ↓                    │
│              Linear                     │
│                    ↓                    │
│             Softmax                     │
│                    ↓                    │
│              输出概率                   │
│                                         │
└─────────────────────────────────────────┘
```

### 使用示例

```python
class TransformerClassifier:
    """
    用 Transformer 做文本分类（只用 Encoder）
    """

    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes):
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # 编码器
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        # 分类头
        self.fc = Linear(d_model, num_classes)

    def __call__(self, x):
        # x: (batch, seq_len) 词索引
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)

        # 平均池化
        x = x.mean(dim=1)
        return self.fc(x)
```

---

## 12.8 Transformer 变体

### BERT（仅 Encoder）

```
BERT = Bidirectional Encoder Representations from Transformers

特点：
  - 只用 Encoder
  - 双向（能看到整个句子）
  - 预训练 + 微调

用途：理解任务（分类、标注等）
```

### GPT（仅 Decoder）

```
GPT = Generative Pre-trained Transformer

特点：
  - 只用 Decoder
  - 单向（因果掩码）
  - 自回归生成

用途：生成任务（文本生成、对话等）
```

### 对比

| 模型 | 架构 | 方向 | 用途 |
|------|------|------|------|
| BERT | Encoder | 双向 | 理解 |
| GPT | Decoder | 单向 | 生成 |
| T5 | Encoder+Decoder | 双向+单向 | 翻译 |

---

## 12.9 常见陷阱

### 陷阱1：忘记位置编码

```python
# 错误：没有位置信息
x = self.embedding(tokens)
x = self.encoder(x)  # 不知道词序！

# 正确
x = self.embedding(tokens)
x = self.pos_encoder(x)  # 加位置编码
x = self.encoder(x)
```

### 陷阱2：Decoder 忘记掩码

```python
# 错误：能看到未来
output = self.decoder(tgt, memory)  # 没有掩码

# 正确：因果掩码
mask = generate_square_subsequent_mask(tgt.shape[1])
output = self.decoder(tgt, memory, tgt_mask=mask)
```

### 陷阱3：维度不匹配

```python
# 多头注意力：embed_dim 必须能被 num_heads 整除
MultiheadAttention(embed_dim=512, num_heads=8)   # ✓ 512/8=64
MultiheadAttention(embed_dim=512, num_heads=7)   # ✗ 512/7 不整除
```

---

## 12.10 一句话总结

| 概念 | 一句话 |
|------|--------|
| 注意力 | Q查询K得到权重，加权组合V |
| 自注意力 | Q=K=V，自己和自己的关系 |
| 多头 | 多个角度同时看，合并结果 |
| 残差连接 | 输入直接加到输出，方便梯度流动 |
| 位置编码 | 给每个位置加标签，区分顺序 |
| 因果掩码 | 不让Decoder看未来 |

---

## 下一章

现在我们学会了 Transformer！

下一章，我们将学习**数据加载** —— 如何高效地加载和处理训练数据。

→ [第十三章：数据加载](13-dataloader.md)

```python
# 预告：下一章你将学到
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch_x, batch_y in dataloader:
    # 自动分批、打乱、并行加载
```
