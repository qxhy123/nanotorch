# 第十六章：推荐系统实战 - DeepFM 模型

## 想象你走进一家你从未去过的书店...

数万册书籍静静排列，你有些迷茫——该从何下手？

这时，一位老店员走来。他不认识你，却仿佛读懂了你的心：
"您上次买了东野圭吾，新到了一本《嫌疑人X的献身》；您浏览过的科幻区，刘慈欣的新作刚上架；对了，像您这样的读者，通常也会喜欢乙一..."

你惊讶于他的眼光。其实，他只是记住了三个东西：
- 你是谁（用户画像）
- 书是什么（物品特征）
- 像你这样的人，通常会喜欢什么书（历史交互规律）

```
茫茫书海中：
  无指引 → 随手翻阅，可能错过挚爱
  有推荐 → 每一本递来的，都正合心意

推荐系统的秘密：
  不是猜测，而是计算
  不是运气，而是规律
  它让"相遇"变成"重逢"
```

**推荐系统，是数字世界的知音** —— 在万千选择中，为你呈上那份"恰好"。

---

## 目录

1. [推荐系统概述](#推荐系统概述)
2. [DeepFM 架构详解](#deepfm-架构详解)
3. [特征工程与数据处理](#特征工程与数据处理)
4. [模型实现](#模型实现)
5. [训练流程](#训练流程)
6. [评估指标](#评估指标)
7. [完整示例](#完整示例)
8. [模型对比](#模型对比)
9. [小结](#小结)

---

## 推荐系统概述

### 什么是推荐系统？

推荐系统的目标是预测用户对物品的偏好，从而向用户推荐可能感兴趣的物品。常见的应用场景包括：

- **电商推荐**：商品推荐、猜你喜欢
- **内容推荐**：新闻、视频、音乐推荐
- **广告推荐**：CTR（点击率）预测

### 推荐任务类型

| 任务类型 | 描述 | 输出 |
|---------|------|------|
| **CTR 预测** | 预测用户是否会点击 | 二分类（0/1） |
| **评分预测** | 预测用户对物品的评分 | 回归（1-5分） |
| **排序学习** | 为候选物品排序 | 排序分数 |
| **序列推荐** | 根据历史行为预测下一个物品 | 物品ID |

### 推荐系统的发展

```
传统方法 → 协同过滤 → 矩阵分解 → 深度学习
                         ↓
         ┌───────────────┼───────────────┐
         ↓               ↓               ↓
      Wide&Deep       DeepFM          DIN/DIEN
```

---

## DeepFM 架构详解

### 核心思想

DeepFM 结合了 **Factorization Machine (FM)** 和 **Deep Neural Network (DNN)** 的优势：

- **FM 组件**：捕获低阶特征交互（二阶交叉）
- **DNN 组件**：捕获高阶特征交互（非线性组合）
- **共享嵌入**：两个组件共享相同的特征嵌入，实现端到端训练

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      Sparse Features                        │
│         (user_id, item_id, category, brand, ...)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Shared Embedding Layer                   │
│              每个稀疏特征 → Dense Vector                     │
└─────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
    ┌────────────┐     ┌─────────────┐    ┌──────────┐
    │    FM      │     │    DNN      │    │  Linear  │
    │  Component │     │  Component  │    │  (1st)   │
    │  (2nd)     │     │  (High)     │    │          │
    └────────────┘     └─────────────┘    └──────────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              │
                              ▼
                       ┌─────────────┐
                       │   Sigmoid   │
                       │  CTR Score  │
                       └─────────────┘
```

### Factorization Machine 组件

FM 用于建模特征之间的二阶交互：

$$y_{FM} = \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

**高效计算**：直接计算二阶项需要 $O(n^2)$ 复杂度，但可以通过以下公式优化到 $O(nk)$：

$$\sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2}\sum_{k=1}^{K}\left[\left(\sum_{i=1}^{n} v_{ik}x_i\right)^2 - \sum_{i=1}^{n} v_{ik}^2 x_i^2\right]$$

**推导过程**：

**第一步**：展开平方项。

$$\left(\sum_{i=1}^{n} v_{ik}x_i\right)^2 = \sum_{i=1}^{n} v_{ik}^2 x_i^2 + 2\sum_{i=1}^{n}\sum_{j=i+1}^{n} v_{ik}v_{jk}x_ix_j$$

**第二步**：移项得到交叉项。

$$\sum_{i=1}^{n}\sum_{j=i+1}^{n} v_{ik}v_{jk}x_ix_j = \frac{1}{2}\left[\left(\sum_{i=1}^{n} v_{ik}x_i\right)^2 - \sum_{i=1}^{n} v_{ik}^2 x_i^2\right]$$

**第三步**：对所有隐维度求和。

$$\boxed{\sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2}\sum_{k=1}^{K}\left[\left(\sum_{i=1}^{n} v_{ik}x_i\right)^2 - \sum_{i=1}^{n} v_{ik}^2 x_i^2\right]}$$

### DNN 组件

DNN 用于学习高阶特征交互：

```
Embeddings → Flatten → Dense → ReLU → Dense → ReLU → Dense
                         │              │              │
                        256            128            64
```

---

## 特征工程与数据处理

### 特征类型

在推荐系统中，特征通常分为两类：

| 类型 | 描述 | 示例 | 处理方式 |
|------|------|------|---------|
| **稀疏特征** | 类别型，高基数 | user_id, item_id | Embedding |
| **稠密特征** | 数值型，连续 | price, rating | 归一化 |

### 数据模式

```python
# 单个样本结构
{
    # 稀疏特征（类别型）
    'user_id': 12345,         # 用户ID
    'item_id': 67890,         # 物品ID
    'category': 15,           # 类别
    'brand': 42,              # 品牌
    'device': 0,              # 设备类型 (0=mobile, 1=desktop)
    
    # 稠密特征（数值型）
    'user_age': 0.45,         # 归一化年龄
    'item_price': 0.23,       # 归一化价格
    'item_rating': 0.85,      # 归一化评分
    
    # 标签
    'label': 1                # 是否点击
}
```

### 数据加载实现

```python
from dataclasses import dataclass
from nanotorch.data import Dataset, DataLoader

@dataclass
class SparseFeat:
    """稀疏特征配置"""
    name: str
    vocabulary_size: int
    embedding_dim: int = 8

@dataclass  
class DenseFeat:
    """稠密特征配置"""
    name: str
    dimension: int = 1

class RecommendationDataset(Dataset):
    """推荐数据集"""
    
    def __init__(
        self,
        sparse_features: Dict[str, np.ndarray],
        dense_features: Dict[str, np.ndarray],
        labels: np.ndarray
    ):
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sparse = {k: v[idx] for k, v in self.sparse_features.items()}
        dense = {k: v[idx] for k, v in self.dense_features.items()}
        return sparse, dense, self.labels[idx]
```

---

## 模型实现

### FM 层实现

```python
from nanotorch.nn.module import Module
from nanotorch.tensor import Tensor
import numpy as np

class FactorizationMachine(Module):
    """Factorization Machine 层
    
    公式: y = 0.5 * (||sum(x)||^2 - sum(||x||^2))
    """
    
    def __init__(self, num_fields: int, embed_dim: int, reduce_sum: bool = True):
        super().__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.reduce_sum = reduce_sum
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, num_fields, embed_dim)
        Returns:
            (batch_size, 1) if reduce_sum else (batch_size, embed_dim)
        """
        # 平方和: (Σ x)^2
        sum_of_x = x.sum(axis=1)            # (batch, embed_dim)
        square_of_sum = sum_of_x * sum_of_x  # (batch, embed_dim)
        
        # 和平方: Σ x^2
        square_of_x = x * x                  # (batch, fields, embed_dim)
        sum_of_square = square_of_x.sum(axis=1)  # (batch, embed_dim)
        
        # FM 交互: 0.5 * (square_of_sum - sum_of_square)
        fm_interaction = (square_of_sum - sum_of_square) * 0.5
        
        if self.reduce_sum:
            return fm_interaction.sum(axis=1, keepdims=True)  # (batch, 1)
        return fm_interaction  # (batch, embed_dim)
```

### DeepFM 完整实现

```python
from nanotorch.nn import Module, Sequential, Linear, ReLU, Dropout, LayerNorm
from nanotorch.nn import Embedding

class DeepFM(Module):
    """DeepFM: FM + DNN for CTR Prediction
    
    Args:
        sparse_features: 稀疏特征配置列表
        dense_features: 稠密特征配置列表
        embed_dim: 嵌入维度
        hidden_dims: DNN 隐藏层维度列表
        dropout: Dropout 比率
    """
    
    def __init__(
        self,
        sparse_features: List[SparseFeat],
        dense_features: List[DenseFeat],
        embed_dim: int = 16,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.num_sparse = len(sparse_features)
        self.num_dense = len(dense_features)
        
        # 共享嵌入层
        self.embeddings = {}
        for feat in sparse_features:
            emb = Embedding(feat.vocabulary_size, embed_dim)
            self.embeddings[feat.name] = emb
            self.register_module(f'emb_{feat.name}', emb)
        
        # FM 组件
        self.fm = FactorizationMachine(self.num_sparse, embed_dim)
        
        # DNN 组件
        total_dense_dim = sum(f.dimension for f in dense_features)
        dnn_input_dim = self.num_sparse * embed_dim + total_dense_dim
        
        self.dnn = Sequential(
            Linear(dnn_input_dim, hidden_dims[0]),
            LayerNorm(hidden_dims[0]),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dims[0], hidden_dims[1]),
            LayerNorm(hidden_dims[1]),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dims[1], 1)
        )
        
        # 一阶线性项
        self.linear = Linear(self.num_sparse + self.num_dense, 1)
    
    def forward(self, sparse_input: Tensor, dense_input: Tensor = None) -> Tensor:
        """
        Args:
            sparse_input: (batch, num_sparse) 整数索引
            dense_input: (batch, num_dense) 浮点值
        
        Returns:
            (batch, 1) CTR 概率
        """
        batch_size = sparse_input.shape[0]
        x_np = sparse_input.data.astype(np.int64)
        
        # 嵌入稀疏特征
        embedded_list = []
        for i, feat in enumerate(self.sparse_features):
            indices = Tensor(x_np[:, i])
            emb = self.embeddings[feat.name](indices)
            embedded_list.append(emb.data)
        
        # 堆叠: (batch, num_sparse, embed_dim)
        embedded = Tensor(
            np.stack(embedded_list, axis=1).astype(np.float32),
            requires_grad=True
        )
        
        # FM 输出
        fm_output = self.fm(embedded)  # (batch, 1)
        
        # DNN 输入: 展平嵌入 + 稠密特征
        embedded_flat = embedded.data.reshape(batch_size, -1)
        if dense_input is not None:
            dnn_input = np.concatenate([embedded_flat, dense_input.data], axis=1)
        else:
            dnn_input = embedded_flat
        
        dnn_input_tensor = Tensor(dnn_input.astype(np.float32), requires_grad=True)
        dnn_output = self.dnn(dnn_input_tensor)  # (batch, 1)
        
        # 一阶线性输出
        if dense_input is not None:
            linear_input = np.concatenate([x_np, dense_input.data], axis=1)
        else:
            linear_input = x_np
        linear_input_tensor = Tensor(linear_input.astype(np.float32), requires_grad=True)
        linear_output = self.linear(linear_input_tensor)
        
        # 组合: FM + DNN + Linear
        combined = fm_output.data + dnn_output.data + linear_output.data
        
        # Sigmoid 激活
        output = Tensor(
            1.0 / (1.0 + np.exp(-np.clip(combined, -15, 15))),
            requires_grad=True
        )
        
        return output
```

---

## 训练流程

### 训练配置

```python
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    num_epochs: int = 20
    learning_rate: float = 0.001
    batch_size: int = 256
    weight_decay: float = 0.0001
    early_stop_patience: int = 3
    gradient_clip_norm: float = 5.0
    lr_scheduler: str = 'cosine_warmup'
    warmup_epochs: int = 2
```

### 训练循环

```python
from nanotorch.optim.adamw import AdamW
from nanotorch.optim.lr_scheduler import CosineWarmupScheduler
from nanotorch.nn import BCELoss
from nanotorch.utils import clip_grad_norm_

def train_model(model, train_loader, val_loader, config):
    """完整训练流程"""
    
    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        max_epochs=config.num_epochs
    )
    
    # 损失函数
    criterion = BCELoss()
    
    # 早停
    best_val_auc = 0.0
    patience_counter = 0
    
    for epoch in range(1, config.num_epochs + 1):
        # === 训练阶段 ===
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            sparse, dense, labels = batch
            
            # 前向传播
            predictions = model(
                Tensor(sparse.astype(np.float32)),
                Tensor(dense.astype(np.float32)) if dense is not None else None
            )
            
            loss = criterion(predictions, Tensor(labels.reshape(-1, 1)))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # === 验证阶段 ===
        val_auc = evaluate(model, val_loader)
        
        print(f"Epoch {epoch}: Loss={train_loss/len(train_loader):.4f}, "
              f"Val AUC={val_auc:.4f}")
        
        # 早停检查
        if val_auc > best_val_auc + 0.0001:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return model
```

---

## 评估指标

### 分类指标

| 指标 | 公式 | 含义 |
|------|------|------|
| **AUC** | ROC 曲线下面积 | 正样本排名高于负样本的概率 |
| **LogLoss** | $-\frac{1}{N}\sum(y\log(p) + (1-y)\log(1-p))$ | 交叉熵损失 |

### 排序指标

| 指标 | 公式 | 含义 |
|------|------|------|
| **Recall@K** | $\frac{\|relevant \cap topK\|}{\|relevant\|}$ | Top-K 中相关物品占比 |
| **NDCG@K** | $\frac{DCG@K}{IDCG@K}$ | 考虑位置的排序质量 |
| **Hit@K** | $1[topK \cap relevant \neq \emptyset]$ | Top-K 是否包含相关物品 |
| **MRR** | $\frac{1}{|Q|}\sum \frac{1}{rank_1}$ | 第一个相关物品的平均倒数排名 |

### 指标实现

```python
def auc_score(predictions, targets):
    """计算 AUC"""
    sorted_indices = np.argsort(-predictions)
    sorted_targets = targets[sorted_indices]
    
    n_pos = np.sum(targets == 1)
    n_neg = np.sum(targets == 0)
    
    tp_cumsum = np.cumsum(sorted_targets == 1)
    auc = np.sum(tp_cumsum[sorted_targets == 0]) / (n_pos * n_neg)
    
    return auc

def ndcg_at_k(predictions, targets, k):
    """计算 NDCG@K"""
    # DCG: sum(rel_i / log2(i+1))
    ranked_indices = np.argsort(-predictions)
    ranked_relevances = targets[ranked_indices][:k]
    
    discounts = 1.0 / np.log2(np.arange(len(ranked_relevances)) + 2)
    dcg = np.sum(ranked_relevances * discounts)
    
    # IDCG: ideal DCG
    ideal_relevances = np.sort(targets)[::-1][:k]
    idcg = np.sum(ideal_relevances * discounts)
    
    return dcg / idcg if idcg > 0 else 0
```

---

## 完整示例

### 生成数据

```python
from examples.recommendation.data import generate_synthetic_data, create_dataloaders

# 生成合成推荐数据
dataset, sparse_configs, dense_configs = generate_synthetic_data(
    num_samples=100000,
    num_users=10000,
    num_items=5000,
    click_rate=0.05,
    random_seed=42
)

# 创建数据加载器
train_loader, val_loader, test_loader = create_dataloaders(
    dataset,
    batch_size=256,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
```

### 创建模型

```python
from nanotorch.nn.recommender import DeepFM

model = DeepFM(
    sparse_features=sparse_configs,
    dense_features=dense_configs,
    embed_dim=16,
    hidden_dims=[256, 128, 64],
    dropout=0.1
)

# 打印参数量
n_params = sum(p.data.size for p in model.parameters())
print(f"Total parameters: {n_params:,}")
```

### 训练和评估

```python
from examples.recommendation.train import train_model, TrainingConfig
from examples.recommendation.evaluate import evaluate_model, print_evaluation_report

# 训练配置
config = TrainingConfig(
    num_epochs=20,
    learning_rate=0.001,
    early_stop_patience=3
)

# 训练
history = train_model(model, train_loader, val_loader, config)

# 评估
test_metrics = evaluate_model(model, test_loader, ks=[1, 5, 10, 20])
print_evaluation_report(test_metrics, "DeepFM Test Results")
```

### 输出示例

```
==================================================
 Evaluation Results
==================================================

  Classification Metrics:
    AUC:      0.7823
    LogLoss:  0.3245

  Ranking Metrics:
    hit@1: 0.2134
    hit@5: 0.4521
    hit@10: 0.5892
    hit@20: 0.7234
    mrr:      0.3892
    map:      0.3456

==================================================
```

---

## 模型对比

### 支持的架构

| 模型 | 特点 | 适用场景 |
|------|------|---------|
| **DeepFM** | FM + DNN，自动特征交叉 | CTR 预测 |
| **Wide & Deep** | 线性 + DNN，记忆+泛化 | 通用推荐 |
| **NeuralCF** | GMF + MLP，协同过滤 | 用户-物品交互 |
| **Two-Tower** | 双塔结构，用户/物品分离 | 大规模检索 |
| **DCN** | Cross Network，显式交叉 | 高阶特征交互 |

### 性能对比示例

```python
from nanotorch.nn.recommender import DeepFM, WideDeep, NeuralCF

models = {
    'DeepFM': DeepFM(sparse_configs, dense_configs, embed_dim=16),
    'Wide&Deep': WideDeep(sparse_configs, dense_configs, embed_dim=16),
    'NeuralCF': NeuralCF(num_users=10000, num_items=5000, embed_dim=16)
}

# 训练并比较
results = {}
for name, model in models.items():
    history = train_model(model, train_loader, val_loader, config)
    metrics = evaluate_model(model, test_loader)
    results[name] = metrics

# 打印对比结果
print(f"{'Model':<15} {'AUC':>8} {'LogLoss':>10}")
print("-" * 35)
for name, m in results.items():
    print(f"{name:<15} {m['auc']:>8.4f} {m['log_loss']:>10.4f}")
```

---

## 小结

本章我们实现了一个接近生产级复杂度的推荐系统，涵盖：

| 组件 | 文件 | 功能 |
|------|------|------|
| FM 层 | `nanotorch/nn/fm.py` | 二阶特征交叉 |
| 推荐模型 | `nanotorch/nn/recommender.py` | DeepFM, Wide&Deep, NeuralCF |
| 评估指标 | `nanotorch/nn/metrics.py` | AUC, NDCG@K, Recall@K, MRR |
| 数据处理 | `examples/recommendation/data.py` | 合成数据生成 |
| 训练流程 | `examples/recommendation/train.py` | 训练循环、早停 |
| 完整示例 | `examples/recommendation/recommender_demo.py` | 端到端演示 |

### 关键公式

**FM 高效计算**：
$$\sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2}\sum_{k}\left[\left(\sum_{i} v_{ik}x_i\right)^2 - \sum_{i} v_{ik}^2 x_i^2\right]$$

**DeepFM 输出**：
$$\hat{y} = \sigma(y_{FM} + y_{DNN} + y_{linear})$$

### 扩展阅读

- [DeepFM 原论文](https://arxiv.org/abs/1703.04247)
- [Wide & Deep Learning](https://arxiv.org/abs/1606.07792)
- [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)

---

**上一章**：[第十五章：高级话题](15-advanced.md)

**返回**：[教程目录](00-overview.md)
