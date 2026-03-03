# Chapter 16: Recommendation System in Practice - DeepFM Model

## Imagine You Walk Into a Bookstore You've Never Been To...

Tens of thousands of books sit quietly on the shelves. You feel a bit lost—where should you start?

Then, an elderly shopkeeper approaches. He doesn't know you, yet he seems to read your mind:
"You bought Higashino Keigo last time; we just got 'The Devotion of Suspect X.' That sci-fi section you were browsing—Liu Cixin's new work just arrived. Oh, and readers like you usually enjoy Otsuichi too..."

You marvel at his insight. In truth, he simply remembers three things:
- Who you are (user profile)
- What the books are (item features)
- What books people like you usually like (historical interaction patterns)

```
In the vast sea of books:
  Without guidance → Browse randomly, might miss your true loves
  With recommendations → Every book handed to you is exactly what you wanted

The secret of recommendation systems:
  Not guessing, but calculating
  Not luck, but patterns
  It turns "encounters" into "reunions"
```

**Recommendation systems are the kindred spirits of the digital world**—presenting you with that "just right" choice among thousands of options.

---

## Table of Contents

1. [Recommendation System Overview](#recommendation-system-overview)
2. [DeepFM Architecture Details](#deepfm-architecture-details)
3. [Feature Engineering and Data Processing](#feature-engineering-and-data-processing)
4. [Model Implementation](#model-implementation)
5. [Training Process](#training-process)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Complete Example](#complete-example)
8. [Model Comparison](#model-comparison)
9. [Summary](#summary)

---

## Recommendation System Overview

### What is a Recommendation System?

The goal of recommendation systems is to predict user preferences for items, thereby recommending items users might be interested in. Common application scenarios include:

- **E-commerce Recommendations**: Product recommendations, "Guess what you like"
- **Content Recommendations**: News, video, music recommendations
- **Ad Recommendations**: CTR (Click-Through Rate) prediction

### Recommendation Task Types

| Task Type | Description | Output |
|-----------|-------------|--------|
| **CTR Prediction** | Predict if user will click | Binary classification (0/1) |
| **Rating Prediction** | Predict user's rating for item | Regression (1-5 stars) |
| **Learning to Rank** | Rank candidate items | Ranking scores |
| **Sequential Recommendation** | Predict next item based on history | Item ID |

### Evolution of Recommendation Systems

```
Traditional Methods → Collaborative Filtering → Matrix Factorization → Deep Learning
                                                                  ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
                Wide&Deep       DeepFM          DIN/DIEN
```

---

## DeepFM Architecture Details

### Core Idea

DeepFM combines the advantages of **Factorization Machine (FM)** and **Deep Neural Network (DNN)**:

- **FM Component**: Captures low-order feature interactions (second-order crosses)
- **DNN Component**: Captures high-order feature interactions (nonlinear combinations)
- **Shared Embedding**: Both components share the same feature embeddings for end-to-end training

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Sparse Features                        │
│         (user_id, item_id, category, brand, ...)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Shared Embedding Layer                   │
│              Each sparse feature → Dense Vector             │
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

### Factorization Machine Component

FM models second-order interactions between features:

$$y_{FM} = \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

**Efficient Computation**: Directly computing the second-order term requires $O(n^2)$ complexity, but can be optimized to $O(nk)$ using:

$$\sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2}\sum_{k=1}^{K}\left[\left(\sum_{i=1}^{n} v_{ik}x_i\right)^2 - \sum_{i=1}^{n} v_{ik}^2 x_i^2\right]$$

**Derivation Process**:

**Step 1**: Expand the squared term.

$$\left(\sum_{i=1}^{n} v_{ik}x_i\right)^2 = \sum_{i=1}^{n} v_{ik}^2 x_i^2 + 2\sum_{i=1}^{n}\sum_{j=i+1}^{n} v_{ik}v_{jk}x_ix_j$$

**Step 2**: Rearrange to get the cross term.

$$\sum_{i=1}^{n}\sum_{j=i+1}^{n} v_{ik}v_{jk}x_ix_j = \frac{1}{2}\left[\left(\sum_{i=1}^{n} v_{ik}x_i\right)^2 - \sum_{i=1}^{n} v_{ik}^2 x_i^2\right]$$

**Step 3**: Sum over all latent dimensions.

$$\boxed{\sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2}\sum_{k=1}^{K}\left[\left(\sum_{i=1}^{n} v_{ik}x_i\right)^2 - \sum_{i=1}^{n} v_{ik}^2 x_i^2\right]}$$

### DNN Component

DNN learns high-order feature interactions:

```
Embeddings → Flatten → Dense → ReLU → Dense → ReLU → Dense
                         │              │              │
                        256            128            64
```

---

## Feature Engineering and Data Processing

### Feature Types

In recommendation systems, features are typically divided into two categories:

| Type | Description | Example | Processing |
|------|-------------|---------|------------|
| **Sparse Features** | Categorical, high cardinality | user_id, item_id | Embedding |
| **Dense Features** | Numerical, continuous | price, rating | Normalization |

### Data Schema

```python
# Single sample structure
{
    # Sparse features (categorical)
    'user_id': 12345,         # User ID
    'item_id': 67890,         # Item ID
    'category': 15,           # Category
    'brand': 42,              # Brand
    'device': 0,              # Device type (0=mobile, 1=desktop)

    # Dense features (numerical)
    'user_age': 0.45,         # Normalized age
    'item_price': 0.23,       # Normalized price
    'item_rating': 0.85,      # Normalized rating

    # Label
    'label': 1                # Click or not
}
```

### Data Loading Implementation

```python
from dataclasses import dataclass
from nanotorch.data import Dataset, DataLoader

@dataclass
class SparseFeat:
    """Sparse feature configuration"""
    name: str
    vocabulary_size: int
    embedding_dim: int = 8

@dataclass
class DenseFeat:
    """Dense feature configuration"""
    name: str
    dimension: int = 1

class RecommendationDataset(Dataset):
    """Recommendation dataset"""

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

## Model Implementation

### FM Layer Implementation

```python
from nanotorch.nn.module import Module
from nanotorch.tensor import Tensor
import numpy as np

class FactorizationMachine(Module):
    """Factorization Machine layer

    Formula: y = 0.5 * (||sum(x)||^2 - sum(||x||^2))
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
        # Square of sum: (Σ x)^2
        sum_of_x = x.sum(axis=1)            # (batch, embed_dim)
        square_of_sum = sum_of_x * sum_of_x  # (batch, embed_dim)

        # Sum of squares: Σ x^2
        square_of_x = x * x                  # (batch, fields, embed_dim)
        sum_of_square = square_of_x.sum(axis=1)  # (batch, embed_dim)

        # FM interaction: 0.5 * (square_of_sum - sum_of_square)
        fm_interaction = (square_of_sum - sum_of_square) * 0.5

        if self.reduce_sum:
            return fm_interaction.sum(axis=1, keepdims=True)  # (batch, 1)
        return fm_interaction  # (batch, embed_dim)
```

### Complete DeepFM Implementation

```python
from nanotorch.nn import Module, Sequential, Linear, ReLU, Dropout, LayerNorm
from nanotorch.nn import Embedding

class DeepFM(Module):
    """DeepFM: FM + DNN for CTR Prediction

    Args:
        sparse_features: Sparse feature configuration list
        dense_features: Dense feature configuration list
        embed_dim: Embedding dimension
        hidden_dims: DNN hidden layer dimension list
        dropout: Dropout ratio
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

        # Shared embedding layer
        self.embeddings = {}
        for feat in sparse_features:
            emb = Embedding(feat.vocabulary_size, embed_dim)
            self.embeddings[feat.name] = emb
            self.register_module(f'emb_{feat.name}', emb)

        # FM component
        self.fm = FactorizationMachine(self.num_sparse, embed_dim)

        # DNN component
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

        # First-order linear term
        self.linear = Linear(self.num_sparse + self.num_dense, 1)

    def forward(self, sparse_input: Tensor, dense_input: Tensor = None) -> Tensor:
        """
        Args:
            sparse_input: (batch, num_sparse) integer indices
            dense_input: (batch, num_dense) float values

        Returns:
            (batch, 1) CTR probability
        """
        batch_size = sparse_input.shape[0]
        x_np = sparse_input.data.astype(np.int64)

        # Embed sparse features
        embedded_list = []
        for i, feat in enumerate(self.sparse_features):
            indices = Tensor(x_np[:, i])
            emb = self.embeddings[feat.name](indices)
            embedded_list.append(emb.data)

        # Stack: (batch, num_sparse, embed_dim)
        embedded = Tensor(
            np.stack(embedded_list, axis=1).astype(np.float32),
            requires_grad=True
        )

        # FM output
        fm_output = self.fm(embedded)  # (batch, 1)

        # DNN input: flatten embeddings + dense features
        embedded_flat = embedded.data.reshape(batch_size, -1)
        if dense_input is not None:
            dnn_input = np.concatenate([embedded_flat, dense_input.data], axis=1)
        else:
            dnn_input = embedded_flat

        dnn_input_tensor = Tensor(dnn_input.astype(np.float32), requires_grad=True)
        dnn_output = self.dnn(dnn_input_tensor)  # (batch, 1)

        # First-order linear output
        if dense_input is not None:
            linear_input = np.concatenate([x_np, dense_input.data], axis=1)
        else:
            linear_input = x_np
        linear_input_tensor = Tensor(linear_input.astype(np.float32), requires_grad=True)
        linear_output = self.linear(linear_input_tensor)

        # Combine: FM + DNN + Linear
        combined = fm_output.data + dnn_output.data + linear_output.data

        # Sigmoid activation
        output = Tensor(
            1.0 / (1.0 + np.exp(-np.clip(combined, -15, 15))),
            requires_grad=True
        )

        return output
```

---

## Training Process

### Training Configuration

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

### Training Loop

```python
from nanotorch.optim.adamw import AdamW
from nanotorch.optim.lr_scheduler import CosineWarmupScheduler
from nanotorch.nn import BCELoss
from nanotorch.utils import clip_grad_norm_

def train_model(model, train_loader, val_loader, config):
    """Complete training process"""

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        max_epochs=config.num_epochs
    )

    # Loss function
    criterion = BCELoss()

    # Early stopping
    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(1, config.num_epochs + 1):
        # === Training phase ===
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            sparse, dense, labels = batch

            # Forward pass
            predictions = model(
                Tensor(sparse.astype(np.float32)),
                Tensor(dense.astype(np.float32)) if dense is not None else None
            )

            loss = criterion(predictions, Tensor(labels.reshape(-1, 1)))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # === Validation phase ===
        val_auc = evaluate(model, val_loader)

        print(f"Epoch {epoch}: Loss={train_loss/len(train_loader):.4f}, "
              f"Val AUC={val_auc:.4f}")

        # Early stopping check
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

## Evaluation Metrics

### Classification Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **AUC** | Area under ROC curve | Probability that positive sample ranks higher than negative |
| **LogLoss** | $-\frac{1}{N}\sum(y\log(p) + (1-y)\log(1-p))$ | Cross-entropy loss |

### Ranking Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Recall@K** | $\frac{\|relevant \cap topK\|}{\|relevant\|}$ | Proportion of relevant items in Top-K |
| **NDCG@K** | $\frac{DCG@K}{IDCG@K}$ | Position-weighted ranking quality |
| **Hit@K** | $1[topK \cap relevant \neq \emptyset]$ | Whether Top-K contains relevant items |
| **MRR** | $\frac{1}{|Q|}\sum \frac{1}{rank_1}$ | Average reciprocal rank of first relevant item |

### Metric Implementation

```python
def auc_score(predictions, targets):
    """Calculate AUC"""
    sorted_indices = np.argsort(-predictions)
    sorted_targets = targets[sorted_indices]

    n_pos = np.sum(targets == 1)
    n_neg = np.sum(targets == 0)

    tp_cumsum = np.cumsum(sorted_targets == 1)
    auc = np.sum(tp_cumsum[sorted_targets == 0]) / (n_pos * n_neg)

    return auc

def ndcg_at_k(predictions, targets, k):
    """Calculate NDCG@K"""
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

## Complete Example

### Generate Data

```python
from examples.recommendation.data import generate_synthetic_data, create_dataloaders

# Generate synthetic recommendation data
dataset, sparse_configs, dense_configs = generate_synthetic_data(
    num_samples=100000,
    num_users=10000,
    num_items=5000,
    click_rate=0.05,
    random_seed=42
)

# Create data loaders
train_loader, val_loader, test_loader = create_dataloaders(
    dataset,
    batch_size=256,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
```

### Create Model

```python
from nanotorch.nn.recommender import DeepFM

model = DeepFM(
    sparse_features=sparse_configs,
    dense_features=dense_configs,
    embed_dim=16,
    hidden_dims=[256, 128, 64],
    dropout=0.1
)

# Print parameter count
n_params = sum(p.data.size for p in model.parameters())
print(f"Total parameters: {n_params:,}")
```

### Train and Evaluate

```python
from examples.recommendation.train import train_model, TrainingConfig
from examples.recommendation.evaluate import evaluate_model, print_evaluation_report

# Training configuration
config = TrainingConfig(
    num_epochs=20,
    learning_rate=0.001,
    early_stop_patience=3
)

# Train
history = train_model(model, train_loader, val_loader, config)

# Evaluate
test_metrics = evaluate_model(model, test_loader, ks=[1, 5, 10, 20])
print_evaluation_report(test_metrics, "DeepFM Test Results")
```

### Sample Output

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

## Model Comparison

### Supported Architectures

| Model | Features | Use Case |
|-------|----------|----------|
| **DeepFM** | FM + DNN, automatic feature crossing | CTR prediction |
| **Wide & Deep** | Linear + DNN, memorization + generalization | General recommendation |
| **NeuralCF** | GMF + MLP, collaborative filtering | User-item interaction |
| **Two-Tower** | Two-tower structure, user/item separation | Large-scale retrieval |
| **DCN** | Cross Network, explicit crossing | High-order feature interaction |

### Performance Comparison Example

```python
from nanotorch.nn.recommender import DeepFM, WideDeep, NeuralCF

models = {
    'DeepFM': DeepFM(sparse_configs, dense_configs, embed_dim=16),
    'Wide&Deep': WideDeep(sparse_configs, dense_configs, embed_dim=16),
    'NeuralCF': NeuralCF(num_users=10000, num_items=5000, embed_dim=16)
}

# Train and compare
results = {}
for name, model in models.items():
    history = train_model(model, train_loader, val_loader, config)
    metrics = evaluate_model(model, test_loader)
    results[name] = metrics

# Print comparison results
print(f"{'Model':<15} {'AUC':>8} {'LogLoss':>10}")
print("-" * 35)
for name, m in results.items():
    print(f"{name:<15} {m['auc']:>8.4f} {m['log_loss']:>10.4f}")
```

---

## Summary

In this chapter, we implemented a near-production-level recommendation system, covering:

| Component | File | Function |
|-----------|------|----------|
| FM Layer | `nanotorch/nn/fm.py` | Second-order feature crossing |
| Recommendation Models | `nanotorch/nn/recommender.py` | DeepFM, Wide&Deep, NeuralCF |
| Evaluation Metrics | `nanotorch/nn/metrics.py` | AUC, NDCG@K, Recall@K, MRR |
| Data Processing | `examples/recommendation/data.py` | Synthetic data generation |
| Training Process | `examples/recommendation/train.py` | Training loop, early stopping |
| Complete Example | `examples/recommendation/recommender_demo.py` | End-to-end demo |

### Key Formulas

**FM Efficient Computation**:
$$\sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2}\sum_{k}\left[\left(\sum_{i} v_{ik}x_i\right)^2 - \sum_{i} v_{ik}^2 x_i^2\right]$$

**DeepFM Output**:
$$\hat{y} = \sigma(y_{FM} + y_{DNN} + y_{linear})$$

### Further Reading

- [DeepFM Original Paper](https://arxiv.org/abs/1703.04247)
- [Wide & Deep Learning](https://arxiv.org/abs/1606.07792)
- [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)

---

**Previous Chapter**: [Chapter 15: Advanced Topics](15-advanced.md)

**Back to**: [Tutorial Overview](00-overview.md)
