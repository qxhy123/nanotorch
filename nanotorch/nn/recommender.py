"""
DeepFM and related recommendation models for nanotorch.

This module provides production-ready recommendation architectures including:
- DeepFM: Combines FM (low-order interactions) with DNN (high-order interactions)
- Wide & Deep: Linear model + DNN for memorization and generalization
- Two-Tower: Separate user and item encoders for retrieval
- NeuralCF: Neural collaborative filtering

Reference:
    "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"
    (Guo et al., 2017) https://arxiv.org/abs/1703.04247
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

from nanotorch.nn.module import Module, Sequential
from nanotorch.nn.linear import Linear
from nanotorch.nn.activation import ReLU
from nanotorch.nn.dropout import Dropout
from nanotorch.nn.normalization import LayerNorm
from nanotorch.nn.embedding import Embedding
from nanotorch.nn.fm import FactorizationMachine, CrossNetwork
from nanotorch.tensor import Tensor
from nanotorch.utils import cat, stack


@dataclass
class SparseFeat:
    """Configuration for a sparse (categorical) feature.
    
    Attributes:
        name: Feature name.
        vocabulary_size: Number of unique values.
        embedding_dim: Embedding dimension.
    """
    name: str
    vocabulary_size: int
    embedding_dim: int = 8


@dataclass
class DenseFeat:
    """Configuration for a dense (numerical) feature.
    
    Attributes:
        name: Feature name.
        dimension: Feature dimension (1 for scalar).
    """
    name: str
    dimension: int = 1


class FeatureEmbedding(Module):
    """Shared embedding layer for multiple sparse features.
    
    Creates separate embedding tables for each sparse feature and
    concatenates them into a unified representation.
    
    Args:
        sparse_features: List of SparseFeat configurations.
    
    Shape:
        - Input: (batch_size, num_sparse_features) - Integer indices
        - Output: (batch_size, num_sparse_features, embed_dim) - Stacked embeddings
    """
    
    def __init__(self, sparse_features: List[SparseFeat]) -> None:
        super().__init__()
        self.sparse_features = sparse_features
        self.num_features = len(sparse_features)
        
        self.embeddings = {}
        for feat in sparse_features:
            emb = Embedding(feat.vocabulary_size, feat.embedding_dim)
            self.embeddings[feat.name] = emb
            self.register_module(feat.name, emb)
    
    def forward(self, x: Tensor) -> Tensor:
        """Look up embeddings for all sparse features.
        
        Args:
            x: (batch_size, num_sparse_features) integer tensor
        
        Returns:
            (batch_size, num_sparse_features, embed_dim) stacked embeddings
            where embed_dim varies per feature
        """
        # batch_size = x.shape[0]
        
        embedded_list = []
        x_np = x.data.astype(np.int64)
        
        for i, feat in enumerate(self.sparse_features):
            indices = Tensor(x_np[:, i])
            emb = self.embeddings[feat.name](indices)  # (batch, embed_dim)
            embedded_list.append(emb)
        
        return stack(embedded_list, dim=1)


class FMLayer(Module):
    """Factorization Machine layer with first and second order terms.
    
    Combines linear (first-order) and pairwise interaction (second-order)
    components of FM.
    
    Args:
        num_fields: Number of feature fields.
        embed_dim: Embedding dimension (should be same for all features).
    
    Shape:
        - Input: (batch_size, num_fields, embed_dim)
        - Output: (batch_size, 1)
    """
    
    def __init__(self, num_fields: int, embed_dim: int) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.fm = FactorizationMachine(num_fields, embed_dim, reduce_sum=True)
    
    def forward(self, x: Tensor) -> Tensor:
        """Compute FM output.
        
        Args:
            x: (batch_size, num_fields, embed_dim) embedded features
        
        Returns:
            (batch_size, 1) FM scores
        """
        return self.fm(x)


class DNNLayer(Module):
    """Deep Neural Network for high-order feature interactions.
    
    Standard MLP with optional LayerNorm and Dropout.
    
    Args:
        input_dim: Input dimension.
        hidden_dims: List of hidden layer dimensions.
        output_dim: Output dimension.
        dropout: Dropout rate.
        use_layernorm: Whether to use LayerNorm after each layer.
    
    Shape:
        - Input: (batch_size, input_dim)
        - Output: (batch_size, output_dim)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        dropout: float = 0.1,
        use_layernorm: bool = True
    ) -> None:
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(Linear(prev_dim, hidden_dim))
            if use_layernorm:
                layers.append(LayerNorm(hidden_dim))
            layers.append(ReLU())
            if dropout > 0:
                layers.append(Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(Linear(prev_dim, output_dim))
        
        self.mlp = Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through MLP.
        
        Args:
            x: (batch_size, input_dim)
        
        Returns:
            (batch_size, output_dim)
        """
        return self.mlp(x)


class DeepFM(Module):
    """DeepFM: Factorization-Machine based Neural Network for CTR Prediction.
    
    DeepFM combines the strengths of Factorization Machines (for low-order
    feature interactions) and Deep Neural Networks (for high-order interactions)
    in a shared embedding architecture.
    
    Architecture:
        ┌─────────────────────────────────────┐
        │        Sparse Features              │
        │  (user_id, item_id, category, ...)  │
        └─────────────────────────────────────┘
                        │
                        ▼
        ┌─────────────────────────────────────┐
        │      Shared Embedding Layer         │
        │   (Feature → Dense Vector)          │
        └─────────────────────────────────────┘
                        │
           ┌────────────┼────────────┐
           ▼            ▼            ▼
        ┌──────┐   ┌─────────┐   ┌───────┐
        │ FM   │   │  DNN    │   │ Bias  │
        │(2nd) │   │(High)   │   │(1st)  │
        └──────┘   └─────────┘   └───────┘
           │            │            │
           └────────────┼────────────┘
                        ▼
                  ┌──────────┐
                  │  CTR     │
                  │  Score   │
                  └──────────┘
    
    Args:
        sparse_features: List of SparseFeat configurations.
        dense_features: List of DenseFeat configurations.
        embed_dim: Embedding dimension (must be same for all sparse features).
        hidden_dims: List of DNN hidden layer dimensions.
        dropout: Dropout rate for DNN.
        use_layernorm: Whether to use LayerNorm in DNN.
    
    Shape:
        - sparse_input: (batch_size, num_sparse_features) integer indices
        - dense_input: (batch_size, num_dense_features) float values
        - Output: (batch_size, 1) CTR probability
    
    Example:
        >>> sparse_features = [
        ...     SparseFeat('user_id', 10000, 32),
        ...     SparseFeat('item_id', 5000, 32),
        ... ]
        >>> dense_features = [DenseFeat('price', 1)]
        >>> model = DeepFM(sparse_features, dense_features, embed_dim=32)
        >>> sparse_input = Tensor([[1, 100], [2, 200]])  # (2, 2)
        >>> dense_input = Tensor([[9.99], [19.99]])       # (2, 1)
        >>> output = model(sparse_input, dense_input)     # (2, 1)
    """
    
    def __init__(
        self,
        sparse_features: List[SparseFeat],
        dense_features: List[DenseFeat],
        embed_dim: int = 16,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
        use_layernorm: bool = True
    ) -> None:
        super().__init__()
        
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.embed_dim = embed_dim
        self.num_sparse = len(sparse_features)
        self.num_dense = len(dense_features)
        
        total_dense_dim = sum(f.dimension for f in dense_features)
        
        self.embeddings = {}
        for feat in sparse_features:
            emb = Embedding(feat.vocabulary_size, embed_dim)
            self.embeddings[feat.name] = emb
            self.register_module(f'emb_{feat.name}', emb)
        
        self.fm = FactorizationMachine(self.num_sparse, embed_dim, reduce_sum=True)
        
        dnn_input_dim = self.num_sparse * embed_dim + total_dense_dim
        self.dnn = DNNLayer(
            input_dim=dnn_input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout=dropout,
            use_layernorm=use_layernorm
        )
        
        self.linear = Linear(self.num_sparse + self.num_dense, 1)
    
    def forward(
        self,
        sparse_input: Tensor,
        dense_input: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass computing CTR prediction.
        
        Args:
            sparse_input: (batch_size, num_sparse_features) integer indices
            dense_input: (batch_size, num_dense_features) float values
        
        Returns:
            (batch_size, 1) CTR probability (after sigmoid)
        """
        batch_size = sparse_input.shape[0]
        x_sparse_np = sparse_input.data.astype(np.int64)
        
        # Embed sparse features
        embedded_list = []
        for i, feat in enumerate(self.sparse_features):
            indices = Tensor(x_sparse_np[:, i])
            emb = self.embeddings[feat.name](indices)
            embedded_list.append(emb)
        
        # Stack embeddings: (batch, num_sparse, embed_dim)
        embedded = stack(embedded_list, dim=1)
        
        # FM component: second-order interactions
        fm_output = self.fm(embedded)  # (batch, 1)
        
        # Prepare DNN input: flatten embeddings + dense features
        embedded_flat = embedded.reshape((batch_size, -1))  # (batch, num_sparse * embed_dim)
        
        if dense_input is not None and self.num_dense > 0:
            dnn_input_tensor = cat([embedded_flat, dense_input], dim=1)
        else:
            dnn_input_tensor = embedded_flat
        dnn_output = self.dnn(dnn_input_tensor)  # (batch, 1)
        
        # Linear component: first-order interactions
        sparse_linear_input = Tensor(x_sparse_np.astype(np.float32), requires_grad=False)
        if dense_input is not None and self.num_dense > 0:
            linear_input_tensor = cat([sparse_linear_input, dense_input], dim=1)
        else:
            linear_input_tensor = sparse_linear_input
        linear_output = self.linear(linear_input_tensor)  # (batch, 1)
        
        # Combine: FM + DNN + Linear
        combined = fm_output + dnn_output + linear_output
        
        return combined.sigmoid()


class WideDeep(Module):
    """Wide & Deep model for recommendation.
    
    Combines a wide linear model (memorization) with a deep neural network
    (generalization) using shared embeddings.
    
    Architecture:
        Wide: Sparse features → Linear → ┐
                                        ├→ Sum → Sigmoid
        Deep: Embedding → MLP ───────────┘
    
    Args:
        sparse_features: List of SparseFeat configurations.
        dense_features: List of DenseFeat configurations.
        embed_dim: Embedding dimension.
        hidden_dims: List of DNN hidden dimensions.
        dropout: Dropout rate.
    
    Reference:
        "Wide & Deep Learning for Recommender Systems" (Cheng et al., 2016)
    """
    
    def __init__(
        self,
        sparse_features: List[SparseFeat],
        dense_features: List[DenseFeat],
        embed_dim: int = 16,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.num_sparse = len(sparse_features)
        self.num_dense = len(dense_features)
        
        total_dense_dim = sum(f.dimension for f in dense_features)
        
        self.embeddings = {}
        for feat in sparse_features:
            emb = Embedding(feat.vocabulary_size, embed_dim)
            self.embeddings[feat.name] = emb
            self.register_module(f'emb_{feat.name}', emb)
        
        self.wide = Linear(self.num_sparse + self.num_dense, 1)
        
        dnn_input_dim = self.num_sparse * embed_dim + total_dense_dim
        self.deep = DNNLayer(
            input_dim=dnn_input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout=dropout
        )
    
    def forward(
        self,
        sparse_input: Tensor,
        dense_input: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass.
        
        Args:
            sparse_input: (batch, num_sparse) integer indices
            dense_input: (batch, num_dense) float values
        
        Returns:
            (batch, 1) probability
        """
        x_sparse_np = sparse_input.data.astype(np.int64)
        
        # Embed sparse features
        embedded_list = []
        for i, feat in enumerate(self.sparse_features):
            indices = Tensor(x_sparse_np[:, i])
            emb = self.embeddings[feat.name](indices)
            embedded_list.append(emb)
        
        embedded_flat = cat(embedded_list, dim=1)
        
        # Wide component
        sparse_wide_input = Tensor(x_sparse_np.astype(np.float32), requires_grad=False)
        if dense_input is not None and self.num_dense > 0:
            wide_input_tensor = cat([sparse_wide_input, dense_input], dim=1)
        else:
            wide_input_tensor = sparse_wide_input
        wide_output = self.wide(wide_input_tensor)
        
        # Deep component
        if dense_input is not None and self.num_dense > 0:
            deep_input_tensor = cat([embedded_flat, dense_input], dim=1)
        else:
            deep_input_tensor = embedded_flat
        deep_output = self.deep(deep_input_tensor)

        combined = wide_output + deep_output
        return combined.sigmoid()


class TwoTowerModel(Module):
    """Two-Tower model for recommendation retrieval.
    
    Separates user and item encoding into two independent towers,
    then computes similarity via dot product.
    
    Architecture:
        User Tower: user_features → MLP → user_embedding ─┐
                                                           ├→ dot → score
        Item Tower: item_features → MLP → item_embedding ─┘
    
    Args:
        user_sparse_features: List of user sparse features.
        user_dense_features: List of user dense features.
        item_sparse_features: List of item sparse features.
        item_dense_features: List of item dense features.
        embed_dim: Embedding dimension for sparse features.
        tower_dim: Output dimension of each tower.
        hidden_dims: Hidden dimensions for towers.
    
    Shape:
        - user_sparse: (batch, num_user_sparse)
        - user_dense: (batch, num_user_dense)
        - item_sparse: (batch, num_item_sparse)
        - item_dense: (batch, num_item_dense)
        - Output: (batch, 1) similarity score
    """
    
    def __init__(
        self,
        user_sparse_features: List[SparseFeat],
        user_dense_features: List[DenseFeat],
        item_sparse_features: List[SparseFeat],
        item_dense_features: List[DenseFeat],
        embed_dim: int = 16,
        tower_dim: int = 64,
        hidden_dims: List[int] = [128, 64]
    ) -> None:
        super().__init__()
        
        self.user_sparse = user_sparse_features
        self.user_dense = user_dense_features
        self.item_sparse = item_sparse_features
        self.item_dense = item_dense_features
        self.tower_dim = tower_dim
        
        user_dense_dim = sum(f.dimension for f in user_dense_features)
        item_dense_dim = sum(f.dimension for f in item_dense_features)
        
        self.user_embeddings = {}
        for feat in user_sparse_features:
            emb = Embedding(feat.vocabulary_size, embed_dim)
            self.user_embeddings[feat.name] = emb
            self.register_module(f'user_emb_{feat.name}', emb)
        
        self.item_embeddings = {}
        for feat in item_sparse_features:
            emb = Embedding(feat.vocabulary_size, embed_dim)
            self.item_embeddings[feat.name] = emb
            self.register_module(f'item_emb_{feat.name}', emb)
        
        user_tower_input = len(user_sparse_features) * embed_dim + user_dense_dim
        self.user_tower = DNNLayer(
            input_dim=user_tower_input,
            hidden_dims=hidden_dims,
            output_dim=tower_dim,
            dropout=0.1
        )
        
        item_tower_input = len(item_sparse_features) * embed_dim + item_dense_dim
        self.item_tower = DNNLayer(
            input_dim=item_tower_input,
            hidden_dims=hidden_dims,
            output_dim=tower_dim,
            dropout=0.1
        )
    
    def _embed_features(
        self,
        sparse_input: Tensor,
        sparse_features: List[SparseFeat],
        embeddings: Dict[str, Embedding]
    ) -> Tensor:
        """Embed sparse features and return concatenated tensor."""
        x_np = sparse_input.data.astype(np.int64)
        embedded_list = []
        for i, feat in enumerate(sparse_features):
            indices = Tensor(x_np[:, i])
            emb = embeddings[feat.name](indices)
            embedded_list.append(emb)
        return cat(embedded_list, dim=1)
    
    def encode_user(
        self,
        user_sparse: Tensor,
        user_dense: Optional[Tensor] = None
    ) -> Tensor:
        """Encode user features to embedding.
        
        Args:
            user_sparse: (batch, num_user_sparse)
            user_dense: (batch, num_user_dense)
        
        Returns:
            (batch, tower_dim) user embedding
        """
        embedded = self._embed_features(user_sparse, self.user_sparse, self.user_embeddings)
        
        if user_dense is not None:
            tower_input_tensor = cat([embedded, user_dense], dim=1)
        else:
            tower_input_tensor = embedded

        return self.user_tower(tower_input_tensor)
    
    def encode_item(
        self,
        item_sparse: Tensor,
        item_dense: Optional[Tensor] = None
    ) -> Tensor:
        """Encode item features to embedding.
        
        Args:
            item_sparse: (batch, num_item_sparse)
            item_dense: (batch, num_item_dense)
        
        Returns:
            (batch, tower_dim) item embedding
        """
        embedded = self._embed_features(item_sparse, self.item_sparse, self.item_embeddings)
        
        if item_dense is not None:
            tower_input_tensor = cat([embedded, item_dense], dim=1)
        else:
            tower_input_tensor = embedded

        return self.item_tower(tower_input_tensor)
    
    def forward(
        self,
        user_sparse: Tensor,
        item_sparse: Tensor,
        user_dense: Optional[Tensor] = None,
        item_dense: Optional[Tensor] = None
    ) -> Tensor:
        """Compute user-item similarity score.
        
        Args:
            user_sparse: (batch, num_user_sparse)
            item_sparse: (batch, num_item_sparse)
            user_dense: (batch, num_user_dense)
            item_dense: (batch, num_item_dense)
        
        Returns:
            (batch, 1) similarity score
        """
        user_emb = self.encode_user(user_sparse, user_dense)
        item_emb = self.encode_item(item_sparse, item_dense)
        
        similarity = (user_emb * item_emb).sum(axis=1, keepdims=True)
        
        return similarity.sigmoid()


class NeuralCF(Module):
    """Neural Collaborative Filtering (NCF) model.
    
    Combines matrix factorization (GMF) with multi-layer perceptron (MLP)
    for user-item recommendation.
    
    Architecture:
        GMF branch: user_emb * item_emb ────────┐
                                             ├→ Concat → Output
        MLP branch: [user_emb, item_emb] → MLP ┘
    
    Args:
        num_users: Number of users.
        num_items: Number of items.
        embed_dim: Embedding dimension.
        hidden_dims: MLP hidden dimensions.
    
    Reference:
        "Neural Collaborative Filtering" (He et al., 2017)
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 32,
        hidden_dims: List[int] = [128, 64, 32]
    ) -> None:
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        
        self.user_embedding_gmf = Embedding(num_users, embed_dim)
        self.item_embedding_gmf = Embedding(num_items, embed_dim)
        self.register_module('user_embedding_gmf', self.user_embedding_gmf)
        self.register_module('item_embedding_gmf', self.item_embedding_gmf)
        
        self.user_embedding_mlp = Embedding(num_users, embed_dim)
        self.item_embedding_mlp = Embedding(num_items, embed_dim)
        self.register_module('user_embedding_mlp', self.user_embedding_mlp)
        self.register_module('item_embedding_mlp', self.item_embedding_mlp)
        
        self.mlp = DNNLayer(
            input_dim=embed_dim * 2,
            hidden_dims=hidden_dims,
            output_dim=embed_dim,
            dropout=0.1
        )
        
        self.output_layer = Linear(embed_dim * 2, 1)
    
    def forward(self, user_indices: Tensor, item_indices: Tensor) -> Tensor:
        """Compute user-item interaction score.
        
        Args:
            user_indices: (batch,) user IDs
            item_indices: (batch,) item IDs
        
        Returns:
            (batch, 1) interaction probability
        """
        user_np = user_indices.data.astype(np.int64)
        item_np = item_indices.data.astype(np.int64)
        
        user_idx = Tensor(user_np)
        item_idx = Tensor(item_np)
        
        user_gmf = self.user_embedding_gmf(user_idx)
        item_gmf = self.item_embedding_gmf(item_idx)
        gmf_output = user_gmf * item_gmf  # (batch, embed_dim)
        
        user_mlp = self.user_embedding_mlp(user_idx)
        item_mlp = self.item_embedding_mlp(item_idx)
        mlp_input = cat([user_mlp, item_mlp], dim=1)
        mlp_output = self.mlp(mlp_input)  # (batch, embed_dim)

        combined = cat([gmf_output, mlp_output], dim=1)
        logits = self.output_layer(combined)  # (batch, 1)

        return logits.sigmoid()


class DCN(DeepFM):
    """Deep & Cross Network for CTR prediction.
    
    Replaces FM component with Cross Network for explicit feature crossing.
    
    Args:
        Same as DeepFM, plus:
        num_cross_layers: Number of cross layers.
    """
    
    def __init__(
        self,
        sparse_features: List[SparseFeat],
        dense_features: List[DenseFeat],
        embed_dim: int = 16,
        hidden_dims: List[int] = [256, 128, 64],
        num_cross_layers: int = 3,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            sparse_features, dense_features, embed_dim,
            hidden_dims, dropout
        )
        
        total_dense_dim = sum(f.dimension for f in dense_features)
        cross_input_dim = len(sparse_features) * embed_dim + total_dense_dim
        self.cross_network = CrossNetwork(cross_input_dim, num_cross_layers)
    
    def forward(
        self,
        sparse_input: Tensor,
        dense_input: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass with cross network."""
        x_sparse_np = sparse_input.data.astype(np.int64)
        
        embedded_list = []
        for i, feat in enumerate(self.sparse_features):
            indices = Tensor(x_sparse_np[:, i])
            emb = self.embeddings[feat.name](indices)
            embedded_list.append(emb)
        
        embedded_flat = cat(embedded_list, dim=1)
        
        if dense_input is not None and self.num_dense > 0:
            combined_tensor = cat([embedded_flat, dense_input], dim=1)
        else:
            combined_tensor = embedded_flat
        
        cross_output = self.cross_network(combined_tensor)
        dnn_output = self.dnn(combined_tensor)
        
        combined = cross_output + dnn_output
        
        return combined[:, :1].sigmoid()
