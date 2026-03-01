"""
Factorization Machine (FM) layer for recommendation systems.

FM efficiently models pairwise feature interactions using factorized parameters.
This is a core component of DeepFM and other recommendation architectures.

Reference:
    "Factorization Machines" (Rendle, 2010)
    https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
"""

import numpy as np
from typing import List
from nanotorch.nn.module import Module
from nanotorch.tensor import Tensor


class FactorizationMachine(Module):
    """Factorization Machine layer for second-order feature interactions.
    
    FM models pairwise feature interactions efficiently using factorized parameters.
    Instead of learning a separate weight w_ij for each feature pair (O(n^2) parameters),
    FM learns a low-rank decomposition v_i · v_j (O(nk) parameters).
    
    Mathematical Formulation:
        y_FM = w_0 + Σ w_i x_i + Σ_{i<j} <v_i, v_j> x_i x_j
             = w_0 + Σ w_i x_i + 0.5 * Σ_k ((Σ_i v_{ik} x_i)^2 - Σ_i v_{ik}^2 x_i^2)
    
    The efficient computation uses the identity:
        Σ_{i<j} <v_i, v_j> x_i x_j = 0.5 * (||Σ_i v_i x_i||^2 - Σ_i ||v_i x_i||^2)
    
    Args:
        num_fields: Number of feature fields (e.g., user_id, item_id, category, etc.)
        embed_dim: Dimension of factorized vectors (k in the formula).
        reduce_sum: If True, sum over the feature dimension. Default True.
    
    Shape:
        - Input: (batch_size, num_fields, embed_dim) - Embedded feature tensors
        - Output: 
            - If reduce_sum=True: (batch_size, 1) - FM output scores
            - If reduce_sum=False: (batch_size, embed_dim) - FM embeddings
    
    Example:
        >>> fm = FactorizationMachine(num_fields=10, embed_dim=32)
        >>> # Input: embedded features from embedding layer
        >>> embedded = embedding_layer(feature_indices)  # (B, num_fields, embed_dim)
        >>> fm_output = fm(embedded)  # (B, 1)
    """
    
    def __init__(
        self, 
        num_fields: int,
        embed_dim: int,
        reduce_sum: bool = True
    ) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.reduce_sum = reduce_sum
    
    def forward(self, x: Tensor) -> Tensor:
        """Compute FM second-order interactions.
        
        Uses the efficient O(nk) formula instead of naive O(n^2k) computation.
        
        Args:
            x: Input tensor of shape (batch_size, num_fields, embed_dim)
               This is typically the output of an embedding layer.
        
        Returns:
            FM output tensor. Shape depends on reduce_sum:
            - If True: (batch_size, 1) - scalar score per sample
            - If False: (batch_size, embed_dim) - vector per sample
        """
        # x shape: (batch_size, num_fields, embed_dim)
        # We compute: 0.5 * (||sum(x)||^2 - sum(||x||^2))
        
        # Square of sum: (Σ_i x_i)^2
        # Sum over fields dimension (dim=1)
        sum_of_x = x.sum(axis=1)  # (batch_size, embed_dim)
        square_of_sum = sum_of_x * sum_of_x  # (batch_size, embed_dim)
        
        # Sum of squares: Σ_i x_i^2
        square_of_x = x * x  # (batch_size, num_fields, embed_dim)
        sum_of_square = square_of_x.sum(axis=1)  # (batch_size, embed_dim)
        
        # FM interaction: 0.5 * (square_of_sum - sum_of_square)
        # This captures all pairwise feature interactions
        fm_interaction = (square_of_sum - sum_of_square) * 0.5  # (batch_size, embed_dim)
        
        if self.reduce_sum:
            # Sum over embedding dimension to get scalar score
            return fm_interaction.sum(axis=1, keepdims=True)  # (batch_size, 1)
        else:
            return fm_interaction  # (batch_size, embed_dim)
    
    def extra_repr(self) -> str:
        return f"num_fields={self.num_fields}, embed_dim={self.embed_dim}, reduce_sum={self.reduce_sum}"


class FieldAwareFactorizationMachine(Module):
    """Field-aware Factorization Machine (FFM) layer.
    
    FFM extends FM by learning separate embeddings for each field pair.
    This allows the model to capture field-specific interactions.
    
    Mathematical Formulation:
        y_FFM = Σ_{i<j} <v_{i,f_j}, v_{j,f_i}> x_i x_j
        
    where v_{i,f_j} is the embedding of feature i for interacting with field j.
    
    Note: FFM has O(n^2 * k) parameters but can capture more nuanced interactions.
    
    Args:
        num_fields: Number of feature fields.
        field_dims: List of vocabulary sizes for each field.
        embed_dim: Dimension of factorized vectors.
    
    Shape:
        - Input: (batch_size, num_fields) - Feature indices
        - Output: (batch_size, 1) - FFM output scores
    
    Reference:
        "Field-aware Factorization Machines for CTR Prediction" (Juan et al., 2016)
    """
    
    def __init__(
        self,
        num_fields: int,
        field_dims: List[int],
        embed_dim: int
    ) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        
        # Create embeddings for each field pair
        # embeddings[i][j] is the embedding for field i when interacting with field j
        from nanotorch.nn.embedding import Embedding
        
        self.embeddings = []
        for i in range(num_fields):
            field_embeddings = []
            for j in range(num_fields):
                if i != j:
                    emb = Embedding(field_dims[i], embed_dim)
                    field_embeddings.append(emb)
                    self.register_parameter(f"emb_{i}_{j}", emb.weight)
                else:
                    field_embeddings.append(None)
            self.embeddings.append(field_embeddings)
    
    def forward(self, x: Tensor) -> Tensor:
        """Compute FFM interactions.
        
        Args:
            x: Input tensor of shape (batch_size, num_fields) containing feature indices.
        
        Returns:
            FFM output of shape (batch_size, 1).
        """
        batch_size = x.shape[0]
        
        # Get all embeddings for all field pairs
        # This is expensive but captures field-specific interactions
        ffm_output = np.zeros((batch_size, 1), dtype=np.float32)
        
        x_np = x.data.astype(np.int64)
        
        for i in range(self.num_fields):
            for j in range(i + 1, self.num_fields):
                # Get embeddings for field i (for interacting with field j)
                emb_i = self.embeddings[i][j]
                # Get embeddings for field j (for interacting with field i)
                emb_j = self.embeddings[j][i]
                
                if emb_i is not None and emb_j is not None:
                    # Get the embeddings for this batch
                    v_i = emb_i(Tensor(x_np[:, i]))  # (batch_size, embed_dim)
                    v_j = emb_j(Tensor(x_np[:, j]))  # (batch_size, embed_dim)
                    
                    # Dot product interaction
                    interaction = (v_i * v_j).sum(axis=1, keepdims=True)  # (batch_size, 1)
                    ffm_output += interaction.data
        
        return Tensor(ffm_output, requires_grad=True)
    
    def extra_repr(self) -> str:
        return f"num_fields={self.num_fields}, embed_dim={self.embed_dim}"


class CrossNetwork(Module):
    """Cross Network for explicit bounded-degree feature interactions.
    
    Part of Deep & Cross Network (DCN), this module learns explicit feature
    crossings through cross layers.
    
    Mathematical Formulation:
        x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
                 = f(x_l, W_l, b_l) + x_l
        
    where x_0 is the original input and the recurrence forms polynomial
    feature crosses of increasing degree.
    
    Args:
        input_dim: Dimension of input features.
        num_layers: Number of cross layers (controls max interaction degree).
    
    Shape:
        - Input: (batch_size, input_dim)
        - Output: (batch_size, input_dim)
    
    Reference:
        "Deep & Cross Network for Ad Click Predictions" (Wang et al., 2017)
        https://arxiv.org/abs/1708.05123
    """
    
    def __init__(
        self,
        input_dim: int,
        num_layers: int = 3
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        
        # Each cross layer has a weight vector and bias
        self.weights = []
        self.biases = []
        
        for i in range(num_layers):
            w = Tensor(
                np.random.randn(input_dim).astype(np.float32) * 0.01,
                requires_grad=True
            )
            b = Tensor(
                np.zeros(input_dim, dtype=np.float32),
                requires_grad=True
            )
            self.weights.append(w)
            self.biases.append(b)
            self.register_parameter(f"weight_{i}", w)
            self.register_parameter(f"bias_{i}", b)
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply cross network layers.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
        
        Returns:
            Output tensor of shape (batch_size, input_dim).
        """
        x0 = x  # Store original input for all layers
        
        for i in range(self.num_layers):
            # Cross operation: x_0 * (W * x + b) + x
            # (batch, dim) * ((dim,) · (batch, dim) + (dim,)) + (batch, dim)
            
            # W · x: element-wise with broadcasting
            wx = x * self.weights[i]  # (batch_size, input_dim)
            
            # Sum over feature dimension for scalar interaction
            wx_sum = wx.sum(axis=1, keepdims=True)  # (batch_size, 1)
            
            # Add bias
            wxb = wx_sum + self.biases[i].sum()  # scalar per batch
            
            # x_0 * interaction + x (residual)
            x = x0 * wxb + x
        
        return x
    
    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, num_layers={self.num_layers}"


class Interaction(Module):
    """Feature interaction layer combining multiple interaction methods.
    
    This layer provides flexible feature interaction computation, supporting:
    - Dot product interactions (like FM)
    - Concatenation for DNN input
    - Cross network interactions
    
    Args:
        num_fields: Number of feature fields.
        embed_dim: Dimension of embeddings.
        interaction_type: Type of interaction ('fm', 'dot', 'concat', 'cross').
    
    Shape:
        - Input: (batch_size, num_fields, embed_dim)
        - Output: Depends on interaction_type
            - 'fm': (batch_size, 1)
            - 'dot': (batch_size, num_fields * (num_fields - 1) // 2)
            - 'concat': (batch_size, num_fields * embed_dim)
            - 'cross': (batch_size, embed_dim)
    """
    
    def __init__(
        self,
        num_fields: int,
        embed_dim: int,
        interaction_type: str = 'fm'
    ) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.interaction_type = interaction_type
        
        if interaction_type == 'fm':
            self.fm = FactorizationMachine(num_fields, embed_dim, reduce_sum=True)
        elif interaction_type == 'cross':
            self.cross = CrossNetwork(embed_dim, num_layers=2)
    
    def forward(self, x: Tensor) -> Tensor:
        """Compute feature interactions.
        
        Args:
            x: Input tensor of shape (batch_size, num_fields, embed_dim).
        
        Returns:
            Interaction output tensor.
        """
        if self.interaction_type == 'fm':
            return self.fm(x)
        
        elif self.interaction_type == 'dot':
            # Compute pairwise dot products
            batch_size = x.shape[0]
            interactions = []
            
            for i in range(self.num_fields):
                for j in range(i + 1, self.num_fields):
                    # Dot product between field i and field j embeddings
                    dot = (x[:, i, :] * x[:, j, :]).sum(axis=1, keepdims=True)
                    interactions.append(dot)
            
            # Concatenate all pairwise interactions
            return Tensor(
                np.concatenate([t.data for t in interactions], axis=1),
                requires_grad=x.requires_grad
            )
        
        elif self.interaction_type == 'concat':
            # Flatten all embeddings
            batch_size = x.shape[0]
            return x.reshape((batch_size, self.num_fields * self.embed_dim))
        
        elif self.interaction_type == 'cross':
            # Apply cross network after pooling
            x_pooled = x.sum(axis=1)  # (batch_size, embed_dim)
            return self.cross(x_pooled)
        
        else:
            raise ValueError(f"Unknown interaction type: {self.interaction_type}")
    
    def extra_repr(self) -> str:
        return f"num_fields={self.num_fields}, interaction_type='{self.interaction_type}'"
