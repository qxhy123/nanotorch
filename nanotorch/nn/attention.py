"""Attention layers for nanotorch Transformer models."""

import math
from typing import Optional, Tuple
import numpy as np
from nanotorch.nn.module import Module
from nanotorch.nn.linear import Linear
from nanotorch.nn.dropout import Dropout
from nanotorch.tensor import Tensor


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """Compute scaled dot product attention.

    Args:
        q: Query tensor of shape (batch, seq_len, head_dim)
        k: Key tensor of shape (batch, seq_len, head_dim)
        v: Value tensor of shape (batch, seq_len, head_dim)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to apply causal mask
        scale: Scale factor for attention scores

    Returns:
        Tuple of (output, attention_weights)
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    attn_weights = q.matmul(k.transpose(-2, -1)) * scale

    if is_causal:
        seq_len = q.shape[-2]
        causal_mask = np.triu(
            np.full((seq_len, seq_len), -np.inf, dtype=np.float32),
            k=1
        )
        attn_weights = attn_weights + Tensor(causal_mask)

    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask

    attn_weights = attn_weights.softmax(dim=-1)

    if dropout_p > 0.0:
        dropout = Dropout(dropout_p)
        attn_weights = dropout(attn_weights)

    output = attn_weights.matmul(v)
    return output, attn_weights


class MultiheadAttention(Module):
    """Multi-head attention mechanism.

    Allows the model to jointly attend to information from different
    representation subspaces at different positions.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads.
        dropout: Dropout probability on attn_output_weights. Default: 0.0.
        bias: If specified, adds bias to input / output projection layers.
            Default: True.
        add_bias_kv: If specified, adds bias to the key and value sequences
            at dim=0. Default: False.
        add_zero_attn: If specified, adds a new batch of zeros to the key
            and value sequences at dim=1. Default: False.
        kdim: Total number of features for keys. Default: None (uses embed_dim).
        vdim: Total number of features for values. Default: None (uses embed_dim).
        batch_first: If True, input is (batch, seq, feature). Default: True.

    Shape:
        - Inputs:
            - query: (batch, seq_len, embed_dim) if batch_first
            - key: (batch, seq_len, embed_dim) if batch_first
            - value: (batch, seq_len, embed_dim) if batch_first
        - Output: (batch, seq_len, embed_dim)

    Examples:
        >>> mha = MultiheadAttention(embed_dim=512, num_heads=8)
        >>> x = Tensor.randn((32, 10, 512))
        >>> output, _ = mha(x, x, x)
        >>> output.shape
        (32, 10, 512)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn

        if add_bias_kv:
            self.bias_k = Tensor(np.zeros((1, 1, embed_dim)), requires_grad=True)
            self.bias_v = Tensor(np.zeros((1, 1, embed_dim)), requires_grad=True)
            self.register_parameter("bias_k", self.bias_k)
            self.register_parameter("bias_v", self.bias_v)
        else:
            self.bias_k = None
            self.bias_v = None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass of multi-head attention.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            key_padding_mask: Mask for padded keys.
            need_weights: Whether to return attention weights.
            attn_mask: Attention mask.
            average_attn_weights: Whether to average attention weights across heads.
            is_causal: Whether to use causal attention.

        Returns:
            Tuple of (output, attention_weights).
        """
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        if self.bias_k is not None and self.bias_v is not None:
            k = k + self.bias_k
            v = v + self.bias_v

        q = q.reshape((batch_size, seq_len_q, self.num_heads, self.head_dim))
        q = q.transpose(1, 2)
        k = k.reshape((batch_size, seq_len_k, self.num_heads, self.head_dim))
        k = k.transpose(1, 2)
        v = v.reshape((batch_size, seq_len_k, self.num_heads, self.head_dim))
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = q.matmul(k.transpose(-2, -1)) * scale

        if is_causal:
            causal_mask = np.triu(
                np.full((seq_len_q, seq_len_k), -np.inf, dtype=np.float32),
                k=1
            )
            attn_weights = attn_weights + Tensor(causal_mask)

        if key_padding_mask is not None:
            mask = key_padding_mask.data[:, None, None, :]
            attn_weights = attn_weights + Tensor(mask * -1e9)

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = attn_weights.softmax(dim=-1)

        if self.dropout > 0.0 and self.training:
            dropout_mask = Tensor(
                (np.random.rand(*attn_weights.shape) > self.dropout).astype(np.float32)
                / (1.0 - self.dropout)
            )
            attn_weights = attn_weights * dropout_mask

        attn_output = attn_weights.matmul(v)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape((batch_size, seq_len_q, self.embed_dim))

        attn_output = self.out_proj(attn_output)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(axis=1)
            return attn_output, attn_weights
        return attn_output, None

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}"


class SelfAttention(Module):
    """Self-attention layer (simplified multi-head attention for self-attention).

    Args:
        embed_dim: Dimension of the input embeddings.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        bias: Whether to use bias in projections.

    Examples:
        >>> attn = SelfAttention(512, 8)
        >>> x = Tensor.randn((32, 10, 512))
        >>> output = attn(x)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.mha = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
        )

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        output, _ = self.mha(x, x, x, attn_mask=attn_mask, is_causal=is_causal)
        return output
