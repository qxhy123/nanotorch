# Tutorial 12: Transformer Architecture

## Table of Contents

1. [Overview](#overview)
2. [Attention Mechanism](#attention-mechanism)
3. [Multi-Head Attention](#multi-head-attention)
4. [Transformer Encoder Layer](#transformer-encoder-layer)
5. [Transformer Decoder Layer](#transformer-decoder-layer)
6. [Positional Encoding](#positional-encoding)
7. [Complete Transformer](#complete-transformer)
8. [Usage Examples](#usage-examples)
9. [Summary](#summary)

---

## Overview

Transformer is a revolutionary architecture proposed by Google in 2017. It is entirely based on **attention mechanisms**, abandoning traditional recurrent and convolutional structures.

Core advantages:
- **Parallel Computation**: Unlike RNNs, doesn't require sequential processing
- **Long-range Dependencies**: Attention mechanism directly connects any positions
- **Scalability**: Easily scales to large models

Components implemented in nanotorch:
- `MultiheadAttention`: Multi-head self-attention
- `TransformerEncoderLayer`: Encoder layer
- `TransformerDecoderLayer`: Decoder layer
- `TransformerEncoder`: Encoder stack
- `TransformerDecoder`: Decoder stack

---

## Attention Mechanism

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ (Query): Query matrix (batch, seq_len, d_k)
- $K$ (Key): Key matrix (batch, seq_len, d_k)
- $V$ (Value): Value matrix (batch, seq_len, d_v)
- $d_k$: Dimension of Key

### Intuitive Understanding

```
Attention mechanism = Similarity between "query" and "key", weighted "value"

Question: "What color is the apple?"
      Q: Query vector of current word
      K: Key vectors of other words (representing "what I'm about")
      V: Value vectors of other words (representing "what information I contain")

Calculation process:
1. Q @ K^T: Calculate relevance of each word to current word
2. softmax: Normalize to probability distribution
3. @ V: Weighted sum based on relevance
```

### Implementation

```python
def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, 
    mask: Optional[Tensor] = None,
    dropout: Optional[float] = None
) -> Tuple[Tensor, Tensor]:
    """Scaled dot-product attention.
    
    Args:
        q: Query (batch, heads, seq_len, d_k)
        k: Key (batch, heads, seq_len, d_k)
        v: Value (batch, heads, seq_len, d_v)
        mask: Optional mask
        dropout: Dropout probability
    
    Returns:
        (output, attention_weights)
    """
    d_k = q.shape[-1]
    
    # Calculate attention scores
    scores = q.matmul(k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask (for decoder's causal attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax normalization
    attn_weights = scores.softmax(dim=-1)
    
    # Optional dropout
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    
    # Weighted sum
    output = attn_weights.matmul(v)
    
    return output, attn_weights
```

---

## Multi-Head Attention

### Principle

Multi-head attention projects Q, K, V into multiple subspaces, computes attention separately, then combines results:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

Where:
head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
```

### Benefits

1. **Multi-head** can simultaneously attend to different representation subspaces at different positions
2. **Parallel** multiple heads can be computed simultaneously
3. **Rich** each head can learn different types of dependencies

### Implementation

```python
# nanotorch/nn/attention.py

class MultiheadAttention(Module):
    """Multi-head attention layer.
    
    Args:
        embed_dim: Model dimension
        num_heads: Number of heads
        dropout: Dropout probability
        bias: Whether to use bias
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
        
        # Q, K, V projections
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = dropout

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward propagation.
        
        Args:
            query: (batch, seq_len, embed_dim)
            key: (batch, seq_len, embed_dim)
            value: (batch, seq_len, embed_dim)
            attn_mask: Optional attention mask
        
        Returns:
            (output, attention_weights)
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape to multi-head form: (batch, seq_len, num_heads, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, key.shape[1], self.num_heads, self.head_dim)
        v = v.reshape(batch_size, value.shape[1], self.num_heads, self.head_dim)
        
        # Transpose to: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention
        attn_output, attn_weights = scaled_dot_product_attention(
            q, k, v, attn_mask, self.dropout
        )
        
        # Merge multi-heads: (batch, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.embed_dim
        )
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output, attn_weights
```

### Usage Example

```python
from nanotorch.nn import MultiheadAttention

# Create multi-head attention layer
mha = MultiheadAttention(embed_dim=512, num_heads=8)

# Self-attention (query = key = value)
x = Tensor.randn((32, 100, 512))  # (batch, seq_len, embed_dim)
output, weights = mha(x, x, x)

print(output.shape)   # (32, 100, 512)
print(weights.shape)  # (32, 8, 100, 100)  # (batch, heads, seq, seq)
```

---

## Transformer Encoder Layer

### Structure

```
Input x
    │
    ├──────────────────┐
    ↓                  │
MultiheadAttention    │
    ↓                  │
Dropout               │
    ↓                  │
Add (residual) ←──────┘
    ↓
LayerNorm
    │
    ├──────────────────┐
    ↓                  │
FeedForward           │
    ↓                  │
Dropout               │
    ↓                  │
Add (residual) ←──────┘
    ↓
LayerNorm
    │
Output
```

### Implementation

```python
# nanotorch/nn/transformer.py

class TransformerEncoderLayer(Module):
    """Transformer encoder layer.
    
    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        dim_feedforward: Feed-forward network dimension
        dropout: Dropout probability
        activation: Activation function ('relu' or 'gelu')
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
        
        # Self-attention
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feed-forward network
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        # Normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Dropout
        self.dropout = Dropout(dropout)
        
        # Activation function
        self.activation = ReLU() if activation == "relu" else GELU()

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """Forward propagation.
        
        Args:
            src: Input (seq_len, batch, d_model) or (batch, seq_len, d_model)
            src_mask: Optional source sequence mask
        
        Returns:
            Output tensor
        """
        # Self-attention + residual + LayerNorm
        src2, _ = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # Feed-forward network + residual + LayerNorm
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        
        return src
```

### Encoder Stack

```python
class TransformerEncoder(Module):
    """Transformer encoder (multi-layer stack)."""

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

### Structure

Decoder has one more **Cross-Attention** than Encoder:

```
Input tgt (target sequence)
    │
    ├──────────────────┐
    ↓                  │
Masked Self-Attention │  (with causal mask)
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

### Implementation

```python
class TransformerDecoderLayer(Module):
    """Transformer decoder layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        
        # Self-attention (with mask)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Cross-attention
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feed-forward network
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        # Normalization
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
        """Forward propagation.
        
        Args:
            tgt: Target sequence
            memory: Encoder output
            tgt_mask: Target sequence causal mask
            memory_mask: Encoder output mask
        """
        # Self-attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention
        tgt2, _ = self.multihead_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt
```

---

## Positional Encoding

### Why Positional Encoding is Needed?

Transformer has no recurrent structure and cannot automatically perceive positional information. Positional encoding must be explicitly added.

### Sinusoidal Positional Encoding

```python
class PositionalEncoding(Module):
    """Sinusoidal positional encoding.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = Dropout(dropout)
        
        # Calculate positional encoding
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = Tensor(pe[np.newaxis, :, :])  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding.
        
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)
```

### Learnable Positional Encoding

```python
class LearnablePositionalEncoding(Module):
    """Learnable positional encoding."""

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

## Complete Transformer

### Seq2Seq Model

```python
class Transformer(Module):
    """Complete Transformer model."""

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
        # Encode
        memory = self.encoder(src, src_mask)
        
        # Decode
        output = self.decoder(tgt, memory, tgt_mask)
        
        return output
```

### Generate Causal Mask

```python
def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generate causal mask for decoder.
    
    Prevents position i from seeing information after position i.
    
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

## Usage Examples

### Text Classification (Encoder Only)

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
        # x: (batch, seq_len) word indices
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        
        # Use [CLS] position or mean pooling
        x = x.mean(dim=1)  # (batch, d_model)
        return self.fc(x)
```

### Machine Translation (Encoder-Decoder)

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

### GPT Style (Decoder Only)

```python
class GPTModel:
    """GPT-style autoregressive language model."""
    
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_model * 4)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        
        self.lm_head = Linear(d_model, vocab_size)
    
    def __call__(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # Causal mask
        mask = generate_square_subsequent_mask(x.shape[1])
        
        # For decoder-only, memory is itself
        output = self.decoder(x, x, tgt_mask=mask)
        
        return self.lm_head(output)
```

---

## Summary

This tutorial introduced the core components of Transformer architecture:

| Component | Function |
|-----------|----------|
| **MultiheadAttention** | Parallel computation of multiple attentions |
| **TransformerEncoder** | Encodes input sequence |
| **TransformerDecoder** | Decodes to generate target sequence |
| **PositionalEncoding** | Injects positional information |

### Key Points

1. **Attention mechanism** calculates relevance through Q, K, V
2. **Multi-head attention** allows model to attend to different subspaces
3. **Residual connection + LayerNorm** stabilizes deep network training
4. **Causal mask** prevents decoder from seeing future information

### Next Steps

In [Tutorial 13: Data Loading](13-dataloader.md), we will learn how to implement DataLoader and Dataset for efficiently loading and processing training data.

---

**References**:
- [Attention Is All You Need (Original Paper)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Transformers from Scratch](https://e2eml.school/transformers.html)
