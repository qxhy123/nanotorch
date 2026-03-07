# Tutorial 12: Transformer

## In 2017, a Revolution Quietly Occurred...

Before that, natural language processing was dominated by "recurrence". RNN, LSTM, GRU—they were like diligent readers, reading word by word from left to right.

This approach has a fatal weakness: **cannot parallelize**. You have to finish reading the last word to understand the whole sentence. Training time was long, and long sentences were a nightmare.

Then, a paper appeared: "Attention Is All You Need".

It asked: Why must we read from left to right? Why can't we see the whole text at once? Why can't every word directly "see" all other words?

This is **self-attention**.

```
RNN's approach:
  "I" → "love" → "you"
  Sequential reading, like turning pages

Transformer's approach:
  "I" ←→ "love" ←→ "you"
  Every word can directly see every other word
  Like spreading the book open, seeing everything at once
```

**Transformer replaced "recurrence" with "attention".** It taught machines: when understanding a sentence, instead of reading word by word, simultaneously attend to all words and find relationships between them.

From then on, the era of large models began.

---

## 12.1 Attention Mechanism

### Life Analogy

```
You're looking at a photo:

  👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦
  A group of people at a party

Your attention:
  - Looking for a friend: mainly look at "faces"
  - Counting people: look at "heads"
  - Looking at background: look at "environment"

Same scene, you "focus" on different points
→ This is attention
```

### Core Idea

```
Attention mechanism: Query(Q) × Key(K) × Value(V)

Library analogy:
  Q (Query) = What you're looking for (query)
  K (Key)   = Each book's label (key)
  V (Value) = Book's content (value)

Process:
  1. Compare Q with all K's (relevance)
  2. Get attention weights (which books are relevant)
  3. Use weights to combine V's (get relevant content)
```

### Self-Attention

```
Self-attention: Q, K, V all come from the same input

Sentence: "I love Beijing Tiananmen"

Q = representation of "love" (what I'm looking for)
K = representations of all words (each word's label)
V = representations of all words (each word's content)

Calculate: relevance of "love" with each word
  - Relevant to "I": 0.7 (subject)
  - Relevant to "love": 0.1 (self)
  - Relevant to "Beijing": 0.15 (object)
  - Relevant to "Tiananmen": 0.05 (modifier)

Use these weights to combine all words → new representation of "love"
```

---

## 12.2 Attention Calculation

### Formula

```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

Step breakdown:
  1. Q @ K^T     : Calculate relevance scores
  2. / √d_k      : Scale (prevent gradient vanishing)
  3. softmax     : Normalize to probabilities
  4. @ V         : Weighted sum
```

### Diagram

```
Input sequence: [word1, word2, word3]

Q = [q1]    K^T = [k1, k2, k3]    V = [v1]
    [q2]                           [v2]
    [q3]                           [v3]

Step 1: Q @ K^T = relevance matrix
        word1  word2  word3
word1 [ 0.9    0.3    0.1 ]    word1 most relevant to word1
word2 [ 0.2    0.8    0.4 ]
word3 [ 0.1    0.3    0.9 ]

Step 2: softmax = attention weights
        word1  word2  word3
word1 [ 0.70   0.20   0.10 ]
word2 [ 0.15   0.60   0.25 ]
word3 [ 0.10   0.25   0.65 ]

Step 3: @ V = weighted combination
Output = [0.7*v1 + 0.2*v2 + 0.1*v3]
         [0.15*v1 + 0.6*v2 + 0.25*v3]
         [0.1*v1 + 0.25*v2 + 0.65*v3]
```

### Implementation

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Scaled dot-product attention

    Analogy:
      q = query (what I'm looking for)
      k = key (each book's label)
      v = value (book's content)
    """
    d_k = q.shape[-1]

    # 1. Calculate relevance scores
    scores = q.matmul(k.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. Apply mask (optional)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 3. softmax normalization
    attn_weights = scores.softmax(dim=-1)

    # 4. Weighted sum
    output = attn_weights.matmul(v)

    return output, attn_weights
```

---

## 12.3 Multi-Head Attention

### Why Multi-Head?

```
Single-head attention:
  Can only learn one type of "relevance"

Multi-head attention:
  Can learn multiple types of "relevance"

Example:
  Head 1: Focus on syntactic relationships (subject-verb)
  Head 2: Focus on semantic relationships (synonyms)
  Head 3: Focus on positional relationships (adjacent words)
  ...

Each head learns independently, then combines
```

### Diagram

```
Input X (batch, seq_len, d_model)
         │
         ├──→ W_Q → Q ──┐
         ├──→ W_K → K ──┼──→ Head 1 attention ──┐
         └──→ W_V → V ──┘                       │
                                                  │
         ┌───────────────────────────────────────┘
         │
         ├──→ W_Q → Q ──┐
         ├──→ W_K → K ──┼──→ Head 2 attention ──┼──→ Concat ──→ W_O ──→ Output
         └──→ W_V → V ──┘                       │
                                                  │
         ┌───────────────────────────────────────┘
         │
         ... (more heads)
```

### Implementation

```python
class MultiheadAttention(Module):
    """
    Multi-head attention

    Analogy: Multiple experts looking at the problem from different angles
    """

    def __init__(
        self,
        embed_dim: int,      # Model dimension
        num_heads: int,      # Number of heads
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim

        # Q, K, V projections
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        # 1. Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2. Reshape to multi-head (batch, seq, heads, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, key.shape[1], self.num_heads, self.head_dim)
        v = v.reshape(batch_size, value.shape[1], self.num_heads, self.head_dim)

        # 3. Transpose (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 4. Calculate attention
        attn_output, attn_weights = scaled_dot_product_attention(
            q, k, v, attn_mask
        )

        # 5. Merge multi-heads
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.embed_dim
        )

        # 6. Output projection
        output = self.out_proj(attn_output)

        return output, attn_weights
```

### Usage

```python
# Create multi-head attention
mha = MultiheadAttention(embed_dim=512, num_heads=8)

# Self-attention (Q = K = V)
x = Tensor.randn((32, 100, 512))  # (batch, seq_len, embed_dim)
output, weights = mha(x, x, x)

print(output.shape)    # (32, 100, 512)
print(weights.shape)   # (32, 8, 100, 100)  attention weights
```

---

## 12.4 Transformer Encoder

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
Add ←─────────────────┘ (residual connection)
   ↓
LayerNorm
   │
   ├──────────────────┐
   ↓                  │
FeedForward           │
   ↓                  │
Dropout               │
   ↓                  │
Add ←─────────────────┘ (residual connection)
   ↓
LayerNorm
   │
Output
```

### Implementation

```python
class TransformerEncoderLayer(Module):
    """
    Transformer encoder layer

    Structure: attention → residual+normalization → feedforward → residual+normalization
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = MultiheadAttention(d_model, nhead, dropout)

        # Feed-forward network
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)

        # Normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # Dropout
        self.dropout = Dropout(dropout)

    def forward(self, src, src_mask=None):
        # 1. Self-attention + residual + LayerNorm
        src2, _ = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout(src2)  # Residual connection
        src = self.norm1(src)

        # 2. Feed-forward network + residual + LayerNorm
        src2 = self.linear2(self.dropout(self.linear1(src).relu()))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src
```

### Residual Connection

```
Why do we need residual connections?

Problem: Deep networks are hard to train
  Input → Layer1 → Layer2 → ... → Layer100 → Output
  Gradient has to propagate 100 layers, easily vanishes

Solution: Residual connection
  Input → Layer → + → Output
         ↑_____|
  Gradient can directly "skip" some layers

Analogy:
  Without: Every layer must be perfect
  With: Every layer only needs to learn "residual" (difference from input)
```

---

## 12.5 Transformer Decoder

### Difference from Encoder

```
Decoder has one more "cross-attention":

Encoder output ────→ cross-attention's K, V
                     ↓
Target sequence ──→ self-attention → cross-attention → feedforward → output
              (with mask)
```

### Causal Mask

```
Decoder's self-attention needs a "causal mask":

Why?
  - During decoding, can't see future words
  - During translation, can't see parts not yet translated

Mask example (4 positions):
  [[1, 0, 0, 0],   Position 1: can only see position 1
   [1, 1, 0, 0],   Position 2: can see positions 1,2
   [1, 1, 1, 0],   Position 3: can see positions 1,2,3
   [1, 1, 1, 1]]   Position 4: can see positions 1,2,3,4

0 positions will be set to -inf, becomes 0 after softmax
```

### Generate Mask

```python
def generate_square_subsequent_mask(sz: int):
    """
    Generate causal mask

    Prevents seeing future information
    """
    mask = np.triu(np.ones((sz, sz)), k=1) == 0
    # Upper triangle is False, lower triangle is True
    return Tensor(mask.astype(np.float32))
```

---

## 12.6 Positional Encoding

### Why Do We Need It?

```
Transformer has no "recurrence", all positions processed simultaneously

Problem:
  "I love you" and "You love me" are processed the same way
  But they mean completely different things!

Solution:
  Add a "position label" to each position
  Let the model know "I'm the Nth word"
```

### Sinusoidal Positional Encoding

```python
class PositionalEncoding(Module):
    """
    Sinusoidal positional encoding

    Encodes position using sine waves of different frequencies

    Position 0: sin(0), cos(0), sin(0), cos(0), ...
    Position 1: sin(1), cos(1), sin(0.01), cos(0.01), ...
    Position 2: sin(2), cos(2), sin(0.02), cos(0.02), ...
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = Dropout(dropout)

        # Calculate positional encoding
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)  # Even positions
        pe[:, 1::2] = np.cos(position * div_term)  # Odd positions

        self.pe = Tensor(pe[np.newaxis, :, :])  # (1, max_len, d_model)

    def forward(self, x: Tensor):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)
```

---

## 12.7 Complete Transformer

### Architecture Diagram

```
┌─────────────────────────────────────────┐
│              Transformer                │
├─────────────────────────────────────────┤
│                                         │
│  Source language ──→ Embedding ──→ + Positional encoding
│                    ↓                    │
│            ┌──────────────┐             │
│            │   Encoder    │ × N layers  │
│            └──────────────┘             │
│                    ↓                    │
│               Memory                    │
│                    ↓                    │
│  Target language ──→ Embedding ──→ + Positional encoding
│                    ↓                    │
│            ┌──────────────┐             │
│            │   Decoder    │ × N layers  │
│            └──────────────┘             │
│                    ↓                    │
│              Linear                     │
│                    ↓                    │
│             Softmax                     │
│                    ↓                    │
│           Output probabilities          │
│                                         │
└─────────────────────────────────────────┘
```

### Usage Example

```python
class TransformerClassifier:
    """
    Text classification using Transformer (Encoder only)
    """

    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes):
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        # Classification head
        self.fc = Linear(d_model, num_classes)

    def __call__(self, x):
        # x: (batch, seq_len) word indices
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)

        # Mean pooling
        x = x.mean(dim=1)
        return self.fc(x)
```

---

## 12.8 Transformer Variants

### BERT (Encoder Only)

```
BERT = Bidirectional Encoder Representations from Transformers

Features:
  - Uses only Encoder
  - Bidirectional (can see entire sentence)
  - Pre-training + Fine-tuning

Use: Understanding tasks (classification, labeling, etc.)
```

### GPT (Decoder Only)

```
GPT = Generative Pre-trained Transformer

Features:
  - Uses only Decoder
  - Unidirectional (causal mask)
  - Autoregressive generation

Use: Generation tasks (text generation, dialogue, etc.)
```

### Comparison

| Model | Architecture | Direction | Use |
|------|------|------|------|
| BERT | Encoder | Bidirectional | Understanding |
| GPT | Decoder | Unidirectional | Generation |
| T5 | Encoder+Decoder | Bidirectional+Unidirectional | Translation |

---

## 12.9 Common Pitfalls

### Pitfall 1: Forgetting Positional Encoding

```python
# Wrong: No position information
x = self.embedding(tokens)
x = self.encoder(x)  # Doesn't know word order!

# Correct
x = self.embedding(tokens)
x = self.pos_encoder(x)  # Add positional encoding
x = self.encoder(x)
```

### Pitfall 2: Decoder Forgetting Mask

```python
# Wrong: Can see future
output = self.decoder(tgt, memory)  # No mask

# Correct: Causal mask
mask = generate_square_subsequent_mask(tgt.shape[1])
output = self.decoder(tgt, memory, tgt_mask=mask)
```

### Pitfall 3: Dimension Mismatch

```python
# Multi-head attention: embed_dim must be divisible by num_heads
MultiheadAttention(embed_dim=512, num_heads=8)   # ✓ 512/8=64
MultiheadAttention(embed_dim=512, num_heads=7)   # ✗ 512/7 not divisible
```

---

## 12.10 Summary in One Sentence

| Concept | One Sentence |
|------|--------|
| Attention | Q queries K to get weights, weighted combination of V |
| Self-attention | Q=K=V, relationship with itself |
| Multi-head | Look from multiple angles simultaneously, merge results |
| Residual connection | Input directly added to output, facilitates gradient flow |
| Positional encoding | Add label to each position, distinguish order |
| Causal mask | Don't let Decoder see the future |

---

## Next Chapter

Now we've learned Transformer!

Next chapter, we'll learn **Data Loading** — how to efficiently load and process training data.

→ [Chapter 13: Data Loading](13-dataloader_EN.md)

```python
# Preview: What you'll learn in the next chapter
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch_x, batch_y in dataloader:
    # Automatic batching, shuffling, parallel loading
```
