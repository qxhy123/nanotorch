# Tutorial 11: Recurrent Neural Networks (RNN/LSTM/GRU)

## Memory in the Stream of Time...

Imagine reading a sentence, word by word.

"The key was in the..." — you wait, anticipating.

"...drawer" — ah, now it makes sense.

But how did you know to expect a location? Because you remembered "key" and "in the" from before. You held those words in your mind, carrying context forward, letting earlier meaning shape later understanding.

**This is what recurrent neural networks do.**

```
The Flow of Memory:

  Time step 1: See "The"     → Remember: {article detected}
       ↓
  Time step 2: See "key"     → Remember: {subject is "key"}
       ↓
  Time step 3: See "was"     → Remember: {past tense, "key" is subject}
       ↓
  Time step 4: See "in"      → Remember: {location coming}
       ↓
  Time step 5: See "the"     → Remember: {specific location}
       ↓
  Time step 6: See "drawer"  → Understanding complete!

Each step carries forward a memory,
updated by what it sees,
shaping what comes next.
```

**RNNs give neural networks the gift of memory.** Unlike feedforward networks that process each input independently, RNNs maintain a hidden state—a memory—that flows through time. What the network saw at step 1 influences what it predicts at step 100.

But basic RNNs have a problem: they forget. On long sequences, gradients vanish like echoes in a canyon. LSTM and GRU solve this with gates—mechanisms that learn what to remember, what to forget, what to pass on. They are the memory masters of deep learning.

In this tutorial, we'll implement RNN, LSTM, and GRU from scratch. We'll see how gates work, how bidirectional processing captures both past and future, and how these architectures power everything from machine translation to speech recognition.

---

## Table of Contents

1. [Overview](#overview)
2. [RNN Basics](#rnn-basics)
3. [LSTM Details](#lstm-details)
4. [GRU Details](#gru-details)
5. [Cell Version vs Full Version](#cell-version-vs-full-version)
6. [Bidirectional RNN](#bidirectional-rnn)
7. [Usage Examples](#usage-examples)
8. [Summary](#summary)

---

## Overview

Recurrent Neural Networks (RNN) are the core architecture for processing **sequential data**. Unlike traditional feedforward networks, RNNs have **memory** capability, enabling them to capture temporal dependencies in sequences.

nanotorch implements three main types of recurrent neural networks:
- **RNN**: Basic recurrent neural network
- **LSTM**: Long Short-Term Memory
- **GRU**: Gated Recurrent Unit

---

## RNN Basics

### Core Idea

RNN passes information between time steps through a hidden state:

```
Time step t calculation:
h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
y_t = h_t  (for basic RNN)
```

### Information Flow

```
Input sequence:  x_1, x_2, x_3, ..., x_T
                   ↓    ↓    ↓         ↓
Hidden states:   h_1→ h_2→ h_3→ ...→ h_T
                   ↓    ↓    ↓         ↓
Output:          y_1, y_2, y_3, ..., y_T
```

### RNNCell Implementation

```python
# nanotorch/nn/rnn.py

class RNNCell(Module):
    """Single-step RNN cell.
    
    h' = tanh(W_{ih} @ x + b_{ih} + W_{hh} @ h + b_{hh})
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden state size
        bias: Whether to use bias
        nonlinearity: Activation function ('tanh' or 'relu')
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        
        # Input to hidden weights
        self.weight_ih = Tensor(
            np.random.randn(input_size, hidden_size).astype(np.float32)
            * math.sqrt(1.0 / hidden_size),
            requires_grad=True,
        )
        
        # Hidden to hidden weights
        self.weight_hh = Tensor(
            np.random.randn(hidden_size, hidden_size).astype(np.float32)
            * math.sqrt(1.0 / hidden_size),
            requires_grad=True,
        )
        
        # Bias
        self.bias_ih = None
        self.bias_hh = None
        if bias:
            self.bias_ih = Tensor(np.zeros(hidden_size), requires_grad=True)
            self.bias_hh = Tensor(np.zeros(hidden_size), requires_grad=True)

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        """Single-step forward propagation.
        
        Args:
            x: Current time step input (batch, input_size)
            h: Previous time step hidden state (batch, hidden_size)
        
        Returns:
            New hidden state (batch, hidden_size)
        """
        if h is None:
            h = Tensor(np.zeros((x.shape[0], self.hidden_size)))
        
        # Compute input part
        ih = x.matmul(self.weight_ih)
        if self.bias_ih is not None:
            ih = ih + self.bias_ih
        
        # Compute hidden part
        hh = h.matmul(self.weight_hh)
        if self.bias_hh is not None:
            hh = hh + self.bias_hh
        
        # Combine and apply activation function
        out = ih + hh
        if self.nonlinearity == "tanh":
            out = out.tanh()
        elif self.nonlinearity == "relu":
            out = out.relu()
        
        return out
```

### Full RNN Implementation

```python
class RNN(RNNBase):
    """Multi-layer RNN network.
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden state size
        num_layers: Number of layers
        nonlinearity: 'tanh' or 'relu'
        bias: Whether to use bias
        batch_first: Whether input is (batch, seq, feature) format
        dropout: Dropout probability between layers
        bidirectional: Whether to use bidirectional RNN
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__(
            mode="RNN",
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            nonlinearity=nonlinearity,
        )
```

### RNN Problem: Vanishing Gradients

Basic RNNs suffer from **vanishing gradients** on long sequences:

```
Gradient propagation through time steps:
∂L/∂h_t = ∂L/∂h_T × ∏(∂h_i/∂h_{i-1})

When the sequence is long, the multiplied gradients decay or explode exponentially.
```

---

## LSTM Details

### Core Innovation

LSTM (Long Short-Term Memory) solves the vanishing gradient problem through **gating mechanisms**:
- **Forget Gate**: Decides what information to discard
- **Input Gate**: Decides what information to update
- **Output Gate**: Decides what information to output
- **Cell State**: Long-term memory

### LSTM Formulas

```
f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)     Forget gate
i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)     Input gate
o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)     Output gate
g_t = tanh(W_g @ [h_{t-1}, x_t] + b_g)  Candidate value

c_t = f_t * c_{t-1} + i_t * g_t         Update cell state
h_t = o_t * tanh(c_t)                   Update hidden state
```

### LSTMCell Implementation

```python
class LSTMCell(Module):
    """LSTM cell.
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden state size
        bias: Whether to use bias
    
    Shape:
        - Input: (batch, input_size)
        - Hidden: ((batch, hidden_size), (batch, hidden_size))  # (h, c)
        - Output: ((batch, hidden_size), (batch, hidden_size))  # (h', c')
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combine parameters for four gates (4 * hidden_size)
        gate_size = 4 * hidden_size
        
        self.weight_ih = Tensor(
            np.random.randn(input_size, gate_size).astype(np.float32)
            * math.sqrt(1.0 / hidden_size),
            requires_grad=True,
        )
        self.weight_hh = Tensor(
            np.random.randn(hidden_size, gate_size).astype(np.float32)
            * math.sqrt(1.0 / hidden_size),
            requires_grad=True,
        )
        
        self.bias_ih = None
        self.bias_hh = None
        if bias:
            self.bias_ih = Tensor(np.zeros(gate_size), requires_grad=True)
            self.bias_hh = Tensor(np.zeros(gate_size), requires_grad=True)

    def forward(
        self,
        x: Tensor,
        state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Single-step forward propagation.
        
        Args:
            x: Current input (batch, input_size)
            state: (h, c) tuple
        
        Returns:
            (h_new, c_new) tuple
        """
        if state is None:
            h = Tensor(np.zeros((x.shape[0], self.hidden_size)))
            c = Tensor(np.zeros((x.shape[0], self.hidden_size)))
        else:
            h, c = state
        
        # Compute all gates
        gates = x.matmul(self.weight_ih) + h.matmul(self.weight_hh)
        if self.bias_ih is not None:
            gates = gates + self.bias_ih
        if self.bias_hh is not None:
            gates = gates + self.bias_hh
        
        # Split into four gates
        i = gates[:, :self.hidden_size].sigmoid()           # Input gate
        f = gates[:, self.hidden_size:2*self.hidden_size].sigmoid()  # Forget gate
        g = gates[:, 2*self.hidden_size:3*self.hidden_size].tanh()    # Candidate value
        o = gates[:, 3*self.hidden_size:].sigmoid()         # Output gate
        
        # Update state
        c_new = f * c + i * g
        h_new = o * c_new.tanh()
        
        return h_new, c_new
```

### Full LSTM

```python
from nanotorch.nn import LSTM

lstm = LSTM(
    input_size=64,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
    bidirectional=True
)

x = Tensor.randn((32, 10, 64))  # (batch, seq_len, input_size)
output, (h_n, c_n) = lstm(x)

print(output.shape)  # (32, 10, 256)  # 256 = 128 * 2 (bidirectional)
print(h_n.shape)     # (4, 32, 128)   # 4 = 2 layers * 2 directions
```

---

## GRU Details

### Core Idea

GRU (Gated Recurrent Unit) is a simplified version of LSTM with only two gates:
- **Reset Gate**: Controls how to combine new input with previous memory
- **Update Gate**: Controls how much of the previous hidden state to retain

### GRU Formulas

```
r_t = σ(W_r @ [h_{t-1}, x_t])      Reset gate
z_t = σ(W_z @ [h_{t-1}, x_t])      Update gate
n_t = tanh(W_n @ [r_t * h_{t-1}, x_t])  Candidate hidden state

h_t = (1 - z_t) * n_t + z_t * h_{t-1}   New hidden state
```

### GRUCell Implementation

```python
class GRUCell(Module):
    """GRU cell.
    
    Shape:
        - Input: (batch, input_size)
        - Hidden: (batch, hidden_size)
        - Output: (batch, hidden_size)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Three gate parameters (3 * hidden_size)
        self.weight_ih = Tensor(
            np.random.randn(input_size, 3 * hidden_size).astype(np.float32)
            * math.sqrt(1.0 / hidden_size),
            requires_grad=True,
        )
        self.weight_hh = Tensor(
            np.random.randn(hidden_size, 3 * hidden_size).astype(np.float32)
            * math.sqrt(1.0 / hidden_size),
            requires_grad=True,
        )
        
        # bias initialization...

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        if h is None:
            h = Tensor(np.zeros((x.shape[0], self.hidden_size)))
        
        # Compute gates
        gi = x.matmul(self.weight_ih)
        gh = h.matmul(self.weight_hh)
        # ... add bias
        
        # Split into three parts
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        
        # Compute gates
        r = (i_r + h_r).sigmoid()  # Reset gate
        z = (i_z + h_z).sigmoid()  # Update gate
        n = (i_n + r * h_n).tanh() # Candidate hidden state
        
        # Update hidden state
        h_new = (1 - z) * n + z * h
        
        return h_new
```

### LSTM vs GRU

| Feature | LSTM | GRU |
|---------|------|-----|
| Number of gates | 3 | 2 |
| Number of states | 2 (h, c) | 1 (h) |
| Parameters | More | Fewer |
| Computation speed | Slower | Faster |
| Expressiveness | Stronger | Similar |
| Training difficulty | Harder | Easier |

---

## Cell Version vs Full Version

### Cell Version

For **manual control** of each time step:

```python
from nanotorch.nn import LSTMCell

cell = LSTMCell(input_size=64, hidden_size=128)

h = None
c = None
outputs = []

for t in range(seq_len):
    x_t = x[:, t, :]  # Current time step input
    h, c = cell(x_t, (h, c))
    outputs.append(h)

output = Tensor(np.stack([o.data for o in outputs], axis=1))
```

### Full Version

Automatically handles the entire sequence:

```python
from nanotorch.nn import LSTM

lstm = LSTM(input_size=64, hidden_size=128, batch_first=True)

# Process entire sequence at once
output, (h_n, c_n) = lstm(x)
```

---

## Bidirectional RNN

### Principle

Bidirectional RNN processes sequences from both forward and backward directions:

```
Forward:  h_1 → h_2 → h_3 → ... → h_T
Backward: h'_1 ← h'_2 ← h'_3 ← ... ← h'_T

Output: [h_1, h'_1], [h_2, h'_2], ..., [h_T, h'_T]
```

### Usage

```python
from nanotorch.nn import LSTM

# Bidirectional LSTM
bi_lstm = LSTM(
    input_size=64,
    hidden_size=128,
    bidirectional=True,  # Enable bidirectional
    batch_first=True
)

x = Tensor.randn((32, 10, 64))
output, (h_n, c_n) = bi_lstm(x)

print(output.shape)  # (32, 10, 256)  # 256 = 128 * 2 directions
```

---

## Usage Examples

### Text Classification

```python
from nanotorch import Tensor
from nanotorch.nn import Embedding, LSTM, Linear

class TextClassifier:
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
    
    def __call__(self, x):
        # x: (batch, seq_len) integer indices
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        output, (h_n, c_n) = self.lstm(x)
        
        # Use last time step output
        # h_n shape: (num_layers * 2, batch, hidden_dim)
        # Take bidirectional output from last layer
        h_forward = h_n[-2]  # Forward last layer
        h_backward = h_n[-1]  # Backward last layer
        h_concat = Tensor(np.concatenate([h_forward.data, h_backward.data], axis=-1))
        
        return self.fc(h_concat)
    
    def parameters(self):
        return self.embedding.parameters() + self.lstm.parameters() + self.fc.parameters()

# Usage
model = TextClassifier(vocab_size=10000, embed_dim=128, hidden_dim=256, num_classes=5)
x = Tensor(np.random.randint(0, 10000, (32, 50)))  # (batch, seq_len)
logits = model(x)
```

### Sequence Labeling (NER)

```python
class SequenceTagger:
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.lstm = LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = Linear(hidden_dim * 2, num_tags)
    
    def __call__(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)  # output: (batch, seq_len, hidden_dim * 2)
        
        # Classify each time step
        batch, seq_len, _ = output.shape
        output = output.reshape(batch * seq_len, -1)
        logits = self.fc(output)
        return logits.reshape(batch, seq_len, -1)
```

### Language Model

```python
class LanguageModel:
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.lstm = LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = Linear(hidden_dim, vocab_size)
    
    def __call__(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        logits = self.fc(output)
        return logits
    
    def generate(self, start_token, max_len=100):
        """Autoregressive text generation"""
        tokens = [start_token]
        h = None
        c = None
        
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

## Summary

This tutorial introduced recurrent neural network implementations in nanotorch:

| Model | Characteristics | Use Case |
|-------|-----------------|----------|
| **RNN** | Simple, but vanishing gradients | Short sequences |
| **LSTM** | Three gates, long-term memory | Long sequences, complex tasks |
| **GRU** | Two gates, simpler | Long sequences, efficiency priority |

### Key Points

1. **Cell version** is suitable for scenarios requiring fine-grained control
2. **Full version** is suitable for directly processing entire sequences
3. **Bidirectional RNN** can utilize both forward and backward context
4. **LSTM/GRU** solve vanishing gradients through gating mechanisms

### Next Steps

In [Tutorial 12: Transformer](12-transformer.md), we will learn how to implement the Transformer architecture, which is the foundation of modern NLP.

---

**References**:
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Learning Phrase Representations using RNN Encoder-Decoder (GRU)](https://arxiv.org/abs/1406.1078)
