# Tutorial 11: Recurrent Neural Networks

## You're Watching a Suspense Movie...

Act 1, the protagonist finds a key. You don't think much of it.

Act 2, the protagonist discovers a locked box. You start to anticipate.

Act 3, the protagonist uses the key to open the box, inside is—you hold your breath.

You're tense because you **remembered the key from Act 1**. If each act were independent, you wouldn't care whether the box could be opened at all.

**Memory is the key to understanding sequences.**

But ordinary neural networks have no memory. Like a goldfish, they only live in the moment. Give it an article, after reading the first word, it forgets the first word; after reading the second word, it forgets the second word.

RNN (Recurrent Neural Network) gives the network a "memory".

```
Ordinary network:
  See "I" → Forgot
  See "love" → Forgot
  See "you" → Output "What are you?"

RNN:
  See "I" → Remember: subject is I
  See "love" → Remember: subject loves me
  See "you" → Understand: Complete sentence "I love you"
```

**RNN gives machines a sense of time.** It no longer lives only in the present, but can understand the past, grasp the present, and foresee the future.

---

## 11.1 Why Do We Need RNN?

### Problem: Ordinary Networks Have No Memory

```
Ordinary network processing a sentence:

"I love Beijing Tiananmen"
 ↓   ↓    ↓     ↓
Process each word independently

Problem:
  - Don't know "love" has subject "I"
  - Don't know "Tiananmen" is in Beijing
  - Each word is isolated
```

### Solution: Introduce Hidden State

```
RNN processing a sentence:

Time 1: "I" + hidden state 0 → output 1 → hidden state 1
Time 2: "love" + hidden state 1 → output 2 → hidden state 2
Time 3: "Beijing" + hidden state 2 → output 3 → hidden state 3
...

Hidden state = memory, passing previous information
```

---

## 11.2 RNN Basics

### Core Formula

```
h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)
y_t = W_hy @ h_t + b_y

Explanation:
  x_t = current input
  h_{t-1} = hidden state from previous time step (memory)
  h_t = current hidden state (updated memory)
  y_t = current output

tanh: Compress values to (-1, 1), prevent numerical explosion
```

### RNN Diagram

```
Time expansion:

       x_1      x_2      x_3
        ↓        ↓        ↓
     ┌─────┐  ┌─────┐  ┌─────┐
h_0→│ RNN │→│ RNN │→│ RNN │→ h_3
     └─────┘  └─────┘  └─────┘
        ↓        ↓        ↓
       y_1      y_2      y_3

Each RNN cell shares the same weights!
```

### RNNCell Implementation

```python
class RNNCell(Module):
    """
    Single-step RNN cell

    Analogy:
      - Input: Current scene you're watching
      - Hidden state: Previous memory
      - Output: Updated memory
    """

    def __init__(
        self,
        input_size: int,    # Input dimension
        hidden_size: int,   # Hidden state dimension
        bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input to hidden layer weights
        self.weight_ih = Tensor(
            np.random.randn(input_size, hidden_size).astype(np.float32) * 0.1,
            requires_grad=True,
        )

        # Hidden to hidden layer weights (memory transfer)
        self.weight_hh = Tensor(
            np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.1,
            requires_grad=True,
        )

        if bias:
            self.bias_ih = Tensor(np.zeros(hidden_size), requires_grad=True)
            self.bias_hh = Tensor(np.zeros(hidden_size), requires_grad=True)

    def forward(self, x: Tensor, h: Tensor = None) -> Tensor:
        """
        Single-step forward propagation

        Args:
            x: Current input (batch, input_size)
            h: Previous hidden state (batch, hidden_size)

        Returns:
            New hidden state
        """
        if h is None:
            h = Tensor(np.zeros((x.shape[0], self.hidden_size)))

        # Calculate new hidden state
        ih = x.matmul(self.weight_ih)
        hh = h.matmul(self.weight_hh)

        if hasattr(self, 'bias_ih'):
            ih = ih + self.bias_ih
            hh = hh + self.bias_hh

        # tanh activation
        h_new = (ih + hh).tanh()

        return h_new
```

### Usage

```python
# Create RNN cell
cell = RNNCell(input_size=64, hidden_size=128)

# Manually loop through sequence
h = None
outputs = []

for t in range(seq_len):
    x_t = x[:, t, :]  # Current time step input
    h = cell(x_t, h)   # Update hidden state
    outputs.append(h)

# outputs is the hidden states for all time steps
```

---

## 11.3 RNN's Problem: Vanishing Gradients

### Problem

```
RNN gradient propagation:

∂L/∂h_1 = ∂L/∂h_T × ∂h_T/∂h_{T-1} × ... × ∂h_2/∂h_1

Problem: Continuous multiplication many times!

If each gradient < 1:
  0.9 × 0.9 × 0.9 × ... × 0.9 (100 times) ≈ 0.00003

Gradient vanishing → Early information cannot be learned
```

### Life Analogy

```
Telephone game:

Original: "Meet at the usual place at 3 PM tomorrow"
Person 1: "Meet at 3 at the usual place"
Person 2: "Meet at 3 tomorrow"
Person 3: "Meet tomorrow"
...
Person 10: "???" (Forgot)

Information is gradually lost during transmission
```

---

## 11.4 LSTM: Long-term Memory

### Core Idea

```
LSTM = Long Short-Term Memory

Introduces "cell state" c_t as long-term memory
Uses "gates" to control information flow

Three gates:
  - Forget gate: Decides what old information to forget
  - Input gate: Decides what new information to remember
  - Output gate: Decides what information to output
```

### LSTM Diagram

```
                c_{t-1} ─────────────────────→ c_t
                   │                    ↑
                   │    ┌─────────────┐ │
                   └──→ │ Forget gate │─┘
                        │ Input gate  │──→ h_t
            h_{t-1} ──→ │ Output gate │──→ output
                ↑       └─────────────┘
                │             ↑
            x_t ─────────────┘
```

### LSTM Formulas

```
Forget gate: f_t = sigmoid(W_f @ [h_{t-1}, x_t])     "How much to forget"
Input gate: i_t = sigmoid(W_i @ [h_{t-1}, x_t])     "How much to remember"
Candidate: g_t = tanh(W_g @ [h_{t-1}, x_t])        "New information"
Output gate: o_t = sigmoid(W_o @ [h_{t-1}, x_t])     "How much to output"

Cell state: c_t = f_t * c_{t-1} + i_t * g_t      "Update long-term memory"
Hidden state: h_t = o_t * tanh(c_t)                "Update short-term memory"
```

### LSTMCell Implementation

```python
class LSTMCell(Module):
    """
    LSTM cell

    Analogy:
      - c (cell state) = long-term memory (remembers things from long ago)
      - h (hidden state) = short-term memory (remembers recent things)
      - gates = brain's control mechanism
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Four gates combined for efficiency
        gate_size = 4 * hidden_size

        self.weight_ih = Tensor(
            np.random.randn(input_size, gate_size).astype(np.float32) * 0.1,
            requires_grad=True,
        )
        self.weight_hh = Tensor(
            np.random.randn(hidden_size, gate_size).astype(np.float32) * 0.1,
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(gate_size), requires_grad=True)

    def forward(self, x: Tensor, state: tuple = None):
        """
        Args:
            x: Input (batch, input_size)
            state: (h, c) tuple
        Returns:
            (h_new, c_new) tuple
        """
        if state is None:
            h = Tensor(np.zeros((x.shape[0], self.hidden_size)))
            c = Tensor(np.zeros((x.shape[0], self.hidden_size)))
        else:
            h, c = state

        # Calculate all gates
        gates = x.matmul(self.weight_ih) + h.matmul(self.weight_hh) + self.bias

        # Split into four gates
        i = gates[:, :self.hidden_size].sigmoid()                    # Input gate
        f = gates[:, self.hidden_size:2*self.hidden_size].sigmoid()  # Forget gate
        g = gates[:, 2*self.hidden_size:3*self.hidden_size].tanh()   # Candidate value
        o = gates[:, 3*self.hidden_size:].sigmoid()                  # Output gate

        # Update state
        c_new = f * c + i * g        # Long-term memory: selective forgetting + selective remembering
        h_new = o * c_new.tanh()     # Short-term memory: selective output

        return h_new, c_new
```

### Why Doesn't LSTM Easily Vanish Gradients?

```
Cell state update:
  c_t = f_t * c_{t-1} + i_t * g_t

Gradient propagation:
  ∂c_t/∂c_{t-1} = f_t

If f_t ≈ 1, gradient can pass through directly!
The forget gate learns when to preserve information.
```

---

## 11.5 GRU: Simplified LSTM

### Core Idea

```
GRU = Gated Recurrent Unit

Simplified version of LSTM:
  - Only 2 gates (LSTM has 3)
  - Only 1 state (LSTM has h and c)
  - Fewer parameters, faster training
```

### GRU Formulas

```
Reset gate: r_t = sigmoid(W_r @ [h_{t-1}, x_t])    "How much to reset"
Update gate: z_t = sigmoid(W_z @ [h_{t-1}, x_t])    "How much to update"
Candidate: n_t = tanh(W_n @ [r_t * h_{t-1}, x_t]) "New information"

New state: h_t = (1-z_t) * n_t + z_t * h_{t-1}
```

### GRUCell Implementation

```python
class GRUCell(Module):
    """
    GRU cell

    Simpler than LSTM, but similar performance
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # 3 gates
        gate_size = 3 * hidden_size

        self.weight_ih = Tensor(
            np.random.randn(input_size, gate_size).astype(np.float32) * 0.1,
            requires_grad=True,
        )
        self.weight_hh = Tensor(
            np.random.randn(hidden_size, gate_size).astype(np.float32) * 0.1,
            requires_grad=True,
        )

    def forward(self, x: Tensor, h: Tensor = None) -> Tensor:
        if h is None:
            h = Tensor(np.zeros((x.shape[0], self.hidden_size)))

        gates = x.matmul(self.weight_ih) + h.matmul(self.weight_hh)

        # Split
        r = gates[:, :self.hidden_size].sigmoid()                     # Reset gate
        z = gates[:, self.hidden_size:2*self.hidden_size].sigmoid()   # Update gate
        n = gates[:, 2*self.hidden_size:].tanh()                       # Candidate value

        # Update
        h_new = (1 - z) * n + z * h

        return h_new
```

---

## 11.6 LSTM vs GRU

| Feature | LSTM | GRU |
|------|------|-----|
| Number of gates | 3 | 2 |
| Number of states | 2 (h, c) | 1 (h) |
| Parameters | More | 30% fewer |
| Computation speed | Slower | Faster |
| Expressiveness | Slightly stronger | Similar |
| Training difficulty | Harder | Easier |

```
Selection suggestions:
  - Pursuing performance: LSTM
  - Pursuing speed: GRU
  - Uncertain: Try GRU first
```

---

## 11.7 Complete RNN Layer

### Encapsulating the Loop

```python
class LSTM(Module):
    """
    Complete LSTM layer

    Automatically handles entire sequence, no manual looping needed
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = False,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        # Create multi-layer LSTM
        self.cells = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.cells.append(LSTMCell(in_size, hidden_size))

    def forward(self, x: Tensor, state: tuple = None):
        """
        Args:
            x: Input sequence
               batch_first=False: (seq_len, batch, input_size)
               batch_first=True:  (batch, seq_len, input_size)
            state: Initial state (h_0, c_0)

        Returns:
            output: Output for all time steps
            (h_n, c_n): State at last time step
        """
        if self.batch_first:
            # Convert to (seq_len, batch, input_size)
            x = x.transpose(0, 1)

        seq_len, batch, _ = x.shape

        # Initialize state
        if state is None:
            h = [Tensor(np.zeros((batch, self.hidden_size))) for _ in range(self.num_layers)]
            c = [Tensor(np.zeros((batch, self.hidden_size))) for _ in range(self.num_layers)]
        else:
            h, c = state

        outputs = []

        # Process time step by time step
        for t in range(seq_len):
            x_t = x[t]

            # Process layer by layer
            for layer, cell in enumerate(self.cells):
                h[layer], c[layer] = cell(x_t, (h[layer], c[layer]))
                x_t = h[layer]

            outputs.append(h[-1])

        output = Tensor(np.stack([o.data for o in outputs], axis=0))

        return output, (h, c)
```

### Usage

```python
# Create LSTM
lstm = LSTM(
    input_size=64,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
    bidirectional=False
)

# Input
x = Tensor.randn((32, 10, 64))  # (batch, seq_len, input_size)

# Forward propagation
output, (h_n, c_n) = lstm(x)

print(output.shape)  # (32, 10, 128)
```

---

## 11.8 Bidirectional RNN

### Principle

```
Ordinary RNN: Only looks at past
  I → love → you → !

Bidirectional RNN: Looks at both past and future
  Forward: I → love → you → !
  Backward: I ← love ← you ← !

Merge: [forward h_t, backward h_t]
```

### Use Cases

```
Suitable for: Tasks requiring complete context
  - Machine translation
  - Named entity recognition
  - Sentiment analysis

Not suitable for: Real-time generation tasks
  - Real-time speech recognition (can't see future)
  - Text generation
```

### Usage

```python
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

## 11.9 Application Examples

### Text Classification

```python
class TextClassifier:
    """
    Text classification using LSTM

    Structure: Embedding → LSTM → Take last hidden state → Classification
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.lstm = LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = Linear(hidden_dim, num_classes)

    def __call__(self, x):
        # x: (batch, seq_len) word indices
        x = self.embedding(x)        # (batch, seq_len, embed_dim)
        output, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]              # Last layer's hidden state
        return self.fc(h_last)

    def parameters(self):
        return self.embedding.parameters() + self.lstm.parameters() + self.fc.parameters()
```

### Language Model

```python
class LanguageModel:
    """
    Language model using LSTM

    Predict next word
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.lstm = LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = Linear(hidden_dim, vocab_size)

    def __call__(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        logits = self.fc(output)  # Predict next word at each position
        return logits

    def generate(self, start_token, max_len=50):
        """Generate text"""
        tokens = [start_token]
        h, c = None, None

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

## 11.10 Common Pitfalls

### Pitfall 1: Forgetting to Pass Hidden State

```python
# Wrong: Reset at every time step
for t in range(seq_len):
    h = cell(x[t], None)  # Reinitializes every time!

# Correct: Pass hidden state
h = None
for t in range(seq_len):
    h = cell(x[t], h)
```

### Pitfall 2: batch_first Confusion

```python
# LSTM defaults to batch_first=False
lstm = LSTM(64, 128)
x = Tensor.randn((32, 10, 64))  # (batch, seq, feature)

# Will error! Should transpose first or set batch_first=True
lstm = LSTM(64, 128, batch_first=True)
```

### Pitfall 3: Gradient Clipping

```python
# RNNs are prone to gradient explosion, need clipping
from nanotorch.utils import clip_grad_norm_

loss.backward()
clip_grad_norm_(model.parameters(), max_norm=1.0)  # Important!
optimizer.step()
```

---

## 11.11 Summary in One Sentence

| Concept | One Sentence |
|------|--------|
| RNN | Neural network with memory, remembers historical information |
| Hidden state | Carrier for transmitting information |
| LSTM | Uses gating to solve long-term memory |
| GRU | Simplified LSTM, faster |
| Bidirectional | Looks at both past and future simultaneously |

---

## Next Chapter

Now we've learned RNNs for processing sequences!

Next chapter, we'll learn **Transformer** — the foundation of modern NLP, the core of ChatGPT.

→ [Chapter 12: Transformer](12-transformer_EN.md)

```python
# Preview: What you'll learn in the next chapter
attention = MultiheadAttention(embed_dim=512, num_heads=8)
# Use attention mechanism to replace recurrence
# Process entire sequence in parallel
```
