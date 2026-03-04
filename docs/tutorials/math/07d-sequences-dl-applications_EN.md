# Chapter 7(d): Applications of Sequences in Deep Learning

The concepts of sequences and series have extensive and profound applications in deep learning. From learning rate scheduling to sequence modeling, from positional encoding to gradient analysis, understanding these mathematical foundations is crucial for mastering deep learning. This section will explore in detail the specific applications of sequences and series in deep learning.

---

## 🎯 Life Analogy: Learning Rate Decay is Like Slowing Down a Car

Imagine you're **driving to find a parking spot**:

```
Start: Fast driving (high learning rate) → Quickly approach target area
      ┌────────────────────────────────────────────┐
      🚗💨━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→ 🅿️
      "General direction is right, go fast!"

Middle: Gradually slowing (learning rate decay) → Precise positioning
      ┌────────────────────────────────────────────┐
      🚗━━━━━━━━━━━━━━━━━━━━━━━━━━━━→ 🅿️
      "Almost there, slow down for precision"

End: Very slow (learning rate → 0) → Fine alignment
      ┌────────────────────────────────────────────┐
      🚗━━→ 🅿️
      "Carefully align with the parking lines"
```

**Three strategies correspond to three types of sequences**:

| Strategy | Sequence Type | Characteristic |
|----------|---------------|----------------|
| **Linear decay** | Arithmetic | Decreases by fixed amount each step |
| **Exponential decay** | Geometric | Fast at first, then slow, most common |
| **Cosine decay** | Trigonometric | Fast-slow-smooth, Transformers love it |

### RNN is Like "Telephone Game"

RNN processing sequences is like playing **telephone (whisper down the lane)**:

```
Original message: "The weather is nice today"
    ↓
Person 1: "The weather is nice today" (100% retained)
    ↓
Person 2: "The weather is... today" (80% retained)
    ↓
Person 3: "The... today" (64% retained)
    ↓
Person 4: "The...today" (51% retained)
    ↓
Person 5: "today" (41% retained)

Information gradually "decays", like sequence 1, 0.8, 0.64, 0.51, 0.41...
This is RNN's "memory" property—older information is remembered less!
```

### 📖 Plain English Translation

| Term | Plain English |
|------|---------------|
| Learning rate schedule | Designing how step size changes over time |
| RNN recurrence | Nonlinear version of recursive sequences |
| Positional encoding | Using sine/cosine to give each position a unique ID |
| Gradient vanishing | Geometric series converging to 0 |
| Gradient exploding | Geometric series growing to infinity |

---

## Table of Contents

1. [Learning Rate Decay Strategies](#learning-rate-decay-strategies)
2. [Sequence Modeling in RNN](#sequence-modeling-in-rnn)
3. [Transformer Positional Encoding](#transformer-positional-encoding)
4. [Self-Attention and Softmax Sequences](#self-attention-and-softmax-sequences)
5. [Sequence Generation and Sampling](#sequence-generation-and-sampling)
6. [Vanishing and Exploding Gradient Analysis](#vanishing-and-exploding-gradient-analysis)

---

## Learning Rate Decay Strategies

Learning rate scheduling is a critical technique for training deep neural networks. Different decay strategies correspond to different types of sequences.

### 1. Exponential Decay

**Geometric sequence form**:

$$
\eta_t = \eta_0 \cdot \gamma^{\lfloor t/T \rfloor}
$$

Where:
- $\eta_0$: Initial learning rate
- $\gamma$: Decay rate (typically $0.9 \sim 0.99$)
- $T$: Decay period

**Properties**: Learning rate decreases as a geometric sequence, converging to $0$.

```python
import numpy as np
import matplotlib.pyplot as plt

class ExponentialDecay:
    """Exponential learning rate decay"""
    def __init__(self, initial_lr, decay_rate, decay_steps):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def get_lr(self, step):
        return self.initial_lr * (self.decay_rate ** (step / self.decay_steps))

    def get_lr_staircase(self, step):
        """Staircase exponential decay"""
        return self.initial_lr * (self.decay_rate ** (step // self.decay_steps))

# Visualization
steps = np.arange(0, 10000)
exp_decay = ExponentialDecay(0.1, 0.96, 1000)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(steps, [exp_decay.get_lr(s) for s in steps])
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Continuous Exponential Decay')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(steps, [exp_decay.get_lr_staircase(s) for s in steps])
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Staircase Exponential Decay')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 2. Cosine Annealing

**Trigonometric function sequence form**:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)
$$

**Characteristics**:
- High learning rate at the beginning, rapid decline
- Stable in the middle phase
- Smoothly approaches minimum at the end

```python
class CosineAnnealingLR:
    """Cosine annealing learning rate"""
    def __init__(self, initial_lr, min_lr, total_steps):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_steps = total_steps

    def get_lr(self, step):
        return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
               (1 + np.cos(np.pi * step / self.total_steps))

# Visualization
cosine = CosineAnnealingLR(0.1, 0.001, 10000)

plt.figure(figsize=(10, 5))
plt.plot(steps, [cosine.get_lr(s) for s in steps])
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Cosine Annealing Learning Rate Schedule')
plt.grid(True, alpha=0.3)
plt.show()
```

### 3. Linear Decay

**Arithmetic sequence form**:

$$
\eta_t = \eta_0 - \frac{\eta_0 - \eta_{\min}}{T} \cdot t
$$

```python
class LinearDecay:
    """Linear learning rate decay"""
    def __init__(self, initial_lr, final_lr, total_steps):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_steps = total_steps
        self.d = (final_lr - initial_lr) / total_steps  # Common difference

    def get_lr(self, step):
        return max(self.final_lr, self.initial_lr + self.d * step)

# Compare different strategies
linear = LinearDecay(0.1, 0.001, 10000)

plt.figure(figsize=(10, 5))
plt.plot(steps, [exp_decay.get_lr(s) for s in steps], label='Exponential Decay')
plt.plot(steps, [cosine.get_lr(s) for s in steps], label='Cosine Annealing')
plt.plot(steps, [linear.get_lr(s) for s in steps], label='Linear Decay')
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Comparison of Learning Rate Scheduling Strategies')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 4. Warmup + Cosine (Transformer Standard)

**Composite sequence**:

$$
\eta_t = \begin{cases}
\frac{\eta_0 \cdot t}{T_{\text{warmup}}} & t < T_{\text{warmup}} \\
\eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{\pi(t - T_{\text{warmup}})}{T - T_{\text{warmup}}}\right)\right) & t \geq T_{\text{warmup}}
\end{cases}
$$

```python
class WarmupCosineSchedule:
    """Warmup + Cosine annealing (Transformer standard)"""
    def __init__(self, initial_lr, min_lr, warmup_steps, total_steps):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, step):
        if step < self.warmup_steps:
            # Linear warmup
            return self.initial_lr * step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

# Visualization
warmup_cosine = WarmupCosineSchedule(0.1, 0.001, 1000, 10000)

plt.figure(figsize=(10, 5))
plt.plot(steps, [warmup_cosine.get_lr(s) for s in steps])
plt.axvline(x=1000, color='r', linestyle='--', label='End of Warmup')
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Warmup + Cosine Learning Rate Schedule (Transformer Standard)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Sequence Modeling in RNN

The core of RNN is recursive processing of sequences, which is a typical application of recursive sequences.

### 1. Recurrence Relation of RNN Hidden State

**Basic RNN**:

$$
h_t = \tanh(W_h h_{t-1} + W_x x_t + b)
$$

This is a **nonlinear recursive sequence**.

**Simplified analysis** (ignoring nonlinearity):

$$
h_t \approx W_h h_{t-1} + W_x x_t
$$

```python
class SimpleRNN:
    """Simple RNN demonstration"""
    def __init__(self, input_size, hidden_size):
        # Initialize parameters (using small values for observation)
        self.W_h = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W_x = np.random.randn(hidden_size, input_size) * 0.1
        self.b = np.zeros(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x_sequence):
        """Forward propagation - recursive computation"""
        seq_len = x_sequence.shape[0]
        h_sequence = np.zeros((seq_len + 1, self.hidden_size))

        for t in range(seq_len):
            # Recurrence relation
            h_sequence[t+1] = np.tanh(
                self.W_h @ h_sequence[t] +
                self.W_x @ x_sequence[t] +
                self.b
            )

        return h_sequence[1:]

# Example
rnn = SimpleRNN(input_size=3, hidden_size=4)
x = np.random.randn(10, 3)  # Sequence length 10
h = rnn.forward(x)

# Visualize hidden state sequence
plt.figure(figsize=(12, 5))
for i in range(4):
    plt.plot(range(1, 11), h[:, i], 'o-', label=f'$h_{{:, {i}}}$')
plt.xlabel('Time Step')
plt.ylabel('Hidden State')
plt.title('RNN Hidden State Sequence (Recursive Sequence)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 2. LSTM Gated Recurrence

LSTM introduces gating mechanisms, making recurrence more complex but effective:

$$
\begin{align}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) & \text{Forget Gate} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) & \text{Input Gate} \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) & \text{Candidate Memory} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t & \text{Memory Update (Core of Recurrence)} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) & \text{Output Gate} \\
h_t &= o_t \odot \tanh(C_t) & \text{Output}
\end{align}
$$

```python
def lstm_gradient_analysis():
    """LSTM gradient analysis"""
    # LSTM gradient propagation: dC_t/dC_{t-1} = f_t
    # When f_t ≈ 1, gradient is preserved; when f_t ≈ 0, gradient vanishes

    # Simulate gradient propagation under different forget gate values
    f_values = [0.5, 0.9, 0.99, 1.0]
    T = 50

    plt.figure(figsize=(10, 5))
    for f in f_values:
        gradients = [f ** t for t in range(T)]
        plt.semilogy(gradients, label=f'$f = {f}$')

    plt.xlabel('Time Step')
    plt.ylabel('Gradient Magnitude (log)')
    plt.title('LSTM Gradient Propagation: $\\nabla C_t / \\nabla C_0 = f^T$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

lstm_gradient_analysis()
```

---

## Transformer Positional Encoding

Transformer uses sine and cosine functions to generate positional encodings, which is a classic application of sequences in deep learning.

### Positional Encoding Formula

$$
\begin{align}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\end{align}
$$

Where:
- $pos$: Position (position in the sequence)
- $i$: Dimension index
- $d_{model}$: Model dimension

### Frequencies Form a Geometric Sequence

The denominator $10000^{2i/d_{model}}$ forms a geometric sequence:

$$
\omega_i = \frac{1}{10000^{2i/d_{model}}} = 10000^{-2i/d_{model}}
$$

The ratio of adjacent frequencies is:

$$
\frac{\omega_{i+1}}{\omega_i} = 10000^{-2/d_{model}}
$$

```python
def positional_encoding(max_len, d_model):
    """Generate Transformer positional encoding"""
    position = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)

    # Compute frequency terms (geometric sequence)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)  # Even dimensions use sin
    pe[:, 1::2] = np.cos(position * div_term)  # Odd dimensions use cos

    return pe

# Generate positional encoding
pe = positional_encoding(100, 64)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Positional encoding matrix
im = axes[0].imshow(pe, aspect='auto', cmap='RdBu')
axes[0].set_xlabel('Dimension')
axes[0].set_ylabel('Position')
axes[0].set_title('Positional Encoding Matrix')
plt.colorbar(im, ax=axes[0])

# 2. Frequency terms (geometric sequence)
div_term = np.exp(np.arange(0, 64, 2) * (-np.log(10000.0) / 64))
axes[1].semilogy(div_term, 'o-')
axes[1].set_xlabel('Dimension Index')
axes[1].set_ylabel('Frequency (log scale)')
axes[1].set_title('Frequency Factor (Geometric Sequence)')
axes[1].grid(True, alpha=0.3)

# 3. Encodings at different positions
for pos in [0, 10, 50, 99]:
    axes[2].plot(pe[pos], label=f'pos={pos}')
axes[2].set_xlabel('Dimension')
axes[2].set_ylabel('Encoding Value')
axes[2].set_title('Encoding Vectors at Different Positions')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Properties of Positional Encoding

1. **Uniqueness**: Encodings at different positions are different
2. **Boundedness**: Values are between $[-1, 1]$
3. **Relative Position**: $PE_{pos+k}$ can be expressed as a linear function of $PE_{pos}$

```python
def analyze_pe_properties():
    """Analyze properties of positional encoding"""
    pe = positional_encoding(100, 64)

    # 1. Similarity matrix
    similarity = pe @ pe.T  # Dot product similarity

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(similarity, cmap='viridis')
    plt.colorbar(label='Dot Product')
    plt.xlabel('Position')
    plt.ylabel('Position')
    plt.title('Positional Encoding Similarity Matrix')

    # 2. Difference between adjacent positions
    plt.subplot(1, 2, 2)
    diff = np.linalg.norm(np.diff(pe, axis=0), axis=1)
    plt.plot(diff)
    plt.xlabel('Position')
    plt.ylabel('||PE[pos+1] - PE[pos]||')
    plt.title('Distance of Adjacent Positional Encodings')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

analyze_pe_properties()
```

---

## Self-Attention and Softmax Sequences

### Softmax as a Normalized Sequence

Given a sequence $\{z_i\}$, Softmax converts it to a probability distribution:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

Properties:
- Each term is between $(0$, $1)$
- Sum of all terms is $1$

```python
def softmax_sequence_demo():
    """Softmax sequence demonstration"""
    # Simulate attention scores
    scores = np.array([3.0, 1.0, 0.5, 0.1, -1.0])
    weights = np.exp(scores) / np.sum(np.exp(scores))

    print("Original scores:", scores)
    print("Softmax weights:", weights)
    print("Sum of weights:", np.sum(weights))

    # Effect of temperature parameter
    temperatures = [0.5, 1.0, 2.0, 5.0]

    plt.figure(figsize=(10, 5))
    for T in temperatures:
        scaled = scores / T
        weights = np.exp(scaled) / np.sum(np.exp(scaled))
        plt.bar(range(len(scores)), weights, alpha=0.5, label=f'T={T}')

    plt.xlabel('Position')
    plt.ylabel('Attention Weight')
    plt.title('Effect of Temperature Parameter on Softmax Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

softmax_sequence_demo()
```

### Sequence Operations in Self-Attention

**Query-Key-Value Mechanism**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

```python
def self_attention_demo():
    """Self-attention demonstration"""
    np.random.seed(42)
    seq_len = 5
    d_model = 8

    # Simulate input sequence
    X = np.random.randn(seq_len, d_model)

    # Simulate Q, K, V projections
    W_q = np.random.randn(d_model, d_model) * 0.1
    W_k = np.random.randn(d_model, d_model) * 0.1
    W_v = np.random.randn(d_model, d_model) * 0.1

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_model)

    # Softmax
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    attention_weights = softmax(scores)

    # Output
    output = attention_weights @ V

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights, cmap='Blues')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Self-Attention Weight Matrix')

    # Add numerical annotations
    for i in range(seq_len):
        for j in range(seq_len):
            plt.text(j, i, f'{attention_weights[i,j]:.2f}',
                    ha='center', va='center', color='black')

    plt.show()

    print("Attention weight matrix:")
    print(attention_weights)
    print("\nRow sums:", attention_weights.sum(axis=1))

self_attention_demo()
```

---

## Sequence Generation and Sampling

### Autoregressive Generation

**Autoregressive models** generate in a sequential manner:

$$
P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t | x_1, \ldots, x_{t-1})
$$

```python
def autoregressive_generation_demo():
    """Autoregressive generation demonstration"""
    np.random.seed(42)

    # Simulate vocabulary
    vocab = ['<START>', 'hello', 'world', 'deep', 'learning', '<END>']
    vocab_size = len(vocab)

    # Simulate model output (logits)
    def get_logits(context):
        """Simulate model outputting logits based on context"""
        # Simplified: predict next word based on last word
        np.random.seed(hash(tuple(context)) % (2**31))
        return np.random.randn(vocab_size)

    def sample_token(logits, temperature=1.0):
        """Sample from logits"""
        scaled = logits / temperature
        probs = np.exp(scaled) / np.sum(np.exp(scaled))
        return np.random.choice(len(probs), p=probs)

    # Generate sequence
    def generate(max_len=10, temperature=1.0):
        context = [0]  # <START>

        for _ in range(max_len):
            logits = get_logits(context)
            next_token = sample_token(logits, temperature)
            context.append(next_token)

            if next_token == vocab_size - 1:  # <END>
                break

        return [vocab[i] for i in context]

    # Generate with different temperatures
    print("Temperature=0.5 (more deterministic):", generate(temperature=0.5))
    print("Temperature=1.0 (standard):", generate(temperature=1.0))
    print("Temperature=2.0 (more random):", generate(temperature=2.0))

autoregressive_generation_demo()
```

---

## Vanishing and Exploding Gradient Analysis

### Theoretical Analysis

For RNNs, gradient propagation involves repeated multiplication:

$$
\frac{\partial h_T}{\partial h_0} = \prod_{t=1}^{T-1} \frac{\partial h_{t+1}}{\partial h_t} = \prod_{t=1}^{T-1} W_h \cdot \text{diag}(\tanh'(h_t))
$$

Simplified analysis (ignoring nonlinearity):

$$
\frac{\partial h_T}{\partial h_0} \approx W_h^{T-1}
$$

If the maximum eigenvalue of $W_h$ is $\lambda_{\max}$:
- $|\lambda_{\max}| < 1$: $\lim_{T \to \infty} \lambda_{\max}^T = 0$ (gradient vanishes)
- $|\lambda_{\max}| > 1$: $\lim_{T \to \infty} \lambda_{\max}^T = \infty$ (gradient explodes)

```python
def gradient_vanish_explode_demo():
    """Gradient vanishing/exploding demonstration"""
    # Different eigenvalue scenarios
    eigenvalues = [0.5, 0.99, 1.0, 1.01, 1.5]
    T_max = 100

    plt.figure(figsize=(14, 5))

    # 1. Gradient magnitude
    plt.subplot(1, 2, 1)
    for eig in eigenvalues:
        gradients = [np.abs(eig ** t) for t in range(T_max)]
        plt.semilogy(gradients, label=f'$\\lambda = {eig}$')

    plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Time Step (T)')
    plt.ylabel('|Gradient| (log scale)')
    plt.title('Gradient Vanishing/Exploding: $|\\lambda^T|$')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Series analysis
    plt.subplot(1, 2, 2)
    for eig in eigenvalues:
        # Cumulative sum of gradients (series)
        cumulative = [np.abs((1 - eig**(t+1)) / (1 - eig)) if eig != 1 else t + 1
                     for t in range(T_max)]
        plt.semilogy(cumulative, label=f'$\\lambda = {eig}$')

    plt.xlabel('Time Step (T)')
    plt.ylabel('Cumulative Gradient (log scale)')
    plt.title('Gradient Accumulation: $\\sum |\\lambda^t|$')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Theoretical analysis
    print("\nTheoretical Analysis:")
    print("=" * 50)
    for eig in eigenvalues:
        if eig < 1:
            infinite_sum = 1 / (1 - eig)
            print(f"λ = {eig}: Convergent geometric series, infinite sum = {infinite_sum:.4f}")
        elif eig > 1:
            print(f"λ = {eig}: Divergent geometric series, gradient explosion")
        else:
            print(f"λ = {eig}: Boundary case, gradient grows linearly")

gradient_vanish_explode_demo()
```

### Gradient Clipping

Common method to solve gradient explosion:

$$
\mathbf{g} = \begin{cases}
\mathbf{g} & \text{if } \|\mathbf{g}\| \leq c \\
\frac{c}{\|\mathbf{g}\|} \mathbf{g} & \text{if } \|\mathbf{g}\| > c
\end{cases}
$$

```python
def gradient_clipping_demo():
    """Gradient clipping demonstration"""
    np.random.seed(42)

    # Simulate gradients
    gradients = np.random.randn(100) * 10  # Large gradients

    def clip_gradient(grad, max_norm):
        """Gradient clipping"""
        norm = np.linalg.norm(grad)
        if norm > max_norm:
            return grad * max_norm / norm
        return grad

    max_norms = [1.0, 5.0, 10.0]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(gradients)), gradients, alpha=0.7, label='Original Gradients')
    plt.xlabel('Parameter Index')
    plt.ylabel('Gradient Value')
    plt.title('Original Gradients')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for max_norm in max_norms:
        clipped = clip_gradient(gradients, max_norm)
        plt.bar(np.arange(len(clipped)) + 0.2*max_norms.index(max_norm),
               clipped, width=0.2, alpha=0.7, label=f'max_norm={max_norm}')

    plt.xlabel('Parameter Index')
    plt.ylabel('Clipped Gradient')
    plt.title('Gradient Clipping Effect')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Statistics
    print(f"Original gradient norm: {np.linalg.norm(gradients):.2f}")
    for max_norm in max_norms:
        clipped = clip_gradient(gradients, max_norm)
        print(f"max_norm={max_norm}: Clipped norm = {np.linalg.norm(clipped):.2f}")

gradient_clipping_demo()
```

---

## Summary

This section introduced key applications of sequences and series in deep learning:

| Application Area | Sequence/Series Concept | Specific Form |
|------------------|------------------------|---------------|
| Learning Rate Scheduling | Arithmetic/Geometric/Trigonometric Sequences | Exponential Decay, Cosine Annealing |
| RNN | Recursive Sequences | Hidden State Update |
| Positional Encoding | Sine/Cosine Function Sequences | Transformer PE |
| Attention | Normalized Sequences | Softmax Weights |
| Gradient Analysis | Geometric Series | Gradient Vanishing/Exploding |
| Sequence Generation | Autoregressive Sequences | Probabilistic Chain Factorization |

**Key Takeaways**:
- Learning rate scheduling is essentially sequence design
- RNN is a nonlinear version of recursive sequences
- Positional encoding leverages the periodicity of trigonometric functions
- Gradient vanishing/exploding is the limiting behavior of geometric series
- Understanding these mathematical foundations helps diagnose and improve deep learning models

---

**Back to**: [Chapter 7: Sequences and Series](07-sequences-series_EN.md) | [Math Fundamentals Tutorial Table of Contents](../math-fundamentals_EN.md)
