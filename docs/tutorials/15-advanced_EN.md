# Tutorial 15: Advanced Topics

## Training a Neural Network is Like Caring for a Child...

You need patience, you need skills, you need to know how to read the signs.

Sometimes, it learns too aggressively, gradients explode, you need to gently tell it: "Slow down, don't rush." — This is **gradient clipping**.

Sometimes, it's just starting out, knowing nothing, you need to let it begin with simple things, gradually increase the load — This is **learning rate warmup**.

Sometimes, it overlearns, starts memorizing by rote, you need to stop it in time — This is **early stopping**.

```
The Wisdom of Training:

  Gradient clipping:
    Gradient too large → Limit it
    Prevents parameters from updating too aggressively, running away

  Learning rate warmup:
    Just starting → Small learning rate
    Slowly increase → Let model adapt first

  Early stopping:
    Validation loss no longer decreasing → Stop training
    Prevent overfitting, know when to stop
```

**Training techniques are a combination of experience and wisdom.** They make models learn more stably, faster, and better.

---

## 15.1 Gradient Clipping

### Problem: Gradient Explosion

```
During deep network or RNN training:

Normal gradient: [0.1, 0.2, -0.1, ...]
Exploding gradient: [100, 200, -50, ...]

Consequences:
  - Parameter updates too large
  - Loss becomes NaN
  - Training crashes
```

### Solution: Gradient Clipping

```
Gradient clipping: Limit gradient magnitude

clip_grad_norm_: Clip by norm
  If ||gradient|| > max_norm:
    gradient = gradient × (max_norm / ||gradient||)

Analogy: Speed limit
  Speeding → Slow down to speed limit
  Not speeding → Maintain current speed
```

### Implementation

```python
def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
    """
    Clip gradients by norm

    Args:
        parameters: Model parameters
        max_norm: Maximum norm (commonly 1.0 or 5.0)

    Returns:
        Gradient norm before clipping
    """
    # Collect all gradients
    grads = []
    for param in parameters:
        if param.grad is not None:
            grads.append(param.grad.flatten())

    if len(grads) == 0:
        return 0.0

    # Calculate total gradient norm
    total_grad = np.concatenate(grads)
    total_norm = np.linalg.norm(total_grad, ord=norm_type)

    # Calculate clipping coefficient
    clip_coef = max_norm / (total_norm + 1e-6)

    # If clipping needed
    if clip_coef < 1:
        for param in parameters:
            if param.grad is not None:
                param.grad = param.grad * clip_coef

    return float(total_norm)
```

### Usage

```python
from nanotorch.utils import clip_grad_norm_

# Training loop
for x, y in dataloader:
    optimizer.zero_grad()

    output = model(Tensor(x))
    loss = criterion(output, Tensor(y))
    loss.backward()

    # Gradient clipping (before optimizer.step())
    grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    if grad_norm > 1.0:
        print(f"Gradient clipped! Original norm: {grad_norm:.2f}")
```

### When to Use?

```
Must use:
  - RNN / LSTM training
  - Transformer training
  - Deep networks

Optional:
  - Regular CNN
  - Shallow networks

Empirical values:
  - RNN: max_norm=1.0 or 5.0
  - Transformer: max_norm=1.0
```

---

## 15.2 Learning Rate Warmup

### Problem: Training Instability at Start

```
When training just begins:
  - Parameters are randomly initialized
  - Features don't have meaning yet
  - Large learning rate may cause oscillation

Analogy: Cold engine start
  - Winter morning, engine is cold
  - Step on gas immediately → Engine damage
  - Warm up first → Runs smoothly
```

### Solution: Warmup

```
Warmup:

Regular training:
  Learning rate = 0.001 (constant throughout)

With warmup:
  Epoch 1:  Learning rate = 0.0001
  Epoch 2:  Learning rate = 0.0002
  Epoch 3:  Learning rate = 0.0003
  ...
  Epoch 10: Learning rate = 0.001  ← Reached target
  Epoch 11: Learning rate = 0.001
  ...
```

### Implementation

```python
class LinearWarmup:
    """
    Linear warmup

    Learning rate increases linearly from start_lr to target learning rate
    """

    def __init__(self, optimizer, warmup_epochs, start_lr=0.0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.target_lr = optimizer.lr
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            alpha = self.current_epoch / self.warmup_epochs
            self.optimizer.lr = self.start_lr + alpha * (self.target_lr - self.start_lr)
        self.current_epoch += 1
```

### Usage

```python
from nanotorch.optim import Adam
from nanotorch.utils import LinearWarmup

optimizer = Adam(model.parameters(), lr=0.001)
warmup = LinearWarmup(optimizer, warmup_epochs=5, start_lr=0.0001)

for epoch in range(100):
    train_one_epoch(...)

    warmup.step()  # Update learning rate
    print(f"Epoch {epoch}, LR: {optimizer.lr:.6f}")
```

### Cosine Warmup + Decay

```python
class CosineWarmupScheduler:
    """
    Cosine warmup + cosine decay

    Most commonly used Transformer scheduler
    """

    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self.current_epoch = 0

    def step(self):
        import math

        if self.current_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            alpha = self.current_epoch / self.warmup_epochs
            lr = self.min_lr + alpha * (self.base_lr - self.min_lr)
        else:
            # Decay phase: cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        self.optimizer.lr = lr
        self.current_epoch += 1
```

---

## 15.3 Early Stopping

### Problem: Overfitting

```
Training process:

Epoch 1-20:  Training loss ↓, Validation loss ↓  ← Learning
Epoch 21-50: Training loss ↓, Validation loss →  ← Starting to overfit
Epoch 51+:   Training loss ↓, Validation loss ↑  ← Severe overfitting

Should stop when validation loss no longer decreases!
```

### Implementation

```python
class EarlyStopping:
    """
    Early stopping mechanism

    Stops training if validation loss doesn't improve for patience consecutive times
    """

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        """Check if should stop"""
        if val_loss < self.best_loss - self.min_delta:
            # Improved
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Should stop
            return False
```

### Usage

```python
early_stop = EarlyStopping(patience=15)
best_loss = float('inf')

for epoch in range(100):
    train_loss = train(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)

    # Save best model
    if val_loss < best_loss:
        best_loss = val_loss
        save_model(model, 'best_model.npz')

    # Check for early stopping
    if early_stop(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## 15.4 Model Saving and Loading

### Saving Models

```python
# Save parameters
def save_model(model, path):
    state_dict = model.state_dict()
    np.savez(path, **state_dict)

# Save checkpoint (including training state)
def save_checkpoint(model, optimizer, epoch, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'loss': loss,
    }
    np.savez(path, **checkpoint)
```

### Loading Models

```python
# Load parameters
def load_model(model, path):
    state_dict = dict(np.load(path))
    model.load_state_dict(state_dict)
    return model

# Load checkpoint
def load_checkpoint(model, path):
    checkpoint = dict(np.load(path, allow_pickle=True))
    model.load_state_dict(checkpoint['model_state'].item())
    return checkpoint['epoch'], checkpoint['loss']
```

---

## 15.5 Training Debugging Tips

### Check Gradients

```python
def check_gradients(model):
    """Check if gradients are normal"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = np.linalg.norm(param.grad)
            has_nan = np.isnan(param.grad).any()
            has_inf = np.isinf(param.grad).any()

            print(f"{name}:")
            print(f"  norm: {grad_norm:.6f}")
            print(f"  has_nan: {has_nan}")
            print(f"  has_inf: {has_inf}")
```

### Verify with Small Dataset First

```python
# Debugging tip: Use 10 samples first
small_dataset = TensorDataset(X[:10], y[:10])
small_loader = DataLoader(small_dataset, batch_size=10)

# Should be able to quickly overfit (loss → 0)
for epoch in range(100):
    loss = train_one_epoch(model, small_loader, optimizer)
    print(f"Epoch {epoch}, Loss: {loss:.6f}")

# If loss doesn't decrease → Model or code has problems
```

### Monitor Training Metrics

```python
def train_with_logging(model, train_loader, val_loader, epochs):
    history = {'train_loss': [], 'val_loss': [], 'grad_norm': []}

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer)
        grad_norm = get_grad_norm(model.parameters())

        # Validate
        val_loss = validate(model, val_loader)

        # Record
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['grad_norm'].append(grad_norm)

        # Print
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Grad Norm: {grad_norm:.4f}")

    return history
```

---

## 15.6 Training Tips Summary

### Must-Use Techniques

```
1. Gradient clipping
   clip_grad_norm_(model.parameters(), max_norm=1.0)

2. Learning rate scheduling
   scheduler.step() after each epoch

3. Early stopping
   Prevent overfitting

4. Save best model
   Based on validation loss
```

### Debugging Workflow

```
1. Overfit small dataset first
   Ensure code is correct

2. Check gradients
   No NaN/Inf

3. Monitor loss curves
   Should decrease smoothly

4. Gradually increase data/model complexity
```

---

## 15.7 One-Line Summary

| Technique | Purpose | When to Use |
|-----------|---------|-------------|
| Gradient clipping | Prevent gradient explosion | RNN, Transformer |
| Learning rate warmup | Stabilize early training | Large models, large learning rates |
| Early stopping | Prevent overfitting | Long training times |
| Small data verification | Check code | During debugging |

---

## Congratulations!

You've completed the nanotorch tutorial series!

```
What you learned:

┌─────────────────────────────────────────┐
│                                         │
│  ① Tensor: Data carrier + autodiff      │
│  ② Autograd: Computation graph + chain rule │
│  ③ Module: Parameter management + module composition │
│  ④ Layer: Linear, Conv, RNN, Attention  │
│  ⑤ Activation: ReLU, Sigmoid, Softmax   │
│  ⑥ Loss: MSE, CrossEntropy, BCE         │
│  ⑦ Optimizer: SGD, Adam, AdamW          │
│  ⑧ Training: Complete training loop     │
│  ⑨ Data: Dataset, DataLoader            │
│  ⑩ Init: Xavier, Kaiming                │
│  ⑪ Advanced: Gradient clipping, early stopping │
│                                         │
└─────────────────────────────────────────┘
```

### Next Steps

1. Read nanotorch source code
2. Implement more features (mixed precision, distributed)
3. Apply to real projects

**You now understand the core principles of deep learning frameworks!**
