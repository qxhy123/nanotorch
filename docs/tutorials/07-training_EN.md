# Chapter 7: Complete Training Loop

## Learning is Never Achieved Overnight...

The first time you rode a bicycle, you fell.
The second time, you fell again.
The third time, you learned to balance, but couldn't turn.
The fourth time, the fifth time... finally, you could ride freely.

**Learning is the art of repetition.**

Training a neural network is the same. It doesn't learn from seeing the data once. It needs to try, make mistakes, correct, and try again, over and over. With each cycle, it improves a little bit.

```
The Rhythm of Training:

  Round 1:  Clueless, accuracy 30%
  Round 10: Starting to get it, accuracy 60%
  Round 50: Getting better, accuracy 85%
  Round 100: Mastery, accuracy 95%

There are no shortcuts, only repetition
```

This repetitive process is called the **training loop**. It contains five steps: look at data, make predictions, calculate the gap, find the cause, update parameters. Round and round, until convergence.

**The training loop is the path every model must take to grow.** Each iteration is a transformation.

---

## 7.1 The Essence of the Training Loop

### The Five-Step Method

```
for epoch in range(num_epochs):
    for batch in dataloader:
        ① Forward pass: predictions = model(inputs)
        ② Calculate loss: loss = criterion(predictions, targets)
        ③ Zero gradients: optimizer.zero_grad()
        ④ Backward pass: loss.backward()
        ⑤ Update parameters: optimizer.step()
```

### Why Loop?

```
Once is not enough:

1st time: Model guesses randomly, loss=2.5
2nd time: Learned a bit, loss=2.0
3rd time: Still improving, loss=1.5
...
100th time: Pretty good, loss=0.1

Learning requires repetition!
Just like memorizing vocabulary, you can't remember after seeing it once.
```

---

## 7.2 Complete Training Code

### Training Function

```python
import numpy as np
from nanotorch import Tensor
from nanotorch.nn import Sequential, Linear, ReLU, Dropout
from nanotorch.nn.loss import CrossEntropyLoss
from nanotorch.optim import Adam
from nanotorch.optim.lr_scheduler import StepLR
from nanotorch.utils import clip_grad_norm_

def train(
    model,
    train_loader,
    val_loader,
    num_epochs: int = 100,
    lr: float = 0.001,
    save_path: str = 'model.npz'
):
    """
    Complete training function

    Analogy:
      model = student
      train_loader = textbook
      val_loader = practice exam
      num_epochs = how many review rounds
      optimizer = study method
    """
    # Initialization
    criterion = CrossEntropyLoss()  # Grading standard
    optimizer = Adam(model.parameters(), lr=lr)  # Study method
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # Study plan

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # ========== Training Phase (student studies from book) ==========
        model.train()  # Switch to training mode
        train_loss = 0.0
        num_batches = 0

        for X_batch, y_batch in train_loader:
            # Convert to Tensor
            X = Tensor(X_batch)
            y = Tensor(y_batch)

            # ① Forward pass: student solves problems
            logits = model(X)
            loss = criterion(logits, y)

            # ③ Zero gradients (clear previous memory)
            optimizer.zero_grad()

            # ④ Backward pass: student checks answers, understands what went wrong
            loss.backward()

            # Gradient clipping (prevent "learning too fast")
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            # ⑤ Update parameters: student remembers the lesson
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches

        # ========== Validation Phase (practice exam) ==========
        model.eval()  # Switch to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in val_loader:
            X = Tensor(X_batch)
            y = Tensor(y_batch)

            logits = model(X)
            loss = criterion(logits, y)

            val_loss += loss.item()

            # Calculate accuracy
            predictions = np.argmax(logits.data, axis=1)
            correct += np.sum(predictions == y.data)
            total += len(y.data)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            state = model.state_dict()
            np.savez(save_path, **state)
            print(f"  ✓ Saved best model (val_loss: {avg_val_loss:.4f})")

        # Update learning rate
        scheduler.step()

    return history
```

---

## 7.3 Training Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      One Epoch                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Batch 1    │ -> │   Batch 2    │ -> │   Batch N    │  │
│  │  fwd + bwd   │    │  fwd + bwd   │    │  fwd + bwd   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                             │
│                      ↓ All batches processed                │
│                                                             │
│              ┌─────────────────────┐                        │
│              │   Validation Phase  │                        │
│              │  evaluate on val    │                        │
│              └─────────────────────┘                        │
│                      ↓                                      │
│              ┌─────────────────────┐                        │
│              │  Save best model?   │                        │
│              │  Update LR?         │                        │
│              └─────────────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Repeat num_epochs times
```

---

## 7.4 Using DataLoader

### Why Do We Need DataLoader?

```
Too much data to process at once:

10000 images → Split into 313 batches (32 each)
             → Process only 32 at a time
             → 313 iterations to complete one round
```

### Code Example

```python
from nanotorch import DataLoader, TensorDataset
import numpy as np

# Prepare data
X_train = np.random.randn(1000, 784).astype(np.float32)  # 1000 images
y_train = np.random.randint(0, 10, 1000).astype(np.int64)  # 1000 labels

X_val = np.random.randn(200, 784).astype(np.float32)
y_val = np.random.randint(0, 10, 200).astype(np.int64)

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# What does DataLoader do?
# 1. Split data into small batches (32 each)
# 2. shuffle=True: Randomize order (important for training!)
# 3. Automatically returns (X_batch, y_batch) when iterating
```

---

## 7.5 Model Saving and Loading

### Why Save Models?

```
Trained model = Knowledge the student has learned

Training is slow (could take hours/days)
After saving, you can:
  1. Use it directly next time without retraining
  2. Deploy to production
  3. Continue training (transfer learning)
```

### Code

```python
# Save model
def save_model(model, path: str):
    """Save model parameters to file"""
    state = model.state_dict()  # Get all parameters
    np.savez(path, **state)

# Load model
def load_model(model, path: str):
    """Restore model parameters from file"""
    state = dict(np.load(path))
    model.load_state_dict(state)
    return model

# Inference (using the model)
def predict(model, X):
    """Make predictions with the model"""
    model.eval()  # Evaluation mode
    with no_grad():  # Don't compute gradients (saves memory)
        X = Tensor(X)
        logits = model(X)
        predictions = np.argmax(logits.data, axis=1)
    return predictions

# Usage
model = load_model(model, 'model.npz')
predictions = predict(model, X_test)
```

---

## 7.6 Training Tips

### 1. Learning Rate Scheduling

```
Learning rate = Speed of learning

Early stage: Big steps, quickly approach the target
Later stage: Small steps, fine-tune

┌───────────────────────────────┐
│    Big steps → Fine-tune      │
│   ↘️                          │
│    ↘️                         │
│     ↘️  ↘️                    │
│       ↘️  ↓  ← Small steps    │
│         ⬇️                    │
└───────────────────────────────┘
```

| Scheduler | Strategy | Use Case |
|-----------|----------|----------|
| StepLR | Multiply by 0.1 every N steps | Simple tasks |
| CosineAnnealingLR | Cosine curve decay | Transformers |
| ReduceLROnPlateau | Reduce when loss stops improving | Unknown convergence speed |

### 2. Gradient Clipping

```python
from nanotorch.utils import clip_grad_norm_

# Gradient clipping: prevent gradient explosion
clip_grad_norm_(model.parameters(), max_norm=1.0)

# Analogy: Limit learning speed to prevent "learning too aggressively"
# Gradient too large → Clip to 1.0
# Gradient normal → Unchanged
```

### 3. Early Stopping

```python
def train_with_early_stopping(model, train_loader, val_loader, patience=10):
    """
    Stop training if validation loss doesn't improve for patience consecutive times

    Analogy:
      Student doesn't improve after 10 consecutive practice exams → Stop reviewing
      Avoid overfitting (rote memorization)
    """
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader)
        val_loss = validate(model, val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model(model)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
```

---

## 7.7 train() vs eval() Modes

```
model.train()  vs  model.eval()

Training mode:
  - Dropout: Randomly drop neurons
  - BatchNorm: Use current batch statistics

Evaluation mode:
  - Dropout: Don't drop
  - BatchNorm: Use global statistics

Analogy:
  train() = Daily practice (with random interference)
  eval()  = Official exam (stable performance)
```

---

## 7.8 Complete Example: MNIST Classification

```python
# examples/mnist_simple.py
import numpy as np
from nanotorch import Tensor, DataLoader, TensorDataset
from nanotorch.nn import Sequential, Linear, ReLU, Dropout
from nanotorch.nn.loss import CrossEntropyLoss
from nanotorch.optim import Adam

# Load MNIST data (using random data for demonstration)
def load_mnist():
    X_train = np.random.randn(60000, 784).astype(np.float32)
    y_train = np.random.randint(0, 10, 60000).astype(np.int64)
    X_test = np.random.randn(10000, 784).astype(np.float32)
    y_test = np.random.randint(0, 10, 10000).astype(np.int64)
    return X_train, y_train, X_test, y_test

# Main function
def main():
    # Data
    X_train, y_train, X_test, y_test = load_mnist()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=64, shuffle=False
    )

    # Model
    model = Sequential(
        Linear(784, 512),
        ReLU(),
        Dropout(0.2),
        Linear(512, 256),
        ReLU(),
        Dropout(0.2),
        Linear(256, 10)
    )

    print(f"Model parameters: {sum(p.data.size for p in model.parameters()):,}")

    # Train
    history = train(
        model, train_loader, test_loader,
        num_epochs=20, lr=0.001
    )

    print(f"\nFinal accuracy: {history['val_acc'][-1]:.4f}")

if __name__ == '__main__':
    main()
```

---

## 7.9 Common Pitfalls

### Pitfall 1: Forgetting zero_grad()

```python
# Wrong: Gradients will accumulate
for epoch in range(100):
    loss = criterion(model(X), y)
    loss.backward()  # Gradients stack on top of previous ones!
    optimizer.step()

# Correct: Zero each time
for epoch in range(100):
    optimizer.zero_grad()  # Zero first!
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
```

### Pitfall 2: Confusing train/eval modes

```python
# Wrong: Model still in train mode during validation
val_loss = validate(model, val_loader)  # model is still in train mode

# Correct
model.eval()
val_loss = validate(model, val_loader)
model.train()  # Switch back
```

### Pitfall 3: Overfitting

```
Overfitting = Rote memorization

Signs:
  - Training loss keeps dropping
  - Validation loss drops then rises

Solutions:
  - Dropout
  - Early stopping
  - Data augmentation
  - Smaller model
```

---

## 7.10 Summary in One Sentence

| Concept | One Sentence |
|---------|--------------|
| Training Loop | Repeat: forward → loss → backward → update |
| Epoch | One pass through all data |
| Batch | A small chunk of data processed at once |
| DataLoader | Automatically batch and shuffle data |
| zero_grad | Clear gradients from last time |
| model.train/eval | Switch between training/evaluation modes |
| Early Stopping | Stop if validation doesn't improve |

---

## Congratulations!

You now understand the complete process of deep learning:

```
┌─────────────────────────────────────────────────┐
│                                                 │
│   ① Tensor: Data carrier + auto-differentiation │
│   ② Autograd: Computational graph + chain rule  │
│   ③ Module: Parameter management + composition  │
│   ④ Layer: Linear, Conv, RNN, Transformer      │
│   ⑤ Activation: ReLU, Sigmoid, Softmax         │
│   ⑥ Loss: Measure gap between prediction & truth│
│   ⑦ Optimizer: Gradient descent to update params│
│   ⑧ Training: Integrate all components ← You here│
│                                                 │
└─────────────────────────────────────────────────┘
```

## Next Steps

- Read [Chapter 8: Data Augmentation](08-transforms.md): Make data more varied
- Learn [Chapter 9: Convolution Layers](09-conv.md): The magic tool for images
- Check complete examples in the `examples/` directory
- Try implementing your own projects with nanotorch!

---

**Congratulations!** You've mastered the core principles of deep learning frameworks! 🎉
