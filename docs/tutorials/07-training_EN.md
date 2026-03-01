# Chapter 7: Complete Training Loop

## Where Everything Comes Together...

Imagine conducting an orchestra.

The violinists have their sheets, the percussion has their drums, the conductor has the score. But until they all play together, there is no symphony—just isolated sounds.

Deep learning is the same.

```
The Symphony of Training:

  Tensor provides the instruments    → Data that flows and differentiates
  Module organizes the players       → Layers that transform and learn
  Loss sets the tempo                → How far from harmony are we?
  Optimizer conducts the change      → How do we improve?

But the training loop IS the performance.
It brings everything together.
Each epoch, a new movement.
Each batch, a new measure.
The model learns, note by note,
until prediction harmonizes with truth.
```

**The training loop is where understanding becomes reality.** All the pieces we've built—Tensors, Modules, Losses, Optimizers—they mean nothing in isolation. Only in the training loop do they come alive, working together to transform random weights into a model that understands.

This chapter is the culmination of everything before it. We'll see the complete picture: how data flows through the model, how loss guides learning, how optimizers update weights, and how all of this fits together in a rhythmic cycle of forward pass, backward pass, and parameter update.

By the end, you'll have run a complete neural network from scratch—one you built yourself, component by component.

---

## 7.1 Training Loop Structure

```
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Forward pass
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        
        # 2. Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # 3. Parameter update
        optimizer.step()
    
    # 4. Validation
    val_loss = validate(model, val_loader)
    
    # 5. Save model
    if val_loss < best_loss:
        save_model(model)
```

## 7.2 Complete Training Code

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
    # Initialization
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # ========== Training Phase ==========
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            # Convert to Tensor
            X = Tensor(X_batch)
            y = Tensor(y_batch)
            
            # Forward pass
            logits = model(X)
            loss = criterion(logits, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevent gradient explosion)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Parameter update
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        
        # ========== Validation Phase ==========
        model.eval()
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

## 7.3 Using DataLoader

```python
from nanotorch import DataLoader, TensorDataset
import numpy as np

# Prepare data
X_train = np.random.randn(1000, 784).astype(np.float32)
y_train = np.random.randint(0, 10, 1000).astype(np.int64)

X_val = np.random.randn(200, 784).astype(np.float32)
y_val = np.random.randint(0, 10, 200).astype(np.int64)

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create model
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Dropout(0.2),
    Linear(256, 128),
    ReLU(),
    Dropout(0.2),
    Linear(128, 10)
)

# Train
history = train(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    lr=0.001
)
```

## 7.4 Model Loading and Inference

```python
# Load model
def load_model(model, path: str):
    state = dict(np.load(path))
    model.load_state_dict(state)
    return model

# Inference
def predict(model, X):
    model.eval()
    with no_grad():
        X = Tensor(X)
        logits = model(X)
        predictions = np.argmax(logits.data, axis=1)
    return predictions

# Usage
model = load_model(model, 'model.npz')
predictions = predict(model, X_test)
```

## 7.5 Training Tips

### Learning Rate Scheduler Selection

| Scheduler | Use Case |
|-----------|----------|
| StepLR | Simple tasks |
| CosineAnnealingLR | Transformers |
| ReduceLROnPlateau | When convergence speed is uncertain |

### Gradient Clipping

```python
from nanotorch.utils import clip_grad_norm_

# Prevent gradient explosion
clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Early Stopping

```python
def train_with_early_stopping(model, train_loader, val_loader, patience=10):
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

## 7.6 Complete Example: MNIST Classification

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

## 7.7 Summary

Congratulations on completing the nanotorch tutorial! You now understand:

1. **Tensor**: Multi-dimensional arrays + automatic differentiation
2. **Autograd**: Computational graphs + chain rule
3. **Module**: Parameter management + module composition
4. **Layer**: Linear, Conv, RNN, Transformer
5. **Loss**: Measuring discrepancy between predictions and ground truth
6. **Optimizer**: Gradient descent to update parameters
7. **Training**: Integrating all components

## Next Steps

- Read [Chapter 8: Data Augmentation](08-transforms.md): Image transforms and data augmentation
- Learn [Chapter 9: Convolution Layers](09-conv.md): Conv2D, transposed convolution
- Check complete examples in the `examples/` directory
- Try implementing your own projects with nanotorch!

---

**Congratulations!** You've mastered the core principles of deep learning frameworks! 🎉
