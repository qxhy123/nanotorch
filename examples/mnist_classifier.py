"""
MNIST classifier example using nanotorch.

This example demonstrates:
1. Creating a CNN for image classification (MNIST-like)
2. Training on synthetic image data
3. Using cross-entropy loss and SGD optimizer
4. Evaluating classification accuracy
"""

import numpy as np
import nanotorch as nt
from nanotorch.nn import Sequential, Conv2D, ReLU, Dropout, Linear, CrossEntropyLoss
from nanotorch.optim import SGD


def generate_mnist_like_data(num_samples=1000, image_size=28, num_classes=10):
    """Generate synthetic MNIST-like image data.
    
    Creates random grayscale images with simple shapes and assigns random labels.
    This is for demonstration purposes only.
    """
    np.random.seed(42)
    
    # Create random images with simple patterns (circles, lines, etc.)
    images = np.random.randn(num_samples, 1, image_size, image_size).astype(np.float32) * 0.1
    
    # Add some simple patterns to make classification non-trivial
    for i in range(num_samples):
        # Add a random rectangle
        h_start = np.random.randint(0, image_size - 4)
        w_start = np.random.randint(0, image_size - 4)
        images[i, 0, h_start:h_start+4, w_start:w_start+4] += 0.5
        
        # Add a random cross
        center_h = np.random.randint(4, image_size - 4)
        center_w = np.random.randint(4, image_size - 4)
        images[i, 0, center_h-2:center_h+3, center_w] += 0.3
        images[i, 0, center_h, center_w-2:center_w+3] += 0.3
    
    # Normalize to have zero mean and unit variance
    images = (images - images.mean()) / (images.std() + 1e-8)
    
    # Generate random labels
    labels = np.random.randint(0, num_classes, size=(num_samples,))
    
    # Convert labels to one-hot encoding for compatibility
    labels_one_hot = np.zeros((num_samples, num_classes), dtype=np.float32)
    labels_one_hot[np.arange(num_samples), labels] = 1.0
    
    return images, labels, labels_one_hot


class MNISTCNN(nt.nn.Module):
    """A simple CNN for MNIST-like image classification."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = Sequential(
            Conv2D(1, 16, kernel_size=(3, 3), padding=1),  # 28x28 -> 28x28
            ReLU(),
            Conv2D(16, 32, kernel_size=(3, 3), stride=2),  # 28x28 -> 13x13 (floor)
            ReLU(),
            Conv2D(32, 64, kernel_size=(3, 3), stride=2),  # 13x13 -> 6x6
            ReLU(),
        )
        self.classifier = Sequential(
            Linear(64 * 6 * 6, 128),
            ReLU(),
            Dropout(p=0.5),
            Linear(128, num_classes),
        )
    
    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        
        # Flatten for linear layers
        N, C, H, W = x.shape
        x = x.reshape((N, C * H * W))
        
        # Classification
        x = self.classifier(x)
        return x


def train():
    """Train the CNN on synthetic MNIST-like data."""
    # Hyperparameters
    image_size = 28
    num_classes = 10
    num_samples = 2000
    batch_size = 32
    learning_rate = 0.01
    num_epochs = 20
    
    # Generate data
    X_np, y_labels, y_one_hot = generate_mnist_like_data(
        num_samples, image_size, num_classes
    )
    
    print(f"Generated {num_samples} synthetic MNIST-like images")
    print(f"Image shape: {X_np.shape[1:]}")
    print(f"Number of classes: {num_classes}")
    print("-" * 50)
    
    # Create model, loss, optimizer
    model = MNISTCNN(num_classes)
    criterion = CrossEntropyLoss(reduction="mean")
    optimizer = SGD(model.parameters(), lr=learning_rate)
    
    # Print model info
    total_params = sum(p.data.size for p in model.parameters())
    print(f"Model: {model}")
    print(f"Total parameters: {total_params}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("-" * 50)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # Shuffle data
        indices = np.random.permutation(num_samples)
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            
            # Get batch data
            X_batch = nt.Tensor(X_np[batch_indices])
            y_batch = nt.Tensor(y_labels[batch_indices])  # class indices
            
            # Forward pass
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            
            # Compute accuracy
            predictions = np.argmax(logits.data, axis=1)
            correct += np.sum(predictions == y_labels[batch_indices])
            total += len(batch_indices)
        
        avg_loss = epoch_loss / (num_samples / batch_size)
        accuracy = 100.0 * correct / total
        
        # Print progress every epoch
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
              f"Loss: {avg_loss:.4f} "
              f"Accuracy: {accuracy:.2f}%")
    
    print("-" * 50)
    print("Training completed!")
    
    # Final evaluation
    with nt.no_grad():
        X_all = nt.Tensor(X_np[:500])  # Evaluate on subset
        y_all = nt.Tensor(y_labels[:500])
        logits = model(X_all)
        predictions = np.argmax(logits.data, axis=1)
        final_accuracy = 100.0 * np.sum(predictions == y_labels[:500]) / 500
        print(f"Final accuracy on 500 samples: {final_accuracy:.2f}%")
    
    return model


def gradient_flow_demo():
    """Demonstrate gradient flow through the CNN."""
    print("\n" + "=" * 50)
    print("Gradient Flow Demonstration")
    print("=" * 50)
    
    # Create a small CNN
    model = Sequential(
        Conv2D(1, 4, kernel_size=(3, 3)),
        ReLU(),
        Conv2D(4, 8, kernel_size=(3, 3)),
        ReLU(),
    )
    
    # Create dummy input
    x = nt.Tensor(np.random.randn(2, 1, 10, 10).astype(np.float32), requires_grad=True)
    
    # Forward pass
    output = model(x)
    
    # Create dummy loss
    target = nt.Tensor(np.random.randn(2, 8, 6, 6).astype(np.float32))
    loss = ((output - target) ** 2).mean()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print("Gradient statistics:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = np.sqrt((param.grad.data ** 2).sum())
            print(f"  {name:20} gradient norm: {grad_norm:.6f}")
        else:
            print(f"  {name:20} gradient: None")
    
    print("\n✓ Gradients flowing through CNN successfully!")


if __name__ == "__main__":
    print("=" * 50)
    print("nanotorch - MNIST Classifier Example")
    print("=" * 50)
    
    # Train the CNN
    model = train()
    
    # Demonstrate gradient flow
    gradient_flow_demo()
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("=" * 50)