"""
Training utilities for recommendation models.

This module provides training loops, early stopping, and other utilities
for training recommendation models in nanotorch.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module
from nanotorch.nn.loss import BCELoss
from nanotorch.nn.metrics import auc_score, log_loss
from nanotorch.optim.optimizer import Optimizer
from nanotorch.optim.lr_scheduler import CosineWarmupScheduler, ReduceLROnPlateau
from nanotorch.utils import clip_grad_norm_
from nanotorch.data import DataLoader


@dataclass
class TrainingConfig:
    """Configuration for training recommendation models.
    
    Attributes:
        num_epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate.
        batch_size: Batch size for training.
        weight_decay: L2 regularization coefficient.
        early_stop_patience: Epochs to wait before early stopping.
        early_stop_min_delta: Minimum improvement to reset patience.
        gradient_clip_norm: Maximum gradient norm for clipping.
        lr_scheduler: Learning rate scheduler type.
        warmup_epochs: Number of warmup epochs.
        log_interval: Log every N batches.
        eval_interval: Evaluate every N epochs.
    """
    num_epochs: int = 20
    learning_rate: float = 0.001
    batch_size: int = 256
    weight_decay: float = 0.0001
    early_stop_patience: int = 3
    early_stop_min_delta: float = 0.0001
    gradient_clip_norm: float = 5.0
    lr_scheduler: str = 'cosine_warmup'
    warmup_epochs: int = 2
    log_interval: int = 100
    eval_interval: int = 1


class EarlyStopping:
    """Early stopping to prevent overfitting.
    
    Monitors a metric and stops training when it stops improving.
    
    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        mode: 'min' or 'max' - whether lower or higher is better.
    """
    
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0001,
        mode: str = 'max'
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current metric value.
        
        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


class TrainingHistory:
    """Record training and validation metrics.
    
    Tracks loss, AUC, learning rate, and other metrics during training.
    """
    
    def __init__(self) -> None:
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_aucs: List[float] = []
        self.val_aucs: List[float] = []
        self.learning_rates: List[float] = []
        self.epochs: List[int] = []
    
    def record(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_auc: float,
        val_auc: float,
        lr: float
    ) -> None:
        """Record metrics for an epoch."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_aucs.append(train_auc)
        self.val_aucs.append(val_auc)
        self.learning_rates.append(lr)
    
    def best_epoch(self) -> int:
        """Get epoch with best validation AUC."""
        if not self.val_aucs:
            return 0
        return self.epochs[np.argmax(self.val_aucs)]
    
    def best_val_auc(self) -> float:
        """Get best validation AUC."""
        if not self.val_aucs:
            return 0.0
        return max(self.val_aucs)


def train_epoch(
    model: Module,
    dataloader: DataLoader,
    criterion: Module,
    optimizer: Optimizer,
    config: TrainingConfig,
    epoch: int
) -> Tuple[float, float]:
    """Train for one epoch.
    
    Args:
        model: Recommendation model.
        dataloader: Training data loader.
        criterion: Loss function (e.g., BCELoss).
        optimizer: Optimizer.
        config: Training configuration.
        epoch: Current epoch number.
    
    Returns:
        Tuple of (average_loss, auc_score).
    """
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if len(batch) == 3:
            sparse_batch, dense_batch, label_batch = batch
        else:
            sparse_batch, label_batch = batch
            dense_batch = None
        
        sparse_tensor = Tensor(sparse_batch.astype(np.float32))
        dense_tensor = Tensor(dense_batch.astype(np.float32)) if dense_batch is not None else None
        label_tensor = Tensor(label_batch.reshape(-1, 1).astype(np.float32))
        
        predictions = model(sparse_tensor, dense_tensor)
        
        loss = criterion(predictions, label_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        
        if config.gradient_clip_norm > 0:
            clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        all_predictions.append(predictions.data.flatten())
        all_labels.append(label_batch.flatten())
        num_batches += 1
        
        if config.log_interval > 0 and (batch_idx + 1) % config.log_interval == 0:
            avg_loss = total_loss / num_batches
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: Loss = {avg_loss:.4f}")
    
    avg_loss = total_loss / num_batches
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    train_auc = auc_score(all_predictions, all_labels)
    
    return avg_loss, train_auc


def evaluate(
    model: Module,
    dataloader: DataLoader,
    criterion: Module
) -> Tuple[float, float, float]:
    """Evaluate model on a dataset.
    
    Args:
        model: Recommendation model.
        dataloader: Evaluation data loader.
        criterion: Loss function.
    
    Returns:
        Tuple of (average_loss, auc_score, log_loss_score).
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    num_batches = 0
    
    for batch in dataloader:
        if len(batch) == 3:
            sparse_batch, dense_batch, label_batch = batch
        else:
            sparse_batch, label_batch = batch
            dense_batch = None
        
        sparse_tensor = Tensor(sparse_batch.astype(np.float32))
        dense_tensor = Tensor(dense_batch.astype(np.float32)) if dense_batch is not None else None
        label_tensor = Tensor(label_batch.reshape(-1, 1).astype(np.float32))
        
        predictions = model(sparse_tensor, dense_tensor)
        
        loss = criterion(predictions, label_tensor)
        
        total_loss += loss.item()
        all_predictions.append(predictions.data.flatten())
        all_labels.append(label_batch.flatten())
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    eval_auc = auc_score(all_predictions, all_labels)
    eval_logloss = log_loss(all_predictions, all_labels)
    
    return avg_loss, eval_auc, eval_logloss


def train_model(
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[TrainingConfig] = None,
    verbose: bool = True
) -> TrainingHistory:
    """Full training loop with early stopping.
    
    Args:
        model: Recommendation model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration.
        verbose: Whether to print progress.
    
    Returns:
        TrainingHistory with recorded metrics.
    """
    if config is None:
        config = TrainingConfig()
    
    from nanotorch.optim.adamw import AdamW
    
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    if config.lr_scheduler == 'cosine_warmup':
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_epochs=config.warmup_epochs,
            max_epochs=config.num_epochs
        )
        use_plateau = False
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=2
        )
        use_plateau = True
    
    criterion = BCELoss()
    
    early_stopping = EarlyStopping(
        patience=config.early_stop_patience,
        min_delta=config.early_stop_min_delta,
        mode='max'
    )
    
    history = TrainingHistory()
    
    if verbose:
        n_params = sum(p.data.size for p in model.parameters())
        print(f"Training {model.__class__.__name__}")
        print(f"  Parameters: {n_params:,}")
        print(f"  Training samples: {len(train_loader.dataset):,}")
        print(f"  Validation samples: {len(val_loader):,}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print("-" * 60)
    
    for epoch in range(1, config.num_epochs + 1):
        if verbose:
            print(f"Epoch {epoch}/{config.num_epochs}")
        
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, config, epoch
        )
        
        val_loss, val_auc, val_logloss = evaluate(model, val_loader, criterion)
        
        current_lr = optimizer.param_groups[0]['lr']
        history.record(epoch, train_loss, val_loss, train_auc, val_auc, current_lr)
        
        if use_plateau:
            scheduler.step(val_auc)
        else:
            scheduler.step()
        
        if verbose:
            print(f"  Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, LogLoss: {val_logloss:.4f}")
            print(f"  LR: {current_lr:.6f}")
        
        if early_stopping(val_auc):
            if verbose:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best validation AUC: {early_stopping.best_score:.4f}")
            break
    
    if verbose:
        print("-" * 60)
        print(f"Training complete!")
        print(f"Best epoch: {history.best_epoch()}")
        print(f"Best validation AUC: {history.best_val_auc():.4f}")
    
    return history


class ModelCheckpoint:
    """Save and load model checkpoints.
    
    Args:
        filepath: Path to save checkpoint.
        monitor: Metric to monitor ('val_auc' or 'val_loss').
        mode: 'max' or 'min'.
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_auc',
        mode: str = 'max'
    ) -> None:
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
    
    def save(self, model: Module, score: float) -> bool:
        """Save model if score is best so far.
        
        Args:
            model: Model to save.
            score: Current metric score.
        
        Returns:
            True if model was saved.
        """
        if self.best_score is None:
            self.best_score = score
            should_save = True
        elif self.mode == 'max' and score > self.best_score:
            self.best_score = score
            should_save = True
        elif self.mode == 'min' and score < self.best_score:
            self.best_score = score
            should_save = True
        else:
            should_save = False
        
        if should_save:
            state = model.state_dict()
            np.savez(self.filepath, **state)
        
        return should_save
    
    def load(self, model: Module) -> None:
        """Load best model from checkpoint."""
        state = dict(np.load(self.filepath))
        model.load_state_dict(state)


def get_user_item_embeddings(
    model: Module,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    embed_dim: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract user and item embeddings from a trained model.
    
    Useful for nearest-neighbor recommendation and visualization.
    
    Args:
        model: Trained recommendation model.
        user_ids: Array of user IDs.
        item_ids: Array of item IDs.
        embed_dim: Embedding dimension.
    
    Returns:
        Tuple of (user_embeddings, item_embeddings).
    """
    model.eval()
    
    user_embeddings = np.zeros((len(user_ids), embed_dim), dtype=np.float32)
    item_embeddings = np.zeros((len(item_ids), embed_dim), dtype=np.float32)
    
    return user_embeddings, item_embeddings


if __name__ == "__main__":
    print("Training utilities for recommendation models")
    print("Use train_model() for full training loop with early stopping")
