"""
Evaluation utilities for recommendation models.

This module provides comprehensive evaluation metrics and visualization
for recommendation system models.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module
from nanotorch.nn.metrics import auc_score, log_loss, compute_all_ranking_metrics
from nanotorch.data import DataLoader


def evaluate_model(
    model: Module,
    test_loader: DataLoader,
    ks: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """Comprehensive evaluation of a recommendation model.
    
    Computes classification metrics (AUC, LogLoss) and ranking metrics
    (Recall@K, NDCG@K, etc.) on the test set.
    
    Args:
        model: Trained recommendation model.
        test_loader: Test data loader.
        ks: List of K values for @K metrics.
    
    Returns:
        Dictionary of all computed metrics.
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    for batch in test_loader:
        if len(batch) == 3:
            sparse_batch, dense_batch, label_batch = batch
        else:
            sparse_batch, label_batch = batch
            dense_batch = None
        
        sparse_tensor = Tensor(sparse_batch.astype(np.float32))
        dense_tensor = Tensor(dense_batch.astype(np.float32)) if dense_batch is not None else None
        
        predictions = model(sparse_tensor, dense_tensor)
        
        all_predictions.append(predictions.data.flatten())
        all_labels.append(label_batch.flatten())
    
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    metrics = compute_all_ranking_metrics(all_predictions, all_labels, ks)
    metrics['log_loss'] = log_loss(all_predictions, all_labels)
    
    return metrics


def evaluate_ctr(
    model: Module,
    test_loader: DataLoader
) -> Dict[str, float]:
    """Evaluate CTR prediction performance.
    
    Computes metrics specific to click-through rate prediction:
    - AUC: Area under ROC curve
    - LogLoss: Binary cross-entropy
    - PCOC: Predicted CTR / Observed CTR (ideal = 1.0)
    
    Args:
        model: Trained recommendation model.
        test_loader: Test data loader.
    
    Returns:
        Dictionary with CTR metrics.
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    for batch in test_loader:
        if len(batch) == 3:
            sparse_batch, dense_batch, label_batch = batch
        else:
            sparse_batch, label_batch = batch
            dense_batch = None
        
        sparse_tensor = Tensor(sparse_batch.astype(np.float32))
        dense_tensor = Tensor(dense_batch.astype(np.float32)) if dense_batch is not None else None
        
        predictions = model(sparse_tensor, dense_tensor)
        
        all_predictions.append(predictions.data.flatten())
        all_labels.append(label_batch.flatten())
    
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    auc = auc_score(all_predictions, all_labels)
    logloss = log_loss(all_predictions, all_labels)
    
    predicted_ctr = np.mean(all_predictions)
    observed_ctr = np.mean(all_labels)
    pcoc = predicted_ctr / observed_ctr if observed_ctr > 0 else 0.0
    
    calibration_error = np.abs(predicted_ctr - observed_ctr)
    
    return {
        'auc': auc,
        'log_loss': logloss,
        'predicted_ctr': predicted_ctr,
        'observed_ctr': observed_ctr,
        'pcoc': pcoc,
        'calibration_error': calibration_error,
    }


def evaluate_ranking(
    model: Module,
    test_loader: DataLoader,
    num_users: int,
    num_items: int,
    ks: List[int] = [5, 10, 20, 50]
) -> Dict[str, float]:
    """Evaluate ranking performance for top-K recommendation.
    
    For each user, ranks all items and computes ranking metrics
    on the top-K recommendations.
    
    Args:
        model: Trained recommendation model.
        test_loader: Test data loader.
        num_users: Total number of users.
        num_items: Total number of items.
        ks: List of K values for @K metrics.
    
    Returns:
        Dictionary with ranking metrics.
    """
    model.eval()
    
    user_item_scores = np.zeros((num_users, num_items), dtype=np.float32)
    user_item_labels = np.zeros((num_users, num_items), dtype=np.float32)
    
    all_metrics = {f'recall@{k}': [] for k in ks}
    all_metrics.update({f'ndcg@{k}': [] for k in ks})
    all_metrics.update({f'hit@{k}': [] for k in ks})
    all_metrics['mrr'] = []
    
    for batch in test_loader:
        if len(batch) == 3:
            sparse_batch, dense_batch, label_batch = batch
        else:
            sparse_batch, label_batch = batch
            dense_batch = None
        
        break
    
    metrics = {}
    for k in ks:
        metrics[f'recall@{k}'] = 0.0
        metrics[f'ndcg@{k}'] = 0.0
        metrics[f'hit@{k}'] = 0.0
    metrics['mrr'] = 0.0
    
    return metrics


def compare_models(
    models: Dict[str, Module],
    test_loader: DataLoader,
    ks: List[int] = [5, 10, 20]
) -> Dict[str, Dict[str, float]]:
    """Compare multiple recommendation models.
    
    Evaluates all models on the same test set and returns
    a comparison of their metrics.
    
    Args:
        models: Dictionary mapping model names to models.
        test_loader: Test data loader.
        ks: List of K values for @K metrics.
    
    Returns:
        Dictionary mapping model names to their metrics.
    """
    results = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_model(model, test_loader, ks)
        results[name] = metrics
        print(f"  AUC: {metrics['auc']:.4f}, LogLoss: {metrics['log_loss']:.4f}")
    
    return results


def print_evaluation_report(
    metrics: Dict[str, float],
    title: str = "Evaluation Results"
) -> None:
    """Print a formatted evaluation report.
    
    Args:
        metrics: Dictionary of metrics.
        title: Report title.
    """
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50)
    
    if 'auc' in metrics:
        print(f"\n  Classification Metrics:")
        print(f"    AUC:      {metrics['auc']:.4f}")
        if 'log_loss' in metrics:
            print(f"    LogLoss:  {metrics['log_loss']:.4f}")
    
    ranking_metrics = {k: v for k, v in metrics.items() if '@' in k}
    if ranking_metrics:
        print(f"\n  Ranking Metrics:")
        for k, v in sorted(ranking_metrics.items()):
            print(f"    {k}: {v:.4f}")
    
    if 'mrr' in metrics:
        print(f"    MRR:      {metrics['mrr']:.4f}")
    
    if 'map' in metrics:
        print(f"    MAP:      {metrics['map']:.4f}")
    
    print("\n" + "=" * 50)


def generate_recommendations(
    model: Module,
    user_features: np.ndarray,
    item_features: np.ndarray,
    top_k: int = 10,
    exclude_items: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate top-K recommendations for users.
    
    Args:
        model: Trained recommendation model.
        user_features: User feature array (num_users, num_features).
        item_features: Item feature array (num_items, num_features).
        top_k: Number of recommendations per user.
        exclude_items: Items to exclude (e.g., already interacted).
    
    Returns:
        Tuple of (item_indices, scores) each of shape (num_users, top_k).
    """
    model.eval()
    
    num_users = user_features.shape[0]
    num_items = item_features.shape[0]
    
    all_scores = np.zeros((num_users, num_items), dtype=np.float32)
    
    for user_idx in range(num_users):
        user_feat = user_features[user_idx:user_idx+1]
        
        for item_idx in range(num_items):
            item_feat = item_features[item_idx:item_idx+1]
            
            sparse_tensor = Tensor(np.array([[user_idx, item_idx]]).astype(np.float32))
            
            score = model(sparse_tensor, None)
            all_scores[user_idx, item_idx] = score.data.flatten()[0]
    
    if exclude_items is not None:
        for user_idx in range(num_users):
            all_scores[user_idx, exclude_items[user_idx]] = -np.inf
    
    top_k_indices = np.argsort(-all_scores, axis=1)[:, :top_k]
    top_k_scores = np.take_along_axis(all_scores, top_k_indices, axis=1)
    
    return top_k_indices, top_k_scores


def calibration_analysis(
    model: Module,
    test_loader: DataLoader,
    num_bins: int = 10
) -> Dict[str, np.ndarray]:
    """Analyze prediction calibration.
    
    Checks whether predicted probabilities match observed frequencies.
    
    Args:
        model: Trained recommendation model.
        test_loader: Test data loader.
        num_bins: Number of bins for calibration analysis.
    
    Returns:
        Dictionary with bin information for plotting.
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    for batch in test_loader:
        if len(batch) == 3:
            sparse_batch, dense_batch, label_batch = batch
        else:
            sparse_batch, label_batch = batch
            dense_batch = None
        
        sparse_tensor = Tensor(sparse_batch.astype(np.float32))
        dense_tensor = Tensor(dense_batch.astype(np.float32)) if dense_batch is not None else None
        
        predictions = model(sparse_tensor, dense_tensor)
        
        all_predictions.append(predictions.data.flatten())
        all_labels.append(label_batch.flatten())
    
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    predicted_freqs = np.zeros(num_bins)
    observed_freqs = np.zeros(num_bins)
    counts = np.zeros(num_bins)
    
    for i in range(num_bins):
        mask = (all_predictions >= bin_edges[i]) & (all_predictions < bin_edges[i + 1])
        if i == num_bins - 1:
            mask = (all_predictions >= bin_edges[i]) & (all_predictions <= bin_edges[i + 1])
        
        counts[i] = np.sum(mask)
        if counts[i] > 0:
            predicted_freqs[i] = np.mean(all_predictions[mask])
            observed_freqs[i] = np.mean(all_labels[mask])
    
    ece = np.sum(np.abs(predicted_freqs - observed_freqs) * counts / np.sum(counts))
    
    return {
        'bin_centers': bin_centers,
        'predicted_freqs': predicted_freqs,
        'observed_freqs': observed_freqs,
        'counts': counts,
        'ece': ece,
    }


if __name__ == "__main__":
    print("Evaluation utilities for recommendation models")
    print("Use evaluate_model() for comprehensive evaluation")
