"""
Evaluation metrics for recommendation systems.

This module provides standard metrics for evaluating recommender system performance,
including ranking metrics (NDCG@K, Recall@K, Hit@K, MRR) and classification metrics (AUC).
"""

import numpy as np
from typing import List, Dict, Optional, Union
from nanotorch.tensor import Tensor


def _to_numpy(data: Union[Tensor, np.ndarray]) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(data, Tensor):
        return data.data
    return np.asarray(data)


def auc_score(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray]
) -> float:
    """Compute Area Under ROC Curve (AUC).
    
    AUC measures the probability that a randomly chosen positive example
    is ranked higher than a randomly chosen negative example.
    
    Args:
        predictions: Predicted scores/probabilities, shape (N,) or (N, 1).
        targets: Binary labels (0 or 1), shape (N,) or (N, 1).
    
    Returns:
        AUC score between 0 and 1. 0.5 indicates random guessing.
    
    Example:
        >>> preds = np.array([0.9, 0.4, 0.3, 0.8])
        >>> targets = np.array([1, 0, 0, 1])
        >>> auc = auc_score(preds, targets)
        >>> print(f"AUC: {auc:.4f}")  # Should be 1.0 (perfect ranking)
    """
    predictions = _to_numpy(predictions).flatten()
    targets = _to_numpy(targets).flatten()
    
    # Sort by predictions in descending order
    sorted_indices = np.argsort(-predictions)
    sorted_targets = targets[sorted_indices]
    
    # Count positives and negatives
    n_pos = np.sum(targets == 1)
    n_neg = np.sum(targets == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined, return random baseline
    
    # Compute AUC using trapezoidal rule
    # Count how many negatives are ranked below each positive
    tp_cumsum = np.cumsum(sorted_targets == 1)

    # AUC = (1 / (P * N)) * Σ TP(t) for each negative at threshold t
    auc = np.sum(tp_cumsum[sorted_targets == 0]) / (n_pos * n_neg)
    
    return float(auc)


def log_loss(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    epsilon: float = 1e-15
) -> float:
    """Compute Binary Cross-Entropy (Log Loss).
    
    Args:
        predictions: Predicted probabilities, shape (N,) or (N, 1).
        targets: Binary labels (0 or 1), shape (N,) or (N, 1).
        epsilon: Small constant to avoid log(0).
    
    Returns:
        Average log loss.
    """
    predictions = _to_numpy(predictions).flatten()
    targets = _to_numpy(targets).flatten()
    
    # Clip predictions to avoid log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
    # Compute binary cross-entropy
    loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    
    return float(loss)


def hit_at_k(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    k: int = 10
) -> float:
    """Compute Hit Rate at K (Hit@K).
    
    Hit@K measures whether at least one relevant item appears in the top-K
    recommendations. It's a binary metric per user/query.
    
    Args:
        predictions: Predicted scores, shape (N,) or (batch, N).
            For single user: (num_items,)
            For multiple users: (batch, num_items)
        targets: Binary relevance labels, same shape as predictions.
        k: Number of top items to consider.
    
    Returns:
        Average Hit@K score across all users/queries.
    
    Example:
        >>> predictions = np.array([[0.9, 0.2, 0.7, 0.1],  # User 1
        ...                        [0.3, 0.8, 0.1, 0.9]])  # User 2
        >>> targets = np.array([[1, 0, 0, 0],  # User 1: item 0 relevant
        ...                     [0, 1, 0, 0]])  # User 2: item 1 relevant
        >>> hit_at_k(predictions, targets, k=2)
        # User 1: top-2 are items 0,2 -> item 0 is relevant -> hit
        # User 2: top-2 are items 1,3 -> item 1 is relevant -> hit
        # Result: 1.0
    """
    predictions = _to_numpy(predictions)
    targets = _to_numpy(targets)
    
    # Handle single user case
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
        targets = targets.reshape(1, -1)
    
    batch_size = predictions.shape[0]
    hits = 0.0
    
    for i in range(batch_size):
        pred_i = predictions[i]
        target_i = targets[i]
        
        # Get indices of top-k items
        top_k_indices = np.argsort(-pred_i)[:k]
        
        # Check if any relevant item is in top-k
        if np.any(target_i[top_k_indices] > 0):
            hits += 1.0
    
    return hits / batch_size


def recall_at_k(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    k: int = 10
) -> float:
    """Compute Recall at K (Recall@K).
    
    Recall@K measures what fraction of relevant items are retrieved in top-K.
    
    Formula:
        Recall@K = |relevant items in top-K| / |total relevant items|
    
    Args:
        predictions: Predicted scores, shape (N,) or (batch, N).
        targets: Binary relevance labels, same shape as predictions.
        k: Number of top items to consider.
    
    Returns:
        Average Recall@K across all users/queries.
    
    Example:
        >>> predictions = np.array([[0.9, 0.8, 0.3, 0.1]])  # 1 user, 4 items
        >>> targets = np.array([[1, 1, 0, 0]])  # Items 0 and 1 are relevant
        >>> recall_at_k(predictions, targets, k=2)
        # Top-2 are items 0,1, both relevant
        # Recall = 2/2 = 1.0
    """
    predictions = _to_numpy(predictions)
    targets = _to_numpy(targets)
    
    # Handle single user case
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
        targets = targets.reshape(1, -1)
    
    batch_size = predictions.shape[0]
    recalls = 0.0
    
    for i in range(batch_size):
        pred_i = predictions[i]
        target_i = targets[i]
        
        # Count total relevant items
        total_relevant = np.sum(target_i > 0)
        
        if total_relevant == 0:
            continue  # Skip if no relevant items
        
        # Get indices of top-k items
        top_k_indices = np.argsort(-pred_i)[:k]
        
        # Count relevant items in top-k
        relevant_in_top_k = np.sum(target_i[top_k_indices] > 0)
        
        recalls += relevant_in_top_k / total_relevant
    
    return recalls / batch_size


def precision_at_k(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    k: int = 10
) -> float:
    """Compute Precision at K (Precision@K).
    
    Precision@K measures what fraction of top-K items are relevant.
    
    Formula:
        Precision@K = |relevant items in top-K| / K
    
    Args:
        predictions: Predicted scores, shape (N,) or (batch, N).
        targets: Binary relevance labels, same shape as predictions.
        k: Number of top items to consider.
    
    Returns:
        Average Precision@K across all users/queries.
    """
    predictions = _to_numpy(predictions)
    targets = _to_numpy(targets)
    
    # Handle single user case
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
        targets = targets.reshape(1, -1)
    
    batch_size = predictions.shape[0]
    precisions = 0.0
    
    for i in range(batch_size):
        pred_i = predictions[i]
        target_i = targets[i]
        
        # Get indices of top-k items
        top_k_indices = np.argsort(-pred_i)[:k]
        
        # Count relevant items in top-k
        relevant_in_top_k = np.sum(target_i[top_k_indices] > 0)
        
        precisions += relevant_in_top_k / k
    
    return precisions / batch_size


def dcg_at_k(
    relevances: Union[Tensor, np.ndarray],
    k: int = 10,
    method: str = 'standard'
) -> float:
    """Compute Discounted Cumulative Gain at K (DCG@K).
    
    DCG measures the ranking quality by summing relevant scores with
    position-based discounts.
    
    Formula (standard):
        DCG@K = Σ_{i=1}^{K} rel_i / log2(i + 1)
    
    Formula (alternative, used by some search engines):
        DCG@K = Σ_{i=1}^{K} (2^rel_i - 1) / log2(i + 1)
    
    Args:
        relevances: Relevance scores in ranked order, shape (K,).
        k: Number of items to consider.
        method: 'standard' or 'alternative' DCG formula.
    
    Returns:
        DCG@K score.
    """
    relevances = _to_numpy(relevances).flatten()[:k]
    
    if len(relevances) == 0:
        return 0.0
    
    # Position-based discount: 1 / log2(rank + 1)
    # rank starts from 1, so we use (i + 2) where i is 0-indexed
    discounts = 1.0 / np.log2(np.arange(len(relevances)) + 2)
    
    if method == 'alternative':
        # Alternative formula: (2^rel - 1)
        relevances = np.power(2, relevances) - 1
    
    return float(np.sum(relevances * discounts))


def ndcg_at_k(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    k: int = 10,
    method: str = 'standard'
) -> float:
    """Compute Normalized Discounted Cumulative Gain at K (NDCG@K).
    
    NDCG normalizes DCG by the ideal DCG (best possible ranking).
    
    Formula:
        NDCG@K = DCG@K / IDCG@K
        
    where IDCG@K is the DCG of the ideal ranking (sorted by relevance).
    
    Args:
        predictions: Predicted scores, shape (N,) or (batch, N).
        targets: Relevance scores (can be graded), same shape as predictions.
        k: Number of top items to consider.
        method: 'standard' or 'alternative' DCG formula.
    
    Returns:
        Average NDCG@K across all users/queries (0 to 1).
    
    Example:
        >>> predictions = np.array([[0.9, 0.3, 0.8, 0.1]])  # 1 user
        >>> targets = np.array([[3, 0, 2, 1]])  # Graded relevance
        >>> ndcg_at_k(predictions, targets, k=4)
        # Predicted order: [0, 2, 3, 1] with relevances [3, 2, 1, 0]
        # Ideal order: [3, 2, 1, 0] with relevances [3, 2, 1, 0]
        # Perfect NDCG = 1.0
    """
    predictions = _to_numpy(predictions)
    targets = _to_numpy(targets)
    
    # Handle single user case
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
        targets = targets.reshape(1, -1)
    
    batch_size = predictions.shape[0]
    ndcgs = 0.0
    
    for i in range(batch_size):
        pred_i = predictions[i]
        target_i = targets[i]
        
        # Get indices sorted by predicted scores (descending)
        ranked_indices = np.argsort(-pred_i)
        
        # Get relevances in predicted ranking order
        ranked_relevances = target_i[ranked_indices]
        
        # Compute DCG@K for predicted ranking
        dcg = dcg_at_k(ranked_relevances, k, method)
        
        # Compute IDCG@K (ideal DCG)
        ideal_relevances = np.sort(target_i)[::-1]
        idcg = dcg_at_k(ideal_relevances, k, method)
        
        if idcg > 0:
            ndcgs += dcg / idcg
    
    return ndcgs / batch_size


def mrr(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray]
) -> float:
    """Compute Mean Reciprocal Rank (MRR).
    
    MRR measures the average reciprocal rank of the first relevant item.
    
    Formula:
        MRR = (1/|Q|) * Σ_{q=1}^{|Q|} 1 / rank_q
        
    where rank_q is the position of the first relevant item for query q.
    
    Args:
        predictions: Predicted scores, shape (N,) or (batch, N).
        targets: Binary relevance labels, same shape as predictions.
    
    Returns:
        MRR score between 0 and 1.
    
    Example:
        >>> predictions = np.array([[0.9, 0.3, 0.8],   # User 1
        ...                        [0.1, 0.8, 0.5]])   # User 2
        >>> targets = np.array([[0, 0, 1],   # Item 2 relevant
        ...                     [1, 0, 0]])  # Item 0 relevant
        >>> mrr(predictions, targets)
        # User 1: order [0, 2, 1], first relevant at rank 2 -> RR = 1/2
        # User 2: order [1, 2, 0], first relevant at rank 3 -> RR = 1/3
        # MRR = (1/2 + 1/3) / 2 = 0.4167
    """
    predictions = _to_numpy(predictions)
    targets = _to_numpy(targets)
    
    # Handle single user case
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
        targets = targets.reshape(1, -1)
    
    batch_size = predictions.shape[0]
    reciprocal_ranks = 0.0
    
    for i in range(batch_size):
        pred_i = predictions[i]
        target_i = targets[i]
        
        # Get ranking order (descending by score)
        ranked_indices = np.argsort(-pred_i)
        
        # Find position of first relevant item
        for rank, idx in enumerate(ranked_indices, start=1):
            if target_i[idx] > 0:
                reciprocal_ranks += 1.0 / rank
                break
    
    return reciprocal_ranks / batch_size


def map_score(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    k: Optional[int] = None
) -> float:
    """Compute Mean Average Precision (MAP).
    
    MAP computes the mean of Average Precision (AP) scores for each query.
    AP is the average of precision values at each relevant item position.
    
    Formula:
        AP = (1/|R|) * Σ_{k=1}^{n} P(k) * rel(k)
        
    where P(k) is precision at position k, rel(k) is 1 if item at k is relevant.
    
    Args:
        predictions: Predicted scores, shape (N,) or (batch, N).
        targets: Binary relevance labels, same shape as predictions.
        k: If specified, compute MAP@K (only consider top-K items).
    
    Returns:
        MAP score between 0 and 1.
    """
    predictions = _to_numpy(predictions)
    targets = _to_numpy(targets)
    
    # Handle single user case
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
        targets = targets.reshape(1, -1)
    
    batch_size = predictions.shape[0]
    ap_scores = 0.0
    
    for i in range(batch_size):
        pred_i = predictions[i]
        target_i = targets[i]
        
        # Count total relevant items
        total_relevant = np.sum(target_i > 0)
        
        if total_relevant == 0:
            continue
        
        # Get ranking order
        ranked_indices = np.argsort(-pred_i)
        
        if k is not None:
            ranked_indices = ranked_indices[:k]
        
        # Compute average precision
        num_relevant_found = 0
        precision_sum = 0.0
        
        for rank, idx in enumerate(ranked_indices, start=1):
            if target_i[idx] > 0:
                num_relevant_found += 1
                precision_sum += num_relevant_found / rank
        
        # Normalize by total relevant (not just those in top-k)
        ap_scores += precision_sum / total_relevant
    
    return ap_scores / batch_size


def compute_all_ranking_metrics(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    ks: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """Compute all ranking metrics at once.
    
    Convenience function to compute multiple metrics in one call.
    
    Args:
        predictions: Predicted scores, shape (N,) or (batch, N).
        targets: Binary relevance labels, same shape as predictions.
        ks: List of K values for @K metrics.
    
    Returns:
        Dictionary containing all metric values:
        - 'auc': AUC score
        - 'mrr': Mean Reciprocal Rank
        - 'map': Mean Average Precision
        - 'hit@{k}': Hit@K for each k in ks
        - 'recall@{k}': Recall@K for each k in ks
        - 'precision@{k}': Precision@K for each k in ks
        - 'ndcg@{k}': NDCG@K for each k in ks
    """
    metrics = {}
    
    # Compute metrics that don't depend on K
    metrics['auc'] = auc_score(predictions, targets)
    metrics['mrr'] = mrr(predictions, targets)
    metrics['map'] = map_score(predictions, targets)
    
    # Compute metrics at each K
    for k in ks:
        metrics[f'hit@{k}'] = hit_at_k(predictions, targets, k)
        metrics[f'recall@{k}'] = recall_at_k(predictions, targets, k)
        metrics[f'precision@{k}'] = precision_at_k(predictions, targets, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(predictions, targets, k)
    
    return metrics


class RecommenderMetrics:
    """Class to track and aggregate recommendation metrics during training.
    
    Useful for computing running averages and reporting metrics during
    training or evaluation loops.
    
    Example:
        >>> metrics_tracker = RecommenderMetrics(ks=[1, 5, 10])
        >>> for batch in dataloader:
        ...     predictions = model(batch)
        ...     metrics_tracker.update(predictions, targets)
        >>> print(metrics_tracker.compute())
    """
    
    def __init__(self, ks: List[int] = [1, 5, 10, 20]):
        """Initialize metrics tracker.
        
        Args:
            ks: List of K values for @K metrics.
        """
        self.ks = ks
        self.reset()
    
    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self._predictions = []
        self._targets = []
        self._count = 0
    
    def update(
        self,
        predictions: Union[Tensor, np.ndarray],
        targets: Union[Tensor, np.ndarray],
        batch_size: Optional[int] = None
    ) -> None:
        """Accumulate predictions and targets.
        
        Args:
            predictions: Batch of predictions.
            targets: Batch of targets.
            batch_size: Optional batch size (inferred if not provided).
        """
        predictions = _to_numpy(predictions)
        targets = _to_numpy(targets)
        
        self._predictions.append(predictions)
        self._targets.append(targets)
        self._count += batch_size if batch_size else len(predictions)
    
    def compute(self) -> Dict[str, float]:
        """Compute metrics on accumulated data.
        
        Returns:
            Dictionary of computed metrics.
        """
        if not self._predictions:
            return {}
        
        # Concatenate all batches
        all_predictions = np.concatenate(self._predictions, axis=0)
        all_targets = np.concatenate(self._targets, axis=0)
        
        return compute_all_ranking_metrics(all_predictions, all_targets, self.ks)
    
    def __str__(self) -> str:
        """String representation of current metrics."""
        metrics = self.compute()
        if not metrics:
            return "RecommenderMetrics(no data)"
        
        parts = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        return "RecommenderMetrics(" + ", ".join(parts) + ")"
