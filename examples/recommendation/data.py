"""
Data utilities for recommendation system examples.

This module provides synthetic data generation for recommendation experiments,
including user features, item features, and interaction labels.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from nanotorch.data import Dataset, TensorDataset, DataLoader
from nanotorch.nn.recommender import SparseFeat, DenseFeat


class RecommendationDataset(Dataset):
    """Dataset for recommendation systems with sparse and dense features.
    
    This dataset handles the common recommendation data format:
    - Sparse features: categorical features that need embedding (user_id, item_id, etc.)
    - Dense features: numerical features (price, rating, etc.)
    - Labels: binary (click/no-click) or numerical (rating)
    
    Args:
        sparse_features: Dict mapping feature name to integer indices.
        dense_features: Dict mapping feature name to numerical values.
        labels: Binary or numerical labels.
    
    Example:
        >>> sparse = {'user_id': [1, 2, 3], 'item_id': [10, 20, 30]}
        >>> dense = {'price': [9.99, 19.99, 29.99]}
        >>> labels = [1, 0, 1]
        >>> dataset = RecommendationDataset(sparse, dense, labels)
    """
    
    def __init__(
        self,
        sparse_features: Dict[str, np.ndarray],
        dense_features: Dict[str, np.ndarray],
        labels: np.ndarray
    ) -> None:
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.labels = labels
        
        # Validate sizes match
        n_samples = len(labels)
        for name, feat in sparse_features.items():
            if len(feat) != n_samples:
                raise ValueError(f"Sparse feature '{name}' has {len(feat)} samples, expected {n_samples}")
        for name, feat in dense_features.items():
            if len(feat) != n_samples:
                raise ValueError(f"Dense feature '{name}' has {len(feat)} samples, expected {n_samples}")
        
        self.sparse_feature_names = list(sparse_features.keys())
        self.dense_feature_names = list(dense_features.keys())
        self._n_samples = n_samples
    
    def __len__(self) -> int:
        return self._n_samples
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, int], Dict[str, float], float]:
        sparse = {name: self.sparse_features[name][idx] for name in self.sparse_feature_names}
        dense = {name: self.dense_features[name][idx] for name in self.dense_feature_names}
        label = self.labels[idx]
        return sparse, dense, label
    
    def get_sparse_array(self) -> np.ndarray:
        """Get sparse features as a 2D array (samples x features)."""
        return np.column_stack([self.sparse_features[name] for name in self.sparse_feature_names])
    
    def get_dense_array(self) -> np.ndarray:
        """Get dense features as a 2D array (samples x features)."""
        return np.column_stack([self.dense_features[name] for name in self.dense_feature_names])
    
    def get_labels(self) -> np.ndarray:
        """Get labels as a 1D array."""
        return self.labels


def generate_synthetic_data(
    num_samples: int = 100000,
    num_users: int = 10000,
    num_items: int = 5000,
    num_categories: int = 50,
    num_brands: int = 200,
    num_cities: int = 100,
    click_rate: float = 0.05,
    random_seed: int = 42
) -> Tuple[RecommendationDataset, List[SparseFeat], List[DenseFeat]]:
    """Generate synthetic recommendation data for CTR prediction.
    
    Creates realistic synthetic data with user features, item features,
    and context features. The click probability is determined by latent
    user and item factors, creating a learnable pattern.
    
    Data Schema:
        Sparse Features:
        - user_id: User identifier (0 to num_users-1)
        - item_id: Item identifier (0 to num_items-1)
        - user_city: User's city
        - item_category: Item category
        - item_brand: Item brand
        - device_type: Device type (mobile, desktop, tablet)
        - hour: Hour of day (0-23)
        
        Dense Features:
        - user_age: User age (normalized)
        - user_tenure: Days since user registration (normalized)
        - item_price: Item price (normalized)
        - item_rating: Average item rating (normalized)
        - user_item_history: Number of past user-item interactions (normalized)
    
    Args:
        num_samples: Number of interaction samples to generate.
        num_users: Number of unique users.
        num_items: Number of unique items.
        num_categories: Number of item categories.
        num_brands: Number of item brands.
        num_cities: Number of user cities.
        click_rate: Target click-through rate (affects class balance).
        random_seed: Random seed for reproducibility.
    
    Returns:
        Tuple of (dataset, sparse_feature_configs, dense_feature_configs)
    """
    np.random.seed(random_seed)
    
    # Generate latent factors for users and items
    # These create the "true" click probability
    latent_dim = 16
    user_factors = np.random.randn(num_users, latent_dim).astype(np.float32) * 0.1
    item_factors = np.random.randn(num_items, latent_dim).astype(np.float32) * 0.1
    
    # User features
    user_ids = np.random.randint(0, num_users, num_samples)
    user_cities = np.random.randint(0, num_cities, num_samples)
    user_ages = np.random.randint(18, 70, num_samples).astype(np.float32)
    user_tenures = np.random.exponential(365, num_samples).astype(np.float32)  # Days
    
    # Item features
    item_ids = np.random.randint(0, num_items, num_samples)
    item_categories = np.random.randint(0, num_categories, num_samples)
    item_brands = np.random.randint(0, num_brands, num_samples)
    item_prices = np.abs(np.random.exponential(50, num_samples).astype(np.float32))
    item_ratings = np.clip(np.random.normal(4.0, 0.5, num_samples).astype(np.float32), 1.0, 5.0)
    
    # Context features
    device_types = np.random.randint(0, 3, num_samples)  # 0=mobile, 1=desktop, 2=tablet
    hours = np.random.randint(0, 24, num_samples)
    
    # Interaction history (user-item pair frequency)
    user_item_history = np.random.poisson(2, num_samples).astype(np.float32)
    
    # Compute click probability using latent factors
    # Click = sigmoid(user_factor · item_factor + biases + noise)
    user_biases = np.random.randn(num_users).astype(np.float32) * 0.1
    item_biases = np.random.randn(num_items).astype(np.float32) * 0.1
    
    # Dot product of latent factors
    latent_scores = np.sum(user_factors[user_ids] * item_factors[item_ids], axis=1)
    
    # Add biases
    scores = latent_scores + user_biases[user_ids] + item_biases[item_ids]
    
    # Add context effects
    # Higher CTR for mobile, lower for tablet
    device_effect = np.where(device_types == 0, 0.2, np.where(device_types == 1, 0.0, -0.1))
    # Higher CTR during peak hours (18-22)
    hour_effect = np.where((hours >= 18) & (hours <= 22), 0.2, 0.0)
    # Higher CTR for cheaper items
    price_effect = -item_prices / 200.0
    # Higher CTR for users with more history
    history_effect = user_item_history * 0.1
    
    scores = scores + device_effect + hour_effect + price_effect + history_effect
    
    # Apply sigmoid to get probabilities
    probs = 1.0 / (1.0 + np.exp(-scores))
    
    # Adjust threshold to achieve target click rate
    # Find threshold that gives approximately click_rate positive samples
    threshold = np.percentile(probs, (1 - click_rate) * 100)
    labels = (probs > threshold).astype(np.float32)
    
    # Normalize dense features
    user_ages_norm = (user_ages - 18) / (70 - 18)  # 0-1
    user_tenures_norm = np.clip(user_tenures / 730, 0, 1)  # Clip at 2 years
    item_prices_norm = np.clip(item_prices / 500, 0, 1)  # Clip at $500
    item_ratings_norm = (item_ratings - 1.0) / 4.0  # 0-1
    user_item_history_norm = np.clip(user_item_history / 10, 0, 1)  # Clip at 10
    
    # Build sparse features dict
    sparse_features = {
        'user_id': user_ids,
        'item_id': item_ids,
        'user_city': user_cities,
        'item_category': item_categories,
        'item_brand': item_brands,
        'device_type': device_types,
        'hour': hours,
    }
    
    # Build dense features dict
    dense_features = {
        'user_age': user_ages_norm,
        'user_tenure': user_tenures_norm,
        'item_price': item_prices_norm,
        'item_rating': item_ratings_norm,
        'user_item_history': user_item_history_norm,
    }
    
    # Define feature configurations
    sparse_feature_configs = [
        SparseFeat('user_id', num_users, embedding_dim=32),
        SparseFeat('item_id', num_items, embedding_dim=32),
        SparseFeat('user_city', num_cities, embedding_dim=8),
        SparseFeat('item_category', num_categories, embedding_dim=8),
        SparseFeat('item_brand', num_brands, embedding_dim=8),
        SparseFeat('device_type', 3, embedding_dim=4),
        SparseFeat('hour', 24, embedding_dim=4),
    ]
    
    dense_feature_configs = [
        DenseFeat('user_age'),
        DenseFeat('user_tenure'),
        DenseFeat('item_price'),
        DenseFeat('item_rating'),
        DenseFeat('user_item_history'),
    ]
    
    dataset = RecommendationDataset(sparse_features, dense_features, labels)
    
    print(f"Generated synthetic recommendation data:")
    print(f"  Samples: {num_samples:,}")
    print(f"  Users: {num_users:,}")
    print(f"  Items: {num_items:,}")
    print(f"  Click rate: {labels.mean():.4f} (target: {click_rate})")
    print(f"  Sparse features: {len(sparse_feature_configs)}")
    print(f"  Dense features: {len(dense_feature_configs)}")
    
    return dataset, sparse_feature_configs, dense_feature_configs


def create_dataloaders(
    dataset: RecommendationDataset,
    batch_size: int = 256,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders from a dataset.
    
    Args:
        dataset: RecommendationDataset instance.
        batch_size: Batch size for all loaders.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        seed: Random seed for reproducibility.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    n_samples = len(dataset)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    
    # Create indices and split
    indices = np.random.RandomState(seed).permutation(n_samples)
    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:n_train + n_val].tolist()
    test_indices = indices[n_train + n_val:].tolist()
    
    # Create TensorDataset for efficient batching
    sparse_array = dataset.get_sparse_array()
    dense_array = dataset.get_dense_array()
    labels = dataset.get_labels()
    
    train_dataset = TensorDataset(
        sparse_array[train_indices],
        dense_array[train_indices],
        labels[train_indices]
    )
    val_dataset = TensorDataset(
        sparse_array[val_indices],
        dense_array[val_indices],
        labels[val_indices]
    )
    test_dataset = TensorDataset(
        sparse_array[test_indices],
        dense_array[test_indices],
        labels[test_indices]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Data split:")
    print(f"  Train: {len(train_dataset):,} samples ({len(train_dataset)/n_samples:.1%})")
    print(f"  Val:   {len(val_dataset):,} samples ({len(val_dataset)/n_samples:.1%})")
    print(f"  Test:  {len(test_dataset):,} samples ({len(test_dataset)/n_samples:.1%})")
    
    return train_loader, val_loader, test_loader


def generate_sequential_data(
    num_users: int = 10000,
    num_items: int = 5000,
    max_seq_len: int = 50,
    min_seq_len: int = 5,
    avg_seq_len: float = 20,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sequential user-item interaction data.
    
    Creates user behavior sequences for sequential recommendation models
    like GRU4Rec, SASRec, etc.
    
    Args:
        num_users: Number of unique users.
        num_items: Number of unique items.
        max_seq_len: Maximum sequence length.
        min_seq_len: Minimum sequence length.
        avg_seq_len: Average sequence length.
        random_seed: Random seed for reproducibility.
    
    Returns:
        Tuple of:
        - user_ids: User identifiers (num_users,)
        - sequences: Item sequences (num_users, max_seq_len)
        - seq_lengths: Actual length of each sequence (num_users,)
    """
    np.random.seed(random_seed)
    
    # Generate sequence lengths from a distribution
    seq_lengths = np.clip(
        np.random.poisson(avg_seq_len, num_users),
        min_seq_len,
        max_seq_len
    ).astype(np.int32)
    
    # Generate item sequences
    sequences = np.zeros((num_users, max_seq_len), dtype=np.int64)
    
    # Item popularity follows a power law
    item_probs = np.random.power(0.5, num_items)
    item_probs = item_probs / item_probs.sum()
    
    for i in range(num_users):
        seq_len = seq_lengths[i]
        # Sample items from popularity distribution
        sequences[i, :seq_len] = np.random.choice(
            num_items, size=seq_len, replace=False, p=item_probs
        )
    
    user_ids = np.arange(num_users)
    
    print(f"Generated sequential data:")
    print(f"  Users: {num_users:,}")
    print(f"  Items: {num_items:,}")
    print(f"  Max seq len: {max_seq_len}")
    print(f"  Avg seq len: {seq_lengths.mean():.1f}")
    
    return user_ids, sequences, seq_lengths


class NegativeSampler:
    """Negative sampler for recommendation training.
    
    Generates negative samples (items not interacted with) for training
    ranking models like BPR, WARP, etc.
    
    Args:
        num_items: Total number of items.
        num_negatives: Number of negative samples per positive.
        sampling_strategy: 'uniform', 'popularity', or 'inbatch'.
    
    Example:
        >>> sampler = NegativeSampler(num_items=1000, num_negatives=4)
        >>> pos_items = np.array([10, 20, 30])
        >>> neg_items = sampler.sample(pos_items)
        >>> # neg_items shape: (3, 4)
    """
    
    def __init__(
        self,
        num_items: int,
        num_negatives: int = 4,
        sampling_strategy: str = 'uniform'
    ) -> None:
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.sampling_strategy = sampling_strategy
        self._item_probs = None
    
    def set_item_distribution(self, item_counts: np.ndarray) -> None:
        """Set item distribution for popularity-based sampling.
        
        Args:
            item_counts: Array of item interaction counts.
        """
        if self.sampling_strategy == 'popularity':
            # Sample proportional to sqrt of frequency (popular items more likely)
            self._item_probs = np.sqrt(item_counts + 1)
            self._item_probs = self._item_probs / self._item_probs.sum()
    
    def sample(
        self,
        positive_items: np.ndarray,
        exclude_items: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Sample negative items.
        
        Args:
            positive_items: Array of positive item indices.
            exclude_items: Optional additional items to exclude.
        
        Returns:
            Array of negative item indices, shape (len(positive_items), num_negatives).
        """
        batch_size = len(positive_items)
        negatives = np.zeros((batch_size, self.num_negatives), dtype=np.int64)
        
        for i in range(batch_size):
            # Items to exclude
            excluded = {positive_items[i]}
            if exclude_items is not None:
                excluded.update(exclude_items[i])
            
            # Sample negatives
            for j in range(self.num_negatives):
                while True:
                    if self.sampling_strategy == 'uniform':
                        neg = np.random.randint(0, self.num_items)
                    elif self.sampling_strategy == 'popularity' and self._item_probs is not None:
                        neg = np.random.choice(self.num_items, p=self._item_probs)
                    else:
                        neg = np.random.randint(0, self.num_items)
                    
                    if neg not in excluded:
                        negatives[i, j] = neg
                        excluded.add(neg)
                        break
        
        return negatives


def create_feature_column_info(
    sparse_configs: List[SparseFeat],
    dense_configs: List[DenseFeat]
) -> Dict[str, Any]:
    """Create feature column information for model initialization.
    
    This helper function aggregates feature configuration into a format
    suitable for model construction.
    
    Args:
        sparse_configs: List of sparse feature configurations.
        dense_configs: List of dense feature configurations.
    
    Returns:
        Dictionary with feature metadata.
    """
    sparse_info = []
    total_embed_dim = 0
    
    for config in sparse_configs:
        sparse_info.append({
            'name': config.name,
            'vocabulary_size': config.vocabulary_size,
            'embedding_dim': config.embedding_dim,
        })
        total_embed_dim += config.embedding_dim
    
    dense_names = [c.name for c in dense_configs]
    num_dense = len(dense_configs)
    
    return {
        'sparse_features': sparse_info,
        'dense_features': dense_names,
        'num_sparse': len(sparse_configs),
        'num_dense': num_dense,
        'total_embed_dim': total_embed_dim,
        'total_input_dim': total_embed_dim + num_dense,
    }


if __name__ == "__main__":
    # Demo: generate synthetic data and create dataloaders
    dataset, sparse_configs, dense_configs = generate_synthetic_data(
        num_samples=10000,
        num_users=1000,
        num_items=500,
        click_rate=0.1
    )
    
    print("\nFeature configurations:")
    print("Sparse features:")
    for sf in sparse_configs:
        print(f"  {sf.name}: vocab_size={sf.vocabulary_size}, embed_dim={sf.embedding_dim}")
    
    print("Dense features:")
    for df in dense_configs:
        print(f"  {df.name}")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, batch_size=128
    )
    
    # Show a sample batch
    print("\nSample batch:")
    for sparse_batch, dense_batch, label_batch in train_loader:
        print(f"  Sparse shape: {sparse_batch.shape}")
        print(f"  Dense shape: {dense_batch.shape}")
        print(f"  Label shape: {label_batch.shape}")
        print(f"  Click rate in batch: {label_batch.mean():.4f}")
        break
