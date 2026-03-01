"""
Recommendation system examples using nanotorch.

This package demonstrates production-quality recommendation models:
- DeepFM: Factorization Machine + DNN for CTR prediction
- Wide & Deep: Linear model + DNN
- NeuralCF: Neural Collaborative Filtering
- Two-Tower: User and item encoders for retrieval

Modules:
    data: Synthetic data generation and dataloaders
    train: Training loops, early stopping, and utilities
    evaluate: Evaluation metrics and model comparison
    recommender_demo: Complete end-to-end demo

Usage:
    from examples.recommendation import data, train, evaluate
    from examples.recommendation.recommender_demo import main
"""

from .data import (
    generate_synthetic_data,
    create_dataloaders,
    RecommendationDataset,
    SparseFeat,
    DenseFeat,
    NegativeSampler,
)

__all__ = [
    'generate_synthetic_data',
    'create_dataloaders',
    'RecommendationDataset',
    'SparseFeat',
    'DenseFeat',
    'NegativeSampler',
]
