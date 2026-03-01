"""
Neural network modules for nanotorch.

This module provides neural network building blocks similar to PyTorch's nn module.
"""

from nanotorch.nn.module import Module, Sequential
from nanotorch.nn.linear import Linear
from nanotorch.nn.activation import ReLU, Sigmoid, Tanh, Softmax, LogSoftmax, LeakyReLU, ELU, GELU, SiLU, PReLU, Softplus, Hardswish, Hardsigmoid, Flatten, Identity
from nanotorch.nn.loss import (
    MSE, CrossEntropyLoss, L1Loss, SmoothL1Loss, BCELoss, BCEWithLogitsLoss, NLLLoss,
    mse_loss, cross_entropy_loss, l1_loss, smooth_l1_loss, bce_loss, bce_with_logits_loss, nll_loss
)
from nanotorch.nn.batchnorm import BatchNorm2d
from nanotorch.nn.dropout import Dropout
from nanotorch.nn.conv import Conv1D, Conv2D, ConvTranspose2D, Conv3D, ConvTranspose3D
from nanotorch.nn.pooling import (
    MaxPool2d, AvgPool2d, max_pool2d, avg_pool2d, 
    AdaptiveAvgPool2d, AdaptiveMaxPool2d, adaptive_avg_pool2d, adaptive_max_pool2d,
    MaxPool1d, AvgPool1d, MaxPool3d, AvgPool3d
)
from nanotorch.nn.normalization import LayerNorm, layer_norm, GroupNorm, group_norm, InstanceNorm2d, instance_norm, BatchNorm1d, BatchNorm3d, InstanceNorm1d, InstanceNorm3d
from nanotorch.nn.embedding import Embedding, EmbeddingBag
from nanotorch.nn.attention import MultiheadAttention, SelfAttention, scaled_dot_product_attention
from nanotorch.nn.transformer import (
    TransformerEncoderLayer, TransformerDecoderLayer,
    TransformerEncoder, TransformerDecoder, Transformer
)
from nanotorch.nn.rnn import RNNCell, LSTMCell, GRUCell, RNN, LSTM, GRU
from nanotorch.nn.fm import FactorizationMachine, FieldAwareFactorizationMachine, CrossNetwork, Interaction
from nanotorch.nn.metrics import (
    auc_score, log_loss, hit_at_k, recall_at_k, precision_at_k,
    dcg_at_k, ndcg_at_k, mrr, map_score, compute_all_ranking_metrics,
    RecommenderMetrics
)

__all__ = [
    "Module",
    "Sequential",
    "Linear",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "LogSoftmax",
    "LeakyReLU",
    "ELU",
    "GELU",
    "SiLU",
    "PReLU",
    "Softplus",
    "Hardswish",
    "Hardsigmoid",
    "Flatten",
    "Identity",
    "MSE",
    "CrossEntropyLoss",
    "L1Loss",
    "SmoothL1Loss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "NLLLoss",
    "mse_loss",
    "cross_entropy_loss",
    "l1_loss",
    "smooth_l1_loss",
    "bce_loss",
    "bce_with_logits_loss",
    "nll_loss",
    "BatchNorm2d",
    "Dropout",
    "Conv1D",
    "Conv2D",
    "ConvTranspose2D",
    "Conv3D",
    "ConvTranspose3D",
    "MaxPool1d",
    "AvgPool1d",
    "MaxPool2d",
    "AvgPool2d",
    "max_pool2d",
    "avg_pool2d",
    "MaxPool3d",
    "AvgPool3d",
    "AdaptiveAvgPool2d",
    "AdaptiveMaxPool2d",
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
    "LayerNorm",
    "layer_norm",
    "GroupNorm",
    "group_norm",
    "InstanceNorm2d",
    "instance_norm",
    "BatchNorm1d",
    "BatchNorm3d",
    "InstanceNorm1d",
    "InstanceNorm3d",
    "Embedding",
    "EmbeddingBag",
    "MultiheadAttention",
    "SelfAttention",
    "scaled_dot_product_attention",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerDecoder",
    "Transformer",
    "RNNCell",
    "LSTMCell",
    "GRUCell",
    "RNN",
    "LSTM",
    "GRU",
    # Factorization Machine layers
    "FactorizationMachine",
    "FieldAwareFactorizationMachine",
    "CrossNetwork",
    "Interaction",
    # Recommendation metrics
    "auc_score",
    "log_loss",
    "hit_at_k",
    "recall_at_k",
    "precision_at_k",
    "dcg_at_k",
    "ndcg_at_k",
    "mrr",
    "map_score",
    "compute_all_ranking_metrics",
    "RecommenderMetrics",
]
