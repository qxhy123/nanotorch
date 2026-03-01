"""
Optimizers for neural network training.

This module provides optimization algorithms similar to PyTorch's optim module.
"""

from nanotorch.optim.optimizer import Optimizer
from nanotorch.optim.sgd import SGD
from nanotorch.optim.adam import Adam
from nanotorch.optim.rmsprop import RMSprop
from nanotorch.optim.adagrad import Adagrad
from nanotorch.optim.adamw import AdamW
from nanotorch.optim.lr_scheduler import (
    LRScheduler,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    ConstantLR,
    LinearWarmup,
    WarmupScheduler,
    CosineWarmupScheduler,
)

__all__ = [
    "Optimizer",
    "SGD",
    "Adam",
    "AdamW",
    "RMSprop",
    "Adagrad",
    "LRScheduler",
    "StepLR",
    "MultiStepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "LinearLR",
    "ReduceLROnPlateau",
    "ConstantLR",
    "LinearWarmup",
    "WarmupScheduler",
    "CosineWarmupScheduler",
]
