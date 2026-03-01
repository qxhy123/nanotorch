"""
Learning rate schedulers for neural network training.

This module provides learning rate scheduling strategies similar to PyTorch's optim.lr_scheduler.
"""

import math
from typing import List, Optional, Union
from nanotorch.optim.optimizer import Optimizer


class LRScheduler:
    """Base class for learning rate schedulers.

    Learning rate schedulers adjust the learning rate during training
    based on the epoch number or other metrics.

    Args:
        optimizer: Wrapped optimizer.
        last_epoch: The index of the last epoch (default: -1).

    Example:
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        ...     train(...)
        ...     validate(...)
        ...     scheduler.step()
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs: List[float] = []
        
        for group in optimizer.param_groups:
            self.base_lrs.append(group["lr"])
        
        self.step()

    def get_lr(self) -> List[float]:
        """Compute learning rate for each parameter group.

        Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_lr method"
        )

    def step(self, metrics: Optional[float] = None) -> None:
        """Step the scheduler.

        Args:
            metrics: Optional metric value for ReduceLROnPlateau.
        """
        self.last_epoch += 1
        new_lrs = self.get_lr()
        
        for i, group in enumerate(self.optimizer.param_groups):
            group["lr"] = new_lrs[i]

    def state_dict(self) -> dict:
        """Return the state of the scheduler as a dict."""
        return {
            "last_epoch": self.last_epoch,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the scheduler state."""
        self.last_epoch = state_dict["last_epoch"]
        self.base_lrs = state_dict["base_lrs"]


class StepLR(LRScheduler):
    """Decays learning rate by gamma every step_size epochs.

    Args:
        optimizer: Wrapped optimizer.
        step_size: Period of learning rate decay.
        gamma: Multiplicative factor of learning rate decay (default: 0.1).
        last_epoch: The index of the last epoch (default: -1).

    Example:
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> # lr = 0.05 for epochs 0-29
        >>> # lr = 0.005 for epochs 30-59
        >>> # lr = 0.0005 for epochs 60-79
    """

    def __init__(
        self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1
    ) -> None:
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute new learning rates."""
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [base_lr * self.gamma for base_lr in self.base_lrs]

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["step_size"] = self.step_size
        state["gamma"] = self.gamma
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.step_size = state_dict["step_size"]
        self.gamma = state_dict["gamma"]


class MultiStepLR(LRScheduler):
    """Decays learning rate by gamma at milestones.

    Args:
        optimizer: Wrapped optimizer.
        milestones: List of epoch indices. Must be increasing.
        gamma: Multiplicative factor of learning rate decay (default: 0.1).
        last_epoch: The index of the last epoch (default: -1).

    Example:
        >>> scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
        >>> # lr = 0.05 for epochs 0-29
        >>> # lr = 0.005 for epochs 30-79
        >>> # lr = 0.0005 for epochs 80-99
    """

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute new learning rates."""
        count = sum(1 for m in self.milestones if self.last_epoch >= m)
        if count == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [base_lr * (self.gamma ** count) for base_lr in self.base_lrs]

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["milestones"] = self.milestones
        state["gamma"] = self.gamma
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.milestones = state_dict["milestones"]
        self.gamma = state_dict["gamma"]


class ExponentialLR(LRScheduler):
    """Decays learning rate by gamma every epoch.

    Args:
        optimizer: Wrapped optimizer.
        gamma: Multiplicative factor of learning rate decay.
        last_epoch: The index of the last epoch (default: -1).

    Example:
        >>> scheduler = ExponentialLR(optimizer, gamma=0.9)
        >>> # lr = 0.05 * 0.9^epoch
    """

    def __init__(
        self, optimizer: Optimizer, gamma: float, last_epoch: int = -1
    ) -> None:
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute new learning rates."""
        return [base_lr * (self.gamma ** self.last_epoch) for base_lr in self.base_lrs]

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["gamma"] = self.gamma
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.gamma = state_dict["gamma"]


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate scheduler.

    Set learning rate using a cosine annealing schedule where eta_min
    is the minimum learning rate and T_max is the period.

    Args:
        optimizer: Wrapped optimizer.
        T_max: Maximum number of iterations (period).
        eta_min: Minimum learning rate (default: 0).
        last_epoch: The index of the last epoch (default: -1).

    Formula:
        lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute new learning rates."""
        if self.last_epoch == 0:
            return self.base_lrs
        
        cosine_factor = (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_factor
            for base_lr in self.base_lrs
        ]

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["T_max"] = self.T_max
        state["eta_min"] = self.eta_min
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.T_max = state_dict["T_max"]
        self.eta_min = state_dict["eta_min"]


class LinearLR(LRScheduler):
    """Linear learning rate scheduler.

    Decays the learning rate linearly from start_factor to end_factor
    over total_iters iterations.

    Args:
        optimizer: Wrapped optimizer.
        start_factor: Multiplicative factor at the start (default: 1/3).
        end_factor: Multiplicative factor at the end (default: 1.0).
        total_iters: Number of iterations to reach end_factor.
        last_epoch: The index of the last epoch (default: -1).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 0.333,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
    ) -> None:
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute new learning rates."""
        if self.last_epoch >= self.total_iters:
            return [base_lr * self.end_factor for base_lr in self.base_lrs]
        
        factor = self.start_factor + (self.end_factor - self.start_factor) * self.last_epoch / self.total_iters
        return [base_lr * factor for base_lr in self.base_lrs]

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["start_factor"] = self.start_factor
        state["end_factor"] = self.end_factor
        state["total_iters"] = self.total_iters
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.start_factor = state_dict["start_factor"]
        self.end_factor = state_dict["end_factor"]
        self.total_iters = state_dict["total_iters"]


class ReduceLROnPlateau:
    """Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer: Wrapped optimizer.
        mode: 'min' or 'max'. In 'min' mode, lr will be reduced when
            the quantity monitored has stopped decreasing. In 'max' mode,
            it will be reduced when the quantity monitored has stopped
            increasing (default: 'min').
        factor: Factor by which the learning rate will be reduced (default: 0.1).
        patience: Number of epochs with no improvement after which learning
            rate will be reduced (default: 10).
        threshold: Threshold for measuring the new optimum (default: 1e-4).
        threshold_mode: 'rel' or 'abs'. In 'rel' mode, dynamic_threshold =
            best * (1 + threshold) in 'max' mode or best * (1 - threshold)
            in 'min' mode. In 'abs' mode, dynamic_threshold = best + threshold
            in 'max' mode or best - threshold in 'min' mode (default: 'rel').
        cooldown: Number of epochs to wait before resuming normal operation
            after lr has been reduced (default: 0).
        min_lr: A scalar or a list of scalars. A lower bound on the learning
            rate of all param groups or each group respectively (default: 0).
        eps: Minimal decay applied to lr. If the difference between new and
            old lr is smaller than eps, the update is ignored (default: 1e-8).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: Union[float, List[float]] = 0.0,
        eps: float = 1e-8,
    ) -> None:
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.eps = eps
        
        if isinstance(min_lr, (list, tuple)):
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)
        
        self.best: Optional[float] = None
        self.num_bad_epochs: int = 0
        self.cooldown_counter: int = 0
        
        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        if threshold_mode not in ["rel", "abs"]:
            raise ValueError(f"threshold_mode must be 'rel' or 'abs', got {threshold_mode}")
        if factor >= 1.0:
            raise ValueError(f"factor should be < 1.0, got {factor}")
        if patience < 0:
            raise ValueError(f"patience should be >= 0, got {patience}")

    def step(self, metrics: float) -> None:
        """Step the scheduler based on metrics.

        Args:
            metrics: The metric to monitor (e.g., validation loss).
        """
        current = float(metrics)
        
        if self.best is None:
            self.best = current
            return
        
        if self._is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        
        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == "min":
            if self.threshold_mode == "rel":
                return current < best * (1 - self.threshold)
            else:
                return current < best - self.threshold
        else:
            if self.threshold_mode == "rel":
                return current > best * (1 + self.threshold)
            else:
                return current > best + self.threshold

    def _reduce_lr(self) -> None:
        """Reduce learning rate by factor."""
        for i, group in enumerate(self.optimizer.param_groups):
            old_lr = group["lr"]
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                group["lr"] = new_lr

    def state_dict(self) -> dict:
        """Return the state of the scheduler as a dict."""
        return {
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
            "cooldown_counter": self.cooldown_counter,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the scheduler state."""
        self.best = state_dict["best"]
        self.num_bad_epochs = state_dict["num_bad_epochs"]
        self.cooldown_counter = state_dict["cooldown_counter"]


class ConstantLR(LRScheduler):
    """Constant learning rate scheduler.

    Keeps learning rate constant for total_iters iterations, then
    optionally decays by a factor.

    Args:
        optimizer: Wrapped optimizer.
        factor: Multiplicative factor to decay by (default: 1.0).
        total_iters: Number of iterations before decay (default: 5).
        last_epoch: The index of the last epoch (default: -1).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
    ) -> None:
        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute new learning rates."""
        if self.last_epoch >= self.total_iters:
            return [base_lr * self.factor for base_lr in self.base_lrs]
        return self.base_lrs

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["factor"] = self.factor
        state["total_iters"] = self.total_iters
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.factor = state_dict["factor"]
        self.total_iters = state_dict["total_iters"]


class LinearWarmup(LRScheduler):
    """Linear warmup scheduler.

    Linearly increases learning rate from 0 to base_lr over warmup_epochs.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of epochs for warmup.
        last_epoch: The index of the last epoch.

    Example:
        >>> scheduler = LinearWarmup(optimizer, warmup_epochs=5)
        >>> for epoch in range(100):
        ...     train(...)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        return self.base_lrs

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["warmup_epochs"] = self.warmup_epochs
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.warmup_epochs = state_dict["warmup_epochs"]


class WarmupScheduler(LRScheduler):
    """Warmup wrapper for any scheduler.

    Applies linear warmup before using the wrapped scheduler.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of epochs for warmup.
        warmup_start_lr: Starting learning rate for warmup (default: 0).
        after_scheduler: Scheduler to use after warmup (optional).
        last_epoch: The index of the last epoch.

    Example:
        >>> scheduler = WarmupScheduler(
        ...     optimizer,
        ...     warmup_epochs=5,
        ...     after_scheduler=CosineAnnealingLR(optimizer, T_max=95)
        ... )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        warmup_start_lr: float = 0.0,
        after_scheduler: Optional[LRScheduler] = None,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.after_scheduler = after_scheduler
        self.finished_warmup = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]

        if self.after_scheduler is not None:
            if not self.finished_warmup:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished_warmup = True
            return self.after_scheduler.get_lr()

        return self.base_lrs

    def step(self, metrics: Optional[float] = None) -> None:
        if self.last_epoch >= self.warmup_epochs and self.after_scheduler is not None:
            if metrics is not None:
                self.after_scheduler.step(metrics)
            else:
                self.after_scheduler.step()
        super().step(metrics)

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["warmup_epochs"] = self.warmup_epochs
        state["warmup_start_lr"] = self.warmup_start_lr
        state["finished_warmup"] = self.finished_warmup
        if self.after_scheduler is not None:
            state["after_scheduler"] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.warmup_epochs = state_dict["warmup_epochs"]
        self.warmup_start_lr = state_dict["warmup_start_lr"]
        self.finished_warmup = state_dict["finished_warmup"]
        if self.after_scheduler is not None and "after_scheduler" in state_dict:
            self.after_scheduler.load_state_dict(state_dict["after_scheduler"])


class CosineWarmupScheduler(LRScheduler):
    """Cosine annealing with warmup.

    Linearly warms up learning rate, then applies cosine annealing.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of epochs for warmup.
        max_epochs: Total number of training epochs.
        warmup_start_lr: Starting learning rate for warmup (default: 1e-6).
        min_lr: Minimum learning rate after cosine annealing (default: 0).
        last_epoch: The index of the last epoch.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-6,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]

        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.max_epochs - self.warmup_epochs
        )
        return [
            self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["warmup_epochs"] = self.warmup_epochs
        state["max_epochs"] = self.max_epochs
        state["warmup_start_lr"] = self.warmup_start_lr
        state["min_lr"] = self.min_lr
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.warmup_epochs = state_dict["warmup_epochs"]
        self.max_epochs = state_dict["max_epochs"]
        self.warmup_start_lr = state_dict["warmup_start_lr"]
        self.min_lr = state_dict["min_lr"]
