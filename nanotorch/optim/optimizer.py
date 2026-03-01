"""
Base Optimizer class.

This module provides the base Optimizer class similar to PyTorch's optim.Optimizer.
"""

from typing import List, Iterator, Dict, Any
import nanotorch.tensor as T


class Optimizer:
    """Base class for all optimizers.

    This class provides the basic functionality for optimization algorithms.
    Subclasses should implement the `step` method.

    Attributes:
        params: List of parameters to optimize.
        defaults: Dictionary of default hyperparameter values.
    """

    def __init__(self, params: Iterator[T.Tensor], defaults: Dict[str, Any]) -> None:
        """Initialize the optimizer.

        Args:
            params: Iterator of parameters to optimize.
            defaults: Dictionary of default hyperparameter values.
        """
        self.param_groups: List[Dict[str, Any]] = []
        self.defaults = defaults

        # Convert params to list if it's an iterator
        param_list = list(params)

        if len(param_list) == 0:
            raise ValueError("Optimizer got an empty parameter list")

        # Create a parameter group
        self.add_param_group({"params": param_list})

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Add a parameter group to the optimizer.

        Args:
            param_group: Dictionary containing the parameter group.
                Must contain 'params' key with parameter list.
        """
        if not isinstance(param_group, dict):
            raise TypeError("param_group must be a dict")

        params = param_group.get("params")
        if params is None:
            raise ValueError("param_group must contain 'params' key")

        # Convert params to list if needed
        if not isinstance(params, (list, tuple)):
            params = [params]

        # Validate that all params are Tensors with requires_grad=True
        for param in params:
            if not isinstance(param, T.Tensor):
                raise TypeError("optimizer can only optimize Tensors")
            if not param.requires_grad:
                raise ValueError("optimized parameters must have requires_grad=True")

        # Set defaults for missing hyperparameters
        param_group = param_group.copy()
        for key, default in self.defaults.items():
            param_group.setdefault(key, default)

        # Store parameters
        param_group["params"] = list(params)
        self.param_groups.append(param_group)

    def zero_grad(self) -> None:
        """Clear gradients of all optimized parameters."""
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.zero_grad()

    def step(self) -> None:
        """Perform a single optimization step.

        Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"Optimizer {self.__class__.__name__} must implement step method"
        )

    def __repr__(self) -> str:
        """String representation of the optimizer."""
        format_string = self.__class__.__name__ + " ("
        for i, group in enumerate(self.param_groups):
            format_string += f"\nParameter Group {i}"
            for key in sorted(group.keys()):
                if key != "params":
                    format_string += f"\n    {key}: {group[key]}"
            format_string += f"\n    params: {len(group['params'])} tensors"
        format_string += "\n)"
        return format_string
