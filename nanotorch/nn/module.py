"""
Base Module class for neural network modules.

This module provides the base Module class similar to PyTorch's nn.Module.
"""

from typing import Dict, Iterator, Tuple, Any, Union
from numpy.typing import NDArray
from collections import OrderedDict
import numpy as np
import nanotorch.tensor as T
from nanotorch.device import Device


class Module:
    """Base class for all neural network modules.

    This class provides functionality for parameter management, gradient
    computation, and module hierarchy similar to PyTorch's nn.Module.

    Attributes:
        training: Whether the module is in training mode.
        _modules: Dictionary of child modules.
        _parameters: Dictionary of module parameters.
    """

    def __init__(self) -> None:
        """Initialize the module."""
        self.training = True
        self._modules: Dict[str, Module] = OrderedDict()
        self._parameters: Dict[str, T.Tensor] = OrderedDict()
        self._buffers: Dict[str, T.Tensor] = OrderedDict()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the module.

        Subclasses must implement this method.

        Returns:
            Output tensor or tuple of tensors.
        """
        raise NotImplementedError(
            f"Module {self.__class__.__name__} must implement forward method"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the module's forward method.

        This enables module instances to be called like functions.

        Returns:
            Output tensor from forward pass.
        """
        return self.forward(*args, **kwargs)

    def register_parameter(self, name: str, param: T.Tensor) -> None:
        """Register a parameter with the module.

        Args:
            name: Name of the parameter.
            param: Parameter tensor.
        """
        if not isinstance(param, T.Tensor):
            raise TypeError(f"Parameter '{name}' must be a Tensor, got {type(param)}")

        self._parameters[name] = param

    def register_module(self, name: str, module: "Module") -> None:
        """Register a child module.

        Args:
            name: Name of the module.
            module: Module instance.
        """
        if not isinstance(module, Module):
            raise TypeError(
                f"Module '{name}' must be a Module instance, got {type(module)}"
            )

        self._modules[name] = module

    def register_buffer(self, name: str, buffer: T.Tensor) -> None:
        """Register a buffer with the module.

        Buffers are tensors that are not considered model parameters but are
        part of the module's state (e.g., running statistics in BatchNorm).

        Args:
            name: Name of the buffer.
            buffer: Buffer tensor.
        """
        if not isinstance(buffer, T.Tensor):
            raise TypeError(f"Buffer '{name}' must be a Tensor, got {type(buffer)}")

        self._buffers[name] = buffer
        super().__setattr__(name, buffer)

    def buffers(self, recurse: bool = True) -> Iterator[T.Tensor]:
        """Return an iterator over module buffers.

        Args:
            recurse: If True, recursively yield buffers of submodules.

        Yields:
            Buffer tensors.
        """
        for buffer in self._buffers.values():
            yield buffer

        if recurse:
            for module in self._modules.values():
                yield from module.buffers(recurse=True)

    def parameters(self, recurse: bool = True) -> Iterator[T.Tensor]:
        """Return an iterator over module parameters.

        Args:
            recurse: If True, recursively yield parameters of submodules.

        Yields:
            Parameter tensors.
        """
        for param in self._parameters.values():
            yield param

        if recurse:
            for module in self._modules.values():
                yield from module.parameters(recurse=True)

    def modules(self, recurse: bool = True) -> Iterator["Module"]:
        """Return an iterator over child modules.

        Args:
            recurse: If True, recursively yield modules and their children.

        Yields:
            Module instances.
        """
        yield self

        if recurse:
            for module in self._modules.values():
                yield from module.modules(recurse=True)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, T.Tensor]]:
        """Return an iterator over module parameters with names.

        Args:
            prefix: Prefix to prepend to parameter names.
            recurse: If True, recursively yield parameters of submodules.

        Yields:
            Tuple of (parameter_name, parameter_tensor).
        """
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param

        if recurse:
            for module_name, module in self._modules.items():
                module_prefix = f"{prefix}.{module_name}" if prefix else module_name
                yield from module.named_parameters(prefix=module_prefix, recurse=True)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, T.Tensor]]:
        """Return an iterator over module buffers with names.

        Args:
            prefix: Prefix to prepend to buffer names.
            recurse: If True, recursively yield buffers of submodules.

        Yields:
            Tuple of (buffer_name, buffer_tensor).
        """
        for name, buffer in self._buffers.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, buffer

        if recurse:
            for module_name, module in self._modules.items():
                module_prefix = f"{prefix}.{module_name}" if prefix else module_name
                yield from module.named_buffers(prefix=module_prefix, recurse=True)

    def state_dict(self) -> Dict[str, NDArray[Any]]:
        """Return a dictionary containing module state.

        The state dict contains all parameters and buffers.

        Returns:
            Dictionary mapping parameter/buffer names to numpy arrays.
        """
        state_dict: Dict[str, NDArray[Any]] = {}
        for name, param in self.named_parameters():
            state_dict[name] = param.data.copy()
        for name, buffer in self.named_buffers():
            state_dict[name] = buffer.data.copy()
        return state_dict

    def load_state_dict(
        self, state_dict: Dict[str, NDArray[Any]], strict: bool = True
    ) -> None:
        """Load state dictionary into module.

        Args:
            state_dict: Dictionary mapping parameter/buffer names to numpy arrays.
            strict: If True, raise an error if keys don't match exactly.

        Raises:
            RuntimeError: If strict=True and keys don't match.
        """
        current_params = dict(self.named_parameters())
        current_buffers = dict(self.named_buffers())
        current_keys = set(current_params.keys()) | set(current_buffers.keys())
        loaded_keys = set(state_dict.keys())

        missing_keys = current_keys - loaded_keys
        unexpected_keys = loaded_keys - current_keys

        if strict:
            if missing_keys:
                raise RuntimeError(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                raise RuntimeError(f"Unexpected keys in state_dict: {unexpected_keys}")
        else:
            if missing_keys:
                print(f"Warning: Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")

        for name, param in current_params.items():
            if name in state_dict:
                param.data = state_dict[name].copy()

        for name, buffer in current_buffers.items():
            if name in state_dict:
                buffer.data = state_dict[name].copy()

    def zero_grad(self) -> None:
        """Zero gradients of all parameters."""
        for param in self.parameters():
            param.zero_grad()

    def train(self, mode: bool = True) -> "Module":
        """Set the module in training mode.

        Args:
            mode: If True, set to training mode; if False, set to evaluation mode.

        Returns:
            Self for method chaining.
        """
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        """Set the module in evaluation mode.

        Returns:
            Self for method chaining.
        """
        return self.train(False)

    def to(self, device: Union[Device, str]) -> "Module":
        """Move all parameters and buffers to the specified device.

        Args:
            device: Target device ('cpu', 'cuda', 'cuda:0', or Device instance).

        Returns:
            Self for method chaining.
        """
        # Move parameters
        for param in self.parameters():
            if param.device != device:
                param_moved = param.to(device)
                # Update parameter in place by modifying its data
                param.data = param_moved.data
                param._device = param_moved._device

        # Move buffers
        for buffer in self.buffers():
            if buffer.device != device:
                buffer_moved = buffer.to(device)
                buffer.data = buffer_moved.data
                buffer._device = buffer_moved._device

        return self

    def cuda(self, device: Union[int, str] = 0) -> "Module":
        """Move all parameters and buffers to CUDA device.

        Args:
            device: CUDA device index or string like 'cuda:0'.

        Returns:
            Self for method chaining.
        """
        if isinstance(device, int):
            target = Device('cuda', device)
        else:
            target = Device.from_string(device)
        return self.to(target)

    def cpu(self) -> "Module":
        """Move all parameters and buffers to CPU.

        Returns:
            Self for method chaining.
        """
        return self.to('cpu')

    def __setattr__(self, name: str, value: object) -> None:
        """Set attribute, handling special cases for parameters and modules."""
        if isinstance(value, T.Tensor):
            # Register as parameter if it has requires_grad=True
            if value.requires_grad:
                self.register_parameter(name, value)
        elif isinstance(value, Module):
            # Register as module
            self.register_module(name, value)

        # Call parent's __setattr__
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        """String representation of the module."""
        lines = [f"{self.__class__.__name__}("]
        child_lines = []

        # Add parameters
        for name, param in self._parameters.items():
            param_repr = (
                f"  ({name}): Tensor(shape={param.shape}, "
                f"requires_grad={param.requires_grad})"
            )
            child_lines.append(param_repr)

        # Add modules
        for name, module in self._modules.items():
            module_repr = repr(module)
            module_lines = module_repr.split("\n")
            child_lines.append(f"  ({name}): {module_lines[0]}")
            for line in module_lines[1:]:
                child_lines.append(f"    {line}")

        if child_lines:
            lines.extend(child_lines)
        lines.append(")")

        return "\n".join(lines)


class Sequential(Module):
    """A sequential container for stacking modules.

    Modules will be added to the container in the order they are passed
    to the constructor. The forward pass sequentially applies each module.

    Example:
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 1)
        )
    """

    def __init__(self, *modules: Module) -> None:
        """Initialize Sequential container with modules.

        Args:
            *modules: Module instances to add sequentially.
        """
        super().__init__()
        for idx, module in enumerate(modules):
            self.register_module(str(idx), module)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Apply modules sequentially to the input.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying all modules.
        """
        for module in self._modules.values():
            x = module(x)
        return x

    def __repr__(self) -> str:
        """String representation of Sequential container."""
        modules_str = ",\n  ".join(repr(module) for module in self._modules.values())
        return f"Sequential(\n  {modules_str}\n)"
