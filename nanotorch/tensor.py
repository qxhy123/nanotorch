"""
Tensor class for nanotorch - Core tensor operations with autograd support.

This module implements the core Tensor class with automatic differentiation
capabilities, similar to PyTorch's Tensor API.
"""

import numpy as np
from numpy.typing import NDArray
from types import TracebackType
from typing import Optional, Tuple, List, Union, Any, Set, cast, Type


class Tensor:
    """A multi-dimensional array with automatic differentiation support.

    The Tensor class stores data, gradients, and computational graph information
    to enable automatic differentiation (autograd).

    Attributes:
        data: The underlying numpy array storing tensor values.
        requires_grad: Whether to track gradients for this tensor.
        grad: Gradient tensor (same shape as data).
        _op: Operation that created this tensor (for computational graph).
        _parents: Parent tensors in the computational graph.
    """

    __slots__ = ["data", "requires_grad", "grad", "_op", "_parents", "_ctx", "_fn"]
    data: NDArray[np.float32]

    def __init__(
        self,
        data: Union[NDArray[Any], List[Any], int, float],
        requires_grad: bool = False,
        _op: Optional[str] = None,
        _parents: Tuple["Tensor", ...] = (),
        _ctx: Optional[Any] = None,
        _fn: Optional[Any] = None,
    ) -> None:
        """Initialize a Tensor.

        Args:
            data: Input data, can be numpy array, list, or scalar.
            requires_grad: Whether to compute gradients for this tensor.
            _op: Internal use - operation that created this tensor.
            _parents: Internal use - parent tensors in computational graph.
        """
        # Convert input to numpy array with float32 dtype
        if isinstance(data, (int, float)):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, list):
            self.data = np.array(data, dtype=np.float32)
        else:
            self.data = data.astype(np.float32) if data.dtype != np.float32 else data

        # Disable gradient tracking if no_grad context is active
        if not _ENABLE_GRAD:
            requires_grad = False

        self.requires_grad = requires_grad
        self.grad: Optional["Tensor"] = None
        self._op = _op
        self._parents = _parents
        self._ctx = _ctx
        self._fn = _fn

        # Initialize gradient if requires_grad is True
        if requires_grad:
            self.zero_grad()

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the tensor shape."""
        return self.data.shape

    @property
    def size(self) -> int:
        """Return total number of elements in the tensor."""
        return self.data.size

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype[np.float32]:
        """Return the data type."""
        return self.data.dtype

    def __repr__(self) -> str:
        """String representation of the tensor."""
        grad_str = f", grad={self.grad}" if self.grad is not None else ""
        return (
            f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}{grad_str})"
        )

    def __getitem__(self, key) -> "Tensor":
        """Index into the tensor.

        Args:
            key: Index or slice.

        Returns:
            Indexed tensor (view, not copy).
        """
        result_data = self.data[key]
        return Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _op="getitem",
            _parents=(self,),
            _ctx={"key": key},
        )

    @staticmethod
    def _reduce_gradient_numpy(
        grad_contrib_data: NDArray[np.float32], target_shape: Tuple[int, ...]
    ) -> NDArray[np.float32]:
        """Reduce gradient to target shape by summing over broadcast dimensions."""
        if grad_contrib_data.shape == target_shape:
            return grad_contrib_data

        grad_shape = grad_contrib_data.shape
        target_shape_aligned = target_shape

        if len(target_shape) < len(grad_shape):
            target_shape_aligned = (1,) * (len(grad_shape) - len(target_shape)) + target_shape

        axes_to_sum = []
        for i in range(len(grad_shape)):
            if i >= len(target_shape_aligned):
                axes_to_sum.append(i)
            elif target_shape_aligned[i] == 1 and grad_shape[i] > 1:
                axes_to_sum.append(i)

        if axes_to_sum:
            reduced_data = grad_contrib_data.sum(
                axis=tuple(axes_to_sum), keepdims=False, dtype=np.float32
            )
            reduced_data = reduced_data.reshape(target_shape)
            return cast(NDArray[np.float32], reduced_data)
        else:
            return grad_contrib_data.reshape(target_shape)

    @staticmethod
    def _accumulate_grad(
        parent: "Tensor", grad_contrib: Union["Tensor", NDArray[Any]]
    ) -> None:
        if not parent.requires_grad:
            return

        if isinstance(grad_contrib, Tensor):
            grad_contrib_data = grad_contrib.data
        else:
            grad_contrib_data = grad_contrib

        if (
            grad_contrib_data.shape == parent.shape
            and grad_contrib_data.dtype == np.float32
            and parent.grad is not None
        ):
            parent.grad.data += grad_contrib_data
            return

        if grad_contrib_data.shape != parent.shape:
            grad_contrib_data = Tensor._reduce_gradient_numpy(
                grad_contrib_data, parent.shape
            )

        if grad_contrib_data.dtype != np.float32:
            grad_contrib_data = grad_contrib_data.astype(np.float32)

        if parent.grad is None:
            # First gradient contribution: set directly instead of zero + add
            parent.grad = Tensor(grad_contrib_data, requires_grad=False)
        else:
            parent.grad.data += grad_contrib_data

    @staticmethod
    def _accumulate_grad_batch(
        accumulations: List[Tuple["Tensor", Union["Tensor", NDArray[Any]]]]
    ) -> None:
        """Batch gradient accumulation for multiple parent-gradient pairs.
        
        More efficient than calling _accumulate_grad multiple times by
        batching common operations and reducing Python overhead.
        
        Args:
            accumulations: List of (parent, gradient) pairs to accumulate.
        """
        if not accumulations:
            return
        
        for parent, grad_contrib in accumulations:
            if not parent.requires_grad:
                continue
            
            if isinstance(grad_contrib, Tensor):
                grad_contrib_data = grad_contrib.data
            else:
                grad_contrib_data = grad_contrib
            
            if (
                grad_contrib_data.shape == parent.shape
                and grad_contrib_data.dtype == np.float32
                and parent.grad is not None
            ):
                parent.grad.data += grad_contrib_data
                continue
            
            if grad_contrib_data.shape != parent.shape:
                grad_contrib_data = Tensor._reduce_gradient_numpy(
                    grad_contrib_data, parent.shape
                )
            
            if grad_contrib_data.dtype != np.float32:
                grad_contrib_data = grad_contrib_data.astype(np.float32)
            
            if parent.grad is None:
                # First gradient contribution: set directly instead of zero + add
                parent.grad = Tensor(grad_contrib_data, requires_grad=False)
            else:
                parent.grad.data += grad_contrib_data

    def zero_grad(self) -> None:
        """Reset gradient to zero."""
        if self.grad is None:
            self.grad = Tensor(np.zeros_like(self.data), requires_grad=False)
        else:
            self.grad.data.fill(0)

    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        """Perform backward pass through the computational graph.

        Computes gradients for all tensors in the computational graph using
        the chain rule.

        Args:
            gradient: Gradient to propagate (default is ones tensor).
        """
        # Build computational graph (topological order)
        topo: List[Tensor] = []
        visited: Set[Tensor] = set()

        def build_topo(v: Tensor) -> None:
            """Build topological order using DFS."""
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    if isinstance(parent, Tensor):
                        build_topo(parent)
                topo.append(v)

        build_topo(self)

        # Set initial gradient
        if gradient is None:
            gradient = Tensor(np.ones_like(self.data), requires_grad=False)

        if self.grad is None:
            self.grad = gradient
        else:
            self.grad.data += gradient.data

        # Propagate gradients backwards
        for v in reversed(topo):
            if v._op is not None and v.requires_grad and v.grad is not None:
                # Dispatch to operation-specific backward pass
                self._backward_operation(v, v._op, v._parents)
            elif v._fn is not None and v.requires_grad and v.grad is not None:
                ctx = v._ctx
                if ctx is not None:
                    grad_inputs = v._fn.backward(ctx, v.grad)

                    if v._parents:
                        if isinstance(grad_inputs, tuple):
                            for parent, grad in zip(v._parents, grad_inputs):
                                if (
                                    isinstance(parent, Tensor)
                                    and parent.requires_grad
                                    and grad is not None
                                ):
                                    if parent.grad is None:
                                        parent.grad = grad
                                    else:
                                        parent.grad.data += grad.data
                        elif isinstance(grad_inputs, Tensor) and len(v._parents) == 1:
                            parent = v._parents[0]
                            if isinstance(parent, Tensor) and parent.requires_grad:
                                if parent.grad is None:
                                    parent.grad = grad_inputs
                                else:
                                    parent.grad.data += grad_inputs.data

    def _backward_operation(
        self, tensor: "Tensor", op: str, parents: Tuple["Tensor", ...]
    ) -> None:
        """Dispatch backward pass to operation-specific implementation.

        Args:
            tensor: The tensor to compute gradients for.
            op: Operation identifier.
            parents: Parent tensors of the operation.
        """
        # This is a simplified implementation - in a full implementation,
        # we would have proper operation classes with backward methods.
        if tensor.grad is None:
            return



        if op == "add":
            if len(parents) == 2:
                a, b = parents
                if a.size <= 100 and b.size <= 100:
                    Tensor._accumulate_grad(a, tensor.grad.data)
                    Tensor._accumulate_grad(b, tensor.grad.data)
                else:
                    Tensor._accumulate_grad_batch([(a, tensor.grad.data), (b, tensor.grad.data)])
        elif op == "mul":
            if len(parents) == 2:
                a, b = parents
                grad_a_data = tensor.grad.data * b.data
                grad_b_data = tensor.grad.data * a.data
                if a.size <= 100 and b.size <= 100:
                    Tensor._accumulate_grad(a, grad_a_data)
                    Tensor._accumulate_grad(b, grad_b_data)
                else:
                    Tensor._accumulate_grad_batch([(a, grad_a_data), (b, grad_b_data)])
        elif op == "matmul":
            if len(parents) == 2:
                a, b = parents
                # Use cached transposes if available (stored during forward pass)
                ctx = tensor._ctx
                if ctx is not None:
                    b_T = ctx.get("b_T")
                    a_T = ctx.get("a_T")
                else:
                    b_T = b.data.T
                    a_T = a.data.T
                grad_a_data = tensor.grad.data @ b_T
                grad_b_data = a_T @ tensor.grad.data
                if a.size <= 100 and b.size <= 100:
                    Tensor._accumulate_grad(a, grad_a_data)
                    Tensor._accumulate_grad(b, grad_b_data)
                else:
                    Tensor._accumulate_grad_batch([(a, grad_a_data), (b, grad_b_data)])
        elif op == "neg":
            # Gradient for negation: dL/da = -dL/dout
            if len(parents) == 1:
                a = parents[0]
                self._accumulate_grad(a, -tensor.grad.data)
        elif op == "sum":
            # Gradient for sum: dL/da = broadcast(dL/dout) to input shape
            if len(parents) == 1 and tensor.grad is not None:
                a = parents[0]
                ctx = tensor._ctx
                if ctx is None:
                    # Default: full reduction (axis=None)
                    axis = None
                    keepdims = False
                else:
                    axis = ctx.get("axis")
                    keepdims = ctx.get("keepdims", False)

                grad_data = tensor.grad.data
                # If keepdims is False and axis is not None, we need to unsqueeze
                # the reduced dimensions
                if axis is not None and not keepdims:
                    # Convert axis to tuple
                    if isinstance(axis, int):
                        axis = (axis,)
                    # Insert dimensions of size 1 at the reduced axes
                    new_shape = list(grad_data.shape)
                    for ax in sorted(axis):
                        # Insert at position ax (but handle negative axis)
                        if ax < 0:
                            ax = (
                                grad_data.ndim + ax + 1
                            )  # +1 because we're inserting before?
                        new_shape.insert(ax, 1)
                    grad_data = grad_data.reshape(new_shape)

                # Broadcast gradient to input shape
                # numpy broadcasting will handle this automatically
                grad_contrib_data = np.broadcast_to(grad_data, a.shape)
                self._accumulate_grad(a, grad_contrib_data)
        elif op == "mean":
            # Gradient for mean: dL/da = broadcast(dL/dout / n) where n is
            # number of reduced elements
            if len(parents) == 1 and tensor.grad is not None:
                a = parents[0]
                ctx = tensor._ctx
                if ctx is None:
                    axis = None
                    keepdims = False
                else:
                    axis = ctx.get("axis")
                    keepdims = ctx.get("keepdims", False)

                # Compute number of reduced elements
                if axis is None:
                    # Reduce over all axes
                    n = a.data.size
                else:
                    # Convert axis to tuple
                    if isinstance(axis, int):
                        axis = (axis,)
                    # Compute product of dimensions being reduced
                    n = 1
                    for ax in axis:
                        if ax < 0:
                            ax = a.data.ndim + ax
                        n *= a.data.shape[ax]

                grad_data = tensor.grad.data / n
                # Handle keepdims=False similarly to sum
                if axis is not None and not keepdims:
                    if isinstance(axis, int):
                        axis = (axis,)
                    new_shape = list(grad_data.shape)
                    for ax in sorted(axis):
                        if ax < 0:
                            ax = grad_data.ndim + ax + 1
                        new_shape.insert(ax, 1)
                    grad_data = grad_data.reshape(new_shape)

                grad_contrib_data = np.broadcast_to(grad_data, a.shape)
                self._accumulate_grad(a, grad_contrib_data)
        elif op == "prod":
            # Gradient for product: dL/da = dL/dout * prod(a) / a
            # (with special handling for zeros)
            if len(parents) == 1 and tensor.grad is not None:
                a = parents[0]
                ctx = tensor._ctx
                if ctx is None:
                    axis = None
                    keepdims = False
                else:
                    axis = ctx.get("axis")
                    keepdims = ctx.get("keepdims", False)

                # Compute product along axis (same as forward pass)
                prod_result = tensor.data  # This is the product result
                grad_output = tensor.grad.data

                # We need to broadcast grad_output to input shape
                # First handle keepdims=False similarly to sum
                grad_data = grad_output
                if axis is not None and not keepdims:
                    if isinstance(axis, int):
                        axis = (axis,)
                    new_shape = list(grad_data.shape)
                    for ax in sorted(axis):
                        if ax < 0:
                            ax = grad_data.ndim + ax + 1
                        new_shape.insert(ax, 1)
                    grad_data = grad_data.reshape(new_shape)

                # Broadcast grad_data to input shape
                grad_broadcast = np.broadcast_to(grad_data, a.shape)

                # Compute gradient using product rule
                # For each element: gradient = grad_broadcast * (prod_result / element)
                # Need to handle division by zero
                a_data = a.data
                epsilon = 1e-12
                # Compute product of all elements along reduced axes
                # We already have prod_result, but need to broadcast it to input shape
                # First, reshape prod_result to have shape with reduced dimensions
                # as size 1
                prod_expanded = prod_result
                if axis is not None:
                    # Create shape with ones at reduced dimensions
                    shape = list(a.shape)
                    if isinstance(axis, int):
                        axis = (axis,)
                    for ax in axis:
                        if ax < 0:
                            ax = a.ndim + ax
                        shape[ax] = 1
                    prod_expanded = prod_result.reshape(shape)

                # Broadcast prod_expanded to input shape
                prod_broadcast = np.broadcast_to(prod_expanded, a.shape)

                # Compute gradient = grad_broadcast * prod_broadcast /
                # (a_data + epsilon)
                # But need special handling for zeros: when a_data == 0,
                # gradient = grad_broadcast * product of other elements
                # For simplicity, we'll use epsilon to avoid division by zero
                # This approximation may cause small errors when a_data is exactly zero
                grad_contrib_data = grad_broadcast * prod_broadcast / (a_data + epsilon)
                self._accumulate_grad(a, grad_contrib_data)
        elif op == "div":
            if len(parents) == 2 and tensor.grad is not None:
                a, b = parents
                
                ctx = tensor._ctx
                if ctx is not None and "denom" in ctx:
                    denom = ctx["denom"]
                else:
                    denom = b.data
                
                denom_sq = denom * denom
                grad_a_data = tensor.grad.data / denom
                grad_b_data = -tensor.grad.data * a.data / denom_sq
                if a.size <= 100 and b.size <= 100:
                    Tensor._accumulate_grad(a, grad_a_data)
                    Tensor._accumulate_grad(b, grad_b_data)
                else:
                    Tensor._accumulate_grad_batch([(a, grad_a_data), (b, grad_b_data)])
        elif op == "pow":
            # Gradient for power: y = a ** exponent
            # dy/da = exponent * a ** (exponent - 1)
            if len(parents) == 1 and tensor.grad is not None:
                a = parents[0]
                ctx = tensor._ctx
                exponent = ctx.get("exponent") if ctx else None
                if exponent is not None:
                    # a_data = a.data (numpy array)
                    grad = tensor.grad.data * exponent * (a.data ** (exponent - 1))
                    self._accumulate_grad(a, grad)
        elif op == "relu":
            # Gradient for ReLU: dy/da = grad if a > 0 else 0
            if len(parents) == 1 and tensor.grad is not None:
                a = parents[0]
                mask: NDArray[np.float32] = (a.data > 0).astype(np.float32)
                grad = tensor.grad.data * mask
                self._accumulate_grad(a, grad)
        elif op == "sigmoid":
            # Gradient for sigmoid: dy/da = grad * sigmoid(a) * (1 - sigmoid(a))
            if len(parents) == 1 and tensor.grad is not None:
                a = parents[0]
                sig = 1 / (1 + np.exp(-np.clip(a.data, -15, 15)))
                grad = tensor.grad.data * sig * (1 - sig)
                self._accumulate_grad(a, grad)
        elif op == "tanh":
            # Gradient for tanh: dy/da = grad * (1 - tanh(a)^2)
            if len(parents) == 1 and tensor.grad is not None:
                a = parents[0]
                tanh_a = np.tanh(a.data)
                grad = tensor.grad.data * (1 - tanh_a * tanh_a)
                self._accumulate_grad(a, grad)
        elif op == "exp":
            # Gradient for exp: dy/da = grad * exp(a)
            if len(parents) == 1 and tensor.grad is not None:
                a = parents[0]
                grad = tensor.grad.data * np.exp(a.data)
                self._accumulate_grad(a, grad)
        elif op == "log":
            # Gradient for log: dy/da = grad / a (matches PyTorch behavior)
            if len(parents) == 1 and tensor.grad is not None:
                a = parents[0]
                grad = tensor.grad.data / a.data
                self._accumulate_grad(a, grad)
        elif op == "reshape":
            # Gradient for reshape: just reshape gradient to input shape
            if len(parents) == 1 and tensor.grad is not None:
                a = parents[0]
                grad = tensor.grad.data.reshape(a.shape)
                self._accumulate_grad(a, grad)
        elif op == "squeeze":
            # Gradient for squeeze: unsqueeze gradient to input shape
            if len(parents) == 1 and tensor.grad is not None:
                a = parents[0]
                # Need to know which axes were squeezed (stored in ctx)
                ctx = tensor._ctx
                if ctx and "axis" in ctx:
                    axis = ctx["axis"]
                    # Insert dimensions of size 1 at the squeezed axes
                    grad_data = tensor.grad.data
                    if axis is not None:
                        if isinstance(axis, int):
                            axis = (axis,)
                        new_shape = list(grad_data.shape)
                        for ax in sorted(axis):
                            if ax < 0:
                                ax = grad_data.ndim + ax + 1
                            new_shape.insert(ax, 1)
                        grad = np.reshape(grad_data, new_shape)
                    else:
                        grad = grad_data
                else:
                    # If no axis info, just match shape (should be same as input)
                    grad = tensor.grad.data.reshape(a.shape)
                self._accumulate_grad(a, grad)
        elif op == "transpose":
            # Gradient for transpose: transpose gradient back
            if len(parents) == 1 and tensor.grad is not None:
                a = parents[0]
                grad = tensor.grad.data.T
                self._accumulate_grad(a, grad)
        elif op == "abs":
            # Gradient for abs: dy/da = grad * sign(a) where sign(0) = 0
            if len(parents) == 1 and tensor.grad is not None:
                a = parents[0]
                sign = np.sign(a.data)
                # sign(0) = 0, but np.sign(0) = 0 already
                grad = tensor.grad.data * sign
                self._accumulate_grad(a, grad)
        elif op == "sqrt":
            # Gradient for sqrt: dy/da = grad / (2 * sqrt(a)) where a > 0
            if len(parents) == 1 and tensor.grad is not None:
                a = parents[0]
                epsilon = 1e-8  # small epsilon to avoid division by zero
                grad = tensor.grad.data / (2 * np.sqrt(np.maximum(a.data, epsilon)))
                self._accumulate_grad(a, grad)
        elif op == "clamp":
            # Gradient for clamp: dy/da = grad where a is within [min, max], else 0
            if len(parents) == 1 and tensor.grad is not None:
                a = parents[0]
                ctx = tensor._ctx
                min_val = ctx.get("min_val") if ctx else None
                max_val = ctx.get("max_val") if ctx else None

                # Create mask for values that contribute to gradient
                mask = np.ones_like(a.data, dtype=np.float32)
                if min_val is not None:
                    mask = mask * (a.data > min_val).astype(np.float32)
                if max_val is not None:
                    mask = mask * (a.data < max_val).astype(np.float32)

                grad = tensor.grad.data * mask
                self._accumulate_grad(a, grad)
        elif op == "softmax":
            # Gradient for softmax: grad_input = s * (grad_output -
            # sum(grad_output * s, dim=dim))
            if len(parents) == 1 and tensor.grad is not None:
                a = parents[0]
                ctx = tensor._ctx
                dim = ctx.get("dim") if ctx else -1
                # Recompute softmax for gradient calculation
                if dim < 0:
                    dim = a.ndim + dim
                shifted = a.data - np.max(a.data, axis=dim, keepdims=True)
                exp = np.exp(shifted)
                s = exp / np.sum(exp, axis=dim, keepdims=True)
                grad_output = tensor.grad.data
                # Compute sum(grad_output * s) along dim
                sum_grad = np.sum(grad_output * s, axis=dim, keepdims=True)
                grad_input = s * (grad_output - sum_grad)
                self._accumulate_grad(a, grad_input)
        elif op == "log_softmax":
            # Gradient for log_softmax: grad_input = grad_output -
            # s * sum(grad_output, dim=dim)
            if len(parents) == 1 and tensor.grad is not None:
                a = parents[0]
                ctx = tensor._ctx
                dim = ctx.get("dim") if ctx else -1
                if dim < 0:
                    dim = a.ndim + dim
                shifted = a.data - np.max(a.data, axis=dim, keepdims=True)
                exp = np.exp(shifted)
                s = exp / np.sum(exp, axis=dim, keepdims=True)
                grad_output = tensor.grad.data
                # Compute sum(grad_output) along dim
                sum_grad = np.sum(grad_output, axis=dim, keepdims=True)
                grad_input = grad_output - s * sum_grad
                self._accumulate_grad(a, grad_input)
        elif op == "permute":
            # Gradient for permute: permute gradient back using inverse permutation
            if len(parents) == 1 and tensor.grad is not None:
                a = parents[0]
                ctx = tensor._ctx
                if ctx and "inverse_perm" in ctx:
                    inverse_perm = ctx["inverse_perm"]
                    # Apply inverse permutation to gradient
                    grad = tensor.grad.data.transpose(inverse_perm)
                    self._accumulate_grad(a, grad)
                else:
                    # Fallback: if no inverse permutation stored, gradient is just transposed back
                    # This shouldn't happen if permute was called correctly
                    grad = tensor.grad.data.T if a.ndim == 2 else tensor.grad.data
                    self._accumulate_grad(a, grad)
        elif op == "split":
            # Gradient for split: place gradient in appropriate slice of parent
            if len(parents) == 1 and tensor.grad is not None:
                a = parents[0]
                ctx = tensor._ctx
                if ctx and "dim" in ctx and "split_index" in ctx and "split_sizes" in ctx:
                    dim = ctx["dim"]
                    split_index = ctx["split_index"]
                    split_sizes = ctx["split_sizes"]
                    
                    # Calculate start and end indices for this slice
                    start_idx = sum(split_sizes[:split_index])
                    end_idx = start_idx + split_sizes[split_index]
                    
                    # Create zero gradient for parent
                    grad_full = np.zeros_like(a.data)
                    
                    # Construct slice indices
                    slice_indices = [slice(None)] * a.ndim
                    slice_indices[dim] = slice(start_idx, end_idx)
                    
                    # Place gradient in appropriate slice
                    grad_full[tuple(slice_indices)] = tensor.grad.data
                    
                    self._accumulate_grad(a, grad_full)
        
        elif op == "gather":
            # Gradient for gather: scatter gradient back to input using indices
            if len(parents) == 2 and tensor.grad is not None:
                input_tensor = parents[0]
                index_tensor = parents[1]
                ctx = tensor._ctx
                
                if ctx and "dim" in ctx:
                    dim = ctx["dim"]
                    # Scatter gradient back to input positions
                    grad_input = np.zeros_like(input_tensor.data)
                    np.put_along_axis(grad_input, index_tensor.data.astype(np.int64), tensor.grad.data, axis=dim)
                    self._accumulate_grad(input_tensor, grad_input)
                    # Index tensor gradient not computed (PyTorch convention)
        
        elif op == "scatter":
            # Gradient for scatter: input gets gradient except at indexed positions,
            # src gets gradient gathered from output at indices
            if len(parents) == 3 and tensor.grad is not None:
                input_tensor = parents[0]
                index_tensor = parents[1]
                src_tensor = parents[2]
                ctx = tensor._ctx
                
                if ctx and "dim" in ctx:
                    dim = ctx["dim"]
                    grad_output = tensor.grad.data
                    
                    # Gradient for input: grad_output with zeros at indexed positions
                    grad_input = grad_output.copy()
                    # Put zeros at indexed positions (overwrites with zero)
                    np.put_along_axis(grad_input, index_tensor.data.astype(np.int64), 0, axis=dim)
                    self._accumulate_grad(input_tensor, grad_input)
                    
                    # Gradient for src: gather gradient from output at indices
                    grad_src = np.take_along_axis(grad_output, index_tensor.data.astype(np.int64), axis=dim)
                    self._accumulate_grad(src_tensor, grad_src)

                    # Index tensor gradient not computed

        elif op == "embedding":
            # Gradient for embedding: scatter gradient back to weight matrix
            if len(parents) == 2 and tensor.grad is not None:
                weight = parents[0]
                input_tensor = parents[1]
                if weight.requires_grad:
                    indices = input_tensor.data.astype(np.int64)
                    grad_weight = np.zeros_like(weight.data)
                    np.add.at(grad_weight, indices, tensor.grad.data)
                    self._accumulate_grad(weight, grad_weight)

        elif op == "embedding_bag":
            # Gradient for embedding_bag: scatter gradient back based on mode
            if len(parents) == 2 and tensor.grad is not None:
                weight = parents[0]
                input_tensor = parents[1]
                if weight.requires_grad:
                    indices = input_tensor.data.astype(np.int64)
                    grad_weight = np.zeros_like(weight.data)
                    ctx = tensor._ctx

                    if input_tensor.data.ndim == 1:
                        # Direct indexing case
                        np.add.at(grad_weight, indices, tensor.grad.data)
                    else:
                        # Bag case - need to scatter gradient back
                        mode = ctx.get("mode", "sum") if ctx else "sum"

                        if mode == "sum":
                            bag_size = indices.shape[1]
                            expanded_grad = np.repeat(tensor.grad.data[:, np.newaxis, :], bag_size, axis=1)
                            np.add.at(grad_weight, indices, expanded_grad)
                        elif mode == "mean":
                            bag_size = indices.shape[1]
                            expanded_grad = np.repeat(tensor.grad.data[:, np.newaxis, :], bag_size, axis=1) / bag_size
                            np.add.at(grad_weight, indices, expanded_grad)
                        elif mode == "max":
                            # Only the max element gets the gradient
                            embedded = weight.data[indices]
                            max_indices = np.argmax(embedded, axis=1)
                            for n in range(indices.shape[0]):
                                for d in range(weight.data.shape[1]):
                                    max_pos = max_indices[n, d]
                                    idx = indices[n, max_pos]
                                    grad_weight[idx, d] += tensor.grad.data[n, d]

                    self._accumulate_grad(weight, grad_weight)

    # Basic arithmetic operations
    def __add__(
        self, other: Union["Tensor", NDArray[np.float32], int, float]
    ) -> "Tensor":
        """Element-wise addition."""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        # Check shape compatibility (support broadcasting)
        try:
            result_data = self.data + other.data
        except ValueError as e:
            raise RuntimeError(
                f"Shape mismatch in addition: {self.shape} vs {other.shape}"
            ) from e

        return Tensor(
            result_data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op="add",
            _parents=(self, other),
        )

    def __radd__(
        self, other: Union["Tensor", NDArray[np.float32], int, float]
    ) -> "Tensor":
        """Reverse addition."""
        return self.__add__(other)

    def __sub__(
        self, other: Union["Tensor", NDArray[np.float32], int, float]
    ) -> "Tensor":
        """Element-wise subtraction."""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        # Negate and add
        return self + (-other)

    def __rsub__(
        self, other: Union["Tensor", NDArray[np.float32], int, float]
    ) -> "Tensor":
        """Reverse subtraction."""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return other - self

    def __neg__(self) -> "Tensor":
        """Element-wise negation."""
        return Tensor(
            -self.data, requires_grad=self.requires_grad, _op="neg", _parents=(self,)
        )

    def __mul__(
        self, other: Union["Tensor", NDArray[np.float32], int, float]
    ) -> "Tensor":
        """Element-wise multiplication."""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        try:
            result_data = self.data * other.data
        except ValueError as e:
            raise RuntimeError(
                f"Shape mismatch in multiplication: {self.shape} vs {other.shape}"
            ) from e

        return Tensor(
            result_data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op="mul",
            _parents=(self, other),
        )

    def __rmul__(
        self, other: Union["Tensor", NDArray[np.float32], int, float]
    ) -> "Tensor":
        """Reverse multiplication."""
        return self.__mul__(other)

    def __truediv__(
        self, other: Union["Tensor", NDArray[np.float32], int, float]
    ) -> "Tensor":
        """Element-wise division."""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        try:
            result_data = self.data / other.data
        except ValueError as e:
            raise RuntimeError(
                f"Shape mismatch in division: {self.shape} vs {other.shape}"
            ) from e

        return Tensor(
            result_data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op="div",
            _parents=(self, other),
            _ctx={
                "denom": other.data,  # Store denominator for gradient reuse
            },
        )

    def __rtruediv__(
        self, other: Union["Tensor", NDArray[np.float32], int, float]
    ) -> "Tensor":
        """Reverse division."""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return other / self

    def __pow__(self, exponent: Union[int, float]) -> "Tensor":
        """Element-wise power."""
        return Tensor(
            self.data**exponent,
            requires_grad=self.requires_grad,
            _op="pow",
            _parents=(self,),
            _ctx={"exponent": exponent},
        )

    def __matmul__(self, other: Union["Tensor", NDArray[np.float32]]) -> "Tensor":
        """Matrix multiplication."""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        if self.ndim != 2 or other.ndim != 2:
            raise RuntimeError("matmul requires 2D tensors")

        if self.shape[1] != other.shape[0]:
            raise RuntimeError(
                f"Shape mismatch in matmul: {self.shape} vs {other.shape}. "
                f"Expected self.shape[1] == other.shape[0]"
            )

        result_data = self.data @ other.data
        ctx = {
            "b_T": other.data.T,
            "a_T": self.data.T,
        }
        return Tensor(
            result_data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op="matmul",
            _parents=(self, other),
            _ctx=ctx,
        )

    def matmul(self, other: Union["Tensor", NDArray[np.float32]]) -> "Tensor":
        """General matrix multiplication supporting batched inputs.

        For 2D inputs: standard matrix multiplication
        For ND inputs: batched matrix multiplication over last two dimensions

        Args:
            other: Tensor to multiply with.

        Returns:
            Result tensor.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        if self.ndim < 1 or other.ndim < 1:
            raise RuntimeError("matmul requires at least 1D tensors")

        if self.shape[-1] != other.shape[-2 if other.ndim > 1 else 0]:
            raise RuntimeError(
                f"Shape mismatch in matmul: {self.shape} vs {other.shape}"
            )

        result_data = np.matmul(self.data, other.data)
        ctx = {
            "a_shape": self.shape,
            "b_shape": other.shape,
        }
        return Tensor(
            result_data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op="batch_matmul",
            _parents=(self, other),
            _ctx=ctx,
        )

    @property
    def T(self) -> "Tensor":
        """Transpose the tensor."""
        return Tensor(
            self.data.T,
            requires_grad=self.requires_grad,
            _op="transpose",
            _parents=(self,),
        )

    # Activation functions
    def relu(self) -> "Tensor":
        """Rectified Linear Unit activation."""
        result_data = np.maximum(0, self.data)
        return Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _op="relu",
            _parents=(self,),
        )

    def sigmoid(self) -> "Tensor":
        """Sigmoid activation."""
        # Stable sigmoid implementation
        result_data = 1 / (1 + np.exp(-np.clip(self.data, -15, 15)))
        return Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _op="sigmoid",
            _parents=(self,),
        )

    def tanh(self) -> "Tensor":
        """Hyperbolic tangent activation."""
        result_data = np.tanh(self.data)
        return Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _op="tanh",
            _parents=(self,),
        )
    
    def gelu(self) -> "Tensor":
        """Gaussian Error Linear Unit activation (fast approximation).
        
        Uses the fast approximation: gelu(x) ≈ x * sigmoid(1.702 * x).
        
        Returns:
            Tensor with GELU activation applied element-wise.
        """
        return self * (self * 1.702).sigmoid()
    
    def swish(self, beta: float = 1.0) -> "Tensor":
        """Swish activation (also called SiLU).
        
        Args:
            beta: Scaling factor for sigmoid. Default 1.0 (standard Swish).
        
        Returns:
            Tensor with Swish activation applied element-wise.
        """
        # Swish: x * sigmoid(beta * x)
        return self * (self * beta).sigmoid()
    
    def leaky_relu(self, negative_slope: float = 0.01) -> "Tensor":
        """Leaky ReLU activation.
        
        Args:
            negative_slope: Slope for negative values. Default 0.01.
        
        Returns:
            Tensor with LeakyReLU activation applied element-wise.
        
        Formula: leaky_relu(x) = max(0, x) + negative_slope * min(0, x)
        """
        return self.relu() + (self - self.relu()) * negative_slope
    
    def elu(self, alpha: float = 1.0) -> "Tensor":
        """Exponential Linear Unit activation.
        
        Args:
            alpha: Scaling factor for negative values. Default 1.0.
        
        Returns:
            Tensor with ELU activation applied element-wise.
        
        Formula: elu(x) = x if x > 0 else alpha * (exp(x) - 1)
        """
        # elu(x) = relu(x) + alpha * (exp(min(x, 0)) - 1)
        return self.relu() + alpha * ((self - self.relu()).exp() - 1)
    
    def softplus(self) -> "Tensor":
        """Softplus activation.
        
        Formula: softplus(x) = log(1 + exp(x))
        
        Returns:
            Tensor with Softplus activation applied element-wise.
        """
        # softplus(x) = log(1 + exp(x))
        # For numerical stability: log(1 + exp(x)) = max(x, 0) + log(1 + exp(-|x|))
        return (self.relu()) + ((-self.abs()).exp() + 1).log()
    
    def hardswish(self) -> "Tensor":
        """Hard Swish activation.
        
        Formula: hardswish(x) = x * relu6(x + 3) / 6
        
        Returns:
            Tensor with HardSwish activation applied element-wise.
        """
        # relu6(x) = min(max(0, x), 6)
        relu6 = (self + 3).relu()
        relu6 = relu6.clamp(0, 6)
        return self * relu6 * (1.0 / 6.0)
    
    def hardsigmoid(self) -> "Tensor":
        """Hard Sigmoid activation.
        
        Formula: hardsigmoid(x) = relu6(x + 3) / 6
        
        Returns:
            Tensor with HardSigmoid activation applied element-wise.
        """
        relu6 = (self + 3).relu()
        relu6 = relu6.clamp(0, 6)
        return relu6 * (1.0 / 6.0)
    
    def silu(self) -> "Tensor":
        """SiLU (Sigmoid Linear Unit) activation, equivalent to Swish with beta=1.
        
        Returns:
            Tensor with SiLU activation applied element-wise.
        """
        return self.swish(beta=1.0)
    
    def prelu(self, weight: "Tensor") -> "Tensor":
        """Parametric ReLU activation.
        
        Args:
            weight: Learnable parameter with same shape as input or broadcastable.
        
        Returns:
            Tensor with PReLU activation applied element-wise.
        
        Formula: prelu(x) = max(0, x) + weight * min(0, x)
        """
        return self.relu() + (self - self.relu()) * weight

    # Shape operations
    def reshape(self, new_shape: Tuple[int, ...]) -> "Tensor":
        """Reshape the tensor.

        Supports -1 in new_shape to infer the dimension size.
        """
        new_shape_list = list(new_shape)
        if -1 in new_shape_list:
            known_size = np.prod([s for s in new_shape_list if s != -1])
            total_size = np.prod(self.shape)
            inferred_size = total_size // known_size
            idx = new_shape_list.index(-1)
            new_shape_list[idx] = int(inferred_size)
            new_shape = tuple(new_shape_list)

        if np.prod(new_shape) != np.prod(self.shape):
            raise ValueError(
                f"Cannot reshape tensor of shape {self.shape} to {new_shape}. "
                f"Total elements must match."
            )

        return Tensor(
            self.data.reshape(new_shape),
            requires_grad=self.requires_grad,
            _op="reshape",
            _parents=(self,),
        )

    def squeeze(self, axis: Union[int, Tuple[int, ...], None] = None) -> "Tensor":
        """Remove dimensions of size 1.

        Args:
            axis: Selects a subset of the dimensions of size 1 to remove.
                If None, all dimensions of size 1 are removed.

        Returns:
            Squeezed tensor.
        """
        result_data = self.data.squeeze(axis=axis)
        return Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _op="squeeze",
            _parents=(self,),
            _ctx={"axis": axis, "input_shape": self.shape},
        )

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        """Transpose two dimensions of the tensor.

        Args:
            dim0: First dimension to transpose.
            dim1: Second dimension to transpose.

        Returns:
            Transposed tensor.
        """
        if dim0 < -self.ndim or dim0 >= self.ndim:
            raise ValueError(f"dim0 out of range: {dim0}")
        if dim1 < -self.ndim or dim1 >= self.ndim:
            raise ValueError(f"dim1 out of range: {dim1}")

        perm = list(range(self.ndim))
        dim0_pos = dim0 if dim0 >= 0 else self.ndim + dim0
        dim1_pos = dim1 if dim1 >= 0 else self.ndim + dim1
        perm[dim0_pos], perm[dim1_pos] = perm[dim1_pos], perm[dim0_pos]

        return self.permute(*perm)

    def permute(self, *dims: int) -> "Tensor":
        """Permute the dimensions of the tensor.

        Args:
            *dims: Desired ordering of dimensions.

        Returns:
            Permuted tensor.

        Raises:
            ValueError: If dimensions are invalid or contain duplicates.
        """
        # Validate dimensions
        if len(dims) != self.ndim:
            raise ValueError(
                f"Number of dimensions must match tensor ndim. "
                f"Expected {self.ndim}, got {len(dims)}"
            )
        
        # Check for duplicates and valid range
        seen = set()
        for d in dims:
            if d < -self.ndim or d >= self.ndim:
                raise ValueError(
                    f"Dimension out of range (expected to be in range "
                    f"[-{self.ndim}, {self.ndim-1}]), got {d}"
                )
            # Convert negative indices to positive
            pos_d = d if d >= 0 else self.ndim + d
            if pos_d in seen:
                raise ValueError(f"Repeated dimension: {d}")
            seen.add(pos_d)
        
        # Convert to positive indices for numpy transpose
        pos_dims = tuple(d if d >= 0 else self.ndim + d for d in dims)
        result_data = self.data.transpose(pos_dims)
        
        # Compute inverse permutation for gradient
        inverse_perm = [0] * len(pos_dims)
        for i, d in enumerate(pos_dims):
            inverse_perm[d] = i
        
        return Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _op="permute",
            _parents=(self,),
            _ctx={"permutation": pos_dims, "inverse_perm": tuple(inverse_perm)},
        )

    def view(self, *shape: int) -> "Tensor":
        """Return a tensor with same data but different shape.
        
        For now, this is an alias for reshape since NumPy doesn't distinguish
        between views and copies in the same way PyTorch does.
        
        Args:
            *shape: Desired shape.
            
        Returns:
            Tensor with new shape.
        """
        # Convert to tuple
        shape_tuple = tuple(shape)
        return self.reshape(shape_tuple)

    def gather(self, dim: int, index: "Tensor") -> "Tensor":
        """Gather values along an axis using indices.

        Args:
            dim: Dimension to gather along.
            index: Tensor containing indices to gather.

        Returns:
            Tensor with same shape as index.

        Raises:
            ValueError: If dimensions are invalid.
            RuntimeError: If index shape is incompatible.
        """
        import numpy as np
        
        # Handle negative dimension
        if dim < 0:
            dim = self.ndim + dim
        
        if dim < 0 or dim >= self.ndim:
            raise ValueError(
                f"Dimension out of range. Expected 0 <= dim < {self.ndim}, got {dim}"
            )
        
        # Check index shape: must have same number of dimensions as input
        if index.ndim != self.ndim:
            raise RuntimeError(
                f"Index tensor must have same number of dimensions as input. "
                f"Expected {self.ndim}, got {index.ndim}"
            )
        
        # Check index shape matches input shape except at gather dimension
        for i in range(self.ndim):
            if i != dim and index.shape[i] != self.shape[i]:
                raise RuntimeError(
                    f"Index shape mismatch at dimension {i}. "
                    f"Expected {self.shape[i]}, got {index.shape[i]}"
                )
        
        # Use numpy's take_along_axis for gathering
        result_data = np.take_along_axis(self.data, index.data.astype(np.int64), axis=dim)
        
        # Determine if gradient tracking is needed
        requires_grad = self.requires_grad
        
        return Tensor(
            result_data,
            requires_grad=requires_grad,
            _op="gather",
            _parents=(self, index),
            _ctx={
                "dim": dim,
                "input_shape": self.shape,
                "index_shape": index.shape
            }
        )

    def scatter(self, dim: int, index: "Tensor", src: "Tensor") -> "Tensor":
        """Scatter values from src into self at positions specified by index along dim.
        
        Args:
            dim: Dimension along which to index.
            index: Tensor containing indices where to scatter.
            src: Tensor containing values to scatter.
            
        Returns:
            Tensor with same shape as self.
            
        Raises:
            ValueError: If dimensions are invalid.
            RuntimeError: If shape mismatches.
        """
        import numpy as np
        
        # Handle negative dimension
        if dim < 0:
            dim = self.ndim + dim
        
        if dim < 0 or dim >= self.ndim:
            raise ValueError(
                f"Dimension out of range. Expected 0 <= dim < {self.ndim}, got {dim}"
            )
        
        # Check index and src have same number of dimensions as self
        if index.ndim != self.ndim or src.ndim != self.ndim:
            raise RuntimeError(
                f"Index and src must have same number of dimensions as self. "
                f"Expected {self.ndim}, got index.ndim={index.ndim}, src.ndim={src.ndim}"
            )
        
        # Check index and src have same shape
        if index.shape != src.shape:
            raise RuntimeError(
                f"Index and src must have same shape. "
                f"Got index.shape={index.shape}, src.shape={src.shape}"
            )
        
        # Check index shape matches self shape except at scatter dimension
        for i in range(self.ndim):
            if i != dim and index.shape[i] != self.shape[i]:
                raise RuntimeError(
                    f"Index shape mismatch at dimension {i}. "
                    f"Expected {self.shape[i]}, got {index.shape[i]}"
                )
        
        # Validate index values
        max_index = self.shape[dim]
        if np.any(index.data < 0) or np.any(index.data >= max_index):
            raise ValueError(
                f"Index values must be in range [0, {max_index})"
            )
        
        # Create result as copy of self data
        result_data = self.data.copy()
        
        # Use numpy's put_along_axis to scatter src values
        np.put_along_axis(result_data, index.data.astype(np.int64), src.data, axis=dim)
        
        # Determine if gradient tracking is needed
        requires_grad = self.requires_grad or src.requires_grad
        
        return Tensor(
            result_data,
            requires_grad=requires_grad,
            _op="scatter",
            _parents=(self, index, src),
            _ctx={
                "dim": dim,
                "input_shape": self.shape,
                "index_shape": index.shape,
                "src_shape": src.shape
            }
        )

    def sum(
        self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False
    ) -> "Tensor":
        """Sum tensor elements along the specified axis(es)."""
        result_data = self.data.sum(axis=axis, keepdims=keepdims)
        return Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _op="sum",
            _parents=(self,),
            _ctx={"axis": axis, "keepdims": keepdims, "input_shape": self.shape},
        )

    def mean(
        self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False
    ) -> "Tensor":
        """Mean of tensor elements along the specified axis(es)."""
        result_data = self.data.mean(axis=axis, keepdims=keepdims)
        return Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _op="mean",
            _parents=(self,),
            _ctx={"axis": axis, "keepdims": keepdims, "input_shape": self.shape},
        )

    def prod(
        self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False
    ) -> "Tensor":
        """Product of tensor elements along the specified axis(es)."""
        result_data = self.data.prod(axis=axis, keepdims=keepdims)
        return Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _op="prod",
            _parents=(self,),
            _ctx={"axis": axis, "keepdims": keepdims, "input_shape": self.shape},
        )

    def var(
        self,
        axis: Union[int, Tuple[int, ...], None] = None,
        keepdims: bool = False,
        ddof: int = 0,
    ) -> "Tensor":
        """Variance of tensor elements along the specified axis(es).

        Args:
            axis: Axis or axes along which to compute variance.
            keepdims: Whether to keep reduced dimensions.
            ddof: Delta degrees of freedom. The divisor used is N - ddof,
                  where N is the number of elements. Default is 0.

        Returns:
            Tensor with variance values.
        """
        # Compute mean along axis (keep dimensions for broadcasting)
        mean = self.mean(axis=axis, keepdims=True)
        # Compute squared differences
        squared_diff = (self - mean) ** 2
        # Compute mean of squared differences (variance with ddof=0)
        var = squared_diff.mean(axis=axis, keepdims=keepdims)

        # Adjust for ddof (degrees of freedom)
        if ddof != 0:
            # Compute number of elements along reduced axes
            if axis is None:
                n = self.data.size
            else:
                if isinstance(axis, int):
                    axis = (axis,)
                n = 1
                for ax in axis:
                    if ax < 0:
                        ax = self.ndim + ax
                    n *= self.shape[ax]
            # Scale variance by n/(n-ddof)
            scale = n / (n - ddof)
            var = var * scale

        return var

    def std(
        self,
        axis: Union[int, Tuple[int, ...], None] = None,
        keepdims: bool = False,
        ddof: int = 0,
    ) -> "Tensor":
        """Standard deviation of tensor elements along the specified axis(es).

        Args:
            axis: Axis or axes along which to compute standard deviation.
            keepdims: Whether to keep reduced dimensions.
            ddof: Delta degrees of freedom. The divisor used is N - ddof,
                  where N is the number of elements. Default is 0.

        Returns:
            Tensor with standard deviation values.
        """
        # Standard deviation is square root of variance
        var = self.var(axis=axis, keepdims=True, ddof=ddof)
        std = var.sqrt()

        # If keepdims is False and axis is not None, we need to squeeze the dimensions
        # that were kept for broadcasting but should be removed
        if not keepdims and axis is not None:
            # squeeze only the dimensions that were reduced
            if isinstance(axis, int):
                axis = (axis,)
            # Convert negative axis to positive
            axis = tuple(ax if ax >= 0 else self.ndim + ax for ax in axis)
            # Sort axis to squeeze from highest to lowest
            for ax in sorted(axis, reverse=True):
                std = std.squeeze(axis=ax)

        return std

    def max(
        self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False
    ) -> "Tensor":
        """Maximum of tensor elements along the specified axis(es)."""
        result_data = np.max(self.data, axis=axis, keepdims=keepdims)
        return Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _op="max",
            _parents=(self,),
        )

    def min(
        self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False
    ) -> "Tensor":
        """Minimum of tensor elements along the specified axis(es)."""
        result_data = np.min(self.data, axis=axis, keepdims=keepdims)
        return Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _op="min",
            _parents=(self,),
        )

    def exp(self) -> "Tensor":
        """Element-wise exponential."""
        result_data = np.exp(self.data)
        return Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _op="exp",
            _parents=(self,),
        )

    def log(self) -> "Tensor":
        """Element-wise natural logarithm (matches PyTorch behavior).
        
        Returns:
            Tensor with natural logarithm of each element.
            log(0) = -inf, log(negative) = nan (matching NumPy behavior).
        """
        result_data = np.log(self.data)
        return Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _op="log",
            _parents=(self,),
        )

    def abs(self) -> "Tensor":
        """Element-wise absolute value."""
        result_data = np.abs(self.data)
        return Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _op="abs",
            _parents=(self,),
        )

    def sqrt(self) -> "Tensor":
        """Element-wise square root."""
        # Use maximum to avoid sqrt(negative) due to numerical errors
        result_data = np.sqrt(np.maximum(self.data, 0.0))
        return Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _op="sqrt",
            _parents=(self,),
        )

    def clamp(
        self, min_val: Optional[float] = None, max_val: Optional[float] = None
    ) -> "Tensor":
        """Clamp tensor values to [min_val, max_val] range.

        Args:
            min_val: Minimum value (if None, no lower bound).
            max_val: Maximum value (if None, no upper bound).

        Returns:
            Clamped tensor.
        """
        result_data = self.data.copy()
        if min_val is not None:
            result_data = np.maximum(result_data, min_val)
        if max_val is not None:
            result_data = np.minimum(result_data, max_val)
        return Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _op="clamp",
            _parents=(self,),
            _ctx={"min_val": min_val, "max_val": max_val},
        )

    def softmax(self, dim: int = -1) -> "Tensor":
        """Apply softmax along the specified dimension.

        Args:
            dim: Dimension along which softmax will be computed.

        Returns:
            Tensor with softmax applied along dimension.
        """
        # Handle negative dimension
        if dim < 0:
            dim = self.ndim + dim

        # Numerical stability: subtract max along dim
        shifted = self.data - np.max(self.data, axis=dim, keepdims=True)
        exp = np.exp(shifted)
        result_data = exp / np.sum(exp, axis=dim, keepdims=True)

        return Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _op="softmax",
            _parents=(self,),
            _ctx={"dim": dim},
        )

    def log_softmax(self, dim: int = -1) -> "Tensor":
        """Apply log-softmax along the specified dimension.

        Args:
            dim: Dimension along which log-softmax will be computed.

        Returns:
            Tensor with log-softmax applied along dimension.
        """
        if dim < 0:
            dim = self.ndim + dim

        # Numerical stability: use logsumexp trick
        shifted = self.data - np.max(self.data, axis=dim, keepdims=True)
        exp = np.exp(shifted)
        log_sum_exp = np.log(np.sum(exp, axis=dim, keepdims=True))
        result_data = shifted - log_sum_exp

        return Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _op="log_softmax",
            _parents=(self,),
            _ctx={"dim": dim},
        )

    # Utility methods
    def item(self) -> float:
        """Convert scalar tensor to Python float."""
        if self.shape != ():
            raise ValueError(
                f"item() can only be called on scalar tensors, got shape {self.shape}"
            )
        return float(self.data)

    def numpy(self) -> NDArray[np.float32]:
        """Return the tensor data as numpy array."""
        return self.data

    def detach(self) -> "Tensor":
        """Return a new tensor detached from the computational graph."""
        return Tensor(self.data, requires_grad=False)

    def clone(self) -> "Tensor":
        """Return a copy of the tensor."""
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        """Flatten a contiguous range of dims into a single dimension.

        Args:
            start_dim: First dim to flatten (default: 0).
            end_dim: Last dim to flatten (default: -1, meaning up to last dim).

        Returns:
            Flattened tensor.
        """
        if end_dim < 0:
            end_dim = self.ndim + end_dim

        if start_dim == end_dim:
            return self

        new_shape = list(self.shape[:start_dim])
        flat_size = 1
        for i in range(start_dim, end_dim + 1):
            flat_size *= self.shape[i]
        new_shape.append(flat_size)
        new_shape.extend(self.shape[end_dim + 1:])

        return self.reshape(tuple(new_shape))

    def expand(self, *sizes: int) -> "Tensor":
        """Expand tensor to given sizes (broadcasting).

        Args:
            *sizes: Desired sizes for each dimension. Use -1 to keep original size.

        Returns:
            Expanded tensor (view).
        """
        sizes_list = list(sizes)
        new_shape = list(self.shape)
        
        # Pad with leading 1s if needed
        while len(new_shape) < len(sizes_list):
            new_shape.insert(0, 1)
        
        # Build expanded shape
        result_shape = []
        for i, (orig, target) in enumerate(zip(reversed(new_shape), reversed(sizes_list))):
            if target == -1:
                result_shape.insert(0, orig)
            elif orig == 1:
                result_shape.insert(0, target)
            elif orig == target:
                result_shape.insert(0, target)
            else:
                raise RuntimeError(
                    f"Cannot expand shape {self.shape} to size {sizes}"
                )
        
        # Use numpy broadcast_to
        result_data = np.broadcast_to(self.data, tuple(result_shape))
        return Tensor(result_data.copy(), requires_grad=self.requires_grad)

    def repeat(self, *repeats: int) -> "Tensor":
        """Repeat tensor along each dimension.

        Args:
            *repeats: Number of repetitions for each dimension.

        Returns:
            Repeated tensor.
        """
        if len(repeats) < self.ndim:
            repeats = (1,) * (self.ndim - len(repeats)) + repeats
        
        result_data = np.tile(self.data, repeats)
        return Tensor(result_data, requires_grad=self.requires_grad)

    def split(self, split_size: int, dim: int = 0) -> List["Tensor"]:
        """Split tensor into chunks of given size along dimension.

        Args:
            split_size: Size of each chunk.
            dim: Dimension along which to split.

        Returns:
            List of tensor chunks.
        """
        chunks = []
        size = self.shape[dim]
        for i in range(0, size, split_size):
            end = min(i + split_size, size)
            # Build indices for slicing
            idx = [slice(None)] * self.ndim
            idx[dim] = slice(i, end)
            chunk_data = self.data[tuple(idx)]
            chunks.append(Tensor(chunk_data, requires_grad=self.requires_grad))
        return chunks

    def chunk(self, chunks: int, dim: int = 0) -> List["Tensor"]:
        """Split tensor into given number of chunks along dimension.

        Args:
            chunks: Number of chunks to create.
            dim: Dimension along which to split.

        Returns:
            List of tensor chunks.
        """
        if chunks <= 0:
            raise ValueError(f"chunks must be greater than 0, got {chunks}")
        
        dim_size = self.shape[dim]
        chunk_size = (dim_size + chunks - 1) // chunks
        return self.split(chunk_size, dim)

    def topk(self, k: int, dim: int = -1, largest: bool = True, sorted: bool = True) -> Tuple["Tensor", "Tensor"]:
        """Return the k largest/smallest elements along a dimension.

        Args:
            k: Number of top elements to return.
            dim: Dimension to sort along.
            largest: If True, return largest. Otherwise, smallest.
            sorted: If True, return elements in sorted order.

        Returns:
            Tuple of (values, indices) tensors.
        """
        if dim < 0:
            dim = self.ndim + dim

        if largest:
            indices = np.argsort(-self.data, axis=dim)
        else:
            indices = np.argsort(self.data, axis=dim)

        # Take top k
        slices = [slice(None)] * self.ndim
        slices[dim] = slice(0, k)
        top_indices = indices[tuple(slices)]
        
        # Gather values
        values = np.take_along_axis(self.data, top_indices, axis=dim)

        return (
            Tensor(values, requires_grad=self.requires_grad),
            Tensor(top_indices.astype(np.int64), requires_grad=False)
        )

    def sort(self, dim: int = -1, descending: bool = False) -> Tuple["Tensor", "Tensor"]:
        """Sort tensor along a dimension.

        Args:
            dim: Dimension to sort along.
            descending: If True, sort in descending order.

        Returns:
            Tuple of (sorted_values, indices) tensors.
        """
        if dim < 0:
            dim = self.ndim + dim

        if descending:
            indices = np.argsort(-self.data, axis=dim)
        else:
            indices = np.argsort(self.data, axis=dim)

        sorted_data = np.take_along_axis(self.data, indices, axis=dim)

        return (
            Tensor(sorted_data, requires_grad=self.requires_grad),
            Tensor(indices.astype(np.int64), requires_grad=False)
        )

    def where(self, condition: "Tensor", y: "Tensor") -> "Tensor":
        """Select elements based on condition.

        Args:
            condition: Boolean tensor mask.
            y: Tensor for False values.

        Returns:
            Tensor with elements from self where condition is True, else from y.
        """
        result = np.where(condition.data.astype(bool), self.data, y.data)
        return Tensor(result, requires_grad=self.requires_grad or y.requires_grad)

    # Static factory methods
    @staticmethod
    def zeros(shape: Tuple[int, ...], requires_grad: bool = False) -> "Tensor":
        """Create a tensor filled with zeros."""
        return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)

    @staticmethod
    def ones(shape: Tuple[int, ...], requires_grad: bool = False) -> "Tensor":
        """Create a tensor filled with ones."""
        return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)

    @staticmethod
    def ones_like(other: "Tensor", requires_grad: bool = False) -> "Tensor":
        """Create a tensor filled with ones with the same shape as another tensor."""
        return Tensor(np.ones_like(other.data), requires_grad=requires_grad)

    @staticmethod
    def zeros_like(other: "Tensor", requires_grad: bool = False) -> "Tensor":
        """Create a tensor filled with zeros with the same shape as another tensor."""
        return Tensor(np.zeros_like(other.data), requires_grad=requires_grad)

    @staticmethod
    def randn(shape: Tuple[int, ...], requires_grad: bool = False) -> "Tensor":
        """Create a tensor with random values from standard normal distribution."""
        return Tensor(
            np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad
        )

    @staticmethod
    def rand(shape: Tuple[int, ...], requires_grad: bool = False) -> "Tensor":
        """Create a tensor with random values from uniform distribution [0, 1)."""
        return Tensor(
            np.random.rand(*shape).astype(np.float32), requires_grad=requires_grad
        )

    @staticmethod
    def eye(n: int, requires_grad: bool = False) -> "Tensor":
        """Create an identity matrix."""
        return Tensor(np.eye(n, dtype=np.float32), requires_grad=requires_grad)

    @staticmethod
    def arange(
        *args: Union[int, float], requires_grad: bool = False, **kwargs: Any
    ) -> "Tensor":
        """Create a 1D tensor with values from start to stop with given step.

        Similar to numpy.arange and torch.arange.
        Supports both numpy-style (start, stop, step) and
        PyTorch-style (start, end, step) signatures.

        Usage:
            Tensor.arange(stop) -> values from 0 to stop-1
            Tensor.arange(start, stop) -> values from start to stop-1
            Tensor.arange(start, stop, step) -> values from start to stop-1 with step

        Args:
            *args: Either (stop,) or (start, stop) or (start, stop, step)
            requires_grad: Whether to track gradients.
            **kwargs: Ignored for compatibility.

        Returns:
            1D Tensor with values [start, start+step, ..., stop-step].
        """
        start: Union[int, float]
        stop: Union[int, float]
        step: Union[int, float]

        if len(args) == 1:
            start, stop, step = 0, args[0], 1
        elif len(args) == 2:
            start, stop, step = args[0], args[1], 1
        elif len(args) == 3:
            start, stop, step = args[0], args[1], args[2]
        else:
            raise ValueError(
                f"arange expected 1-3 positional arguments, got {len(args)}"
            )

        return Tensor(
            np.arange(start, stop, step, dtype=np.float32), requires_grad=requires_grad
        )

    @staticmethod
    def cat(tensors: List["Tensor"], dim: int = 0) -> "Tensor":
        """Concatenate tensors along a dimension.
        
        Args:
            tensors: List of tensors to concatenate.
            dim: Dimension along which to concatenate.
            
        Returns:
            Concatenated tensor.
        """
        from nanotorch.utils import cat
        return cat(tensors, dim=dim)
    
    @staticmethod
    def stack(tensors: List["Tensor"], dim: int = 0) -> "Tensor":
        """Stack tensors along a new dimension.
        
        Args:
            tensors: List of tensors to stack.
            dim: Dimension to insert for stacking.
            
        Returns:
            Stacked tensor.
        """
        from nanotorch.utils import stack
        return stack(tensors, dim=dim)


# Context manager for gradient tracking
class no_grad:
    """Context manager to disable gradient tracking.

    Example:
        with no_grad():
            y = x * 2  # No gradient tracking
    """

    def __init__(self) -> None:
        self.prev_enable_grad = _ENABLE_GRAD

    def __enter__(self) -> None:
        global _ENABLE_GRAD
        self.prev_enable_grad = _ENABLE_GRAD
        _ENABLE_GRAD = False

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        global _ENABLE_GRAD
        _ENABLE_GRAD = self.prev_enable_grad


# Global flag for gradient tracking
_ENABLE_GRAD = True
