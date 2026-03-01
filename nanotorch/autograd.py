"""
Automatic differentiation engine for nanotorch.

This module provides the core autograd functionality similar to PyTorch's autograd.
It includes the Function base class for defining custom operations with
forward and backward passes.
"""

from typing import Optional, Tuple, Any, List, Set, Dict, Union
import numpy as np
from numpy.typing import NDArray
from nanotorch.tensor import Tensor


class Function:
    """Base class for creating autograd operations.

    Subclasses should implement the forward and backward static methods.
    This is similar to PyTorch's torch.autograd.Function.

    Example:
        class Add(Function):
            @staticmethod
            def forward(ctx, a, b):
                ctx.save_for_backward(a, b)
                return a + b

            @staticmethod
            def backward(ctx, grad_output):
                a, b = ctx.saved_tensors
                return grad_output, grad_output
    """

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the operation.

        Args:
            ctx: Context object to save information for backward pass.
            *args: Input arguments.
            **kwargs: Keyword arguments.

        Returns:
            Output tensor(s) or tuple of outputs. Non-tensor outputs are allowed
            but won't participate in gradient computation.
        """
        raise NotImplementedError("Subclasses must implement forward")

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """Backward pass of the operation.

        Args:
            ctx: Context object with saved information from forward pass.
            *grad_outputs: Gradients of the output with respect to some scalar.

        Returns:
            Gradients with respect to each input.
        """
        raise NotImplementedError("Subclasses must implement backward")

    @classmethod
    def apply(cls, *args: Any, **kwargs: Any) -> Any:
        """Apply the function to inputs.

        This creates a new tensor with the function as its _op and sets up
        the computational graph.

        Returns:
            Tensor or tuple of Tensors/non-Tensor values. If a tuple is returned,
            only Tensor elements will have gradient tracking enabled.
        """
        # Create context object
        ctx = FunctionContext()

        # Run forward pass
        output = cls.forward(ctx, *args, **kwargs)

        def process_output(out):
            if isinstance(out, Tensor):
                out._ctx = ctx
                out._fn = cls
                out._parents = args
                return out
            else:
                # Non-tensor output (e.g., indices as numpy array)
                return out

        if isinstance(output, tuple):
            # Process each element, keep tuple structure
            processed = tuple(process_output(elem) for elem in output)
            return processed
        else:
            # Single output (should be Tensor, but we allow non-Tensor for flexibility)
            return process_output(output)


class FunctionContext:
    """Context object for saving information between forward and backward passes.

    Similar to PyTorch's ctx object in Function.forward.
    """

    def __init__(self) -> None:
        self.saved_tensors: List[Tensor] = []
        self.saved_values: Dict[str, Any] = {}

    def save_for_backward(self, *tensors: Tensor) -> None:
        """Save tensors for use in the backward pass."""
        self.saved_tensors.extend(tensors)

    def save_value(self, key: str, value: Any) -> None:
        """Save a value for use in the backward pass."""
        self.saved_values[key] = value

    def get_saved_tensors(self) -> Tuple[Tensor, ...]:
        """Get saved tensors."""
        return tuple(self.saved_tensors)

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a saved value."""
        return self.saved_values.get(key, default)


def backward(tensor: Tensor, gradient: Optional[Tensor] = None) -> None:
    """Compute gradients of tensors in the computational graph.

    This is the main entry point for backpropagation.

    Args:
        tensor: Root tensor to start backpropagation from.
        gradient: Gradient with respect to the tensor (default is ones).
    """
    if gradient is None:
        gradient = Tensor(np.ones_like(tensor.data), requires_grad=False)

    # Set gradient on the root tensor
    if tensor.grad is None:
        tensor.grad = gradient
    else:
        tensor.grad = tensor.grad + gradient

    # Build topological order of the computational graph
    topo: List[Tensor] = []
    visited: Set[Tensor] = set()

    def build_topo(v: Tensor) -> None:
        if v not in visited:
            visited.add(v)
            if hasattr(v, "_parents"):
                for parent in v._parents:
                    if isinstance(parent, Tensor):
                        build_topo(parent)

    build_topo(tensor)

    # Backward pass in reverse topological order
    for v in reversed(topo):
        if hasattr(v, "_fn") and v.grad is not None:
            # Use Function-based backward
            ctx = getattr(v, "_ctx", None)
            if ctx is not None:
                # Compute gradients using the function's backward method
                grad_inputs = v._fn.backward(ctx, v.grad)

                # Distribute gradients to parent tensors
                if v._parents:
                    if isinstance(grad_inputs, tuple):
                        for parent, grad in zip(v._parents, grad_inputs):
                            if isinstance(parent, Tensor) and parent.requires_grad:
                                if grad is not None:
                                    if parent.grad is None:
                                        parent.grad = grad
                                    else:
                                        parent.grad = parent.grad + grad
                    elif isinstance(grad_inputs, Tensor) and len(v._parents) == 1:
                        parent = v._parents[0]
                        if isinstance(parent, Tensor) and parent.requires_grad:
                            if grad_inputs is not None:
                                if parent.grad is None:
                                    parent.grad = grad_inputs
                                else:
                                    parent.grad = parent.grad + grad_inputs
        elif hasattr(v, "_op") and v.grad is not None:
            # Use the old string-based operation system (for compatibility)
            v.backward(v.grad)


# Example Function implementations for common operations


class Add(Function):
    """Element-wise addition."""

    @staticmethod
    def forward(ctx: Any, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        return Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Tensor, Tensor]:
        grad_output = grad_outputs[0]
        a, b = ctx.saved_tensors
        return grad_output, grad_output


class Mul(Function):
    """Element-wise multiplication."""

    @staticmethod
    def forward(ctx: Any, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        return Tensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        grad_output = grad_outputs[0]
        a, b = ctx.saved_tensors
        grad_a: Optional[Tensor] = None
        grad_b: Optional[Tensor] = None
        if a.requires_grad:
            grad_a_data = grad_output.data * b.data
            grad_a = Tensor(grad_a_data, requires_grad=False)
        if b.requires_grad:
            grad_b_data = a.data * grad_output.data
            grad_b = Tensor(grad_b_data, requires_grad=False)
        return grad_a, grad_b


class MatMul(Function):
    """Matrix multiplication."""

    @staticmethod
    def forward(ctx: Any, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        # Cache transposes for efficient backward pass
        ctx.save_value("b_T", b.data.T)
        ctx.save_value("a_T", a.data.T)
        return Tensor(a.data @ b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        grad_output = grad_outputs[0]
        a, b = ctx.saved_tensors
        grad_a: Optional[Tensor] = None
        grad_b: Optional[Tensor] = None
        # Use cached transposes if available, otherwise compute
        b_T = ctx.get_value("b_T")
        if b_T is None:
            b_T = b.data.T
        a_T = ctx.get_value("a_T")
        if a_T is None:
            a_T = a.data.T
        if a.requires_grad:
            grad_a_data = grad_output.data @ b_T
            grad_a = Tensor(grad_a_data, requires_grad=False)
        if b.requires_grad:
            grad_b_data = a_T @ grad_output.data
            grad_b = Tensor(grad_b_data, requires_grad=False)
        return grad_a, grad_b


class Conv2DFunction(Function):
    """2D convolution operation with corrected backward pass.

    ROOT CAUSE OF BUG: Input gradient computation was using incorrect broadcasting.
    The fix: Use grad_output_flat[pos_idx, np.newaxis, :] * weight.data[c_out, np.newaxis, np.newaxis, :]
    to properly broadcast the scalar gradient with the weight tensor.
    
    NOTE ON NUMERICAL PRECISION: Gradient checks using finite differences with float32
    tensors may show relatively high errors (up to ~1e-2 for input gradients, ~3e-2 for
    weight gradients) due to float32 precision limitations in finite difference
    computations. The analytic gradient formulas are mathematically correct; these
    thresholds are acceptable for educational purposes.
    """

    @staticmethod
    def _forward_loops(
        input_data: NDArray[np.float32],
        weight_data: NDArray[np.float32],
        bias_data: Optional[NDArray[np.float32]],
        N: int,
        C_in: int,
        C_out: int,
        H_in: int,
        W_in: int,
        K_H: int,
        K_W: int,
        H_out: int,
        W_out: int,
        stride: int,
        padding: int,
        dilation: int,
        requires_grad_input: bool,
        requires_grad_weight: bool,
        requires_grad_bias: bool,
    ) -> NDArray[np.float32]:
        """Loop-based convolution implementation for fallback."""
        output: NDArray[np.float32] = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)

        for n in range(N):
            for c_out in range(C_out):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride
                        w_start = w_out * stride
                        if dilation == 1:
                            h_end = h_start + K_H
                            w_end = w_start + K_W
                            window = input_data[n, :, h_start:h_end, w_start:w_end]
                        else:
                            h_end = h_start + dilation * (K_H - 1) + 1
                            w_end = w_start + dilation * (K_W - 1) + 1
                            window = input_data[n, :, h_start:h_end:dilation, w_start:w_end:dilation]
                        weight_slice = weight_data[c_out]
                        conv_sum = np.sum(window * weight_slice)
                        output[n, c_out, h_out, w_out] = conv_sum

        if bias_data is not None:
            output += bias_data.reshape(1, C_out, 1, 1)

        return output

    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> Tensor:
        ctx.save_for_backward(input, weight, bias)
        ctx.save_value("stride", stride)
        ctx.save_value("padding", padding)
        ctx.save_value("dilation", dilation)

        if input.ndim != 4:
            raise ValueError(f"Conv2D expects 4D input (N, C, H, W), got {input.ndim}D")
        if weight.ndim != 4:
            raise ValueError(
                f"Conv2D expects 4D weight (C_out, C_in, K_H, K_W), got {weight.ndim}D"
            )

        N, C_in, H_in, W_in = input.shape
        C_out, C_in_w, K_H, K_W = weight.shape

        if C_in != C_in_w:
            raise ValueError(
                f"Input channels mismatch: input has {C_in}, weight expects {C_in_w}"
            )

        def output_size(
            input_size: int, kernel_size: int, stride: int, padding: int, dilation: int
        ) -> int:
            kernel_size_dilated = dilation * (kernel_size - 1) + 1
            return (input_size + 2 * padding - kernel_size_dilated) // stride + 1

        H_out = output_size(H_in, K_H, stride, padding, dilation)
        W_out = output_size(W_in, K_W, stride, padding, dilation)

        if padding > 0:
            padded_input: NDArray[np.float32] = np.zeros(
                (N, C_in, H_in + 2 * padding, W_in + 2 * padding), dtype=np.float32
            )
            padded_input[
                :, :, padding : padding + H_in, padding : padding + W_in
            ] = input.data
            input_data = padded_input
        else:
            input_data = input.data

        requires_grad_input = input.requires_grad
        requires_grad_weight = weight.requires_grad
        requires_grad_bias = bias is not None and bias.requires_grad

        if dilation != 1:
            output_data = Conv2DFunction._forward_loops(
                input_data=input_data,
                weight_data=weight.data,
                bias_data=bias.data if bias is not None else None,
                N=N,
                C_in=C_in,
                C_out=C_out,
                H_in=H_in,
                W_in=W_in,
                K_H=K_H,
                K_W=K_W,
                H_out=H_out,
                W_out=W_out,
                stride=stride,
                padding=padding,
                dilation=dilation,
                requires_grad_input=requires_grad_input,
                requires_grad_weight=requires_grad_weight,
                requires_grad_bias=requires_grad_bias,
            )

            requires_grad = (
                requires_grad_input
                or requires_grad_weight
                or (bias is not None and requires_grad_bias)
            )
            return Tensor(output_data, requires_grad=requires_grad)

        try:
            from numpy.lib.stride_tricks import sliding_window_view
        except ImportError:
            return Conv2DFunction._forward_loops(
                input_data=input_data,
                weight_data=weight.data,
                bias_data=bias.data if bias is not None else None,
                N=N,
                C_in=C_in,
                C_out=C_out,
                H_in=H_in,
                W_in=W_in,
                K_H=K_H,
                K_W=K_W,
                H_out=H_out,
                W_out=W_out,
                stride=stride,
                padding=padding,
                dilation=dilation,
                requires_grad_input=requires_grad_input,
                requires_grad_weight=requires_grad_weight,
                requires_grad_bias=requires_grad_bias,
            )

        windows = sliding_window_view(input_data, (K_H, K_W), axis=(-2, -1))
        windows = windows[:, :, ::stride, ::stride, :, :]
        windows_flat = windows.transpose(0, 2, 3, 1, 4, 5).reshape(N * H_out * W_out, C_in * K_H * K_W)
        weight_flat = weight.data.reshape(C_out, -1)
        output_flat = windows_flat @ weight_flat.T
        output = output_flat.T.reshape(C_out, N, H_out, W_out).transpose(1, 0, 2, 3)

        if bias is not None:
            output += bias.data.reshape(1, C_out, 1, 1)

        requires_grad = (
            requires_grad_input
            or requires_grad_weight
            or (bias is not None and requires_grad_bias)
        )
        return Tensor(output, requires_grad=requires_grad)

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: Any,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], None, None, None]:
        grad_output = grad_outputs[0]
        input, weight, bias = ctx.saved_tensors
        stride = ctx.get_value("stride")
        padding = ctx.get_value("padding")
        dilation = ctx.get_value("dilation")

        N, C_in, H_in, W_in = input.shape
        C_out, C_in_w, K_H, K_W = weight.shape
        _, _, H_out, W_out = grad_output.shape

        grad_weight: Optional[NDArray[np.float32]] = (
            np.zeros_like(weight.data) if weight.requires_grad else None
        )
        grad_bias: Optional[NDArray[np.float32]] = (
            np.zeros_like(bias.data)
            if bias is not None and bias.requires_grad
            else None
        )

        if input.requires_grad:
            grad_input_padded: Optional[NDArray[np.float32]] = np.zeros(
                (N, C_in, H_in + 2 * padding, W_in + 2 * padding), dtype=np.float32
            )
        else:
            grad_input_padded = None

        grad_output_data = grad_output.data

        if grad_bias is not None:
            grad_bias = grad_output_data.sum(axis=(0, 2, 3)).reshape(C_out, 1, 1)

        if dilation == 1:
            if grad_weight is not None:
                if padding > 0:
                    padded_input: NDArray[np.float32] = np.zeros(
                        (N, C_in, H_in + 2 * padding, W_in + 2 * padding), dtype=np.float32
                    )
                    padded_input[
                        :, :, padding : padding + H_in, padding : padding + W_in
                    ] = input.data
                    input_data = padded_input
                else:
                    input_data = input.data

                from numpy.lib.stride_tricks import sliding_window_view
                windows = sliding_window_view(input_data, (K_H, K_W), axis=(-2, -1))
                windows = windows[:, :, ::stride, ::stride, :, :]
                windows_flat = windows.transpose(0, 2, 3, 1, 4, 5).reshape(N * H_out * W_out, C_in * K_H * K_W)
                grad_output_flat = grad_output_data.transpose(0, 2, 3, 1).reshape(N * H_out * W_out, C_out)
                grad_weight_flat = grad_output_flat.T @ windows_flat
                grad_weight = grad_weight_flat.reshape(C_out, C_in, K_H, K_W)
            
            if grad_input_padded is not None:
                # Vectorized input gradient computation for dilation=1
                # Flatten weight: (C_out, C_in, K_H, K_W) -> (C_out, C_in*K_H*K_W)
                weight_flat = weight.data.reshape(C_out, -1)
                
                # Reshape grad_output: (N, C_out, H_out, W_out) -> (N*H_out*W_out, C_out)
                grad_output_flat = grad_output_data.transpose(0, 2, 3, 1).reshape(N * H_out * W_out, C_out)
                
                # Compute contributions for each patch: (N*H_out*W_out, C_in*K_H*K_W)
                contributions_flat = grad_output_flat @ weight_flat
                
                # Reshape contributions: (N, H_out, W_out, C_in, K_H, K_W)
                contributions = contributions_flat.reshape(N, H_out, W_out, C_in, K_H, K_W)
                
                # Loop over kernel positions (small loops: K_H * K_W, typically 3x3=9)
                for kh in range(K_H):
                    for kw in range(K_W):
                        # Contributions for this kernel position: (N, H_out, W_out, C_in)
                        contrib_slice = contributions[:, :, :, :, kh, kw]
                        # Transpose to (N, C_in, H_out, W_out)
                        contrib_slice_t = contrib_slice.transpose(0, 3, 1, 2)
                        
                        # Add to grad_input_padded at strided positions
                        h_start = kh
                        w_start = kw
                        grad_input_padded[:, :, h_start:h_start + H_out * stride:stride, w_start:w_start + W_out * stride:stride] += contrib_slice_t
        else:
            if padding > 0:
                padded_input: NDArray[np.float32] = np.zeros(
                    (N, C_in, H_in + 2 * padding, W_in + 2 * padding), dtype=np.float32
                )
                padded_input[
                    :, :, padding : padding + H_in, padding : padding + W_in
                ] = input.data
                input_data = padded_input
            else:
                input_data = input.data
            
            for n in range(N):
                for c_out in range(C_out):
                    for h_out in range(H_out):
                        for w_out in range(W_out):
                            h_start = h_out * stride
                            w_start = w_out * stride
                            h_end = h_start + dilation * (K_H - 1) + 1
                            w_end = w_start + dilation * (K_W - 1) + 1

                            if grad_weight is not None:
                                window = input_data[n, :, h_start:h_end:dilation, w_start:w_end:dilation]
                                grad_weight[c_out] += (
                                    grad_output_data[n, c_out, h_out, w_out] * window
                                )

                            if grad_input_padded is not None:
                                grad_input_padded[n, :, h_start:h_end:dilation, w_start:w_end:dilation] += (
                                    grad_output_data[n, c_out, h_out, w_out]
                                    * weight.data[c_out]
                                )

        grad_input: Optional[NDArray[np.float32]] = None
        if grad_input_padded is not None:
            if padding > 0:
                grad_input = grad_input_padded[
                    :, :, padding : padding + H_in, padding : padding + W_in
                ]
            else:
                grad_input = grad_input_padded

        return (
            Tensor(grad_input, requires_grad=False) if grad_input is not None else None,
            Tensor(grad_weight, requires_grad=False)
            if grad_weight is not None
            else None,
            Tensor(grad_bias, requires_grad=False) if grad_bias is not None else None,
            None,
            None,
            None,
        )


class Conv1DFunction(Function):
    """1D convolution operation with forward and backward passes.

    Performs 1D convolution on 3D input tensors, useful for sequence data
    or 1D signals.

    Input shape: (N, C_in, L_in)
    Weight shape: (C_out, C_in, K_L)
    Bias shape: (C_out, 1) (optional)
    Output shape: (N, C_out, L_out)
    """

    @staticmethod
    def _forward_loops(
        input_data: NDArray[np.float32],
        weight_data: NDArray[np.float32],
        bias_data: Optional[NDArray[np.float32]],
        N: int,
        C_in: int,
        C_out: int,
        L_in: int,
        K_L: int,
        L_out: int,
        stride: int,
        padding: int,
        dilation: int,
    ) -> NDArray[np.float32]:
        output: NDArray[np.float32] = np.zeros((N, C_out, L_out), dtype=np.float32)

        for n in range(N):
            for c_out in range(C_out):
                for l_out in range(L_out):
                    l_start = l_out * stride
                    if dilation == 1:
                        l_end = l_start + K_L
                        window = input_data[n, :, l_start:l_end]
                    else:
                        l_end = l_start + dilation * (K_L - 1) + 1
                        window = input_data[n, :, l_start:l_end:dilation]
                    weight_slice = weight_data[c_out]
                    conv_sum = np.sum(window * weight_slice)
                    output[n, c_out, l_out] = conv_sum

        if bias_data is not None:
            output += bias_data.reshape(1, C_out, 1)

        return output

    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> Tensor:
        ctx.save_for_backward(input, weight, bias)
        ctx.save_value("stride", stride)
        ctx.save_value("padding", padding)
        ctx.save_value("dilation", dilation)

        if input.ndim != 3:
            raise ValueError(f"Conv1D expects 3D input (N, C, L), got {input.ndim}D")
        if weight.ndim != 3:
            raise ValueError(
                f"Conv1D expects 3D weight (C_out, C_in, K_L), got {weight.ndim}D"
            )

        N, C_in, L_in = input.shape
        C_out, C_in_w, K_L = weight.shape

        if C_in != C_in_w:
            raise ValueError(
                f"Input channels mismatch: input has {C_in}, weight expects {C_in_w}"
            )

        def output_size(
            input_size: int, kernel_size: int, stride: int, padding: int, dilation: int
        ) -> int:
            kernel_size_dilated = dilation * (kernel_size - 1) + 1
            return (input_size + 2 * padding - kernel_size_dilated) // stride + 1

        L_out = output_size(L_in, K_L, stride, padding, dilation)

        if padding > 0:
            padded_input: NDArray[np.float32] = np.zeros(
                (N, C_in, L_in + 2 * padding), dtype=np.float32
            )
            padded_input[:, :, padding : padding + L_in] = input.data
            input_data = padded_input
        else:
            input_data = input.data

        if dilation != 1:
            output_data = Conv1DFunction._forward_loops(
                input_data=input_data,
                weight_data=weight.data,
                bias_data=bias.data if bias is not None else None,
                N=N,
                C_in=C_in,
                C_out=C_out,
                L_in=L_in,
                K_L=K_L,
                L_out=L_out,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            requires_grad = (
                input.requires_grad
                or weight.requires_grad
                or (bias is not None and bias.requires_grad)
            )
            return Tensor(output_data, requires_grad=requires_grad)

        try:
            from numpy.lib.stride_tricks import sliding_window_view
        except ImportError:
            output_data = Conv1DFunction._forward_loops(
                input_data=input_data,
                weight_data=weight.data,
                bias_data=bias.data if bias is not None else None,
                N=N,
                C_in=C_in,
                C_out=C_out,
                L_in=L_in,
                K_L=K_L,
                L_out=L_out,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            requires_grad = (
                input.requires_grad
                or weight.requires_grad
                or (bias is not None and bias.requires_grad)
            )
            return Tensor(output_data, requires_grad=requires_grad)

        windows = sliding_window_view(input_data, K_L, axis=-1)
        windows = windows[:, :, ::stride]
        windows_flat = windows.transpose(0, 2, 1, 3).reshape(N * L_out, C_in * K_L)
        weight_flat = weight.data.reshape(C_out, -1)
        output_flat = windows_flat @ weight_flat.T
        output = output_flat.T.reshape(C_out, N, L_out).transpose(1, 0, 2)

        if bias is not None:
            output += bias.data.reshape(1, C_out, 1)

        requires_grad = (
            input.requires_grad
            or weight.requires_grad
            or (bias is not None and bias.requires_grad)
        )
        return Tensor(output, requires_grad=requires_grad)

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: Any,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], None, None, None]:
        grad_output = grad_outputs[0]
        input, weight, bias = ctx.saved_tensors
        stride = ctx.get_value("stride")
        padding = ctx.get_value("padding")
        dilation = ctx.get_value("dilation")

        N, C_in, L_in = input.shape
        C_out, C_in_w, K_L = weight.shape
        _, _, L_out = grad_output.shape

        grad_weight: Optional[NDArray[np.float32]] = (
            np.zeros_like(weight.data) if weight.requires_grad else None
        )
        grad_bias: Optional[NDArray[np.float32]] = (
            np.zeros_like(bias.data)
            if bias is not None and bias.requires_grad
            else None
        )

        if input.requires_grad:
            grad_input_padded: Optional[NDArray[np.float32]] = np.zeros(
                (N, C_in, L_in + 2 * padding), dtype=np.float32
            )
        else:
            grad_input_padded = None

        grad_output_data = grad_output.data

        if grad_bias is not None:
            grad_bias = grad_output_data.sum(axis=(0, 2)).reshape(C_out, 1)

        if dilation == 1:
            if grad_weight is not None:
                if padding > 0:
                    padded_input: NDArray[np.float32] = np.zeros(
                        (N, C_in, L_in + 2 * padding), dtype=np.float32
                    )
                    padded_input[:, :, padding : padding + L_in] = input.data
                    input_data = padded_input
                else:
                    input_data = input.data

                from numpy.lib.stride_tricks import sliding_window_view
                windows = sliding_window_view(input_data, K_L, axis=-1)
                windows = windows[:, :, ::stride]
                windows_flat = windows.transpose(0, 2, 1, 3).reshape(N * L_out, C_in * K_L)
                grad_output_flat = grad_output_data.transpose(0, 2, 1).reshape(N * L_out, C_out)
                grad_weight_flat = grad_output_flat.T @ windows_flat
                grad_weight = grad_weight_flat.reshape(C_out, C_in, K_L)

            if grad_input_padded is not None:
                weight_flat = weight.data.reshape(C_out, -1)
                grad_output_flat = grad_output_data.transpose(0, 2, 1).reshape(N * L_out, C_out)
                contributions_flat = grad_output_flat @ weight_flat
                contributions = contributions_flat.reshape(N, L_out, C_in, K_L)

                for kl in range(K_L):
                    contrib_slice = contributions[:, :, :, kl]
                    contrib_slice_t = contrib_slice.transpose(0, 2, 1)
                    l_start = kl
                    grad_input_padded[:, :, l_start:l_start + L_out * stride:stride] += contrib_slice_t
        else:
            if padding > 0:
                padded_input: NDArray[np.float32] = np.zeros(
                    (N, C_in, L_in + 2 * padding), dtype=np.float32
                )
                padded_input[:, :, padding : padding + L_in] = input.data
                input_data = padded_input
            else:
                input_data = input.data

            for n in range(N):
                for c_out in range(C_out):
                    for l_out in range(L_out):
                        l_start = l_out * stride
                        l_end = l_start + dilation * (K_L - 1) + 1

                        if grad_weight is not None:
                            window = input_data[n, :, l_start:l_end:dilation]
                            grad_weight[c_out] += (
                                grad_output_data[n, c_out, l_out] * window
                            )

                        if grad_input_padded is not None:
                            grad_input_padded[n, :, l_start:l_end:dilation] += (
                                grad_output_data[n, c_out, l_out]
                                * weight.data[c_out]
                            )

        grad_input: Optional[NDArray[np.float32]] = None
        if grad_input_padded is not None:
            if padding > 0:
                grad_input = grad_input_padded[:, :, padding : padding + L_in]
            else:
                grad_input = grad_input_padded

        return (
            Tensor(grad_input, requires_grad=False) if grad_input is not None else None,
            Tensor(grad_weight, requires_grad=False)
            if grad_weight is not None
            else None,
            Tensor(grad_bias, requires_grad=False) if grad_bias is not None else None,
            None,
            None,
            None,
        )


class ConvTranspose2DFunction(Function):
    """2D transposed convolution operation (deconvolution).

    This implements the transposed convolution operation, which is the gradient
    of regular convolution with respect to its input. It's commonly used for
    upsampling in generative models and segmentation networks.

    Note: Weight shape is (in_channels, out_channels, K_H, K_W) for groups=1,
    different from Conv2D's (out_channels, in_channels, K_H, K_W).
    """

    @staticmethod
    def _forward_loops(
        input_data: NDArray[np.float32],
        weight_data: NDArray[np.float32],
        bias_data: Optional[NDArray[np.float32]],
        N: int,
        C_in: int,
        C_out: int,
        H_in: int,
        W_in: int,
        K_H: int,
        K_W: int,
        H_out: int,
        W_out: int,
        stride: int,
        padding: int,
        output_padding: int,
        dilation: int,
        requires_grad_input: bool,
        requires_grad_weight: bool,
        requires_grad_bias: bool,
    ) -> NDArray[np.float32]:
        """Loop-based transposed convolution implementation for fallback."""
        output: NDArray[np.float32] = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
        
        # Transposed convolution algorithm: For each output position (h_out, w_out),
        # find which input positions (h_in, w_in) contribute through each kernel element (kh, kw).
        # The relationship is: h_in = (h_out + padding - dilation * kh) / stride
        # Must be integer division with exact match.
        for n in range(N):
            for c_out in range(C_out):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        total = 0.0
                        for kh in range(K_H):
                            for kw in range(K_W):
                                h_in_numerator = h_out + padding - dilation * kh
                                w_in_numerator = w_out + padding - dilation * kw
                                
                                if h_in_numerator % stride != 0 or w_in_numerator % stride != 0:
                                    continue
                                    
                                h_in = h_in_numerator // stride
                                w_in = w_in_numerator // stride
                                
                                if 0 <= h_in < H_in and 0 <= w_in < W_in:
                                    for c_in in range(C_in):
                                        total += (
                                            input_data[n, c_in, h_in, w_in] *
                                            weight_data[c_in, c_out, kh, kw]
                                        )
                        
                        output[n, c_out, h_out, w_out] = total
        
        if bias_data is not None:
            output += bias_data.reshape(1, C_out, 1, 1)

        return output

    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ) -> Tensor:
        ctx.save_for_backward(input, weight, bias)
        ctx.save_value("stride", stride)
        ctx.save_value("padding", padding)
        ctx.save_value("output_padding", output_padding)
        ctx.save_value("dilation", dilation)
        ctx.save_value("groups", groups)

        if input.ndim != 4:
            raise ValueError(f"ConvTranspose2D expects 4D input (N, C, H, W), got {input.ndim}D")
        if weight.ndim != 4:
            raise ValueError(
                f"ConvTranspose2D expects 4D weight (C_in, C_out//groups, K_H, K_W), got {weight.ndim}D"
            )

        N, C_in, H_in, W_in = input.shape
        C_in_w, C_out_div_groups, K_H, K_W = weight.shape
        
        if groups != 1:
            raise NotImplementedError("Groups > 1 not yet implemented for ConvTranspose2D")
        
        C_out = C_out_div_groups * groups
        
        if C_in != C_in_w:
            raise ValueError(
                f"Input channels mismatch: input has {C_in}, weight expects {C_in_w}"
            )

        # Output size formula from PyTorch documentation
        def output_size(
            input_size: int, kernel_size: int, stride: int, padding: int, 
            output_padding: int, dilation: int
        ) -> int:
            kernel_size_dilated = dilation * (kernel_size - 1) + 1
            return (input_size - 1) * stride - 2 * padding + kernel_size_dilated + output_padding

        H_out = output_size(H_in, K_H, stride, padding, output_padding, dilation)
        W_out = output_size(W_in, K_W, stride, padding, output_padding, dilation)
        
        if H_out <= 0 or W_out <= 0:
            raise ValueError(
                f"Output size would be non-positive: ({H_out}, {W_out}). "
                f"Check input size, padding, stride, and dilation parameters."
            )

        requires_grad_input = input.requires_grad
        requires_grad_weight = weight.requires_grad
        requires_grad_bias = bias is not None and bias.requires_grad

        # Use loop-based implementation for now
        # TODO: Add vectorized implementation similar to Conv2DFunction
        output_data = ConvTranspose2DFunction._forward_loops(
            input_data=input.data,
            weight_data=weight.data,
            bias_data=bias.data if bias is not None else None,
            N=N,
            C_in=C_in,
            C_out=C_out,
            H_in=H_in,
            W_in=W_in,
            K_H=K_H,
            K_W=K_W,
            H_out=H_out,
            W_out=W_out,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            requires_grad_input=requires_grad_input,
            requires_grad_weight=requires_grad_weight,
            requires_grad_bias=requires_grad_bias,
        )

        requires_grad = (
            requires_grad_input
            or requires_grad_weight
            or (bias is not None and requires_grad_bias)
        )
        return Tensor(output_data, requires_grad=requires_grad)

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: Any,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], None, None, None, None, None]:
        grad_output = grad_outputs[0]
        input, weight, bias = ctx.saved_tensors
        stride = ctx.get_value("stride")
        padding = ctx.get_value("padding")
        dilation = ctx.get_value("dilation")
        groups = ctx.get_value("groups")
        
        N, C_in, H_in, W_in = input.shape
        C_in_w, C_out_div_groups, K_H, K_W = weight.shape
        C_out = C_out_div_groups * groups
        _, _, H_out, W_out = grad_output.shape
        
        grad_weight_data = np.zeros_like(weight.data) if weight.requires_grad else None
        grad_bias_data = np.zeros_like(bias.data) if bias is not None and bias.requires_grad else None
        grad_input_data = np.zeros_like(input.data) if input.requires_grad else None
        
        grad_output_data = grad_output.data
        
        if grad_bias_data is not None:
            grad_bias_data = grad_output_data.sum(axis=(0, 2, 3)).reshape(C_out, 1, 1)
        
        # Loop-based backward implementation
        # Based on the forward formula:
        # output[n, c_out, h_out, w_out] += 
        #   input[n, c_in, h_in, w_in] * weight[c_in, c_out, kh, kw]
        # where h_in = (h_out + padding - dilation*kh) / stride (must be integer)
        
        for n in range(N):
            for c_out in range(C_out):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        grad_val = grad_output_data[n, c_out, h_out, w_out]
                        
                        for kh in range(K_H):
                            for kw in range(K_W):
                                h_in_numerator = h_out + padding - dilation * kh
                                w_in_numerator = w_out + padding - dilation * kw
                                
                                if h_in_numerator % stride != 0 or w_in_numerator % stride != 0:
                                    continue
                                    
                                h_in = h_in_numerator // stride
                                w_in = w_in_numerator // stride
                                
                                if 0 <= h_in < H_in and 0 <= w_in < W_in:
                                    for c_in in range(C_in):
                                        if grad_input_data is not None:
                                            grad_input_data[n, c_in, h_in, w_in] += (
                                                grad_val * weight.data[c_in, c_out, kh, kw]
                                            )
                                        
                                        if grad_weight_data is not None:
                                            grad_weight_data[c_in, c_out, kh, kw] += (
                                                grad_val * input.data[n, c_in, h_in, w_in]
                                            )
        
        grad_input = Tensor(grad_input_data, requires_grad=False) if grad_input_data is not None else None
        grad_weight = Tensor(grad_weight_data, requires_grad=False) if grad_weight_data is not None else None
        grad_bias = Tensor(grad_bias_data, requires_grad=False) if grad_bias_data is not None else None
        
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class Conv3DFunction(Function):
    """3D convolution operation with forward and backward passes.

    Performs 3D convolution on 5D input tensors, supporting depth, height, and width
    dimensions. Useful for volumetric data such as medical imaging (CT/MRI) or video
    data where time is treated as an additional spatial dimension.

    Input shape: (N, C_in, D_in, H_in, W_in)
    Weight shape: (C_out, C_in, K_D, K_H, K_W)
    Bias shape: (C_out, 1, 1, 1) (optional)
    Output shape: (N, C_out, D_out, H_out, W_out)

    NOTE ON NUMERICAL PRECISION: Gradient checks using finite differences with float32
    tensors may show relatively high errors (up to ~1e-2 for input gradients, ~3e-2 for
    weight gradients) due to float32 precision limitations in finite difference
    computations. The analytic gradient formulas are mathematically correct; these
    thresholds are acceptable for educational purposes.
    """

    @staticmethod
    def _forward_loops(
        input_data: NDArray[np.float32],
        weight_data: NDArray[np.float32],
        bias_data: Optional[NDArray[np.float32]],
        N: int,
        C_in: int,
        C_out: int,
        D_in: int,
        H_in: int,
        W_in: int,
        K_D: int,
        K_H: int,
        K_W: int,
        D_out: int,
        H_out: int,
        W_out: int,
        stride_d: int,
        stride_h: int,
        stride_w: int,
        padding_d: int,
        padding_h: int,
        padding_w: int,
        dilation_d: int,
        dilation_h: int,
        dilation_w: int,
        requires_grad_input: bool,
        requires_grad_weight: bool,
        requires_grad_bias: bool,
    ) -> NDArray[np.float32]:
        """Loop-based convolution implementation for fallback."""
        output: NDArray[np.float32] = np.zeros((N, C_out, D_out, H_out, W_out), dtype=np.float32)

        for n in range(N):
            for c_out in range(C_out):
                for d_out in range(D_out):
                    for h_out in range(H_out):
                        for w_out in range(W_out):
                            d_start = d_out * stride_d
                            h_start = h_out * stride_h
                            w_start = w_out * stride_w
                            if dilation_d == 1 and dilation_h == 1 and dilation_w == 1:
                                d_end = d_start + K_D
                                h_end = h_start + K_H
                                w_end = w_start + K_W
                                window = input_data[n, :, d_start:d_end, h_start:h_end, w_start:w_end]
                            else:
                                d_end = d_start + dilation_d * (K_D - 1) + 1
                                h_end = h_start + dilation_h * (K_H - 1) + 1
                                w_end = w_start + dilation_w * (K_W - 1) + 1
                                window = input_data[n, :, d_start:d_end:dilation_d, h_start:h_end:dilation_h, w_start:w_end:dilation_w]
                            weight_slice = weight_data[c_out]
                            conv_sum = np.sum(window * weight_slice)
                            output[n, c_out, d_out, h_out, w_out] = conv_sum

        if bias_data is not None:
            output += bias_data.reshape(1, C_out, 1, 1, 1)

        return output

    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0),
        dilation: Tuple[int, int, int] = (1, 1, 1),
    ) -> Tensor:
        ctx.save_for_backward(input, weight, bias)
        ctx.save_value("stride", stride)
        ctx.save_value("padding", padding)
        ctx.save_value("dilation", dilation)

        if input.ndim != 5:
            raise ValueError(f"Conv3D expects 5D input (N, C, D, H, W), got {input.ndim}D")
        if weight.ndim != 5:
            raise ValueError(
                f"Conv3D expects 5D weight (C_out, C_in, K_D, K_H, K_W), got {weight.ndim}D"
            )

        N, C_in, D_in, H_in, W_in = input.shape
        C_out, C_in_w, K_D, K_H, K_W = weight.shape

        if C_in != C_in_w:
            raise ValueError(
                f"Input channels mismatch: input has {C_in}, weight expects {C_in_w}"
            )

        def output_size(
            input_size: int, kernel_size: int, stride: int, padding: int, dilation: int
        ) -> int:
            kernel_size_dilated = dilation * (kernel_size - 1) + 1
            return (input_size + 2 * padding - kernel_size_dilated) // stride + 1

        # Unpack tuple parameters for per-dimension calculations
        stride_d, stride_h, stride_w = stride
        padding_d, padding_h, padding_w = padding
        dilation_d, dilation_h, dilation_w = dilation

        D_out = output_size(D_in, K_D, stride_d, padding_d, dilation_d)
        H_out = output_size(H_in, K_H, stride_h, padding_h, dilation_h)
        W_out = output_size(W_in, K_W, stride_w, padding_w, dilation_w)

        # Check if any padding is needed (per-dimension)
        if padding_d > 0 or padding_h > 0 or padding_w > 0:
            padded_input: NDArray[np.float32] = np.zeros(
                (N, C_in, D_in + 2 * padding_d, H_in + 2 * padding_h, W_in + 2 * padding_w), dtype=np.float32
            )
            padded_input[
                :, :, padding_d : padding_d + D_in, padding_h : padding_h + H_in, padding_w : padding_w + W_in
            ] = input.data
            input_data = padded_input
        else:
            input_data = input.data

        requires_grad_input = input.requires_grad
        requires_grad_weight = weight.requires_grad
        requires_grad_bias = bias is not None and bias.requires_grad

        # Check if any dilation is not 1 (per-dimension)
        if dilation_d != 1 or dilation_h != 1 or dilation_w != 1:
            output_data = Conv3DFunction._forward_loops(
                input_data=input_data,
                weight_data=weight.data,
                bias_data=bias.data if bias is not None else None,
                N=N,
                C_in=C_in,
                C_out=C_out,
                D_in=D_in,
                H_in=H_in,
                W_in=W_in,
                K_D=K_D,
                K_H=K_H,
                K_W=K_W,
                D_out=D_out,
                H_out=H_out,
                W_out=W_out,
                stride_d=stride_d,
                stride_h=stride_h,
                stride_w=stride_w,
                padding_d=padding_d,
                padding_h=padding_h,
                padding_w=padding_w,
                dilation_d=dilation_d,
                dilation_h=dilation_h,
                dilation_w=dilation_w,
                requires_grad_input=requires_grad_input,
                requires_grad_weight=requires_grad_weight,
                requires_grad_bias=requires_grad_bias,
            )

            requires_grad = (
                requires_grad_input
                or requires_grad_weight
                or (bias is not None and requires_grad_bias)
            )
            return Tensor(output_data, requires_grad=requires_grad)

        try:
            from numpy.lib.stride_tricks import sliding_window_view
        except ImportError:
            return Conv3DFunction._forward_loops(
                input_data=input_data,
                weight_data=weight.data,
                bias_data=bias.data if bias is not None else None,
                N=N,
                C_in=C_in,
                C_out=C_out,
                D_in=D_in,
                H_in=H_in,
                W_in=W_in,
                K_D=K_D,
                K_H=K_H,
                K_W=K_W,
                D_out=D_out,
                H_out=H_out,
                W_out=W_out,
                stride_d=stride_d,
                stride_h=stride_h,
                stride_w=stride_w,
                padding_d=padding_d,
                padding_h=padding_h,
                padding_w=padding_w,
                dilation_d=dilation_d,
                dilation_h=dilation_h,
                dilation_w=dilation_w,
                requires_grad_input=requires_grad_input,
                requires_grad_weight=requires_grad_weight,
                requires_grad_bias=requires_grad_bias,
            )

        windows = sliding_window_view(input_data, (K_D, K_H, K_W), axis=(-3, -2, -1))
        windows = windows[:, :, ::stride_d, ::stride_h, ::stride_w, :, :, :]
        windows_flat = windows.transpose(0, 2, 3, 4, 1, 5, 6, 7).reshape(N * D_out * H_out * W_out, C_in * K_D * K_H * K_W)
        weight_flat = weight.data.reshape(C_out, -1)
        output_flat = windows_flat @ weight_flat.T
        output = output_flat.T.reshape(C_out, N, D_out, H_out, W_out).transpose(1, 0, 2, 3, 4)

        if bias is not None:
            output += bias.data.reshape(1, C_out, 1, 1, 1)

        requires_grad = (
            requires_grad_input
            or requires_grad_weight
            or (bias is not None and requires_grad_bias)
        )
        return Tensor(output, requires_grad=requires_grad)

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: Any,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], None, None, None]:
        grad_output = grad_outputs[0]
        input, weight, bias = ctx.saved_tensors
        stride = ctx.get_value("stride")
        padding = ctx.get_value("padding")
        dilation = ctx.get_value("dilation")

        # Unpack tuple parameters
        stride_d, stride_h, stride_w = stride
        padding_d, padding_h, padding_w = padding
        dilation_d, dilation_h, dilation_w = dilation

        N, C_in, D_in, H_in, W_in = input.shape
        C_out, C_in_w, K_D, K_H, K_W = weight.shape
        _, _, D_out, H_out, W_out = grad_output.shape

        grad_weight: Optional[NDArray[np.float32]] = (
            np.zeros_like(weight.data) if weight.requires_grad else None
        )
        grad_bias: Optional[NDArray[np.float32]] = (
            np.zeros_like(bias.data)
            if bias is not None and bias.requires_grad
            else None
        )

        if input.requires_grad:
            grad_input_padded: Optional[NDArray[np.float32]] = np.zeros(
                (N, C_in, D_in + 2 * padding_d, H_in + 2 * padding_h, W_in + 2 * padding_w), dtype=np.float32
            )
        else:
            grad_input_padded = None

        grad_output_data = grad_output.data

        if grad_bias is not None:
            grad_bias = grad_output_data.sum(axis=(0, 2, 3, 4)).reshape(C_out, 1, 1, 1)

        # Check if all dilations are 1
        if dilation_d == 1 and dilation_h == 1 and dilation_w == 1:
            if grad_weight is not None:
                if padding_d > 0 or padding_h > 0 or padding_w > 0:
                    padded_input: NDArray[np.float32] = np.zeros(
                        (N, C_in, D_in + 2 * padding_d, H_in + 2 * padding_h, W_in + 2 * padding_w), dtype=np.float32
                    )
                    padded_input[
                        :, :, padding_d : padding_d + D_in, padding_h : padding_h + H_in, padding_w : padding_w + W_in
                    ] = input.data
                    input_data = padded_input
                else:
                    input_data = input.data

                from numpy.lib.stride_tricks import sliding_window_view
                windows = sliding_window_view(input_data, (K_D, K_H, K_W), axis=(-3, -2, -1))
                windows = windows[:, :, ::stride_d, ::stride_h, ::stride_w, :, :, :]
                windows_flat = windows.transpose(0, 2, 3, 4, 1, 5, 6, 7).reshape(N * D_out * H_out * W_out, C_in * K_D * K_H * K_W)
                grad_output_flat = grad_output_data.transpose(0, 2, 3, 4, 1).reshape(N * D_out * H_out * W_out, C_out)
                grad_weight_flat = grad_output_flat.T @ windows_flat
                grad_weight = grad_weight_flat.reshape(C_out, C_in, K_D, K_H, K_W)

            if grad_input_padded is not None:
                # Vectorized input gradient computation for dilation=1
                # Flatten weight: (C_out, C_in, K_D, K_H, K_W) -> (C_out, C_in*K_D*K_H*K_W)
                weight_flat = weight.data.reshape(C_out, -1)

                # Reshape grad_output: (N, C_out, D_out, H_out, W_out) -> (N*D_out*H_out*W_out, C_out)
                grad_output_flat = grad_output_data.transpose(0, 2, 3, 4, 1).reshape(N * D_out * H_out * W_out, C_out)

                # Compute contributions for each patch: (N*D_out*H_out*W_out, C_in*K_D*K_H*K_W)
                contributions_flat = grad_output_flat @ weight_flat

                # Reshape contributions: (N, D_out, H_out, W_out, C_in, K_D, K_H, K_W)
                contributions = contributions_flat.reshape(N, D_out, H_out, W_out, C_in, K_D, K_H, K_W)

                # Loop over kernel positions (small loops: K_D * K_H * K_W, typically 3x3x3=27)
                for kd in range(K_D):
                    for kh in range(K_H):
                        for kw in range(K_W):
                            # Contributions for this kernel position: (N, D_out, H_out, W_out, C_in)
                            contrib_slice = contributions[:, :, :, :, :, kd, kh, kw]
                            # Transpose to (N, C_in, D_out, H_out, W_out)
                            contrib_slice_t = contrib_slice.transpose(0, 4, 1, 2, 3)

                            # Add to grad_input_padded at strided positions
                            d_start = kd
                            h_start = kh
                            w_start = kw
                            grad_input_padded[:, :, d_start:d_start + D_out * stride_d:stride_d, h_start:h_start + H_out * stride_h:stride_h, w_start:w_start + W_out * stride_w:stride_w] += contrib_slice_t
        else:
            if padding_d > 0 or padding_h > 0 or padding_w > 0:
                padded_input: NDArray[np.float32] = np.zeros(
                    (N, C_in, D_in + 2 * padding_d, H_in + 2 * padding_h, W_in + 2 * padding_w), dtype=np.float32
                )
                padded_input[
                    :, :, padding_d : padding_d + D_in, padding_h : padding_h + H_in, padding_w : padding_w + W_in
                ] = input.data
                input_data = padded_input
            else:
                input_data = input.data

            for n in range(N):
                for c_out in range(C_out):
                    for d_out in range(D_out):
                        for h_out in range(H_out):
                            for w_out in range(W_out):
                                d_start = d_out * stride_d
                                h_start = h_out * stride_h
                                w_start = w_out * stride_w
                                d_end = d_start + dilation_d * (K_D - 1) + 1
                                h_end = h_start + dilation_h * (K_H - 1) + 1
                                w_end = w_start + dilation_w * (K_W - 1) + 1

                                if grad_weight is not None:
                                    window = input_data[n, :, d_start:d_end:dilation_d, h_start:h_end:dilation_h, w_start:w_end:dilation_w]
                                    grad_weight[c_out] += (
                                        grad_output_data[n, c_out, d_out, h_out, w_out] * window
                                    )

                                if grad_input_padded is not None:
                                    grad_input_padded[n, :, d_start:d_end:dilation_d, h_start:h_end:dilation_h, w_start:w_end:dilation_w] += (
                                        grad_output_data[n, c_out, d_out, h_out, w_out]
                                        * weight.data[c_out]
                                    )

        grad_input: Optional[NDArray[np.float32]] = None
        if grad_input_padded is not None:
            if padding_d > 0 or padding_h > 0 or padding_w > 0:
                grad_input = grad_input_padded[
                    :, :, padding_d : padding_d + D_in, padding_h : padding_h + H_in, padding_w : padding_w + W_in
                ]
            else:
                grad_input = grad_input_padded

        return (
            Tensor(grad_input, requires_grad=False) if grad_input is not None else None,
            Tensor(grad_weight, requires_grad=False)
            if grad_weight is not None
            else None,
            Tensor(grad_bias, requires_grad=False) if grad_bias is not None else None,
            None,
            None,
            None,
        )


class ConvTranspose3DFunction(Function):
    """3D transposed convolution operation.

    This implements the transposed convolution operation for 3D data, which is the
    gradient of regular convolution with respect to its input.

    Input shape: (N, C_in, D_in, H_in, W_in)
    Weight shape: (C_in, C_out, K_D, K_H, K_W) for groups=1
    Bias shape: (C_out, 1, 1, 1) (optional)
    Output shape: (N, C_out, D_out, H_out, W_out)
    """

    @staticmethod
    def _forward_loops(
        input_data: NDArray[np.float32],
        weight_data: NDArray[np.float32],
        bias_data: Optional[NDArray[np.float32]],
        N: int,
        C_in: int,
        C_out: int,
        D_in: int,
        H_in: int,
        W_in: int,
        K_D: int,
        K_H: int,
        K_W: int,
        D_out: int,
        H_out: int,
        W_out: int,
        stride_d: int,
        stride_h: int,
        stride_w: int,
        padding_d: int,
        padding_h: int,
        padding_w: int,
        output_padding_d: int,
        output_padding_h: int,
        output_padding_w: int,
        dilation_d: int,
        dilation_h: int,
        dilation_w: int,
    ) -> NDArray[np.float32]:
        output: NDArray[np.float32] = np.zeros((N, C_out, D_out, H_out, W_out), dtype=np.float32)

        for n in range(N):
            for c_out in range(C_out):
                for d_out in range(D_out):
                    for h_out in range(H_out):
                        for w_out in range(W_out):
                            total = 0.0
                            for kd in range(K_D):
                                for kh in range(K_H):
                                    for kw in range(K_W):
                                        d_in_numerator = d_out + padding_d - dilation_d * kd
                                        h_in_numerator = h_out + padding_h - dilation_h * kh
                                        w_in_numerator = w_out + padding_w - dilation_w * kw

                                        if (d_in_numerator % stride_d != 0 or
                                            h_in_numerator % stride_h != 0 or
                                            w_in_numerator % stride_w != 0):
                                            continue

                                        d_in = d_in_numerator // stride_d
                                        h_in = h_in_numerator // stride_h
                                        w_in = w_in_numerator // stride_w

                                        if 0 <= d_in < D_in and 0 <= h_in < H_in and 0 <= w_in < W_in:
                                            for c_in in range(C_in):
                                                total += (
                                                    input_data[n, c_in, d_in, h_in, w_in] *
                                                    weight_data[c_in, c_out, kd, kh, kw]
                                                )

                            output[n, c_out, d_out, h_out, w_out] = total

        if bias_data is not None:
            output += bias_data.reshape(1, C_out, 1, 1, 1)

        return output

    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0),
        output_padding: Tuple[int, int, int] = (0, 0, 0),
        dilation: Tuple[int, int, int] = (1, 1, 1),
        groups: int = 1,
    ) -> Tensor:
        ctx.save_for_backward(input, weight, bias)
        ctx.save_value("stride", stride)
        ctx.save_value("padding", padding)
        ctx.save_value("output_padding", output_padding)
        ctx.save_value("dilation", dilation)
        ctx.save_value("groups", groups)

        if input.ndim != 5:
            raise ValueError(f"ConvTranspose3D expects 5D input (N, C, D, H, W), got {input.ndim}D")
        if weight.ndim != 5:
            raise ValueError(
                f"ConvTranspose3D expects 5D weight (C_in, C_out//groups, K_D, K_H, K_W), got {weight.ndim}D"
            )

        N, C_in, D_in, H_in, W_in = input.shape
        C_in_w, C_out_div_groups, K_D, K_H, K_W = weight.shape

        if groups != 1:
            raise NotImplementedError("Groups > 1 not yet implemented for ConvTranspose3D")

        C_out = C_out_div_groups * groups

        if C_in != C_in_w:
            raise ValueError(
                f"Input channels mismatch: input has {C_in}, weight expects {C_in_w}"
            )

        stride_d, stride_h, stride_w = stride
        padding_d, padding_h, padding_w = padding
        output_padding_d, output_padding_h, output_padding_w = output_padding
        dilation_d, dilation_h, dilation_w = dilation

        def output_size(
            input_size: int, kernel_size: int, stride: int, padding: int,
            output_padding: int, dilation: int
        ) -> int:
            kernel_size_dilated = dilation * (kernel_size - 1) + 1
            return (input_size - 1) * stride - 2 * padding + kernel_size_dilated + output_padding

        D_out = output_size(D_in, K_D, stride_d, padding_d, output_padding_d, dilation_d)
        H_out = output_size(H_in, K_H, stride_h, padding_h, output_padding_h, dilation_h)
        W_out = output_size(W_in, K_W, stride_w, padding_w, output_padding_w, dilation_w)

        if D_out <= 0 or H_out <= 0 or W_out <= 0:
            raise ValueError(
                f"Output size would be non-positive: ({D_out}, {H_out}, {W_out}). "
                f"Check input size, padding, stride, and dilation parameters."
            )

        requires_grad_input = input.requires_grad
        requires_grad_weight = weight.requires_grad
        requires_grad_bias = bias is not None and bias.requires_grad

        output_data = ConvTranspose3DFunction._forward_loops(
            input_data=input.data,
            weight_data=weight.data,
            bias_data=bias.data if bias is not None else None,
            N=N,
            C_in=C_in,
            C_out=C_out,
            D_in=D_in,
            H_in=H_in,
            W_in=W_in,
            K_D=K_D,
            K_H=K_H,
            K_W=K_W,
            D_out=D_out,
            H_out=H_out,
            W_out=W_out,
            stride_d=stride_d,
            stride_h=stride_h,
            stride_w=stride_w,
            padding_d=padding_d,
            padding_h=padding_h,
            padding_w=padding_w,
            output_padding_d=output_padding_d,
            output_padding_h=output_padding_h,
            output_padding_w=output_padding_w,
            dilation_d=dilation_d,
            dilation_h=dilation_h,
            dilation_w=dilation_w,
        )

        requires_grad = (
            requires_grad_input
            or requires_grad_weight
            or (bias is not None and requires_grad_bias)
        )
        return Tensor(output_data, requires_grad=requires_grad)

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: Any,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], None, None, None, None, None]:
        grad_output = grad_outputs[0]
        input, weight, bias = ctx.saved_tensors
        stride = ctx.get_value("stride")
        padding = ctx.get_value("padding")
        dilation = ctx.get_value("dilation")
        groups = ctx.get_value("groups")

        stride_d, stride_h, stride_w = stride
        padding_d, padding_h, padding_w = padding
        dilation_d, dilation_h, dilation_w = dilation

        N, C_in, D_in, H_in, W_in = input.shape
        C_in_w, C_out_div_groups, K_D, K_H, K_W = weight.shape
        C_out = C_out_div_groups * groups
        _, _, D_out, H_out, W_out = grad_output.shape

        grad_weight_data = np.zeros_like(weight.data) if weight.requires_grad else None
        grad_bias_data = np.zeros_like(bias.data) if bias is not None and bias.requires_grad else None
        grad_input_data = np.zeros_like(input.data) if input.requires_grad else None

        grad_output_data = grad_output.data

        if grad_bias_data is not None:
            grad_bias_data = grad_output_data.sum(axis=(0, 2, 3, 4)).reshape(C_out, 1, 1, 1)

        for n in range(N):
            for c_out in range(C_out):
                for d_out in range(D_out):
                    for h_out in range(H_out):
                        for w_out in range(W_out):
                            grad_val = grad_output_data[n, c_out, d_out, h_out, w_out]

                            for kd in range(K_D):
                                for kh in range(K_H):
                                    for kw in range(K_W):
                                        d_in_numerator = d_out + padding_d - dilation_d * kd
                                        h_in_numerator = h_out + padding_h - dilation_h * kh
                                        w_in_numerator = w_out + padding_w - dilation_w * kw

                                        if (d_in_numerator % stride_d != 0 or
                                            h_in_numerator % stride_h != 0 or
                                            w_in_numerator % stride_w != 0):
                                            continue

                                        d_in = d_in_numerator // stride_d
                                        h_in = h_in_numerator // stride_h
                                        w_in = w_in_numerator // stride_w

                                        if 0 <= d_in < D_in and 0 <= h_in < H_in and 0 <= w_in < W_in:
                                            for c_in in range(C_in):
                                                if grad_input_data is not None:
                                                    grad_input_data[n, c_in, d_in, h_in, w_in] += (
                                                        grad_val * weight.data[c_in, c_out, kd, kh, kw]
                                                    )

                                                if grad_weight_data is not None:
                                                    grad_weight_data[c_in, c_out, kd, kh, kw] += (
                                                        grad_val * input.data[n, c_in, d_in, h_in, w_in]
                                                    )

        grad_input = Tensor(grad_input_data, requires_grad=False) if grad_input_data is not None else None
        grad_weight = Tensor(grad_weight_data, requires_grad=False) if grad_weight_data is not None else None
        grad_bias = Tensor(grad_bias_data, requires_grad=False) if grad_bias_data is not None else None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class CatFunction(Function):
    """Concatenation of tensors along a dimension."""

    @staticmethod
    def forward(ctx: Any, *tensors: Tensor, dim: int = 0) -> Tensor:
        ctx.save_for_backward(*tensors)
        ctx.save_value("dim", dim)
        arrays = [t.data for t in tensors]
        result_data = np.concatenate(arrays, axis=dim)
        requires_grad = any(t.requires_grad for t in tensors)
        return Tensor(result_data, requires_grad=requires_grad)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        tensors = ctx.saved_tensors
        dim = ctx.get_value("dim")
        split_indices = np.cumsum([t.shape[dim] for t in tensors[:-1]])
        grad_arrays = np.split(grad_output.data, split_indices, axis=dim)
        grad_inputs: List[Optional[Tensor]] = []
        for i, t in enumerate(tensors):
            if t.requires_grad:
                grad_inputs.append(Tensor(grad_arrays[i], requires_grad=False))
            else:
                grad_inputs.append(None)
        grad_inputs.append(None)
        return tuple(grad_inputs)


class StackFunction(Function):
    """Stack tensors along a new dimension."""

    @staticmethod
    def forward(ctx: Any, *tensors: Tensor, dim: int = 0) -> Tensor:
        ctx.save_for_backward(*tensors)
        ctx.save_value("dim", dim)
        arrays = [t.data for t in tensors]
        result_data = np.stack(arrays, axis=dim)
        requires_grad = any(t.requires_grad for t in tensors)
        return Tensor(result_data, requires_grad=requires_grad)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        tensors = ctx.saved_tensors
        dim = ctx.get_value("dim")
        grad_arrays = np.split(grad_output.data, len(tensors), axis=dim)
        grad_arrays = [np.squeeze(g, axis=dim) for g in grad_arrays]
        grad_inputs: List[Optional[Tensor]] = []
        for i, t in enumerate(tensors):
            if t.requires_grad:
                grad_inputs.append(Tensor(grad_arrays[i], requires_grad=False))
            else:
                grad_inputs.append(None)
        return tuple(grad_inputs)


class MaxPool2dFunction(Function):
    """2D max pooling operation.

    Applies a 2D max pooling over an input signal composed of several input planes.

    Args:
        kernel_size: Size of the pooling window. Can be a single number or tuple.
        stride: Stride of the pooling. Default: kernel_size.
        padding: Zero-padding added to both sides of the input. Default: 0.
        dilation: Spacing between kernel elements. Default: 1.
        ceil_mode: If True, will use ceil instead of floor to compute output shape. Default: False.
        return_indices: If True, will return the indices along with the outputs. Default: False.

    Shape:
        - Input: (N, C, H_in, W_in)
        - Output: (N, C, H_out, W_out)

        where (when ceil_mode=False):
            H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
            W_out = floor((W_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
    """

    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        kernel_size: Union[int, Tuple[int, int]] = (2, 2),
        stride: Union[int, Tuple[int, int]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, NDArray]]:
        # Normalize parameters to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        K_H, K_W = kernel_size
        stride_h, stride_w = stride
        pad_h, pad_w = padding
        dilation_h, dilation_w = dilation

        # Validate input
        if input.ndim != 4:
            raise ValueError(f"MaxPool2d expects 4D input (N, C, H, W), got {input.ndim}D")

        N, C, H_in, W_in = input.shape

        # Compute output size
        def output_size(input_size: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
            kernel_size_dilated = dilation * (kernel_size - 1) + 1
            if ceil_mode:
                return int(np.ceil((input_size + 2 * padding - kernel_size_dilated) / stride + 1))
            else:
                return (input_size + 2 * padding - kernel_size_dilated) // stride + 1

        H_out = output_size(H_in, K_H, stride_h, pad_h, dilation_h)
        W_out = output_size(W_in, K_W, stride_w, pad_w, dilation_w)

        if H_out <= 0 or W_out <= 0:
            raise ValueError(
                f"Output size would be non-positive: ({H_out}, {W_out}). "
                f"Check input size, padding, kernel_size, and dilation parameters."
            )

        # Pad input if needed
        if pad_h > 0 or pad_w > 0:
            padded_input: NDArray[np.float32] = np.zeros(
                (N, C, H_in + 2 * pad_h, W_in + 2 * pad_w), dtype=np.float32
            )
            padded_input[:, :, pad_h : pad_h + H_in, pad_w : pad_w + W_in] = input.data
            input_data = padded_input
        else:
            input_data = input.data

        # Prepare output and indices
        output_data: NDArray[np.float32] = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        indices: NDArray[np.int32] = np.zeros((N, C, H_out, W_out), dtype=np.int32)

        # Perform max pooling with index tracking
        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride_h
                        w_start = w_out * stride_w

                        # Calculate actual window considering dilation
                        h_end = h_start + dilation_h * (K_H - 1) + 1
                        w_end = w_start + dilation_w * (K_W - 1) + 1

                        # Extract window with dilation
                        window = input_data[n, c, h_start:h_end:dilation_h, w_start:w_end:dilation_w]

                        # Find max and its index
                        max_val = np.max(window)
                        output_data[n, c, h_out, w_out] = max_val

                        # Find index in flattened window, then convert to absolute position
                        flat_idx = np.argmax(window)
                        h_offset, w_offset = np.unravel_index(flat_idx, (K_H, K_W))
                        abs_h = h_start + dilation_h * h_offset
                        abs_w = w_start + dilation_w * w_offset

                        # Store flattened absolute index for backward pass
                        # Store as: n * (C*H*W) + c * (H*W) + h * W + w
                        total_h = H_in + 2 * pad_h
                        total_w = W_in + 2 * pad_w
                        indices[n, c, h_out, w_out] = n * (C * total_h * total_w) + c * (total_h * total_w) + abs_h * total_w + abs_w

        # Save context for backward pass
        ctx.save_for_backward(input)
        ctx.save_value("kernel_size", kernel_size)
        ctx.save_value("stride", stride)
        ctx.save_value("padding", padding)
        ctx.save_value("dilation", dilation)
        ctx.save_value("indices", indices)
        ctx.save_value("ceil_mode", ceil_mode)
        ctx.save_value("H_out", H_out)
        ctx.save_value("W_out", W_out)

        requires_grad = input.requires_grad
        output = Tensor(output_data, requires_grad=requires_grad)

        if return_indices:
            return output, indices
        else:
            return output

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Any
    ) -> Tuple[Optional[Tensor], None, None, None, None, None, None]:
        grad_output = grad_outputs[0]
        input = ctx.saved_tensors[0]
        kernel_size = ctx.get_value("kernel_size")
        stride = ctx.get_value("stride")
        padding = ctx.get_value("padding")
        indices = ctx.get_value("indices")
        H_out = ctx.get_value("H_out")
        W_out = ctx.get_value("W_out")

        K_H, K_W = kernel_size
        stride_h, stride_w = stride
        pad_h, pad_w = padding

        N, C, H_in, W_in = input.shape
        total_h = H_in + 2 * pad_h
        total_w = W_in + 2 * pad_w

        # Initialize gradient
        grad_input: Optional[NDArray[np.float32]] = np.zeros(
            (N, C, total_h, total_w), dtype=np.float32
        )

        # Backward pass: distribute gradient to max positions
        grad_output_data = grad_output.data
        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        idx = indices[n, c, h_out, w_out]
                        # Unpack flat index
                        n_idx = idx // (C * total_h * total_w)
                        c_idx = (idx % (C * total_h * total_w)) // (total_h * total_w)
                        h_idx = ((idx % (C * total_h * total_w)) % (total_h * total_w)) // total_w
                        w_idx = ((idx % (C * total_h * total_w)) % (total_h * total_w)) % total_w

                        grad_input[n_idx, c_idx, h_idx, w_idx] += grad_output_data[n, c, h_out, w_out]

        # Remove padding from gradient
        if pad_h > 0 or pad_w > 0:
            grad_input_unpadded = grad_input[:, :, pad_h : pad_h + H_in, pad_w : pad_w + W_in]
        else:
            grad_input_unpadded = grad_input

        return (
            Tensor(grad_input_unpadded, requires_grad=False),
            None,
            None,
            None,
            None,
            None,
            None,
        )


class AvgPool2dFunction(Function):
    """2D average pooling operation.

    Applies a 2D average pooling over an input signal composed of several input planes.

    Args:
        kernel_size: Size of the pooling window. Can be a single number or tuple.
        stride: Stride of the pooling. Default: kernel_size.
        padding: Zero-padding added to both sides of the input. Default: 0.
        ceil_mode: If True, will use ceil instead of floor to compute output shape. Default: False.
        count_include_pad: If True, will include zero-padding in averaging. Default: True.
        divisor_override: If specified, it will be used as divisor, otherwise kernel_size will be used.

    Shape:
        - Input: (N, C, H_in, W_in)
        - Output: (N, C, H_out, W_out)

        where (when ceil_mode=False):
            H_out = floor((H_in + 2*padding - (kernel_size-1) - 1) / stride + 1)
            W_out = floor((W_in + 2*padding - (kernel_size-1) - 1) / stride + 1)
    """

    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        kernel_size: Union[int, Tuple[int, int]] = (2, 2),
        stride: Union[int, Tuple[int, int]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
    ) -> Tensor:
        # Normalize parameters to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        K_H, K_W = kernel_size
        stride_h, stride_w = stride
        pad_h, pad_w = padding

        # Validate input
        if input.ndim != 4:
            raise ValueError(f"AvgPool2d expects 4D input (N, C, H, W), got {input.ndim}D")

        N, C, H_in, W_in = input.shape

        # Compute output size
        def output_size(input_size: int, kernel_size: int, stride: int, padding: int) -> int:
            if ceil_mode:
                return int(np.ceil((input_size + 2 * padding - kernel_size) / stride + 1))
            else:
                return (input_size + 2 * padding - kernel_size) // stride + 1

        H_out = output_size(H_in, K_H, stride_h, pad_h)
        W_out = output_size(W_in, K_W, stride_w, pad_w)

        if H_out <= 0 or W_out <= 0:
            raise ValueError(
                f"Output size would be non-positive: ({H_out}, {W_out}). "
                f"Check input size, padding, and kernel_size parameters."
            )

        # Pad input if needed
        if pad_h > 0 or pad_w > 0:
            padded_input: NDArray[np.float32] = np.zeros(
                (N, C, H_in + 2 * pad_h, W_in + 2 * pad_w), dtype=np.float32
            )
            padded_input[:, :, pad_h : pad_h + H_in, pad_w : pad_w + W_in] = input.data
            input_data = padded_input
        else:
            input_data = input.data

        # Prepare output
        output_data: NDArray[np.float32] = np.zeros((N, C, H_out, W_out), dtype=np.float32)

        # Perform average pooling
        divisor = divisor_override if divisor_override is not None else (K_H * K_W)

        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride_h
                        w_start = w_out * stride_w

                        h_end = h_start + K_H
                        w_end = w_start + K_W

                        window = input_data[n, c, h_start:h_end, w_start:w_end]

                        if count_include_pad:
                            # Include padding in divisor (simple average)
                            avg_val = np.sum(window) / divisor
                        else:
                            # Exclude padding from divisor (only count actual elements)
                            if divisor_override is not None:
                                avg_val = np.sum(window) / divisor_override
                            else:
                                # Count non-padding elements
                                actual_h = min(h_end, H_in + 2 * pad_h) - h_start
                                actual_w = min(w_end, W_in + 2 * pad_w) - w_start
                                actual_count = actual_h * actual_w
                                avg_val = np.sum(window) / actual_count

                        output_data[n, c, h_out, w_out] = avg_val

        # Save context for backward pass
        ctx.save_for_backward(input)
        ctx.save_value("kernel_size", kernel_size)
        ctx.save_value("stride", stride)
        ctx.save_value("padding", padding)
        ctx.save_value("ceil_mode", ceil_mode)
        ctx.save_value("count_include_pad", count_include_pad)
        ctx.save_value("divisor_override", divisor_override)
        ctx.save_value("H_out", H_out)
        ctx.save_value("W_out", W_out)

        requires_grad = input.requires_grad
        return Tensor(output_data, requires_grad=requires_grad)

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Any
    ) -> Tuple[Optional[Tensor], None, None, None, None, None, None]:
        grad_output = grad_outputs[0]
        input = ctx.saved_tensors[0]
        kernel_size = ctx.get_value("kernel_size")
        stride = ctx.get_value("stride")
        padding = ctx.get_value("padding")
        count_include_pad = ctx.get_value("count_include_pad")
        divisor_override = ctx.get_value("divisor_override")
        H_out = ctx.get_value("H_out")
        W_out = ctx.get_value("W_out")

        K_H, K_W = kernel_size
        stride_h, stride_w = stride
        pad_h, pad_w = padding

        N, C, H_in, W_in = input.shape
        total_h = H_in + 2 * pad_h
        total_w = W_in + 2 * pad_w

        # Initialize gradient
        grad_input: Optional[NDArray[np.float32]] = np.zeros(
            (N, C, total_h, total_w), dtype=np.float32
        )

        # Base divisor
        base_divisor = divisor_override if divisor_override is not None else (K_H * K_W)

        # Backward pass: distribute gradient evenly across window
        grad_output_data = grad_output.data
        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride_h
                        w_start = w_out * stride_w

                        h_end = h_start + K_H
                        w_end = w_start + K_W

                        if count_include_pad or divisor_override is not None:
                            # Distribute evenly across all positions
                            grad_val = grad_output_data[n, c, h_out, w_out] / base_divisor
                            grad_input[n, c, h_start:h_end, w_start:w_end] += grad_val
                        else:
                            # Distribute evenly excluding padding
                            actual_h = min(h_end, H_in + 2 * pad_h) - h_start
                            actual_w = min(w_end, W_in + 2 * pad_w) - w_start
                            actual_count = actual_h * actual_w
                            grad_val = grad_output_data[n, c, h_out, w_out] / actual_count

                            # Only distribute to actual (non-padding) positions
                            h_clip = min(h_end, total_h)
                            w_clip = min(w_end, total_w)
                            grad_input[n, c, h_start:h_clip, w_start:w_clip] += grad_val

        # Remove padding from gradient
        if pad_h > 0 or pad_w > 0:
            grad_input_unpadded = grad_input[:, :, pad_h : pad_h + H_in, pad_w : pad_w + W_in]
        else:
            grad_input_unpadded = grad_input

        return (
            Tensor(grad_input_unpadded, requires_grad=False),
            None,
            None,
            None,
            None,
            None,
            None,
        )


class AdaptiveAvgPool2dFunction(Function):
    """2D adaptive average pooling operation.

    Applies a 2D adaptive average pooling over an input signal composed of several
    input planes. The output size is specified rather than kernel size/stride.

    Args:
        output_size: Target output size. Can be a single integer (applied to both
            height and width) or a tuple (H_out, W_out).

    Shape:
        - Input: (N, C, H_in, W_in)
        - Output: (N, C, H_out, W_out)
    """

    @staticmethod
    def _start_index(a: int, b: int, c: int) -> int:
        """Calculate start index in input for output index a.
        
        PyTorch formula: start = (a // b) * c + ((a % b) * c) // b
        """
        return (a // b) * c + ((a % b) * c) // b

    @staticmethod
    def _end_index(a: int, b: int, c: int) -> int:
        """Calculate end index (exclusive) in input for output index a.
        
        PyTorch formula: end = 1 + ((a + 1) * c - 1) // b
        """
        return 1 + ((a + 1) * c - 1) // b

    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        output_size: Union[int, Tuple[int, int]] = 1,
    ) -> Tensor:
        # Normalize output_size to tuple
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        
        H_out, W_out = output_size

        # Validate input
        if input.ndim != 4:
            raise ValueError(
                f"AdaptiveAvgPool2d expects 4D input (N, C, H, W), got {input.ndim}D"
            )
        
        N, C, H_in, W_in = input.shape

        if H_out <= 0 or W_out <= 0:
            raise ValueError(
                f"output_size must be positive, got {output_size}"
            )
        
        # Prepare output
        output_data: NDArray[np.float32] = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        input_data = input.data

        # Perform adaptive average pooling
        for n in range(N):
            for c in range(C):
                for oh in range(H_out):
                    # Compute pooling region for height
                    ih0 = AdaptiveAvgPool2dFunction._start_index(oh, H_out, H_in)
                    ih1 = AdaptiveAvgPool2dFunction._end_index(oh, H_out, H_in)
                    kh = ih1 - ih0  # actual kernel height
                    
                    for ow in range(W_out):
                        # Compute pooling region for width
                        iw0 = AdaptiveAvgPool2dFunction._start_index(ow, W_out, W_in)
                        iw1 = AdaptiveAvgPool2dFunction._end_index(ow, W_out, W_in)
                        kw = iw1 - iw0  # actual kernel width
                        
                        # Extract window and compute average
                        window = input_data[n, c, ih0:ih1, iw0:iw1]
                        if kh > 0 and kw > 0:
                            output_data[n, c, oh, ow] = np.sum(window) / (kh * kw)
                        else:
                            output_data[n, c, oh, ow] = 0.0
        
        # Save context for backward pass
        ctx.save_for_backward(input)
        ctx.save_value("output_size", output_size)
        ctx.save_value("H_out", H_out)
        ctx.save_value("W_out", W_out)
        ctx.save_value("H_in", H_in)
        ctx.save_value("W_in", W_in)

        requires_grad = input.requires_grad
        return Tensor(output_data, requires_grad=requires_grad)

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Any
    ) -> Tuple[Optional[Tensor], None]:
        grad_output = grad_outputs[0]
        input = ctx.saved_tensors[0]
        output_size = ctx.get_value("output_size")
        H_out, W_out = output_size
        H_in = ctx.get_value("H_in")
        W_in = ctx.get_value("W_in")

        N, C, _, _ = input.shape
        grad_output_data = grad_output.data

        # Initialize gradient
        grad_input: NDArray[np.float32] = np.zeros(
            (N, C, H_in, W_in), dtype=np.float32
        )

        # Backward pass: distribute gradient evenly across pooling region
        for n in range(N):
            for c in range(C):
                for oh in range(H_out):
                    # Compute pooling region for height
                    ih0 = AdaptiveAvgPool2dFunction._start_index(oh, H_out, H_in)
                    ih1 = AdaptiveAvgPool2dFunction._end_index(oh, H_out, H_in)
                    kh = ih1 - ih0
                    
                    for ow in range(W_out):
                        # Compute pooling region for width
                        iw0 = AdaptiveAvgPool2dFunction._start_index(ow, W_out, W_in)
                        iw1 = AdaptiveAvgPool2dFunction._end_index(ow, W_out, W_in)
                        kw = iw1 - iw0
                        
                        if kh > 0 and kw > 0:
                            # Distribute gradient evenly across all pixels in region
                            grad_val = grad_output_data[n, c, oh, ow] / (kh * kw)
                            grad_input[n, c, ih0:ih1, iw0:iw1] += grad_val

        return (
            Tensor(grad_input, requires_grad=False),
            None,
        )


class AdaptiveMaxPool2dFunction(Function):
    """2D adaptive max pooling operation.

    Applies a 2D adaptive max pooling over an input signal composed of several
    input planes. The output size is specified rather than kernel size/stride.

    Args:
        output_size: Target output size. Can be a single integer (applied to both
            height and width) or a tuple (H_out, W_out).
        return_indices: If True, will return the indices along with the outputs.
            Default: False.

    Shape:
        - Input: (N, C, H_in, W_in)
        - Output: (N, C, H_out, W_out)
    """

    @staticmethod
    def _start_index(a: int, b: int, c: int) -> int:
        """Calculate start index in input for output index a."""
        return (a // b) * c + ((a % b) * c) // b

    @staticmethod
    def _end_index(a: int, b: int, c: int) -> int:
        """Calculate end index (exclusive) in input for output index a."""
        return 1 + ((a + 1) * c - 1) // b

    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        output_size: Union[int, Tuple[int, int]] = 1,
        return_indices: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, NDArray]]:
        # Normalize output_size to tuple
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        
        H_out, W_out = output_size

        # Validate input
        if input.ndim != 4:
            raise ValueError(
                f"AdaptiveMaxPool2d expects 4D input (N, C, H, W), got {input.ndim}D"
            )
        
        N, C, H_in, W_in = input.shape

        if H_out <= 0 or W_out <= 0:
            raise ValueError(
                f"output_size must be positive, got {output_size}"
            )
        
        # Prepare output and indices
        output_data: NDArray[np.float32] = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        indices_data: NDArray[np.int32] = np.zeros((N, C, H_out, W_out), dtype=np.int32)
        input_data = input.data

        # Perform adaptive max pooling
        for n in range(N):
            for c in range(C):
                for oh in range(H_out):
                    # Compute pooling region for height
                    ih0 = AdaptiveMaxPool2dFunction._start_index(oh, H_out, H_in)
                    ih1 = AdaptiveMaxPool2dFunction._end_index(oh, H_out, H_in)
                    
                    for ow in range(W_out):
                        # Compute pooling region for width
                        iw0 = AdaptiveMaxPool2dFunction._start_index(ow, W_out, W_in)
                        iw1 = AdaptiveMaxPool2dFunction._end_index(ow, W_out, W_in)
                        
                        # Extract window
                        window = input_data[n, c, ih0:ih1, iw0:iw1]
                        
                        if window.size > 0:
                            # Find maximum value and its index
                            # Flatten window to find global max
                            flat_window = window.flatten()
                            max_val = np.max(flat_window)
                            flat_idx = 0  # default, will be overwritten
                            
                            # Handle NaN values (NaN always "wins" in PyTorch)
                            if np.any(np.isnan(flat_window)):
                                max_val = np.nan
                                # Find first NaN position
                                nan_positions = np.where(np.isnan(flat_window))[0]
                                # nan_positions should have at least one element since np.any returned True
                                flat_idx = nan_positions[0]
                            else:
                                flat_idx = np.argmax(flat_window)
                            
                            # Convert flat index back to 2D coordinates in input
                            h_idx, w_idx = np.unravel_index(flat_idx, (ih1 - ih0, iw1 - iw0))
                            abs_h = ih0 + h_idx
                            abs_w = iw0 + w_idx
                            
                            output_data[n, c, oh, ow] = max_val
                            # Store as flattened index: n * (C*H*W) + c * (H*W) + h * W + w
                            indices_data[n, c, oh, ow] = (
                                n * (C * H_in * W_in) + 
                                c * (H_in * W_in) + 
                                abs_h * W_in + 
                                abs_w
                            )
                        else:
                            output_data[n, c, oh, ow] = 0.0
                            indices_data[n, c, oh, ow] = 0
        
        # Save context for backward pass
        ctx.save_for_backward(input)
        ctx.save_value("output_size", output_size)
        ctx.save_value("indices", indices_data)
        ctx.save_value("H_out", H_out)
        ctx.save_value("W_out", W_out)
        ctx.save_value("H_in", H_in)
        ctx.save_value("W_in", W_in)

        requires_grad = input.requires_grad
        output = Tensor(output_data, requires_grad=requires_grad)
        
        if return_indices:
            return output, indices_data
        else:
            return output

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Any
    ) -> Tuple[Optional[Tensor], None, None]:
        grad_output = grad_outputs[0]
        input = ctx.saved_tensors[0]
        output_size = ctx.get_value("output_size")
        indices = ctx.get_value("indices")
        H_out, W_out = output_size
        H_in = ctx.get_value("H_in")
        W_in = ctx.get_value("W_in")

        N, C, _, _ = input.shape
        grad_output_data = grad_output.data

        # Initialize gradient
        grad_input: NDArray[np.float32] = np.zeros(
            (N, C, H_in, W_in), dtype=np.float32
        )

        # Backward pass: distribute gradient only to max positions
        for n in range(N):
            for c in range(C):
                for oh in range(H_out):
                    for ow in range(W_out):
                        idx = indices[n, c, oh, ow]
                        
                        # Unpack flattened index
                        n_idx = idx // (C * H_in * W_in)
                        remainder = idx % (C * H_in * W_in)
                        c_idx = remainder // (H_in * W_in)
                        remainder2 = remainder % (H_in * W_in)
                        h_idx = remainder2 // W_in
                        w_idx = remainder2 % W_in
                        
                        # Ensure indices are within bounds
                        if (0 <= n_idx < N and 0 <= c_idx < C and 
                            0 <= h_idx < H_in and 0 <= w_idx < W_in):
                            grad_input[n_idx, c_idx, h_idx, w_idx] += grad_output_data[n, c, oh, ow]

        return (
            Tensor(grad_input, requires_grad=False),
            None,
            None,
        )


class LayerNormFunction(Function):
    """Layer Normalization operation.
    
    Applies Layer Normalization over the last D dimensions where D is the length
    of normalized_shape.
    
    Args:
        normalized_shape: Shape of the normalized dimensions.
        weight: Optional weight tensor (gamma) of shape normalized_shape.
        bias: Optional bias tensor (beta) of shape normalized_shape.
        eps: A value added to the denominator for numerical stability.
    """
    
    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        normalized_shape: Union[int, Tuple[int, ...]],
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        eps: float = 1e-5,
    ) -> Tensor:
        # Convert normalized_shape to tuple if needed
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        
        # Validate input shape
        input_shape = input.shape
        normalized_ndim = len(normalized_shape)
        
        if input_shape[-normalized_ndim:] != normalized_shape:
            raise ValueError(
                f"Expected input with trailing dimensions {normalized_shape}, "
                f"got {input_shape}"
            )
        
        # Reshape to (M, N) where M = product of non-normalized dimensions,
        # N = product of normalized dimensions
        M = np.prod(input_shape[:-normalized_ndim], dtype=int)
        N = np.prod(normalized_shape, dtype=int)
        input_reshaped = input.data.reshape(M, N)
        
        # Compute mean and variance over normalized dimension (axis=1)
        mean = np.mean(input_reshaped, axis=1, keepdims=True)  # Shape: (M, 1)
        variance = np.var(input_reshaped, axis=1, keepdims=True)  # Biased variance (ddof=0)
        
        # Compute inverse standard deviation (rstd)
        rstd = 1.0 / np.sqrt(variance + eps)  # Shape: (M, 1)
        
        # Center and scale
        x_centered = input_reshaped - mean  # Shape: (M, N)
        x_hat = x_centered * rstd  # Shape: (M, N) - normalized values
        
        # Apply affine transformation if weight and bias are provided
        if weight is not None:
            gamma = weight.data.reshape(N)  # Shape: (N,)
            if bias is not None:
                beta = bias.data.reshape(N)  # Shape: (N,)
                output_reshaped = x_hat * gamma + beta
            else:
                output_reshaped = x_hat * gamma
        else:
            output_reshaped = x_hat
        
        # Reshape back to original shape
        output = output_reshaped.reshape(input_shape)
        
        # Save for backward pass
        ctx.save_for_backward(
            input.data,
            mean.reshape(-1),  # Shape: (M,)
            rstd.reshape(-1),  # Shape: (M,)
            weight.data if weight is not None else None,
            bias.data if bias is not None else None,
            x_hat,  # Normalized values
            M,
            N,
            normalized_ndim,
            input_shape,
            eps,
        )
        
        return Tensor(output, requires_grad=input.requires_grad)
    
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        # Retrieve saved tensors
        (
            input_data,
            mean,
            rstd,
            weight_data,
            bias_data,
            x_hat,
            M,
            N,
            normalized_ndim,
            input_shape,
            eps,
        ) = ctx.saved_tensors
        
        grad_output = grad_outputs[0]
        # Reshape grad_output data to (M, N)
        grad_output_reshaped = grad_output.data.reshape(M, N)
        
        # Prepare gamma (weight) for gradient computation
        gamma = weight_data.reshape(N) if weight_data is not None else 1.0
        
        # Gradient w.r.t weight (gamma)
        grad_weight = None
        if weight_data is not None:
            # dgamma = sum(dY * x_hat) over batch dimension (axis=0)
            grad_weight = np.sum(grad_output_reshaped * x_hat, axis=0)  # Shape: (N,)
            grad_weight = grad_weight.reshape(weight_data.shape)
        
        # Gradient w.r.t bias (beta)
        grad_bias = None
        if bias_data is not None:
            # dbeta = sum(dY) over batch dimension (axis=0)
            grad_bias = np.sum(grad_output_reshaped, axis=0)  # Shape: (N,)
            grad_bias = grad_bias.reshape(bias_data.shape)
        
        # Gradient w.r.t input
        # Reshape mean and rstd for broadcasting
        mean_b = mean.reshape(M, 1)  # Shape: (M, 1)
        rstd_b = rstd.reshape(M, 1)  # Shape: (M, 1)
        
        # Reshape input for computation
        input_reshaped = input_data.reshape(M, N)
        
        # Compute ds and db statistics
        # ds = sum(dY * X * gamma) over normalized dimension (N)
        # db = sum(dY * gamma) over normalized dimension (N)
        ds = np.sum(grad_output_reshaped * input_reshaped * gamma, axis=1, keepdims=True)  # Shape: (M, 1)
        db = np.sum(grad_output_reshaped * gamma, axis=1, keepdims=True)  # Shape: (M, 1)
        
        # Compute gradients w.r.t input using PyTorch's formula
        # a = rstd
        # b = (db * mean - ds) * a^3 * scale, where scale = 1/N
        # c = -b * mean - db * a * scale
        # dX = a * dY * gamma + b * X + c
        scale = 1.0 / N
        a = rstd_b
        a_cubed = a * a * a
        b = (db * mean_b - ds) * a_cubed * scale
        c = -b * mean_b - db * a * scale
        
        grad_input_reshaped = a * grad_output_reshaped * gamma + b * input_reshaped + c
        
        # Reshape back to original input shape
        grad_input = grad_input_reshaped.reshape(input_shape)

        # Return gradients: input, normalized_shape, weight, bias, eps
        # normalized_shape and eps have no gradient
        return (
            Tensor(grad_input, requires_grad=False),
            None,
            Tensor(grad_weight, requires_grad=False) if grad_weight is not None else None,
            Tensor(grad_bias, requires_grad=False) if grad_bias is not None else None,
            None,
        )


class GroupNormFunction(Function):
    """Group Normalization operation.
    
    Applies Group Normalization as described in the paper
    "Group Normalization" (https://arxiv.org/abs/1803.08494).
    
    Args:
        num_groups: Number of groups to separate the channels into.
        weight: Optional weight tensor (gamma) of shape (C,).
        bias: Optional bias tensor (beta) of shape (C,).
        eps: A value added to the denominator for numerical stability.
    """
    
    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        num_groups: int,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        eps: float = 1e-5,
    ) -> Tensor:
        # Validate input shape
        input_shape = input.shape
        if len(input_shape) < 2:
            raise ValueError(f"GroupNorm expects at least 2D input, got {len(input_shape)}D")
        
        N, C = input_shape[0], input_shape[1]
        spatial_shape = input_shape[2:]
        
        # Validate num_channels divisible by num_groups
        if C % num_groups != 0:
            raise ValueError(
                f"num_channels ({C}) must be divisible by num_groups ({num_groups})"
            )
        
        group_size = C // num_groups
        
        # Reshape to group-wise: (N, num_groups, group_size, *spatial)
        # Then flatten to (N * num_groups, group_size * spatial_elements)
        spatial_elements = 1
        for dim in spatial_shape:
            spatial_elements *= dim
        
        # Reshape input to (N, num_groups, group_size, spatial_elements)
        input_reshaped = input.data.reshape(N, num_groups, group_size, spatial_elements)
        
        # Compute mean and variance over (group_size, spatial_elements) dimension (axis=(2, 3))
        mean = np.mean(input_reshaped, axis=(2, 3), keepdims=True)  # Shape: (N, num_groups, 1, 1)
        variance = np.var(input_reshaped, axis=(2, 3), keepdims=True)  # Biased variance (ddof=0)
        
        # Compute inverse standard deviation (rstd)
        rstd = 1.0 / np.sqrt(variance + eps)  # Shape: (N, num_groups, 1, 1)
        
        # Center and scale
        x_centered = input_reshaped - mean  # Shape: (N, num_groups, group_size, spatial_elements)
        x_hat = x_centered * rstd  # Normalized values
        
        # Reshape back to original shape for affine transformation
        x_hat_reshaped = x_hat.reshape(input_shape)
        
        # Apply affine transformation if weight and bias are provided
        if weight is not None:
            # Reshape gamma to (C, 1, 1, ...) for broadcasting across spatial dimensions
            gamma_shape = (C,) + (1,) * len(spatial_shape)
            gamma = weight.data.reshape(gamma_shape)
            if bias is not None:
                beta = bias.data.reshape(gamma_shape)
                output = x_hat_reshaped * gamma + beta
            else:
                output = x_hat_reshaped * gamma
        else:
            output = x_hat_reshaped
        
        # Save for backward pass
        ctx.save_for_backward(
            input.data,
            mean.reshape(N * num_groups, 1),  # Shape: (M, 1) where M = N * num_groups
            rstd.reshape(N * num_groups, 1),  # Shape: (M, 1)
            weight.data if weight is not None else None,
            bias.data if bias is not None else None,
            x_hat,  # Normalized values shape: (N, num_groups, group_size, spatial_elements)
            N,
            C,
            num_groups,
            group_size,
            spatial_elements,
            input_shape,
            eps,
        )
        
        return Tensor(output, requires_grad=input.requires_grad)
    
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        # Retrieve saved tensors
        (
            input_data,
            mean,
            rstd,
            weight_data,
            bias_data,
            x_hat,
            N,
            C,
            num_groups,
            group_size,
            spatial_elements,
            input_shape,
            eps,
        ) = ctx.saved_tensors
        
        grad_output = grad_outputs[0]

        # Reshape grad_output to (N, num_groups, group_size, spatial_elements)
        grad_output_reshaped = grad_output.data.reshape(N, num_groups, group_size, spatial_elements)

        # Gradient w.r.t weight (gamma)
        grad_weight = None
        if weight_data is not None:
            # dgamma = sum(dY * x_hat) over (N, spatial_elements) dimensions
            # x_hat is already normalized values shape: (N, num_groups, group_size, spatial_elements)
            # Need to reshape to (N, C, spatial_elements) first
            x_hat_channels = x_hat.reshape(N, C, spatial_elements)
            grad_output_channels = grad_output.data.reshape(N, C, spatial_elements)
            grad_weight = np.sum(grad_output_channels * x_hat_channels, axis=(0, 2))  # Shape: (C,)
            grad_weight = grad_weight.reshape(weight_data.shape)
        
        # Gradient w.r.t bias (beta)
        grad_bias = None
        if bias_data is not None:
            # dbeta = sum(dY) over (N, spatial_elements) dimensions
            grad_output_channels = grad_output.data.reshape(N, C, spatial_elements)
            grad_bias = np.sum(grad_output_channels, axis=(0, 2))  # Shape: (C,)
            grad_bias = grad_bias.reshape(bias_data.shape)
        
        # Gradient w.r.t input
        # Reshape for computation: flatten to (M, K) where M = N * num_groups, K = group_size * spatial_elements
        M = N * num_groups
        K = group_size * spatial_elements
        
        input_flat = input_data.reshape(M, K)
        grad_output_flat = grad_output_reshaped.reshape(M, K)
        mean_b = mean.reshape(M, 1)  # Shape: (M, 1)
        rstd_b = rstd.reshape(M, 1)  # Shape: (M, 1)
        
        # Prepare gamma for broadcasting: need gamma per channel, reshape to (M, K)
        if weight_data is not None:
            # gamma shape: (C, 1, 1) -> need to expand to (N, num_groups, group_size, 1)
            gamma_per_group = weight_data.reshape(num_groups, group_size, 1)
            gamma_expanded = np.repeat(gamma_per_group, spatial_elements, axis=2)  # (num_groups, group_size, spatial_elements)
            gamma_expanded = gamma_expanded.reshape(1, num_groups, group_size, spatial_elements)
            gamma_expanded = np.repeat(gamma_expanded, N, axis=0)  # (N, num_groups, group_size, spatial_elements)
            gamma_flat = gamma_expanded.reshape(M, K)
        else:
            gamma_flat = 1.0
        
        # Compute ds and db statistics
        # ds = sum(dY * X * gamma) over normalized dimension (K)
        # db = sum(dY * gamma) over normalized dimension (K)
        ds = np.sum(grad_output_flat * input_flat * gamma_flat, axis=1, keepdims=True)  # Shape: (M, 1)
        db = np.sum(grad_output_flat * gamma_flat, axis=1, keepdims=True)  # Shape: (M, 1)
        
        # Compute gradients w.r.t input using LayerNorm formula
        scale = 1.0 / K
        a = rstd_b
        a_cubed = a * a * a
        b = (db * mean_b - ds) * a_cubed * scale
        c = -b * mean_b - db * a * scale
        
        grad_input_flat = a * grad_output_flat * gamma_flat + b * input_flat + c
        
        # Reshape back to original input shape
        grad_input = grad_input_flat.reshape(input_shape)

        # Return gradients: input, num_groups, weight, bias, eps
        # num_groups and eps have no gradient
        return (
            Tensor(grad_input, requires_grad=False),
            None,
            Tensor(grad_weight, requires_grad=False) if grad_weight is not None else None,
            Tensor(grad_bias, requires_grad=False) if grad_bias is not None else None,
            None,
        )


class InstanceNormFunction(Function):
    """Instance Normalization operation.
    
    Applies Instance Normalization independently for each channel in each sample.
    For 2D input (N, C, H, W), computes statistics over spatial dimensions (H, W)
    for each channel in each sample independently.
    
    Args:
        weight: Optional weight tensor (gamma) of shape (C,).
        bias: Optional bias tensor (beta) of shape (C,).
        eps: A value added to the denominator for numerical stability.
    """
    
    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        eps: float = 1e-5,
    ) -> Tensor:
        # Validate input shape
        input_shape = input.shape
        if len(input_shape) < 2:
            raise ValueError(
                f"InstanceNorm expects at least 2D input, got {len(input_shape)}D"
            )
        
        N, C = input_shape[0], input_shape[1]
        spatial_shape = input_shape[2:]
        
        # Compute spatial elements product
        spatial_elements = 1
        for dim in spatial_shape:
            spatial_elements *= dim
        
        # Reshape to (N*C, spatial_elements) for batch normalization
        M = N * C
        K = spatial_elements
        input_reshaped = input.data.reshape(M, K)
        
        # Compute mean and variance over spatial dimension (axis=1)
        mean = np.mean(input_reshaped, axis=1, keepdims=True)  # Shape: (M, 1)
        variance = np.var(input_reshaped, axis=1, keepdims=True)  # Biased variance (ddof=0)
        
        # Compute inverse standard deviation (rstd)
        rstd = 1.0 / np.sqrt(variance + eps)  # Shape: (M, 1)
        
        # Center and scale
        x_centered = input_reshaped - mean  # Shape: (M, K)
        x_hat = x_centered * rstd  # Normalized values
        
        # Reshape back to original shape for affine transformation
        x_hat_reshaped = x_hat.reshape(input_shape)
        
        # Apply affine transformation if weight and bias are provided
        if weight is not None:
            # Reshape gamma to (C, 1, 1, ...) for broadcasting across spatial dimensions
            gamma_shape = (C,) + (1,) * len(spatial_shape)
            gamma = weight.data.reshape(gamma_shape)
            if bias is not None:
                beta = bias.data.reshape(gamma_shape)
                output = x_hat_reshaped * gamma + beta
            else:
                output = x_hat_reshaped * gamma
        else:
            output = x_hat_reshaped
        
        # Save for backward pass
        ctx.save_for_backward(
            input.data,
            mean,        # Shape: (M, 1)
            rstd,        # Shape: (M, 1)
            weight.data if weight is not None else None,
            bias.data if bias is not None else None,
            x_hat,       # Normalized values shape: (M, K)
            N,
            C,
            spatial_elements,
            input_shape,
            eps,
        )
        
        return Tensor(output, requires_grad=input.requires_grad)
    
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        # Retrieve saved tensors
        (
            input_data,
            mean,
            rstd,
            weight_data,
            bias_data,
            x_hat,
            N,
            C,
            spatial_elements,
            input_shape,
            eps,
        ) = ctx.saved_tensors
        
        grad_output = grad_outputs[0]
        
        # Reshape grad_output to (M, K)
        M = N * C
        K = spatial_elements
        grad_output_reshaped = grad_output.data.reshape(M, K)
        
        # Prepare gamma (weight) for gradient computation
        gamma = 1.0 if weight_data is None else weight_data
        
        # Gradient w.r.t weight (gamma)
        grad_weight = None
        if weight_data is not None:
            # dgamma = sum(dY * x_hat) over (N, spatial_elements) dimensions
            # Reshape x_hat and grad_output to (N, C, K) first
            x_hat_channels = x_hat.reshape(N, C, K)
            grad_output_channels = grad_output.data.reshape(N, C, K)
            grad_weight = np.sum(grad_output_channels * x_hat_channels, axis=(0, 2))  # Shape: (C,)
            grad_weight = grad_weight.reshape(weight_data.shape)
        
        # Gradient w.r.t bias (beta)
        grad_bias = None
        if bias_data is not None:
            # dbeta = sum(dY) over (N, spatial_elements) dimensions
            grad_output_channels = grad_output.data.reshape(N, C, K)
            grad_bias = np.sum(grad_output_channels, axis=(0, 2))  # Shape: (C,)
            grad_bias = grad_bias.reshape(bias_data.shape)
        
        # Gradient w.r.t input
        # Prepare gamma for broadcasting: need gamma per channel, reshape to (M, K)
        if weight_data is not None:
            # gamma shape: (C,) -> need to expand to (N, C, 1) then repeat for spatial elements
            gamma_per_channel = gamma.reshape(1, C, 1)
            gamma_expanded = np.repeat(gamma_per_channel, K, axis=2)  # (1, C, K)
            gamma_expanded = np.repeat(gamma_expanded, N, axis=0)  # (N, C, K)
            gamma_flat = gamma_expanded.reshape(M, K)
        else:
            gamma_flat = 1.0
        
        # Reshape input to (M, K)
        input_flat = input_data.reshape(M, K)
        
        # Compute ds and db statistics using LayerNorm formula
        # ds = sum(dY * X * gamma) over spatial dimension (K)
        # db = sum(dY * gamma) over spatial dimension (K)
        ds = np.sum(grad_output_reshaped * input_flat * gamma_flat, axis=1, keepdims=True)  # Shape: (M, 1)
        db = np.sum(grad_output_reshaped * gamma_flat, axis=1, keepdims=True)  # Shape: (M, 1)
        
        # Compute gradients w.r.t input using LayerNorm formula
        scale = 1.0 / K
        a = rstd  # Shape: (M, 1)
        a_cubed = a * a * a
        b = (db * mean - ds) * a_cubed * scale
        c = -b * mean - db * a * scale
        
        grad_input_flat = a * grad_output_reshaped * gamma_flat + b * input_flat + c
        
        # Reshape back to original input shape
        grad_input = grad_input_flat.reshape(input_shape)

        # Return gradients: input, weight, bias, eps
        # eps has no gradient
        return (
            Tensor(grad_input, requires_grad=False),
            Tensor(grad_weight, requires_grad=False) if grad_weight is not None else None,
            Tensor(grad_bias, requires_grad=False) if grad_bias is not None else None,
            None,
        )


# Convenience functions for common operations
def add(a: Tensor, b: Tensor) -> Tensor:
    return Add.apply(a, b)


def mul(a: Tensor, b: Tensor) -> Tensor:
    return Mul.apply(a, b)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatMul.apply(a, b)


def cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    return CatFunction.apply(*tensors, dim=dim)


def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
    return StackFunction.apply(*tensors, dim=dim)
