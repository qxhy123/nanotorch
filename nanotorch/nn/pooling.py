"""
Pooling layers for nanotorch.

This module provides MaxPool and AvgPool layers for 1D, 2D, and 3D data,
similar to PyTorch's nn pooling modules.
"""

import numpy as np
from typing import Tuple, Optional, Union
from nanotorch.nn.module import Module
from nanotorch.tensor import Tensor
from nanotorch.autograd import MaxPool2dFunction, AvgPool2dFunction, AdaptiveAvgPool2dFunction, AdaptiveMaxPool2dFunction


class MaxPool1d(Module):
    """1D max pooling layer.

    Applies a 1D max pooling over an input signal composed of several input planes.

    Args:
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling. Default: kernel_size.
        padding: Zero-padding added to both sides. Default: 0.
        dilation: Spacing between kernel elements. Default: 1.
        return_indices: If True, returns indices. Default: False.
        ceil_mode: If True, uses ceil for output shape. Default: False.

    Shape:
        - Input: (N, C, L_in)
        - Output: (N, C, L_out)

        where:
            L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
    """

    def __init__(
        self,
        kernel_size: int = 2,
        stride: Optional[int] = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, np.ndarray]]:
        if x.ndim != 3:
            raise ValueError(f"MaxPool1d expects 3D input (N, C, L), got {x.ndim}D")

        N, C, L_in = x.shape
        
        def output_size(input_size: int) -> int:
            kernel_dilated = self.dilation * (self.kernel_size - 1) + 1
            if self.ceil_mode:
                return int(np.ceil((input_size + 2 * self.padding - kernel_dilated) / self.stride + 1))
            return (input_size + 2 * self.padding - kernel_dilated) // self.stride + 1
        
        L_out = output_size(L_in)
        
        if self.padding > 0:
            padded_data = np.zeros((N, C, L_in + 2 * self.padding), dtype=np.float32)
            padded_data[:, :, self.padding:self.padding + L_in] = x.data
        else:
            padded_data = x.data
        
        output_data = np.zeros((N, C, L_out), dtype=np.float32)
        indices = np.zeros((N, C, L_out), dtype=np.int64)
        
        total_L = padded_data.shape[2]
        
        for n in range(N):
            for c in range(C):
                for l_out in range(L_out):
                    l_start = l_out * self.stride
                    l_end = l_start + self.dilation * (self.kernel_size - 1) + 1
                    window = padded_data[n, c, l_start:l_end:self.dilation]
                    output_data[n, c, l_out] = np.max(window)
                    indices[n, c, l_out] = l_start + self.dilation * np.argmax(window)
        
        output = Tensor(output_data, requires_grad=x.requires_grad)
        
        if self.return_indices:
            return output, indices
        return output

    def extra_repr(self) -> str:
        s = f"kernel_size={self.kernel_size}"
        if self.stride != self.kernel_size:
            s += f", stride={self.stride}"
        if self.padding != 0:
            s += f", padding={self.padding}"
        return s

    def __repr__(self) -> str:
        return f"MaxPool1d({self.extra_repr()})"


class AvgPool1d(Module):
    """1D average pooling layer.

    Args:
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling. Default: kernel_size.
        padding: Zero-padding added to both sides. Default: 0.
        ceil_mode: If True, uses ceil for output shape. Default: False.
        count_include_pad: If True, includes padding in average. Default: True.

    Shape:
        - Input: (N, C, L_in)
        - Output: (N, C, L_out)
    """

    def __init__(
        self,
        kernel_size: int = 2,
        stride: Optional[int] = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"AvgPool1d expects 3D input (N, C, L), got {x.ndim}D")

        N, C, L_in = x.shape
        
        def output_size(input_size: int) -> int:
            if self.ceil_mode:
                return int(np.ceil((input_size + 2 * self.padding - self.kernel_size) / self.stride + 1))
            return (input_size + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        L_out = output_size(L_in)
        
        if self.padding > 0:
            padded_data = np.zeros((N, C, L_in + 2 * self.padding), dtype=np.float32)
            padded_data[:, :, self.padding:self.padding + L_in] = x.data
        else:
            padded_data = x.data
        
        output_data = np.zeros((N, C, L_out), dtype=np.float32)
        
        for n in range(N):
            for c in range(C):
                for l_out in range(L_out):
                    l_start = l_out * self.stride
                    l_end = l_start + self.kernel_size
                    window = padded_data[n, c, l_start:l_end]
                    if self.count_include_pad:
                        output_data[n, c, l_out] = np.mean(window)
                    else:
                        actual_count = min(l_end, L_in + 2 * self.padding) - max(l_start, self.padding)
                        if actual_count > 0:
                            output_data[n, c, l_out] = np.sum(window) / actual_count
        
        return Tensor(output_data, requires_grad=x.requires_grad)

    def extra_repr(self) -> str:
        s = f"kernel_size={self.kernel_size}"
        if self.stride != self.kernel_size:
            s += f", stride={self.stride}"
        if self.padding != 0:
            s += f", padding={self.padding}"
        return s

    def __repr__(self) -> str:
        return f"AvgPool1d({self.extra_repr()})"


class MaxPool3d(Module):
    """3D max pooling layer.

    Args:
        kernel_size: Size of the pooling window. Can be a single number or tuple of 3.
        stride: Stride of the pooling. Default: kernel_size.
        padding: Zero-padding added to all sides. Default: 0.
        dilation: Spacing between kernel elements. Default: 1.
        return_indices: If True, returns indices. Default: False.
        ceil_mode: If True, uses ceil for output shape. Default: False.

    Shape:
        - Input: (N, C, D_in, H_in, W_in)
        - Output: (N, C, D_out, H_out, W_out)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]] = 2,
        stride: Optional[Union[int, Tuple[int, int, int]]] = None,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, np.ndarray]]:
        if x.ndim != 5:
            raise ValueError(f"MaxPool3d expects 5D input (N, C, D, H, W), got {x.ndim}D")

        N, C, D_in, H_in, W_in = x.shape
        K_D, K_H, K_W = self.kernel_size
        S_D, S_H, S_W = self.stride
        P_D, P_H, P_W = self.padding
        D_D, D_H, D_W = self.dilation
        
        def output_size(input_size: int, kernel: int, stride: int, pad: int, dil: int) -> int:
            kernel_dilated = dil * (kernel - 1) + 1
            if self.ceil_mode:
                return int(np.ceil((input_size + 2 * pad - kernel_dilated) / stride + 1))
            return (input_size + 2 * pad - kernel_dilated) // stride + 1
        
        D_out = output_size(D_in, K_D, S_D, P_D, D_D)
        H_out = output_size(H_in, K_H, S_H, P_H, D_H)
        W_out = output_size(W_in, K_W, S_W, P_W, D_W)
        
        if any(p > 0 for p in self.padding):
            padded_data = np.zeros((N, C, D_in + 2*P_D, H_in + 2*P_H, W_in + 2*P_W), dtype=np.float32)
            padded_data[:, :, P_D:P_D+D_in, P_H:P_H+H_in, P_W:P_W+W_in] = x.data
        else:
            padded_data = x.data
        
        output_data = np.zeros((N, C, D_out, H_out, W_out), dtype=np.float32)
        indices = np.zeros((N, C, D_out, H_out, W_out), dtype=np.int64)
        
        for n in range(N):
            for c in range(C):
                for d_out in range(D_out):
                    for h_out in range(H_out):
                        for w_out in range(W_out):
                            d_start, h_start, w_start = d_out * S_D, h_out * S_H, w_out * S_W
                            d_end = d_start + D_D * (K_D - 1) + 1
                            h_end = h_start + D_H * (K_H - 1) + 1
                            w_end = w_start + D_W * (K_W - 1) + 1
                            
                            window = padded_data[n, c, d_start:d_end:D_D, h_start:h_end:D_H, w_start:w_end:D_W]
                            output_data[n, c, d_out, h_out, w_out] = np.max(window)
        
        output = Tensor(output_data, requires_grad=x.requires_grad)
        
        if self.return_indices:
            return output, indices
        return output

    def extra_repr(self) -> str:
        s = f"kernel_size={self.kernel_size}"
        if self.stride != self.kernel_size:
            s += f", stride={self.stride}"
        if self.padding != (0, 0, 0):
            s += f", padding={self.padding}"
        return s

    def __repr__(self) -> str:
        return f"MaxPool3d({self.extra_repr()})"


class AvgPool3d(Module):
    """3D average pooling layer.

    Args:
        kernel_size: Size of the pooling window. Can be a single number or tuple of 3.
        stride: Stride of the pooling. Default: kernel_size.
        padding: Zero-padding added to all sides. Default: 0.
        ceil_mode: If True, uses ceil for output shape. Default: False.
        count_include_pad: If True, includes padding in average. Default: True.

    Shape:
        - Input: (N, C, D_in, H_in, W_in)
        - Output: (N, C, D_out, H_out, W_out)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]] = 2,
        stride: Optional[Union[int, Tuple[int, int, int]]] = None,
        padding: Union[int, Tuple[int, int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"AvgPool3d expects 5D input (N, C, D, H, W), got {x.ndim}D")

        N, C, D_in, H_in, W_in = x.shape
        K_D, K_H, K_W = self.kernel_size
        S_D, S_H, S_W = self.stride
        P_D, P_H, P_W = self.padding
        
        def output_size(input_size: int, kernel: int, stride: int, pad: int) -> int:
            if self.ceil_mode:
                return int(np.ceil((input_size + 2 * pad - kernel) / stride + 1))
            return (input_size + 2 * pad - kernel) // stride + 1
        
        D_out = output_size(D_in, K_D, S_D, P_D)
        H_out = output_size(H_in, K_H, S_H, P_H)
        W_out = output_size(W_in, K_W, S_W, P_W)
        
        if any(p > 0 for p in self.padding):
            padded_data = np.zeros((N, C, D_in + 2*P_D, H_in + 2*P_H, W_in + 2*P_W), dtype=np.float32)
            padded_data[:, :, P_D:P_D+D_in, P_H:P_H+H_in, P_W:P_W+W_in] = x.data
        else:
            padded_data = x.data
        
        output_data = np.zeros((N, C, D_out, H_out, W_out), dtype=np.float32)
        
        for n in range(N):
            for c in range(C):
                for d_out in range(D_out):
                    for h_out in range(H_out):
                        for w_out in range(W_out):
                            d_start, h_start, w_start = d_out * S_D, h_out * S_H, w_out * S_W
                            window = padded_data[n, c, d_start:d_start+K_D, h_start:h_start+K_H, w_start:w_start+K_W]
                            output_data[n, c, d_out, h_out, w_out] = np.mean(window)
        
        return Tensor(output_data, requires_grad=x.requires_grad)

    def extra_repr(self) -> str:
        s = f"kernel_size={self.kernel_size}"
        if self.stride != self.kernel_size:
            s += f", stride={self.stride}"
        if self.padding != (0, 0, 0):
            s += f", padding={self.padding}"
        return s

    def __repr__(self) -> str:
        return f"AvgPool3d({self.extra_repr()})"


class MaxPool2d(Module):
    """2D max pooling layer.

    Applies a 2D max pooling over an input signal composed of several input planes.

    Args:
        kernel_size: Size of the pooling window. Can be a single number or a tuple.
        stride: Stride of the pooling. Default: kernel_size.
        padding: Zero-padding added to both sides of the input. Default: 0.
        dilation: Spacing between kernel elements. Default: 1.
        return_indices: If True, will return the indices along with the outputs. Default: False.
        ceil_mode: If True, will use ceil instead of floor to compute output shape. Default: False.

    Shape:
        - Input: (N, C, H_in, W_in)
        - Output: (N, C, H_out, W_out)

        where (when ceil_mode=False):
            H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
            W_out = floor((W_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = (2, 2),
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size

        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride if isinstance(stride, tuple) else (stride, stride)

        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, np.ndarray]]:
        """Forward pass of MaxPool2d layer.

        Args:
            x: Input tensor of shape (N, C, H_in, W_in).

        Returns:
            Output tensor of shape (N, C, H_out, W_out).
            If return_indices=True, also returns indices tensor.
        """
        if x.ndim != 4:
            raise ValueError(f"MaxPool2d expects 4D input (N, C, H, W), got {x.ndim}D")

        output = MaxPool2dFunction.apply(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )

        return output

    def extra_repr(self) -> str:
        """Extra representation string."""
        s = f"kernel_size={self.kernel_size}"
        if self.stride != self.kernel_size:
            s += f", stride={self.stride}"
        if self.padding != (0, 0):
            s += f", padding={self.padding}"
        if self.dilation != (1, 1):
            s += f", dilation={self.dilation}"
        if self.ceil_mode:
            s += f", ceil_mode={self.ceil_mode}"
        if self.return_indices:
            s += f", return_indices={self.return_indices}"
        return s

    def __repr__(self) -> str:
        return f"MaxPool2d({self.extra_repr()})"


class AvgPool2d(Module):
    """2D average pooling layer.

    Applies a 2D average pooling over an input signal composed of several input planes.

    Args:
        kernel_size: Size of the pooling window. Can be a single number or a tuple.
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

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = (2, 2),
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size

        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride if isinstance(stride, tuple) else (stride, stride)

        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of AvgPool2d layer.

        Args:
            x: Input tensor of shape (N, C, H_in, W_in).

        Returns:
            Output tensor of shape (N, C, H_out, W_out).
        """
        if x.ndim != 4:
            raise ValueError(f"AvgPool2d expects 4D input (N, C, H, W), got {x.ndim}D")

        output = AvgPool2dFunction.apply(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )

        return output

    def extra_repr(self) -> str:
        """Extra representation string."""
        s = f"kernel_size={self.kernel_size}"
        if self.stride != self.kernel_size:
            s += f", stride={self.stride}"
        if self.padding != (0, 0):
            s += f", padding={self.padding}"
        if self.ceil_mode:
            s += f", ceil_mode={self.ceil_mode}"
        if not self.count_include_pad:
            s += f", count_include_pad={self.count_include_pad}"
        if self.divisor_override is not None:
            s += f", divisor_override={self.divisor_override}"
        return s

    def __repr__(self) -> str:
        return f"AvgPool2d({self.extra_repr()})"


class AdaptiveAvgPool2d(Module):
    """2D adaptive average pooling layer.

    Applies a 2D adaptive average pooling over an input signal composed of several
    input planes. The output size is specified rather than kernel size/stride.

    Args:
        output_size: Target output size. Can be a single integer (applied to both
            height and width) or a tuple (H_out, W_out). Default: 1.

    Shape:
        - Input: (N, C, H_in, W_in)
        - Output: (N, C, H_out, W_out)
    """

    def __init__(
        self,
        output_size: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        super().__init__()
        
        # Normalize output_size to tuple
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of AdaptiveAvgPool2d layer.

        Args:
            x: Input tensor of shape (N, C, H_in, W_in).

        Returns:
            Output tensor of shape (N, C, H_out, W_out).
        """
        if x.ndim != 4:
            raise ValueError(
                f"AdaptiveAvgPool2d expects 4D input (N, C, H, W), got {x.ndim}D"
            )
        
        output = AdaptiveAvgPool2dFunction.apply(x, self.output_size)
        return output

    def extra_repr(self) -> str:
        """Extra representation string."""
        return f"output_size={self.output_size}"

    def __repr__(self) -> str:
        return f"AdaptiveAvgPool2d({self.extra_repr()})"


class AdaptiveMaxPool2d(Module):
    """2D adaptive max pooling layer.

    Applies a 2D adaptive max pooling over an input signal composed of several
    input planes. The output size is specified rather than kernel size/stride.

    Args:
        output_size: Target output size. Can be a single integer (applied to both
            height and width) or a tuple (H_out, W_out). Default: 1.
        return_indices: If True, will return the indices along with the outputs.
            Default: False.

    Shape:
        - Input: (N, C, H_in, W_in)
        - Output: (N, C, H_out, W_out)
    """

    def __init__(
        self,
        output_size: Union[int, Tuple[int, int]] = 1,
        return_indices: bool = False,
    ) -> None:
        super().__init__()
        
        # Normalize output_size to tuple
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        
        self.output_size = output_size
        self.return_indices = return_indices

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, np.ndarray]]:
        """Forward pass of AdaptiveMaxPool2d layer.

        Args:
            x: Input tensor of shape (N, C, H_in, W_in).

        Returns:
            Output tensor of shape (N, C, H_out, W_out).
            If return_indices=True, also returns indices tensor.
        """
        if x.ndim != 4:
            raise ValueError(
                f"AdaptiveMaxPool2d expects 4D input (N, C, H, W), got {x.ndim}D"
            )
        
        output = AdaptiveMaxPool2dFunction.apply(x, self.output_size, self.return_indices)
        return output

    def extra_repr(self) -> str:
        """Extra representation string."""
        s = f"output_size={self.output_size}"
        if self.return_indices:
            s += f", return_indices={self.return_indices}"
        return s

    def __repr__(self) -> str:
        return f"AdaptiveMaxPool2d({self.extra_repr()})"


# Functional pooling interfaces
def max_pool2d(
    input: Tensor,
    kernel_size: Union[int, Tuple[int, int]] = (2, 2),
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
) -> Union[Tensor, Tuple[Tensor, np.ndarray]]:
    """Functional interface for 2D max pooling.

    Args:
        input: Input tensor of shape (N, C, H_in, W_in).
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling. Default: kernel_size.
        padding: Zero-padding added to both sides.
        dilation: Spacing between kernel elements.
        return_indices: If True, will return the indices along with the outputs.
        ceil_mode: If True, will use ceil instead of floor to compute output shape.

    Returns:
        Output tensor of shape (N, C, H_out, W_out).
        If return_indices=True, also returns indices tensor.
    """
    return MaxPool2dFunction.apply(
        input, kernel_size, stride, padding, dilation, ceil_mode, return_indices
    )


def avg_pool2d(
    input: Tensor,
    kernel_size: Union[int, Tuple[int, int]] = (2, 2),
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> Tensor:
    """Functional interface for 2D average pooling.

    Args:
        input: Input tensor of shape (N, C, H_in, W_in).
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling. Default: kernel_size.
        padding: Zero-padding added to both sides.
        ceil_mode: If True, will use ceil instead of floor to compute output shape.
        count_include_pad: If True, will include zero-padding in averaging.
        divisor_override: If specified, it will be used as divisor.

    Returns:
        Output tensor of shape (N, C, H_out, W_out).
    """
    return AvgPool2dFunction.apply(
        input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
    )


def adaptive_avg_pool2d(
    input: Tensor,
    output_size: Union[int, Tuple[int, int]] = 1,
) -> Tensor:
    """Functional interface for 2D adaptive average pooling.

    Args:
        input: Input tensor of shape (N, C, H_in, W_in).
        output_size: Target output size. Can be a single integer (applied to both
            height and width) or a tuple (H_out, W_out). Default: 1.

    Returns:
        Output tensor of shape (N, C, H_out, W_out).
    """
    return AdaptiveAvgPool2dFunction.apply(input, output_size)


def adaptive_max_pool2d(
    input: Tensor,
    output_size: Union[int, Tuple[int, int]] = 1,
    return_indices: bool = False,
) -> Union[Tensor, Tuple[Tensor, np.ndarray]]:
    """Functional interface for 2D adaptive max pooling.

    Args:
        input: Input tensor of shape (N, C, H_in, W_in).
        output_size: Target output size. Can be a single integer (applied to both
            height and width) or a tuple (H_out, W_out). Default: 1.
        return_indices: If True, will return the indices along with the outputs.
            Default: False.

    Returns:
        Output tensor of shape (N, C, H_out, W_out).
        If return_indices=True, also returns indices tensor.
    """
    return AdaptiveMaxPool2dFunction.apply(input, output_size, return_indices)
