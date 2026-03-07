"""
Convolution layers for nanotorch.

This module provides Conv1D, Conv2D, ConvTranspose2D, and Conv3D layers similar to PyTorch's nn.
"""

import numpy as np
from typing import Tuple, Optional, Union
from nanotorch.nn.module import Module
from nanotorch.tensor import Tensor
from nanotorch.autograd import Conv1DFunction, Conv2DFunction, ConvTranspose2DFunction, Conv3DFunction, ConvTranspose3DFunction


class Conv1D(Module):
    """1D convolution layer.

    Applies a 1D convolution over an input signal composed of several input planes.

    Args:
        in_channels: Number of channels in the input signal.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Default: 1.
        padding: Zero-padding added to both sides of the input. Default: 0.
        dilation: Spacing between kernel elements. Default: 1.
        bias: If True, adds a learnable bias to the output. Default: True.

    Shape:
        - Input: (N, C_in, L_in)
        - Output: (N, C_out, L_out)

        where:
            L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        weight_shape = (out_channels, in_channels, kernel_size)
        self.weight = Tensor(
            np.random.randn(*weight_shape).astype(np.float32) * 0.1, requires_grad=True
        )

        self.bias: Optional[Tensor] = None
        if bias:
            self.bias = Tensor(
                np.zeros((out_channels, 1), dtype=np.float32), requires_grad=True
            )

        self.register_parameter("weight", self.weight)
        if self.bias is not None:
            self.register_parameter("bias", self.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Conv1D expects 3D input (N, C, L), got {x.ndim}D")

        N, C_in, L_in = x.shape

        if C_in != self.in_channels:
            raise ValueError(f"Expected input channels {self.in_channels}, got {C_in}")

        output = Conv1DFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation
        )

        return output

    def extra_repr(self) -> str:
        s = f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}"
        if self.stride != 1:
            s += f", stride={self.stride}"
        if self.padding != 0:
            s += f", padding={self.padding}"
        if self.dilation != 1:
            s += f", dilation={self.dilation}"
        s += f", bias={self.bias is not None}"
        return s


class Conv2D(Module):
    """2D convolution layer.

    Applies a 2D convolution over an input signal composed of several input planes.

    Args:
        in_channels: Number of channels in the input image.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel. Can be a single number or a tuple.
        stride: Stride of the convolution. Default: 1.
        padding: Zero-padding added to both sides of the input. Default: 0.
        dilation: Spacing between kernel elements. Default: 1.
        bias: If True, adds a learnable bias to the output. Default: True.

    Shape:
        - Input: (N, C_in, H_in, W_in)
        - Output: (N, C_out, H_out, W_out)

        where:
            H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1) /
                stride + 1)
            W_out = floor((W_in + 2*padding - dilation*(kernel_size-1) - 1) /
                stride + 1)
    
    Note:
        Gradient checks using finite differences may show relatively high errors
        (up to ~1e-2 for input gradients, ~3e-2 for weight gradients) due to
        float32 precision limitations. The analytic gradient formulas are
        mathematically correct and suitable for educational purposes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = (3, 3),
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        # Convert kernel_size to tuple if needed
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Initialize weights
        # Weight shape: (out_channels, in_channels, kernel_height, kernel_width)
        weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])
        self.weight = Tensor(
            np.random.randn(*weight_shape).astype(np.float32) * 0.1, requires_grad=True
        )

        # Initialize bias
        self.bias: Optional[Tensor] = None
        if bias:
            self.bias = Tensor(
                np.zeros((out_channels, 1, 1), dtype=np.float32), requires_grad=True
            )

        self.register_parameter("weight", self.weight)
        if self.bias is not None:
            self.register_parameter("bias", self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of Conv2D layer.

        Args:
            x: Input tensor of shape (N, C_in, H_in, W_in).

        Returns:
            Output tensor of shape (N, C_out, H_out, W_out).
        """
        if x.ndim != 4:
            raise ValueError(f"Conv2D expects 4D input (N, C, H, W), got {x.ndim}D")

        N, C_in, H_in, W_in = x.shape

        if C_in != self.in_channels:
            raise ValueError(f"Expected input channels {self.in_channels}, got {C_in}")

        output = Conv2DFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation
        )

        return output

    def _output_size(
        self,
        input_size: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
    ) -> int:
        """Calculate output size for a given dimension."""
        kernel_size_dilated = dilation * (kernel_size - 1) + 1
        return (input_size + 2 * padding - kernel_size_dilated) // stride + 1

    def _pad_input(self, x: Tensor) -> Tensor:
        """Pad input tensor."""
        N, C, H, W = x.shape
        padded_H = H + 2 * self.padding
        padded_W = W + 2 * self.padding

        # Create padded tensor
        x_padded = Tensor.zeros(
            (N, C, padded_H, padded_W), requires_grad=x.requires_grad
        )

        # Copy original data to center
        x_padded.data[
            :, :, self.padding : self.padding + H, self.padding : self.padding + W
        ] = x.data

        return x_padded

    def _conv2d_forward(self, x: Tensor, H_out: int, W_out: int) -> Tensor:
        """Perform 2D convolution (simple implementation).

        This is a simple, non-optimized implementation for educational purposes.
        In practice, you would use optimized convolution algorithms.
        """
        N, C_in, H_in, W_in = x.shape
        C_out = self.out_channels
        K_H, K_W = self.kernel_size

        # Initialize output tensor
        output = Tensor.zeros((N, C_out, H_out, W_out), requires_grad=x.requires_grad)

        # Simple sliding window convolution
        for n in range(N):  # Batch dimension
            for c_out in range(C_out):  # Output channels
                for h_out in range(H_out):  # Output height
                    for w_out in range(W_out):  # Output width
                        # Calculate input window position
                        h_start = h_out * self.stride
                        w_start = w_out * self.stride
                        h_end = h_start + K_H
                        w_end = w_start + K_W

                        # Extract input window
                        window = x.data[n, :, h_start:h_end, w_start:w_end]

                        # Get corresponding weight slice
                        weight_slice = self.weight.data[c_out]

                        # Compute convolution sum
                        conv_sum = np.sum(window * weight_slice)

                        # Store result
                        output.data[n, c_out, h_out, w_out] = conv_sum

        return output

    def extra_repr(self) -> str:
        """Extra representation string."""
        s = f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}"
        if self.stride != 1:
            s += f", stride={self.stride}"
        if self.padding != 0:
            s += f", padding={self.padding}"
        if self.dilation != 1:
            s += f", dilation={self.dilation}"
        s += f", bias={self.bias is not None}"
        return s


class ConvTranspose2D(Module):
    """2D transposed convolution layer.

    Applies a 2D transposed convolution over an input signal composed of several input planes.
    This is sometimes called "deconvolution" but is actually the gradient of convolution
    with respect to its input.

    Args:
        in_channels: Number of channels in the input image.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel. Can be a single number or a tuple.
        stride: Stride of the convolution. Default: 1.
        padding: Zero-padding added to both sides of the input. Default: 0.
        output_padding: Additional size added to output shape. Default: 0.
        dilation: Spacing between kernel elements. Default: 1.
        groups: Number of blocked connections. Default: 1.
        bias: If True, adds a learnable bias to the output. Default: True.

    Shape:
        - Input: (N, C_in, H_in, W_in)
        - Output: (N, C_out, H_out, W_out)

        where:
            H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size-1) + output_padding + 1
            W_out = (W_in - 1) * stride - 2 * padding + dilation * (kernel_size-1) + output_padding + 1
    
    Note:
        Weight shape is (in_channels, out_channels // groups, K_H, K_W) for ConvTranspose2D,
        different from Conv2D's (out_channels, in_channels, K_H, K_W).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = (3, 3),
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

        # Weight shape for ConvTranspose2D: (in_channels, out_channels // groups, K_H, K_W)
        weight_shape = (in_channels, out_channels // groups, kernel_size[0], kernel_size[1])
        self.weight = Tensor(
            np.random.randn(*weight_shape).astype(np.float32) * 0.1, requires_grad=True
        )

        self.bias: Optional[Tensor] = None
        if bias:
            self.bias = Tensor(
                np.zeros((out_channels, 1, 1), dtype=np.float32), requires_grad=True
            )

        self.register_parameter("weight", self.weight)
        if self.bias is not None:
            self.register_parameter("bias", self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of ConvTranspose2D layer.

        Args:
            x: Input tensor of shape (N, C_in, H_in, W_in).

        Returns:
            Output tensor of shape (N, C_out, H_out, W_out).
        """
        if x.ndim != 4:
            raise ValueError(f"ConvTranspose2D expects 4D input (N, C, H, W), got {x.ndim}D")

        N, C_in, H_in, W_in = x.shape

        if C_in != self.in_channels:
            raise ValueError(f"Expected input channels {self.in_channels}, got {C_in}")

        output = ConvTranspose2DFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding, 
            self.output_padding, self.dilation, self.groups
        )

        return output

    def extra_repr(self) -> str:
        """Extra representation string."""
        s = f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}"
        if self.stride != 1:
            s += f", stride={self.stride}"
        if self.padding != 0:
            s += f", padding={self.padding}"
        if self.output_padding != 0:
            s += f", output_padding={self.output_padding}"
        if self.dilation != 1:
            s += f", dilation={self.dilation}"
        s += f", bias={self.bias is not None}"
        return s

    def __repr__(self) -> str:
        return f"ConvTranspose2d({self.extra_repr()})"


class Conv3D(Module):
    """3D convolution layer.

    Applies a 3D convolution over an input signal composed of several input planes.

    Args:
        in_channels: Number of channels in the input image.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel. Can be a single number or a tuple of 3.
        stride: Stride of the convolution. Can be a single number or a tuple of 3. Default: 1.
        padding: Zero-padding added to all sides of the input. Can be a single number or a tuple of 3. Default: 0.
        dilation: Spacing between kernel elements. Can be a single number or a tuple of 3. Default: 1.
        bias: If True, adds a learnable bias to the output. Default: True.

    Shape:
        - Input: (N, C_in, D_in, H_in, W_in)
        - Output: (N, C_out, D_out, H_out, W_out)

        where:
            D_out = floor((D_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
            H_out = floor((H_in + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1) / stride[1] + 1)
            W_out = floor((W_in + 2*padding[2] - dilation[2]*(kernel_size[2]-1) - 1) / stride[2] + 1)

    Note:
        Gradient checks using finite differences may show relatively high errors
        (up to ~1e-2 for input gradients, ~3e-2 for weight gradients) due to
        float32 precision limitations. The analytic gradient formulas are
        mathematically correct and suitable for educational purposes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = (3, 3, 3),
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        # Convert kernel_size to tuple of 3 if needed
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        # Convert stride to tuple if needed
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        # Convert padding to tuple if needed
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        # Convert dilation to tuple if needed
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Initialize weights
        # Weight shape: (out_channels, in_channels, kernel_depth, kernel_height, kernel_width)
        weight_shape = (
            out_channels,
            in_channels,
            kernel_size[0],
            kernel_size[1],
            kernel_size[2],
        )
        self.weight = Tensor(
            np.random.randn(*weight_shape).astype(np.float32) * 0.1,
            requires_grad=True,
        )

        # Initialize bias
        self.bias: Optional[Tensor] = None
        if bias:
            self.bias = Tensor(
                np.zeros((out_channels, 1, 1, 1), dtype=np.float32),
                requires_grad=True,
            )

        self.register_parameter("weight", self.weight)
        if self.bias is not None:
            self.register_parameter("bias", self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of Conv3D layer.

        Args:
            x: Input tensor of shape (N, C_in, D_in, H_in, W_in).

        Returns:
            Output tensor of shape (N, C_out, D_out, H_out, W_out).
        """
        if x.ndim != 5:
            raise ValueError(f"Conv3D expects 5D input (N, C, D, H, W), got {x.ndim}D")

        N, C_in, D_in, H_in, W_in = x.shape

        if C_in != self.in_channels:
            raise ValueError(
                f"Expected input channels {self.in_channels}, got {C_in}"
            )

        output = Conv3DFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation
        )

        return output

    def _output_size(
        self,
        input_size: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
    ) -> int:
        """Calculate output size for a given dimension."""
        kernel_size_dilated = dilation * (kernel_size - 1) + 1
        return (input_size + 2 * padding - kernel_size_dilated) // stride + 1

    def _pad_input(self, x: Tensor) -> Tensor:
        """Pad input tensor for 3D convolution with per-dimension padding."""
        N, C, D, H, W = x.shape
        padding_d, padding_h, padding_w = self.padding

        padded_D = D + 2 * padding_d
        padded_H = H + 2 * padding_h
        padded_W = W + 2 * padding_w

        # Create padded tensor
        x_padded = Tensor.zeros(
            (N, C, padded_D, padded_H, padded_W), requires_grad=x.requires_grad
        )

        # Copy original data to center
        x_padded.data[
            :, :, padding_d : padding_d + D, padding_h : padding_h + H, padding_w : padding_w + W
        ] = x.data

        return x_padded

    def _conv3d_forward(self, x: Tensor, D_out: int, H_out: int, W_out: int) -> Tensor:
        """Perform 3D convolution (simple implementation).

        This is a simple, non-optimized implementation for educational purposes.
        In practice, you would use optimized convolution algorithms.
        """
        N, C_in, D_in, H_in, W_in = x.shape
        C_out = self.out_channels
        K_D, K_H, K_W = self.kernel_size

        # Initialize output tensor
        output = Tensor.zeros(
            (N, C_out, D_out, H_out, W_out), requires_grad=x.requires_grad
        )

        # Simple sliding window convolution
        for n in range(N):  # Batch dimension
            for c_out in range(C_out):  # Output channels
                for d_out in range(D_out):  # Output depth
                    for h_out in range(H_out):  # Output height
                        for w_out in range(W_out):  # Output width
                            # Calculate input window position
                            d_start = d_out * self.stride
                            h_start = h_out * self.stride
                            w_start = w_out * self.stride
                            d_end = d_start + K_D
                            h_end = h_start + K_H
                            w_end = w_start + K_W

                            # Extract input window
                            window = x.data[
                                n,
                                :,
                                d_start:d_end,
                                h_start:h_end,
                                w_start:w_end,
                            ]

                            # Get corresponding weight slice
                            weight_slice = self.weight.data[c_out]

                            # Compute convolution sum
                            conv_sum = np.sum(window * weight_slice)

                            # Store result
                            output.data[n, c_out, d_out, h_out, w_out] = conv_sum

        return output

    def extra_repr(self) -> str:
        """Extra representation string."""
        s = f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}"
        if self.stride != (1, 1, 1):
            s += f", stride={self.stride}"
        if self.padding != (0, 0, 0):
            s += f", padding={self.padding}"
        if self.dilation != (1, 1, 1):
            s += f", dilation={self.dilation}"
        s += f", bias={self.bias is not None}"
        return s


class ConvTranspose3D(Module):
    """3D transposed convolution layer.

    Applies a 3D transposed convolution over an input signal composed of several input planes.
    This is sometimes called "deconvolution" but is actually the gradient of convolution
    with respect to its input.

    Args:
        in_channels: Number of channels in the input image.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel. Can be a single number or a tuple of 3.
        stride: Stride of the convolution. Can be a single number or a tuple of 3. Default: 1.
        padding: Zero-padding added to all sides of the input. Can be a single number or a tuple of 3. Default: 0.
        output_padding: Additional size added to output shape. Can be a single number or a tuple of 3. Default: 0.
        groups: Number of blocked connections from input to output channels. Default: 1.
        dilation: Spacing between kernel elements. Can be a single number or a tuple of 3. Default: 1.
        bias: If True, adds a learnable bias to the output. Default: True.

    Shape:
        - Input: (N, C_in, D_in, H_in, W_in)
        - Output: (N, C_out, D_out, H_out, W_out)

        where:
            D_out = (D_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0]-1) + output_padding[0] + 1
            H_out = (H_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1]-1) + output_padding[1] + 1
            W_out = (W_in - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kernel_size[2]-1) + output_padding[2] + 1
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = (3, 3, 3),
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        output_padding: Union[int, Tuple[int, int, int]] = 0,
        groups: int = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding, output_padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups})")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by groups ({groups})")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation

        # Weight shape for grouped convolution: (in_channels, out_channels // groups, K_D, K_H, K_W)
        weight_shape = (in_channels, out_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2])
        self.weight = Tensor(
            np.random.randn(*weight_shape).astype(np.float32) * 0.1,
            requires_grad=True,
        )

        self.bias: Optional[Tensor] = None
        if bias:
            self.bias = Tensor(
                np.zeros((out_channels, 1, 1, 1), dtype=np.float32),
                requires_grad=True,
            )

        self.register_parameter("weight", self.weight)
        if self.bias is not None:
            self.register_parameter("bias", self.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"ConvTranspose3D expects 5D input (N, C, D, H, W), got {x.ndim}D")

        N, C_in, D_in, H_in, W_in = x.shape

        if C_in != self.in_channels:
            raise ValueError(f"Expected input channels {self.in_channels}, got {C_in}")

        output = ConvTranspose3DFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding,
            self.output_padding, self.dilation, self.groups
        )

        return output

    def extra_repr(self) -> str:
        s = f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}"
        if self.stride != (1, 1, 1):
            s += f", stride={self.stride}"
        if self.padding != (0, 0, 0):
            s += f", padding={self.padding}"
        if self.output_padding != (0, 0, 0):
            s += f", output_padding={self.output_padding}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.dilation != (1, 1, 1):
            s += f", dilation={self.dilation}"
        s += f", bias={self.bias is not None}"
        return s


# Functional convolution (simplified version)
def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> Tensor:
    """Functional interface for 2D convolution.

    Args:
        input: Input tensor of shape (N, C_in, H_in, W_in).
        weight: Weight tensor of shape (C_out, C_in, K_H, K_W).
        bias: Optional bias tensor of shape (C_out, 1, 1).
        stride: Stride of the convolution.
        padding: Zero-padding added to both sides.
        dilation: Spacing between kernel elements.

    Returns:
        Output tensor of shape (N, C_out, H_out, W_out).
    """
    # Create a temporary Conv2D layer with the given parameters
    C_out, C_in, K_H, K_W = weight.shape

    # Simple implementation for now - just use the layer
    layer = Conv2D(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=(K_H, K_W),
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias is not None,
    )

    # Set weights and bias
    layer.weight = weight
    layer.weight.requires_grad = weight.requires_grad

    if bias is not None:
        layer.bias = bias
        layer.bias.requires_grad = bias.requires_grad

    return layer(input)
