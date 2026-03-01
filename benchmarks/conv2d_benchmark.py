"""Quick benchmark for Conv2D performance."""
import numpy as np
import time
from nanotorch import Tensor
from nanotorch.nn import Conv2D

def benchmark_conv2d():
    # Input shape: (batch, channels, height, width)
    batch = 4
    in_channels = 3
    out_channels = 16
    height, width = 32, 32
    kernel_size = (3, 3)
    stride = 1
    padding = 1
    
    # Create random input
    x_np = np.random.randn(batch, in_channels, height, width).astype(np.float32)
    x = Tensor(x_np, requires_grad=True)
    
    # Create Conv2D layer
    conv = Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias=True)
    
    # Warmup
    for _ in range(10):
        y = conv(x)
        y.backward(Tensor.ones_like(y))
        conv.zero_grad()
    
    # Forward pass timing
    forward_times = []
    for _ in range(100):
        start = time.perf_counter()
        y = conv(x)
        forward_times.append(time.perf_counter() - start)
    
    # Backward pass timing
    y = conv(x)
    grad_output = Tensor.ones_like(y)
    backward_times = []
    for _ in range(100):
        conv.zero_grad()
        start = time.perf_counter()
        y.backward(grad_output)
        backward_times.append(time.perf_counter() - start)
    
    forward_median = np.median(forward_times) * 1000  # ms
    backward_median = np.median(backward_times) * 1000
    
    print(f"Conv2D forward median time: {forward_median:.3f} ms")
    print(f"Conv2D backward median time: {backward_median:.3f} ms")
    
    # Compare with naive NumPy implementation (no gradient)
    # Simple sliding window convolution for reference
    weight_np = conv.weight.data
    bias_np = conv.bias.data.reshape(1, out_channels, 1, 1)
    
    # Pad input
    if padding > 0:
        padded = np.pad(x_np, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
    else:
        padded = x_np
    
    H_out = (height + 2*padding - kernel_size[0]) // stride + 1
    W_out = (width + 2*padding - kernel_size[1]) // stride + 1
    
    numpy_times = []
    for _ in range(100):
        start = time.perf_counter()
        output = np.zeros((batch, out_channels, H_out, W_out), dtype=np.float32)
        for n in range(batch):
            for c_out in range(out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * stride
                        w_start = w * stride
                        window = padded[n, :, h_start:h_start+kernel_size[0], w_start:w_start+kernel_size[1]]
                        output[n, c_out, h, w] = np.sum(window * weight_np[c_out]) + bias_np[0, c_out, 0, 0]
        numpy_times.append(time.perf_counter() - start)
    
    numpy_median = np.median(numpy_times) * 1000
    print(f"NumPy naive forward median time: {numpy_median:.3f} ms")
    print(f"Overhead (nanotorch / NumPy): {forward_median / numpy_median:.2f}x")
    
    return forward_median, backward_median, numpy_median

if __name__ == "__main__":
    benchmark_conv2d()