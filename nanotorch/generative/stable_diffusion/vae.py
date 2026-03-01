"""
Stable Diffusion - Latent Diffusion Model for Text-to-Image Generation

Stable Diffusion is a latent text-to-image diffusion model capable of 
generating photo-realistic images given any text input.

Key components:
1. VAE (Variational Autoencoder) - Compresses images to latent space
2. U-Net - Denoising network with cross-attention
3. CLIP Text Encoder - Encodes text prompts (simplified here)
4. Diffusion Process - Forward/reverse diffusion in latent space

Reference:
    "High-Resolution Image Synthesis with Latent Diffusion Models"
    Rombach, Blattmann, et al.
    CVPR 2022
    https://arxiv.org/abs/2112.10752
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module, Sequential
from nanotorch.nn.conv import Conv2D
from nanotorch.nn.linear import Linear
from nanotorch.nn.activation import SiLU, GELU, Sigmoid
from nanotorch.nn.normalization import GroupNorm
from nanotorch.nn.pooling import AvgPool2d


class ResnetBlock(Module):
    """Residual block with group normalization.
    
    Architecture:
        Input -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> + Input
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 32,
        eps: float = 1e-5
    ):
        super().__init__()
        
        self.norm1 = GroupNorm(groups, in_channels, eps)
        self.act1 = SiLU()
        self.conv1 = Conv2D(in_channels, out_channels, 3, 1, 1, bias=False)
        
        self.norm2 = GroupNorm(groups, out_channels, eps)
        self.act2 = SiLU()
        self.conv2 = Conv2D(out_channels, out_channels, 3, 1, 1, bias=False)
        
        self.skip = None
        if in_channels != out_channels:
            self.skip = Conv2D(in_channels, out_channels, 1, 1, 0, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        
        if self.skip is not None:
            x = self.skip(x)
        
        return Tensor(h.data + x.data, requires_grad=x.requires_grad)


class Downsample(Module):
    """Downsampling layer using strided convolution."""
    
    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        self.conv = Conv2D(channels, channels, 3, stride, 1, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(Module):
    """Upsampling layer using nearest neighbor interpolation + convolution."""
    
    def __init__(self, channels: int, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = Conv2D(channels, channels, 3, 1, 1, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        data = x.data
        n, c, h, w = data.shape
        
        new_h = h * self.scale_factor
        new_w = w * self.scale_factor
        
        upsampled = np.zeros((n, c, new_h, new_w), dtype=data.dtype)
        
        for i in range(new_h):
            for j in range(new_w):
                src_i = min(i // self.scale_factor, h - 1)
                src_j = min(j // self.scale_factor, w - 1)
                upsampled[:, :, i, j] = data[:, :, src_i, src_j]
        
        x = Tensor(upsampled, requires_grad=x.requires_grad)
        return self.conv(x)


class SelfAttention(Module):
    """Self-attention layer for spatial features.
    
    Simplified attention mechanism:
        Q, K, V = Linear(x)
        Attention = softmax(Q @ K.T / sqrt(d)) @ V
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        head_dim: int = 64
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        inner_dim = num_heads * head_dim
        
        self.norm = GroupNorm(32, channels, 1e-5)
        self.to_q = Conv2D(channels, inner_dim, 1, 1, 0, bias=False)
        self.to_k = Conv2D(channels, inner_dim, 1, 1, 0, bias=False)
        self.to_v = Conv2D(channels, inner_dim, 1, 1, 0, bias=False)
        self.to_out = Conv2D(inner_dim, channels, 1, 1, 0, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        
        q = self.to_q(h)
        k = self.to_k(h)
        v = self.to_v(h)
        
        n, c, hh, ww = q.shape
        
        q = q.data.reshape(n, self.num_heads, self.head_dim, hh * ww)
        k = k.data.reshape(n, self.num_heads, self.head_dim, hh * ww)
        v = v.data.reshape(n, self.num_heads, self.head_dim, hh * ww)
        
        attn = np.zeros((n, self.num_heads, hh * ww, hh * ww), dtype=np.float32)
        out = np.zeros_like(q)
        
        for b in range(n):
            for head in range(self.num_heads):
                q_head = q[b, head]
                k_head = k[b, head]
                v_head = v[b, head]
                
                scores = q_head.T @ k_head * self.scale
                
                scores_max = scores.max(axis=1, keepdims=True)
                exp_scores = np.exp(scores - scores_max)
                attn_weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)
                
                out[b, head] = (attn_weights @ v_head.T).T
        
        out = out.reshape(n, c, hh, ww)
        out = Tensor(out, requires_grad=x.requires_grad)
        out = self.to_out(out)
        
        return Tensor(out.data + x.data, requires_grad=x.requires_grad)


class Encoder(Module):
    """VAE Encoder.
    
    Compresses images from (B, 3, H, W) to latent space (B, latent_dim, H/8, W/8).
    
    Architecture:
        - Initial convolution
        - 3 downsampling stages with ResNet blocks
        - Mid block with attention
        - Output to latent distribution parameters (mean, log_var)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 8,
        block_out_channels: Tuple[int, ...] = (128, 256, 512),
        layers_per_block: int = 2
    ):
        super().__init__()
        
        self.conv_in = Conv2D(in_channels, block_out_channels[0], 3, 1, 1)
        
        self.down_blocks = []
        self.down_samples = []
        
        ch = block_out_channels[0]
        for i, out_ch in enumerate(block_out_channels):
            for _ in range(layers_per_block):
                self.down_blocks.append(ResnetBlock(ch, out_ch))
                self.register_module(f'down_block_{len(self.down_blocks)-1}', self.down_blocks[-1])
                ch = out_ch
            
            self.down_samples.append(Downsample(ch))
            self.register_module(f'down_sample_{len(self.down_samples)-1}', self.down_samples[-1])
        
        self.mid_block_1 = ResnetBlock(ch, ch)
        self.mid_attn = SelfAttention(ch)
        self.mid_block_2 = ResnetBlock(ch, ch)
        
        self.conv_norm_out = GroupNorm(32, ch, 1e-5)
        self.conv_act = SiLU()
        self.conv_out = Conv2D(ch, out_channels * 2, 3, 1, 1)
        
        self.out_channels = out_channels
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.conv_in(x)
        
        block_idx = 0
        sample_idx = 0
        
        for i, _ in enumerate([128, 256, 512]):
            for _ in range(2):
                h = self.down_blocks[block_idx](h)
                block_idx += 1
            
            h = self.down_samples[sample_idx](h)
            sample_idx += 1
        
        h = self.mid_block_1(h)
        h = self.mid_attn(h)
        h = self.mid_block_2(h)
        
        h = self.conv_norm_out(h)
        h = self.conv_act(h)
        h = self.conv_out(h)
        
        mean = Tensor(h.data[:, :self.out_channels], requires_grad=x.requires_grad)
        log_var = Tensor(h.data[:, self.out_channels:], requires_grad=x.requires_grad)
        
        return mean, log_var


class Decoder(Module):
    """VAE Decoder.
    
    Reconstructs images from latent space (B, latent_dim, H/8, W/8) to (B, 3, H, W).
    
    Architecture:
        - Initial convolution from latent
        - Mid block with attention
        - 3 upsampling stages with ResNet blocks
        - Output convolution to RGB
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (128, 256, 512),
        layers_per_block: int = 2
    ):
        super().__init__()
        
        reversed_channels = list(reversed(block_out_channels))
        ch = reversed_channels[0]
        
        self.conv_in = Conv2D(in_channels, ch, 3, 1, 1)
        
        self.mid_block_1 = ResnetBlock(ch, ch)
        self.mid_attn = SelfAttention(ch)
        self.mid_block_2 = ResnetBlock(ch, ch)
        
        self.up_blocks = []
        self.up_samples = []
        
        for i, out_ch in enumerate(reversed_channels):
            for j in range(layers_per_block + 1):
                if i == len(reversed_channels) - 1 and j == layers_per_block:
                    final_out_ch = out_ch
                else:
                    final_out_ch = out_ch
                
                self.up_blocks.append(ResnetBlock(ch, final_out_ch))
                self.register_module(f'up_block_{len(self.up_blocks)-1}', self.up_blocks[-1])
                ch = final_out_ch
            
            self.up_samples.append(Upsample(ch))
            self.register_module(f'up_sample_{len(self.up_samples)-1}', self.up_samples[-1])
        
        self.conv_norm_out = GroupNorm(32, ch, 1e-5)
        self.conv_act = SiLU()
        self.conv_out = Conv2D(ch, out_channels, 3, 1, 1)
    
    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        
        h = self.mid_block_1(h)
        h = self.mid_attn(h)
        h = self.mid_block_2(h)
        
        block_idx = 0
        sample_idx = 0
        
        reversed_channels = [512, 256, 128]
        
        for i, _ in enumerate(reversed_channels):
            for j in range(3):
                h = self.up_blocks[block_idx](h)
                block_idx += 1
            
            h = self.up_samples[sample_idx](h)
            sample_idx += 1
        
        h = self.conv_norm_out(h)
        h = self.conv_act(h)
        h = self.conv_out(h)
        
        return h


class VAE(Module):
    """Variational Autoencoder for Stable Diffusion.
    
    Encodes images to latent space and decodes back.
    Uses reparameterization trick for training.
    
    Args:
        in_channels: Number of input image channels (default: 3 for RGB)
        latent_channels: Number of latent channels (default: 4)
        out_channels: Number of output channels (default: 3 for RGB)
        scaling_factor: Latent space scaling factor (default: 0.18215)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        out_channels: int = 3,
        scaling_factor: float = 0.18215
    ):
        super().__init__()
        
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=(128, 256, 512),
            layers_per_block=2
        )
        
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=(128, 256, 512),
            layers_per_block=2
        )
        
        self.scaling_factor = scaling_factor
        self.latent_channels = latent_channels
    
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode image to latent distribution parameters.
        
        Args:
            x: Input image tensor (N, 3, H, W)
        
        Returns:
            Tuple of (mean, log_variance) of latent distribution
        """
        mean, log_var = self.encoder(x)
        
        mean = Tensor(mean.data * self.scaling_factor, requires_grad=x.requires_grad)
        log_var = Tensor(log_var.data + 2 * np.log(self.scaling_factor), requires_grad=x.requires_grad)
        
        return mean, log_var
    
    def decode(self, z: Tensor) -> Tensor:
        """Decode latent to image.
        
        Args:
            z: Latent tensor (N, latent_channels, H/8, W/8)
        
        Returns:
            Reconstructed image (N, 3, H, W)
        """
        z_scaled = Tensor(z.data / self.scaling_factor, requires_grad=z.requires_grad)
        return self.decoder(z_scaled)
    
    def sample(self, mean: Tensor, log_var: Tensor) -> Tensor:
        """Sample from latent distribution using reparameterization trick.
        
        Args:
            mean: Mean of latent distribution
            log_var: Log variance of latent distribution
        
        Returns:
            Sampled latent tensor
        """
        std = Tensor(np.exp(0.5 * log_var.data), requires_grad=mean.requires_grad)
        
        noise = Tensor(
            np.random.randn(*mean.data.shape).astype(np.float32),
            requires_grad=mean.requires_grad
        )
        
        z = Tensor(mean.data + std.data * noise.data, requires_grad=mean.requires_grad)
        
        return z
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Full VAE forward pass.
        
        Args:
            x: Input image tensor (N, 3, H, W)
        
        Returns:
            Tuple of (reconstructed_image, mean, log_variance)
        """
        mean, log_var = self.encode(x)
        z = self.sample(mean, log_var)
        x_recon = self.decode(z)
        
        return x_recon, mean, log_var


def build_vae(
    in_channels: int = 3,
    latent_channels: int = 4,
    out_channels: int = 3
) -> VAE:
    """Build VAE model.
    
    Args:
        in_channels: Number of input channels
        latent_channels: Number of latent channels
        out_channels: Number of output channels
    
    Returns:
        VAE model
    """
    return VAE(
        in_channels=in_channels,
        latent_channels=latent_channels,
        out_channels=out_channels
    )
