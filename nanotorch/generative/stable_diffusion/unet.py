"""
Stable Diffusion U-Net - Denoising Network

The U-Net is the core denoising network in Stable Diffusion.
It predicts the noise in latent space given:
- Noisy latent (z_t)
- Timestep (t)
- Text embedding (c)

Key features:
- Time embedding via sinusoidal positional encoding
- Cross-attention for conditioning on text
- Skip connections between encoder and decoder
- Self-attention in bottleneck

Reference:
    "High-Resolution Image Synthesis with Latent Diffusion Models"
    Rombach, Blattmann, et al.
    CVPR 2022
"""

import numpy as np
from typing import Tuple, List, Optional

from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module
from nanotorch.nn.conv import Conv2D
from nanotorch.nn.linear import Linear
from nanotorch.nn.activation import SiLU, GELU
from nanotorch.nn.normalization import GroupNorm


def get_timestep_embedding(timesteps: np.ndarray, embedding_dim: int) -> np.ndarray:
    """Sinusoidal timestep embedding.
    
    Args:
        timesteps: Array of timestep values (batch_size,)
        embedding_dim: Dimension of output embedding
    
    Returns:
        Timestep embeddings of shape (batch_size, embedding_dim)
    """
    assert len(timesteps.shape) == 1
    
    half_dim = embedding_dim // 2
    emb = np.log(10000.0) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim, dtype=np.float32) * -emb)
    
    emb = timesteps[:, np.newaxis] * emb[np.newaxis, :]
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
    
    if embedding_dim % 2 == 1:
        emb = np.pad(emb, ((0, 0), (0, 1)), mode='constant')
    
    return emb


class TimestepEmbedding(Module):
    """Timestep embedding module.
    
    Projects sinusoidal embedding to model dimension.
    """
    
    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        
        self.linear_1 = Linear(in_channels, time_embed_dim)
        
        if act_fn == "silu":
            self.act = SiLU()
        elif act_fn == "gelu":
            self.act = GELU()
        else:
            self.act = SiLU()
        
        self.linear_2 = Linear(time_embed_dim, time_embed_dim)
    
    def forward(self, sample: Tensor) -> Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class TimeEmbedResnetBlock(Module):
    """ResNet block with time embedding injection.
    
    Architecture:
        Input -> GN -> SiLU -> Conv -> GN -> SiLU -> Conv -> + TimeEmb -> + Input
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
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
        
        if temb_channels > 0:
            self.time_emb_proj = Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None
        
        self.skip = None
        if in_channels != out_channels:
            self.skip = Conv2D(in_channels, out_channels, 1, 1, 0, bias=False)
    
    def forward(self, x: Tensor, temb: Optional[Tensor] = None) -> Tensor:
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        if self.time_emb_proj is not None and temb is not None:
            temb_out = self.time_emb_proj(temb)
            temb_out = Tensor(
                temb_out.data[:, :, np.newaxis, np.newaxis],
                requires_grad=temb.requires_grad
            )
            h = Tensor(h.data + temb_out.data, requires_grad=h.requires_grad)
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        
        if self.skip is not None:
            x = self.skip(x)
        
        return Tensor(h.data + x.data, requires_grad=x.requires_grad)


class CrossAttention(Module):
    """Cross-attention for conditioning.
    
    Query from spatial features, Key/Value from text embedding.
    
    Simplified implementation for educational purposes.
    """
    
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int = 512,
        heads: int = 8,
        dim_head: int = 64
    ):
        super().__init__()
        
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        inner_dim = heads * dim_head
        
        self.to_q = Linear(query_dim, inner_dim, bias=False)
        self.to_k = Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_out = Linear(inner_dim, query_dim)
    
    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        batch_size, channels, height, width = x.shape
        
        x_flat = x.data.transpose(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        x_flat = Tensor(x_flat, requires_grad=x.requires_grad)
        
        if context is None:
            context = x_flat
        
        q = self.to_q(x_flat).data
        k = self.to_k(context).data
        v = self.to_v(context).data
        
        q = q.reshape(batch_size, height * width, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        
        attn = np.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        
        attn_max = attn.max(axis=-1, keepdims=True)
        attn = np.exp(attn - attn_max)
        attn = attn / attn.sum(axis=-1, keepdims=True)
        
        out = np.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, height * width, -1)
        out = Tensor(out, requires_grad=x.requires_grad)
        out = self.to_out(out)
        
        out = out.data.reshape(batch_size, height, width, -1).transpose(0, 3, 1, 2)
        out = Tensor(out, requires_grad=x.requires_grad)
        
        return out


class TransformerBlock(Module):
    """Transformer block with self and cross attention."""
    
    def __init__(
        self,
        dim: int,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 512
    ):
        super().__init__()
        
        self.norm1 = GroupNorm(32, dim, 1e-5)
        self.attn1 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim
        )
        
        self.norm2 = GroupNorm(32, dim, 1e-5)
        self.attn2 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim
        )
        
        self.norm3 = GroupNorm(32, dim, 1e-5)
        self.act = GELU()
        self.ff1 = Linear(dim, dim * 4)
        self.ff2 = Linear(dim * 4, dim)
    
    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        h = self.norm1(x)
        h = self.attn1(h)
        x = Tensor(x.data + h.data, requires_grad=x.requires_grad)
        
        h = self.norm2(x)
        if context is not None:
            h = self.attn2(h, context)
            x = Tensor(x.data + h.data, requires_grad=x.requires_grad)
        
        h = self.norm3(x)
        
        batch_size, channels, height, width = h.shape
        h_flat = h.data.transpose(0, 2, 3, 1).reshape(batch_size, -1, channels)
        h_flat = Tensor(h_flat, requires_grad=h.requires_grad)
        
        h_flat = self.ff1(h_flat)
        h_flat = self.act(h_flat)
        h_flat = self.ff2(h_flat)
        
        h = h_flat.data.reshape(batch_size, height, width, -1).transpose(0, 3, 1, 2)
        x = Tensor(x.data + h, requires_grad=x.requires_grad)
        
        return x


class DownBlock2D(Module):
    """Downsampling block with ResNet and optional transformer."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 2,
        use_attention: bool = False,
        cross_attention_dim: int = 512
    ):
        super().__init__()
        
        self.resnets = []
        self.attentions = []
        
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            resnet = TimeEmbedResnetBlock(in_ch, out_channels, temb_channels)
            self.resnets.append(resnet)
            self.register_module(f'resnet_{i}', resnet)
            
            if use_attention:
                attn = TransformerBlock(
                    dim=out_channels,
                    cross_attention_dim=cross_attention_dim
                )
                self.attentions.append(attn)
                self.register_module(f'attention_{i}', attn)
    
    def forward(
        self,
        x: Tensor,
        temb: Tensor,
        context: Optional[Tensor] = None
    ) -> Tuple[Tensor, List[Tensor]]:
        output_states = []
        
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, temb)
            
            if i < len(self.attentions):
                x = self.attentions[i](x, context)
            
            output_states.append(x)
        
        return x, output_states


class UpBlock2D(Module):
    """Upsampling block with ResNet and optional transformer."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channels: int,
        temb_channels: int,
        num_layers: int = 2,
        use_attention: bool = False,
        cross_attention_dim: int = 512
    ):
        super().__init__()
        
        self.resnets = []
        self.attentions = []
        
        for i in range(num_layers):
            resnet_in_channels = prev_output_channels if i == 0 else out_channels
            
            resnet = TimeEmbedResnetBlock(
                resnet_in_channels + out_channels,
                out_channels,
                temb_channels
            )
            self.resnets.append(resnet)
            self.register_module(f'resnet_{i}', resnet)
            
            if use_attention:
                attn = TransformerBlock(
                    dim=out_channels,
                    cross_attention_dim=cross_attention_dim
                )
                self.attentions.append(attn)
                self.register_module(f'attention_{i}', attn)
    
    def forward(
        self,
        x: Tensor,
        res_hidden_states: List[Tensor],
        temb: Tensor,
        context: Optional[Tensor] = None
    ) -> Tensor:
        for i, resnet in enumerate(self.resnets):
            res_state = res_hidden_states[-(i + 1)]
            x = Tensor(
                np.concatenate([x.data, res_state.data], axis=1),
                requires_grad=x.requires_grad
            )
            x = resnet(x, temb)
            
            if i < len(self.attentions):
                x = self.attentions[i](x, context)
        
        return x


class UNet2DConditionModel(Module):
    """Conditional U-Net for Stable Diffusion.
    
    Predicts noise in latent space conditioned on text embeddings.
    
    Architecture:
        - Time embedding projection
        - Initial convolution
        - 4 downsampling stages
        - Mid block with attention
        - 4 upsampling stages with skip connections
        - Output convolution
    
    Args:
        sample_size: Size of input latent (default: 64)
        in_channels: Number of input channels (default: 4)
        out_channels: Number of output channels (default: 4)
        block_out_channels: Channel dimensions for each stage
        cross_attention_dim: Dimension of text embeddings
        attention_head_dim: Dimension per attention head
    """
    
    def __init__(
        self,
        sample_size: int = 64,
        in_channels: int = 4,
        out_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        cross_attention_dim: int = 512,
        attention_head_dim: int = 8
    ):
        super().__init__()
        
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        
        time_embed_dim = block_out_channels[0] * 4
        
        self.time_proj = Conv2D(in_channels, block_out_channels[0], 3, 1, 1)
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim)
        
        self.conv_in = Conv2D(in_channels, block_out_channels[0], 3, 1, 1)
        
        self.down_blocks = []
        
        output_channel = block_out_channels[0]
        for i, channel in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = channel
            
            use_attention = i >= 1
            
            down_block = DownBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_layers=2,
                use_attention=use_attention,
                cross_attention_dim=cross_attention_dim
            )
            self.down_blocks.append(down_block)
            self.register_module(f'down_block_{i}', down_block)
        
        self.mid_block_1 = TimeEmbedResnetBlock(
            block_out_channels[-1],
            block_out_channels[-1],
            time_embed_dim
        )
        self.mid_attn = TransformerBlock(
            dim=block_out_channels[-1],
            cross_attention_dim=cross_attention_dim
        )
        self.mid_block_2 = TimeEmbedResnetBlock(
            block_out_channels[-1],
            block_out_channels[-1],
            time_embed_dim
        )
        
        self.up_blocks = []
        
        reversed_channels = list(reversed(block_out_channels))
        prev_output_channel = block_out_channels[-1]
        
        for i, channel in enumerate(reversed_channels):
            output_channel = channel
            
            use_attention = i < len(block_out_channels) - 1
            
            up_block = UpBlock2D(
                in_channels=prev_output_channel if i == 0 else reversed_channels[i - 1],
                out_channels=output_channel,
                prev_output_channels=prev_output_channel,
                temb_channels=time_embed_dim,
                num_layers=2,
                use_attention=use_attention,
                cross_attention_dim=cross_attention_dim
            )
            self.up_blocks.append(up_block)
            self.register_module(f'up_block_{i}', up_block)
            
            prev_output_channel = output_channel
        
        self.conv_norm_out = GroupNorm(32, block_out_channels[0], 1e-5)
        self.conv_act = SiLU()
        self.conv_out = Conv2D(block_out_channels[0], out_channels, 3, 1, 1)
    
    def forward(
        self,
        sample: Tensor,
        timestep: np.ndarray,
        encoder_hidden_states: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass of the U-Net.
        
        Args:
            sample: Noisy latent tensor (N, C, H, W)
            timestep: Timestep values (N,)
            encoder_hidden_states: Text embeddings (N, seq_len, cross_attention_dim)
        
        Returns:
            Predicted noise tensor (N, C, H, W)
        """
        timesteps = timestep
        
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])
        t_emb = Tensor(t_emb.astype(np.float32), requires_grad=sample.requires_grad)
        
        t_emb = self.time_embedding(t_emb)
        
        sample = self.conv_in(sample)
        
        down_block_res_samples = [sample]
        
        for down_block in self.down_blocks:
            sample, res_samples = down_block(sample, t_emb, encoder_hidden_states)
            down_block_res_samples.extend(res_samples)
        
        sample = self.mid_block_1(sample, t_emb)
        sample = self.mid_attn(sample, encoder_hidden_states)
        sample = self.mid_block_2(sample, t_emb)
        
        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-2:]
            down_block_res_samples = down_block_res_samples[:-2]
            
            sample = up_block(sample, res_samples, t_emb, encoder_hidden_states)
        
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        
        return sample


def build_unet(
    sample_size: int = 64,
    in_channels: int = 4,
    out_channels: int = 4,
    cross_attention_dim: int = 512
) -> UNet2DConditionModel:
    """Build U-Net model.
    
    Args:
        sample_size: Size of input latent
        in_channels: Number of input channels
        out_channels: Number of output channels
        cross_attention_dim: Dimension of text embeddings
    
    Returns:
        UNet2DConditionModel
    """
    return UNet2DConditionModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        cross_attention_dim=cross_attention_dim
    )
