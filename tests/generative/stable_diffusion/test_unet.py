"""
Unit tests for Stable Diffusion U-Net components.
"""

import pytest
import numpy as np
from nanotorch.tensor import Tensor
from nanotorch.generative.stable_diffusion.unet import (
    get_timestep_embedding,
    TimestepEmbedding,
    TimeEmbedResnetBlock,
    CrossAttention,
    TransformerBlock,
    DownBlock2D,
    UpBlock2D,
    UNet2DConditionModel,
    build_unet,
)


class TestGetTimestepEmbedding:
    """Tests for sinusoidal timestep embedding."""
    
    def test_output_shape(self):
        """Test output shape matches embedding dimension."""
        timesteps = np.array([0, 100, 500, 999])
        embedding_dim = 128
        
        emb = get_timestep_embedding(timesteps, embedding_dim)
        
        assert emb.shape == (4, embedding_dim)
    
    def test_embedding_dim_odd(self):
        """Test embedding with odd dimension."""
        timesteps = np.array([100])
        embedding_dim = 127
        
        emb = get_timestep_embedding(timesteps, embedding_dim)
        
        assert emb.shape == (1, embedding_dim)
    
    def test_different_timesteps_different_embeddings(self):
        """Test that different timesteps produce different embeddings."""
        t1 = np.array([0])
        t2 = np.array([500])
        
        emb1 = get_timestep_embedding(t1, 128)
        emb2 = get_timestep_embedding(t2, 128)
        
        assert not np.allclose(emb1, emb2)
    
    def test_batch_processing(self):
        """Test that batch processing is equivalent to individual."""
        timesteps = np.array([100, 200])
        
        batch_emb = get_timestep_embedding(timesteps, 64)
        single_emb1 = get_timestep_embedding(np.array([100]), 64)
        single_emb2 = get_timestep_embedding(np.array([200]), 64)
        
        np.testing.assert_allclose(batch_emb[0], single_emb1[0], rtol=1e-5)
        np.testing.assert_allclose(batch_emb[1], single_emb2[0], rtol=1e-5)


class TestTimestepEmbedding:
    """Tests for TimestepEmbedding module."""
    
    def test_init(self):
        """Test initialization."""
        temb = TimestepEmbedding(128, 512)
        
        assert temb is not None
        assert temb.linear_1 is not None
        assert temb.linear_2 is not None
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        temb = TimestepEmbedding(128, 512)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))
        
        out = temb(x)
        
        assert out.shape == (2, 512)


class TestTimeEmbedResnetBlock:
    """Tests for ResNet block with time embedding."""
    
    def test_init(self):
        """Test initialization."""
        block = TimeEmbedResnetBlock(64, 128, 512)
        
        assert block is not None
        assert block.time_emb_proj is not None
    
    def test_forward_without_temb(self):
        """Test forward pass without time embedding."""
        block = TimeEmbedResnetBlock(64, 64, 0)
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        
        out = block(x)
        
        assert out.shape == (2, 64, 16, 16)
    
    def test_forward_with_temb(self):
        """Test forward pass with time embedding."""
        block = TimeEmbedResnetBlock(64, 128, 512)
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        temb = Tensor(np.random.randn(2, 512).astype(np.float32))
        
        out = block(x, temb)
        
        assert out.shape == (2, 128, 16, 16)
    
    def test_skip_connection(self):
        """Test skip connection for channel mismatch."""
        block = TimeEmbedResnetBlock(64, 128, 256)
        
        assert block.skip is not None


class TestCrossAttention:
    """Tests for CrossAttention module."""
    
    def test_init(self):
        """Test initialization."""
        attn = CrossAttention(query_dim=256)
        
        assert attn is not None
        assert attn.heads == 8
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        attn = CrossAttention(query_dim=256, cross_attention_dim=512)
        x = Tensor(np.random.randn(2, 256, 16, 16).astype(np.float32))
        context = Tensor(np.random.randn(2, 77, 512).astype(np.float32))
        
        out = attn(x, context)
        
        assert out.shape == x.shape
    
    def test_forward_without_context(self):
        """Test forward pass without context (self-attention)."""
        attn = CrossAttention(query_dim=128, cross_attention_dim=128)
        x = Tensor(np.random.randn(1, 128, 8, 8).astype(np.float32))
        
        out = attn(x)
        
        assert out.shape == x.shape


class TestTransformerBlock:
    """Tests for TransformerBlock module."""
    
    def test_init(self):
        """Test initialization."""
        block = TransformerBlock(dim=256)
        
        assert block is not None
        assert block.attn1 is not None
        assert block.attn2 is not None
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        block = TransformerBlock(dim=256, cross_attention_dim=512)
        x = Tensor(np.random.randn(2, 256, 16, 16).astype(np.float32))
        context = Tensor(np.random.randn(2, 77, 512).astype(np.float32))
        
        out = block(x, context)
        
        assert out.shape == x.shape


class TestDownBlock2D:
    """Tests for DownBlock2D module."""
    
    def test_init(self):
        """Test initialization."""
        block = DownBlock2D(64, 128, 512, num_layers=2)
        
        assert block is not None
        assert len(block.resnets) == 2
    
    def test_forward_without_attention(self):
        """Test forward pass without attention."""
        block = DownBlock2D(64, 128, 512, num_layers=2, use_attention=False)
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        temb = Tensor(np.random.randn(2, 512).astype(np.float32))
        
        out, states = block(x, temb)
        
        assert out.shape == (2, 128, 16, 16)
        assert len(states) == 2
    
    def test_forward_with_attention(self):
        """Test forward pass with attention."""
        block = DownBlock2D(
            64, 128, 512, 
            num_layers=2, 
            use_attention=True,
            cross_attention_dim=512
        )
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        temb = Tensor(np.random.randn(2, 512).astype(np.float32))
        context = Tensor(np.random.randn(2, 77, 512).astype(np.float32))
        
        out, states = block(x, temb, context)
        
        assert out.shape == (2, 128, 16, 16)


class TestUpBlock2D:
    """Tests for UpBlock2D module."""
    
    def test_init(self):
        """Test initialization."""
        block = UpBlock2D(128, 64, 128, 512, num_layers=2)
        
        assert block is not None
        assert len(block.resnets) == 2
    
    def test_forward(self):
        """Test forward pass with skip connections."""
        block = UpBlock2D(
            128, 64, 128, 512,
            num_layers=2,
            use_attention=False
        )
        x = Tensor(np.random.randn(2, 128, 16, 16).astype(np.float32))
        res = [Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32)) for _ in range(2)]
        temb = Tensor(np.random.randn(2, 512).astype(np.float32))
        
        out = block(x, res, temb)
        
        assert out.shape == (2, 64, 16, 16)


class TestUNet2DConditionModel:
    """Tests for full U-Net model."""
    
    def test_init(self):
        """Test U-Net initialization."""
        unet = UNet2DConditionModel()
        
        assert unet is not None
        assert unet.conv_in is not None
        assert unet.conv_out is not None
        assert len(unet.down_blocks) > 0
        assert len(unet.up_blocks) > 0
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        unet = UNet2DConditionModel(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            block_out_channels=(64, 128, 256, 256),
            cross_attention_dim=256
        )
        sample = Tensor(np.random.randn(2, 4, 32, 32).astype(np.float32))
        timestep = np.array([100, 500])
        context = Tensor(np.random.randn(2, 77, 256).astype(np.float32))
        
        out = unet(sample, timestep, context)
        
        assert out.shape == sample.shape
    
    def test_forward_without_context(self):
        """Test forward pass without text conditioning."""
        unet = UNet2DConditionModel(
            sample_size=16,
            block_out_channels=(64, 128, 256, 256)
        )
        sample = Tensor(np.random.randn(1, 4, 16, 16).astype(np.float32))
        timestep = np.array([500])
        
        out = unet(sample, timestep)
        
        assert out.shape == sample.shape
    
    def test_parameters_not_empty(self):
        """Test that U-Net has parameters."""
        unet = UNet2DConditionModel()
        params = list(unet.parameters())
        
        assert len(params) > 0
    
    def test_state_dict(self):
        """Test state dict can be extracted."""
        unet = UNet2DConditionModel()
        state_dict = unet.state_dict()
        
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

    def test_backward_propagates_to_parameters(self):
        """Test a small U-Net keeps parameter gradients connected."""
        unet = UNet2DConditionModel(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            block_out_channels=(32, 64, 64, 64),
            cross_attention_dim=32,
        )
        sample = Tensor(np.random.randn(1, 4, 8, 8).astype(np.float32), requires_grad=True)
        timestep = np.array([10])
        context = Tensor(np.random.randn(1, 8, 32).astype(np.float32), requires_grad=True)

        unet.zero_grad()
        loss = unet(sample, timestep, context).sum()
        loss.backward()

        grads = [param.grad for param in unet.parameters() if param.grad is not None]
        assert grads
        assert any(np.any(np.abs(grad.data) > 0) for grad in grads)


class TestBuildUNet:
    """Tests for U-Net builder function."""
    
    def test_build_unet_default(self):
        """Test building U-Net with defaults."""
        unet = build_unet()
        
        assert isinstance(unet, UNet2DConditionModel)
        assert unet.in_channels == 4
        assert unet.out_channels == 4
    
    def test_build_unet_custom(self):
        """Test building U-Net with custom parameters."""
        unet = build_unet(
            sample_size=32,
            in_channels=8,
            out_channels=8,
            cross_attention_dim=768
        )
        
        assert isinstance(unet, UNet2DConditionModel)
        assert unet.in_channels == 8
        assert unet.out_channels == 8
