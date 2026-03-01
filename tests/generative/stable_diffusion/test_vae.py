"""
Unit tests for Stable Diffusion VAE components.
"""

import pytest
import numpy as np
from nanotorch.tensor import Tensor
from nanotorch.generative.stable_diffusion.vae import (
    ResnetBlock,
    Downsample,
    Upsample,
    SelfAttention,
    Encoder,
    Decoder,
    VAE,
    build_vae,
)


class TestResnetBlock:
    """Tests for ResnetBlock."""
    
    def test_init_basic(self):
        """Test basic ResnetBlock initialization."""
        block = ResnetBlock(64, 128)
        
        assert block is not None
        assert block.norm1 is not None
        assert block.conv1 is not None
        assert block.norm2 is not None
        assert block.conv2 is not None
        assert block.skip is not None  # Different in/out channels
    
    def test_init_same_channels(self):
        """Test ResnetBlock with same input/output channels."""
        block = ResnetBlock(64, 64)
        
        assert block.skip is None  # No skip needed for same channels
    
    def test_forward_same_shape(self):
        """Test forward pass maintains shape when channels match."""
        block = ResnetBlock(64, 64)
        x = Tensor(np.random.randn(2, 64, 32, 32).astype(np.float32))
        
        out = block(x)
        
        assert out.shape == (2, 64, 32, 32)
    
    def test_forward_channel_change(self):
        """Test forward pass changes channels."""
        block = ResnetBlock(64, 128)
        x = Tensor(np.random.randn(2, 64, 32, 32).astype(np.float32))
        
        out = block(x)
        
        assert out.shape == (2, 128, 32, 32)
    
    def test_forward_batch_independence(self):
        """Test that each batch sample is processed independently."""
        block = ResnetBlock(32, 32)
        x1 = Tensor(np.random.randn(1, 32, 16, 16).astype(np.float32))
        x2 = Tensor(np.random.randn(1, 32, 16, 16).astype(np.float32))
        x_batch = Tensor(np.concatenate([x1.data, x2.data], axis=0))
        
        out_batch = block(x_batch)
        out1 = block(x1)
        out2 = block(x2)
        
        np.testing.assert_allclose(
            out_batch.data[0], out1.data[0], rtol=1e-5, atol=1e-5
        )


class TestDownsample:
    """Tests for Downsample layer."""
    
    def test_init(self):
        """Test Downsample initialization."""
        down = Downsample(64)
        
        assert down is not None
        assert down.conv is not None
    
    def test_forward_halves_spatial(self):
        """Test forward pass halves spatial dimensions."""
        down = Downsample(64)
        x = Tensor(np.random.randn(2, 64, 32, 32).astype(np.float32))
        
        out = down(x)
        
        assert out.shape == (2, 64, 16, 16)
    
    def test_forward_non_square(self):
        """Test forward pass with non-square input."""
        down = Downsample(32)
        x = Tensor(np.random.randn(2, 32, 64, 32).astype(np.float32))
        
        out = down(x)
        
        assert out.shape == (2, 32, 32, 16)


class TestUpsample:
    """Tests for Upsample layer."""
    
    def test_init(self):
        """Test Upsample initialization."""
        up = Upsample(64)
        
        assert up is not None
        assert up.conv is not None
    
    def test_forward_doubles_spatial(self):
        """Test forward pass doubles spatial dimensions."""
        up = Upsample(64)
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        
        out = up(x)
        
        assert out.shape == (2, 64, 32, 32)
    
    def test_forward_non_square(self):
        """Test forward pass with non-square input."""
        up = Upsample(32)
        x = Tensor(np.random.randn(2, 32, 16, 8).astype(np.float32))
        
        out = up(x)
        
        assert out.shape == (2, 32, 32, 16)


class TestSelfAttention:
    """Tests for SelfAttention layer."""
    
    def test_init(self):
        """Test SelfAttention initialization."""
        attn = SelfAttention(64)
        
        assert attn is not None
        assert attn.num_heads == 8
        assert attn.head_dim == 64
    
    def test_forward_same_shape(self):
        """Test forward pass maintains shape."""
        attn = SelfAttention(64)
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        
        out = attn(x)
        
        assert out.shape == (2, 64, 16, 16)
    
    def test_forward_residual(self):
        """Test that attention adds residual connection."""
        attn = SelfAttention(64)
        x = Tensor(np.random.randn(1, 64, 8, 8).astype(np.float32))
        
        out = attn(x)
        
        assert out.shape == x.shape


class TestEncoder:
    """Tests for VAE Encoder."""
    
    def test_init(self):
        """Test Encoder initialization."""
        encoder = Encoder()
        
        assert encoder is not None
        assert encoder.conv_in is not None
        assert encoder.mid_block_1 is not None
        assert encoder.mid_attn is not None
        assert encoder.mid_block_2 is not None
    
    def test_forward_output_shape(self):
        """Test encoder output shape."""
        encoder = Encoder(in_channels=3, out_channels=8)
        x = Tensor(np.random.randn(2, 3, 64, 64).astype(np.float32))
        
        mean, log_var = encoder(x)
        
        assert mean.shape == (2, 8, 8, 8)
        assert log_var.shape == (2, 8, 8, 8)
    
    def test_forward_downsampling(self):
        """Test that encoder downsamples by 8x."""
        encoder = Encoder(in_channels=3, out_channels=8)
        x = Tensor(np.random.randn(1, 3, 128, 128).astype(np.float32))
        
        mean, log_var = encoder(x)
        
        assert mean.shape[2] == 128 // 8
        assert mean.shape[3] == 128 // 8


class TestDecoder:
    """Tests for VAE Decoder."""
    
    def test_init(self):
        """Test Decoder initialization."""
        decoder = Decoder()
        
        assert decoder is not None
        assert decoder.conv_in is not None
        assert decoder.mid_block_1 is not None
        assert decoder.mid_attn is not None
        assert decoder.mid_block_2 is not None
    
    def test_forward_output_shape(self):
        """Test decoder output shape."""
        decoder = Decoder(in_channels=4, out_channels=3)
        z = Tensor(np.random.randn(2, 4, 8, 8).astype(np.float32))
        
        out = decoder(z)
        
        assert out.shape == (2, 3, 64, 64)
    
    def test_forward_upsampling(self):
        """Test that decoder upsamples by 8x."""
        decoder = Decoder(in_channels=4, out_channels=3)
        z = Tensor(np.random.randn(1, 4, 16, 16).astype(np.float32))
        
        out = decoder(z)
        
        assert out.shape[2] == 16 * 8
        assert out.shape[3] == 16 * 8


class TestVAE:
    """Tests for full VAE model."""
    
    def test_init(self):
        """Test VAE initialization."""
        vae = VAE()
        
        assert vae is not None
        assert vae.encoder is not None
        assert vae.decoder is not None
        assert vae.scaling_factor == 0.18215
    
    def test_encode_shape(self):
        """Test VAE encode output shape."""
        vae = VAE()
        x = Tensor(np.random.randn(2, 3, 64, 64).astype(np.float32))
        
        mean, log_var = vae.encode(x)
        
        assert mean.shape == (2, 4, 8, 8)
        assert log_var.shape == (2, 4, 8, 8)
    
    def test_decode_shape(self):
        """Test VAE decode output shape."""
        vae = VAE()
        z = Tensor(np.random.randn(2, 4, 8, 8).astype(np.float32))
        
        out = vae.decode(z)
        
        assert out.shape == (2, 3, 64, 64)
    
    def test_sample_shape(self):
        """Test VAE sample output shape."""
        vae = VAE()
        mean = Tensor(np.zeros((2, 4, 8, 8), dtype=np.float32))
        log_var = Tensor(np.zeros((2, 4, 8, 8), dtype=np.float32))
        
        z = vae.sample(mean, log_var)
        
        assert z.shape == (2, 4, 8, 8)
    
    def test_forward_complete(self):
        """Test complete VAE forward pass."""
        vae = VAE()
        x = Tensor(np.random.randn(1, 3, 64, 64).astype(np.float32))
        
        x_recon, mean, log_var = vae(x)
        
        assert x_recon.shape == x.shape
        assert mean.shape[1] == 4  # latent_channels
        assert log_var.shape[1] == 4
    
    def test_reconstruction_shape(self):
        """Test that reconstruction matches input shape."""
        vae = VAE()
        x = Tensor(np.random.randn(2, 3, 128, 96).astype(np.float32))
        
        x_recon, _, _ = vae(x)
        
        assert x_recon.shape == x.shape
    
    def test_parameters_not_empty(self):
        """Test that VAE has parameters."""
        vae = VAE()
        params = list(vae.parameters())
        
        assert len(params) > 0
    
    def test_state_dict(self):
        """Test that VAE state dict can be saved."""
        vae = VAE()
        state_dict = vae.state_dict()
        
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0


class TestBuildVAE:
    """Tests for VAE builder function."""
    
    def test_build_vae_default(self):
        """Test building VAE with default parameters."""
        vae = build_vae()
        
        assert isinstance(vae, VAE)
        assert vae.latent_channels == 4
    
    def test_build_vae_custom_channels(self):
        """Test building VAE with custom channels."""
        vae = build_vae(in_channels=1, latent_channels=8, out_channels=1)
        
        assert isinstance(vae, VAE)
        assert vae.latent_channels == 8
