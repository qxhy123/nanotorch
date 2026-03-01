"""
Integration tests for Stable Diffusion components.
"""

import pytest
import numpy as np
from nanotorch.tensor import Tensor
from nanotorch.generative.stable_diffusion import (
    VAE,
    build_vae,
    UNet2DConditionModel,
    build_unet,
    NoiseScheduler,
    LatentDiffusion,
    build_scheduler,
)


class TestVAEIntegration:
    """Integration tests for VAE."""
    
    def test_vae_encode_decode_roundtrip_shape(self):
        """Test VAE encode-decode roundtrip maintains shape."""
        vae = build_vae()
        
        x = Tensor(np.random.randn(1, 3, 64, 64).astype(np.float32) * 0.5)
        
        mean, log_var = vae.encode(x)
        z = vae.sample(mean, log_var)
        x_recon = vae.decode(z)
        
        assert x_recon.shape == x.shape
    
    def test_vae_different_resolutions(self):
        """Test VAE with different input resolutions."""
        vae = build_vae()
        
        resolutions = [(64, 64), (128, 64), (256, 256)]
        
        for h, w in resolutions:
            x = Tensor(np.random.randn(1, 3, h, w).astype(np.float32))
            
            mean, log_var = vae.encode(x)
            z = vae.sample(mean, log_var)
            x_recon = vae.decode(z)
            
            assert mean.shape == (1, 4, h // 8, w // 8)
            assert z.shape == (1, 4, h // 8, w // 8)
            assert x_recon.shape == (1, 3, h, w)
    
    def test_vae_batch_processing(self):
        """Test VAE with batch input."""
        vae = build_vae()
        
        batch_sizes = [1, 2, 4]
        
        for bs in batch_sizes:
            x = Tensor(np.random.randn(bs, 3, 32, 32).astype(np.float32))
            
            mean, log_var = vae.encode(x)
            z = vae.sample(mean, log_var)
            x_recon = vae.decode(z)
            
            assert mean.shape[0] == bs
            assert x_recon.shape[0] == bs


class TestUNetIntegration:
    """Integration tests for U-Net."""
    
    def test_unet_different_latent_sizes(self):
        """Test U-Net with different latent sizes."""
        unet = build_unet()
        
        sizes = [16, 32, 64]
        
        for size in sizes:
            sample = Tensor(np.random.randn(1, 4, size, size).astype(np.float32))
            timestep = np.array([500])
            
            out = unet(sample, timestep)
            
            assert out.shape == sample.shape
    
    def test_unet_with_text_conditioning(self):
        """Test U-Net with text embeddings."""
        unet = build_unet(cross_attention_dim=512)
        
        sample = Tensor(np.random.randn(2, 4, 32, 32).astype(np.float32))
        timestep = np.array([100, 900])
        context = Tensor(np.random.randn(2, 77, 512).astype(np.float32) * 0.1)
        
        out = unet(sample, timestep, context)
        
        assert out.shape == sample.shape
    
    def test_unet_timestep_range(self):
        """Test U-Net with different timesteps."""
        unet = build_unet()
        
        sample = Tensor(np.random.randn(1, 4, 32, 32).astype(np.float32))
        
        timesteps = [0, 100, 500, 999]
        
        for t in timesteps:
            out = unet(sample, np.array([t]))
            assert out.shape == sample.shape


class TestDiffusionIntegration:
    """Integration tests for diffusion process."""
    
    def test_forward_reverse_diffusion_consistency(self):
        """Test forward and reverse diffusion consistency."""
        scheduler = build_scheduler(num_train_timesteps=100)
        
        x0 = np.random.randn(1, 4, 16, 16).astype(np.float32)
        t = 50
        noise = np.random.randn(1, 4, 16, 16).astype(np.float32)
        
        xt = scheduler.add_noise(x0, noise, np.array([t]))
        
        x0_pred = scheduler.predict_start_from_noise(xt, np.array([t]), noise)
        
        np.testing.assert_allclose(x0_pred, x0, rtol=1e-3, atol=1e-3)
    
    def test_diffusion_schedule_degradation(self):
        """Test that alpha_cumprod decreases with timestep."""
        scheduler = build_scheduler()
        
        prev_alpha = 1.0
        for t in [0, 100, 500, 900]:
            alpha = scheduler.alphas_cumprod[t]
            assert alpha < prev_alpha
            prev_alpha = alpha
    
    def test_scheduler_inference_loop(self):
        """Test scheduler for inference loop."""
        scheduler = build_scheduler()
        scheduler.set_timesteps(10)
        
        x = np.random.randn(1, 4, 16, 16).astype(np.float32)
        
        for t in scheduler.timesteps:
            noise_pred = np.random.randn(1, 4, 16, 16).astype(np.float32)
            x, _ = scheduler.step(noise_pred, int(t), x)
        
        assert x.shape == (1, 4, 16, 16)


class TestPipelineIntegration:
    """Integration tests for full Latent Diffusion pipeline."""
    
    def test_pipeline_components_interaction(self):
        """Test that pipeline components work together."""
        vae = build_vae()
        unet = build_unet()
        scheduler = build_scheduler()
        pipeline = LatentDiffusion(vae, unet, scheduler)
        
        assert pipeline.vae is vae
        assert pipeline.unet is unet
        assert pipeline.noise_scheduler is scheduler
    
    def test_train_step_integration(self):
        """Test training step with all components."""
        vae = build_vae()
        unet = build_unet()
        scheduler = build_scheduler()
        pipeline = LatentDiffusion(vae, unet, scheduler)
        
        pixel_values = Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32))
        
        loss, loss_value = pipeline.train_step(pixel_values)
        
        assert loss_value >= 0
    
    def test_pipeline_state_dict_roundtrip(self):
        """Test saving and loading pipeline components."""
        vae = build_vae()
        unet = build_unet()
        
        vae_state = vae.state_dict()
        unet_state = unet.state_dict()
        
        assert isinstance(vae_state, dict)
        assert isinstance(unet_state, dict)
        
        vae2 = build_vae()
        unet2 = build_unet()
        
        vae2.load_state_dict(vae_state)
        unet2.load_state_dict(unet_state)
        
        x = Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32))
        
        vae.eval()
        vae2.eval()
        
        mean1, log_var1 = vae.encode(x)
        mean2, log_var2 = vae2.encode(x)
        
        np.testing.assert_allclose(mean1.data, mean2.data, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(log_var1.data, log_var2.data, rtol=1e-5, atol=1e-5)


class TestEndToEnd:
    """End-to-end tests."""
    
    def test_vae_unet_diffusion_shapes(self):
        """Test shapes through VAE -> U-Net -> Diffusion."""
        vae = build_vae()
        unet = build_unet()
        scheduler = build_scheduler()
        
        x = Tensor(np.random.randn(1, 3, 64, 64).astype(np.float32))
        
        mean, log_var = vae.encode(x)
        z = vae.sample(mean, log_var)
        
        assert z.shape == (1, 4, 8, 8)
        
        t = 500
        noise = np.random.randn(*z.shape).astype(np.float32)
        zt = scheduler.add_noise(z, noise, np.array([t]))
        zt_tensor = Tensor(zt)
        
        noise_pred = unet(zt_tensor, np.array([t]))
        
        assert noise_pred.shape == z.shape
        
        x_recon = vae.decode(z)
        
        assert x_recon.shape == x.shape
    
    def test_minimal_training_simulation(self):
        """Test minimal training simulation."""
        vae = build_vae()
        unet = build_unet()
        scheduler = build_scheduler(num_train_timesteps=100)
        
        pixel_values = Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32))
        
        mean, log_var = vae.encode(pixel_values)
        latents = vae.sample(mean, log_var)
        
        t = np.random.randint(0, 100, (1,))
        noise = np.random.randn(*latents.shape).astype(np.float32)
        noisy_latents = scheduler.add_noise(latents, noise, t)
        noisy_latents_tensor = Tensor(noisy_latents)
        
        noise_pred = unet(noisy_latents_tensor, t)
        
        loss = np.mean((noise_pred.data - noise) ** 2)
        
        assert loss >= 0
        assert isinstance(loss, (float, np.floating))
    
    def test_generation_simulation(self):
        """Test generation simulation with few steps."""
        vae = build_vae()
        unet = build_unet()
        scheduler = build_scheduler(num_train_timesteps=100)
        
        scheduler.set_timesteps(5)
        
        latents = np.random.randn(1, 4, 8, 8).astype(np.float32)
        
        for t in scheduler.timesteps:
            latents_tensor = Tensor(latents)
            noise_pred = unet(latents_tensor, np.array([int(t)]))
            latents, _ = scheduler.step(noise_pred.data, int(t), latents)
        
        latents_tensor = Tensor(latents)
        image = vae.decode(latents_tensor)
        
        assert image.shape == (1, 3, 64, 64)
