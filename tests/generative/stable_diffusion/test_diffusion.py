"""
Unit tests for Stable Diffusion diffusion process components.
"""

import pytest
import numpy as np
from nanotorch.tensor import Tensor
from nanotorch.generative.stable_diffusion.diffusion import (
    NoiseScheduler,
    LatentDiffusion,
    build_scheduler,
)


class TestNoiseScheduler:
    """Tests for NoiseScheduler."""
    
    def test_init_default(self):
        """Test default initialization."""
        scheduler = NoiseScheduler()
        
        assert scheduler.num_train_timesteps == 1000
        assert len(scheduler.betas) == 1000
        assert len(scheduler.alphas) == 1000
    
    def test_init_linear_schedule(self):
        """Test linear beta schedule."""
        scheduler = NoiseScheduler(beta_schedule="linear")
        
        assert scheduler.betas[0] == pytest.approx(0.00085, rel=1e-5)
        assert scheduler.betas[-1] == pytest.approx(0.012, rel=1e-5)
    
    def test_init_scaled_linear_schedule(self):
        """Test scaled linear beta schedule."""
        scheduler = NoiseScheduler(beta_schedule="scaled_linear")
        
        assert scheduler.betas[0] < scheduler.betas[-1]
    
    def test_init_cosine_schedule(self):
        """Test cosine beta schedule."""
        scheduler = NoiseScheduler(beta_schedule="squaredcos_cap_v2")
        
        assert np.all(scheduler.betas >= 0)
        assert np.all(scheduler.betas < 1)
    
    def test_alphas_cumprod_range(self):
        """Test that alphas_cumprod is in valid range."""
        scheduler = NoiseScheduler()
        
        assert np.all(scheduler.alphas_cumprod >= 0)
        assert np.all(scheduler.alphas_cumprod <= 1)
        assert scheduler.alphas_cumprod[-1] < scheduler.alphas_cumprod[0]
    
    def test_add_noise_shape(self):
        """Test add_noise output shape."""
        scheduler = NoiseScheduler()
        samples = np.random.randn(2, 4, 32, 32).astype(np.float32)
        noise = np.random.randn(2, 4, 32, 32).astype(np.float32)
        timesteps = np.array([100, 500])
        
        noisy = scheduler.add_noise(samples, noise, timesteps)
        
        assert noisy.shape == samples.shape
    
    def test_add_noise_tensor_input(self):
        """Test add_noise with Tensor input."""
        scheduler = NoiseScheduler()
        samples = Tensor(np.random.randn(1, 4, 16, 16).astype(np.float32))
        noise = Tensor(np.random.randn(1, 4, 16, 16).astype(np.float32))
        timesteps = np.array([500])
        
        noisy = scheduler.add_noise(samples, noise, timesteps)
        
        assert noisy.shape == (1, 4, 16, 16)
    
    def test_add_noise_at_t0(self):
        """Test that noise at t=0 is minimal."""
        scheduler = NoiseScheduler()
        samples = np.ones((1, 1, 4, 4), dtype=np.float32)
        noise = np.zeros((1, 1, 4, 4), dtype=np.float32)
        timesteps = np.array([0])
        
        noisy = scheduler.add_noise(samples, noise, timesteps)
        
        # At t=0, alpha_bar is close to 1 but not exactly 1
        # So the result should be very close to the original
        np.testing.assert_allclose(noisy, samples, rtol=1e-3, atol=1e-3)
    
    def test_add_noise_at_tmax(self):
        """Test that noise at max t is dominant."""
        scheduler = NoiseScheduler()
        samples = np.ones((1, 1, 4, 4), dtype=np.float32)
        noise = np.zeros((1, 1, 4, 4), dtype=np.float32)
        timesteps = np.array([scheduler.num_train_timesteps - 1])
        
        noisy = scheduler.add_noise(samples, noise, timesteps)
        
        # At max t, alpha_bar is close to 0 but not exactly 0
        # So the result should be close to 0
        assert np.abs(noisy).mean() < 0.2
    
    def test_predict_start_from_noise_shape(self):
        """Test predict_start_from_noise output shape."""
        scheduler = NoiseScheduler()
        sample = np.random.randn(2, 4, 32, 32).astype(np.float32)
        noise_pred = np.random.randn(2, 4, 32, 32).astype(np.float32)
        timesteps = np.array([100, 500])
        
        pred = scheduler.predict_start_from_noise(sample, timesteps, noise_pred)
        
        assert pred.shape == sample.shape
    
    def test_get_velocity_shape(self):
        """Test get_velocity output shape."""
        scheduler = NoiseScheduler()
        sample = np.random.randn(2, 4, 32, 32).astype(np.float32)
        noise = np.random.randn(2, 4, 32, 32).astype(np.float32)
        timesteps = np.array([100, 500])
        
        velocity = scheduler.get_velocity(sample, noise, timesteps)
        
        assert velocity.shape == sample.shape
    
    def test_step_shape(self):
        """Test step output shape."""
        scheduler = NoiseScheduler()
        noise_pred = np.random.randn(1, 4, 16, 16).astype(np.float32)
        sample = np.random.randn(1, 4, 16, 16).astype(np.float32)
        
        prev_sample, pred_original = scheduler.step(noise_pred, 500, sample)
        
        assert prev_sample.shape == sample.shape
        assert pred_original.shape == sample.shape
    
    def test_step_at_t0(self):
        """Test step at t=0 returns posterior mean without noise."""
        scheduler = NoiseScheduler()
        sample = np.zeros((1, 4, 8, 8), dtype=np.float32)
        noise_pred = np.zeros((1, 4, 8, 8), dtype=np.float32)
        
        prev_sample, _ = scheduler.step(noise_pred, 0, sample)
        
        np.testing.assert_allclose(prev_sample, sample, atol=1e-5)
    
    def test_set_timesteps(self):
        """Test set_timesteps configures inference steps."""
        scheduler = NoiseScheduler()
        
        scheduler.set_timesteps(50)
        
        assert len(scheduler.timesteps) == 50
        assert scheduler.num_inference_steps == 50
    
    def test_set_timesteps_ordering(self):
        """Test timesteps are in descending order."""
        scheduler = NoiseScheduler()
        scheduler.set_timesteps(50)
        
        for i in range(len(scheduler.timesteps) - 1):
            assert scheduler.timesteps[i] > scheduler.timesteps[i + 1]
    
    def test_len(self):
        """Test len returns num_train_timesteps."""
        scheduler = NoiseScheduler(num_train_timesteps=500)
        
        assert len(scheduler) == 500
    
    def test_v_prediction_mode(self):
        """Test scheduler with v-prediction mode."""
        scheduler = NoiseScheduler(prediction_type="v_prediction")
        
        assert scheduler.prediction_type == "v_prediction"
    
    def test_clip_sample(self):
        """Test that samples are clipped when enabled."""
        scheduler = NoiseScheduler(clip_sample=True, clip_sample_range=1.0)
        sample = np.zeros((1, 1, 4, 4), dtype=np.float32)
        noise_pred = np.zeros((1, 1, 4, 4), dtype=np.float32)
        
        _, pred_original = scheduler.step(noise_pred, 500, sample)
        
        assert np.all(pred_original >= -1.0)
        assert np.all(pred_original <= 1.0)


class TestLatentDiffusion:
    """Tests for LatentDiffusion pipeline."""
    
    def test_init(self):
        """Test LatentDiffusion initialization."""
        vae = self._create_mock_vae()
        unet = self._create_mock_unet()
        scheduler = NoiseScheduler()
        
        pipeline = LatentDiffusion(vae, unet, scheduler)
        
        assert pipeline.vae is vae
        assert pipeline.unet is unet
        assert pipeline.noise_scheduler is scheduler
    
    def test_encode_text_shape(self):
        """Test encode_text output shape."""
        pipeline = self._create_pipeline()
        
        embedding = pipeline.encode_text("test prompt")
        
        assert embedding.shape == (1, 77, 512)
    
    def test_prepare_latents_shape(self):
        """Test prepare_latents output shape."""
        pipeline = self._create_pipeline()
        
        latents = pipeline.prepare_latents(batch_size=2, height=8, width=8)
        
        assert latents.shape == (2, 4, 8, 8)
    
    def test_prepare_latents_reproducible(self):
        """Test prepare_latents with generator is reproducible."""
        pipeline = self._create_pipeline()
        gen1 = np.random.Generator(np.random.PCG64(42))
        gen2 = np.random.Generator(np.random.PCG64(42))
        
        latents1 = pipeline.prepare_latents(1, 8, 8, generator=gen1)
        latents2 = pipeline.prepare_latents(1, 8, 8, generator=gen2)
        
        np.testing.assert_array_equal(latents1, latents2)
    
    def test_call_with_classifier_free_guidance(self):
        """Test end-to-end sampling path with guidance enabled."""
        pipeline = self._create_pipeline()

        image = pipeline(
            "test prompt",
            num_inference_steps=2,
            height=4,
            width=4,
            guidance_scale=7.5,
        )

        assert image.shape == (1, 3, 32, 32)

    def test_train_step_shape(self):
        """Test train_step output."""
        pipeline = self._create_pipeline()
        pixel_values = Tensor(np.random.randn(2, 3, 32, 32).astype(np.float32))
        
        loss, loss_value = pipeline.train_step(pixel_values)
        
        assert isinstance(loss_value, (float, np.floating))
        assert loss_value >= 0
    
    def _create_mock_vae(self):
        """Create a mock VAE for testing."""
        class MockVAE:
            def __init__(self):
                self.latent_channels = 4
            
            def encode(self, x):
                batch = x.shape[0]
                mean = Tensor(np.zeros((batch, 4, x.shape[2]//8, x.shape[3]//8), dtype=np.float32))
                log_var = Tensor(np.zeros((batch, 4, x.shape[2]//8, x.shape[3]//8), dtype=np.float32))
                return mean, log_var
            
            def sample(self, mean, log_var):
                return mean
            
            def decode(self, z):
                batch = z.shape[0]
                h, w = z.shape[2] * 8, z.shape[3] * 8
                return Tensor(np.zeros((batch, 3, h, w), dtype=np.float32))
        
        return MockVAE()
    
    def _create_mock_unet(self):
        """Create a mock U-Net for testing."""
        class MockUNet:
            def __call__(self, sample, timestep, context=None):
                return Tensor(np.zeros_like(sample.data))
        
        return MockUNet()
    
    def _create_pipeline(self):
        """Create a LatentDiffusion pipeline for testing."""
        return LatentDiffusion(
            self._create_mock_vae(),
            self._create_mock_unet(),
            NoiseScheduler()
        )


class TestBuildScheduler:
    """Tests for scheduler builder function."""
    
    def test_build_scheduler_default(self):
        """Test building scheduler with defaults."""
        scheduler = build_scheduler()
        
        assert isinstance(scheduler, NoiseScheduler)
        assert scheduler.num_train_timesteps == 1000
    
    def test_build_scheduler_custom(self):
        """Test building scheduler with custom parameters."""
        scheduler = build_scheduler(
            num_train_timesteps=500,
            beta_schedule="linear",
            prediction_type="v_prediction"
        )
        
        assert scheduler.num_train_timesteps == 500
        assert scheduler.beta_schedule == "linear"
        assert scheduler.prediction_type == "v_prediction"
