"""
Stable Diffusion - Diffusion Process and Noise Scheduler

Implements the forward and reverse diffusion processes for training and inference.

Forward Diffusion (adding noise):
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

Reverse Diffusion (denoising):
    x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * epsilon_theta) + sigma_t * z

Key concepts:
1. Noise Schedule: Controls how much noise is added at each timestep
2. Forward Process: Gradually adds noise to data
3. Reverse Process: Uses neural network to predict and remove noise
4. DDPM/DDIM: Different sampling strategies

Reference:
    "Denoising Diffusion Probabilistic Models"
    Ho, Jain, Abbeel
    NeurIPS 2020
    https://arxiv.org/abs/2006.11239
"""

import numpy as np
from typing import Tuple, Optional, Union
from nanotorch.tensor import Tensor, no_grad
from nanotorch.utils import cat


class NoiseScheduler:
    """Noise scheduler for diffusion models.
    
    Manages the noise schedule and provides methods for forward/reverse diffusion.
    
    Args:
        num_train_timesteps: Number of diffusion steps (default: 1000)
        beta_start: Starting beta value (default: 0.00085)
        beta_end: Ending beta value (default: 0.012)
        beta_schedule: Schedule type - "linear", "scaled_linear", "squaredcos_cap_v2"
        prediction_type: "epsilon" (predict noise) or "v_prediction" (predict velocity)
        clip_sample: Whether to clip samples to [-1, 1] range
        clip_sample_range: Range for clipping samples
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        
        # Compute betas based on schedule
        if beta_schedule == "linear":
            self.betas = np.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=np.float32
            )
        elif beta_schedule == "scaled_linear":
            # Linear in sqrt space
            self.betas = np.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=np.float32
            ) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        # Compute alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        
        # Precompute useful values for diffusion
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Posterior variance: q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # Posterior log variance clamped
        self.posterior_log_variance_clipped = np.log(
            np.maximum(self.posterior_variance, 1e-20)
        )
        
        # Posterior mean coefficient
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        
        # For DDIM sampling
        self.timesteps = np.arange(num_train_timesteps)[::-1].copy()
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> np.ndarray:
        """Cosine schedule as proposed in improved DDPM paper.
        
        Args:
            timesteps: Number of diffusion steps
            s: Offset for small noise at start
        
        Returns:
            Beta schedule array
        """
        steps = timesteps + 1
        x = np.linspace(0, timesteps, steps, dtype=np.float32)
        alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)
    
    def add_noise(
        self,
        original_samples: Union[Tensor, np.ndarray],
        noise: Union[Tensor, np.ndarray],
        timesteps: np.ndarray
    ) -> np.ndarray:
        """Add noise to samples at given timesteps (forward diffusion).
        
        Implements: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        
        Args:
            original_samples: Clean samples x_0 (N, C, H, W)
            noise: Noise to add epsilon (N, C, H, W)
            timesteps: Timestep indices (N,)
        
        Returns:
            Noisy samples x_t
        """
        if isinstance(original_samples, Tensor):
            original_samples = original_samples.data
        if isinstance(noise, Tensor):
            noise = noise.data
        
        # Get alpha values for timesteps
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod[:, np.newaxis]
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[:, np.newaxis]
        
        # Forward diffusion formula
        noisy_samples = (
            sqrt_alpha_prod * original_samples + 
            sqrt_one_minus_alpha_prod * noise
        )
        
        return noisy_samples
    
    def get_velocity(
        self,
        sample: Union[Tensor, np.ndarray],
        noise: Union[Tensor, np.ndarray],
        timesteps: np.ndarray
    ) -> np.ndarray:
        """Compute v-prediction target.
        
        v = sqrt(alpha_bar) * epsilon - sqrt(1 - alpha_bar) * x_0
        
        Args:
            sample: Clean samples x_0
            noise: Noise epsilon
            timesteps: Timestep indices
        
        Returns:
            Velocity prediction target
        """
        if isinstance(sample, Tensor):
            sample = sample.data
        if isinstance(noise, Tensor):
            noise = noise.data
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod[:, np.newaxis]
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[:, np.newaxis]
        
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        
        return velocity
    
    def predict_start_from_noise(
        self,
        sample: Union[Tensor, np.ndarray],
        timesteps: np.ndarray,
        noise_pred: Union[Tensor, np.ndarray]
    ) -> np.ndarray:
        """Predict x_0 from x_t and predicted noise (epsilon prediction).
        
        x_0 = (x_t - sqrt(1 - alpha_bar) * epsilon) / sqrt(alpha_bar)
        
        Args:
            sample: Noisy samples x_t
            timesteps: Timestep indices
            noise_pred: Predicted noise epsilon_theta
        
        Returns:
            Predicted clean samples x_0
        """
        if isinstance(sample, Tensor):
            sample = sample.data
        if isinstance(noise_pred, Tensor):
            noise_pred = noise_pred.data
        
        sqrt_recip = self.sqrt_recip_alphas_cumprod[timesteps]
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[timesteps]
        
        while len(sqrt_recip.shape) < len(sample.shape):
            sqrt_recip = sqrt_recip[:, np.newaxis]
            sqrt_recipm1 = sqrt_recipm1[:, np.newaxis]
        
        pred_original_sample = sqrt_recip * sample - sqrt_recipm1 * noise_pred
        
        return pred_original_sample
    
    def predict_start_from_velocity(
        self,
        sample: Union[Tensor, np.ndarray],
        timesteps: np.ndarray,
        velocity: Union[Tensor, np.ndarray]
    ) -> np.ndarray:
        """Predict x_0 from x_t and predicted velocity (v-prediction).
        
        x_0 = sqrt(alpha_bar) * x_t - sqrt(1 - alpha_bar) * v
        
        Args:
            sample: Noisy samples x_t
            timesteps: Timestep indices
            velocity: Predicted velocity v_theta
        
        Returns:
            Predicted clean samples x_0
        """
        if isinstance(sample, Tensor):
            sample = sample.data
        if isinstance(velocity, Tensor):
            velocity = velocity.data
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod[:, np.newaxis]
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[:, np.newaxis]
        
        pred_original_sample = (
            sqrt_alpha_prod * sample - sqrt_one_minus_alpha_prod * velocity
        )
        
        return pred_original_sample
    
    def get_posterior_mean(
        self,
        sample: Union[Tensor, np.ndarray],
        timesteps: np.ndarray,
        pred_original_sample: Union[Tensor, np.ndarray]
    ) -> np.ndarray:
        """Compute posterior mean q(x_{t-1} | x_t, x_0).
        
        Args:
            sample: Noisy samples x_t
            timesteps: Timestep indices
            pred_original_sample: Predicted clean samples x_0
        
        Returns:
            Posterior mean
        """
        if isinstance(sample, Tensor):
            sample = sample.data
        if isinstance(pred_original_sample, Tensor):
            pred_original_sample = pred_original_sample.data
        
        coef1 = self.posterior_mean_coef1[timesteps]
        coef2 = self.posterior_mean_coef2[timesteps]
        
        while len(coef1.shape) < len(sample.shape):
            coef1 = coef1[:, np.newaxis]
            coef2 = coef2[:, np.newaxis]
        
        posterior_mean = coef1 * pred_original_sample + coef2 * sample
        
        return posterior_mean
    
    def step(
        self,
        noise_pred: Union[Tensor, np.ndarray],
        timestep: int,
        sample: Union[Tensor, np.ndarray],
        generator: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one step of reverse diffusion.
        
        Args:
            noise_pred: Predicted noise from U-Net
            timestep: Current timestep
            sample: Current noisy sample x_t
            generator: Random generator for reproducibility
        
        Returns:
            Tuple of (previous sample x_{t-1}, predicted original sample x_0)
        """
        if isinstance(noise_pred, Tensor):
            noise_pred = noise_pred.data
        if isinstance(sample, Tensor):
            sample = sample.data
        
        t = timestep
        
        # Get predicted original sample
        if self.prediction_type == "epsilon":
            pred_original_sample = self.predict_start_from_noise(sample, np.array([t]), noise_pred)
        elif self.prediction_type == "v_prediction":
            pred_original_sample = self.predict_start_from_velocity(sample, np.array([t]), noise_pred)
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")
        
        # Clip predicted sample
        if self.clip_sample:
            pred_original_sample = np.clip(
                pred_original_sample, -self.clip_sample_range, self.clip_sample_range
            )
        
        # Compute posterior mean
        posterior_mean = self.get_posterior_mean(sample, np.array([t]), pred_original_sample)
        
        # Add noise (except at t=0)
        if t > 0:
            if generator is not None:
                noise = generator.standard_normal(sample.shape, dtype=np.float32)
            else:
                noise = np.random.randn(*sample.shape).astype(np.float32)
            
            posterior_variance = np.array(self.posterior_variance[t])
            for _ in range(len(sample.shape)):
                posterior_variance = np.expand_dims(posterior_variance, -1)
            
            prev_sample = posterior_mean + np.sqrt(posterior_variance) * noise
        else:
            prev_sample = posterior_mean
        
        return prev_sample, pred_original_sample
    
    def set_timesteps(self, num_inference_steps: int) -> None:
        """Set timesteps for inference.
        
        Args:
            num_inference_steps: Number of inference steps (<= num_train_timesteps)
        """
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = np.arange(0, num_inference_steps) * step_ratio
        self.timesteps = timesteps[::-1].copy()
        self.num_inference_steps = num_inference_steps
    
    def __len__(self) -> int:
        return self.num_train_timesteps


class LatentDiffusion:
    """Latent Diffusion Model for image generation.
    
    Combines VAE and U-Net for text-to-image generation.
    
    Args:
        vae: Variational Autoencoder
        unet: Conditional U-Net
        noise_scheduler: Noise scheduler for diffusion
        text_encoder: Optional text encoder (simplified placeholder)
    """
    
    def __init__(
        self,
        vae,
        unet,
        noise_scheduler: NoiseScheduler,
        text_encoder=None
    ):
        self.vae = vae
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.text_encoder = text_encoder
    
    def encode_text(self, text: str) -> Tensor:
        """Encode text prompt to embeddings.
        
        Simplified placeholder - in real SD uses CLIP text encoder.
        
        Args:
            text: Text prompt string
        
        Returns:
            Text embedding tensor (1, 77, 512)
        """
        # Placeholder: random embeddings
        # Real implementation would use CLIP or similar
        return Tensor(
            np.random.randn(1, 77, 512).astype(np.float32) * 0.1,
            requires_grad=False
        )
    
    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        generator: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Prepare initial latents for generation.
        
        Args:
            batch_size: Number of images to generate
            height: Latent height
            width: Latent width
            generator: Random generator
        
        Returns:
            Initial noise latents
        """
        shape = (batch_size, self.vae.latent_channels, height, width)
        
        if generator is not None:
            latents = generator.standard_normal(shape, dtype=np.float32)
        else:
            latents = np.random.randn(*shape).astype(np.float32)
        
        return latents
    
    def __call__(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        height: int = 64,
        width: int = 64,
        guidance_scale: float = 7.5,
        generator: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Generate image from text prompt.
        
        Args:
            prompt: Text description
            num_inference_steps: Number of denoising steps
            height: Output height (in latent space)
            width: Output width (in latent space)
            guidance_scale: Classifier-free guidance scale
            generator: Random generator
        
        Returns:
            Generated image array (1, 3, H*8, W*8)
        """
        # Encode text prompt
        text_embeddings = self.encode_text(prompt)
        
        # For classifier-free guidance, encode empty prompt
        if guidance_scale > 1.0:
            unconditional_embeddings = self.encode_text("")
            # Concatenate for batch processing
            text_embeddings = cat([unconditional_embeddings, text_embeddings], dim=0)
        
        # Prepare latents
        latents = self.prepare_latents(1, height, width, generator)
        
        # Set timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        # Denoising loop
        for t in self.noise_scheduler.timesteps:
            # Prepare latent input
            if guidance_scale > 1.0:
                latent_input = np.repeat(latents, 2, axis=0)
            else:
                latent_input = latents

            latent_tensor = Tensor(latent_input, requires_grad=False)
            timestep = np.full((latent_input.shape[0],), t, dtype=np.int64)
            
            # Predict noise
            noise_pred = self.unet(latent_tensor, timestep, text_embeddings)
            noise_pred = noise_pred.data
            
            # Classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2, axis=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous sample
            latents, _ = self.noise_scheduler.step(noise_pred, t, latents, generator)
        
        # Decode latents to image
        latent_tensor = Tensor(latents, requires_grad=False)
        image = self.vae.decode(latent_tensor)
        
        return image.data
    
    def train_step(
        self,
        pixel_values: Tensor,
        text_embeddings: Optional[Tensor] = None
    ) -> Tuple[Tensor, np.ndarray]:
        """Perform one training step.
        
        Args:
            pixel_values: Training images (N, 3, H, W)
            text_embeddings: Text embeddings for conditioning (N, 77, 512)
        
        Returns:
            Tuple of (loss tensor, loss value)
        """
        # Encode images to latent space
        with no_grad():
            mean, log_var = self.vae.encode(pixel_values)
            # Sample from distribution
            latents = self.vae.sample(mean, log_var)
        
        # Sample timesteps
        batch_size = pixel_values.shape[0]
        timesteps = np.random.randint(
            0, self.noise_scheduler.num_train_timesteps, (batch_size,)
        )
        
        # Sample noise
        noise = Tensor(
            np.random.randn(*latents.shape).astype(np.float32),
            requires_grad=False
        )
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        noisy_latents = Tensor(noisy_latents, requires_grad=True)

        # Predict noise
        noise_pred = self.unet(noisy_latents, timesteps, text_embeddings)

        # Compute loss (MSE) - keep computation graph for backward
        diff = noise_pred - noise
        loss = (diff * diff).mean()

        return loss, loss.data


def build_scheduler(
    num_train_timesteps: int = 1000,
    beta_schedule: str = "scaled_linear",
    prediction_type: str = "epsilon"
) -> NoiseScheduler:
    """Build noise scheduler.
    
    Args:
        num_train_timesteps: Number of training timesteps
        beta_schedule: Beta schedule type
        prediction_type: Prediction type (epsilon or v_prediction)
    
    Returns:
        NoiseScheduler instance
    """
    return NoiseScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
        prediction_type=prediction_type
    )
