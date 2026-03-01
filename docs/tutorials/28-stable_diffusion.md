# Stable Diffusion Tutorial

## Overview

Stable Diffusion is a latent text-to-image diffusion model that generates photo-realistic images from text descriptions. This tutorial explains the implementation in nanotorch.

## Key Components

### 1. VAE (Variational Autoencoder)

The VAE compresses images to a latent space (8x spatial compression) and reconstructs them.

```python
from nanotorch.generative.stable_diffusion import VAE, build_vae

vae = build_vae()  # Creates VAE with 4 latent channels

# Encode image to latent space
image = Tensor(np.random.randn(1, 3, 512, 512).astype(np.float32))
mean, log_var = vae.encode(image)  # Shape: (1, 4, 64, 64)

# Sample from latent distribution
z = vae.sample(mean, log_var)

# Decode back to image
reconstructed = vae.decode(z)  # Shape: (1, 3, 512, 512)
```

**Architecture:**
- Encoder: 3 downsampling stages with ResNet blocks + attention
- Decoder: 3 upsampling stages with ResNet blocks + attention
- Latent space: 4 channels with 8x spatial compression

### 2. U-Net (Conditional Denoising Network)

The U-Net predicts noise in the latent space, conditioned on text embeddings and timesteps.

```python
from nanotorch.generative.stable_diffusion import UNet2DConditionModel, build_unet

unet = build_unet()  # Default: 320, 640, 1280, 1280 channels

# Prepare inputs
noisy_latent = Tensor(np.random.randn(1, 4, 64, 64).astype(np.float32))
timestep = np.array([500])
text_embedding = Tensor(np.random.randn(1, 77, 512).astype(np.float32))

# Predict noise
noise_pred = unet(noisy_latent, timestep, text_embedding)
```

**Architecture:**
- Timestep embedding: Sinusoidal encoding + MLP
- 4 downsampling stages with ResNet + Cross-Attention
- Mid block with self-attention
- 4 upsampling stages with skip connections
- Cross-attention for text conditioning

### 3. Noise Scheduler

Manages the forward/reverse diffusion process.

```python
from nanotorch.generative.stable_diffusion import NoiseScheduler, build_scheduler

scheduler = build_scheduler(num_train_timesteps=1000)

# Forward diffusion: add noise
x0 = np.random.randn(1, 4, 64, 64).astype(np.float32)
noise = np.random.randn(1, 4, 64, 64).astype(np.float32)
xt = scheduler.add_noise(x0, noise, np.array([500]))

# Reverse diffusion: denoise step
prev_xt, x0_pred = scheduler.step(noise_pred, 500, xt)
```

**Key Methods:**
- `add_noise()`: Forward diffusion (add noise at timestep t)
- `step()`: Reverse diffusion (single denoising step)
- `set_timesteps()`: Configure inference steps
- `predict_start_from_noise()`: Predict x0 from noise prediction

## Training Process

```python
from nanotorch.generative.stable_diffusion import (
    build_vae, build_unet, build_scheduler, LatentDiffusion
)

vae = build_vae()
unet = build_unet()
scheduler = build_scheduler()
pipeline = LatentDiffusion(vae, unet, scheduler)

# Training step
images = Tensor(np.random.randn(batch_size, 3, 512, 512).astype(np.float32))
text_embeddings = Tensor(np.random.randn(batch_size, 77, 512).astype(np.float32))

loss, loss_value = pipeline.train_step(images, text_embeddings)
```

**Training Algorithm:**
1. Encode images to latent space using VAE
2. Sample random timesteps
3. Add noise to latents (forward diffusion)
4. Predict noise using U-Net
5. Compute MSE loss between predicted and actual noise

## Generation Process

```python
# Configure inference
scheduler.set_timesteps(50)  # Use 50 denoising steps

# Start from random noise
latents = np.random.randn(1, 4, 64, 64).astype(np.float32)

# Encode text prompt
text_embedding = encode_text("a beautiful sunset over the ocean")

# Denoising loop
for t in scheduler.timesteps:
    # Predict noise
    noise_pred = unet(
        Tensor(latents), 
        np.array([int(t)]), 
        text_embedding
    )
    
    # Remove noise
    latents, _ = scheduler.step(noise_pred.data, int(t), latents)

# Decode to image
image = vae.decode(Tensor(latents))
```

## Mathematical Background

### Forward Diffusion

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$$

where:
- $x_0$ is the original image
- $x_t$ is the noisy version at timestep $t$
- $\bar{\alpha}_t = \prod_{i=0}^{t} \alpha_i$ (cumulative product of alphas)
- $\epsilon \sim \mathcal{N}(0, I)$ is Gaussian noise

### Reverse Diffusion

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \cdot \epsilon_\theta(x_t, t) \right) + \sigma_t \cdot z$$

where:
- $\epsilon_\theta(x_t, t)$ is the predicted noise from U-Net
- $\beta_t = 1 - \alpha_t$ is the noise schedule
- $z \sim \mathcal{N}(0, I)$ is random noise (except at $t=0$)

### VAE Loss (ELBO)

$$\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

where:
- $q(z|x)$ is the encoder distribution
- $p(x|z)$ is the decoder distribution  
- $p(z) = \mathcal{N}(0, I)$ is the prior
- $D_{KL}$ is the Kullback-Leibler divergence

## Implementation Details

### Timestep Embedding
```python
def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = np.log(10000.0) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim) * -emb)
    emb = timesteps[:, np.newaxis] * emb[np.newaxis, :]
    return np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
```

### Cross-Attention
- Query: from spatial features
- Key/Value: from text embeddings
- Enables conditioning on text

### Classifier-Free Guidance
```python
# Unconditional + conditional prediction
noise_uncond = unet(latents, t, empty_embedding)
noise_cond = unet(latents, t, text_embedding)

# Guidance
noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
```

## File Structure

```
nanotorch/generative/stable_diffusion/
├── __init__.py          # Module exports
├── vae.py               # VAE implementation
├── unet.py              # U-Net implementation
└── diffusion.py         # Noise scheduler and pipeline

tests/generative/stable_diffusion/
├── test_vae.py          # 30 VAE tests
├── test_unet.py         # 27 U-Net tests
├── test_diffusion.py    # 25 diffusion tests
└── test_integration.py  # 15 integration tests

examples/stable_diffusion/
└── demo.py              # Demo script
```

## References

1. "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., CVPR 2022)
2. "Denoising Diffusion Probabilistic Models" (Ho et al., NeurIPS 2020)
3. Stable Diffusion GitHub: https://github.com/CompVis/stable-diffusion
