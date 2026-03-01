"""
Stable Diffusion Demo - Text-to-Image Generation

This demo shows how to use the Stable Diffusion implementation for
text-to-image generation using:
1. VAE for encoding/decoding between image and latent space
2. U-Net for noise prediction conditioned on text embeddings
3. DDPM scheduler for the diffusion process

Note: This is a simplified educational implementation.
For actual image generation, you would need:
- Pre-trained weights
- CLIP text encoder
- More inference steps (typically 20-50)
"""

import numpy as np
from nanotorch.tensor import Tensor, no_grad
from nanotorch.generative.stable_diffusion import (
    VAE,
    UNet2DConditionModel,
    NoiseScheduler,
    LatentDiffusion,
    build_vae,
    build_unet,
    build_scheduler,
)


def demo_vae_encode_decode():
    """Demonstrate VAE encoding and decoding."""
    print("=" * 60)
    print("VAE Encode/Decode Demo")
    print("=" * 60)
    
    vae = build_vae()
    vae.eval()
    
    image = Tensor(np.random.randn(1, 3, 64, 64).astype(np.float32) * 0.5)
    print(f"Input image shape: {image.shape}")
    
    mean, log_var = vae.encode(image)
    print(f"Encoded mean shape: {mean.shape}")
    print(f"Encoded log_var shape: {log_var.shape}")
    
    z = vae.sample(mean, log_var)
    print(f"Sampled latent shape: {z.shape}")
    
    reconstructed = vae.decode(z)
    print(f"Reconstructed image shape: {reconstructed.shape}")
    
    print("\nVAE compression ratio: 8x spatial + channel reduction")
    print(f"Image size: {np.prod(image.shape)} -> Latent size: {np.prod(z.shape)}")
    print()


def demo_unet_denoising():
    """Demonstrate U-Net noise prediction."""
    print("=" * 60)
    print("U-Net Denoising Demo")
    print("=" * 60)
    
    unet = build_unet()
    
    latent = Tensor(np.random.randn(2, 4, 32, 32).astype(np.float32))
    timesteps = np.array([100, 500])
    text_embedding = Tensor(np.random.randn(2, 77, 512).astype(np.float32) * 0.1)
    
    print(f"Input latent shape: {latent.shape}")
    print(f"Timesteps: {timesteps}")
    print(f"Text embedding shape: {text_embedding.shape}")
    
    noise_pred = unet(latent, timesteps, text_embedding)
    print(f"Predicted noise shape: {noise_pred.shape}")
    
    print("\nU-Net architecture:")
    print("  - 4 downsampling stages with ResNet + Cross-Attention")
    print("  - Mid block with self-attention")
    print("  - 4 upsampling stages with skip connections")
    print()


def demo_diffusion_scheduler():
    """Demonstrate the diffusion scheduler."""
    print("=" * 60)
    print("Diffusion Scheduler Demo")
    print("=" * 60)
    
    scheduler = build_scheduler(num_train_timesteps=1000)
    
    print(f"Number of training timesteps: {len(scheduler)}")
    print(f"Beta schedule: scaled_linear")
    print(f"Beta range: [{scheduler.betas[0]:.6f}, {scheduler.betas[-1]:.6f}]")
    print(f"Alpha_cumprod range: [{scheduler.alphas_cumprod[-1]:.6f}, {scheduler.alphas_cumprod[0]:.6f}]")
    
    x0 = np.random.randn(1, 4, 16, 16).astype(np.float32)
    noise = np.random.randn(1, 4, 16, 16).astype(np.float32)
    t = 500
    
    xt = scheduler.add_noise(x0, noise, np.array([t]))
    print(f"\nForward diffusion at t={t}:")
    print(f"  x0 shape: {x0.shape}")
    print(f"  xt shape: {xt.shape}")
    
    x0_pred = scheduler.predict_start_from_noise(xt, np.array([t]), noise)
    print(f"  x0_pred shape: {x0_pred.shape}")
    
    prev_x, _ = scheduler.step(noise, t, xt)
    print(f"  prev_x shape: {prev_x.shape}")
    
    print()


def demo_generation_simulation():
    """Simulate the image generation process."""
    print("=" * 60)
    print("Generation Simulation Demo")
    print("=" * 60)
    
    vae = build_vae()
    unet = build_unet()
    scheduler = build_scheduler(num_train_timesteps=100)
    
    vae.eval()
    
    scheduler.set_timesteps(10)
    print(f"Number of inference steps: {len(scheduler.timesteps)}")
    print(f"Timesteps: {scheduler.timesteps}")
    
    latents = np.random.randn(1, 4, 8, 8).astype(np.float32)
    print(f"\nInitial latent shape: {latents.shape}")
    
    text_embedding = Tensor(np.random.randn(1, 77, 512).astype(np.float32) * 0.1)
    
    for i, t in enumerate(scheduler.timesteps):
        latents_tensor = Tensor(latents, requires_grad=False)
        noise_pred = unet(latents_tensor, np.array([int(t)]), text_embedding)
        latents, _ = scheduler.step(noise_pred.data, int(t), latents)
    
    print(f"Final latent shape: {latents.shape}")
    
    latents_tensor = Tensor(latents, requires_grad=False)
    image = vae.decode(latents_tensor)
    print(f"Generated image shape: {image.shape}")
    
    print("\nGeneration process complete!")
    print()


def demo_training_step():
    """Demonstrate a training step."""
    print("=" * 60)
    print("Training Step Demo")
    print("=" * 60)
    
    vae = build_vae()
    unet = build_unet()
    scheduler = build_scheduler(num_train_timesteps=1000)
    
    vae.eval()
    
    pixel_values = Tensor(np.random.randn(2, 3, 32, 32).astype(np.float32))
    text_embedding = Tensor(np.random.randn(2, 77, 512).astype(np.float32) * 0.1)
    
    print(f"Training images shape: {pixel_values.shape}")
    
    with no_grad():
        mean, log_var = vae.encode(pixel_values)
        latents = vae.sample(mean, log_var)
    
    print(f"Encoded latents shape: {latents.shape}")
    
    t = np.random.randint(0, 1000, (2,))
    print(f"Random timesteps: {t}")
    
    noise = Tensor(np.random.randn(*latents.shape).astype(np.float32))
    noisy_latents = scheduler.add_noise(latents, noise, t)
    noisy_latents_tensor = Tensor(noisy_latents, requires_grad=False)
    
    print(f"Noisy latents shape: {noisy_latents.shape}")
    
    noise_pred = unet(noisy_latents_tensor, t, text_embedding)
    
    loss = np.mean((noise_pred.data - noise.data) ** 2)
    print(f"Predicted noise shape: {noise_pred.shape}")
    print(f"MSE Loss: {loss:.6f}")
    
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("STABLE DIFFUSION DEMO - nanotorch Implementation")
    print("=" * 60 + "\n")
    
    demo_vae_encode_decode()
    demo_unet_denoising()
    demo_diffusion_scheduler()
    demo_generation_simulation()
    demo_training_step()
    
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    print("\nKey Components:")
    print("  1. VAE: Encodes images to latent space (8x compression)")
    print("  2. U-Net: Predicts noise conditioned on text embeddings")
    print("  3. Scheduler: Manages forward/reverse diffusion process")
    print("\nFor actual image generation, you would need:")
    print("  - Pre-trained weights")
    print("  - CLIP text encoder")
    print("  - More inference steps (20-50)")


if __name__ == "__main__":
    main()
