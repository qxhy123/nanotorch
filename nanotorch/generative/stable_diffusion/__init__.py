"""
Stable Diffusion - Latent Diffusion Model for Text-to-Image Generation

This module implements Stable Diffusion, a latent text-to-image diffusion model
capable of generating photo-realistic images from text prompts.

Key components:
- VAE: Variational Autoencoder for compressing images to latent space
- U-Net: Denoising network with cross-attention conditioning
- NoiseScheduler: Manages the forward/reverse diffusion process
- LatentDiffusion: Combines all components for image generation

Example:
    >>> from nanotorch.generative.stable_diffusion import VAE, UNet2DConditionModel, NoiseScheduler
    >>> vae = build_vae()
    >>> unet = build_unet()
    >>> scheduler = build_scheduler()
    >>> model = LatentDiffusion(vae, unet, scheduler)
    >>> image = model("a beautiful sunset")

Reference:
    "High-Resolution Image Synthesis with Latent Diffusion Models"
    Rombach, Blattmann, et al.
    CVPR 2022
    https://arxiv.org/abs/2112.10752
"""

from nanotorch.generative.stable_diffusion.vae import (
    VAE,
    Encoder,
    Decoder,
    ResnetBlock,
    SelfAttention,
    Downsample,
    Upsample,
    build_vae,
)

from nanotorch.generative.stable_diffusion.unet import (
    UNet2DConditionModel,
    TimestepEmbedding,
    TimeEmbedResnetBlock,
    CrossAttention,
    TransformerBlock,
    DownBlock2D,
    UpBlock2D,
    get_timestep_embedding,
    build_unet,
)

from nanotorch.generative.stable_diffusion.diffusion import (
    NoiseScheduler,
    LatentDiffusion,
    build_scheduler,
)

__all__ = [
    # VAE components
    "VAE",
    "Encoder",
    "Decoder",
    "ResnetBlock",
    "SelfAttention",
    "Downsample",
    "Upsample",
    "build_vae",
    # U-Net components
    "UNet2DConditionModel",
    "TimestepEmbedding",
    "TimeEmbedResnetBlock",
    "CrossAttention",
    "TransformerBlock",
    "DownBlock2D",
    "UpBlock2D",
    "get_timestep_embedding",
    "build_unet",
    # Diffusion components
    "NoiseScheduler",
    "LatentDiffusion",
    "build_scheduler",
]
