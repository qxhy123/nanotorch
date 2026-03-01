# Stable Diffusion 教程

## 概述

Stable Diffusion 是一种潜在文本到图像的扩散模型，能够根据文本描述生成逼真的图像。本教程将详细介绍 nanotorch 中的实现。

## 核心组件

### 1. VAE（变分自编码器）

VAE 将图像压缩到潜在空间（8倍空间压缩）并重建图像。

```python
from nanotorch.generative.stable_diffusion import VAE, build_vae

vae = build_vae()  # 创建具有4个潜在通道的VAE

# 将图像编码到潜在空间
image = Tensor(np.random.randn(1, 3, 512, 512).astype(np.float32))
mean, log_var = vae.encode(image)  # 形状: (1, 4, 64, 64)

# 从潜在分布中采样
z = vae.sample(mean, log_var)

# 解码回图像
reconstructed = vae.decode(z)  # 形状: (1, 3, 512, 512)
```

**架构说明：**
- 编码器：3个下采样阶段，包含ResNet块和注意力机制
- 解码器：3个上采样阶段，包含ResNet块和注意力机制
- 潜在空间：4通道，8倍空间压缩

### 2. U-Net（条件去噪网络）

U-Net 在潜在空间中预测噪声，以文本嵌入和时间步为条件。

```python
from nanotorch.generative.stable_diffusion import UNet2DConditionModel, build_unet

unet = build_unet()  # 默认通道数: 320, 640, 1280, 1280

# 准备输入
noisy_latent = Tensor(np.random.randn(1, 4, 64, 64).astype(np.float32))
timestep = np.array([500])
text_embedding = Tensor(np.random.randn(1, 77, 512).astype(np.float32))

# 预测噪声
noise_pred = unet(noisy_latent, timestep, text_embedding)
```

**架构说明：**
- 时间步嵌入：正弦编码 + MLP
- 4个下采样阶段，包含ResNet + 交叉注意力
- 中间块包含自注意力
- 4个上采样阶段，带有跳跃连接
- 交叉注意力用于文本条件控制

### 3. 噪声调度器

管理前向/反向扩散过程。

```python
from nanotorch.generative.stable_diffusion import NoiseScheduler, build_scheduler

scheduler = build_scheduler(num_train_timesteps=1000)

# 前向扩散：添加噪声
x0 = np.random.randn(1, 4, 64, 64).astype(np.float32)
noise = np.random.randn(1, 4, 64, 64).astype(np.float32)
xt = scheduler.add_noise(x0, noise, np.array([500]))

# 反向扩散：去噪步骤
prev_xt, x0_pred = scheduler.step(noise_pred, 500, xt)
```

**核心方法：**
- `add_noise()`：前向扩散（在时间步t添加噪声）
- `step()`：反向扩散（单步去噪）
- `set_timesteps()`：配置推理步数
- `predict_start_from_noise()`：从噪声预测预测x0

## 训练过程

```python
from nanotorch.generative.stable_diffusion import (
    build_vae, build_unet, build_scheduler, LatentDiffusion
)

vae = build_vae()
unet = build_unet()
scheduler = build_scheduler()
pipeline = LatentDiffusion(vae, unet, scheduler)

# 训练步骤
images = Tensor(np.random.randn(batch_size, 3, 512, 512).astype(np.float32))
text_embeddings = Tensor(np.random.randn(batch_size, 77, 512).astype(np.float32))

loss, loss_value = pipeline.train_step(images, text_embeddings)
```

**训练算法：**
1. 使用VAE将图像编码到潜在空间
2. 随机采样时间步
3. 向潜在变量添加噪声（前向扩散）
4. 使用U-Net预测噪声
5. 计算预测噪声与实际噪声之间的MSE损失

## 生成过程

```python
# 配置推理参数
scheduler.set_timesteps(50)  # 使用50步去噪

# 从随机噪声开始
latents = np.random.randn(1, 4, 64, 64).astype(np.float32)

# 编码文本提示
text_embedding = encode_text("海上美丽的日落")

# 去噪循环
for t in scheduler.timesteps:
    # 预测噪声
    noise_pred = unet(
        Tensor(latents), 
        np.array([int(t)]), 
        text_embedding
    )
    
    # 去除噪声
    latents, _ = scheduler.step(noise_pred.data, int(t), latents)

# 解码为图像
image = vae.decode(Tensor(latents))
```

## 数学原理

### 前向扩散公式

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$$

其中：
- $x_0$ 是原始图像
- $x_t$ 是时间步 $t$ 处的噪声版本
- $\bar{\alpha}_t = \prod_{i=0}^{t} \alpha_i$（alpha的累积乘积）
- $\epsilon \sim \mathcal{N}(0, I)$ 是高斯噪声

### 反向扩散公式

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \cdot \epsilon_\theta(x_t, t) \right) + \sigma_t \cdot z$$

其中：
- $\epsilon_\theta(x_t, t)$ 是 U-Net 预测的噪声
- $\beta_t = 1 - \alpha_t$ 是噪声调度参数
- $z \sim \mathcal{N}(0, I)$ 是随机噪声（$t=0$ 时除外）

### VAE 损失函数（ELBO）

$$\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

其中：
- $q(z|x)$ 是编码器分布
- $p(x|z)$ 是解码器分布
- $p(z) = \mathcal{N}(0, I)$ 是先验分布
- $D_{KL}$ 是 KL 散度

## 实现细节

### 时间步嵌入
```python
def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = np.log(10000.0) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim) * -emb)
    emb = timesteps[:, np.newaxis] * emb[np.newaxis, :]
    return np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
```

### 交叉注意力机制
- Query：来自空间特征
- Key/Value：来自文本嵌入
- 实现基于文本的条件控制

### 无分类器引导（Classifier-Free Guidance）
```python
# 无条件 + 条件预测
noise_uncond = unet(latents, t, empty_embedding)
noise_cond = unet(latents, t, text_embedding)

# 引导
noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
```

## 文件结构

```
nanotorch/generative/stable_diffusion/
├── __init__.py          # 模块导出
├── vae.py               # VAE实现
├── unet.py              # U-Net实现
└── diffusion.py         # 噪声调度器和管道

tests/generative/stable_diffusion/
├── test_vae.py          # 30个VAE测试
├── test_unet.py         # 27个U-Net测试
├── test_diffusion.py    # 25个扩散测试
└── test_integration.py  # 15个集成测试

examples/stable_diffusion/
└── demo.py              # 演示脚本
```

## 测试结果

运行测试：
```bash
python -m pytest tests/generative/stable_diffusion/ -v
```

所有 **97个测试** 全部通过：
- VAE测试：30个
- U-Net测试：27个
- 扩散测试：25个
- 集成测试：15个

## 运行演示

```bash
cd /Users/yangyang/ai_projs/nanotorch
PYTHONPATH=. python examples/stable_diffusion/demo.py
```

演示内容包括：
1. VAE编码/解码
2. U-Net去噪
3. 扩散调度器
4. 生成模拟
5. 训练步骤

## 参考文献

1. "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., CVPR 2022)
2. "Denoising Diffusion Probabilistic Models" (Ho et al., NeurIPS 2020)
3. Stable Diffusion GitHub: https://github.com/CompVis/stable-diffusion
