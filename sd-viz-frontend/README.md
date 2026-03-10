# SD-Viz - Stable Diffusion Visualization Education Platform

<div align="center">

  **Interactive Stable Diffusion Architecture Visualization Platform**

  [English](#english) | [中文](#中文)

</div>

---

## 中文

### 项目简介

SD-Viz 是一个用于教育目的的 Stable Diffusion 架构可视化平台。该项目通过交互式可视化帮助用户理解 Stable Diffusion 的工作原理，包括扩散过程、UNet 架构、注意力机制等核心概念。

### 主要特性

- 🔬 **架构概览** - SD 三大核心组件：CLIP 文本编码器、UNet 去噪网络、VAE
- 🌊 **扩散过程** - 前向和反向扩散过程的交互式动画
- 🏗️ **UNet 架构** - 3D 结构图，展示下采样、上采样和跳跃连接
- 👁️ **注意力机制** - 自注意力和交叉注意力的热力图可视化
- 📝 **文本编码器** - CLIP Token 分解和嵌入向量可视化
- ⚡ **采样方法** - DDPM、DDIM、Euler、DPM-Solver++ 对比
- 📦 **潜空间** - VAE 压缩和向量插值探索
- 🎛️ **控制生成** - ControlNet、T2I-Adapter、IP-Adapter 可视化
- 🎮 **生成游乐场** - 参数调节和生成模拟
- 📚 **交互式教程** - 分步学习扩散模型概念

### 技术栈

- **前端框架**: React 19 + TypeScript
- **构建工具**: Vite 7
- **样式**: TailwindCSS v4
- **状态管理**: Zustand
- **路由**: React Router v6
- **图表**: Recharts
- **动画**: Framer Motion
- **3D**: Three.js + React Three Fiber
- **数学公式**: KaTeX + React KaTeX
- **图标**: Lucide React

### 快速开始

#### 安装依赖

```bash
npm install
```

#### 开发服务器

```bash
npm run dev
```

访问 http://localhost:5173

#### 生产构建

```bash
npm run build
```

### 页面导航

| 路由 | 页面 | 描述 |
|------|------|------|
| `/` | 架构概览 | Stable Diffusion 三大核心组件介绍 |
| `/diffusion` | 扩散过程 | 前向/反向扩散过程可视化 |
| `/unet` | UNet 架构 | 3D 结构图和层详情 |
| `/attention` | 注意力机制 | 自注意力和交叉注意力可视化 |
| `/text-encoder` | 文本编码器 | CLIP token 分解和嵌入 |
| `/sampling` | 采样方法 | 不同采样算法对比 |
| `/latent-space` | 潜空间 | VAE 压缩和向量插值 |
| `/control` | 控制生成 | ControlNet 和 T2I-Adapter |
| `/playground` | 生成游乐场 | 参数调节模拟 |
| `/tutorials` | 教程 | 交互式学习教程 |

### 项目结构

```
sd-viz-frontend/
├── src/
│   ├── components/         # UI 组件
│   │   ├── ui/              # 基础组件
│   │   └── layout/           # 布局组件
│   ├── data/               # 示例数据
│   ├── stores/             # Zustand 状态管理
│   ├── types/              # TypeScript 类型
│   ├── views/              # 页面视图
│   └── App.tsx            # 主应用
├── public/                # 静态资源
└── package.json
```

### 参数说明

| 参数 | 说明 | 范围 |
|------|------|------|
| Steps | 采样步数 | 10-100 |
| CFG Scale | 提示词引导强度 | 1-20 |
| Seed | 随机种子 | 任意整数 |

### 许可证

MIT License

---

## English

### Project Overview

SD-Viz is an educational visualization platform for Stable Diffusion architecture. It helps users understand how Stable Diffusion works through interactive visualizations.

### Key Features

- 🔬 Architecture Overview
- 🌊 Diffusion Process
- 🏗️ UNet Architecture
- 👁️ Attention Mechanism
- 📝 Text Encoder
- ⚡ Sampling Methods
- 📦 Latent Space
- 🎛️ Controlled Generation
- 🎮 Generation Playground
- 📚 Interactive Tutorials

### Tech Stack

- React 19 + TypeScript
- Vite 7
- TailwindCSS v4
- Zustand
- React Router v6
- Recharts
- Framer Motion
- Three.js
- KaTeX

### Quick Start

```bash
# Install dependencies
npm install

# Development
npm run dev

# Build
npm run build
```

### License

MIT License
