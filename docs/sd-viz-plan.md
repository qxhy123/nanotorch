# Stable Diffusion 可视化子项目实施计划

## 项目概述

创建一个独立的 Stable Diffusion 交互式可视化子项目，用于教育和理解扩散模型的工作原理。

**项目定位：**
- 教育性质，帮助理解 Stable Diffusion 的核心概念
- 纯前端实现，使用预计算/模拟数据
- 与 nanotorch 主项目独立，但保持设计风格一致
- 支持中英文双语

---

## 一、技术架构

### 1.1 技术栈

**前端:**
```
- React 19 + TypeScript
- Vite (构建工具)
- TailwindCSS (样式)
- shadcn/ui (UI 组件)
- Recharts / D3.js (数据可视化)
- Framer Motion (动画)
- KaTeX (数学公式)
- Three.js (3D 潜空间可视化)
- Zustand (状态管理)
- React Query (数据获取)
```

**后端 (可选):**
```
- Python + FastAPI
- diffusers (Hugging Face)
- PyTorch
- CUDA/GPU 支持
```

**项目结构:**
```
sd-viz/
├── sd-viz-frontend/          # 前端项目
│   ├── src/
│   │   ├── components/      # 可视化组件
│   │   ├── views/           # 页面视图
│   │   ├── stores/          # 状态管理
│   │   ├── services/        # API 服务
│   │   ├── utils/           # 工具函数
│   │   ├── data/            # 预计算数据/模拟数据
│   │   └── types/           # TypeScript 类型
│   ├── public/
│   └── package.json
│
├── sd-viz-backend/          # 后端项目 (可选)
│   ├── app/
│   │   ├── api/            # API 路由
│   │   ├── core/           # 核心逻辑
│   │   └── models/         # SD 模型封装
│   └── requirements.txt
│
└── docs/                    # 文档
    ├── design.md
    └── api.md
```

---

## 二、功能模块设计

### 2.1 核心可视化模块

#### 模块 1: 架构概览 (Architecture Overview)
**路径:** `/`

**功能:**
- SD 整体架构流程图
- 三大组件交互 (CLIP, UNet, VAE)
- 数据流向可视化
- 组件详解卡片

**实现要点:**
```typescript
- React Flow 架构图
- 交互式组件点击展开详情
- 动画展示数据流向
- 支持缩放和拖拽
```

---

#### 模块 2: 扩散过程可视化 (Diffusion Process)
**路径:** `/diffusion`

**功能:**
- 前向扩散过程 (加噪)
- 反向扩散过程 (去噪)
- 逐步去噪动画
- 不同时间步的图像/潜空间状态
- 噪声调度曲线

**子组件:**
```typescript
- ForwardProcess: 展示 T 步加噪过程
- ReverseProcess: 展示 T 步去噪过程
- StepSlider: 时间步滑块控制
- NoiseSchedule: 噪声调度 β_t 可视化
- SideBySideComparison: 同步对比不同时间步
```

**数据结构:**
```typescript
interface DiffusionStep {
  timestep: number;
  noise: number[];        // 潜空间噪声
  image: number[];        // 图像/潜空间表示
  alpha_bar: number;      // 累积噪声调度
  snr: number;            // 信噪比
}
```

---

#### 模块 3: UNet 架构可视化 (UNet Architecture)
**路径:** `/unet`

**功能:**
- UNet 3D 架构图
- 下采样/上采样路径
- ResNet Block 结构
- Attention Block 详情
- 时间步和类别嵌入注入点

**子组件:**
```typescript
- UNet3DView: Three.js 3D 模型
- LayerDetailCard: 层详情卡片
- AttentionMap: 注意力地图可视化
- SkipConnection: 跳跃连接动画
- TensorShapeDisplay: 张量形状标注
```

**架构层级:**
```
输入: [B, 4, 64, 64] (潜空间)
  ↓
DownBlock × N (每个: ResNet + Attention)
  → [B, 320/640/1280, 8/16/32, 8/16/32]
  ↓
MiddleBlock
  ↓
UpBlock × N (每个: ResNet + Attention)
  → [B, 4, 64, 64]
  ↓
输出: 预测噪声
```

---

#### 模块 4: 注意力机制可视化 (Attention)
**路径:** `/attention`

**功能:**
- Self-Attention 地图
- Cross-Attention 地图 (文本-图像)
- 多头注意力分解
- 注意力权重随时间步变化
- 不同 token 对生成的影响

**子组件:**
```typescript
- SelfAttentionGrid: Self-attention 热力图
- CrossAttentionMatrix: Token-token 注意力矩阵
- AttentionHeadComparison: 多头对比
- TemporalAttentionEvolution: 时间演化动画
- TokenInfluenceBar: Token 影响力条形图
```

**示例数据:**
```typescript
interface AttentionData {
  timestep: number;
  layer: number;
  head: number;
  tokens: string[];           // 文本 tokens
  attentionMap: number[][];   // [spatial, token]
  queryPosition: [number, number];
}
```

---

#### 模块 5: 文本编码器 (Text Encoder)
**路径:** `/text-encoder`

**功能:**
- CLIP 文本编码器架构
- Token 到嵌入的转换
- Text Embeddings 可视化
- Token 对生成的影响分析
- 提示词工程工具

**子组件:**
```typescript
- PromptEditor: 提示词编辑器
- TokenVisualization: Token 展示
- EmbeddingProjection: 2D 投影
- AttentionWeights: Token 注意力权重
- PromptOptimizer: 提示词优化建议
```

---

#### 模块 6: 采样算法对比 (Sampling Methods)
**路径:** `/sampling`

**功能:**
- DDPM vs DDIM vs Euler vs DPM-Solver
- 采样步数对比
- 生成速度 vs 质量权衡
- 采样轨迹可视化
- 自定义采样参数

**子组件:**
```typescript
- SamplingMethodSelector: 算法选择
- StepComparison: 步数并排对比
- QualityMetrics: 质量指标 (FID, CLIP Score)
- SpeedChart: 速度对比图表
- TrajectoryVisualization: 潜空间轨迹
- ParameterTuner: 参数调节器
```

**对比表格:**
```typescript
interface SamplingMethod {
  name: string;
  steps: number[];
  speed: number;
  quality: number;
  deterministic: boolean;
  description: string;
}
```

---

#### 模块 7: 潜空间探索 (Latent Space)
**路径:** `/latent-space`

**功能:**
- VAE 编码/解码可视化
- 潜空间 3D 交互
- 潜向量插值
- 方向操纵 (smile, age 等)
- 扰动影响

**子组件:**
```typescript
- LatentSpace3D: Three.js 3D 散点图
- VAEEncoderDecoder: VAE 可视化
- VectorInterpolator: 向量插值工具
- DirectionEditor: 方向编辑器
- NoiseExplorer: 噪声探索
```

---

#### 模块 8: 控制生成 (Controlled Generation)
**路径:** `/control`

**功能:**
- ControlNet 架构
- T2I-Adapter 流程
- 输入条件可视化 (depth, edge, pose)
- ZeroConv 和控制注入
- 多控制器组合

**子组件:**
```typescript
- ControlNetArch: ControlNet 架构图
- ConditionInput: 条件输入上传
- ControlWeightSlider: 权重滑块
- LayerInjection: 层注入可视化
- MultiControlComposer: 多控制器组合
```

---

#### 模块 9: 生成游乐场 (Generation Playground)
**路径:** `/playground`

**功能:**
- 完整生成流程演示
- 参数实时调节
- 逐步生成动画
- 中间结果查看
- Batch 生成对比

**子组件:**
```typescript
- PromptBuilder: 提示词构建器
- ParameterPanel: 参数面板
  - prompt: string
  - negative_prompt: string
  - steps: 1-100
  - guidance_scale: 1-20
  - seed: number
  - width/height: [256, 512, 768, 1024]
- GenerationProgress: 生成进度条
- StepPreview: 步骤预览
- ResultGallery: 结果画廊
- BatchComparison: 批量对比
```

---

#### 模块 10: 教学教程 (Tutorials)
**路径:** `/tutorials`

**功能:**
- 交互式教程
- 概念解释动画
- 小测验
- 进度追踪

**教程内容:**
```typescript
const tutorials = [
  {
    id: 'basics',
    title: '扩散模型基础',
    lessons: [
      '什么是扩散模型',
      '前向过程',
      '反向过程',
      '训练目标',
    ]
  },
  {
    id: 'architecture',
    title: 'SD 架构解析',
    lessons: [
      '整体架构',
      'UNet详解',
      'VAE编码器',
      'CLIP编码器',
    ]
  },
  {
    id: 'advanced',
    title: '进阶话题',
    lessons: [
      '采样算法',
      'ControlNet',
      'LoRA',
      '提示词工程',
    ]
  }
]
```

---

### 2.2 交互设计

#### 导航结构
```typescript
const navigation = [
  { name: '架构概览', path: '/', icon: Architecture },
  { name: '扩散过程', path: '/diffusion', icon: Waves },
  { name: 'UNet架构', path: '/unet', icon: Box },
  { name: '注意力', path: '/attention', icon: Eye },
  { name: '文本编码', path: '/text-encoder', icon: Type },
  { name: '采样算法', path: '/sampling', icon: Zap },
  { name: '潜空间', path: '/latent-space', icon: Cube },
  { name: '控制生成', path: '/control', icon: Sliders },
  { name: '生成游乐场', path: '/playground', icon: Sparkles },
  { name: '教程', path: '/tutorials', icon: BookOpen },
]
```

---

### 2.3 数据设计

#### 预计算数据集

**示例生成过程:**
```typescript
interface GenerationProcess {
  id: string;
  prompt: string;
  steps: GenerationStep[];
  config: GenerationConfig;
}

interface GenerationStep {
  timestep: number;
  latent: number[];           // [4, 64, 64] 潜空间
  noise_pred: number[];       // 预测的噪声
  denoised: number[];         // 去噪后的潜空间
  image: string;              // base64 图像
}
```

**架构数据:**
```typescript
interface UNetLayer {
  name: string;
  type: 'resnet' | 'attention' | 'downsample' | 'upsample';
  inputShape: number[];
  outputShape: number[];
  channels: number;
  parameters: number;
}
```

---

## 三、实施阶段

### 阶段 1: 项目初始化 (1-2 天)

**任务:**
1. 创建前端项目 (Vite + React + TS)
2. 配置 TailwindCSS + shadcn/ui
3. 设置路由 (React Router)
4. 配置状态管理 (Zustand)
5. 创建基础布局
6. 设置中英文 i18n

**验收标准:**
- 项目可运行
- 路由正常工作
- UI 组件库可用

---

### 阶段 2: 架构概览 (2-3 天)

**任务:**
1. 实现 SD 架构流程图
2. 三大组件卡片
3. 数据流向动画
4. 组件详情弹窗

**组件:**
```typescript
- ArchitectureFlow.tsx
- ComponentCard.tsx
- DataFlowAnimation.tsx
- ComponentDetailModal.tsx
```

---

### 阶段 3: 扩散过程可视化 (3-4 天)

**任务:**
1. 前向过程演示
2. 反向过程演示
3. 时间步滑块控制
4. 噪声调度曲线
5. 预计算/生成示例数据

**组件:**
```typescript
- DiffusionProcess.tsx
- ForwardProcess.tsx
- ReverseProcess.tsx
- StepSlider.tsx
- NoiseScheduleChart.tsx
```

**数据生成脚本:**
```python
# scripts/generate_diffusion_data.py
- 生成不同时间步的潜空间状态
- 模拟加噪/去噪过程
- 保存为 JSON
```

---

### 阶段 4: UNet 架构 (3-4 天)

**任务:**
1. 3D UNet 可视化 (Three.js)
2. 下/上采样路径
3. ResNet Block 详情
4. Attention Block 可视化
5. 张量形状标注

**组件:**
```typescript
- UNet3DView.tsx
- UNetLayerDetail.tsx
- ResNetBlockCard.tsx
- AttentionBlockCard.tsx
- TensorShapeLabel.tsx
```

---

### 阶段 5: 注意力可视化 (3-4 天)

**任务:**
1. Self-Attention 热力图
2. Cross-Attention 矩阵
3. 多头对比
4. 时间演化动画
5. Token 影响力分析

**组件:**
```typescript
- AttentionView.tsx
- SelfAttentionGrid.tsx
- CrossAttentionMatrix.tsx
- HeadComparison.tsx
- TokenInfluenceBar.tsx
```

---

### 阶段 6: 采样算法对比 (2-3 天)

**任务:**
1. 算法选择器
2. 步数对比
3. 质量/速度图表
4. 采样轨迹
5. 参数调节

**组件:**
```typescript
- SamplingComparison.tsx
- MethodSelector.tsx
- StepComparisonView.tsx
- QualityMetricsChart.tsx
- TrajectoryViz.tsx
```

---

### 阶段 7: 潜空间探索 (2-3 天)

**任务:**
1. VAE 可视化
2. 3D 潜空间散点图
3. 向量插值
4. 方向操纵

**组件:**
```typescript
- LatentSpaceView.tsx
- VAEEncoderDecoder.tsx
- Latent3DScatter.tsx
- VectorInterpolator.tsx
```

---

### 阶段 8: 生成游乐场 (3-4 天)

**任务:**
1. 提示词编辑器
2. 参数面板
3. 生成进度
4. 结果画廊

**组件:**
```typescript
- Playground.tsx
- PromptBuilder.tsx
- ParameterPanel.tsx
- GenerationProgress.tsx
- ResultGallery.tsx
```

**后端集成 (可选):**
```python
# sd-viz-backend/app/api/routes/generation.py
@router.post("/generate")
async def generate_image(request: GenerationRequest):
    # 调用 diffusers
    # 返回生成过程
```

---

### 阶段 9: 教程系统 (2-3 天)

**任务:**
1. 教程列表
2. 课程播放器
3. 交互式演示
4. 进度追踪

**组件:**
```typescript
- TutorialsList.tsx
- LessonPlayer.tsx
- ConceptAnimation.tsx
- Quiz.tsx
- ProgressTracker.tsx
```

---

### 阶段 10: 打包与部署 (1-2 天)

**任务:**
1. 性能优化
2. 响应式适配
3. 打包配置
4. 文档完善
5. 部署

---

## 四、UI/UX 设计

### 4.1 设计系统

**颜色方案:**
```typescript
const colors = {
  primary: {
    50: '#f0f9ff',
    500: '#0ea5e9',
    900: '#0c4a6e',
  },
  diffusion: {
    from: '#3b82f6',  // 蓝色 (清晰)
    to: '#8b5cf6',    // 紫色 (噪声)
  },
  attention: {
    low: '#ef4444',
    medium: '#f59e0b',
    high: '#22c55e',
  }
}
```

**布局组件:**
```typescript
- MainLayout: 主布局 (顶部导航 + 内容区)
- Sidebar: 侧边栏 (导航)
- ContentCard: 内容卡片
- ControlPanel: 控制面板
- VisualizationCanvas: 可视化画布
```

---

### 4.2 交互模式

**滑块控制:**
```typescript
<Slider
  value={[timestep]}
  min={0}
  max={1000}
  step={1}
  onChange={setTimestep}
/>
<Label>时间步: {timestep}</Label>
```

**参数调节:**
```typescript
<ParameterControl
  name="Guidance Scale"
  value={guidanceScale}
  min={1}
  max={20}
  step={0.5}
  onChange={setGuidanceScale}
/>
```

**播放控制:**
```typescript
<PlaybackControls
  isPlaying={isPlaying}
  onPlay={handlePlay}
  onPause={handlePause}
  onReset={handleReset}
  onStepForward={handleStepForward}
  onStepBackward={handleStepBackward}
/>
```

---

## 五、数据准备

### 5.1 示例数据生成

**扩散过程数据:**
```python
# scripts/generate_diffusion_samples.py
import numpy as np
import json

def generate_diffusion_process(num_samples=10):
    """生成扩散过程示例数据"""
    processes = []

    for i in range(num_samples):
        steps = []
        for t in range(0, 1001, 50):  # 每50步
            alpha_bar = compute_alpha_bar(t)
            noise = np.random.randn(4, 64, 64)
            x0 = np.random.randn(4, 64, 64)  # "清晰"图像
            xt = np.sqrt(alpha_bar) * x0 + np.sqrt(1 - alpha_bar) * noise

            steps.append({
                'timestep': t,
                'latent': xt.tolist(),
                'noise': noise.tolist(),
                'alpha_bar': float(alpha_bar),
                'snr': float(alpha_bar / (1 - alpha_bar))
            })

        processes.append({
            'id': f'sample_{i}',
            'steps': steps
        })

    return processes
```

**注意力数据:**
```python
def generate_attention_maps():
    """生成注意力地图示例"""
    # 使用真实的 SD 模型提取注意力
    # 或使用模拟数据
    pass
```

---

### 5.2 数据格式

**JSON 结构:**
```json
{
  "version": "1.0",
  "samples": [
    {
      "id": "sample_001",
      "prompt": "a beautiful landscape",
      "negative_prompt": "",
      "steps": 50,
      "guidance_scale": 7.5,
      "seed": 42,
      "size": [512, 512],
      "process": [
        {
          "timestep": 0,
          "latent": [...],
          "noise_pred": [...],
          "image": "data:image/png;base64,..."
        }
      ],
      "final_image": "data:image/png;base64,..."
    }
  ]
}
```

---

## 六、技术挑战与解决方案

### 6.1 性能优化

**挑战:** 大量数据可视化导致性能问题

**解决方案:**
```typescript
// 1. 虚拟化长列表
import { useVirtualizer } from '@tanstack/react-virtual';

// 2. 懒加载组件
const LazyComponent = lazy(() => import('./HeavyComponent'));

// 3. Web Worker 处理数据
const worker = new Worker('/data-processor.worker.js');

// 4. 图像懒加载
<img loading="lazy" src={imageSrc} />

// 5. 数据分页
const paginatedData = useMemo(() =>
  data.slice(page * pageSize, (page + 1) * pageSize),
  [data, page, pageSize]
);
```

---

### 6.2 3D 渲染优化

**挑战:** Three.js 性能问题

**解决方案:**
```typescript
// 1. 实例化渲染
const instancedMesh = new InstancedMesh(geometry, material, count);

// 2. LOD (Level of Detail)
const lod = new LOD();
lod.addLevel(highPolyMesh, 0);
lod.addLevel(lowPolyMesh, 50);

// 3. 降低面数
const simplifiedGeometry = geometry.clone();
simplifiedGeometry.scale(0.5, 0.5, 0.5);

// 4. 使用 BufferGeometry
const geometry = new BufferGeometry();
```

---

### 6.3 状态管理

**挑战:** 复杂的跨组件状态

**解决方案:**
```typescript
// 使用 Zustand
interface AppState {
  timestep: number;
  setTimestep: (t: number) => void;
  prompt: string;
  setPrompt: (p: string) => void;
  // ...
}

const useAppStore = create<AppState>((set) => ({
  timestep: 0,
  setTimestep: (t) => set({ timestep: t }),
  prompt: '',
  setPrompt: (p) => set({ prompt: p }),
}));
```

---

## 七、项目时间线

| 阶段 | 任务 | 预计时间 |
|-----|------|---------|
| 1 | 项目初始化 | 1-2 天 |
| 2 | 架构概览 | 2-3 天 |
| 3 | 扩散过程 | 3-4 天 |
| 4 | UNet 架构 | 3-4 天 |
| 5 | 注意力可视化 | 3-4 天 |
| 6 | 采样算法 | 2-3 天 |
| 7 | 潜空间探索 | 2-3 天 |
| 8 | 生成游乐场 | 3-4 天 |
| 9 | 教程系统 | 2-3 天 |
| 10 | 打包部署 | 1-2 天 |

**总计:** 约 25-35 天

---

## 八、成功指标

### 8.1 功能完整性

- [ ] 10 个核心可视化模块全部实现
- [ ] 交互功能正常运行
- [ ] 动画流畅 (60fps)
- [ ] 响应式设计适配移动端

### 8.2 教育价值

- [ ] 概念解释清晰易懂
- [ ] 可视化直观准确
- [ ] 教程覆盖核心概念
- [ ] 交互式学习体验

### 8.3 技术质量

- [ ] TypeScript 类型覆盖率 > 90%
- [ ] 单元测试覆盖率 > 70%
- [ ] 构建体积 < 2MB (gzipped)
- [ ] 首屏加载 < 3s

---

## 九、后续扩展

### 9.1 短期扩展

- LoRA 可视化
- IP-Adapter 适配器
- AnimateDiff 动画生成
- SDXL 支持
- SD3 架构

### 9.2 长期扩展

- 其他扩散模型 (MusicGen, AudioLDM)
- 视频生成模型 (SVD, CogVideoX)
- 3D 生成模型 (Shap-E)
- 模型微调界面
- A/B 测试工具

---

## 十、参考资源

### 论文
- Ho et al. "Denoising Diffusion Probabilistic Models" (2020)
- Song et al. "Denoising Diffusion Implicit Models" (2021)
- Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" (2022)
- Zhang et al. "ControlNet" (2023)

### 代码
- Hugging Face Diffusers
- CompVis/stable-diffusion
- AUTOMATIC1111/stable-diffusion-webui

### 可视化参考
- Hugging Face Diffusion Visualization
- Google's DreamBooth Demo
- Various explainer articles

---

**文档版本:** v1.0
**最后更新:** 2026-03-09
**维护者:** nanotorch 团队
