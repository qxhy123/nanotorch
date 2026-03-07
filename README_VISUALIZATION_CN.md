# Transformer 可视化项目

基于 nanotorch 的 Transformer 模型完整交互式可视化系统。

## ✅ 测试状态

所有组件已通过测试：
- ✅ nanotorch 导入测试
- ✅ TransformerWrapper 测试
- ✅ 前端构建测试
- ✅ 后端 API 测试

## 📁 项目结构

```
nanotorch/
├── frontend/                    # React + TypeScript 前端
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/              # shadcn/ui 基础组件
│   │   │   ├── visualization/   # 可视化组件
│   │   │   │   ├── embedding/   # Token/Positional Encoding
│   │   │   │   ├── attention/   # Multi-Head Attention
│   │   │   │   ├── feedforward/ # FFN
│   │   │   │   ├── normalization/ # LayerNorm
│   │   │   │   └── transformer/ # 完整流程
│   │   │   ├── controls/        # 参数控制面板
│   │   │   └── layout/          # 布局组件
│   │   ├── stores/              # Zustand 状态管理
│   │   ├── services/            # API 服务
│   │   ├── types/               # TypeScript 类型
│   │   └── utils/               # 工具函数
│   └── package.json
│
├── backend/                     # FastAPI 后端
│   ├── app/
│   │   ├── main.py              # FastAPI 应用入口
│   │   ├── api/routes/          # API 路由
│   │   ├── core/                # Transformer 包装类
│   │   ├── models/              # Pydantic 模型
│   │   └── utils/               # 工具函数
│   └── requirements.txt
│
├── start-backend.sh             # 启动后端脚本
├── start-frontend.sh            # 启动前端脚本
└── test_setup.sh               # 测试脚本
```

## 🚀 快速开始

### 方法一：使用启动脚本（推荐）

**终端 1 - 启动后端：**
```bash
./start-backend.sh
```

**终端 2 - 启动前端：**
```bash
./start-frontend.sh
```

然后访问：http://localhost:5173

### 方法二：手动启动

**后端：**
```bash
cd backend
export PYTHONPATH=/Users/yangyang/ai_projs/nanotorch:/Users/yangyang/ai_projs/nanotorch/backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**前端：**
```bash
cd frontend
npm run dev
```

## 🎨 可视化功能

### 1. Token Embedding（词嵌入）
- 热力图展示每个 token 的 embedding 向量
- 统计信息（最小值、最大值、均值）
- 支持显示/隐藏数值

### 2. Positional Encoding（位置编码）
- 正弦波形可视化
- 2D 热力图展示编码矩阵
- 公式说明

### 3. Attention Matrix（注意力矩阵）
- 交互式注意力权重热力图
- 多头选择器（H1-H8）
- 多种配色方案（viridis, plasma, blues, reds, inferno）
- 悬停显示具体数值

### 4. Multi-Head Attention（多头注意力）
- 架构图展示
- 计算步骤说明
- 头配置详情

### 5. Feed Forward Network（前馈网络）
- 网络架构可视化
- 激活函数对比（ReLU/GELU）
- 参数数量计算

### 6. Layer Normalization（层归一化）
- 归一化前后对比
- 交互式数据生成
- Gamma/Beta 参数效果

### 7. Transformer Flow（完整流程）
- 流水线动画
- 逐步导航
- 播放/暂停控制

## 🔧 配置选项

### 模型参数
- `d_model`: 模型维度（512）
- `nhead`: 注意力头数（8）
- `num_encoder_layers`: 编码器层数（6）
- `num_decoder_layers`: 解码器层数（6）
- `dim_feedforward`: FFN 维度（2048）
- `dropout`: Dropout 率（0.1）
- `activation`: 激活函数（relu/gelu）
- `max_seq_len`: 最大序列长度（128）
- `vocab_size`: 词汇表大小（10000）

## 📡 API 端点

- `GET /health` - 健康检查
- `POST /api/v1/transformer/forward` - 前向传播
- `POST /api/v1/transformer/attention` - 获取注意力权重
- `POST /api/v1/transformer/embeddings` - 获取嵌入
- `GET /api/v1/transformer/positional-encoding` - 获取位置编码
- `POST /api/v1/transformer/validate-config` - 验证配置

API 文档：http://localhost:8000/docs

## 🛠️ 技术栈

### 前端
- React 18.3 + TypeScript 5.3
- Vite 5.0（构建工具）
- TailwindCSS v3（样式）
- shadcn/ui（UI 组件）
- Zustand（状态管理）
- Axios（HTTP 客户端）

### 后端
- FastAPI（Web 框架）
- uvicorn（ASGI 服务器）
- Pydantic（数据验证）
- nanotorch（Tensor/NN 库）

## 📝 开发说明

### 运行测试
```bash
bash test_setup.sh
```

### 构建前端
```bash
cd frontend
npm run build
```

### 安装后端依赖
```bash
cd backend
pip install -r requirements.txt
```

## ⚠️ 注意事项

1. 确保 nanotorch 在项目根目录下可用
2. 后端需要设置正确的 PYTHONPATH
3. 前端默认连接 `http://localhost:8000`
4. 如需更改 API 地址，编辑 `frontend/.env`

## 📄 许可证

本项目基于 nanotorch 项目开发。
