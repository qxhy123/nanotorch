# Transformer 可视化 - 快速入门指南

## ✅ 已验证：所有组件正常工作

- ✅ 后端 API 所有端点测试通过
- ✅ 前端构建成功
- ✅ nanotorch 集成正常

## 🚀 启动应用

### 快速启动（推荐）

```bash
# 终端 1 - 启动后端
cd /Users/yangyang/ai_projs/nanotorch/backend
PYTHONPATH=/Users/yangyang/ai_projs/nanotorch:/Users/yangyang/ai_projs/nanotorch/backend python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 终端 2 - 启动前端
cd /Users/yangyang/ai_projs/nanotorch/frontend
npm run dev
```

### 或使用启动脚本

```bash
# 终端 1
./start-backend.sh

# 终端 2
./start-frontend.sh
```

## 📍 访问地址

- **前端应用**: http://localhost:5173
- **后端 API**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs

## 🧪 测试 API

```bash
python test_api.py
```

或使用 curl:

```bash
# 健康检查
curl http://localhost:8000/health

# 前向传播
curl -X POST http://localhost:8000/api/v1/transformer/forward \
  -H "Content-Type: application/json" \
  -d '{
    "config": {"d_model": 512, "nhead": 8, "num_encoder_layers": 2, "num_decoder_layers": 0, "dim_feedforward": 2048, "dropout": 0.1, "activation": "relu", "max_seq_len": 128, "vocab_size": 10000, "layer_norm_eps": 1e-5, "batch_first": true, "norm_first": false},
    "input_data": {"text": "Hello world"},
    "options": {"return_attention": true, "return_all_layers": true, "return_embeddings": true}
  }'
```

## 📋 API 请求格式

### Forward 端点

```json
POST /api/v1/transformer/forward

{
  "config": {
    "d_model": 512,
    "nhead": 8,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "activation": "relu",
    "max_seq_len": 128,
    "vocab_size": 10000,
    "layer_norm_eps": 1e-5,
    "batch_first": true,
    "norm_first": false
  },
  "input_data": {
    "text": "Hello world",
    "tokens": [72, 101, 108, 108, 111]  // optional
  },
  "options": {
    "return_attention": true,
    "return_all_layers": true,
    "return_embeddings": true
  }
}
```

**响应格式：**

```json
{
  "success": true,
  "data": {
    "final_output": {
      "shape": [1, 11, 512],
      "data": [...],
      "dtype": "float32"
    },
    "embeddings": {
      "token_embeddings": {...},
      "positional_encodings": {...},
      "combined": {...}
    },
    "attention_weights": [...],
    "layer_outputs": [...]
  }
}
```

## ⚠️ 重要说明

1. **参数名称**：后端使用 `input_data`（不是 `input`）
2. **Python 路径**：需要设置 `PYTHONPATH` 包含项目根目录和 backend 目录
3. **CORS 配置**：后端已配置允许 `localhost:5173` 的跨域请求

## 🔍 故障排查

### 后端启动失败

```bash
# 检查 nanotorch 是否可用
python -c "
import sys
sys.path.insert(0, '/Users/yangyang/ai_projs/nanotorch')
from nanotorch.nn.transformer import Transformer
print('✓ nanotorch OK')
"
```

### 前端无法连接后端

1. 检查后端是否运行：`curl http://localhost:8000/health`
2. 检查 `.env` 文件中的 `VITE_API_BASE_URL`
3. 查看浏览器控制台的网络请求

### API 参数错误

确保请求体使用 `input_data`（不是 `input`）：
```json
{
  "input_data": {...},   // ✓ 正确
  "input": {...}         // ✗ 错误
}
```

## 📚 相关文档

- [完整项目文档](./README_VISUALIZATION_CN.md)
- [nanotorch README](./README_CN.md)
- API 文档：http://localhost:8000/docs
