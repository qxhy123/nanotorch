# Transformer 可视化快速入门

这是仓库内可视化应用的中文快速入门。英文版请查看 [`QUICKSTART.md`](./QUICKSTART.md)。

本指南面向仓库级 frontend/backend 可视化系统，而不是发布到包管理器里的 `nanotorch` Python 包本体。要获得完整体验，需要同时启动前端和后端。

## 应用包含的能力

- 前端界面：Transformer 结构、embedding、attention、layer flow、tokenization、inference、training 相关视图
- FastAPI 后端：封装 Transformer、tokenizer、layer analysis 等接口
- 共享底层 `nanotorch` Python 包能力

## 环境要求

- Python `3.8+`
- Node.js `18+` 与 `npm`
- 本仓库的本地代码副本

## 安装

### 1. 准备 Python 环境

在仓库根目录执行：

```bash
uv venv
source .venv/bin/activate
uv sync
python -m pip install -r backend/requirements.txt
```

`uv sync` 安装的是 `pyproject.toml` 中定义的库依赖；可视化后端还需要 `backend/requirements.txt` 中的 FastAPI 相关依赖。

### 2. 安装前端依赖

```bash
cd frontend
npm install
cd ..
```

## 启动应用

### 推荐：手动分别启动两个服务

后端：

```bash
cd backend
PYTHONPATH="$(pwd)/..:$(pwd)" python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

前端，在第二个终端中执行：

```bash
cd frontend
npm run dev
```

然后访问：

- 前端应用：`http://localhost:5173`
- 后端健康检查：`http://localhost:8000/health`
- 后端 API 文档：`http://localhost:8000/docs`

### 可选：使用仓库内启动脚本

仓库中还提供了 `./start-backend.sh` 和 `./start-frontend.sh`。这两个脚本是当前工作区的便捷封装；如果你移动了仓库位置，请先确认里面的硬编码路径是否仍然正确。

## 快速验证

### 检查后端是否存活

```bash
curl http://localhost:8000/health
```

期望结果：返回包含 backend 状态和 nanotorch 版本信息的 JSON。

### 运行 API 冒烟脚本

在仓库根目录执行：

```bash
python test_api.py
```

## 主要 API 能力

当前后端主要暴露三组接口：

- `/api/v1/transformer` 下的 Transformer 接口
- `/api/v1/tokenizer` 下的 tokenizer 接口
- `/api/v1/layer` 下的 layer analysis 接口

常见示例：

```bash
curl -X POST http://localhost:8000/api/v1/transformer/forward \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "d_model": 128,
      "nhead": 8,
      "num_encoder_layers": 2,
      "num_decoder_layers": 0,
      "dim_feedforward": 256,
      "dropout": 0.1,
      "activation": "relu",
      "max_seq_len": 64,
      "vocab_size": 1000,
      "layer_norm_eps": 1e-5,
      "batch_first": true,
      "norm_first": false
    },
    "input_data": {
      "text": "Hello world"
    },
    "options": {
      "return_attention": true,
      "return_all_layers": true,
      "return_embeddings": true
    }
  }'
```

```bash
curl -X POST http://localhost:8000/api/v1/tokenizer/tokenize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "hello nanotorch",
    "tokenizer_type": "char"
  }'
```

## 重要说明

- Transformer 路由使用的是 `input_data`，不是 `input`。
- 前端默认访问 `http://localhost:8000` 上的后端。
- 完整可视化能力依赖前端 dev server 和后端 API 同时运行。
- 本快速入门只覆盖可视化应用；Python 包的使用示例请看 [`README.md`](./README.md) 和 [`README_CN.md`](./README_CN.md)。

## 故障排查

### 后端导入失败

确认虚拟环境已经激活，并且已安装 `backend/requirements.txt`。如果你是手动启动后端，要保证 `PYTHONPATH` 同时包含仓库根目录和 `backend/`。

### 前端连不上后端

- 确认后端已启动：`curl http://localhost:8000/health`
- 确认前端 dev server 在 `http://localhost:5173` 正常运行
- 如果你改过 API 地址，检查前端环境变量配置

### API 参数校验失败

先打开 `http://localhost:8000/docs` 查看当前 schema，再核对 `input_data`、`config`、`options` 等字段名。

## 相关文档

- [`README.md`](./README.md)：英文总览与包使用说明
- [`README_CN.md`](./README_CN.md)：中文总览与包使用说明
- [`README_VISUALIZATION.md`](./README_VISUALIZATION.md)：英文可视化专项文档
- [`README_VISUALIZATION_CN.md`](./README_VISUALIZATION_CN.md)：中文可视化专项文档
