# Transformer 可视化 - 调试指南

## 问题：点击 Forward Pass 后 Attention 页面没有反应

### 可能的原因和解决方案

#### 1. 打开浏览器控制台查看调试信息

**Chrome/Edge:**
- 按 `F12` 或右键 → 检查
- 切换到 "Console" 标签

**Firefox:**
- 按 `F12` 或右键 → 检查
- 切换到 "控制台" 标签

#### 2. 查看控制台日志

点击 "Run Forward Pass" 后，控制台应该显示类似以下信息：

```
API Request: POST /api/v1/transformer/forward
Attention weights loaded: {count: X, firstItem: {...}, hasWeights: true}
Processing attention data: {...}
Weights shape: [1, 8, seq_len, seq_len]
```

#### 3. 常见问题排查

**问题 1: CORS 错误**
```
Access to fetch at 'http://localhost:8000' has been blocked by CORS policy
```

**解决:** 确保后端正在运行：
```bash
cd backend
PYTHONPATH=/Users/yangyang/ai_projs/nanotorch:/Users/yangyang/ai_projs/nanotorch/backend python -m uvicorn app.main:app --reload
```

**问题 2: API 参数错误**
```
{"detail": [{"type": "missing", "loc": ["body", "input_data"], ...}]}
```

这已经在最新的代码中修复了。

**问题 3: 数据格式不匹配**
控制台显示：`Could not process weights data`

这说明后端返回的数据结构与前端期望的不符。

#### 4. 检查实际返回的数据

在控制台中手动执行以下代码查看数据：

```javascript
// 在浏览器控制台中执行
window.addEventListener('load', () => {
  // 点击 Forward Pass 后，在控制台执行：
  console.log('Store state:', window.useTransformerStore?.getState?.());
});
```

或者在 InputPanel.tsx 中的 `handleRun` 函数的 `setOutput(result)` 调用后添加日志：

```typescript
if (result.success && result.data) {
  console.log('API Response data:', result.data);
  console.log('Attention weights:', result.data.attention_weights);
  setOutput(result);
}
```

#### 5. 验证后端返回的数据结构

使用 curl 测试后端：

```bash
curl -X POST http://localhost:8000/api/v1/transformer/forward \
  -H "Content-Type: application/json" \
  -d '{
    "config": {"d_model": 512, "nhead": 8, "num_encoder_layers": 2, "num_decoder_layers": 0, "dim_feedforward": 2048, "dropout": 0.1, "activation": "relu", "max_seq_len": 128, "vocab_size": 10000, "layer_norm_eps": 1e-5, "batch_first": true, "norm_first": false},
    "input_data": {"text": "Hello"},
    "options": {"return_attention": true, "return_all_layers": true, "return_embeddings": true}
  }' | python -m json.tool
```

检查返回的 `attention_weights` 结构。

#### 6. 临时解决方案：添加数据加载状态指示器

在 `InputPanel.tsx` 的成功响应后添加状态显示：

```typescript
if (result.success && result.data) {
  console.log('✓ Forward pass successful');
  console.log('  - Final output shape:', result.data.final_output?.shape);
  console.log('  - Has embeddings:', !!result.data.embeddings);
  console.log('  - Attention weights:', result.data.attention_weights?.length || 0, 'layers');

  setOutput(result);

  // 显示成功消息
  alert(`Forward pass successful!\nOutput shape: ${result.data.final_output?.shape}`);
} else {
  console.error('✗ Forward pass failed:', result.error);
  alert(`Forward pass failed: ${result.error}`);
}
```

### 完整的调试步骤

1. **启动后端并验证**
```bash
# 检查后端健康状态
curl http://localhost:8000/health

# 测试 forward 端点（应该返回成功）
curl -X POST http://localhost:8000/api/v1/transformer/forward \
  -H "Content-Type: application/json" \
  -d '{"config": {"d_model": 512, "nhead": 8, "num_encoder_layers": 2, "num_decoder_layers": 0, "dim_feedforward": 2048, "dropout": 0.1, "activation": "relu", "max_seq_len": 128, "vocab_size": 10000, "layer_norm_eps": 1e-5, "batch_first": true, "norm_first": false}, "input_data": {"text": "Hello"}, "options": {"return_attention": true, "return_all_layers": true, "return_embeddings": true}}'
```

2. **打开前端并启动开发者工具**
   - F12 打开开发者工具
   - 切换到 Console 标签

3. **点击 "Run Forward Pass" 按钮**
   - 观察控制台输出
   - 检查是否有错误信息

4. **切换到 Attention 标签**
   - 如果仍显示 "Run a forward pass to see attention patterns"
   - 检查控制台是否有数据处理错误

### 已添加的调试功能

代码已更新，包含：
- Console 日志显示 attention weights 加载状态
- 更详细的数据处理逻辑
- 更好的错误消息显示

### 如果问题仍然存在

请提供以下信息：

1. **浏览器控制台的完整输出**
2. **后端 API 返回的原始 JSON**（使用 curl 测试）
3. **前端构建版本**（`npm run dev` 的输出）
4. **浏览器和版本信息**

这样我可以更准确地定位和解决问题。
