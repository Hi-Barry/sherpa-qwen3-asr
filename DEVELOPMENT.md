# DEVELOPMENT.md — 开发日志

## v0.1.0 (2026-05-22)

### 做了什么
创建 `sherpa-qwen3-asr` 新项目，基于 Qwen3-ASR 0.6B int8 的纯语音识别 API。

### 架构
```
sherpa-qwen3-asr/
├── src/
│   ├── api.py              # FastAPI 服务入口
│   ├── engine.py           # Qwen3 ASR 引擎 + VAD
│   ├── models.py           # Pydantic 数据模型
│   └── openai_compat.py    # OpenAI 兼容路由
├── config/config.yaml      # 配置文件
├── scripts/download_models.sh
├── tests/
│   ├── test_models.py      # 13 个测试
│   ├── test_engine.py      # 14 个测试
│   └── test_api.py         # 12 个测试
├── requirements.txt        # CPU 版
├── requirements-gpu.txt    # GPU 版 (CUDA 11.8)
└── Dockerfile
```

### 踩坑
1. **YAML 的 1e-6 被解析为字符串** — PyYAML (YAML 1.1) 把 `1e-6` 当字符串处理，导致 `from_qwen3_asr()` 报 `TypeError`。修复：配置文件用 `0.000001` 替代。
2. **TestClient lifespan** — `TestClient(app)` 不触发 `lifespan` handler，需要用 `with TestClient(app) as client:` 上下文管理器。
3. **Qwen3 feature_dim=128** — 与 SenseVoice (80) 不同，首次加载模型时需要注意。

### 学到什么
- `sherpa_onnx.OfflineRecognizer.from_qwen3_asr()` API 签名：需要 conv_frontend/encoder/decoder/tokenizer 四个路径
- Qwen3 ASR 使用 LLM Decoder (KV Cache)，输出为 `result.text`，无情感/事件标签
- 模型文件：conv_frontend.onnx (43MB) + encoder.int8.onnx (175MB) + decoder.int8.onnx (721MB) + tokenizer/ 目录
- Silero VAD 对长音频分割效果良好，每段 30s 以内适配 Qwen3 的 max_new_tokens
