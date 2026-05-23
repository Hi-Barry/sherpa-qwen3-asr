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

## v0.1.1 (2026-05-22) — GPU 版安装 + 测试

### 做了什么
安装 CUDA 11.8 + cuDNN 8.9 + GPU 版 sherpa-onnx 1.13.2+cuda，并在 NVIDIA GTX 1050 Ti 4GB 上完成全量测试。

### GPU 环境搭建步骤
1. 清理旧 CUDA 残留 → 用 apt 安装 CUDA Toolkit 11.8（需 NVIDIA apt 源）
2. `apt install libcudnn8=8.9.7.*+cuda11.8` 安装 cuDNN 8
3. 创建独立 `venv-gpu` 虚拟环境
4. `pip install sherpa-onnx==1.13.2+cuda -f https://k2-fsa.github.io/sherpa/onnx/cuda.html`（需代理）
5. 安装 `nvidia-cudnn-cu11` + `nvidia-cublas-cu11` 两个 Python wheel（解决找不到 libcublasLt.so.11 的问题）

### API 差异（重要）
GPU 版 sherpa-onnx 1.13.2+cuda 与 CPU 版 1.13.2 存在 API 差异：
- `from_qwen3_asr()` 签名相同（独立文件路径），✅ 兼容
- OfflineStream **无** `input_finished()` 方法 — 直接 `accept_waveform()` 后 `decode_stream()` 即可
- OfflineStream 有 `set_option("language", ...)` — ✅ 兼容

### 性能测试结果（GTX 1050 Ti vs CPU）
| 项目 | CPU (12-core) | GPU (1050 Ti) |
|------|--------------|---------------|
| 模型加载时间 | 14.0s | **6.6s** |
| ASR 推理 (20.8s 音频) | 10.88s | 10.49s |
| RTF | 0.524 | 0.505 |
| 测试全量 | 39/39 ✅ | 39/39 ✅ |

**结论：** Qwen3 0.6B 在小显存（4GB）的低端 GPU 上推理速度无实质提升（RTF 都在 ~0.5 左右，因为 LLM 解码器是自回归的，GPU 编解码内核未充分利用）。但模型加载速度翻倍（6.6s vs 14s），对服务器冷启动延迟有明显改善。

### 踩坑（GPU 版）
1. **CUDA 版本冲突** — NVIDIA apt 源默认到 CUDA 12.x，需用 `cuda-11-8` 指定版本且要添加 `cuda11.8` pin 优先级
2. **cuDNN 版本不匹配** — apt 安装的 `libcudnn8` 默认也装 12.x 版，需 `apt-cache policy libcudnn8` 找到 cuDNN 8 的 CUDA 11.8 专用包
3. **libcublasLt.so.11 找不到** — sherpa-onnx 1.13.2+cuda 内部调 `libcublasLt.so.11`，但 CUDA 11.8 toolkit 和 cuDNN apt 包不包含 cuBLAS。解决方案：`pip install nvidia-cublas-cu11 nvidia-cudnn-cu11`，这两个 wheel 自带 libcublasLt.so.11

---

## v0.1.1 (2026-05-23)

### 做了什么

修复 M4A/AAC/MP3/OPUS/WEBM 等格式的解码失败问题。

### 根因

`engine.py` 的 `load_audio()` 使用 `soundfile`（libsndfile）解码音频文件。但 libsndfile 不支持 M4A、AAC、MP3、OPUS、WEBM 等常见压缩音频格式。当 LiveSpeaker Android App 上传 M4A 录音文件时，服务端返回 `Failed to decode audio file` 错误。

### 修复

添加 `_ffmpeg_decode()` 静态方法 — 当文件扩展名不在 soundfile 原生支持列表（wav/flac/ogg/aiff/w64/caf）时，自动调用 ffmpeg 解码为 16kHz mono WAV 临时文件，再交由 soundfile 读取。

### 改动

| 文件 | 改动 |
|------|------|
| `src/engine.py` | +58 行 — 新增 `_SF_FORMATS`、`_ffmpeg_decode()`、修改 `load_audio()` 增加 ffmpeg fallback |

### 依赖

- 需要 `ffmpeg` 命令（Ubuntu 下系统自带，无需额外 pip 包）
