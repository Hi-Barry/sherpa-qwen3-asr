# sherpa-qwen3-asr — Qwen3-ASR Speech Recognition API

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Pure CPU / CUDA GPU 语音识别 API — 基于 Qwen3-ASR 0.6B int8 + sherpa-onnx**

支持 **52 种语言**，纯 ONNX Runtime 推理，无需 PyTorch，无需 GPU（可选 CUDA 加速）。

---

## 特性

| 功能 | 支持 |
|------|------|
| **多语言 ASR** | 30 种语言 + 22 种中文方言（自动检测） |
| **纯 CPU 推理** | ONNX int8 量化，无需 GPU |
| **CUDA GPU 加速** | 可选 NVIDIA GPU 加速（CUDA 11.8） |
| **长音频智能切分** | Silero VAD 分段 + 强制 30s 子切片（防止模型溢出） |
| **热词 (Hotwords)** | 支持热词偏置，提升特定词汇识别率 |
| **语言强制** | 可指定语言（如 `"Korean"`, `"Chinese"`） |
| **OpenAI 兼容** | 兼容 OpenAI Whisper API 调用方式 |
| **全格式解码** | 内置 ffmpeg 回退 — 支持 M4A/AAC/MP3/OPUS/WEBM 等

---

## 环境要求

| 项目 | CPU 模式 | GPU 模式（可选） |
|------|---------|----------------|
| Python | 3.10+ | 3.10+ |
| 内存 | ~2 GB | ~1.5 GB + 1.5 GB VRAM |
| CPU | 2 核+ | 2 核+ |
| 磁盘 | 2 GB（模型 ~1.5 GB） | 2 GB |
| NVIDIA GPU | — | GTX 1050 Ti 4GB+ |
| CUDA Toolkit | — | **11.8** + cuDNN 8 |
| **ffmpeg** | ✅ 推荐 | ✅ 推荐 |

---

## 快速开始

### 1. 获取代码

```bash
git clone https://github.com/Hi-Barry/sherpa-qwen3-asr.git
cd sherpa-qwen3-asr
```

### 2. 安装依赖

```bash
python -m venv venv
source venv/bin/activate

# CPU 模式（推荐先试这个）
pip install -r requirements.txt

# 或 GPU 模式（需要 CUDA 11.8）
# pip install -r requirements-gpu.txt
```

### 3. 下载模型

```bash
chmod +x scripts/download_models.sh
./scripts/download_models.sh
```

下载内容：
- **Qwen3-ASR 0.6B int8** (~1.5 GB) — 主 ASR 模型
- **Silero VAD v5** (~2.2 MB) — 语音活动检测

### 4. 启动服务

```bash
# 默认 CPU 模式
python -m src.api

# 或指定 GPU（需 config.yaml 中设 provider: "cuda" + GPU 版 sherpa-onnx）
# python -m src.api
```

### 5. 验证

```bash
curl http://localhost:8000/api/v1/health
# {"status":"ok","models":{"asr":true},"provider":"cpu"}
```

---

## API 文档

### `POST /api/v1/recognize`

上传音频文件，返回识别结果。

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | file | ✅ | 音频文件 (wav/mp3/flac/ogg/m4a) |
| `language` | string | ❌ | 强制语言："Chinese", "English", "Korean" 等，空值=自动 |

**返回：**

```json
{
  "success": true,
  "result": {
    "language": "en",
    "duration": 12.34,
    "segments": [
      {"start": 0.0, "end": 3.0, "text": "Hello, this is a test."},
      {"start": 3.5, "end": 7.0, "text": "How are you today?"}
    ],
    "text": "Hello, this is a test. How are you today?",
    "stats": {
      "asr_time": 2.1,
      "total_time": 2.5
    }
  }
}
```

### `GET /api/v1/health`

服务健康检查。

### OpenAI 兼容 API

标准 OpenAI Whisper API 兼容端点：

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=Qwen/Qwen3-ASR-0.6B"

curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=Qwen/Qwen3-ASR-0.6B" \
  -F "response_format=verbose_json"

curl http://localhost:8000/v1/models
```

### API 文档

访问 `http://localhost:8000/docs` 查看 Swagger UI。

---

## 远程调用

项目自带 `scripts/qwen3_client.py`，纯 Python 标准库实现，零依赖，可直接复制到远程机器运行。

> ⚠️ **跨平台注意：** Linux/macOS 用 `python3`，Windows 用 `python`（虚拟环境中 `python3` 不可用，只识别 `python`）。

```bash
# 复制到远程机器（只需这一个文件）
scp scripts/qwen3_client.py user@remote-pc:~/

# 在远程机器上使用
python3 qwen3_client.py health --server 10.88.88.5:8000

python3 qwen3_client.py transcribe audio.wav --server 10.88.88.5:8000

python3 qwen3_client.py transcribe audio.wav \
  --server 10.88.88.5:8000 --language Chinese

# OpenAI 兼容接口
python3 qwen3_client.py openai audio.wav \
  --server 10.88.88.5:8000 --format verbose_json

python3 qwen3_client.py openai audio.wav \
  --server 10.88.88.5:8000 --format text
```

也可在代码中作为库调用：

```python
from qwen3_client import Qwen3Client

client = Qwen3Client(server="10.88.88.5:8000")

# 健康检查
print(client.health())

# 识别音频
result = client.transcribe("audio.wav", language="Chinese")
print(result["result"]["text"])

# OpenAI 兼容
text = client.openai_transcribe("audio.wav", response_format="text")
print(text["text"])
```

---

## 配置

编辑 `config/config.yaml`：

```yaml
asr:
  provider: "cpu"         # "cpu" 或 "cuda"
  num_threads: 2
  feature_dim: 128        # Qwen3 专用，非 80！
  max_total_len: 2048     # 512→2048（修复长音频截断，配合 30s 切片）
  max_new_tokens: 256     # 128→256（支持更长句子）
  temperature: 0.000001
  hotwords: ""            # 逗号分隔的热词

vad:
  enabled: true           # 长音频自动 VAD 分割（每请求新建实例）
  threshold: 0.3          # 0.5→0.3（嘈杂环境更灵敏）
  min_silence_duration: 1.0  # 0.25→1.0（按句子边界切分）
  min_speech_duration: 0.5   # 0.25→0.5（过滤短噪声）

processing:
  max_chunk_duration: 30  # ★ 新增：强制子切片最长 30s
  chunk_overlap: 0.0      # 不重叠（简单可靠）
  preprocess:
    normalize: true       # ★ 新增：音量归一化
    highpass_cutoff: 80   # ★ 新增：80Hz 高通滤波（去低频噪音）
```

---

## Docker 部署

```bash
# 1. 先下载模型到本地
bash scripts/download_models.sh

# 2. Docker 构建并运行
docker build -t sherpa-qwen3-asr .
docker run -d --name sherpa-qwen3-asr \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  sherpa-qwen3-asr
```

---

## 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 只跑单元测试（不需要模型）
python -m pytest tests/test_models.py -v
python -m pytest tests/test_engine.py -v -k "not TestEngineInit and not TestAsrInference"

# 跑完整的集成测试（需要模型已下载）
python -m pytest tests/ -v
```

---

## 许可证

MIT
