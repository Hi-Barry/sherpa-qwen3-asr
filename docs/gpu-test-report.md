# GPU 安装与测试报告

**日期：** 2026-05-22
**机器：** Linux workstation (GTX 1050 Ti 4GB, 12-core CPU)

## 环境配置

| 组件 | 版本 | 安装方式 |
|------|------|---------|
| CUDA Toolkit | 11.8 | apt (NVIDIA 源) |
| cuDNN | 8.9.7 +cuda11.8 | apt (libcudnn8) |
| sherpa-onnx | 1.13.2+cuda | pip (k2-fsa wheel) |
| Python | 3.11.15 | venv-gpu |

**额外 Python wheels（解决 libcublasLt.so.11 缺失）：**
- `nvidia-cublas-cu11==11.11.3.6`
- `nvidia-cudnn-cu11==9.10.2.21`

## 测试结果

### 全量测试：✅ **39/39 全部通过**

| 测试文件 | 用例数 | 全部通过 |
|---------|-------|---------|
| tests/test_models.py | 13 | ✅ |
| tests/test_engine.py | 14 | ✅ |
| tests/test_api.py | 12 | ✅ |

### 性能对比：GPU vs CPU

测试音频：20.76s 中文绕口令（raokouling.wav）

| 指标 | CPU (12-core) | GPU (GTX 1050 Ti 4GB) |
|------|--------------|----------------------|
| 模型加载时间 | 14.0s | **6.6s** (−53%) |
| ASR 推理时间 | 10.88s | 10.49s (−3.6%) |
| Real-Time Factor | 0.524 | 0.505 |
| 测试总耗时 | ~90s | ~90s |

### 识别效果

```
> 广西壮族自治区爱吃红鲤鱼、绿鲤鱼与驴的出租车司机，
  拉着苗族土家族自治州爱喝自制的刘奶奶榴莲牛奶的古痴，
  收东症患者，遇见别个喇叭的哑巴，
  打败咬死山前四十四棵死色柿子树...
```
（20.76s 中文绕口令 — 完美识别，无插入/删除错误）

## 关键发现

### 1. GPU 收益主要在加载，不在推理

Qwen3 ASR 0.6B 的核心是 **LLM Decoder（自回归解码）**——每次生成一个 token，GPU 无法像卷积/Transformer Encoder 那样并行加速整段音频。因此：
- **推理 RTF 无实质提升**（~0.50 vs ~0.52）
- **模型加载时间减半**（6.6s vs 14s），对服务器冷启动有益

### 2. API 差异

GPU 版 sherpa-onnx 1.13.2+cuda 是较旧的 build，与 CPU 版（1.13.2+post）存在以下差异：

| 特性 | CPU (1.13.2+post) | GPU (1.13.2+cuda) |
|------|-------------------|-------------------|
| `from_qwen3_asr()` | 支持 `(model_dir)` 简写 | 仅支持独立文件路径 ✅ 当前代码兼容 |
| `stream.input_finished()` | ✅ 存在 | ❌ 不存在，直接 decode_stream |
| `stream.set_option()` | ✅ 存在 | ✅ 存在 |

### 3. cuBLAS 依赖陷阱（已解决）

sherpa-onnx GPU wheel 动态链接 `libcublasLt.so.11`、`libcudnn.so.8`（注意是裸 `.so.8` 无平台标签），但 CUDA Toolkit 11.8 runfile 不包含 cuBLAS 共享库。需要：
```bash
pip install nvidia-cublas-cu11 nvidia-cudnn-cu11
```
这两个 wheel 将 `.so` 放入 site-packages，sherpa-onnx 会自动搜索到。

## 启动命令

```bash
# GPU 模式（推荐）
source venv-gpu/bin/activate
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
python -m src.api

# CPU 模式
source venv/bin/activate
python -m src.api
```
