# CUDA 11.8 + cuDNN 8 安装指南（apt 版）

> 📍 **本机已验证环境：** GTX 1050 Ti 4GB | 驱动 535.309.01
>
> 这是实际安装成功的步骤记录，包含所有踩坑和解决方案。

---

## 环境检查

```bash
nvidia-smi
```

确认输出：
- **Driver Version** ≥ 520.61.05（CUDA 11.8 的最低要求）
- 当前驱动支持 CUDA 12.x runtime 也没关系，CUDA Toolkit 11.8 可以兼容安装

> ⚠️ **不要用 runfile 安装！** 实际测试发现 runfile 安装到 `~/cuda/` 后，sherpa-onnx GPU wheel 找不到 `libcublasLt.so.11`（cuBLAS 不在 CUDA Toolkit 中）。**推荐用 apt 方式安装。**

---

## 第一步：添加 NVIDIA apt 源

```bash
# 添加 NVIDIA CUDA apt 仓库
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

---

## 第二步：安装 CUDA Toolkit 11.8

```bash
# 安装 CUDA 11.8 Toolkit（不装驱动，用系统已有驱动）
sudo apt install cuda-toolkit-11-8
```

> ⚠️ **版本锁定：** NVIDIA apt 源还有其他 CUDA 版本（如 12.x），务必指定 `cuda-toolkit-11-8` 而不是 `cuda-toolkit`，否则会装最新版。
>
> 安装位置：`/usr/local/cuda-11.8/`

---

## 第三步：安装 cuDNN 8.9 for CUDA 11.8

```bash
# 查看可用的 cuDNN 版本
apt-cache policy libcudnn8

# 安装 cuDNN 8.9 对应 CUDA 11.8 的版本
sudo apt install libcudnn8=8.9.7.*+cuda11.8
```

> ⚠️ **版本坑：** 不加版本号会装 cuDNN for CUDA 12.x（依赖 CUDA 12 runtime，与 CUDA 11.8 不兼容）。
>
> 安装位置：`/usr/lib/x86_64-linux-gnu/libcudnn.so.8`

---

## 第四步：设置环境变量

追加到 `~/.bashrc`：

```bash
# CUDA 11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8
```

生效并验证：

```bash
source ~/.bashrc
nvcc --version
# 应输出: Cuda compilation tools, release 11.8, V11.8.89
```

---

## 第五步：安装 GPU 版 sherpa-onnx

```bash
cd ~/Projects/sherpa-qwen3-asr

# 创建 GPU 专用虚拟环境
deactivate 2>/dev/null
python3 -m venv venv-gpu
source venv-gpu/bin/activate

# 升级 pip
pip install --upgrade pip

# ⚠️ 需要代理（如国内网络）：
# export http_proxy=socks5://10.88.88.3:10808
# export https_proxy=socks5://10.88.88.3:10808

# 安装 GPU 版 sherpa-onnx 1.13.2+cuda
pip install sherpa-onnx==1.13.2+cuda \
  -f https://k2-fsa.github.io/sherpa/onnx/cuda.html

# 安装其他依赖
pip install fastapi uvicorn[standard] python-multipart \
  soundfile librosa pyyaml

# 安装 cuBLAS + cuDNN Python wheels（关键！）
pip install nvidia-cublas-cu11 nvidia-cudnn-cu11
```

> ⚠️ **巨坑！libcublasLt.so.11 找不到**
>
> sherpa-onnx GPU wheel 链接了 `libcublasLt.so.11`，但 CUDA Toolkit 11.8 和 cuDNN apt 包**都不包含 cuBLAS 共享库**。必须通过 pip 安装：
> - `nvidia-cublas-cu11` — 提供 `libcublasLt.so.11`
> - `nvidia-cudnn-cu11` — 提供 `libcudnn.so.8`
>
> 这两个 wheel 会把 `.so` 文件放入 site-packages，sherpa-onnx 运行时自动搜索到。

---

## 第六步：验证 GPU 可用

```bash
source venv-gpu/bin/activate
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

python -c "
import sherpa_onnx

model_dir = '/home/barry/Projects/sherpa-qwen3-asr/models/qwen3-asr'
recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
    conv_frontend=f'{model_dir}/conv_frontend.onnx',
    encoder=f'{model_dir}/encoder.int8.onnx',
    decoder=f'{model_dir}/decoder.int8.onnx',
    tokenizer=f'{model_dir}/tokenizer',
    provider='cuda',
    num_threads=4,
)
print('✅ GPU recognizer loaded OK')

import soundfile as sf
audio, sr = sf.read(f'{model_dir}/test_wavs/raokouling.wav')
stream = recognizer.create_stream()
stream.accept_waveform(sr, audio)
recognizer.decode_stream(stream)
result = stream.result.text
print(f'✅ ASR: {result[:80]}...')
print(f'sherpa-onnx version: {sherpa_onnx.__version__}')
"
```

成功输出表示 GPU 模式已生效。

---

## 第七步：启动 GPU 版服务

```bash
source ~/Projects/sherpa-qwen3-asr/venv-gpu/bin/activate
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
python -m src.api
```

配置文件中 `provider: "cuda"` 已设为默认。

> ⚠️ 每次新开终端都要重新设置环境变量（PATH / LD_LIBRARY_PATH），建议加到 `~/.bashrc`。

---

## 性能说明

| 指标 | CPU (12-core) | GPU (GTX 1050 Ti 4GB) |
|------|--------------|----------------------|
| 模型加载 | 14.0s | **6.6s** (−53%) |
| ASR 推理 RTF | 0.524 | 0.505 |
| 测试全量 (39 tests) | ~90s | ~90s |

> Qwen3 ASR 0.6B 的核心是 **LLM Decoder（自回归解码）**——每次生成一个 token，GPU 无法像 Encoder 那样并行加速整段音频。因此推理速度提升有限，但**模型加载时间减半**。

---

## 常见问题

### Q: 出现 `libcublasLt.so.11: cannot open shared object file`
**原因：** cuBLAS 共享库缺失。
**解决：** 在 venv-gpu 中运行：
```bash
pip install nvidia-cublas-cu11 nvidia-cudnn-cu11
```

### Q: 出现 `libcudnn.so.8: cannot open shared object file`
**原因：** cuDNN 共享库缺失。
**解决：** 同上，或确认 `libcudnn8=8.9.7.*+cuda11.8` 已正确安装。

### Q: 出现 `CUDA driver is insufficient`
**原因：** 环境变量未正确设置。
**解决：**
```bash
export CUDA_VISIBLE_DEVICES=0
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

### Q: 如何切回 CPU 模式？
```bash
# 1. 切换到 CPU venv
source venv/bin/activate
# 2. 修改 config.yaml 中 provider: "cpu"
```

### Q: GPU API 和 CPU API 有什么区别？
GPU 版 sherpa-onnx 1.13.2+cuda 是较早的 build，存在以下差异：
- `from_qwen3_asr()` 只接受独立文件路径，不接受目录简写（当前代码兼容）
- OfflineStream **没有** `input_finished()` 方法，直接 `accept_waveform()` 后 `decode_stream()`（当前代码兼容）
- OfflineStream 有 `set_option("language", ...)`（兼容）

---

## 参考链接

- [sherpa-onnx 官方 CUDA 安装文档](https://k2-fsa.github.io/sherpa/onnx/python/install.html#method-2-from-pre-compiled-wheels-cpu-cuda-11-8)
- [CUDA Toolkit 11.8 下载](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- [cuDNN 下载](https://developer.nvidia.com/cudnn)
