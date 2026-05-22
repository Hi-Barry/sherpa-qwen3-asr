# CUDA 11.8 + cuDNN 8 安装指南（无 sudo 权限版）

## 环境信息

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce GTX 1050 Ti (4GB) |
| 驱动版本 | 535.309.01 |
| 当前 CUDA 驱动支持 | 12.2 |
| 目标 CUDA Toolkit | **11.8**（兼容当前驱动） |
| 目标 cuDNN | **8.9.x for CUDA 11.x** |

---

## 第一步：下载 CUDA Toolkit 11.8

```bash
cd ~/Downloads

# 下载 CUDA 11.8 runfile（约 3.2GB）
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# 如 wget 慢，可用 curl
# curl -L -o cuda_11.8.0_520.61.05_linux.run \
#   https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
```

---

## 第二步：安装 CUDA Toolkit（无 sudo 模式）

```bash
chmod +x cuda_11.8.0_520.61.05_linux.run

# 只解压（不安装驱动），安装到 ~/cuda/
./cuda_11.8.0_520.61.05_linux.run --toolkit --silent \
  --toolkitpath=$HOME/cuda/ \
  --no-opengl-libs --no-man-page
```

> ⚠️ 注意：**不要勾选 Driver**（你的驱动 535 已经比 520 新）
> 如果出现交互界面，取消勾选 "Driver"，只保留 "CUDA Toolkit 11.8"

---

## 第三步：设置环境变量

将以下内容追加到 `~/.bashrc`：

```bash
# CUDA 11.8 （自定义安装路径）
export PATH=$HOME/cuda/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$HOME/cuda
```

然后生效：

```bash
source ~/.bashrc

# 验证
nvcc --version
# 应输出: Cuda compilation tools, release 11.8, V11.8.89
```

---

## 第四步：下载并安装 cuDNN 8.9 for CUDA 11.x

cuDNN 需要从 NVIDIA Developer 网站下载（需免费注册账号）。

**方法 A — 手动下载（推荐）：**

1. 访问 https://developer.nvidia.com/cudnn
2. 注册/登录 NVIDIA Developer 账号
3. 下载 **cuDNN for CUDA 11.x** → **Local Installer for Linux x86_64 (Tar)**
   - 文件名类似 `cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz`
4. 解压并复制到 CUDA 目录：

```bash
tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz

# 复制到自定义 CUDA 目录
cp cudnn-linux-x86_64-8.9.7.29_cuda11-archive/include/cudnn*.h $HOME/cuda/include/
cp -P cudnn-linux-x86_64-8.9.7.29_cuda11-archive/lib/libcudnn* $HOME/cuda/lib64/

# 设置权限
chmod a+r $HOME/cuda/include/cudnn*.h $HOME/cuda/lib64/libcudnn*

# 验证
cat $HOME/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

**方法 B — 用 pip 安装（无需注册，推荐的替代方案）：**

```bash
# Python 版的 cuDNN 封装 — sherpa-onnx 可能需要
pip install nvidia-cudnn-cu11
```

> 注：sherpa-onnx 的 CUDA 版 wheel 已内嵌 ONNX Runtime 所需依赖，
> 部分功能可能不需要手动装 cuDNN。

---

## 第五步：安装 GPU 版 sherpa-onnx

```bash
cd ~/Projects/sherpa-qwen3-asr

# 确保 CUDA 环境变量在当前 shell 生效
source ~/.bashrc
which nvcc  # 应显示 ~/cuda/bin/nvcc

# 创建新 venv（或复用现有的）
deactivate 2>/dev/null
rm -rf venv-gpu
python3 -m venv venv-gpu
source venv-gpu/bin/activate

# 安装 GPU 版依赖
# ⚠️ 先注释掉 requirements.txt 中的 sherpa-onnx 行
# 然后安装 GPU 版
pip install --upgrade pip
pip install sherpa-onnx==1.13.2+cuda \
  -f https://k2-fsa.github.io/sherpa/onnx/cuda.html

# 安装其他依赖
pip install fastapi uvicorn[standard] python-multipart soundfile librosa pyyaml
```

---

## 第六步：验证 GPU 可用

```bash
python -c "
import sherpa_onnx
print('sherpa-onnx:', sherpa_onnx.__version__)
# 尝试创建 Qwen3 recognizer 使用 GPU
from pathlib import Path
import yaml

config = yaml.safe_load(open('config/config.yaml'))
config['asr']['provider'] = 'cuda'
config['models']['qwen3_asr_dir'] = str(Path('models/qwen3-asr').resolve())

from src.engine import SpeechEngine
engine = SpeechEngine(config)
print('GPU provider:', engine.provider)
print('Ready:', engine.is_ready)
"
```

如果成功输出 `GPU provider: cuda`，说明 GPU 模式已生效。

---

## 第七步：启动 GPU 版服务

```bash
# 方式 1：直接用 python 启动（默认使用 config.yaml 中的 provider）
source venv-gpu/bin/activate
python -m src.api

# 方式 2：用 uvicorn 启动
# uvicorn src.api:app --host 0.0.0.0 --port 8000
```

---

## 常见问题

### Q: 出现 `CUDA driver is insufficient` 错误
原因：驱动版本不匹配。当前驱动 535 支持 CUDA 12.2 runtime，
但 CUDA 11.8 toolkit 要求驱动 >= 520.61.05。你的驱动满足要求。
如果仍有问题，尝试：
```bash
export CUDA_VISIBLE_DEVICES=0
```

### Q: 出现 `libcudart.so.11.0: cannot open shared object file`
原因：LD_LIBRARY_PATH 没设置对。检查：
```bash
echo $LD_LIBRARY_PATH
ls $HOME/cuda/lib64/libcuda*  # 应该能看到文件
```

### Q: 4GB VRAM 够用吗？
Qwen3-ASR 0.6B int8 模型约 1.5GB，加上 ONNX Runtime 开销，
总共约 2-2.5GB VRAM。GTX 1050 Ti 的 4GB **完全够用**。

### Q: 还是报错，想切回 CPU 模式
把 `config/config.yaml` 中的 `provider` 改回 `"cpu"` 即可。
两个模式可以随时切换。

---

## 参考链接

- [sherpa-onnx 官方 CUDA 安装文档](https://k2-fsa.github.io/sherpa/onnx/python/install.html#method-2-from-pre-compiled-wheels-cpu-cuda-11-8)
- [CUDA Toolkit 11.8 下载](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- [cuDNN 下载](https://developer.nvidia.com/cudnn)
- [K2 CUDA 安装指南](https://k2-fsa.github.io/k2/installation/cuda-cudnn.html)
