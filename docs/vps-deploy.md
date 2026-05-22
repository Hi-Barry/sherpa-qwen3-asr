# VPS 部署指南 — CPU 版 + Nginx 反向代理

**目标：** 在 VPS 上部署纯 CPU 版本的 sherpa-qwen3-asr，通过 Nginx 反向代理对外提供服务，支持 HTTPS（Let's Encrypt）和开机自启。

---

## 环境要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **CPU** | 2 核 | 4 核+ |
| **内存** | 2 GB | 4 GB+ |
| **磁盘** | 3 GB（模型 ~1.5 GB） | 10 GB |
| **OS** | Ubuntu 22.04 / Debian 12 | Ubuntu 22.04 |
| **Python** | 3.10+ | 3.11 |
| **Nginx** | 1.18+ | 最新版 |

> ⚠️ **CPU 推理速度预期：** Qwen3-ASR 0.6B int8 在 4 核 CPU 上 RTF ≈ 0.5–1.0，即 10 秒音频约需 5–10 秒处理。2 核 VPS 上会更慢。

---

## 第一步：安装系统依赖

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装 Python + 基础工具
sudo apt install -y python3 python3-venv python3-pip \
  curl wget git nginx certbot python3-certbot-nginx

# 验证
python3 --version   # 应 >= 3.10
nginx -v            # 应 >= 1.18
```

---

## 第二步：创建用户并拉取代码

```bash
# 创建专用用户（非 root 运行服务，更安全）
sudo useradd -m -s /bin/bash sherpa
sudo usermod -aG sudo sherpa  # 可选，方便 sudo 操作

# 切换到 sherpa 用户
sudo -u sherpa -H bash
cd ~

# 拉取代码
git clone https://github.com/Hi-Barry/sherpa-qwen3-asr.git
cd sherpa-qwen3-asr
```

---

## 第三步：安装 Python 依赖

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 升级 pip
pip install --upgrade pip

# 安装 CPU 版依赖（直接走 pip，无需代理）
pip install -r requirements.txt

# 验证 sherpa-onnx 安装
python -c "import sherpa_onnx; print('sherpa-onnx:', sherpa_onnx.__version__)"
```

> ✅ 国内 VPS 如果连接 GitHub/ PyPI 慢，可以用镜像：
> ```bash
> pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -r requirements.txt
> ```

---

## 第四步：下载模型

```bash
# 模型约 1.5 GB，下载可能需要几分钟
chmod +x scripts/download_models.sh
./scripts/download_models.sh

# 验证
ls -la models/qwen3-asr/
# 应看到: conv_frontend.onnx, encoder.int8.onnx, decoder.int8.onnx, tokenizer/
ls -la models/vad/
# 应看到: silero_vad.onnx
```

> ⚠️ **模型下载慢？** 如果 GitHub Releases 下载慢，可以先用代理下载模型文件，然后 scp 到 VPS：
> ```bash
> # 在本机下载后 scp 到 VPS
> scp -r models/ user@your-vps:~/sherpa-qwen3-asr/models/
> ```

---

## 第五步：修改配置

编辑 `config/config.yaml`：

```yaml
service:
  host: "127.0.0.1"    # ⚠️ 改为 127.0.0.1！只接受 Nginx 转发，不对外暴露
  port: 8000

models:
  qwen3_asr_dir: "models/qwen3-asr"
  vad_dir: "models/vad"
  vad_model_file: "silero_vad.onnx"

asr:
  provider: "cpu"        # ✅ CPU 模式
  num_threads: 4         # ⚠️ 改成 VPS 的 CPU 核数（性能关键！）
  feature_dim: 128
  max_total_len: 512
  max_new_tokens: 128
  temperature: 0.000001
  top_p: 0.8
  seed: 42
  hotwords: ""

vad:
  enabled: true
  threshold: 0.5
  min_speech_duration: 0.25
  min_silence_duration: 0.25
  max_speech_duration: 30

processing:
  max_file_size: 52428800      # 50 MB
  max_audio_duration: 3600     # 1 hour
  temp_dir: "/tmp/sherpa-qwen3-asr"

logging:
  level: "INFO"
```

> ⚠️ **关键：** `num_threads` 设置为 VPS 的 CPU 核数（`nproc` 命令查看）。ONNX Runtime 用这个值控制线程池。

---

## 第六步：配置 Systemd 服务

创建 `/etc/systemd/system/sherpa-qwen3-asr.service`：

```ini
[Unit]
Description=sherpa-qwen3-asr — Qwen3-ASR Speech Recognition API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=sherpa
Group=sherpa
WorkingDirectory=/home/sherpa/sherpa-qwen3-asr
Environment=PATH=/home/sherpa/sherpa-qwen3-asr/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/home/sherpa/sherpa-qwen3-asr/venv/bin/python -m src.api
Restart=always
RestartSec=10

# 日志管理
StandardOutput=journal
StandardError=journal

# 安全加固（可选）
NoNewPrivileges=true
ProtectSystem=full
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

启用并启动：

```bash
# 复制服务文件
sudo cp docs/sherpa-qwen3-asr.service /etc/systemd/system/

# 重载 systemd
sudo systemctl daemon-reload

# 启动服务
sudo systemctl enable sherpa-qwen3-asr
sudo systemctl start sherpa-qwen3-asr

# 检查状态
sudo systemctl status sherpa-qwen3-asr
# 应显示 active (running)

# 本地验证
curl http://127.0.0.1:8000/api/v1/health
# {"status":"ok","models":{"asr":true},"provider":"cpu"}
```

> 查看日志：`sudo journalctl -u sherpa-qwen3-asr -f`

---

## 第七步：配置 Nginx 反向代理

创建 `/etc/nginx/sites-available/sherpa-qwen3-asr`：

```nginx
# ==========================================================================
# sherpa-qwen3-asr — Nginx 反向代理配置
# ==========================================================================

upstream sherpa_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name your-domain.com;  # ⚠️ 改成你的域名

    # 文件上传大小限制（对应 config.yaml 中的 max_file_size）
    client_max_body_size 60M;

    # ---- 健康检查（无需认证） ----
    location /api/v1/health {
        proxy_pass http://sherpa_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # ---- ASR 识别接口 ----
    location /api/v1/recognize {
        proxy_pass http://sherpa_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 长音频请求可能需要更久
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;

        # 禁用缓冲，流式响应
        proxy_buffering off;
    }

    # ---- OpenAI 兼容接口 ----
    location /v1/ {
        proxy_pass http://sherpa_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_buffering off;
    }

    # ---- Swagger UI 文档 ----
    location /docs {
        proxy_pass http://sherpa_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /openapi.json {
        proxy_pass http://sherpa_backend;
        proxy_set_header Host $host;
    }

    # ---- 拒绝其他路径 ----
    location / {
        return 404;
    }
}
```

启用配置：

```bash
# 保存配置
sudo cp docs/sherpa-qwen3-asr.nginx.conf /etc/nginx/sites-available/sherpa-qwen3-asr

# 编辑配置文件中的 server_name 为你的域名
sudo sed -i 's/your-domain.com/你的实际域名/g' /etc/nginx/sites-available/sherpa-qwen3-asr

# 启用站点
sudo ln -sf /etc/nginx/sites-available/sherpa-qwen3-asr /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default  # 删除默认站点

# 测试配置
sudo nginx -t

# 重载 Nginx
sudo systemctl reload nginx

# 验证外部访问
curl http://your-domain.com/api/v1/health
```

---

## 第八步：配置 HTTPS（Let's Encrypt）

> ⚠️ 需要有一个**公网 IP** 和 **已解析的域名**（DNS A 记录指向你的 VPS）。

```bash
# 申请证书（certbot 会自动修改 Nginx 配置）
sudo certbot --nginx -d your-domain.com

# 按提示输入邮箱、同意条款、选择是否重定向 HTTP→HTTPS（推荐选 yes）

# 验证
curl https://your-domain.com/api/v1/health

# 测试自动续期
sudo certbot renew --dry-run
```

HTTP→HTTPS 重定向配置（如果 certbot 没有自动做，手动加在 `server` block 中）：

```nginx
# 在 /etc/nginx/sites-available/sherpa-qwen3-asr 中添加
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$host$request_uri;
}
```

---

## 第九步：验证完整链路

```bash
# 从 VPS 本地测试
curl http://127.0.0.1:8000/api/v1/health

# 从外部测试（HTTP）
curl http://your-domain.com/api/v1/health

# 从外部测试（HTTPS）
curl https://your-domain.com/api/v1/health

# 测试识别（需一个音频文件）
curl https://your-domain.com/v1/audio/transcriptions \
  -F "file=@test.wav" \
  -F "model=Qwen/Qwen3-ASR-0.6B"

# OpenAI 兼容格式
curl https://your-domain.com/v1/models
```

---

## 可选优化

### 1. 调高 num_threads

```bash
# 查看 CPU 核数
nproc

# 设置 num_threads = CPU 核数（不要超过，否则上下文切换反而变慢）
# 编辑 config/config.yaml 中的 num_threads
```

### 2. 关闭 VAD 提升短音频速度

短音频（<30s）可以关闭 VAD 跳过分割开销，在请求中加参数：
```
language=Chinese&vad=false
```

或者在 `config.yaml` 中全局关闭：`vad.enabled: false`

### 3. 内存优化

如果 VPS 只有 2GB 内存，考虑：

- 关闭不必要的服务（MySQL、Redis 等）
- 增加 swap：
  ```bash
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  ```

### 4. 热词（Hotwords）

针对特定领域（如医疗、法律、技术术语），可以配置热词：
```yaml
asr:
  hotwords: "深度学习,神经网络,Transformer,GPT"
```

---

## 故障排除

### 服务启动失败

```bash
# 查看日志
sudo journalctl -u sherpa-qwen3-asr -n 50 --no-pager

# 尝试手动启动（看在 venv 中是否能运行）
sudo -u sherpa -H bash
cd ~/sherpa-qwen3-asr
source venv/bin/activate
python -m src.api
```

### 502 Bad Gateway（Nginx 连不上后端）

```bash
# 检查后端是否运行
sudo systemctl status sherpa-qwen3-asr
curl http://127.0.0.1:8000/api/v1/health

# 检查端口监听
sudo ss -tlnp | grep 8000

# 检查 Nginx 配置
sudo nginx -t
```

### 413 Request Entity Too Large

上传文件超过 Nginx 限制，检查 `client_max_body_size`：

```bash
# 查看当前配置
grep client_max_body_size /etc/nginx/sites-available/sherpa-qwen3-asr

# 修改后重载
sudo systemctl reload nginx
```

### 504 Gateway Timeout

长音频处理超过 Nginx 超时，增大 `proxy_read_timeout`：

```nginx
proxy_read_timeout 900s;  # 改为 15 分钟
```

### 请求返回 404

确认 Nginx `location` 规则是否匹配。测试不同路径：

```bash
curl https://your-domain.com/api/v1/health          # 应返回 JSON
curl https://your-domain.com/v1/models               # 应返回 JSON
curl https://your-domain.com/                        # 应返回 404（安全）
```

---

## 维护命令速查

```bash
# 服务管理
sudo systemctl start|stop|restart|status sherpa-qwen3-asr
sudo journalctl -u sherpa-qwen3-asr -f        # 实时日志
sudo journalctl -u sherpa-qwen3-asr -n 100    # 最近 100 行

# Nginx 管理
sudo systemctl reload|restart|status nginx
sudo nginx -t                                 # 测试配置

# HTTPS 证书
sudo certbot renew                            # 手动续期
sudo certbot certificates                     # 查看证书状态

# 更新代码
cd ~/sherpa-qwen3-asr
git pull
source venv/bin/activate
pip install -r requirements.txt               # 如果有新依赖
sudo systemctl restart sherpa-qwen3-asr
```

---

## 参考

- [sherpa-onnx 官方文档](https://k2-fsa.github.io/sherpa/onnx/)
- [Nginx 反向代理文档](https://nginx.org/en/docs/http/ngx_http_proxy_module.html)
- [Certbot 文档](https://certbot.eff.org/docs/)
- [GitHub 项目](https://github.com/Hi-Barry/sherpa-qwen3-asr)
