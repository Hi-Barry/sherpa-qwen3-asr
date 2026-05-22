#!/usr/bin/env python3
"""
sherpa-qwen3-asr Remote Client
===============================

自包含的 Python 客户端，用于远程调用 sherpa-qwen3-asr API。
纯标准库，零额外依赖，在远程机器上直接运行。

Usage:
    # 识别音频
    python qwen3_client.py transcribe audio.wav --server 10.88.88.5:8000

    # 指定语言
    python qwen3_client.py transcribe audio.wav --server 10.88.88.5:8000 --language Chinese

    # OpenAI 兼容接口
    python qwen3_client.py openai audio.wav --server 10.88.88.5:8000

    # 详细格式
    python qwen3_client.py openai audio.wav --server 10.88.88.5:8000 --format verbose_json

    # 纯文本输出
    python qwen3_client.py openai audio.wav --server 10.88.88.5:8000 --format text

    # 健康检查
    python qwen3_client.py health --server 10.88.88.5:8000

    # 列出可用模型
    python qwen3_client.py models --server 10.88.88.5:8000
"""

import json
import sys
import os
import argparse
from urllib.request import Request, urlopen, OpenerDirector, ProxyHandler, HTTPHandler
from urllib.parse import urlencode
from urllib.error import URLError, HTTPError
from io import BytesIO
from mimetypes import guess_type
from pathlib import Path

# ---------------------------------------------------------------------------
# Multipart form-data builder (标准库实现，无需 requests)
# ---------------------------------------------------------------------------


class MultipartFormData:
    """Build multipart/form-data bodies without external dependencies."""

    BOUNDARY = b"----Qwen3ClientBoundary7MA4YWxkTrZu0gW"

    @classmethod
    def build(cls, fields: dict, files: dict) -> tuple[bytes, str]:
        """
        Build multipart body.

        Args:
            fields: {name: value} — text form fields
            files:  {name: filepath_or_bytesio} — file uploads

        Returns:
            (body_bytes, content_type_header_value)
        """
        parts = []

        for name, value in fields.items():
            part = cls._text_field(name, str(value))
            parts.append(part)

        for name, file_src in files.items():
            if isinstance(file_src, (str, Path)):
                file_path = str(file_src)
                filename = os.path.basename(file_path)
                with open(file_path, "rb") as f:
                    file_data = f.read()
            elif isinstance(file_src, bytes):
                filename = "audio.wav"
                file_data = file_src
            elif hasattr(file_src, "read"):
                filename = getattr(file_src, "name", "audio.wav")
                file_data = file_src.read()
            else:
                raise TypeError(f"Unsupported file source: {type(file_src)}")

            content_type, _ = guess_type(filename)
            if content_type is None:
                content_type = "application/octet-stream"

            part = cls._file_field(name, filename, file_data, content_type)
            parts.append(part)

        parts.append(cls._close_boundary())
        body = b"".join(parts)
        content_type = f"multipart/form-data; boundary={cls.BOUNDARY.decode()}"
        return body, content_type

    @classmethod
    def _text_field(cls, name: str, value: str) -> bytes:
        return (
            b"--" + cls.BOUNDARY + b"\r\n"
            b'Content-Disposition: form-data; name="' + name.encode() + b'"\r\n'
            b"\r\n"
            + value.encode("utf-8")
            + b"\r\n"
        )

    @classmethod
    def _file_field(cls, name: str, filename: str, data: bytes, content_type: str) -> bytes:
        return (
            b"--" + cls.BOUNDARY + b"\r\n"
            b'Content-Disposition: form-data; name="' + name.encode() + b'"; '
            b'filename="' + filename.encode() + b'"\r\n'
            b"Content-Type: " + content_type.encode() + b"\r\n"
            b"\r\n"
            + data
            + b"\r\n"
        )

    @classmethod
    def _close_boundary(cls) -> bytes:
        return b"--" + cls.BOUNDARY + b"--\r\n"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _request(
    url: str,
    method: str = "GET",
    body: bytes = None,
    headers: dict = None,
    timeout: int = 600,
) -> tuple[int, dict, bytes]:
    """Send HTTP request, return (status_code, response_json_dict, raw_body)."""
    req = Request(url, data=body, method=method)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)

    # Default headers
    req.add_header("Accept", "application/json")

    try:
        resp = urlopen(req, timeout=timeout)
        raw = resp.read()
        status = resp.status
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            data = {"text": raw.decode("utf-8", errors="replace")}
        return status, data, raw
    except HTTPError as e:
        raw = e.read()
        status = e.code
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            data = {"text": raw.decode("utf-8", errors="replace"), "error": str(e)}
        return status, data, raw
    except URLError as e:
        return 0, {"error": f"Connection failed: {e.reason}"}, b""


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------


class Qwen3Client:
    """
    Remote client for sherpa-qwen3-asr API.

    Args:
        server: Server address (host:port), e.g. "10.88.88.5:8000"
        timeout: HTTP request timeout in seconds
    """

    BASE_PATH = "/api/v1"
    OPENAI_PATH = "/v1"

    def __init__(self, server: str = "localhost:8000", timeout: int = 600):
        self.server = server.rstrip("/")
        if not self.server.startswith("http"):
            self.server = "http://" + self.server
        self.timeout = timeout

    # -- Health -----------------------------------------------------------------

    def health(self) -> dict:
        """GET /api/v1/health"""
        url = f"{self.server}{self.BASE_PATH}/health"
        _, data, _ = _request(url, timeout=self.timeout)
        return data

    # -- Models -----------------------------------------------------------------

    def list_models(self) -> dict:
        """GET /v1/models"""
        url = f"{self.server}{self.OPENAI_PATH}/models"
        _, data, _ = _request(url, timeout=self.timeout)
        return data

    # -- Native API -------------------------------------------------------------

    def transcribe(
        self,
        audio_path: str,
        language: str = "",
    ) -> dict:
        """
        POST /api/v1/recognize — 原生 API.

        Args:
            audio_path: Local audio file path
            language:   Force language (e.g. "Chinese", "English", "Korean").
                        Empty string = auto-detect.

        Returns:
            Parsed JSON response dict.
        """
        url = f"{self.server}{self.BASE_PATH}/recognize"

        fields = {"language": language}
        files = {"file": audio_path}

        body, content_type = MultipartFormData.build(fields, files)
        headers = {"Content-Type": content_type}

        status, data, _ = _request(url, method="POST", body=body, headers=headers, timeout=self.timeout)
        return data

    # -- OpenAI Compatible API --------------------------------------------------

    def openai_transcribe(
        self,
        audio_path: str,
        model: str = "Qwen/Qwen3-ASR-0.6B",
        language: str = "",
        response_format: str = "json",
    ) -> dict:
        """
        POST /v1/audio/transcriptions — OpenAI 兼容接口.

        Args:
            audio_path:      Local audio file path
            model:           Model name (default: Qwen/Qwen3-ASR-0.6B)
            language:        Force language (optional)
            response_format: "json", "text", or "verbose_json"

        Returns:
            Parsed response. For format="text", returns {"text": "<plain text>"}.
        """
        url = f"{self.server}{self.OPENAI_PATH}/audio/transcriptions"

        fields = {"model": model, "response_format": response_format}
        if language:
            fields["language"] = language

        files = {"file": audio_path}

        body, content_type = MultipartFormData.build(fields, files)
        headers = {"Content-Type": content_type}

        status, data, raw = _request(url, method="POST", body=body, headers=headers, timeout=self.timeout)

        # text format returns raw text
        if response_format == "text":
            return {"text": data.get("text", "") if isinstance(data, dict) else raw.decode("utf-8", errors="replace")}

        return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    # Parent parser with shared connection options
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--server", "-s",
        default="127.0.0.1:8000",
        help="服务器地址 (host:port)，默认 127.0.0.1:8000",
    )
    parent.add_argument(
        "--timeout", "-t",
        type=int,
        default=600,
        help="HTTP 超时秒数，默认 600",
    )

    parser = argparse.ArgumentParser(
        description="sherpa-qwen3-asr 远程客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        parents=[parent],
    )

    sub = parser.add_subparsers(dest="command", help="可用命令")

    # transcribe
    p_t = sub.add_parser("transcribe", parents=[parent], help="原生 API：识别音频")
    p_t.add_argument("audio", help="音频文件路径")
    p_t.add_argument("--language", "-l", default="", help="强制语言 (Chinese/English/Korean/...)")

    # openai
    p_o = sub.add_parser("openai", parents=[parent], help="OpenAI 兼容 API：识别音频")
    p_o.add_argument("audio", help="音频文件路径")
    p_o.add_argument("--language", "-l", default="", help="强制语言")
    p_o.add_argument("--model", "-m", default="Qwen/Qwen3-ASR-0.6B", help="模型名称")
    p_o.add_argument("--format", "-f", default="json",
                     choices=["json", "text", "verbose_json"],
                     help="响应格式，默认 json")

    # health
    sub.add_parser("health", parents=[parent], help="健康检查")

    # models
    sub.add_parser("models", parents=[parent], help="列出可用模型")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    client = Qwen3Client(server=args.server, timeout=args.timeout)

    try:
        if args.command == "health":
            result = client.health()
            print_json(result)

        elif args.command == "models":
            result = client.list_models()
            print_json(result)

        elif args.command == "transcribe":
            if not os.path.isfile(args.audio):
                print(f"❌ 文件不存在: {args.audio}", file=sys.stderr)
                sys.exit(1)
            result = client.transcribe(args.audio, language=args.language)
            print_result(result)

        elif args.command == "openai":
            if not os.path.isfile(args.audio):
                print(f"❌ 文件不存在: {args.audio}", file=sys.stderr)
                sys.exit(1)
            result = client.openai_transcribe(
                args.audio,
                model=args.model,
                language=args.language,
                response_format=args.format,
            )
            if args.format == "text":
                print(result.get("text", ""))
            else:
                print_result(result)

    except KeyboardInterrupt:
        print("\n⏹️  已取消", file=sys.stderr)
        sys.exit(130)


def print_json(data: dict):
    """Pretty-print JSON response."""
    print(json.dumps(data, ensure_ascii=False, indent=2))


def print_result(result: dict):
    """Pretty-print recognition result."""
    # Check for errors first
    if isinstance(result, dict):
        if result.get("error"):
            print(f"❌ 错误: {result['error']}", file=sys.stderr)
            sys.exit(1)
        if result.get("detail"):
            print(f"❌ 错误: {result['detail']}", file=sys.stderr)
            print_json(result)
            sys.exit(1)

        # Native API format
        if "success" in result:
            if not result.get("success"):
                print(f"❌ API 返回失败: {result.get('error', 'unknown')}", file=sys.stderr)
                sys.exit(1)
            r = result.get("result", {})
            _print_recognition(r)
            return

        # OpenAI-compatible format
        if "text" in result and "segments" not in result:
            print(result["text"])
            return

        if "segments" in result:
            _print_openai_verbose(result)
            return

        # Fallback: full JSON
        print_json(result)
    else:
        print(result)


def _print_recognition(r: dict):
    """Pretty-print RecognitionResult from native API."""
    text = r.get("text", "")
    segments = r.get("segments", [])
    duration = r.get("duration", 0)
    language = r.get("language", "unknown")
    stats = r.get("stats", {})

    print(f"\n📝 {text}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"⏱  音频时长:    {duration:.1f}s")
    print(f"🌐 语言:        {language}")
    print(f"🧩 分段数:      {len(segments)}")
    if stats:
        print(f"🔊 ASR 推理:   {stats.get('asr_time', 0):.2f}s")
        print(f"📊 总计耗时:   {stats.get('total_time', 0):.2f}s")

    if segments and len(segments) > 1:
        print(f"\n── 分段详情 ──")
        for i, seg in enumerate(segments):
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            seg_text = seg.get("text", "")
            print(f"  [{i}] {start:>7.2f}s → {end:>7.2f}s | {seg_text}")

    print()


def _print_openai_verbose(r: dict):
    """Pretty-print verbose_json format."""
    text = r.get("text", "")
    segments = r.get("segments", [])
    duration = r.get("duration", 0)
    language = r.get("language", "unknown")

    print(f"\n📝 {text}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"⏱  音频时长:    {duration:.1f}s")
    print(f"🌐 语言:        {language}")
    print(f"🧩 分段数:      {len(segments)}")

    if segments:
        print(f"\n── 分段详情 ──")
        for s in segments:
            start = s.get("start", 0)
            end = s.get("end", 0)
            seg_text = s.get("text", "")
            print(f"  [{s.get('id', '?')}] {start:>7.2f}s → {end:>7.2f}s | {seg_text}")

    print()


if __name__ == "__main__":
    main()
