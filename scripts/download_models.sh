#!/bin/bash
# ==========================================================================
# sherpa-qwen3-asr — Model Download Script
#
# Downloads:
#   1. Qwen3-ASR 0.6B int8   (~1.5 GB)  — Main ASR model
#   2. Silero VAD v5         (~2.2 MB)  — Voice Activity Detection
#
# Usage:
#   chmod +x scripts/download_models.sh
#   ./scripts/download_models.sh
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

echo "============================================"
echo " sherpa-qwen3-asr — Model Download"
echo "============================================"
echo "Project: $PROJECT_DIR"
echo "Models:  $MODELS_DIR"
echo ""

mkdir -p "$MODELS_DIR"

# ---- Helper: download + extract tar.bz2 ----
download_and_extract() {
    local url="$1"
    local dest_dir="$2"
    local name="$3"

    echo ""
    echo "━━━ Downloading ${name}... ━━━"

    local archive="${MODELS_DIR}/$(basename "$url")"

    if [ -d "$dest_dir" ] && [ -n "$(ls -A "$dest_dir" 2>/dev/null)" ]; then
        echo "  ✓ Already exists, skipping: $dest_dir"
        return
    fi

    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$archive" "$url"
    elif command -v curl &>/dev/null; then
        curl -# -L -o "$archive" "$url"
    else
        echo "ERROR: Neither wget nor curl found. Please install one."
        exit 1
    fi

    echo "  → Extracting..."
    tar xf "$archive" -C "$MODELS_DIR"
    rm "$archive"

    # Flatten: move extracted contents into dest_dir
    local archive_basename
    archive_basename=$(basename "$url" .tar.bz2)
    local extracted_dir="${MODELS_DIR}/${archive_basename}"
    if [ -d "$extracted_dir" ]; then
        mkdir -p "$dest_dir"
        mv "$extracted_dir"/* "$dest_dir"/ 2>/dev/null || true
        mv "$extracted_dir"/.[!.]* "$dest_dir"/ 2>/dev/null || true
        rmdir "$extracted_dir" 2>/dev/null || true
    fi

    echo "  ✓ Done: $dest_dir"
}

# ---- Helper: download single file ----
download_file() {
    local url="$1"
    local dest="$2"
    local name="$3"

    echo ""
    echo "━━━ Downloading ${name}... ━━━"

    if [ -f "$dest" ]; then
        echo "  ✓ Already exists, skipping: $dest"
        return
    fi

    mkdir -p "$(dirname "$dest")"

    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest" "$url"
    elif command -v curl &>/dev/null; then
        curl -# -L -o "$dest" "$url"
    else
        echo "ERROR: Neither wget nor curl found."
        exit 1
    fi

    echo "  ✓ Done: $dest"
}

# =====================================================================
# 1. Qwen3-ASR 0.6B int8 (main ASR model)
# =====================================================================
QWEN3_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2"
QWEN3_DIR="$MODELS_DIR/qwen3-asr"

download_and_extract "$QWEN3_URL" "$QWEN3_DIR" "Qwen3-ASR 0.6B int8"

# =====================================================================
# 2. Silero VAD v5 (voice activity detection)
# =====================================================================
# v5 is the latest version with better accuracy
VAD_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad_v5.onnx"
VAD_FILE="$MODELS_DIR/vad/silero_vad.onnx"

download_file "$VAD_URL" "$VAD_FILE" "Silero VAD v5"

# =====================================================================
# Summary
# =====================================================================
echo ""
echo "============================================"
echo " All models downloaded successfully!"
echo "============================================"
echo ""
echo "Model files:"
du -sh "$MODELS_DIR"/*/ 2>/dev/null
du -sh "$MODELS_DIR"/vad/*.onnx 2>/dev/null
echo ""
echo "Total size:"
du -sh "$MODELS_DIR"
echo ""
echo "Next steps:"
echo "  1. Start the server:    python -m src.api"
echo "  2. Test the API:        curl http://localhost:8000/api/v1/health"
echo ""
