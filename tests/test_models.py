"""
Tests for Pydantic models — data serialization & validation.
"""
import sys
sys.path.insert(0, "..")

import pytest
from src.models import (
    Segment,
    RecognitionResult,
    RecognitionResponse,
    HealthResponse,
    OpenAICompletionResponse,
    OpenAIVerboseResponse,
    OpenAIVerboseSegment,
    OpenAIModelList,
    OpenAIModelObject,
)


class TestSegment:
    def test_minimal(self):
        seg = Segment(start=0.0, end=1.5, text="hello world")
        assert seg.start == 0.0
        assert seg.end == 1.5
        assert seg.text == "hello world"
        assert seg.language is None

    def test_with_language(self):
        seg = Segment(start=0.0, end=2.0, text="你好", language="Chinese")
        assert seg.language == "Chinese"

    def test_serialization(self):
        seg = Segment(start=0.0, end=1.0, text="test", language="en")
        d = seg.model_dump()
        assert d["start"] == 0.0
        assert d["end"] == 1.0
        assert d["text"] == "test"
        assert d["language"] == "en"


class TestRecognitionResult:
    def test_minimal(self):
        result = RecognitionResult(duration=10.0)
        assert result.duration == 10.0
        assert result.language == "unknown"
        assert result.segments == []
        assert result.text == ""

    def test_with_segments(self):
        segments = [
            Segment(start=0.0, end=1.0, text="hello"),
            Segment(start=1.5, end=2.5, text="world"),
        ]
        result = RecognitionResult(
            language="en",
            duration=3.0,
            segments=segments,
            text="hello world",
        )
        assert len(result.segments) == 2
        assert result.text == "hello world"

    def test_serialization(self):
        seg = Segment(start=0.0, end=1.0, text="test")
        result = RecognitionResult(
            language="en",
            duration=1.0,
            segments=[seg],
            text="test",
        )
        d = result.model_dump()
        assert d["language"] == "en"
        assert d["duration"] == 1.0
        assert len(d["segments"]) == 1
        assert d["text"] == "test"


class TestRecognitionResponse:
    def test_success(self):
        result = RecognitionResult(duration=5.0)
        resp = RecognitionResponse(success=True, result=result)
        assert resp.success is True
        assert resp.result is not None
        assert resp.error is None

    def test_error(self):
        resp = RecognitionResponse(success=False, error="Test error")
        assert resp.success is False
        assert resp.error == "Test error"
        assert resp.result is None

    def test_serialization(self):
        resp = RecognitionResponse(success=True, result=RecognitionResult(duration=1.0))
        d = resp.model_dump()
        assert d["success"] is True
        assert d["result"]["duration"] == 1.0


class TestOpenAICompat:
    def test_completion_response(self):
        resp = OpenAICompletionResponse(text="hello world")
        assert resp.text == "hello world"
        assert resp.model_dump() == {"text": "hello world"}

    def test_verbose_segment(self):
        seg = OpenAIVerboseSegment(id=0, start=0.0, end=1.0, text="hello")
        assert seg.id == 0
        d = seg.model_dump()
        assert d["start"] == 0.0
        assert d["end"] == 1.0
        assert d["text"] == "hello"

    def test_verbose_response(self):
        seg = OpenAIVerboseSegment(id=0, start=0.0, end=1.0, text="hello")
        resp = OpenAIVerboseResponse(
            task="transcribe",
            language="en",
            duration=1.0,
            text="hello",
            segments=[seg],
        )
        assert resp.task == "transcribe"
        assert len(resp.segments) == 1

    def test_model_list(self):
        model = OpenAIModelObject(
            id="Qwen/Qwen3-ASR-0.6B",
            created=1710000000,
            owned_by="Qwen",
        )
        model_list = OpenAIModelList(data=[model])
        d = model_list.model_dump()
        assert d["object"] == "list"
        assert len(d["data"]) == 1
        assert d["data"][0]["id"] == "Qwen/Qwen3-ASR-0.6B"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
