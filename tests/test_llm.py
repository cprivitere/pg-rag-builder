"""Tests for the LLM client's streaming (SSE) parsing."""

import pytest

from pgrag.rag import llm


class _FakeResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        yield from self._chunks


def test_stream_generate_parses_sse_deltas(monkeypatch):
    seen = {}

    def fake_post(prompt, stream):
        seen["stream"] = stream
        chunks = [
            'data: {"choices": [{"delta": {"content": "Hel"}}]}',
            "event: keepalive",
            'data: {"choices": [{"delta": {"content": "lo"}}]}',
            'data: {"choices": [{"delta": {}}]}',
            "data: [DONE]",
            'data: {"choices": [{"delta": {"content": "IGNORED"}}]}',
        ]
        return _FakeResponse(chunks)

    monkeypatch.setattr(llm, "_post", fake_post)

    assert list(llm.stream_generate("prompt")) == ["Hel", "lo"]
    assert seen["stream"] is True


def test_stream_generate_raises_server_error(monkeypatch):
    import requests

    def boom(prompt, stream):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(llm, "_post", boom)

    with pytest.raises(llm.LLMServerError):
        list(llm.stream_generate("prompt"))