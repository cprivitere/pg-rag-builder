import pytest

from rag import pipeline


def _lookup_result():
    return {
        "ids": [["skill_Pooping_chunk_0"]],
        "documents": [["Skill: Dungcrafting level requirements"]],
        "metadatas": [[{"name": "Dungcrafting", "table": "skills"}]],
        "distances": [[0.3]],
    }


def test_lookup_query_passes_valid_count(monkeypatch):
    captured = {}

    def fake_retrieve(question, metadata_filter=None, query_type=None, count=None, hybrid=None):
        captured["count"] = count
        captured["hybrid"] = hybrid
        return _lookup_result()

    monkeypatch.setattr("rag.pipeline.classify_query", lambda q: "lookup")
    monkeypatch.setattr("rag.pipeline.retrieve", fake_retrieve)
    monkeypatch.setattr("rag.pipeline.should_synthesize", lambda *a, **k: False)
    monkeypatch.setattr("rag.pipeline.generate", lambda p: "Level 5.")

    result = pipeline.ask("what level is Dungcrafting")

    assert result["query_type"] == "lookup"
    assert result["answer"] == "Level 5."
    assert isinstance(captured["count"], int) and captured["count"] > 0
    assert captured["hybrid"] is False
