from pathlib import Path
from unittest.mock import patch

from rag import pipeline


FAKE_RESULTS = {
    "documents": [["doc one", "doc two", "doc three"]],
    "ids": [["d1", "d2", "d3"]],
    "distances": [[1.5, 1.6, 1.7]],
    "metadatas": [
        [{"source": "wiki", "table": "wiki", "name": "one"},
         {"source": "cdn", "table": "items", "name": "two"},
         {"source": "wiki", "table": "wiki", "name": "three"}],
    ],
}


def _run_ask(tmp_path, question="where are the level 30 areas"):
    patches = [
        patch.object(pipeline, "CURATED_DIR", tmp_path),
        patch.object(pipeline, "retrieve", return_value=FAKE_RESULTS),
        patch.object(pipeline, "should_synthesize", return_value=True),
        patch.object(pipeline, "synthesize_answer", return_value="synthesized answer text"),
        patch.object(pipeline, "generate", return_value="final answer"),
    ]
    for p in patches:
        p.start()
    try:
        return pipeline.ask(question)
    finally:
        for p in reversed(patches):
            p.stop()


def test_v24_synthesis_persisted_to_curated_dir(tmp_path):
    _run_ask(tmp_path)
    files = list(Path(tmp_path).glob("synthesized_*_curated.txt"))
    assert len(files) == 1
    assert "synthesized answer text" in files[0].read_text()


def test_v24_synthesis_persisted_id_stable(tmp_path):
    _run_ask(tmp_path)
    _run_ask(tmp_path)
    files = list(Path(tmp_path).glob("synthesized_*_curated.txt"))
    assert len(files) == 1, "same query must reuse same file"


def test_v24_create_curated_doc_returns_shape():
    from rag.synthesis_generator import create_curated_doc
    results = [
        {"text": "one", "metadata": {"source": "wiki", "name": "one"}},
        {"text": "two", "metadata": {"source": "cdn", "name": "two"}},
    ]
    doc = create_curated_doc("question here", results, synthesized_text="custom text")
    assert doc["type"] == "synthesized"
    assert doc["text"] == "custom text"
    assert doc["metadata"]["source"] == "synthesized"
    assert doc["id"].startswith("synthesized_")
