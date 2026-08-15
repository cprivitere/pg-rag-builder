import tempfile
from pathlib import Path
from unittest.mock import patch

import chromadb
import pytest
from chromadb.api.models.Collection import Collection

from pgrag.config import EMBEDDING_DIM
from pgrag.vectorstore.build_index import build_index


def fake_embed_batch(texts):
    return [[0.1] * EMBEDDING_DIM for _ in texts]


TMP_KW = {"ignore_cleanup_errors": True}


@patch("pgrag.vectorstore.build_index.embed_batch", side_effect=fake_embed_batch)
def test_build_upserts_docs_with_correct_dim(mock_embed):
    docs = [
        {"id": "a", "type": "item", "text": "alpha", "metadata": {"source": "cdn", "table": "items"}},
        {"id": "b", "type": "item", "text": "beta", "metadata": {"source": "cdn", "table": "items"}},
    ]
    with tempfile.TemporaryDirectory(**TMP_KW) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        build_index(documents=docs, chroma_path=chroma_path)
        client = chromadb.PersistentClient(path=chroma_path)
        coll = client.get_collection("project_gorgon")
        assert coll.count() == 2
        result = coll.get(include=["embeddings", "metadatas"])
        assert set(result["ids"]) == {"a", "b"}
        assert len(result["embeddings"][0]) == EMBEDDING_DIM


@patch("pgrag.vectorstore.build_index.embed_batch", side_effect=fake_embed_batch)
def test_build_deleted_doc_purged(mock_embed):
    docs_a = [
        {"id": "a", "type": "item", "text": "alpha", "metadata": {"source": "cdn", "table": "items"}},
        {"id": "b", "type": "item", "text": "beta", "metadata": {"source": "cdn", "table": "items"}},
    ]
    docs_b = [
        {"id": "a", "type": "item", "text": "alpha", "metadata": {"source": "cdn", "table": "items"}},
    ]
    with tempfile.TemporaryDirectory(**TMP_KW) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        build_index(documents=docs_a, chroma_path=chroma_path)
        build_index(documents=docs_b, chroma_path=chroma_path)
        client = chromadb.PersistentClient(path=chroma_path)
        coll = client.get_collection("project_gorgon")
        assert coll.count() == 1
        assert coll.get()["ids"] == ["a"]


@patch("pgrag.vectorstore.build_index.embed_batch", side_effect=fake_embed_batch)
def test_build_metadata_only_skips_reembed(mock_embed):
    docs_first = [
        {"id": "x", "type": "item", "text": "same", "metadata": {"source": "cdn", "table": "items", "name": "old"}},
    ]
    docs_second = [
        {"id": "x", "type": "item", "text": "same", "metadata": {"source": "cdn", "table": "items", "name": "new"}},
    ]
    with tempfile.TemporaryDirectory(**TMP_KW) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        build_index(documents=docs_first, chroma_path=chroma_path)
        build_index(documents=docs_second, chroma_path=chroma_path)
        client = chromadb.PersistentClient(path=chroma_path)
        coll = client.get_collection("project_gorgon")
        result = coll.get(include=["metadatas"])
        assert result["metadatas"][0]["name"] == "new"
        assert coll.count() == 1


def test_build_dimension_mismatch_aborts():
    docs_first = [
        {"id": "a", "type": "item", "text": "alpha", "metadata": {"source": "cdn", "table": "items"}},
    ]
    docs_second = [
        {"id": "b", "type": "item", "text": "beta", "metadata": {"source": "cdn", "table": "items"}},
    ]
    with tempfile.TemporaryDirectory(**TMP_KW) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        with patch("pgrag.vectorstore.build_index.embed_batch", return_value=[[0.1, 0.2, 0.3, 0.4]]), \
             patch("pgrag.vectorstore.build_index.EMBEDDING_DIM", 4):
            build_index(documents=docs_first, chroma_path=chroma_path)
        with patch("pgrag.vectorstore.build_index.embed_batch", return_value=[[0.1, 0.2]]), \
             patch("pgrag.vectorstore.build_index.EMBEDDING_DIM", 4):
            with pytest.raises(Exception, match="expected 4"):
                build_index(documents=docs_second, chroma_path=chroma_path)


def test_v23_build_start_aborts_on_dim_mismatch():
    docs_first = [
        {"id": "a", "type": "item", "text": "alpha", "metadata": {"source": "cdn", "table": "items"}},
    ]
    docs_second = [
        {"id": "b", "type": "item", "text": "beta", "metadata": {"source": "cdn", "table": "items"}},
    ]
    with tempfile.TemporaryDirectory(**TMP_KW) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        with patch("pgrag.vectorstore.build_index.embed_batch", return_value=[[0.1, 0.2, 0.3, 0.4]]), \
             patch("pgrag.vectorstore.build_index.EMBEDDING_DIM", 4):
            build_index(documents=docs_first, chroma_path=chroma_path)
        with patch("pgrag.vectorstore.build_index.embed_batch", side_effect=AssertionError("must not embed")):
            with pytest.raises(ValueError, match="EMBEDDING_DIM"):
                build_index(documents=docs_second, chroma_path=chroma_path)


def test_build_interleaves_embed_and_upsert_per_batch():
    docs = [
        {"id": f"d{i}", "type": "item", "text": f"text {i}", "metadata": {"source": "cdn"}}
        for i in range(9)
    ]
    real_upsert = Collection.upsert
    events = []

    def tracked_embed(texts):
        events.append(("embed", len(texts)))
        return [[0.1] * EMBEDDING_DIM for _ in texts]

    def tracked_upsert(self, ids=None, embeddings=None, metadatas=None, documents=None, **kw):
        events.append(("upsert", len(ids)))
        return real_upsert(
            self, ids=ids, embeddings=embeddings,
            metadatas=metadatas, documents=documents, **kw
        )

    with tempfile.TemporaryDirectory(**TMP_KW) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        with patch("pgrag.vectorstore.build_index.embed_batch", side_effect=tracked_embed), \
             patch("pgrag.vectorstore.build_index.EMBED_BATCH_SIZE", 4), \
             patch("pgrag.vectorstore.build_index.BATCH_SIZE", 2), \
             patch.object(Collection, "upsert", side_effect=tracked_upsert, autospec=True):
            build_index(documents=docs, chroma_path=chroma_path)

    # each embed batch must be upserted before the next embed batch starts
    assert events == [
        ("embed", 4), ("upsert", 2), ("upsert", 2),
        ("embed", 4), ("upsert", 2), ("upsert", 2),
        ("embed", 1), ("upsert", 1),
    ]


def test_build_interruption_persists_completed_batches_and_resumes():
    docs = [
        {"id": f"d{i}", "type": "item", "text": f"text {i}", "metadata": {"source": "cdn"}}
        for i in range(8)
    ]
    embed_calls = {"n": 0}

    def failing_embed(texts):
        embed_calls["n"] += 1
        if embed_calls["n"] > 1:
            raise RuntimeError("simulated crash")
        return [[0.1] * EMBEDDING_DIM for _ in texts]

    with tempfile.TemporaryDirectory(**TMP_KW) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        with patch("pgrag.vectorstore.build_index.embed_batch", side_effect=failing_embed), \
             patch("pgrag.vectorstore.build_index.EMBED_BATCH_SIZE", 4), \
             patch("pgrag.vectorstore.build_index.BATCH_SIZE", 4):
            with pytest.raises(RuntimeError, match="simulated crash"):
                build_index(documents=docs, chroma_path=chroma_path)

        client = chromadb.PersistentClient(path=chroma_path)
        coll = client.get_collection("project_gorgon")
        # first batch persisted despite the crash
        assert coll.count() == 4

        # resume: only the remaining docs are embedded, not re-embedding persisted ones
        embedded_texts = []

        def record_embed(texts):
            embedded_texts.extend(texts)
            return [[0.1] * EMBEDDING_DIM for _ in texts]

        with patch("pgrag.vectorstore.build_index.embed_batch", side_effect=record_embed), \
             patch("pgrag.vectorstore.build_index.EMBED_BATCH_SIZE", 4), \
             patch("pgrag.vectorstore.build_index.BATCH_SIZE", 4):
            build_index(documents=docs, chroma_path=chroma_path)

        assert embedded_texts == ["text 4", "text 5", "text 6", "text 7"]
        assert coll.count() == 8
