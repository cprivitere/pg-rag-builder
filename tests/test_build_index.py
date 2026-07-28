from unittest.mock import MagicMock, patch

from vectorstore.hashes import embedding_hash, metadata_hash
from vectorstore.build_index import _get_existing_dim


def test_v7_deleted_ids_computation():
    existing_ids = {"a", "b", "c", "d"}
    current_ids = {"b", "d", "e"}
    deleted = existing_ids - current_ids
    assert deleted == {"a", "c"}


def test_v4_unchanged_doc_skipped():
    doc = {"id": "x", "text": "hello", "metadata": {"source": "cdn"}}
    embed_h = embedding_hash(doc)
    meta_h = metadata_hash(doc)
    assert embed_h == embedding_hash(doc)
    assert meta_h == metadata_hash(doc)
    is_unchanged = embed_h == embedding_hash(doc) and meta_h == metadata_hash(doc)
    assert is_unchanged


def test_v4_metadata_only_change_detected():
    doc_old = {"id": "x", "text": "hello", "metadata": {"source": "cdn"}}
    doc_new = {"id": "x", "text": "hello", "metadata": {"source": "wiki"}}
    same_embed = embedding_hash(doc_old) == embedding_hash(doc_new)
    diff_meta = metadata_hash(doc_old) != metadata_hash(doc_new)
    assert same_embed, "embedding_hash must be same (text unchanged)"
    assert diff_meta, "metadata_hash must differ (metadata changed)"


def test_v4_embed_change_detected():
    doc_old = {"id": "x", "text": "hello", "metadata": {"source": "cdn"}}
    doc_new = {"id": "x", "text": "goodbye", "metadata": {"source": "cdn"}}
    same_embed = embedding_hash(doc_old) == embedding_hash(doc_new)
    diff_meta = metadata_hash(doc_old) == metadata_hash(doc_new)
    assert not same_embed, "embedding_hash must differ (text changed)"
    assert diff_meta, "metadata_hash must be same (metadata unchanged)"


def test_get_existing_dim_with_embeddings():
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "ids": ["a"],
        "embeddings": [[0.1, 0.2, 0.3, 0.4]],
    }
    assert _get_existing_dim(mock_collection) == 4


def test_get_existing_dim_empty_collection():
    mock_collection = MagicMock()
    mock_collection.get.return_value = {"ids": [], "embeddings": []}
    assert _get_existing_dim(mock_collection) is None


def test_get_existing_dim_no_embeddings():
    mock_collection = MagicMock()
    mock_collection.get.return_value = {"ids": ["a"], "embeddings": [[]]}
    assert _get_existing_dim(mock_collection) is None
