from unittest.mock import patch, MagicMock

from rag.retriever import retrieve
from rag.pipeline import ask


@patch("rag.retriever.embed_text")
@patch("rag.retriever.chromadb.PersistentClient")
def test_retrieve_default_no_filter(mock_client, mock_embed):
    mock_embed.return_value = [0.1] * 128
    mock_col = MagicMock()
    mock_client.return_value.get_collection.return_value = mock_col

    retrieve("test question", count=3)

    _, kwargs = mock_col.query.call_args
    assert "where" not in kwargs


@patch("rag.retriever.embed_text")
@patch("rag.retriever.chromadb.PersistentClient")
def test_retrieve_with_source_filter(mock_client, mock_embed):
    mock_embed.return_value = [0.1] * 128
    mock_col = MagicMock()
    mock_client.return_value.get_collection.return_value = mock_col

    retrieve("test question", count=5, metadata_filter={"source": "cdn"})

    _, kwargs = mock_col.query.call_args
    assert kwargs.get("where") == {"source": "cdn"}
    assert kwargs.get("n_results") == 15


@patch("rag.retriever.embed_text")
@patch("rag.retriever.chromadb.PersistentClient")
def test_retrieve_with_table_filter(mock_client, mock_embed):
    mock_embed.return_value = [0.1] * 128
    mock_col = MagicMock()
    mock_client.return_value.get_collection.return_value = mock_col

    retrieve("test question", count=3, metadata_filter={"table": "items"})

    _, kwargs = mock_col.query.call_args
    assert kwargs.get("where") == {"table": "items"}


@patch("rag.retriever.embed_text")
@patch("rag.retriever.chromadb.PersistentClient")
def test_retrieve_with_composite_filter(mock_client, mock_embed):
    mock_embed.return_value = [0.1] * 128
    mock_col = MagicMock()
    mock_client.return_value.get_collection.return_value = mock_col

    retrieve("test", count=10, metadata_filter={"source": "wiki", "table": "skills"})

    _, kwargs = mock_col.query.call_args
    assert kwargs.get("where") == {"source": "wiki", "table": "skills"}


@patch("rag.pipeline.retrieve")
@patch("rag.pipeline.generate")
def test_ask_passes_metadata_filter(mock_generate, mock_retrieve):
    mock_generate.return_value = "mock answer"
    mock_retrieve.return_value = {
        "documents": [[]],
        "ids": [[]],
        "distances": [[]],
        "metadatas": [[]]
    }

    ask("test", metadata_filter={"source": "cdn"})

    mock_retrieve.assert_called_once()
    _, kwargs = mock_retrieve.call_args
    assert kwargs.get("metadata_filter") == {"source": "cdn"}


@patch("rag.pipeline.retrieve")
@patch("rag.pipeline.generate")
def test_source_citation_format_with_name(mock_generate, mock_retrieve):
    mock_generate.return_value = "mock answer"
    mock_retrieve.return_value = {
        "documents": [["doc text"]],
        "ids": [["item_96"]],
        "distances": [[0.42]],
        "metadatas": [[{"name": "Bunny Juice", "table": "items", "source": "cdn"}]]
    }

    result = ask("test")

    assert len(result["sources"]) == 1
    src = result["sources"][0]
    assert src["id"] == "item_96"
    assert src["citation"] == "Bunny Juice (items)"
    assert src["distance"] == 0.42


@patch("rag.pipeline.retrieve")
@patch("rag.pipeline.generate")
def test_source_citation_fallback_to_id(mock_generate, mock_retrieve):
    mock_generate.return_value = "mock answer"
    mock_retrieve.return_value = {
        "documents": [["doc text"]],
        "ids": [["item_96"]],
        "distances": [[0.42]],
        "metadatas": [[{"table": "items", "source": "cdn"}]]
    }

    result = ask("test")

    assert result["sources"][0]["citation"] == "item_96 (items)"


@patch("rag.pipeline.retrieve")
@patch("rag.pipeline.generate")
def test_source_citation_unknown_table(mock_generate, mock_retrieve):
    mock_generate.return_value = "mock answer"
    mock_retrieve.return_value = {
        "documents": [["doc text"]],
        "ids": [["item_96"]],
        "distances": [[0.42]],
        "metadatas": [[{"name": "Bunny Juice"}]]
    }

    result = ask("test")

    assert result["sources"][0]["citation"] == "Bunny Juice (unknown)"
