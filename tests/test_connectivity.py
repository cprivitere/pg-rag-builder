import pytest
import requests
from unittest.mock import patch

from embeddings.llama_embeddings import embed_batch, EmbeddingServerError, EMBEDDING_URL
from rag.llm import generate, LLMServerError, LLM_URL


def test_embedding_server_unreachable():
    with patch("embeddings.llama_embeddings.requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError()
        with pytest.raises(EmbeddingServerError) as exc:
            embed_batch(["test"])
        msg = str(exc.value)
        assert "8081" in msg
        assert EMBEDDING_URL in msg


def test_embedding_server_timeout():
    with patch("embeddings.llama_embeddings.requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout()
        with pytest.raises(EmbeddingServerError) as exc:
            embed_batch(["test"])
        msg = str(exc.value)
        assert "timed out" in msg.lower()
        assert EMBEDDING_URL in msg


def test_embedding_server_http_error_still_raised():
    with patch("embeddings.llama_embeddings.requests.post") as mock_post:
        mock_response = mock_post.return_value
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        with pytest.raises(requests.exceptions.HTTPError):
            embed_batch(["test"])


def test_llm_server_unreachable():
    with patch("rag.llm.requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError()
        with pytest.raises(LLMServerError) as exc:
            generate("test prompt")
        msg = str(exc.value)
        assert "8080" in msg
        assert LLM_URL in msg


def test_llm_server_timeout():
    with patch("rag.llm.requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout()
        with pytest.raises(LLMServerError) as exc:
            generate("test prompt")
        msg = str(exc.value)
        assert "timed out" in msg.lower()
        assert LLM_URL in msg


def test_llm_server_http_error_still_raised():
    with patch("rag.llm.requests.post") as mock_post:
        mock_response = mock_post.return_value
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        with pytest.raises(requests.exceptions.HTTPError):
            generate("test prompt")
