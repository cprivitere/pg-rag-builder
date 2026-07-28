import requests


EMBEDDING_URL = "http://localhost:8081/embedding"


def embed_text(text):
    return embed_batch([text])[0]


class EmbeddingServerError(ConnectionError):
    pass


def embed_batch(texts):
    try:
        response = requests.post(
            EMBEDDING_URL,
            json={
                "content": texts
            },
            timeout=300
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise EmbeddingServerError(
            f"Cannot connect to embedding server at {EMBEDDING_URL}. "
            "Ensure llama.cpp is running on port 8081."
        ) from e
    except requests.exceptions.Timeout as e:
        raise EmbeddingServerError(
            f"Embedding server at {EMBEDDING_URL} timed out after 300s."
        ) from e

    data = response.json()

    vectors = [
        item["embedding"][0]
        for item in data
    ]

    vectors = validate_embeddings(vectors)

    return vectors


class EmbeddingValidationError(ValueError):
    pass


def validate_embeddings(vectors, expected_dim=None):
    if not vectors:
        raise EmbeddingValidationError("Empty embedding response")

    for i, vec in enumerate(vectors):
        if not isinstance(vec, list):
            raise EmbeddingValidationError(f"Vector {i}: expected list, got {type(vec).__name__}")
        if not vec:
            raise EmbeddingValidationError(f"Vector {i} is empty")
        for j, v in enumerate(vec):
            if not isinstance(v, (int, float)):
                raise EmbeddingValidationError(
                    f"Vector {i}[{j}]: expected float, got {type(v).__name__}"
                )

    if expected_dim is not None:
        for i, vec in enumerate(vectors):
            if len(vec) != expected_dim:
                raise EmbeddingValidationError(
                    f"Vector {i} length {len(vec)} != expected {expected_dim}"
                )

    return vectors
