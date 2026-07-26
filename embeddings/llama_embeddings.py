import requests


EMBEDDING_URL = "http://localhost:8081/embedding"


def embed_text(text):
    return embed_batch([text])[0]


def embed_batch(texts):
    response = requests.post(
        EMBEDDING_URL,
        json={
            "content": texts
        },
        timeout=300
    )

    response.raise_for_status()

    data = response.json()

    return [
        item["embedding"][0]
        for item in data
    ]