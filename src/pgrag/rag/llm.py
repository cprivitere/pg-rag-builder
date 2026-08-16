import json

import requests


LLM_URL = "http://localhost:8080/v1/chat/completions"


class LLMServerError(ConnectionError):
    pass


def _post(prompt, stream):
    return requests.post(
        LLM_URL,
        json={
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 8192,
            "stream": stream,
        },
        timeout=300,
        stream=stream,
    )


def _server_error(exc):
    if isinstance(exc, requests.exceptions.ConnectionError):
        return LLMServerError(
            f"Cannot connect to LLM server at {LLM_URL}. "
            "Ensure llama.cpp is running on port 8080."
        )
    if isinstance(exc, requests.exceptions.Timeout):
        return LLMServerError(
            f"LLM server at {LLM_URL} timed out after 300s."
        )
    return exc


def generate(prompt):

    try:
        response = _post(prompt, stream=False)
        response.raise_for_status()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        raise _server_error(e) from e

    return response.json()["choices"][0]["message"]["content"]


def stream_generate(prompt):
    """Stream an LLM completion as a generator of text deltas.

    Uses llama.cpp's OpenAI-compatible /v1/chat/completions endpoint
    (SSE). Yields incremental token strings; raises LLMServerError if
    the server is unreachable.
    """
    try:
        response = _post(prompt, stream=True)
        response.raise_for_status()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        raise _server_error(e) from e

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            delta = json.loads(data)["choices"][0]["delta"].get("content")
        except (ValueError, KeyError, IndexError, TypeError):
            continue
        if delta:
            yield delta