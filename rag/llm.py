import requests


LLM_URL = "http://localhost:8080/v1/chat/completions"


class LLMServerError(ConnectionError):
    pass


def generate(prompt):

    try:
        response = requests.post(
            LLM_URL,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 8192
            },
            timeout=300
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise LLMServerError(
            f"Cannot connect to LLM server at {LLM_URL}. "
            "Ensure llama.cpp is running on port 8080."
        ) from e
    except requests.exceptions.Timeout as e:
        raise LLMServerError(
            f"LLM server at {LLM_URL} timed out after 120s."
        ) from e

    return response.json()["choices"][0]["message"]["content"]

