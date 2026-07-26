import requests


LLM_URL = "http://localhost:8080/v1/chat/completions"


def generate(prompt):

    response = requests.post(
        LLM_URL,
        json={
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2
        }
    )

    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]