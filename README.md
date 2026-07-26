Embedding model
llama-server -hf jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:Q8_0 --host 0.0.0.0 --port 8081 --embedding --pooling last -ngl 99 -np 16

LLM
llama-server -hf unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL --spec-type draft-mtp --spec-draft-n-max 4 -ngl 999 -fa on -c 8192

Open-Webui Launch command
uv run open-webui serve --port 3000
