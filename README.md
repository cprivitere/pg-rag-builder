# pg-rag-builder

RAG pipeline for Project Gorgon game data (CDN exports + wiki). Build docs → embed → ChromaDB → hybrid retrieval → LLM answers.

## Server launch commands (see `mise.toml` / `.mise/tasks/*.ps1` for the canonical versions)

Embedding (mxbai-embed-xsmall-v1, pooling mean):
```
llama-server -hf twine-network/mxbai-embed-xsmall-v1-Q8_0-GGUF:Q8_0 --host 0.0.0.0 --port 8081 --embedding --pooling mean -ngl 99 --ubatch-size 8192 -np 1 -c 4096
```

LLM (Gemma 4 26B, MTP draft, reasoning budget):
```
llama-server -hf unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL --spec-type draft-mtp --spec-draft-n-max 4 -ngl 999 -fa on -c 16384 --reasoning-budget 1024 --host 0.0.0.0 --port 8080
```

Reranker (bge-reranker-v2-m3 Q4_K_M):
```
llama-server -hf gpustack/bge-reranker-v2-m3-GGUF:Q4_K_M --host 0.0.0.0 --port 8082 --reranking --pooling rank -c 32768 -ngl 99
```

Open WebUI:
```
uv run open-webui serve --port 3000
```

See `AGENTS.md` for the full command reference, mise tasks, architecture, and operational gotchas.