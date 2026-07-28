# pg-rag-builder — agent guide

## Commands

```bash
uv run python download_cdn.py      # fetch CDN JSON -> data/cdn/
uv run python download_wiki.py     # fetch wiki pages -> data/wiki/ (via mwclient)
uv run python main.py              # build data/documents.json from CDN+wiki
uv run python -m vectorstore.build_index  # upsert ChromaDB index
uv run python search.py            # interactive substring search
uv run python -m vectorstore.health_check  # validate index integrity
```

Mise tasks: `mise r` (refresh = download_cdn + download_wiki + main + build_index), `mise t` (test), `mise v` (validate = health-check then rebuild if needed).

## Pipeline order

```
download_cdn.py → download_wiki.py → main.py → vectorstore.build_index
```

Each step depends on prior output. Never build index without documents. Never build documents without data downloaded.

## Architecture

- `database.GameDatabase` — two buckets: `tables` (CDN data), `wiki` (page text)
- `loaders/` — read CDN JSON + wiki txt from disk into DB
- `documents/builder.py` — chunk CDN+w初i into docs, each `{id, type, text, metadata}`
- `embeddings/llama_embeddings.py` — POST to llama.cpp `:8081` for vectors
- `vectorstore/build_index.py` — incremental upsert to ChromaDB `data/chroma/`
- `rag/retriever.py` — query ChromaDB, optional rerank + hybrid BM25
- `rag/bm25.py` — custom BM25 on `data/documents.json`

## External services

| Service | Port | Required for |
|---------|------|-------------|
| llama.cpp embedding | :8081 | `embed_batch`, `embed_text` |
| llama.cpp LLM | :8080 | RAG Q&A (`test_rag.py`) |

LLM server not needed for build/refresh — only for interactive RAG queries.

## Data directory

`data/` is gitignored, not tracked. Contents:
- `data/cdn/*.json` — game CDN exports (27 tables)
- `data/wiki/*.txt` — wiki page dumps (originally from `wiki.projectgorgon.com`)
- `data/documents.json` — built by `main.py`
- `data/chroma/` — ChromaDB persistent store
- `data/wiki/.meta.json` — touched timestamps for freshness (auto-managed)

## Testing

- `pytest` — 90+ tests
- Interactive tests excluded from pytest run: `test_rag.py`, `test_retrieval.py`, `test_embedding.py`, `test_similarity.py`
- Integration tests use temp ChromaDB dirs — no external deps needed
- Health-check tests use temp dirs + synthetic docs

## SPEC.md

cavekit format: §G (Goal), §C (Constraints), §I (Interfaces), §V (Invariants), §T (Tasks), §B (Bugs). Source of truth for what's built and why. All T items currently `x` (complete).

## Python

Requires ≥3.14. Package manager: `uv`. Deps in `pyproject.toml`. Lockfile: `uv.lock`. Run any Python via `uv run python ...`.

## OpenCode config — never commit

`.agents/`, `skills-lock.json`, `docs/superpowers/` — local-only tooling, gitignored.
