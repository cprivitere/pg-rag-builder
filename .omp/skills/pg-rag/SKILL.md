---
name: pg-rag
description: How the pg-rag-builder pipeline actually works — loaders, document generation, indexing, retrieval stages, and test discipline. Use when modifying the RAG pipeline, retrieval behavior, indexing, chunking, embeddings, reranking, or query processing in this repo.
---

# pg-rag — project architecture

pg-rag-builder turns Project Gorgon CDN/database tables + externally synced wiki
content into a searchable knowledge base. Everything below already exists — do
not redesign it without reading the code first.

## Pipeline

```
source data  →  loaders  →  normalized records  →  document generation
  →  documents.json  →  chunking → embeddings  →  Chroma (project_gorgon)
  →  hybrid retrieval: dense + BM25 → RRF → reranker (:8082, lexical fallback)
  →  answer generation (LLM :8080)
```

## Commands (mise)

- `mise refresh` / `refresh-full` / `refresh-cdn` / `refresh-wiki` / `refresh-documents` / `refresh-vectors` — piecemeal pipeline steps
- `mise validate` — index health check (rebuilds if broken)
- `mise status` — service health; `mise start` / `mise down` — services up/down
- `mise chat` — Gradio chat; `mise test` — `uv run -m pytest tests/`
- `uv run pgrag build-index --source cdn|wiki|computed|curated` — partial rebuild of one source only

## Key files

- `src/pgrag/loaders/` — disk→DB: `cdn_loader.py`, `wiki_loader.py`, `download_cdn.py`, `download_wiki.py` (wiki sync), `database.py`
- `src/pgrag/documents/builder.py` — CDN entity → docs (items, recipes, skills, quests, abilities, …)
- `src/pgrag/documents/wiki_builder.py` — wiki → sections (chunks by `==heading==`, `parent_id` metadata links page chunks)
- `src/pgrag/documents/chunking.py` — 1024c/100ov (lorebook+skillprofile 2048, summary+curated 8192)
- `src/pgrag/documents/resolver.py` + `skill_profiles.py` + `summaries.py` — cross-refs, leveling dossiers, gathering summaries
- `src/pgrag/embeddings/llama_embeddings.py` → :8081 (mxbai-xsmall-Q8, fixed dim contract)
- `src/pgrag/vectorstore/build_index.py` — incremental upsert; metadata-only changes skip re-embedding
- `src/pgrag/rag/retriever.py` — chroma query + `_hybrid_fuse` (RRF) + rerank
- `src/pgrag/rag/bm25.py` — lexical arm; currently re-parses `documents.json` per hybrid query
- `src/pgrag/rag/query_classifier.py` — entity / comparison / general (+ leveling-intent → skill entity)
- `src/pgrag/rag/pipeline.py` — `ask()` orchestration: entity dossier or general top-k, `_gap_fill` re-retrieval, streaming variant `ask_stream`
- `src/pgrag/rag/prompts.py` — prompt builder incl. leveling ranking block
- `scripts/golden_check.py` + `data/golden/` — fact-presence eval (see `evaluation` skill)

See `AGENTS.md` (repo root) for the compressed quick reference — it is kept in
sync with this skill; if they drift, AGENTS.md wins for commands, this skill
wins for architecture detail.

## Testing discipline

- All 341+ tests are offline; 5 golden tests skip unless :8080+:8081 are up.
- Retrieval changes ⇒ run `tests/test_bm25.py`, `tests/test_retrieval_unit.py`, `tests/test_rerank*.py`, `tests/test_retriever_spelling.py`.
- Document-generation changes ⇒ inspect representative docs after `build-documents`; builder tests assert keys, not exact dicts.
- Index changes ⇒ `tests/test_build_index.py`, `tests/test_hashes.py`, `tests/test_health_check.py`, then `pgrag validate`.
- Wiki loader changes ⇒ `tests/test_download_wiki.py` MUST keep tmp-dir isolation (real `data/wiki/.meta.json` is not to be touched).