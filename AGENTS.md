# pg-rag-builder — agent guide

## Commands

```bash
uv run python download_cdn.py      # fetch CDN JSON -> data/cdn/
uv run python download_wiki.py     # fetch wiki pages -> data/wiki/ (via mwclient)
uv run python main.py              # build data/documents.json from CDN+wiki
uv run python -m vectorstore.build_index  # upsert ChromaDB index
uv run python search.py            # interactive substring search
uv run python -m vectorstore.health_check  # validate index integrity
uv run python -m scripts.rag       # interactive RAG Q&A (requires LLM server)
uv run python -m scripts.retrieval # interactive Chroma similarity search (requires embedding server)
```

### Mise Tasks

**Pipeline:**
- `mise r` / `mise refresh` — Full refresh: download_cdn + download_wiki + main + build_index
- `mise t` / `mise test` — Run pytest test suite
- `mise v` / `mise validate` — Health-check index, rebuild if issues found

**Service Management (with logging):**
- `mise start` / `mise start-all` — Start all services (embed + llm + webui) with logging
- `mise down` / `mise stop-all` — Stop all services
- `mise se` / `mise start-embed` — Start embedding server (production mode) with logging
- `mise eb` / `mise start-embed-build` — Start embedding server (build mode) with logging
- `mise sl` / `mise start-llm` — Start LLM server with logging
- `mise sw` / `mise start-webui` — Start Open WebUI with logging
- `mise xe` / `mise stop-embed` — Stop embedding server
- `mise xl` / `mise stop-llm` — Stop LLM server
- `mise xw` / `mise stop-webui` — Stop Open WebUI (kills process on port 3000)

**Debug Mode (foreground, direct stdout):**
- `mise dse` / `mise debug-embed` — Start embedding server in foreground (production)
- `mise deb` / `mise debug-embed-build` — Start embedding server in foreground (build mode)
- `mise dsl` / `mise debug-llm` — Start LLM server in foreground
- `mise dsw` / `mise debug-webui` — Start Open WebUI in foreground

**Monitoring:**
- `mise st` / `mise status` — Check status of all services (embed, llm, webui)
- `mise logs` — List all log files in logs/ directory
- `mise tle` / `mise tail-embed` — Tail embedding server log (live, last 20 lines)
- `mise tll` / `mise tail-llm` — Tail LLM server log (live, last 20 lines)
- `mise tlw` / `mise tail-webui` — Tail Open WebUI log (live, last 20 lines)

**Note:** Use `mise start` instead of `mise up` to avoid conflict with built-in `mise update` command

**Logging:**
All background services write to `logs/` directory:
- `logs/embed.log` / `logs/embed-error.log` — Embedding server output
- `logs/llm.log` / `logs/llm-error.log` — LLM server output
- `logs/webui.log` / `logs/webui-error.log` — Open WebUI output
- `logs/embed-build.log` / `logs/embed-build-error.log` — Embedding build mode output

## Pipeline order

```
download_cdn.py → download_wiki.py → main.py → vectorstore.build_index
```

Each step depends on prior output. Never build index without documents. Never build documents without data downloaded.

## Architecture

- `database.GameDatabase` — two buckets: `tables` (CDN data), `wiki` (page text)
- `loaders/` — read CDN JSON + wiki txt from disk into DB
- `documents/builder.py` — chunk CDN+wiki into docs, each `{id, type, text, metadata}`
- `embeddings/llama_embeddings.py` — POST to llama.cpp `:8081` for vectors
- `vectorstore/build_index.py` — incremental upsert to ChromaDB `data/chroma/`
- `rag/retriever.py` — query ChromaDB, optional rerank + hybrid BM25
- `rag/bm25.py` — custom BM25 on `data/documents.json`
- `rag/pipeline.py` — full RAG flow: retrieve → build prompt → LLM generate
- `rag/prompts.py` — system prompt template for PG game assistant
- `rag/llm.py` — HTTP client for llama.cpp LLM on :8080
- `scripts/` — interactive tools: `rag.py` (Q&A), `retrieval.py` (Chroma search), `embedding.py` (inspect vectors), `similarity.py` (cosine similarity)
- `check_services.py` — service status checker for mise status command
- `pg_rag.py` — Open WebUI Pipe Function (imports from this project)

## External services

| Service | Port | Required for |
|---------|------|-------------|
| llama.cpp embedding | :8081 | `embed_batch`, `embed_text`, Chroma similarity search |
| llama.cpp LLM | :8080 | RAG Q&A (`scripts.rag`) |
| Open WebUI | :3000 | Web interface for LLM interaction |

LLM and embedding servers not needed for build/refresh — only for interactive RAG queries. Open WebUI requires Python 3.11 (separate from main project's Python ≥3.14).

## Data directory

`data/` is gitignored, not tracked. Contents:
- `data/cdn/*.json` — game CDN exports (27 tables)
- `data/wiki/*.txt` — wiki page dumps (originally from `wiki.projectgorgon.com`)
- `data/documents.json` — built by `main.py`
- `data/chroma/` — ChromaDB persistent store
- `data/wiki/.meta.json` — touched timestamps for freshness (auto-managed)

## Logs directory

`logs/` is gitignored, not tracked. Contains service output from background processes:
- `logs/embed.log` / `logs/embed-error.log` — Embedding server output
- `logs/llm.log` / `logs/llm-error.log` — LLM server output
- `logs/webui.log` / `logs/webui-error.log` — Open WebUI output
- `logs/embed-build.log` / `logs/embed-build-error.log` — Embedding build mode output

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

## Open WebUI integration

Pipe function bridges the custom RAG pipeline into Open WebUI's chat interface.

- **Pipe file:** `pg_rag.py` (repo root)
- **Setup:** Import via Admin Panel > Functions > Import > select `pg_rag.py`
- **Rename:** After import, edit the function name to "PG RAG" in the admin panel
- **Usage:** Select "PG RAG" from model dropdown, ask game questions
- **Config (Valves):** TOP_K (default 3), USE_HYBRID (default true), USE_RERANK (default true)
- **How it works:** Extracts user query → `retrieve()` (hybrid BM25 + Chroma + rerank) → `build_prompt()` → `generate()` via LLM on :8080 → answer + source citations

**Connecting the LLM directly (no RAG):**
The LLM server (:8080) auto-registers via `OPENAI_BASE_URL` in launch.ps1. To rename the model:
1. Admin Panel > Workspace > Models
2. Create new model > set name to "Gemma 4 26B"
3. Set base model to the auto-detected gemma model ID

Prerequisites: embedding server (:8081) and LLM server (:8080) must be running.

## Additional gitignored items

- `.venv-openwebui/` — Separate Python 3.11 venv for Open WebUI
- `logs/` — Service output logs from background processes
