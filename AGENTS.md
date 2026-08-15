# pg-rag-builder — agent guide (compressed)

## mise
`mise tasks` = full list

## pipeline
`uv run pgrag download-cdn → download-wiki → build-documents → build-index` (entry: `src/pgrag/cli.py`)

## layout
- `src/pgrag/` — importable package (uv installs it editable):
  - `cli.py` — `pgrag` CLI: download-cdn/wiki, build-documents, build-index, validate
  - `config.py` — paths (PROJECT_ROOT = repo root), EMBEDDING_DIM, CONTEXT_BUDGET
  - `build.py` — `generate_documents()` orchestration (CDN+wiki → documents.json)
  - `loaders/`: disk→DB — `cdn_loader.py`, `wiki_loader.py`, `download_cdn.py`, `download_wiki.py` (wiki sync), `database.py` (`GameDatabase` = `tables`(CDN) + `wiki`(text) handoff bag)
  - `documents/`: `builder.py` CDN, `wiki_builder.py` wiki→sections, `chunking.py` 1024c/100ov (lorebook+skillprofile 2048, summary+curated 8192), `resolver.py` xrefs, `skill_profiles.py`, `summaries.py`
  - `embeddings/llama_embeddings.py` → :8081
  - `vectorstore/`: `build_index.py` incremental, `hashes.py`, `health_check.py`
  - `rag/`: `retriever.py` chroma+rerank+BM25fuse, `reranker_client.py` :8082+stats, `bm25.py`, `query_classifier.py` entity/comparison/general, `spelling.py`, `entity_retrieval.py` dossier, `synthesis_detector.py`+`synthesis_generator.py`, `pipeline.py` entity+gap-fill, `prompts.py`, `llm.py` :8080
- `scripts/`: `rag.py`, `retrieval.py`, `curator.py`+`curator_scheduler.py`→`data/wiki/curated/`, `golden_check.py`→`data/golden/`, `embed_eval.py`+`embed_vram_probe.py`+`bakeoff_corpus.py`, `check_services.py` status, `pg_rag.py` OpenWebUI pipe
- `tests/` — pytest, imports `pgrag` (installed package)

## services
svc,port,note
embed mxbai-xsmall-Q8,8081,embed_text/batch retrieval
LLM gemma-4-26B,8080,RAG Q&A
rerank bge-reranker-v2-m3,8082,opt; lexical fallback if down
OpenWebUI,3000,../mywebui

build/refresh needs no servers. `CONTEXT_BUDGET=24000` (config.py) caps entity ctx.

## data
`data/` gitignored: `cdn/ wiki/ documents.json chroma/ wiki/curated/ golden/ rerank_stats.json wiki/.meta.json` · `logs/`: `embed.log llm.log webui.log rerank.log`

## tests
294 (308 coll −14 slow). All offline; 5 skip golden (need :8080+:8081). Temp-dir integration.

## env
- py≥3.14, `uv`. `hf` global CLI, cache `F:\AI\models\hub\`; GGUF via `-hf org/repo:quant`

## gotchas
- **wiki cats**: `TARGET_CATEGORIES` flat + `RECURSIVE_CATEGORIES` walk (`Creatures` d2, `Items` d1) — monsters/items only via recursion (drops live there). Subcats bare (no `Category:` prefix).
- **LLM draft model**: OOM if already running → `mise down` first.
- **scripts/pg_rag.py**: hardcodes `PG_ROOT=F:\ProjectGorgon\pg-rag-builder` + `os.chdir()`, adds `PG_ROOT/src` to `sys.path` — update if repo moves.
- **mise.toml `[env]`**: `WEBUI_DIR`, `LOGS_DIR` — update if paths move.
