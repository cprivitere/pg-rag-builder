# pg-rag-builder — agent guide (compressed)

## mise
`mise tasks` = full list

## pipeline
`uv run pgrag download-cdn → download-wiki → build-documents → build-index` (+ `validate` for index health; entry: `src/pgrag/cli.py`). `build-index --source cdn|wiki|computed|curated` = partial rebuild: embeds/purges only that source's docs, leaves the rest untouched. Wiki harvest summaries are tagged `source=wiki`, so `--source wiki` refreshes them too.

## layout
- `src/pgrag/` — importable package (uv installs it editable):
  - `cli.py` — `pgrag` CLI: download-cdn/wiki, build-documents, build-index, validate
  - `config.py` — paths (PROJECT_ROOT = repo root), EMBEDDING_DIM, CONTEXT_BUDGET
  - `build.py` — `generate_documents()` orchestration (CDN+wiki → documents.json)
  - `loaders/`: disk→DB — `cdn_loader.py`, `wiki_loader.py`, `download_cdn.py`, `download_wiki.py` (wiki sync), `database.py` (`GameDatabase` = `tables`(CDN) + `wiki`(text) handoff bag)
  - `documents/`: `builder.py` CDN, `wiki_builder.py` wiki→sections, `chunking.py` 1024c/100ov (lorebook+skillprofile 2048, summary+curated 8192), `resolver.py` xrefs, `skill_profiles.py`, `summaries.py`
  - `embeddings/llama_embeddings.py` → :8081
  - `vectorstore/`: `build_index.py` incremental, `hashes.py`, `health_check.py`
  - `rag/`: `retriever.py` chroma+rerank+BM25fuse, `reranker_client.py` :8082+stats, `bm25.py`, `query_classifier.py` entity/comparison/general (+ leveling-intent → skill entity), `spelling.py`, `entity_retrieval.py` dossier (skill recipes sorted by required level), `synthesis_detector.py`+`synthesis_generator.py`, `pipeline.py` entity+gap-fill (+ streaming `ask_stream` events), `prompts.py` (+ leveling ranking block), `llm.py` :8080 (+ `stream_generate` SSE)
- `scripts/`: `rag.py`, `retrieval.py`, `curator.py`+`curator_scheduler.py`→`data/wiki/curated/`, `golden_check.py`→`data/golden/`, `embed_eval.py`+`embed_vram_probe.py`+`bakeoff_corpus.py`, `check_services.py` status, `pg_rag.py` OpenWebUI pipe, `rag_chat.py` Gradio chat (`mise chat`; `uv run --with gradio`)
- `tests/` — pytest, imports `pgrag` (installed package)

## services
svc,port,note
embed mxbai-xsmall-Q8,8081,embed_text/batch retrieval
LLM gemma-4-26B,8080,RAG Q&A
rerank bge-reranker-v2-m3,8082,opt; lexical fallback if down
OpenWebUI,3000,../mywebui
chat Gradio,7860,`mise chat`; needs embed+LLM up; history in browser localStorage

build/refresh needs no servers. `CONTEXT_BUDGET=34000` (config.py) caps entity ctx.

## data
`data/` gitignored: `cdn/ wiki/ documents.json chroma/ wiki/curated/ golden/ rerank_stats.json wiki/.meta.json curator_state.json` · eval records: `embed_eval_*.log`, `embed_vram.json`, `bakeoff_*.json` · `logs/`: `embed.log llm.log webui.log rerank.log`

## tests
341 offline (360 coll −14 slow). All offline; 5 golden skip unless :8080+:8081 are up (runtime skip + single retry for LLM nondeterminism). Temp-dir integration.

## env
- py≥3.14, `uv`. `hf` global CLI, cache `F:\AI\models\hub\`; GGUF via `-hf org/repo:quant`

## gotchas
- **wiki cats**: `TARGET_CATEGORIES` flat + `RECURSIVE_CATEGORIES` walk (`Creatures` d2, `Items` d1) — monsters/items only via recursion (drops live there); `Quests` is flat-synced. Subcats bare (no `Category:` prefix). Wiki filenames are `{safe_title}_<sha256-8>.txt`; `wiki_loader` names pages via `.meta.json` real titles (fallback: hash stripped) so doc names/sources show clean titles. Sync ends with `remove_orphan_files` — `.txt` files not in meta (leftovers from older enumerations/aborted runs) are deleted.
- **LLM draft model**: OOM if already running → `mise down` first.
- **scripts/pg_rag.py**: hardcodes `PG_ROOT=F:\ProjectGorgon\pg-rag-builder` + `os.chdir()`, adds `PG_ROOT/src` to `sys.path` — update if repo moves.
- **mise.toml `[env]`**: `WEBUI_DIR`, `LOGS_DIR` — update if paths move.
- **test isolation**: `test_download_wiki.py` `main()` tests patch `META_FILE` + `WIKI_DIR` to tmp — they must, or they clobber the real `data/wiki/.meta.json` (was a silent 14k→1 entry destroyer).

## roadmap
- **Agentic retrieval / parent-child chunks (not built)**: pipeline is one-shot — LLM gets a fixed context (entity dossier or top-k) and answers once. No hierarchical chunking (`chunk_index` only, no `parent_id`), no tool-calling, no multi-turn browse. Only second chance is rule-based `_gap_fill` (auto re-retrieve on "I don't know"). If built: (1) record `parent_id` per chunk in chunking/builder, (2) add a resolve endpoint (full parent text or next-level chunks, like `_hub_chunks` but callable by the LLM), (3) make `generate()` tools-aware or add a structured "request more" two-stage loop.
