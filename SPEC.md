# Project Gorgon RAG Builder — cavekit spec

## §G — Goal

Local RAG assistant for Project Gorgon game data. Ingest CDN/wiki exports → embed → ChromaDB. Retrieve context → local LLM answers player questions. Survive frequent data updates w/o full rebuild.

**Self-improvement capability:** System identifies scattered/fragmented knowledge across wiki pages, synthesizes into curated documents, and auto-indexes. Enables Open WebUI interface to dig through information, create summaries, and improve itself over time.

**Anti-pattern:** System must NOT respond: "I do not know the specific names of the level 30 areas, but the context mentions that players should head to appropriate level 20 areas, then 30, then 40, etc." — When information exists in wiki but is scattered across pages, system should synthesize and provide specific answers (e.g., "Eltibule: 20-30, Serbule Hills: 18-24").

## §C — Constraints

C1. Local only — ⊥ cloud APIs for embedding or LLM
C2. Embedding server: llama.cpp @ `:8081`
C3. LLM server: llama.cpp @ `:8080` (OpenAI-compat `/v1/chat/completions`)
C4. Vector store: ChromaDB PersistentClient @ `data/chroma`
C5. Incremental indexing — ⊥ full rebuild on data update
C6. Python ≥3.14. Deps: chromadb, requests, mwparserfromhell
C7. OpenCode superpowers/agent config — `.agents/`, `skills-lock.json`, `docs/superpowers/` — ⊥ commit to git. Local-only tooling, not project assets.

## §I — Interfaces

cmd: `uv run python download_cdn.py` → fetch latest CDN JSON from `cdn.projectgorgon.com` to `data/cdn/`
cmd: `uv run python download_wiki.py` → fetch wiki page text from `wiki.projectgorgon.com` to `data/wiki/` via mwclient
cmd: `uv run python main.py` → writes `data/documents.json`
cmd: `uv run python vectorstore/build_index.py` → build/update ChromaDB index
cmd: `uv run python search.py` → interactive substring search on documents.json (stdio)
cmd: `uv run python -m scripts.rag` → interactive RAG Q&A (stdio)
api: POST `:8081/embedding` → embedding vector (llama.cpp)
api: POST `:8080/v1/chat/completions` → LLM response (llama.cpp)
db: `data/chroma` → ChromaDB persistent store
db: `data/documents.json` → generated JSON corpus
cfg: `config.py` → DATA_DIR, CDN_DIR, WIKI_DIR
coll: `project_gorgon` → ChromaDB collection name
cmd: `uv run python -m scripts.retrieval` → interactive Chroma similarity search (stdio)
cmd: `uv run python -m scripts.embedding` → manual embedding vector inspect (stdio)
cmd: `uv run python -m scripts.similarity` → manual cosine similarity print (stdio)
cmd: `uv run python vectorstore/health_check.py` → validate existing ChromaDB index, exit 0 if OK, exit 1 with report on issues
cmd: `uv run python check_services.py` → service status checker for mise status command
cmd: `uv run python -m scripts.curator` → background curator agent — scan wiki for scattered info, create curated documents
api: POST `:8080/v1/chat/completions` → LLM synthesis for auto-generating curated documents

## §V — Invariants

V1: Pipeline order: download CDN + wiki → load CDN + wiki → build documents → index vectors. ⊥ index before documents exist. ⊥ build before data downloaded.
V2: Chroma path ≡ `data/chroma`. ∀ reference uses `PersistentClient(path="data/chroma")`. Old path `vectorstore/chroma` gone.
V3: Embedding hash ⊥ include metadata. `embedding_hash()` = sha256(`{id, text}`). `metadata_hash()` = sha256(metadata). Separate concerns.
V4: Metadata-only change → `collection.update(metadatas=...)` only. Skip re-embed. Implemented in `build_index.py:126`.
V5: Embed batch ≤ 1000 docs (`EMBED_BATCH_SIZE`). Chroma upsert batch ≤ 5000 (`BATCH_SIZE`). Tune per embedding server context: `-np 1 -c 32000` for wiki-length texts.
V6: ∀ document: `{id, type, text, metadata}`. Metadata: `{source, table, name?, type, embedding_hash, metadata_hash}`.
V7: Deleted documents purged from ChromaDB each build pass. `collection.delete(ids=deleted_ids)` in `build_index.py:68-74`.
V8: ChromaDB collection name ≡ `project_gorgon`. ∀ reference uses same name. ⊥ drift between build & retrieve.
V9: Embedding response ∀ entry: shape `{embedding: [[float]]}`, embed vec len = known dimension. ⊥ pass unvalidated response to ChromaDB upsert.
V10: Collection embedding dimension stable ∀ build passes. On build start: query existing Dim, assert match, abort on mismatch.
V11: Document before hash: `id` ∈ keys ∧ `text` ∈ keys. ⊥ pass malformed doc to hash — ⊥ partial index state on crash.
V12: Embedding vector validated before ChromaDB upsert. ∀ vector: non-empty, list of float, length matches expected dimension. ⊥ pass untyped/empty/mismatched vector.
V13: On build start: query existing collection embedding dimension, assert match against expected, abort on mismatch.
V14: Full build pipeline validated via integration test against a temp ChromaDB — verify docs upserted, dims match, deleted docs purged, metadata-only updates skip re-embed. ⊥ changes to build_index without integration test.
V15: Health-check/audit command exists to validate existing ChromaDB index outside the build process. Check: embedding dim matches config, doc count vs expected, no orphaned metadata.
V16: Health-check verifies hash integrity of every indexed doc — embedding_hash(id+text) and metadata_hash(metadata sans type/hash fields) compared against stored values. ⊥ corrupted hash passes health-check.
V17: documents.json write atomic — temp file then `os.replace()`. ⊥ partial/corrupt documents.json reaches build.
V18: Wiki download purge .txt files for pages gone from target category. ⊥ stale wiki docs persist in index.
V19: Comparison/aggregation queries (highest/lowest/best/most) trigger adaptive retrieval — count=20 instead of default. Pre-computed summary docs stored in ChromaDB, retrieved semantically. Prompt includes comparison reasoning instructions.
V20: Curated documents stored in `data/wiki/curated/` with `_curated` suffix. Auto-generated from wiki page synthesis. Rebuilt during index pass.
V21: Synthesis capability — pipeline detects scattered answers (multiple sources, low relevance scores) and triggers LLM synthesis to create curated summary documents.
V22: Background curator agent runs periodically, scans wiki for fragmented knowledge (area levels, skill progressions, crafting chains), creates curated documents, rebuilds index.

## §T — Tasks

| id | status | task | cites |
|----|--------|------|-------|
| T1 | x | Fix dead code: merge duplicate `elif "ItemKeys"` branches in `documents/builder.py:69` & `:77` | B1 |
| T2 | x | Implement wiki loader — fill `loaders/wiki_loader.py` stub | I.wiki_loader |
| T3 | x | Add connectivity error handling — fail visible on :8081/:8080 unreachable (⊥ silent crash) | C2,C3 |
| T4 | x | Add metadata filtering to `rag/retriever.py` — filter by table/source to reduce noise | V6 |
| T5 | x | Improve citation display — show human-readable name + source table instead of raw `item_96 {...}` | |
| T6 | x | Remove orphan `processors/document_builder.py` — calls undefined `db` | |
| T7 | x | Add test framework (pytest) + assertion-based tests covering pipeline, hash, retrieval | V1-V11 |
| T8 | x | Update SPEC.md to reflect hash split already implemented (code ahead of spec) | B2 |
| T9 | x | Add reranking step after Chroma similarity search — improve top-k relevance | V6 |
| T10 | x | Add hybrid search (dense + keyword BM25) for broader recall | V6 |
| T11 | x | Improve chunking strategy — split large item/recipe text into smaller chunks | V6 |
| T12 | x | Build wiki documents from `db.wiki` — parse wiki markup via mwparserfromhell, section-split, match V6 metadata shape | V6, I.wiki_loader |
| T13 | x | Validate embedding response shape before upsert — non-empty, list of float, consistent length | V12, B3 |
| T14 | x | Check ChromaDB collection dimension at build start — abort on mismatch | V13, B4 |
| T15.1 | x | Integration test: build_index against temp ChromaDB, verify docs upserted with correct dim | V14 |
| T15.2 | x | Integration test: deleted doc purged from ChromaDB after build | V14 |
| T15.3 | x | Integration test: metadata-only change uses update, not re-embed | V14 |
| T16 | x | Add health-check/audit script to validate existing ChromaDB index — check dim, doc count, orphaned metadata | V15 |
| T17 | x | Add CDN download script — fetch latest game data from `cdn.projectgorgon.com` before build pipeline | C5 |
| T18 | x | Atomic documents.json write — temp file + os.replace() in main.py | V17, B5 |
| T19 | x | Prune stale wiki .txt files — delete files not in current page set after download | V18, B6 |
| T20 | x | Move resolver.py from processors/ to documents/, rm processors/ | |
| T21 | x | Delete dead vectorstore/check_index.py — hardcoded debug script | |
| T22 | x | Move interactive scripts from tests/ to scripts/ — test_rag, test_retrieval, test_embedding, test_similarity | |
| T23 | x | Improve chunking: overlap (100 chars), semantic split at sentence boundaries, type-aware limits (items/recipes 512, wiki 1024), default 1024 | V6 |
| T24 | x | Add query classifier — detect comparison patterns (highest/lowest/best/most) in `rag/query_classifier.py` | V19 |
| T25 | x | Add pre-computed summaries — group recipes by skill, rank by level in `documents/summaries.py` | V19 |
| T26 | x | Integrate summaries into document builder — call `build_summary_documents()` in `documents/builder.py` | V19, T25 |
| T27 | x | Adaptive retrieval — accept `query_type` param, count=20 for comparisons in `rag/retriever.py` | V19, T24 |
| T28 | x | Enhanced prompt — add comparison reasoning instructions in `rag/prompts.py` | V19, T24 |
| T29 | x | Pipeline integration — wire `classify_query()` through `rag/pipeline.py` | V19, T24, T27, T28 |
| T30 | x | Update Open WebUI pipe — pass `query_type` through in `pg_rag.py` | V19, T29 |
| T31 | x | Create curated wiki document template — `data/wiki/curated/` directory, naming convention `_curated` suffix | V20 |
| T32 | x | Create area-levels curated doc — consolidate scattered area/dungeon level info from wiki pages | V20, T31 |
| T33 | x | Create skill-trainers curated doc — consolidate skill trainer locations across wiki | V20, T31 |
| T34 | x | Create crafting-progressions curated doc — consolidate crafting skill progressions | V20, T31 |
| T35 | x | Update build pipeline — load curated docs from `data/wiki/curated/` into documents.json | V20, T31 |
| T36 | x | Add synthesis detector — `rag/synthesis_detector.py` identify scattered answers (multiple sources, low scores) | V21 |
| T37 | x | Add synthesis generator — `rag/synthesis_generator.py` use LLM to create curated docs from retrieved chunks | V21, T36 |
| T38 | x | Integrate synthesis into pipeline — `rag/pipeline.py` trigger synthesis on scattered answers, store in ChromaDB | V21, T36, T37 |
| T39 | x | Add curator agent — `scripts/curator.py` background process scans wiki, identifies fragmented knowledge, creates curated docs | V22 |
| T40 | x | Add curator scheduler — periodic runs (daily/weekly), diff detection, rebuild index after changes | V22, T39 |
| T41 | x | Test curated doc loading — verify curated docs indexed correctly, queries return consolidated answers | V20, T35 |
| T42 | x | Test synthesis flow — verify scattered answers trigger synthesis, new docs improve future queries | V21, T38 |
| T43 | x | Update Open WebUI pipe — surface synthesis status in answers ("synthesized from X sources") | V21, T38 |

## §B — Bugs

| id | date | cause | fix |
|----|------|-------|-----|
| B1 | 2026-07-26 | Duplicate `elif "ItemKeys"` at `documents/builder.py:77` — same condition as line 69 → dead code. Second branch intended for keyword group ingredients | T1 |
| B2 | 2026-07-26 | SPEC.md out of date — describes hash split as "future improvement." Code in `vectorstore/hashes.py` already implements separate `embedding_hash()`/`metadata_hash()` | T8 |
| B3 | 2026-07-27 | Embedding response shape not validated before upsert. `llama_embeddings.py:24` trusts `item["embedding"][0]` blindly — silent garbage vector corrupts index | V12 |
| B4 | 2026-07-27 | No ChromaDB dimension check at build start. Model config change → build succeeds with mismatched dim, corrupting index without abort | V13 |
| B5 | 2026-07-27 | main.py writes documents.json directly — crash mid-write corrupts file, next build_index fails on json.load | V17, T18 |
| B6 | 2026-07-27 | download_wiki.py never deletes .txt for removed/renamed pages → stale documents persist in index forever | V18, T19 |
| B7 | 2026-07-29 | Scattered knowledge across wiki pages causes "I do not know" responses — area-level info exists but fragmented across game updates, dungeon pages, area pages | V20-V22, T31-T43 |
