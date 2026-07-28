# Project Gorgon RAG Builder — cavekit spec

## §G — Goal

Local RAG assistant for Project Gorgon game data. Ingest CDN/wiki exports → embed → ChromaDB. Retrieve context → local LLM answers player questions. Survive frequent data updates w/o full rebuild.

## §C — Constraints

C1. Local only — ⊥ cloud APIs for embedding or LLM
C2. Embedding server: llama.cpp @ `:8081`
C3. LLM server: llama.cpp @ `:8080` (OpenAI-compat `/v1/chat/completions`)
C4. Vector store: ChromaDB PersistentClient @ `data/chroma`
C5. Incremental indexing — ⊥ full rebuild on data update
C6. Python ≥3.14. Deps: chromadb, requests, mwparserfromhell
C7. OpenCode superpowers/agent config — `.agents/`, `skills-lock.json`, `docs/superpowers/` — ⊥ commit to git. Local-only tooling, not project assets.

## §I — Interfaces

cmd: `uv run python main.py` → writes `data/documents.json`
cmd: `uv run python vectorstore/build_index.py` → build/update ChromaDB index
cmd: `uv run python search.py` → interactive substring search on documents.json (stdio)
cmd: `uv run python -m tests.test_rag` → interactive RAG Q&A (stdio)
api: POST `:8081/embedding` → embedding vector (llama.cpp)
api: POST `:8080/v1/chat/completions` → LLM response (llama.cpp)
db: `data/chroma` → ChromaDB persistent store
db: `data/documents.json` → generated JSON corpus
cfg: `config.py` → DATA_DIR, CDN_DIR, WIKI_DIR, OUTPUT_DIR, KNOWLEDGE_DIR
coll: `project_gorgon` → ChromaDB collection name
cmd: `uv run python -m tests.test_retrieval` → interactive Chroma similarity search (stdio)
cmd: `uv run python -m tests.test_embedding` → manual embedding vector inspect (stdio)
cmd: `uv run python -m tests.test_similarity` → manual cosine similarity print (stdio)
cmd: `uv run python vectorstore/health_check.py` → validate existing ChromaDB index, exit 0 if OK, exit 1 with report on issues

## §V — Invariants

V1: Pipeline order: load CDN → build documents → index vectors. ⊥ index before documents exist.
V2: Chroma path ≡ `data/chroma`. ∀ reference uses `PersistentClient(path="data/chroma")`. Old path `vectorstore/chroma` gone.
V3: Embedding hash ⊥ include metadata. `embedding_hash()` = sha256(`{id, text}`). `metadata_hash()` = sha256(metadata). Separate concerns.
V4: Metadata-only change → `collection.update(metadatas=...)` only. Skip re-embed. Implemented in `build_index.py:79-81`.
V5: Embed batch ≤ 1000 docs (`EMBED_BATCH_SIZE`). Chroma upsert batch ≤ 5000 (`BATCH_SIZE`). Tune per embedding server context: `-np 1 -c 32000` for wiki-length texts.
V6: ∀ document: `{id, type, text, metadata}`. Metadata: `{source, table, name?, type, embedding_hash, metadata_hash}`.
V7: Deleted documents purged from ChromaDB each build pass. `collection.delete(ids=deleted_ids)` in `build_index.py:56-57`.
V8: ChromaDB collection name ≡ `project_gorgon`. ∀ reference uses same name. ⊥ drift between build & retrieve.
V9: Embedding response ∀ entry: shape `{embedding: [[float]]}`, embed vec len = known dimension. ⊥ pass unvalidated response to ChromaDB upsert.
V10: Collection embedding dimension stable ∀ build passes. On build start: query existing Dim, assert match, abort on mismatch.
V11: Document before hash: `id` ∈ keys ∧ `text` ∈ keys. ⊥ pass malformed doc to hash — ⊥ partial index state on crash.
V12: Embedding vector validated before ChromaDB upsert. ∀ vector: non-empty, list of float, length matches expected dimension. ⊥ pass untyped/empty/mismatched vector.
V13: On build start: query existing collection embedding dimension, assert match against expected, abort on mismatch.
V14: Full build pipeline validated via integration test against a temp ChromaDB — verify docs upserted, dims match, deleted docs purged, metadata-only updates skip re-embed. ⊥ changes to build_index without integration test.
V15: Health-check/audit command exists to validate existing ChromaDB index outside the build process. Check: embedding dim matches config, doc count vs expected, no orphaned metadata.
V16: Health-check verifies hash integrity of every indexed doc — embedding_hash(id+text) and metadata_hash(metadata sans type/hash fields) compared against stored values. ⊥ corrupted hash passes health-check.

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

## §B — Bugs

| id | date | cause | fix |
|----|------|-------|-----|
| B1 | 2026-07-26 | Duplicate `elif "ItemKeys"` at `documents/builder.py:77` — same condition as line 69 → dead code. Second branch intended for keyword group ingredients | T1 |
| B2 | 2026-07-26 | SPEC.md out of date — describes hash split as "future improvement." Code in `vectorstore/hashes.py` already implements separate `embedding_hash()`/`metadata_hash()` | T8 |
| B3 | 2026-07-27 | Embedding response shape not validated before upsert. `llama_embeddings.py:24` trusts `item["embedding"][0]` blindly — silent garbage vector corrupts index | V12 |
| B4 | 2026-07-27 | No ChromaDB dimension check at build start. Model config change → build succeeds with mismatched dim, corrupting index without abort | V13 |
