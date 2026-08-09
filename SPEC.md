# Project Gorgon RAG Builder — cavekit spec

## §G — Goal

Local RAG for Project Gorgon game data. CDN/wiki → embed → ChromaDB → retrieve → local LLM answers. Survive frequent data updates, ⊥ full rebuild.

**Self-improvement:** System detects scattered wiki knowledge, synthesizes curated docs, auto-indexes. Open WebUI interface → dig, summarize, self-improve.

**Anti-pattern:** ⊥ "I do not know the specific names of the level 30 areas, but..." When info exists scattered across wiki, synthesize → specific answers (e.g. "Eltibule: 20-30, Serbule Hills: 18-24").

**Skill profiles:** Per-skill knowledge docs from CDN cross-refs (abilities, advancement, XP table, recipes, quests, trainers). New skills (Dungcrafting) answer from raw table data, ⊥ wiki dependency.

**Dossier parity:** Entity questions ("tell me everything about X") match manual full-source quality — whole hub docs + cross-ref facets, ⊥ top-N fragments. General queries count=20. LLM context 16K.

**Cross-encoder rerank:** retrieval precision via llama.cpp reranker :8082 — BAAI/bge-reranker-v2-m3 Q4_K_M (2026-08-08 sweep), failure-surfaced fallback, mise-managed, golden-verified.

**Embedder swap:** pending subset-eval sweep — jina v5-text-small-retrieval (current) vs 6 candidates (22M-335M), VRAM pre-filter + golden gate (V65).

## §C — Constraints

C1. Local only — ⊥ cloud APIs
C2. Embedding: llama.cpp @ `:8081`
C3. LLM: llama.cpp @ `:8080` (OpenAI-compat `/v1/chat/completions`)
C4. Vector store: ChromaDB PersistentClient @ `data/chroma`
C5. Incremental indexing — ⊥ full rebuild on update
C6. Python ≥3.14. Deps: chromadb, requests, mwparserfromhell
C7. OpenCode config (`.agents/`, `skills-lock.json`, `docs/superpowers/`) ⊥ commit. Local-only.
C8. Latency flexible — multi-pass + re-answer OK. ⊥ latency-first.
C9. LLM context 16384 tokens (`-c 16384`) — fits CONTEXT_BUDGET (≤5.5K tok) + output (≤8192) + headroom; KV cache f16 (⊥ q8_0 — breaks MTP acceptance, B27). ⊥ 8K context.
C10. Quality verified vs golden dossiers — 5 answers, fact-list assertions. ⊥ eyeball-only.
C11. Reranker: llama.cpp @ `:8082`, model `BAAI/bge-reranker-v2-m3` Q4_K_M GGUF (gpustack conv, llama.cpp rank pooling; chosen 2026-08-08 sweep: 5/5 golden vs Qwen3-0.6B best 4/5). Winner eval: `-c 32768` (ctx trim = no-op for bge-class KV). Flags `--rerank --pooling rank --embedding`. ⊥ non-llama.cpp serving.
C12. Out of scope: GraphRAG, golden-set expansion, hybrid/BM25/classifier logic change. Embedding swap in-scope — gate V65 (subset eval + golden + VRAM ≥ jina before swap). ⊥ swap-on-hunch.
C13. Reranker failure ⊥ silent — log + flag + persistent stats. Fallback = current lexical order, ⊥ raise.

## §I — Interfaces

cmd: `uv run python download_cdn.py` → fetch CDN JSON → `data/cdn/`
cmd: `uv run python download_wiki.py` → fetch wiki text → `data/wiki/`
cmd: `uv run python main.py` → writes `data/documents.json`
cmd: `uv run python vectorstore/build_index.py` → build/update ChromaDB index
cmd: `uv run python search.py` → interactive substring search (stdio)
cmd: `uv run python -m scripts.rag` → interactive RAG Q&A (stdio)
cmd: `uv run python -m scripts.retrieval` → interactive Chroma similarity search (stdio)
cmd: `uv run python -m scripts.embedding` → manual embedding inspect (stdio)
cmd: `uv run python -m scripts.similarity` → manual cosine similarity (stdio)
cmd: `uv run python vectorstore/health_check.py` → validate ChromaDB index (exit 0/1)
cmd: `uv run python check_services.py` → service status (mise status)
cmd: `uv run python -m scripts.curator` → scan wiki → curated docs
cmd: `uv run python -m scripts.curator_scheduler` → curator + change detect + rebuild
cmd: `uv run python -m scripts.golden_check` → RAG vs golden fact report (needs :8080+:8081)
api: POST `:8081/embedding` → embedding vector (llama.cpp)
api: POST `:8080/v1/chat/completions` → LLM response / synthesis (llama.cpp)
db: `data/chroma` → ChromaDB persistent store
db: `data/documents.json` → generated corpus
cfg: `config.py` → DATA_DIR, CDN_DIR, WIKI_DIR, EMBEDDING_DIM, CONTEXT_BUDGET=24000
cfg: `vectorstore/build_index.py` → EMBED_BATCH_SIZE=1000, BATCH_SIZE=5000
coll: `project_gorgon` → collection name
mod: `rag/entity_retrieval.py` → entity path: hub whole load, facet expansion, context pack
data: `data/golden/` → golden answer dossiers (5, fact lists)
api: POST `:8082/v1/rerank` → `{query, documents, top_n}` → `results[].index` + `relevance_score` (llama.cpp)
mod: `rag/reranker_client.py` → rerank client + stats + fallback
data: `data/rerank_stats.json` → `{failures, last_failure, last_success}`
mise: `rerank-start`/`rerank-stop` ps1 → :8082, logs `logs/rerank.log`
cmd: `uv run python scripts/embed_vram_probe.py` → per-candidate VRAM delta → `data/embed_vram.json`

## §R — Research

| id | topic | finding | src |
|----|-------|---------|-----|
| R1 | model pick | Qwen3-Reranker-0.6B — HF text-ranking downloads #3 (2.7M), MTEB-R 65.80 @0.6B vs bge-v2-m3 57.03, Apache-2.0 | huggingface.co/api/models?pipeline_tag=text-ranking&sort=downloads; arXiv 2506.05176 |
| R2 | llama.cpp rerank | native since PR #9510: `--rerank --pooling rank --embedding`, routes `/rerank` `/v1/rerank` `/v1/reranking`; qwen3-reranker needs ≥ b6578 | github.com/ggml-org/llama.cpp#9510; tools/server/README.md |
| R3 | GGUF trap | community Qwen3-Reranker GGUFs broken — missing `cls.output.weight` + `pooling_type=RANK` → scores 4.5e-23 (#16407); use verified conv (Voodisss) | github.com/ggml-org/llama.cpp/issues/16407 |
| R4 | jina v3/v3.5 | LBNL decoder arch + CC-BY-NC-4.0 → ⊥ llama.cpp rerank, blobs commercial | jina.ai/models/jina-reranker-v3.5 |
| R5 | response schema | README documents ⊥ response shape; jina-style `results[].index`/`relevance_score` per API — curl-test before client | tools/server/README.md |
| R6 | bge-reranker-large | ⊥ rerank support anywhere in llama.cpp tests/docs | github.com/ggml-org/llama.cpp#9510 |
| R7 | host binding | llama-server `--host [IP_ADDRESS]` → IPv6 link-local on this box; new servers ! `--host Scamper` (ASCII host) — `[IP_ADDRESS]` ∉ .NET Uri | measured 2026-08-08 |

## §V — Invariants

V1: Pipeline order: download CDN+wiki → load CDN+wiki → build documents → index vectors. ⊥ index before documents. ⊥ build before data.
V2: Chroma path ≡ `data/chroma`. ∀ ref uses `PersistentClient(path="data/chroma")`.
V3: `embedding_hash()` = sha256(`{id, text}`). `metadata_hash()` = sha256(metadata). Separate.
V4: Metadata-only change → `collection.update(metadatas=...)` only. Skip re-embed. `build_index.py:126`.
V5: Embed batch ≤ 1000 (`EMBED_BATCH_SIZE`). Upsert batch ≤ 5000 (`BATCH_SIZE`).
V6: ∀ doc: `{id, type, text, metadata}`. Metadata: `{source, table, name?, type, embedding_hash, metadata_hash}`.
V7: Deleted docs purged each build. `collection.delete(ids=deleted_ids)` @ `build_index.py:68-74`.
V8: Collection name ≡ `project_gorgon`. ∀ ref same name. ⊥ drift build vs retrieve.
V9: Embedding response ∀ entry: `{embedding: [[float]]}`, vec len = known dim. ⊥ unvalidated upsert.
V10: Collection dim stable ∀ builds. Build start: query existing dim, assert match, abort on mismatch.
V11: Doc before hash: `id` ∈ keys ∧ `text` ∈ keys. ⊥ malformed doc → hash.
V12: Embedding vector validated before upsert. ∀ vec: non-empty, list of float, length = expected dim.
V14: Full build validated via integration test (temp ChromaDB): docs upserted, dims match, deleted purged, metadata-only skips re-embed.
V15: Health-check exists: validate dim, doc count, orphaned metadata, missing docs, hash integrity.
V16: Health-check verifies hash integrity ∀ indexed doc — embedding_hash(id+text) and metadata_hash(metadata sans hash fields; type ∈ hash) vs stored.
V17: documents.json write atomic — temp + `os.replace()`. ⊥ partial/corrupt.
V18: Wiki download purge .txt for pages gone from category. ⊥ stale wiki in index.
V19: Comparison queries (highest/lowest/best/most) → adaptive retrieval count=20. Summary docs in ChromaDB. Prompt includes comparison reasoning.
V20: Curated docs in `data/wiki/curated/` with `_curated` suffix. Auto-generated from wiki synthesis. Rebuilt during index pass.
V21: Synthesis: pipeline detects scattered answers (multiple sources, low scores) → LLM synthesis → curated summary docs.
V22: Background curator scans wiki for fragmented knowledge → curated docs, rebuilds index. ⊥ periodic scheduling.
V23: `EMBEDDING_DIM` in config.py. Build start: existing dim ≠ EMBEDDING_DIM → abort. Health-check: dim vs EMBEDDING_DIM.
V24: Synthesis persisted — writes `data/wiki/curated/synthesized_*_curated.txt`, loaded by builder (V20), survives purge (V7). ⊥ inline-only, ⊥ Chroma upsert (V7 purges ∉ documents.json), ⊥ dead `create_curated_doc()`.
V25: Curator rebuild: curated file write → main.py → build_index. ⊥ build_index direct after curator.
V26: Builder ∀ nested CDN access — isinstance guard before `.get()`/`int()`. ⊥ crash mid-build. ⊥ index-based doc ids. Known gaps: `build_item_documents`, `build_recipe_documents` lack top-level guard; inner loops (ingredients, results, objectives) also unguarded.
V27: ∀ doc → {id,type,text,metadata}; type ∈ known set; source ∈ {cdn,wiki,computed,curated}.
V28: cdn text → ⊥ `{{` `}}` `[[` `]]` `{|` residue, `:\s*None` leak, raw `item_\d+` id. wiki text → ⊥ `{{` `}}` only. curated → ⊥ nothing (markup intended). Names via GameResolver.
V29: ∀ doc id unique. 2× `_assemble_documents` → identical id→text map (pure fn). ⊥ duplicate/nondeterministic.
V30: ∀ split chunk → normalized text ⊆ normalized parent; chunk id unique; chunk_index/chunk_count present; unchunked passthrough.
V31: ∀ itemuse → key ∈ items table keys; recipe count = len(`RecipesThatUseItem`); name via GameResolver.
V32: ∀ skill → `skillprofile_<key>` doc. Sections: description, type, parents, rewards, abilities (Skill==key), advancement (advtable `<num>_<key>`), XP table (XpTable→xptables), recipes (Skill/RewardSkill==key, sorted SkillLevelReq, cap 25), quests (Rewards SkillXp or MinSkillLevel, cap 25), trainers (NPC Services Training). ∅ sections omitted.
V33: Health-check ∀ `collection.get()` paginated `BATCH_SIZE=5000`. ⊥ unbounded → SQLite var limit crash.
V34: Entity query → hub docs loaded whole, subject to CONTEXT_BUDGET cap — ∀ chunks (chunk_index order) into context. Exceeds budget → silent truncation of tail chunks. ⊥ top-N similarity cuts hub doc.
V35: Entity context: hub whole + facet sub-queries (parallel, type-filtered `where`, top-3) + dedupe + pack to `CONTEXT_BUDGET`, hub-first.
V36: Gap-fill: answer self-reports missing (regex) → targeted retrieval → append → re-answer. Max 1 pass.
V37: General query → count=20 requested (may be fewer if collection small).
V38: Golden dossiers — `data/golden/` 5 entries (3 entity + 2 general), each fact list. `golden_check.py` asserts every fact ∈ answer, exit nonzero on miss. Servers down → skip.
V39: Entity hub-miss → general path fallback.
V40: Entity path bypasses synthesis.
V41: Gap-fill subject: regex capture from missing-phrase; empty → question as sub-query.
V42: Golden fact = normalized variant list (≥1 match). ⊥ single brittle substring.
V43: Wiki content fetch batches ≤50 titles/request (`action=query&prop=revisions&rvprop=content&rvslots=main`); changed pages only. Goal: minimize API calls, reduce sync time, respect wiki API rate limits. Function must enforce ≤50 internally; current caller sends single title → violates goal.
V44: ∀ wiki API (including skip pacing) → serial, delay ≥ `BASE_DELAY=0.5s`, `maxlag=5`, retry exp backoff ≤ `MAX_RETRIES=5`, honor `Retry-After`.
V45: Timestamp response complete iff every title appears as page or explicit `missing`; absent → abort before content/purge. Timestamp `missing` (pageid ≤ 0) → delete local `.txt` + meta tombstone, skip download; absent (title ∉ response) → abort. ⊥ abort on benign missing.
V46: Content batch writes pages atomically via temp+`os.replace()`; persist metadata per iteration. Stale purge after complete sync. ⊥ partial run deletes data.
V47: ∀ wiki transport: one `requests` session, headers available for `Retry-After`. ⊥ separate mwclient path.
V48: Category/inventory failure → abort before timestamps/content/purge.
V49: `missing`/redirect → no content write; remove old `.txt` after successful sync. ⊥ retain stale forever. Redirect detection currently absent — redirects saved as content.
V50: Content fetch failure → try/except → abort sync with exit code 1. ⊥ unhandled crash.
V51: Title→filename stable across runs, persisted; cleanup uses persisted mapping. ⊥ enumeration index.
V52: Category enumeration: same session, UA, pacing, maxlag, retry, Retry-After, continuation.
V53: Title→filename detects collisions; deterministic unique suffix or abort.
V54: `.meta.json` schema: `{pages:{<title>:{touched,filename}}}`; atomic write via temp+`os.replace()`.
V55: Filename collision suffix deterministic from title; ⊥ enumeration index.
V56: Tests hitting real CDN/wiki data tagged `@pytest.mark.slow`. Default `pytest` skips slow. `pytest --runslow` or `pytest -m slow` runs them. ⊥ slow tests block fast feedback.
V57: CDN text flatten — stable field order + canonical fields only. Derived/volatile/server-state excluded → refetch same source → identical text → `embedding_hash` stable → build skips re-embed. Field value change (e.g. XP) → that doc re-embeds; siblings skip. ⊥ unstable text → mass re-embed each refresh.
V58: cdn item/recipe text — flat translations only, ⊥ raw `{TOKEN}` template/brace residue. Effect tokens resolved via `attributes.json` `Label`. Item: `Slot: <slot>` + `Stat: <Label> +<value>` lines. Recipe: `Awards <Skill> XP` line `+<XP> <RewardSkill> XP` + first-time when present. ⊥ `{MAX_ARMOR}` in doc text.
V59: Wiki recursion — `RECURSIVE_CATEGORIES` root walk: ns14 subcats queued BARE (⊥ `Category:` twice → 0-member crawl), dedup by category/page, depth cap per root; flat `TARGET_CATEGORIES` unchanged.
V60: Rerank via :8082 cross-encoder only; error/timeout → fallback, ⊥ raising.
V61: Fallback visible — `[WARN]` log + `results["rerank_used"]` flag + `data/rerank_stats.json` counter (atomic). ⊥ silent fallback.
V62: Rerank reorders input set, preserves id/text/metadata pairing, returns requested count.
V63: Golden fact-miss count post-swap ≤ pre-swap (golden_check exit). ⊥ retrieval regression.
V64: Rerank stats: best-effort write (failure ⊥ retrieval), atomic temp+`os.replace`, concurrency loss tolerated.
V65: Embedder swap gate — candidate ! win subset eval + golden facts + VRAM before swap; swap → wipe `data/chroma` + full rebuild (embedding_hash = text-only, T84 trap). ⊥ incremental swap.

## §T — Tasks

| id | status | task | cites |
|----|--------|------|-------|
| T1 | x | Fix dead code: merge duplicate `elif "ItemKeys"` in `documents/builder.py:69` & `:77` | B1 |
| T2 | x | Wiki loader — fill `loaders/wiki_loader.py` stub | I.wiki_loader |
| T3 | x | Connectivity error handling — fail visible :8081/:8080 (⊥ silent crash) | C2,C3 |
| T4 | x | Metadata filtering — `rag/retriever.py` filter by table/source | V6 |
| T5 | x | Citation display — name + source table, ⊥ raw `item_96 {...}` | |
| T6 | x | Remove orphan `processors/document_builder.py` | |
| T7 | x | Test framework (pytest) + pipeline/hash/retrieval tests | V1-V11 |
| T8 | x | SPEC update — hash split already implemented | B2 |
| T9 | x | Reranking after Chroma search — improve top-k relevance | V6 |
| T10 | x | Hybrid search (dense + BM25) | V6 |
| T11 | x | Chunking improvement — split large item/recipe text | V6 |
| T12 | x | Wiki docs — mwparserfromhell parse, section-split, V6 shape | V6, I.wiki_loader |
| T13 | x | Embedding validation before upsert — non-empty, list float, consistent length | V12, B3 |
| T14 | x | ChromaDB dim check at build start — abort on mismatch | V10, V23, B4 |
| T15.1 | x | Integration test: build_index temp ChromaDB, docs + dim | V14 |
| T15.2 | x | Integration test: deleted doc purged after build | V14 |
| T15.3 | x | Integration test: metadata-only uses update, not re-embed | V14 |
| T16 | x | Health-check script — dim, doc count, orphaned metadata | V15 |
| T17 | x | CDN download — fetch game data from `cdn.projectgorgon.com` | C5 |
| T18 | x | Atomic documents.json — temp + os.replace() | V17, B5 |
| T19 | x | Prune stale wiki .txt — delete removed pages | V18, B6 |
| T20 | x | Move resolver.py: processors/ → documents/ | |
| T21 | x | Delete dead `vectorstore/check_index.py` | |
| T22 | x | Move interactive scripts tests/ → scripts/ | |
| T23 | x | Chunking: overlap 100, sentence split, type-aware limits (item/recipe 512, wiki 1024) | V6 |
| T24 | x | Query classifier — comparison patterns in `rag/query_classifier.py` | V19 |
| T25 | x | Pre-computed summaries — recipes by skill, rank by level | V19 |
| T26 | x | Integrate summaries into builder | V19, T25 |
| T27 | x | Adaptive retrieval — count=20 for comparisons | V19, T24 |
| T28 | x | Comparison reasoning prompt | V19, T24 |
| T29 | x | Pipeline wiring — `classify_query()` | V19, T24, T27, T28 |
| T30 | x | Open WebUI pipe — pass `query_type` | V19, T29 |
| T31 | x | Curated template — `data/wiki/curated/`, `_curated` suffix | V20 |
| T32 | x | Area-levels curated doc | V20, T31 |
| T33 | x | Skill-trainers curated doc | V20, T31 |
| T34 | x | Crafting-progressions curated doc | V20, T31 |
| T35 | x | Load curated docs into documents.json | V20, T31 |
| T36 | x | Synthesis detector — `rag/synthesis_detector.py` | V21 |
| T37 | x | Synthesis generator — `rag/synthesis_generator.py` | V21, T36 |
| T38 | x | Synthesis into pipeline — trigger on scattered, persist via curated txt | V21, T36, T37, V24 |
| T39 | x | Curator agent — `scripts/curator.py` | V22 |
| T40 | x | Curator scheduler — change detect + rebuild | V22, T39 |
| T41 | x | Curated doc loading test | V20, T35 |
| T42 | x | Synthesis flow test | V21, T38 |
| T43 | x | Open WebUI synthesis status | V21, T38 |
| T44 | x | Persist synthesis — `synthesized_*_curated.txt` | V24, T38 |
| T45 | x | Fix curator rebuild — main.py before build_index | V25, T40 |
| T46 | x | EMBEDDING_DIM in config.py + build start assert + health-check | V23 |
| T47 | x | Gathering summaries — items by skill level (CDN + wiki) | V19, T25 |
| T48 | x | Curator refresh — regenerate stale, ⊥ `if not exists` skip | V20, V25 |
| T49 | x | CDN table builders: skill, quest, ability, npc, effect, lorebook, directedgoal, area, itemuse, landmark, title, vault | V6 |
| T50 | x | CDN table builders: advancementtable, ai, attribute, source, tsys, xptable, abilitykeyword | V6 |
| T51 | x | Type-aware chunk limits in `documents/chunking.py` | V6, T23 |
| T52 | x | Quality suite `tests/test_doc_quality.py`: shape+hygiene ∀ 23 builders + real sweep | V27, V28 |
| T53 | x | Determinism + id dedup tests — 2× build → identical, ids unique | V29 |
| T54 | x | Chunk↔doc integration — text ⊆ parent, unique ids, chunk_index/chunk_count | V30 |
| T55 | x | Cross-source consistency — itemuse counts, name resolution, skill rewards | V31 |
| T56 | x | Skill profile builder — `documents/skill_profiles.py` | V32 |
| T57 | x | Wire profile builder into `_assemble_documents` | V32, T56 |
| T58 | x | Health-check pagination — batch `collection.get()` @ 5000 | V33, B17 |
| T59 | x | Rebuild index — clear 132-doc drift + 37 orphaned; DIM 512→1024 | B18, V1 |
| T60 | x | Profile tests — cross-refs, recipe cap 25, advtable, xptable, determinism | V32, V29 |
| T61 | x | Health-check pagination test — >5k docs | V33 |
| T62 | x | Entity detection — regex + name lookup → hub key | V34 |
| T63 | x | Entity retrieval — hub whole + facet expansion + dedupe + pack | V34, V35 |
| T64 | x | Pipeline wiring — entity path + gap-fill (max 1 pass) | V36, V40, V41 |
| T65 | x | General path TOP_K 20 | V37 |
| T66 | x | LLM context 16384 — mise.toml `-c 16384` + CONTEXT_BUDGET | C9 |
| T67 | x | Golden harness — 5 dossiers + `golden_check.py` + pytest wrapper | V38, V42 |
| T68 | x | Wiki API sync: one session, continuation, ≤50 batches, pacing, retry, stable filename, atomic metadata, abort on failure; add `tests/test_download_wiki.py` | V18, V43-V55, B22, B23 |
| T69 | x | Test split: add `slow` marker to pyproject.toml, tag `test_doc_quality` real-data tests + `test_skill_profiles` real-data test, default skip slow | V56 |
| T70 | x | Flatten recipe reward fields into recipe text: RewardSkill, RewardSkillXp, RewardSkillXpFirstTime, RewardSkillXpDropOffLevel/Pct/Rate, ResetTimeInSeconds, MaxUses. Fused XP line `Awards <Skill> XP: +<Xp>, first-time +<FirstTime>`; ⊥ dup `Skill` line when RewardSkill==Skill | V57, V58, B24 |
| T71 | x | Flatten item combat/equip fields into item text: EquipSlot, SkillReqs, EffectDescs (tokens → attributes.json Label, `Slot:` + `Stat: <Label> +<value>`), BestowRecipes, CraftingTargetLevel, FoodDesc, TSysProfile | V57, V58, B25 |
| T72 | x | Churn test — flatten-only change → hash stability invariant: identical source → identical text; single field change → 1 re-embed, n-1 skip | V57, V2, V3, V4 |
| T73 | x | Wiki recursion — `RECURSIVE_CATEGORIES` walk (`Creatures` depth 2, `Items` depth 1), subcats queued bare, dedupe + depth limit; tests | V43-V55, V59, B26 |
| T74 | x | Missing-title handle: timestamp `missing` (pageid ≤ 0) → `None`, flow to content phase delete + tombstone, ⊥ abort; absent (truncation) still aborts; tests | V45 |
| T75 | x | KV cache f16 ⊥ q8_0: drop `-ctk q8_0 -ctv q8_0` from llm-start.ps1 + mise.toml debug-llm; C9 updated | C9, B27 |
| T76 | x | Golden baseline — record fact-miss count (0 misses, 2026-08-08) | V38, V63 |
| T77 | x | `rag/reranker_client.py` — curl-verify /v1/rerank shape first (R5), skip call when pool ≤ count, POST, timeout 60s, parse, stats | V60, V61, R2, R5 |
| T78 | x | Wire `_rerank` — cross-encoder when server up, lexical fallback | V60, T77 |
| T79 | x | `data/rerank_stats.json` — best-effort atomic write, failures/last_failure/last_success | V61, V64 |
| T80 | x | mise `rerank-start`/`rerank-stop` ps1 + status shows :8082 + stats failures counter | V60, V61, R2 |
| T81 | x | Fallback tests — server down/error/timeout → lexical order, flag False, stats increment | V60, V61 |
| T83 | x | Rerank memory trims — RESOLVED 2026-08-08: obsoleted by model swap (BAAI/bge-reranker-v2-m3 Q4_K_M). Measured sweep: Qwen3-0.6B Q8_0/-c32768 = 23.3 GB stack, Q8/-c8192 = 21.4 GB (KV cut real), Q4_K_M = 22.1 GB, Q2_K = 3/5 golden (quality floor). bge-m3 + jina-v2 (Q4_K_M) = 19.6-19.8 GB both, 5/5 golden; ctx 8192 = no-op for both (KV ~0.34 GB, ctx-independent pool allocs). Final: bge-m3 Q4_K_M @ -c 32768 → stack 96% -> 82% VRAM (19.8 GB), rerank delta 4.1 GB -> 0.34 GB | V60, T80 |
| T84 | x | Embedder swap eval — RESOLVED 2026-08-08: current embedder is jina-embeddings-v5-text-small-retrieval (677M, 71.7 MTEB-Eng-v2 avg; measured 610 MB Q8_0). Candidate Qwen3-Embedding-0.6B (610 MB Q8_0, 70.7 avg) = size-neutral, slightly weaker on paper; swap would cost full 27k-doc re-embed + instruct-prefix alignment for ~nil gain. KEEP jina. Only bigger gain = Qwen3-Embedding-4B (74.6 avg, +3.5 GB VRAM — requires T83 trims first if ever re-evaluated). Swap note: text-hash unchanged → incremental skips → needs wipe + forced rebuild (no --force flag today). Memory-neutral; retrieval quality TBD via golden eval | V2, V23, T76 |
| T85 | x | Embedder VRAM pre-filter — resolved host resolution for IPv6 link-local (`socket.getaddrinfo` with scope_id) | T83,C11,R7 |
| T86 | x | Embedder subset eval — scripts/embed_eval.py + data/eval_subset.json (1.2k docs, 13 queries, sources-derived labels), MRR@10/hit@3-5/recall@10 vs jina | T85,V65 |
| T87 | . | Sweep runlist — jina-base, MiniLM-L6 22M, mxbai-xsmall 24M⚠, bge-small Q8, embeddinggemma-300m Q8, mxbai-large 335M, KaLM-mini 0.5B; +2 mxbai prefix runs | T86 |
| T88 | . | Decision gate — subset winner → golden facts (V63) → VRAM; swap only if win, then wipe data/chroma + full re-embed | T87,V63,T84 |
| T89 | x | Post-swap golden — RESOLVED 2026-08-08: bge-m3 Q4_K_M @ c32k + c8k both 5/5 (0 miss) | V63 |
| T90 | x | `scripts.rag` prints `rerank_used` — ⊥ `scripts.retrieval` (raw query, no client) | V61 |

## §B — Bugs

| id | date | cause | fix |
|----|------|-------|-----|
| B1 | 2026-07-26 | Duplicate `elif "ItemKeys"` @ `builder.py:77` — dead code | T1 |
| B2 | 2026-07-26 | SPEC outdated — hash split already in `hashes.py` | T8 |
| B3 | 2026-07-27 | Embedding response ⊥ validated. `llama_embeddings.py:24` trusts `item["embedding"][0]` — garbage vector corrupts index | V12 |
| B4 | 2026-07-27 | ⊥ ChromaDB dim check at build start → mismatched dim corrupts silently | V10 |
| B5 | 2026-07-27 | main.py writes documents.json direct — crash → corrupt file | V17, T18 |
| B6 | 2026-07-27 | download_wiki.py ⊥ delete removed page .txt → stale forever | V18, T19 |
| B7 | 2026-07-29 | Scattered wiki → "I do not know" despite data present | V20-V22, T31-T43 |
| B8 | 2026-08-01 | curator_scheduler rebuild skips main.py → curated docs ⊥ indexed | V25 |
| B9 | 2026-08-01 | `create_curated_doc()` dead (0 callers) — synthesis inline ⊥ persist | V24 |
| B10 | 2026-08-01 | V24 "upsert to ChromaDB" conflicts V7 (purge ∉ documents.json) → deleted next build | V24 |
| B11 | 2026-08-01 | `ab_data.get()` unguarded — AI Abilities non-dict → AttributeError | V26 |
| B12 | 2026-08-01 | `int()` on wiki level cell — non-numeric → ValueError kills build | V26 |
| B13 | 2026-08-01 | xptable `amounts[:30]` — real tables ≤125 → levels 31+ dropped | T50 |
| B14 | 2026-08-01 | None leak — `Parents:\nNone` (731 docs), ai Abilities, tsys Slots, xptable, abilitykeyword | V28 |
| B15 | 2026-08-01 | source text `Source: item_1` — raw CDN key leaked (6361 docs) | V28 |
| B16 | 2026-08-01 | wiki_builder template residue — `{{Ambox}}` shell survives strip_code (3/6200) | V28 |
| B17 | 2026-08-01 | health_check `collection.get()` unbounded — 90k ids → SQLite var limit crash | V33 |
| B18 | 2026-08-01 | Index drift — 132 docs ⊄ ChromaDB, 37 orphaned abkeyword ids; B17 hid it | T58, T59 |
| B19 | 2026-08-02 | `EMBEDDING_DIM = 512` — jina-embeddings-v5 returns 1024. V23 guard worked correctly | V23 |
| B20 | 2026-08-02 | Top-N cuts hub docs — chunk_1 scores low vs "what is X" → facets missing → "I do not know" | V34-V37 |
| B21 | 2026-08-02 | Thinking model eats budget — reasoning_content fills max_tokens, zero content. Fix: `--reasoning-budget 1024` + `max_tokens 8192` + retry once | C9, V36 |
| B22 | 2026-08-05 | download_wiki.py metadata in-memory only, saved once at end. Crash → all metadata lost | V46 |
| B23 | 2026-08-05 | download_wiki.py:313 ⊥ try/except — content fetch failure → unhandled crash, entire sync lost | V50 |
| B24 | 2026-08-06 | recipe build drops reward fields — "how much XP does X give" unanswerable | T70 |
| B25 | 2026-08-06 | item build drops equip/combat fields — "what does X give" unanswerable | T71 |
| B26 | 2026-08-07 | `download_wiki.py` recursion queued subcats WITH `Category:` prefix → crawl hit `Category:Category:X`, 0 members → monster/item pages never fetched → "I do not know" on drops | T73 |
| B27 | 2026-08-08 | `-ctk/-ctv q8_0` quantized KV cache breaks Gemma 4 MTP draft acceptance (→0%), e.g. Gemma4 MTP guide: "Quantized KV cache (like Q8_0) breaks acceptance rates (drops to 0%)"; llama.cpp #23636/#23658. Was in llm-start.ps1 + mise.toml | C9 covered |
| B28 | 2026-08-08 | `_persist_synthesized` — question `?` in doc id → Windows-invalid filename `synthesized_...?_curated.txt` → `Errno 22` → synthesis persistence always fails for `?`-ending queries (all general goldens). Fallback path answers OK — silent data loss | pipeline.py sanitize `[<>:"/\\|?*]` -> `-` |
