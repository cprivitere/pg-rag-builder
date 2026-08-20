---
name: evaluation
description: The pg-rag RAG evaluation harness — golden set (fact-presence LLM checks) + pre-LLM IR suite (Recall@k/MRR/NDCG/Hit@k, reranker uplift, entity accuracy), before/after workflow. Use before and after any retrieval or document-generation change that should be measured.
---

# evaluation — RAG eval harness

The point of this repo's evaluation is making agent changes measurable instead
of "this seems better."

## Golden set — `data/golden/*.json`

Each file:

```json
{
  "id": "recipes-using-spider-silk",
  "question": "What recipes can I make with Spider Silk at Nature Appreciation 25?",
  "type": "recipe",
  "facts": [
    ["Spider Silk Tunic", "Spider Silk Robe"],
    ["Nature Appreciation", "25"]
  ]
}
```

- `facts` is a list of variant groups: the answer PASSES if ANY variant in
  each group appears (normalized substring) in the LLM answer.
- 5 files exist today; the planned expansion targets ~20–30 covering
  recipes-by-ingredient, level-gated crafting, item acquisition/drops, ability
  lookups, comparisons, quest requirements, wiki lore, wiki how-to assembly.

## Running it

- `uv run python scripts/golden_check.py` — needs embed (:8081) + LLM (:8080)
  up (`mise start`). Exits 1 with the missing facts on failure.
- `tests/test_golden_check.py` parametrizes over `data/golden/*.json` at
  collection time → new goldens are AUTO-collected as offline-skipped tests
  (skip unless servers up). No test code needed per golden.

## Retrieval eval suite (pre-LLM IR metrics)

- `mise eval-retrieval` (alias `ev`) — full per-stage run (needs embed :8081
  + rerank :8082 up; `build-index` required first). `scripts/retrieval_eval.py`.
- `mise eval-offline` — `--offline` variant: BM25 + query classifier + entity
  hub resolution only, no server deps. Use when you only touched retrieval
  stages that don't need Chroma/rerank.
- **Dataset**: `evaluation/queries.jsonl` — hand-authored gold input (tracked).
  Each line: `id`, `query`, `query_type`, `expected_classifier`,
  `target_entities`, `relevant_ids`, `relevant_names`, `category`.
- **Output**: `evaluation/results/latest.json` (gitignored, generated).
  `--baseline` reruns against a saved snapshot for diff/regression comparison
  (`compare_benchmarks`).
- The offline entity-hub path reuses `build_entity_context_offline`, so it
  measures the same wiki-linked dossier the live pipeline builds.
- Metrics count **canonical retrievable units**, not raw index docs: a wiki
  table's `_coverage` + `_row_<n>` docs and any `_chunk_<n>` split collapse to
  one unit (applied to both relevant and ranked, deduped). Without this the
  per-row explosion deflates recall@k and chunked ranked docs never match base
  relevant ids.
- **Comparison queries** (`query_type == "comparison"`, ≥2 entities) are scored
  on the multi-entity dossier production feeds the LLM
  (`build_multi_entity_context`), not the single-hub rerank window — both
  subjects must be present in the scored context.
- Retrieval surfaces terse-stub lookup docs (e.g. ability/area stubs) by
  **exact entity-name injection + promotion**: the bounded longest multi-token
  span that resolves to a known entity name is injected into the query and the
  matching explicit-id doc is promoted, so a stub that never appears via dense
  embedding still ranks. Fragments that only partially match are excluded.
- Add queries to `evaluation/queries.jsonl` to cover a regression case; prefer
  a new per-stage assertion in `tests/test_retrieval_eval.py` for pure metric
  or well-formedness guarantees.

## Two layers of measurement

1. **LLM end-to-end** — fact presence (subset recall of stated facts) over
   `data/golden/*.json`. Cheapest signal that the answer is right.
2. **Pre-LLM IR suite** (`src/pgrag/rag/retrieval_eval.py`) — per-stage
   ranking metrics over `evaluation/queries.jsonl`, so a regression can be
   pinned to the exact stage (BM25, dense, hybrid RRF, rerank, entity hub)
   instead of "LLM said something wrong":
   - Recall@k, MRR, NDCG@k, Hit@k
   - Reranker Uplift (Δ NDCG@5 rerank vs hybrid)
   - Entity accuracy (Jaccard / strict match on entity linkage)

## Before/after workflow

1. Pick a focused query set (or write a new golden for the failing case).
2. Run `golden_check.py` / targeted tests → record baseline.
3. Make the change (one stage).
4. Re-run — compare pass/fail AND inspect which docs ranked (scripts/retrieval.py).
5. If the "fix" only moved a ranking problem downstream, go back to stage
   localization (`retrieval` skill).

## Planned direction

- Expand goldens to ~20–30, including named regression cases:
  - `recipes-using-spider-silk` (metadata/keyword search) — fails before
    recipe metadata enrichment, passes after.
  - `grow-field-mushrooms` ("How do I grow Field Mushrooms?") — fails before
    wiki page expansion, passes after.
- Extend `evaluation/queries.jsonl` (45 cases today) across the same
  categories, and promote benchmark snapshots into regression thresholds.