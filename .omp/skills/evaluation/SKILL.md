---
name: evaluation
description: The pg-rag RAG evaluation harness — golden set, fact-presence checking, metrics, before/after workflow. Use before and after any retrieval or document-generation change that should be measured.
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

## Metrics (target)

Current harness measures **fact presence (subset recall of stated facts)**.
The forward-looking targets, captured per stage:

- Recall@5 / Recall@10
- MRR, NDCG
- reranker improvement (rerank vs no-rerank on the same query set)
- entity-resolution accuracy

## Before/after workflow

1. Pick a focused query set (or write a new golden for the failing case).
2. Run `golden_check.py` / targeted tests → record baseline.
3. Make the change (one stage).
4. Re-run — compare pass/fail AND inspect which docs ranked (scripts/retrieval.py).
5. If the "fix" only moved a ranking problem downstream, go back to stage
   localization (`retrieval` skill).

## Planned direction (see .opencode/plans/metadata-bm25-eval.md)

- Expand goldens to ~20–30, including named regression cases:
  - `recipes-using-spider-silk` (metadata/keyword search) — fails before
    recipe metadata enrichment, passes after.
  - `grow-field-mushrooms` ("How do I grow Field Mushrooms?") — fails before
    wiki page expansion, passes after.
- Convert to per-stage metrics (retrieval-level, pre-LLM) for ranking-aware
  measurement: `evaluation/` dir with `queries.jsonl`, `expected.jsonl`,
  `results/`.