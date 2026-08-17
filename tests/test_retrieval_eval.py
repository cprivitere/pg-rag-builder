"""Offline retrieval metrics (Recall@k, MRR, NDCG@k, entity resolution).

Pure, deterministic metric functions run against hand-built relevance
scenarios — the deterministic signal for retrieval quality (services-off),
complementing the greedy live golden checks.
"""

import json
import math
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "retrieval_cases.json"


# --- metric definitions (binary relevance) ---


def recall_at_k(ranked, relevant, k):
    """|relevant ∩ top-k| / |relevant| (0 when no relevant ids given)."""
    if not relevant:
        return 0.0
    top = set(ranked[:k])
    return len(top & set(relevant)) / len(relevant)


def reciprocal_rank(ranked, relevant):
    """1/rank of first relevant doc, 0 if none appears in ranked."""
    for i, doc_id in enumerate(ranked, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def mrr(queries):
    """Mean reciprocal rank over (ranked, relevant) pairs."""
    if not queries:
        return 0.0
    return sum(reciprocal_rank(r, rel) for r, rel in queries) / len(queries)


def dcg_at_k(ranked, relevant, k):
    """Σ_{i=1..k} rel_i / log2(i+1), binary relevance; 1-based rank discount."""
    return sum(
        1.0 / math.log2(i + 2)
        for i, doc_id in enumerate(ranked[:k])
        if doc_id in relevant
    )


def ndcg_at_k(ranked, relevant, k):
    """DCG@k / IDCG@k; ideal = the min(k, |relevant|) relevant docs at top."""
    ideal = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / idcg


def entity_accuracy(resolved_hubs, expected_hubs):
    """Fraction of queries whose hub resolution matched expectation."""
    if not resolved_hubs:
        return 0.0
    matches = sum(1 for a, b in zip(resolved_hubs, expected_hubs) if a == b)
    return matches / len(resolved_hubs)


# --- deterministic scenarios (no services) ---


def test_exact_metric_values():
    ranked = [["d1", "d3", "d2"], ["d2", "d5", "d4"], ["d1", "d2", "d5"]]
    relevant = [["d1"], ["d2", "d4"], ["d5"]]
    queries = list(zip(ranked, relevant))

    # q1: d1 at rank 1
    assert recall_at_k(ranked[0], relevant[0], 3) == 1.0
    # q2: d2 in top-2, d4 at rank 3 -> Recall@2 = 1/2
    assert recall_at_k(ranked[1], relevant[1], 2) == 0.5
    # q3: d5 at rank 3 -> Recall@3 = 1/1
    assert recall_at_k(ranked[2], relevant[2], 3) == 1.0

    # MRR: (1 + 1 + 1/3) / 3 = 7/9
    assert mrr(queries) == pytest.approx(7 / 9)

    # NDCG@2, q2: DCG = 1/log2(2)=1.0 (d2); IDCG = 1 + 1/log2(3)
    assert ndcg_at_k(ranked[1], relevant[1], 2) == pytest.approx(
        1.0 / (1.0 + 1.0 / math.log2(3))
    )
    # perfect ranking -> NDCG 1
    assert ndcg_at_k(["d2", "d4", "d5"], ["d2", "d4"], 2) == 1.0


def test_recall_mrr_detect_rank_regression():
    """Swapping a relevant doc out of top-1 must change Recall@1 and MRR."""
    good = ["d1", "d2"]
    bad = ["d2", "d1"]  # d1 (the relevant one) dropped from top-1
    relevant = ["d1"]

    assert recall_at_k(good, relevant, 1) == 1.0
    assert recall_at_k(bad, relevant, 1) == 0.0
    assert reciprocal_rank(good, relevant) == 1.0
    assert reciprocal_rank(bad, relevant) == 0.5


def test_entity_accuracy():
    assert entity_accuracy(["Punch", "Front Kick"], ["Punch", "Front Kick"]) == 1.0
    assert entity_accuracy(["Punch", "Punch"], ["Punch", "Front Kick"]) == 0.5
    assert entity_accuracy([], ["Punch"]) == 0.0


# --- fixture conformance (the 5 canonical cases must be present) ---


def test_fixture_has_all_canonical_cases():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    ids = {c["id"] for c in data["cases"]}
    assert {
        "punch-vs-front-kick",
        "grow-field-mushrooms",
        "field-mushroom-locations",
        "mushroom-farming-gathering",
        "spider-silk-recipe",
    } <= ids


@pytest.mark.parametrize("case_id", [
    "punch-vs-front-kick", "grow-field-mushrooms",
    "field-mushroom-locations", "mushroom-farming-gathering",
    "spider-silk-recipe",
])
def test_fixture_case_well_formed(case_id):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    case = next(c for c in data["cases"] if c["id"] == case_id)
    assert isinstance(case["query"], str) and case["query"]
    assert case["classifier"] in {"general", "lookup", "comparison"}
    assert isinstance(case["expected_ranked_ids"], list)
    assert isinstance(case["relevant_ids"], list) and case["relevant_ids"]