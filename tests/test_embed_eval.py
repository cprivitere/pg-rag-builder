import pytest

from scripts.embed_eval import (
    _type_budgets,
    base_key,
    gen_subset,
    metrics,
    truncate,
    with_deltas,
)

TYPES = ["wiki", "effect", "item", "source", "tsys", "recipe", "ability", "quest",
         "advancementtable", "attribute", "skillprofile", "itemuse", "title",
         "skill", "ai", "curated"]


def make_docs(per=100):
    docs = []
    for t in TYPES:
        for i in range(per):
            if t == "wiki":
                did = f"wiki_Page_{i % 7}_chunk_{i // 7}"
            else:
                did = f"{t}_{i}"
            docs.append({"id": did, "type": t, "text": f"text of {did} " * 20})
    return docs


def test_budgets_sum_to_target():
    counts = {t: 100 for t in TYPES}
    budgets = _type_budgets(counts, 1200)
    assert sum(budgets.values()) == 1200


def test_subset_deterministic():
    docs = make_docs()
    g1 = gen_subset(docs)
    g2 = gen_subset(docs)
    assert g1 == g2


def test_subset_shape():
    g = gen_subset(make_docs())
    ids = [d["id"] for d in g["docs"]]
    assert len(ids) == 1200
    assert len(set(ids)) == 1200
    assert len(g["queries"]) == 13
    id_set = set(ids)
    for q in g["queries"]:
        assert q["q"].strip()
        assert q["relevant"], q
        assert set(q["relevant"]) <= id_set


def test_truncate_limit():
    long = ["x" * 600]
    assert len(truncate(long)[0]) == 512
    assert truncate(["short"]) == ["short"]


def test_metrics_math():
    docs = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    queries = [
        [1.0, 0.8, 0.0, 0.0],
        [0.95, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.5],
    ]
    labels = [[1], [2], [2, 3]]
    got = metrics(queries, docs, labels)
    assert got["mrr10"] == pytest.approx(0.6111, abs=1e-4)
    assert got["hit3"] == 1.0
    assert got["hit5"] == 1.0
    assert got["recall10"] == pytest.approx(1.0)


def test_with_deltas():
    base = {"mrr10": 0.6, "hit3": 0.5, "hit5": 0.6, "recall10": 0.7}
    cand = {"mrr10": 0.7, "hit3": 0.4, "hit5": 0.6, "recall10": 0.8}
    out = with_deltas(cand, base)
    assert out["delta_mrr10"] == 0.1
    assert out["delta_hit3"] == -0.1
    assert out["delta_hit5"] == 0.0
    assert out["delta_recall10"] == 0.1


def test_base_key_strips_chunks():
    assert base_key("wiki_Page_chunk_3") == "wiki_Page"
    assert base_key("recipe_1") == "recipe_1"