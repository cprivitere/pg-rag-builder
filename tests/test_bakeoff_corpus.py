"""Tests the scripts.bakeoff_corpus helpers.

Contract (see scripts/bakeoff_corpus.py): the corpus is cluster-sampled — docs
are grouped into entity clusters (same case-normalized metadata.name across the
cross-source tables + multi-chunk page families), whole clusters are selected
(type-stratified, seeded, no table > cap_share), and each query's golds are its
whole cluster (by construction present in the corpus). base_key strips the
_chunk_N suffix. golds are never empty.
"""

from scripts.bakeoff_corpus import (
    base_key,
    build_clusters,
    build_queries,
    pick_clusters,
)


def make_docs():
    docs = []
    # Recipes, each with a unique name.
    for i in range(20):
        docs.append({"id": f"recipe_{i}", "type": "recipe",
                      "text": "x" * (2000 - i * 50),
                      "metadata": {"name": f"Recipe {i}"}})
    # Items: item_3 shares its name with recipe_3 -> cross-source link.
    for i in range(20):
        name = "Recipe 3" if i == 3 else f"Item {i}"
        docs.append({"id": f"item_{i}", "type": "item", "text": "short",
                      "metadata": {"name": name}})
    # Wiki page with multiple chunks sharing one name/base key.
    for i in range(6):
        docs.append({"id": f"wiki_PageA_chunk_{i}", "type": "wiki",
                      "text": "y" * 1500, "metadata": {"name": "Page A"}})
    return docs


def ids(docs):
    return {d["id"] for d in docs}


def test_base_key():
    assert base_key("wiki_Page_chunk_3") == "wiki_Page"
    assert base_key("recipe_1") == "recipe_1"



def test_build_clusters_hybrid_gold():
    """Entity clusters link same-name docs across tables AND multi-chunk pages."""
    docs = make_docs()
    by_id = {d["id"]: d for d in docs}
    clusters, owner = build_clusters(docs, by_id)
    # item_3 + recipe_3 share a name -> one cluster (cross-source link).
    assert owner["recipe_3"] == owner["item_3"]
    # All six Page A chunks share one cluster (page family).
    wiki_cluster = clusters[owner["wiki_PageA_chunk_0"]]
    assert all(f"wiki_PageA_chunk_{i}" in wiki_cluster for i in range(6))


def test_pick_clusters_deterministic_and_bounded():
    docs = make_docs()
    by_id = {d["id"]: d for d in docs}
    clusters, _ = build_clusters(docs, by_id)
    a = pick_clusters(clusters, by_id, 25)
    assert a == pick_clusters(clusters, by_id, 25)          # deterministic (SEED)
    chosen = [i for c in a for i in clusters[c]]
    assert len(chosen) <= 25                                # respects target
    assert len(chosen) == len(set(chosen))                  # no dup docs


def test_build_queries_has_expected_fields_and_golds_in_corpus():
    docs = make_docs()
    by_id = {d["id"]: d for d in docs}
    clusters, _ = build_clusters(docs, by_id)
    chosen_cids = pick_clusters(clusters, by_id, 25)
    queries = build_queries(docs, clusters, chosen_cids, n=5)
    assert len(queries) >= 5
    corpus_ids = ids(docs)  # docs are all pipelines' corpus input in this unit
    for q in queries:
        assert "id" in q and "text" in q
        assert "expected_doc_ids" in q
        assert len(q["expected_doc_ids"]) > 0               # golds never empty
        for g in q["expected_doc_ids"]:
            assert g in corpus_ids                          # gold is retrievable
    # The wiki page cluster yields a multi-gold query.
    assert any(len(q["expected_doc_ids"]) >= 2 for q in queries)