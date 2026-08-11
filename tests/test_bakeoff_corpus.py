import pytest
from scripts.bakeoff_corpus import pick_fat_docs, family_labels, build_queries, base_key


def make_docs():
    docs = []
    # Fat docs
    for i in range(20):
        docs.append({"id": f"recipe_{i}", "type": "recipe",
                      "text": "x" * (2000 - i * 50),
                      "metadata": {"name": f"Recipe {i}"}})
    # Thin docs
    for i in range(20):
        docs.append({"id": f"item_{i}", "type": "item",
                      "text": "short",
                      "metadata": {"name": f"Item {i}"}})
    # Wiki chunks (same base)
    for i in range(6):
        docs.append({"id": f"wiki_PageA_chunk_{i}", "type": "wiki",
                      "text": "y" * 1500,
                      "metadata": {"name": "Page A"}})
    return docs


def test_pick_fat_docs_descending():
    docs = make_docs()
    picked = pick_fat_docs(docs, 10)
    lengths = [len(d["text"]) for d in picked]
    assert lengths == sorted(lengths, reverse=True)
    assert len(picked) == 10


def test_family_labels_groups_chunks():
    ids = ["wiki_PageA_chunk_0", "wiki_PageA_chunk_1", "recipe_1"]
    assert family_labels(ids, "wiki_PageA_chunk_0") == [
        "wiki_PageA_chunk_0", "wiki_PageA_chunk_1"]


def test_build_queries_has_expected_fields():
    docs = make_docs()
    selected = pick_fat_docs(docs, 25)
    queries = build_queries(selected, docs, n=5)
    assert len(queries) >= 5
    for q in queries:
        assert "id" in q
        assert "text" in q
        assert "expected_doc_ids" in q
        assert len(q["expected_doc_ids"]) > 0


def test_base_key():
    assert base_key("wiki_Page_chunk_3") == "wiki_Page"
    assert base_key("recipe_1") == "recipe_1"
