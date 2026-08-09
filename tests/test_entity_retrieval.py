import pytest

from rag import entity_retrieval as er


def _mk(doc_id, text, chunk_index=None, dtype="skill", name="X"):
    meta = {"type": dtype, "name": name, "table": "skills"}
    if chunk_index is not None:
        meta["chunk_index"] = chunk_index
        meta["chunk_count"] = 3
    return {"id": doc_id, "text": text, "metadata": meta}


def _mkdocs():
    return [
        _mk("skillprofile_Pooping_chunk_1", "Reuse 1800s XP Table Quests", 1),
        _mk("skillprofile_Pooping_chunk_0", "Skill Profile: Dungcrafting Abilities", 0),
    ]


def _empty_retrieve(**kwargs):
    return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


@pytest.fixture(autouse=True)
def fake_docs(monkeypatch):
    monkeypatch.setattr(er, "_load_docs", _mkdocs)
    monkeypatch.setattr(er, "retrieve", _empty_retrieve)


def test_hub_whole_loaded_in_order():
    r = er.build_entity_context("what is Dungcrafting", "skillprofile_Pooping")
    assert r["ids"][0] == [
        "skillprofile_Pooping_chunk_0",
        "skillprofile_Pooping_chunk_1",
    ]
    assert r["documents"][0][0].startswith("Skill Profile")
    assert "XP Table" in r["documents"][0][1]


def test_unchunked_hub_included():
    docs = [_mk("item_42", "Item text")]
    er._load_docs = lambda: docs
    r = er.build_entity_context("what is cheese", "item_42")
    assert r["ids"][0] == ["item_42"]


def test_facet_type_filters(monkeypatch):
    calls = []

    def fake_retrieve(question, count=3, metadata_filter=None, hybrid=True, rerank=True):
        calls.append((question, metadata_filter))
        return _empty_retrieve()

    monkeypatch.setattr(er, "retrieve", fake_retrieve)
    er.build_entity_context("what is Dungcrafting", "skillprofile_Pooping")
    filters = [c[1] for c in calls]
    assert {"type": "recipe"} in filters
    assert {"type": "quest"} in filters
    assert {"type": "npc"} in filters
    assert {"type": "advancementtable"} in filters


def test_facet_uses_entity_name_not_full_question(monkeypatch):
    calls = []

    def fake_retrieve(question, count=3, metadata_filter=None, hybrid=True, rerank=True):
        calls.append(question)
        return _empty_retrieve()

    monkeypatch.setattr(er, "retrieve", fake_retrieve)
    er.build_entity_context(
        "Tell me what recipes I will use while leveling saddlery from 0 to 15",
        "skillprofile_Saddlery",
    )
    for q in calls:
        assert "Tell me" not in q
        assert "leveling" not in q
        assert "Saddlery" in q


def test_facet_dedupe(monkeypatch):
    def fake_retrieve(question, count=3, metadata_filter=None, hybrid=True, rerank=True):
        return {
            "ids": [["skillprofile_Pooping_chunk_1", "recipe_906"]],
            "documents": [["dup text", "Recipe text"]],
            "metadatas": [[{"type": "skill"}, {"type": "recipe"}]],
            "distances": [[0.5, 0.3]],
        }

    monkeypatch.setattr(er, "retrieve", fake_retrieve)
    r = er.build_entity_context("what is Dungcrafting", "skillprofile_Pooping")
    ids = r["ids"][0]
    assert ids.count("skillprofile_Pooping_chunk_1") == 1
    assert "recipe_906" in ids


def test_budget_cap(monkeypatch):
    monkeypatch.setattr(er, "CONTEXT_BUDGET", 100)

    def fake_retrieve(question, count=3, metadata_filter=None, hybrid=True, rerank=True):
        return {
            "ids": [["recipe_906"]],
            "documents": [["R" * 300]],
            "metadatas": [[{"type": "recipe"}]],
            "distances": [[0.3]],
        }

    monkeypatch.setattr(er, "retrieve", fake_retrieve)
    r = er.build_entity_context("what is Dungcrafting", "skillprofile_Pooping")
    texts = r["documents"][0]
    assert sum(len(t) for t in texts) <= 100
    assert texts[0].startswith("Skill Profile")


def test_hub_miss_returns_none():
    er._load_docs = lambda: []
    assert er.build_entity_context("what is missing", "skillprofile_Nope") is None


def test_non_entity_prefix_no_facets():
    docs = [_mk("summary_cheese", "Cheese summary", dtype="summary", name="Cheese Summary")]
    er._load_docs = lambda: docs
    r = er.build_entity_context("what is summary", "summary_cheese")
    assert r["ids"][0] == ["summary_cheese"]
