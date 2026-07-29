from unittest.mock import patch

from documents.builder import build_item_documents, build_recipe_documents


def _make_db(items=None, recipes=None):
    class FakeDB:
        def __init__(self):
            self.tables = {}
            if items is not None:
                self.tables["items"] = items
            if recipes is not None:
                self.tables["recipes"] = recipes
            self.wiki = {}
    return FakeDB()


def test_item_document_shape():
    items = {"item_1": {"Name": "Bunny Juice", "Description": "Turns you into a rabbit"}}
    db = _make_db(items=items)
    docs = build_item_documents(db)
    assert len(docs) == 1
    doc = docs[0]
    assert set(doc.keys()) == {"id", "type", "text", "metadata"}
    assert doc["id"] == "item_1"
    assert doc["type"] == "item"
    assert "Bunny Juice" in doc["text"]
    meta = doc["metadata"]
    assert meta["source"] == "cdn"
    assert meta["table"] == "items"
    assert meta["name"] == "Bunny Juice"


def test_recipe_document_shape():
    recipes = {"recipe_1": {"Name": "Healing Potion", "Skill": "Alchemy", "SkillLevelReq": 5, "Ingredients": [], "ResultItems": []}}
    db = _make_db(recipes=recipes)
    docs = build_recipe_documents(db)
    assert len(docs) == 1
    doc = docs[0]
    assert set(doc.keys()) == {"id", "type", "text", "metadata"}
    assert doc["id"] == "recipe_1"
    assert doc["type"] == "recipe"
    assert "Healing Potion" in doc["text"]
    meta = doc["metadata"]
    assert meta["source"] == "cdn"
    assert meta["table"] == "recipes"


def test_build_documents_sets_type_in_metadata():
    items = {"item_1": {"Name": "Bunny Juice"}}
    recipes = {"recipe_1": {"Name": "Healing Potion", "Skill": "Alchemy", "SkillLevelReq": 5, "Ingredients": [], "ResultItems": []}}
    db = _make_db(items=items, recipes=recipes)
    from documents.builder import build_documents
    docs = build_documents(db)
    for doc in docs:
        assert doc["metadata"]["type"] == doc["type"]


def test_build_documents_derives_name_from_text_when_missing():
    items = {"item_1": {"Name": "Bunny Juice"}}
    db = _make_db(items=items)
    from documents.builder import build_documents
    docs = build_documents(db)
    item_doc = next(d for d in docs if d["id"] == "item_1")
    assert item_doc["metadata"]["name"] == "Bunny Juice"


def test_build_documents_includes_summaries():
    from documents.builder import build_documents

    recipes = {
        "r1": {"Name": "Cheese A", "Skill": "Cheesemaking", "SkillLevelReq": 10, "Ingredients": [], "ResultItems": []},
        "r2": {"Name": "Cheese B", "Skill": "Cheesemaking", "SkillLevelReq": 50, "Ingredients": [], "ResultItems": []},
    }
    db = _make_db(recipes=recipes)
    docs = build_documents(db)
    summaries = [d for d in docs if d["type"] == "summary"]
    assert len(summaries) >= 1
    cheesemaking_summary = next(s for s in summaries if "Cheesemaking" in s["metadata"]["name"])
    assert "Cheese B (50)" in cheesemaking_summary["text"]
