from pgrag.documents.wiki_builder import build_wiki_documents, _preserve_template_names


class FakeDB:
    def __init__(self, wiki):
        self.wiki = wiki
        self.tables = {}


# --- _preserve_template_names unit tests ---


def test_item_template_preserved():
    raw = "{{Item|Mortaferus Mushroom}} || 95 || [[Vidaria]]"
    result = _preserve_template_names(raw)
    assert "Mortaferus Mushroom" in result
    assert "{{Item|" not in result


def test_npc_template_preserved():
    raw = "Talk to {{NPC|Rita}} in [[Serbule]]."
    result = _preserve_template_names(raw)
    assert "Rita" in result
    assert "{{NPC|" not in result


def test_quest_template_preserved():
    raw = "Complete {{Quest|Graffiti_Glyph27}} for XP."
    result = _preserve_template_names(raw)
    assert "Graffiti_Glyph27" in result


def test_skill_template_preserved():
    raw = "Requires {{Skill|Alchemy}} level 50."
    result = _preserve_template_names(raw)
    assert "Alchemy" in result


def test_area_template_preserved():
    raw = "Found in {{Area|AreaSunVale}}."
    result = _preserve_template_names(raw)
    assert "AreaSunVale" in result


def test_recipe_template_preserved():
    raw = "Uses {{Recipe|Healing Potion}} recipe."
    result = _preserve_template_names(raw)
    assert "Healing Potion" in result


def test_lorebook_template_preserved():
    raw = "Read {{LoreBook|The Wasted Wishes}}."
    result = _preserve_template_names(raw)
    assert "The Wasted Wishes" in result


def test_ability_template_preserved():
    raw = "Cast {{Ability|Fireball}} on enemies."
    result = _preserve_template_names(raw)
    assert "Fireball" in result


def test_ignores_unknown_templates():
    raw = "{{RandomTemplate|KeepMe}} and {{Item|Mushroom}}"
    result = _preserve_template_names(raw)
    assert "{{RandomTemplate|KeepMe}}" in result
    assert "Mushroom" in result


def test_template_with_extra_args():
    raw = "{{Item|Boletus Mushroom|link=no|icon=small}}"
    result = _preserve_template_names(raw)
    assert "Boletus Mushroom" in result
    assert "{{Item|" not in result


def test_no_templates_unchanged():
    raw = "Just plain text with no templates."
    assert _preserve_template_names(raw) == raw


def test_multiple_templates():
    raw = "Get {{Skill|Mycology}} 60, then farm {{Item|Ghostshroom}} in {{Area|Vidaria}}."
    result = _preserve_template_names(raw)
    assert "Mycology" in result
    assert "Ghostshroom" in result
    assert "Vidaria" in result
    assert "{{" not in result


# --- build_wiki_documents integration tests ---


def test_wiki_page_preserves_item_names():
    raw = """__NOTOC__
== Harvestables ==
{| class="wikitable"
!Name
!Level
|-
| {{Item|Parasol Mushroom}} || 00
|-
| {{Item|Mycena Mushroom}} || 05
|-
| {{Item|Mortaferus Mushroom}} || 95
|}
"""
    db = FakeDB({"Mycology": raw})
    docs = build_wiki_documents(db)
    all_text = " ".join(d["text"] for d in docs)
    assert "Parasol Mushroom" in all_text
    assert "Mycena Mushroom" in all_text
    assert "Mortaferus Mushroom" in all_text


def test_wiki_page_still_strips_other_wikicode():
    raw = """__NOTOC__
== Overview ==
This is [[Mycology]], a skill for gathering mushrooms.
"""
    db = FakeDB({"Mycology": raw})
    docs = build_wiki_documents(db)
    all_text = " ".join(d["text"] for d in docs)
    assert "Mycology" in all_text
    assert "[[" not in all_text
    assert "]]" not in all_text


def test_wiki_page_preserves_npc_names():
    raw = """__NOTOC__
== Trainers ==
Talk to {{NPC|Mushroom Jack}} in [[Serbule]] to learn.
"""
    db = FakeDB({"Mycology": raw})
    docs = build_wiki_documents(db)
    all_text = " ".join(d["text"] for d in docs)
    assert "Mushroom Jack" in all_text


def test_empty_wiki_produces_no_docs():
    db = FakeDB({})
    docs = build_wiki_documents(db)
    assert docs == []


def test_short_section_skipped():
    raw = """__NOTOC__
== Tiny ==
Hi.
"""
    db = FakeDB({"TestPage": raw})
    docs = build_wiki_documents(db)
    assert len(docs) == 0
