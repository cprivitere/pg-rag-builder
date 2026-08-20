import json
import re
from pathlib import Path

from pgrag.rag.spelling import correct_query

COMPARISON_PATTERNS = [
    r"\bhighest\b",
    r"\blowest\b",
    r"\bbest\b",
    r"\bworst\b",
    r"\bmost\b",
    r"\bleast\b",
    r"\bmaximum\b",
    r"\bminimum\b",
    r"\btop\b",
    r"\bstrongest\b",
    r"\bweakest\b",
    r"\bbiggest\b",
    r"\bsmallest\b",
    r"\bfastest\b",
    r"\bslowest\b",
    r"\bvs\.?\b",
    r"\b(?:more|less) (?:damage|powerful|effective|xp)\b",
    r"\bbetween .+ and .+\b",
]

# "How do I raise skill X" — even when phrased as "most efficient way to level
# X" (which trips the comparison patterns above), a named skill entity should
# route to the entity dossier, not item comparison.
LEVELING_PATTERNS = [
    r"\bhow to level(?: up)?\b",
    r"\bhow do i level(?: up)?\b",
    r"\bhow do you level(?: up)?\b",
    r"\b(?:way|ways) to level\b",
    r"\bmost efficient way to (?:level|raise)\b",
    r"\befficient way to (?:level|raise)\b",
    r"\bto level up\b",
    r"\bhow do i raise (?:my |the )?(\w+) skill\b",
]

LOOKUP_INDICATORS = [
    r"\bwhat level is\b",
    r"\bwhat level does\b",
    r"\bwhat level should\b",
    r"\bwhat level do\b",
    r"\bwhat level can\b",
    r"\bhow much\b",
    r"\bhow many\b",
    r"\bwhere is\b",
    r"\bwhere can\b",
]

ENTITY_PATTERNS = [
    r"\bwhat is\b",
    r"\bwhat are\b",
    r"\btell me about\b",
    r"\bhow do i get\b",
    r"\bhow do you get\b",
    r"\bwhat does\b",
    r"\bwhat can\b",
]

# Aggregation/listing intent — a named entity is a *filter or aspect* of the
# question, not the answer itself. "What recipes use Animal Feces?" lists
# recipes (general), whereas "What skill and level do I need to make Orcish
# Flour?" targets the item's dossier (entity). Likewise "How do I grow Field
# Mushrooms?" spans skill + wiki, not a single item dossier, and "Who gives
# quests in the Ranalon Den area?" enumerates quests rather than a single
# quest dossier. Lore books ("the Chalice Saga", "the lore book …") are
# multi-book narrative synthesis with no single-hub dossier, so they stay
# general too. This mirrors the existing anti-hijack guard
# (test_gardeningrelated_query_classifies_general): a surface entity mention
# must not upgrade an aggregation/listing query to a single-entity route.
# Checked after leveling intent (which is a stronger, skill-specific how-to).
AGGREGATION_PATTERNS = [
    r"\brecipes?\s+(?:using|that\s+use|use|can\s+(?:i|you)\s+make)\b",
    r"\bhow\s+do\s+i\s+grow\b",
    r"\bcan\s+i\s+grow\b",
    r"\bwho\s+gives?\s+quests?\s+in\b",
    r"\blore\s+book\b",
    r"\bsaga\b",
]

ENTITY_TYPES = ("skill", "item", "ability", "quest", "recipe", "effect", "area", "npc", "lorebook")

_ENTITY_INDEX = None
_NAME_RE_CACHE = {}

# Query-facing aliases that don't match any single doc `name`. Each alias key
# ("Chalice Saga") is injected into the entity index pointing at the real docs
# (name, doc_id, type) it denotes, so the alias resolves like any real name.
# Multiple target docs each get an alias entry (a series, quest-cluster, item
# family all resolve the alias).
_ENTITY_ALIASES: dict[str, list[tuple[str, str, str]]] = {
    "Chalice Saga": [
        ("The Chalice Saga, Vol 1", "lorebook_Book_103", "lorebook"),
        ("The Chalice Saga, Vol 2", "lorebook_Book_104", "lorebook"),
        ("The Chalice Saga, Vol 3", "lorebook_Book_105", "lorebook"),
    ],
    "Ranalon Den": [
        (f"quest {i}", f"quest_quest_{i}", "quest") for i in range(25401, 25416)
    ],
    "Animal Feces": [
        ("Meager Animal Poop", "item_1501", "item"),
        ("Cow Poop", "item_1493", "item"),
        ("Deer Poop", "item_1494", "item"),
        ("Pig Poop", "item_1495", "item"),
        ("Rabbit Poop", "item_1496", "item"),
    ],
}


def _name_regex(name):
    key = name.lower()
    pattern = _NAME_RE_CACHE.get(key)
    if pattern is None:
        # Strict whole-word match, plus a plural-append variant: `{entity}s` /
        # `{entity}es` (e.g. "Field Mushrooms" = "Field Mushroom" + "s").
        # This is NOT a stem/prefix match — "gardens" is not "Gardening"+"s",
        # and "StaffCaptain"/"BarleySoup" have no word boundary after the
        # entity name, so they stay unmatched (see test_lowercase_word_exten-
        # sion_not_matched / test_capitalized_compound_does_not_overmatch).
        pattern = re.compile(rf"\b{re.escape(key)}(?:es|s)?\b")
        _NAME_RE_CACHE[key] = pattern
    return pattern


def _load_entity_index():
    global _ENTITY_INDEX
    path = Path("data/documents.json")
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        mtime = None
    if _ENTITY_INDEX is not None and _ENTITY_INDEX[0] == mtime:
        return _ENTITY_INDEX[1]

    index = []
    if path.exists():
        seen = set()
        for doc in json.loads(path.read_text(encoding="utf-8")):
            meta = doc.get("metadata", {})
            name = meta.get("name")
            dtype = meta.get("type")
            if not name or not dtype:
                continue
            if dtype not in ENTITY_TYPES:
                continue
            doc_id = re.sub(r"_chunk_\d+$", "", doc.get("id", ""))
            if (name.lower(), doc_id) in seen:
                continue
            seen.add((name.lower(), doc_id))
            index.append((name, doc_id, dtype))

    # Inject query-facing aliases (multi-target: each target is its own entry
    # sharing the alias name, so the alias resolves to every real doc).
    for alias, targets in _ENTITY_ALIASES.items():
        for _real_name, doc_id, dtype in targets:
            if (alias.lower(), doc_id) in seen:
                continue
            seen.add((alias.lower(), doc_id))
            index.append((alias, doc_id, dtype))

    index.sort(key=lambda t: len(t[0]), reverse=True)
    _ENTITY_INDEX = (mtime, index)
    _NAME_RE_CACHE.clear()
    return index


def _hub_id(doc_id, dtype):
    if dtype == "skill":
        key = doc_id[len("skill_"):]
        return f"skillprofile_{key}"
    return doc_id


def _match_entity(text):
    for name, doc_id, dtype in _load_entity_index():
        if _name_regex(name).search(text):
            return _hub_id(doc_id, dtype), dtype
    return None


def find_entity(query):
    lower = query.lower()
    hit = _match_entity(lower)
    if hit:
        return hit
    corrected = correct_query(query)
    if corrected != lower:
        hit = _match_entity(corrected)
        if hit:
            return hit
    return None, None


def find_entities(query) -> list:
    """All entities named in the query, greedy longest-first and non-overlapping.

    Returns [(name, hub_id, dtype), ...] in query order. Matches against the
    original text, falling back to the spelling-corrected text when the
    original yields nothing. Names come from the entity index (not the hub
    filename), so ability hubs keep their real name.
    """
    corrected = correct_query(query)
    texts = [query]
    if corrected != query.lower():
        texts.append(corrected)

    for text in texts:
        t = text.lower()
        picked = []
        spans = []
        # The index is sorted longest-first, so the first match to claim a
        # span is the longest; overlapping shorter names are skipped.
        for name, doc_id, dtype in _load_entity_index():
            m = _name_regex(name).search(t)
            if not m:
                continue
            s, e = m.span()
            if any(not (e <= ss or s >= ee) for ss, ee in spans):
                continue
            spans.append((s, e))
            picked.append((s, name, doc_id, dtype))
        if not picked:
            continue
        picked.sort(key=lambda x: x[0])  # query order
        out = []
        seen = set()
        for _s, name, doc_id, dtype in picked:
            hub = _hub_id(doc_id, dtype)
            if hub in seen:
                continue
            seen.add(hub)
            out.append((name, hub, dtype))
        return out
    return []


def is_leveling_intent(query: str) -> bool:
    """True when the query asks how to level a named skill (one of
    LEVELING_PATTERNS + a skill entity). The pipeline uses this to pull the
    computed leveling_<Skill> dossier into the skill hub context — and
    deliberately NOT for unrelated skill questions (mushroom locations etc),
    which would otherwise lose wiki/table rows to the front-loaded ladder.
    """
    if any(re.search(p, query.lower()) for p in LEVELING_PATTERNS):
        hub, dtype = find_entity(query)
        if hub and dtype == "skill":
            return True
    return False


def classify_query(query: str) -> str:
    lower = query.lower()

    # Leveling intent with a named skill wins over comparison phrasing
    # ("most efficient way to level Cheesemaking" is a how-to, not a comparison).
    # Restricted to skills: generic words that happen to match NPC names
    # ("way to level up" -> NPC "Way") must not hijack the route.
    if is_leveling_intent(query):
        return "entity"

    # Aggregation/listing intent wins over comparison AND single-entity
    # routing: the named entity is a filter, not the answer. "recipes can I
    # make with Spider Silk at Tailoring 4" is a list of recipes (general),
    # not an item comparison even though two entities appear.
    for pattern in AGGREGATION_PATTERNS:
        if re.search(pattern, lower):
            return "general"

    for pattern in COMPARISON_PATTERNS:
        if re.search(pattern, lower):
            return "comparison"

    # Two (or more) entities named in one question is a comparison even when
    # no superlative/intensifier pattern trips — e.g. "Punch or Front Kick".
    if len(find_entities(query)) >= 2:
        return "comparison"

    for pattern in LOOKUP_INDICATORS:
        if re.search(pattern, lower):
            return "lookup"

    for pattern in ENTITY_PATTERNS:
        if re.search(pattern, lower):
            hub, _ = find_entity(query)
            return "entity" if hub else "general"

    hub, _ = find_entity(query)
    if hub:
        return "entity"

    return "general"
