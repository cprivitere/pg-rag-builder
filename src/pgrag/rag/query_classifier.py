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

ENTITY_TYPES = ("skill", "item", "ability", "quest", "recipe", "effect", "area", "npc")

_ENTITY_INDEX = None
_NAME_RE_CACHE = {}


def _name_regex(name):
    key = name.lower()
    pattern = _NAME_RE_CACHE.get(key)
    if pattern is None:
        pattern = re.compile(rf"\b{re.escape(key)}\b")
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


def classify_query(query: str) -> str:
    lower = query.lower()

    # Leveling intent with a named skill wins over comparison phrasing
    # ("most efficient way to level Cheesemaking" is a how-to, not a comparison).
    # Restricted to skills: generic words that happen to match NPC names
    # ("way to level up" -> NPC "Way") must not hijack the route.
    if any(re.search(p, lower) for p in LEVELING_PATTERNS):
        hub, dtype = find_entity(query)
        if hub and dtype == "skill":
            return "entity"

    for pattern in COMPARISON_PATTERNS:
        if re.search(pattern, lower):
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
