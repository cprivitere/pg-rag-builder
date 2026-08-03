import json
import re
from pathlib import Path

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


def _load_entity_index():
    global _ENTITY_INDEX
    if _ENTITY_INDEX is not None:
        return _ENTITY_INDEX

    index = []
    path = Path("data/documents.json")
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
    _ENTITY_INDEX = index
    return index


def _hub_id(doc_id, dtype):
    if dtype == "skill":
        key = doc_id[len("skill_"):]
        return f"skillprofile_{key}"
    return doc_id


def find_entity(query):
    lower = query.lower()
    for name, doc_id, dtype in _load_entity_index():
        if re.search(rf"\b{re.escape(name.lower())}\b", lower):
            return _hub_id(doc_id, dtype), dtype
    return None, None


def classify_query(query: str) -> str:
    lower = query.lower()

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
