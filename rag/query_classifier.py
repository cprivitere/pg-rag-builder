import re

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


def classify_query(query: str) -> str:
    lower = query.lower()

    for pattern in COMPARISON_PATTERNS:
        if re.search(pattern, lower):
            return "comparison"

    for pattern in LOOKUP_INDICATORS:
        if re.search(pattern, lower):
            return "lookup"

    return "general"
