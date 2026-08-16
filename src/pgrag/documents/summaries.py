import re
from collections import defaultdict


def _extract_skill_and_level(doc):
    skill_match = re.search(r"Skill:\s*\n(.+)", doc["text"])
    level_match = re.search(r"Required Skill Level:\s*\n(\d+)", doc["text"])
    if skill_match and level_match:
        return skill_match.group(1).strip(), int(level_match.group(1))
    return None, None


def _extract_recipe_name(doc):
    return doc.get("metadata", {}).get("name", doc["id"])


def build_summary_documents(documents):
    skill_groups = defaultdict(list)

    for doc in documents:
        if doc.get("type") != "recipe":
            continue
        skill, level = _extract_skill_and_level(doc)
        if skill is not None:
            name = _extract_recipe_name(doc)
            skill_groups[skill].append((level, name))

    summaries = []
    for skill, items in skill_groups.items():
        items.sort(key=lambda x: x[0], reverse=True)

        lines = [f"{skill} recipes ranked by skill level:"]
        for rank, (level, name) in enumerate(items, 1):
            lines.append(f"{rank}. {name} ({level})")

        summary_id = f"summary_{skill.lower().replace(' ', '_')}"
        summaries.append({
            "id": summary_id,
            "type": "summary",
            "text": "\n".join(lines),
            "metadata": {
                "source": "computed",
                "table": "summaries",
                "name": f"{skill} Summary",
                "type": "summary",
            }
        })

    return summaries


# ---------------------------------------------------------------------------
# Gathering summaries: link items to skill-level requirements via CDN data
# ---------------------------------------------------------------------------

# Patterns for numbered keyword prefixes (e.g., Mushroom1, Skin1)
_NUMBERED_KW_RE = re.compile(r"^([A-Za-z]+)(\d+)$")

# Skills whose gathering recipes use ItemMenuKeywordReq
_GATHERING_SKILLS = {
    "Mushroom": "Mycology",
    "Skin": "Tanning",
}

# Named fish keywords → Fishing skill
_FISH_SKILL = "Fishing"


def _item_name(item):
    return item.get("Name", "")


def _item_keywords(item):
    kws = item.get("Keywords", [])
    return kws if isinstance(kws, list) else []


def build_gathering_summaries(items, recipes):
    """Build summaries linking raw items to gathering skill levels via CDN recipes.

    Args:
        items: dict of item_id -> item_data (from db.tables["items"])
        recipes: dict of recipe_id -> recipe_data (from db.tables["recipes"])
    """
    # Index recipes by ItemMenuKeywordReq
    keyword_recipes = {}
    for recipe in recipes.values():
        if not isinstance(recipe, dict):
            continue
        kw_req = recipe.get("ItemMenuKeywordReq")
        if kw_req:
            keyword_recipes[kw_req] = recipe

    # Group items by skill
    skill_items = defaultdict(list)  # skill -> [(level, item_name)]

    for item in items.values():
        if not isinstance(item, dict):
            continue
        name = _item_name(item)
        if not name:
            continue

        for kw in _item_keywords(item):
            # Numbered keywords: Mushroom1, Skin2, etc.
            m = _NUMBERED_KW_RE.match(kw)
            if m:
                prefix, _num = m.group(1), m.group(2)
                if prefix in _GATHERING_SKILLS:
                    skill = _GATHERING_SKILLS[prefix]
                    recipe = keyword_recipes.get(kw)
                    if recipe:
                        level = recipe.get("SkillLevelReq", 0)
                        skill_items[skill].append((level, name))
                continue

            # Named fish keywords (plain name matches recipe keyword)
            recipe = keyword_recipes.get(kw)
            if recipe and recipe.get("Skill") == _FISH_SKILL:
                level = recipe.get("SkillLevelReq", 0)
                skill_items[_FISH_SKILL].append((level, name))

    summaries = []
    for skill, entries in skill_items.items():
        entries.sort(key=lambda x: x[0], reverse=True)
        lines = [f"{skill} items ranked by required skill level:"]
        for rank, (level, name) in enumerate(entries, 1):
            lines.append(f"{rank}. {name} ({level})")

        summary_id = f"summary_gathering_{skill.lower().replace(' ', '_')}"
        summaries.append({
            "id": summary_id,
            "type": "summary",
            "text": "\n".join(lines),
            "metadata": {
                "source": "computed",
                "table": "summaries",
                "name": f"{skill} Gathering Summary",
                "type": "summary",
            }
        })

    return summaries


# ---------------------------------------------------------------------------
# Wiki gathering summaries: parse harvestable tables for complete data
# ---------------------------------------------------------------------------

# Maps wiki page names to skill names
_WIKI_SKILL_MAP = {
    "Fishing": "Fishing",
    "Foraging": "Foraging",
    "Mining": "Mining",
    "Mycology": "Mycology",
    "Myconic": "Mycology",
}

# Regex to extract rows from wiki tables: {{Item|Name}} || Level || ...
_WIKI_ROW_RE = re.compile(
    r"\{\{Item\|([^}|]+)\}\}\s*\|\|\s*(\d+)\??"
)


def _parse_wiki_harvest_rows(wiki):
    """Parse harvestable tables from wiki pages.

    Returns:
        dict: skill -> list of (level, item_name) tuples.
    """
    skill_entries = defaultdict(list)  # skill -> [(level, name)]

    for page_name, raw_text in wiki.items():
        skill = _WIKI_SKILL_MAP.get(page_name)
        if skill is None:
            continue

        # Match line by line to avoid table syntax interfering with regex
        for line in raw_text.splitlines():
            for row_match in _WIKI_ROW_RE.finditer(line):
                name = row_match.group(1).strip()
                try:
                    level = int(row_match.group(2))
                except ValueError:
                    continue  # non-numeric level cell — skip row (V26)
                skill_entries[skill].append((level, name))

    return skill_entries


def build_wiki_harvest_map(wiki):
    """Map item name (lowercased) -> (skill, required level) from wiki harvest tables.

    When an item appears in more than one skill table, the highest level
    (binding requirement) wins so the item doc shows what it really takes to
    gather it.
    """
    harvest_map = {}
    for skill, entries in _parse_wiki_harvest_rows(wiki).items():
        for level, name in entries:
            key = name.strip().lower()
            current = harvest_map.get(key)
            if current is None or level > current[1]:
                harvest_map[key] = (skill, level)
    return harvest_map


def build_wiki_gathering_summaries(wiki):
    """Parse harvestable tables from wiki pages and build skill-level summaries.

    Args:
        wiki: dict of page_name -> raw_text (from db.wiki)
    """
    skill_entries = _parse_wiki_harvest_rows(wiki)

    summaries = []
    for skill, entries in skill_entries.items():
        # Deduplicate (Myconic page has same mushrooms as Mycology)
        seen = set()
        unique = []
        for level, name in sorted(entries, reverse=True):
            if name not in seen:
                seen.add(name)
                unique.append((level, name))
        entries = unique

        entries.sort(key=lambda x: x[0], reverse=True)
        lines = [f"{skill} items ranked by required skill level (wiki):"]
        for rank, (level, name) in enumerate(entries, 1):
            lines.append(f"{rank}. {name} ({level})")

        summary_id = f"summary_wiki_{skill.lower().replace(' ', '_')}"
        summaries.append({
            "id": summary_id,
            "type": "summary",
            "text": "\n".join(lines),
            "metadata": {
                "source": "wiki",
                "table": "summaries",
                "name": f"{skill} Wiki Gathering Summary",
                "type": "summary",
            }
        })

    return summaries
