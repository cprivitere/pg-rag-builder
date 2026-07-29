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
