import os
import re
import json
from pathlib import Path
from collections import defaultdict

CDN_DIR = Path("F:/ProjectGorgon/cdn")
WIKI_DIR = Path("project_gorgon_wiki_backup")
OUTPUT_DIR = Path("knowledge")


def strip_wikitext(text):
    text = re.sub(r"\[\[[^\]]*?\|?([^\]]+?)\]\]", r"\1", text)
    text = re.sub(r"'''(.*?)'''", r"\1", text)
    text = re.sub(r"''(.*?)''", r"\1", text)
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"__\w+__", "", text)
    text = re.sub(r"^[=]+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"&[a-z]+;", "", text)
    text = re.sub(r"\/[a-zA-Z_]+\}\}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^[-\s]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"Category:\S+", "", text)
    return text.strip()


def load_wiki_pages():
    pages = {}
    if not WIKI_DIR.exists():
        return pages
    for f in WIKI_DIR.iterdir():
        if f.suffix == ".txt":
            name = f.stem.replace("_", " ").lower()
            pages[name] = f.read_text(encoding="utf-8")
    return pages


def wiki_match(wiki_pages, names):
    for name in names:
        key = name.lower()
        if key in wiki_pages:
            text = strip_wikitext(wiki_pages[key])
            return f"\n\n**Wiki Guide:**\n{text}" if text else ""
    return ""


def save_doc(filename, content):
    path = OUTPUT_DIR / filename
    path.write_text(content, encoding="utf-8")


def build_npcs(wiki_pages):
    npcs = json.load(open(CDN_DIR / "npcs.json"))
    count = 0
    for key, data in npcs.items():
        if not key.startswith("NPC_"):
            continue
        name = data.get("Name", key)
        area = data.get("AreaFriendlyName", "")
        desc = data.get("Desc", "")

        lines = [f"# {name}", ""]
        if area:
            lines.append(f"**Location:** {area}")
        if desc:
            lines.append(f"**Description:** {desc}")
        services = data.get("Services", [])
        if services:
            lines.append("")
            lines.append("**Services:**")
            for s in services:
                stype = s.get("Type", "")
                favor = s.get("Favor", "")
                parts = [f"  - {stype}"]
                if favor:
                    parts.append(f"(requires {favor} favor)")
                if stype == "Training" and s.get("Skills"):
                    parts.append(": " + ", ".join(sk for sk in s["Skills"] if sk != "Unknown"))
                lines.append(" ".join(parts))
        lines.append(wiki_match(wiki_pages, [name]))
        content = "\n".join(lines).strip()
        if content:
            save_doc(f"npc_{key}.txt", content)
            count += 1
    return count


def build_skills(wiki_pages):
    skills = json.load(open(CDN_DIR / "skills.json"))
    count = 0
    for key, data in skills.items():
        desc = data.get("Description", "")
        combat = data.get("Combat", False)
        hints = data.get("AdvancementHints", {})

        lines = [f"# {key}", ""]
        if desc:
            lines.append(f"**Description:** {desc}")
        lines.append(f"**Combat Skill:** {'Yes' if combat else 'No'}")
        if hints:
            lines.append("")
            lines.append("**Advancement Hints:**")
            for level, hint in sorted(hints.items()):
                lines.append(f"  - Level {level}: {hint}")
        lines.append(wiki_match(wiki_pages, [key]))
        content = "\n".join(lines).strip()
        if content:
            save_doc(f"skill_{key}.txt", content)
            count += 1
    return count


def build_lorebooks():
    lorebooks = json.load(open(CDN_DIR / "lorebooks.json"))
    count = 0
    for key, data in lorebooks.items():
        name = data.get("InternalName", key)
        category = data.get("Category", "")
        location = data.get("LocationHint", "")
        text = data.get("Text", "")

        lines = [f"# {name}", ""]
        if category:
            lines.append(f"**Category:** {category}")
        if location:
            lines.append(f"**Location:** {location}")
        lines.append("")
        lines.append(strip_wikitext(text).strip())
        content = "\n".join(lines).strip()
        if content:
            save_doc(f"lore_{key}.txt", content)
            count += 1
    return count


def build_recipes():
    recipes = json.load(open(CDN_DIR / "recipes.json"))
    items = json.load(open(CDN_DIR / "items.json"))
    item_names = {}
    for ik, iv in items.items():
        code = ik.replace("item_", "")
        item_names[code] = iv.get("InternalName", ik)

    count = 0
    for key, data in recipes.items():
        name = data.get("Name", key)
        desc = data.get("Description", "")
        skill = data.get("RewardSkill", "")
        xp = data.get("RewardSkillXp", "")
        ingredients = data.get("Ingredients", [])
        results = data.get("ResultItems", [])

        lines = [f"# {name}", ""]
        if desc:
            lines.append(f"**Description:** {desc}")
        if skill:
            line = f"**Skill:** {skill}"
            if xp:
                line += f" (+{xp} XP)"
            lines.append(line)
        if ingredients:
            lines.append("")
            lines.append("**Ingredients:**")
            for ing in ingredients:
                qty = ing.get("StackSize", 1)
                code = str(ing.get("ItemCode", ""))
                iname = item_names.get(code, f"Item#{code}")
                chance = ing.get("ChanceToConsume")
                line = f"  - {iname} x{qty}"
                if chance and chance < 1:
                    line += f" ({int(chance*100)}% chance to consume)"
                lines.append(line)
        if results:
            lines.append("")
            lines.append("**Results:**")
            for res in results:
                qty = res.get("StackSize", 1)
                code = str(res.get("ItemCode", ""))
                iname = item_names.get(code, f"Item#{code}")
                pct = res.get("PercentChance", 1)
                line = f"  - {iname} x{qty}"
                if pct < 1:
                    line += f" ({int(pct*100)}% chance)"
                lines.append(line)

        content = "\n".join(lines).strip()
        if content:
            save_doc(f"recipe_{key}.txt", content)
            count += 1
    return count


def build_quests(wiki_pages):
    quests = json.load(open(CDN_DIR / "quests.json"))
    cdn_lookup = {}
    for k, v in quests.items():
        name = v.get("Name", "")
        if name:
            cdn_lookup[name.lower().replace("-", " ").replace("'", "")] = k

    count = 0
    for key, data in quests.items():
        name = data.get("Name", key)
        desc = data.get("Description", "")
        objectives = data.get("Objectives", [])

        lines = [f"# {name}", ""]
        if desc:
            lines.append(f"**Description:** {desc}")
        if objectives:
            lines.append("")
            lines.append("**Objectives:**")
            for obj in objectives:
                otype = obj.get("Type", "")
                target = obj.get("Target", obj.get("Description", ""))
                number = obj.get("Number", 1)
                lines.append(f"  - {otype}: {target} ({number})")
        lines.append(wiki_match(wiki_pages, [name, key]))
        content = "\n".join(lines).strip()
        if content:
            save_doc(f"quest_{key}.txt", content)
            count += 1
    return count


def build_abilities():
    abilities = json.load(open(CDN_DIR / "abilities.json"))
    batch = defaultdict(list)
    for key, data in abilities.items():
        kws = data.get("Keywords", [])
        group = "Uncategorized"
        for kw in kws:
            if kw in ("Attack", "Passive", "Buff"):
                group = kw
                break
        name = data.get("Name", key)
        tier = data.get("Tier", "")
        description = data.get("DescriptionOverride", data.get("Description", ""))
        desc_short = (description[:120] + "...") if len(description) > 120 else description
        line = f"- **{name}** (Tier {tier}): {desc_short}"
        batch[group].append(line)

    count = 0
    for group, items in sorted(batch.items()):
        lines = [f"# Abilities - {group}", "", f"Total: {len(items)} abilities", ""]
        lines.extend(sorted(items))
        save_doc(f"abilities_{group}.txt", "\n".join(lines))
        count += 1
    return count


def build_items():
    items = json.load(open(CDN_DIR / "items.json"))
    source_items = json.load(open(CDN_DIR / "sources_items.json"))
    batch = defaultdict(list)
    for key, data in items.items():
        kws = data.get("Keywords", [])
        group = "Uncategorized"
        for kw in kws:
            if kw in ("Equipment", "Consumable", "CraftingIngredient", "Loot", "Document", "Book", "Recipe"):
                group = kw
                break
        name = data.get("InternalName", key)
        desc = data.get("Description", "")
        desc_short = (desc[:120] + "...") if len(desc) > 120 else desc
        line = f"- **{name}**: {desc_short}"
        batch[group].append(line)

    count = 0
    for group, items_list in sorted(batch.items()):
        lines = [f"# Items - {group}", "", f"Total: {len(items_list)} items", ""]
        lines.extend(sorted(items_list))
        save_doc(f"items_{group}.txt", "\n".join(lines))
        count += 1
    return count


def build_effects():
    effects = json.load(open(CDN_DIR / "effects.json"))
    batch = defaultdict(list)
    for key, data in effects.items():
        name = data.get("Name", key)
        desc = data.get("Description", "")
        if desc:
            desc_short = (desc[:120] + "...") if len(desc) > 120 else desc
            batch["All"].append(f"- **{name}**: {desc_short}")

    count = 0
    for group, items in batch.items():
        lines = [f"# Effects - {group}", "", f"Total: {len(items)} effects", ""]
        lines.extend(sorted(items))
        save_doc(f"effects_{group}.txt", "\n".join(lines))
        count += 1
    return count


def build_ai():
    ai = json.load(open(CDN_DIR / "ai.json"))
    count = 0
    for key, data in ai.items():
        name = data.get("Name", key)
        desc = data.get("Description", "")
        lines = [f"# {name}", ""]
        if desc:
            lines.append(f"**Description:** {desc}")
        personality = data.get("Personality", [])
        if personality:
            lines.append(f"**Personality:** {', '.join(personality)}")
        content = "\n".join(lines).strip()
        if content:
            save_doc(f"ai_{key}.txt", content)
            count += 1
    return count


def build_attributes():
    attrs = json.load(open(CDN_DIR / "attributes.json"))
    lines = ["# Attributes", ""]
    for key, data in attrs.items():
        name = data.get("Name", key)
        desc = data.get("Description", "")
        lines.append(f"- **{name}**: {desc}")
    save_doc("attributes.txt", "\n".join(lines))
    return 1


def build_areas():
    areas = json.load(open(CDN_DIR / "areas.json"))
    lines = ["# Areas", ""]
    for key, data in areas.items():
        name = data.get("Name", key)
        parent = data.get("Parent", "")
        line = f"- **{name}**"
        if parent:
            line += f" (parent: {parent})"
        lines.append(line)
    save_doc("areas.txt", "\n".join(lines))
    return 1


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wiki_pages = load_wiki_pages()
    print(f"Loaded {len(wiki_pages)} wiki pages")

    total = 0
    total += build_npcs(wiki_pages)
    total += build_skills(wiki_pages)
    total += build_lorebooks()
    total += build_recipes()
    total += build_quests(wiki_pages)
    total += build_abilities()
    total += build_items()
    total += build_effects()
    total += build_ai()
    total += build_attributes()
    total += build_areas()

    print(f"Built {total} knowledge documents in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
