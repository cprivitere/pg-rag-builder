import json
import re

import mwparserfromhell

from config import WIKI_DIR

MIN_SECTION_CHARS = 50

CACHE_FILE = WIKI_DIR / ".parsed.json"

# Templates whose first argument is a meaningful name that gets
# destroyed by strip_code().  We rewrite them to plain text first.
_TEMPLATE_PATTERN = re.compile(
    r"\{\{(Item|NPC|Quest|Skill|Area|Recipe|LoreBook|Ability)"
    r"\|([^}|]+)(?:\|[^}]*)?\}\}"
)


def _preserve_template_names(wikicode_text):
    """Replace {{Template|Name|...}} with just Name before stripping."""
    return _TEMPLATE_PATTERN.sub(r"\2", wikicode_text)


def _parse_page(page_name, raw_text):
    documents = []
    seen_ids = set()

    wikicode = mwparserfromhell.parse(raw_text)
    sections = wikicode.get_sections(
        levels=[2], include_lead=True
    )

    for section in sections:
        heading = ""
        for h in section.filter_headings():
            h_clean = mwparserfromhell.parse(
                str(h.title)
            ).strip_code(
                normalize=False, collapse=True
            ).strip()
            heading = h_clean
            break

        text = _preserve_template_names(str(section))
        text = mwparserfromhell.parse(text).strip_code(
            normalize=False, collapse=True
        ).strip()

        # mwparserfromhell glitch: unclosed ''' before a == heading leaves
        # preceding template shells intact (B16). Drop leftover shells.
        text = re.sub(r"\{\{[^{}]*\}\}", "", text).strip()
        # unclosed openers and stray closers left behind (nested templates)
        text = re.sub(r"\{\{[^{}]*$", "", text, flags=re.M).strip()
        text = re.sub(r"^\}\}", "", text, flags=re.M).strip()

        if not text or len(text) < MIN_SECTION_CHARS:
            continue

        if text.startswith("__NOTOC__"):
            continue

        doc_id = f"wiki_{page_name}"
        if heading:
            safe_heading = (
                heading
                .replace(" ", "_")
                .replace("/", "_")
                .replace("&", "and")[:80]
            )
            doc_id = f"wiki_{page_name}_{safe_heading}"

        if doc_id in seen_ids:
            counter = 2
            while f"{doc_id}_{counter}" in seen_ids:
                counter += 1
            doc_id = f"{doc_id}_{counter}"
        seen_ids.add(doc_id)

        documents.append({
            "id": doc_id,
            "type": "wiki",
            "text": text,
            "metadata": {
                "source": "wiki",
                "table": "wiki",
                "name": page_name.replace("_", " "),
                "section": heading if heading else None,
            },
        })

    return documents


def _load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}
    return {}


def _save_cache(cache):
    CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False),
        encoding="utf-8",
    )


def build_wiki_documents(db):
    if not hasattr(db, "wiki_mtimes"):
        documents = []
        for page_name, raw_text in db.wiki.items():
            documents.extend(_parse_page(page_name, raw_text))
        return documents

    cache = _load_cache()
    changed = set()

    documents = []
    for page_name, raw_text in db.wiki.items():
        mtime = db.wiki_mtimes.get(page_name)
        cached = cache.get(page_name)

        if cached is not None and cached.get("mtime") == mtime:
            documents.extend(cached["docs"])
            continue

        docs = _parse_page(page_name, raw_text)
        cache[page_name] = {"mtime": mtime, "docs": docs}
        changed.add(page_name)
        documents.extend(docs)

    for page_name in list(cache):
        if page_name not in db.wiki:
            del cache[page_name]
            changed.add(page_name)

    if changed:
        _save_cache(cache)

    return documents