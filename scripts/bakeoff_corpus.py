"""Bake-off corpus generator.

Grabs ~500 fattest docs from data/documents.json and builds 10-15
queries with known-relevant doc IDs for embedding model comparison.

Run: uv run python scripts/bakeoff_corpus.py
Writes: data/bakeoff_corpus.json
"""
import json
import re
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DOCS_PATH = DATA / "documents.json"
OUT_PATH = DATA / "bakeoff_corpus.json"

TARGET = 500
SEED = 42

CHUNK_SUFFIX = re.compile(r"_chunk_\d+$")


def base_key(doc_id):
    return CHUNK_SUFFIX.sub("", doc_id)


def doc_name(doc):
    meta = doc.get("metadata", {})
    return meta.get("name") or base_key(doc["id"])


def load_docs():
    with open(DOCS_PATH, encoding="utf-8") as f:
        return json.load(f)


def pick_fat_docs(docs, target):
    """Sort by text length descending, take top target."""
    valid = [d for d in docs if d.get("text", "").strip()]
    valid.sort(key=lambda d: len(d["text"]), reverse=True)
    return valid[:target]


def family_labels(selected_ids, doc_id):
    """All chunks sharing the same base key."""
    key = base_key(doc_id)
    return sorted(i for i in selected_ids if base_key(i) == key)


def build_queries(selected_docs, all_docs, n=12):
    """Hand-craft queries targeting specific docs with known relevance."""
    selected_ids = {d["id"] for d in selected_docs}
    by_id = {d["id"]: d for d in all_docs}
    selected_by_id = {d["id"]: d for d in selected_docs}

    queries = []

    # Type-stratified picks: grab one doc per interesting type, craft query
    type_picks = {}
    for d in selected_docs:
        t = d["type"]
        if t not in type_picks and t in ("recipe", "item", "wiki", "ability", "quest", "skillprofile", "effect"):
            type_picks[t] = d

    for type_, doc in type_picks.items():
        name = doc_name(doc)
        q_id = f"q_{type_}_{len(queries)}"

        if type_ == "recipe":
            q_text = f"What materials are needed to craft {name}?"
        elif type_ == "item":
            q_text = f"What does the item {name} do?"
        elif type_ == "wiki":
            title = base_key(doc["id"])
            if title.startswith("wiki_"):
                title = title[len("wiki_"):]
            q_text = f"What does the wiki say about {title.replace('_', ' ')}?"
        elif type_ == "ability":
            q_text = f"What does the ability {name} do?"
        elif type_ == "quest":
            q_text = f"How do I start the quest {name}?"
        elif type_ == "skillprofile":
            q_text = f"Tell me everything about the {name} skill"
        elif type_ == "effect":
            q_text = f"What effect does {name} have?"
        else:
            continue

        relevant = family_labels(selected_ids, doc["id"])
        if relevant:
            queries.append({"id": q_id, "text": q_text, "expected_doc_ids": relevant})

    # Fill remaining slots with cross-type queries
    used_ids = set()
    for q in queries:
        used_ids.update(q.get("expected_doc_ids", []))
    fill_candidates = [d for d in selected_docs if d["id"] not in used_ids]
    for doc in fill_candidates:
        if len(queries) >= n:
            break
        name = doc_name(doc)
        q_id = f"q_fill_{len(queries)}"
        if doc["type"] == "wiki":
            title = base_key(doc["id"])
            if title.startswith("wiki_"):
                title = title[len("wiki_"):]
            q_text = f"What information is available about {title.replace('_', ' ')}?"
        elif doc["type"] == "skillprofile":
            q_text = f"What are the details of the {name} skill profile?"
        elif doc["type"] == "recipe":
            q_text = f"What materials are needed to craft {name}?"
        elif doc["type"] == "item":
            q_text = f"What is {name} used for?"
        elif doc["type"] == "ability":
            q_text = f"What does the ability {name} do?"
        elif doc["type"] == "quest":
            q_text = f"How do I complete the quest {name}?"
        elif doc["type"] == "effect":
            q_text = f"What effect does {name} have?"
        else:
            q_text = f"Tell me about {name}"
        relevant = family_labels(selected_ids, doc["id"])
        if relevant:
            queries.append({"id": q_id, "text": q_text, "expected_doc_ids": relevant})

    return queries[:n]


def main():
    docs = load_docs()
    sys.stderr.write(f"loaded {len(docs)} docs\n")

    selected = pick_fat_docs(docs, TARGET)
    sys.stderr.write(f"selected {len(selected)} fattest docs (max len: {len(selected[0]['text'])} chars)\n")

    queries = build_queries(selected, docs)
    sys.stderr.write(f"built {len(queries)} queries\n")

    corpus = {
        "docs": [{"id": d["id"], "type": d["type"], "text": d["text"],
                   "metadata": d.get("metadata", {})} for d in selected],
        "queries": queries,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(corpus, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(OUT_PATH)
    sys.stderr.write(f"wrote {OUT_PATH}\n")
    print(json.dumps({"docs": len(corpus["docs"]), "queries": len(corpus["queries"])}))


if __name__ == "__main__":
    main()
