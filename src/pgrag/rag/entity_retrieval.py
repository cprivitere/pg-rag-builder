import json
import logging
import re
from pathlib import Path

from pgrag.config import CONTEXT_BUDGET
from pgrag.rag.retriever import retrieve

logger = logging.getLogger(__name__)

FACET_PLANS = {
    "skill": [
        ("recipes", "recipe"),
        ("quests", "quest"),
        ("trainers", "npc"),
        ("advancement", "advancementtable"),
        ("XP requirements", "xptable"),
        ("abilities", "ability"),
    ],
    "item": [
        ("recipes that use", "recipe"),
        ("uses", "itemuse"),
        ("sources", "source"),
    ],
    "ability": [
        ("skill", "skill"),
        ("keywords", "abilitykeyword"),
    ],
    "quest": [
        ("rewards", "item"),
        ("requirements", "abilitykeyword"),
        ("location", "npc"),
    ],
    "recipe": [
        ("ingredients", "item"),
        ("skill", "skill"),
        ("results", "item"),
    ],
    "effect": [
        ("items", "item"),
    ],
    "area": [
        ("NPCs", "npc"),
        ("quests", "quest"),
    ],
    "npc": [
        ("services", "skill"),
        ("quests", "quest"),
    ],
}

_DOCS_CACHE = None


def _load_docs():
    global _DOCS_CACHE
    path = Path("data/documents.json")
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return None
    if _DOCS_CACHE is None or _DOCS_CACHE[0] != mtime:
        _DOCS_CACHE = (mtime, json.loads(path.read_text(encoding="utf-8")))
    return _DOCS_CACHE[1]


def _hub_chunks(hub_id, docs):
    chunks = []
    for doc in docs:
        doc_id = doc["id"]
        if doc_id == hub_id or doc_id.startswith(hub_id + "_chunk_"):
            chunks.append(doc)
    chunks.sort(key=lambda d: int(d["metadata"].get("chunk_index", -1)))
    return chunks


def _entity_type(hub_id):
    for prefix, dtype in [
        ("skillprofile_", "skill"),
        ("item_", "item"),
        ("ability_", "ability"),
        ("quest_", "quest"),
        ("recipe_", "recipe"),
        ("effect_", "effect"),
        ("area_", "area"),
        ("npc_", "npc"),
    ]:
        if hub_id.startswith(prefix):
            return dtype
    return None


def _entity_name_from_hub(hub_id):
    """Extract human-readable entity name from hub_id."""
    for prefix in ("skillprofile_", "item_", "ability_", "quest_",
                    "recipe_", "effect_", "area_", "npc_"):
        if hub_id.startswith(prefix):
            return hub_id[len(prefix):]
    return hub_id


def build_entity_context(question, hub_id):
    docs = _load_docs()
    if not docs:
        return None
    hub_chunks = _hub_chunks(hub_id, docs)
    if not hub_chunks:
        return None

    dtype = _entity_type(hub_id)
    entity_name = _entity_name_from_hub(hub_id)

    ids = [d["id"] for d in hub_chunks]
    texts = [d["text"] for d in hub_chunks]
    metas = [d["metadata"] for d in hub_chunks]
    dists = [0.0] * len(hub_chunks)

    seen = set(ids)

    rerank_used = False
    FACET_COUNTS = {"recipe": 20}
    for facet_q, ftype in FACET_PLANS.get(dtype, []):
        facet_query = f"{facet_q} {entity_name}"
        try:
            res = retrieve(
                facet_query,
                count=FACET_COUNTS.get(ftype, 10),
                metadata_filter={"type": ftype},
                hybrid=True,
                rerank=True,
            )
        except Exception as exc:
            logger.warning(
                "facet %r failed for hub %r: %s", facet_q, hub_id, exc
            )
            continue
        rerank_used = rerank_used or res.get("rerank_used", False)
        for i in range(len(res["ids"][0])):
            did = res["ids"][0][i]
            if did in seen:
                continue
            seen.add(did)
            ids.append(did)
            texts.append(res["documents"][0][i])
            metas.append(res["metadatas"][0][i])
            dists.append(res["distances"][0][i])

    if dtype == "skill":
        # Put low-level recipes first so the LLM sees what is usable at the
        # player's target level, and truncation keeps the relevant ones.
        hub_count = len(hub_chunks)
        heads = list(zip(ids, texts, metas, dists))[:hub_count]
        tails = list(zip(ids, texts, metas, dists))[hub_count:]

        def _req_level(item):
            m = re.search(r"Required Skill Level:\s*\n(\d+)", item[1])
            return int(m.group(1)) if m else 10**9

        recipes = sorted(
            (t for t in tails if t[2].get("type") == "recipe"),
            key=_req_level,
        )
        others = [t for t in tails if t[2].get("type") != "recipe"]
        ordered = heads + recipes + others
        if ordered:
            ids, texts, metas, dists = (
                list(x) for x in zip(*ordered)
            )

    total = 0
    cut = len(texts)
    for i, t in enumerate(texts):
        if total + len(t) > CONTEXT_BUDGET and i > 0:
            cut = i
            break
        total += len(t)

    return {
        "ids": [ids[:cut]],
        "documents": [texts[:cut]],
        "metadatas": [metas[:cut]],
        "distances": [dists[:cut]],
        "rerank_used": rerank_used,
    }
