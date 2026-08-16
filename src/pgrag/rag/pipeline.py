import logging
import re

import chromadb
from pathlib import Path

from pgrag.embeddings.llama_embeddings import embed_text
from pgrag.rag.query_classifier import classify_query, find_entity
from pgrag.rag.entity_retrieval import build_entity_context
from pgrag.rag.retriever import retrieve
from pgrag.rag.bm25 import load_bm25_index
from pgrag.rag.prompts import build_prompt
from pgrag.rag.llm import generate, stream_generate
from pgrag.rag.synthesis_detector import should_synthesize
from pgrag.rag.synthesis_generator import synthesize_answer, create_curated_doc

logger = logging.getLogger(__name__)

CURATED_DIR = Path("data/wiki/curated")

# Wiki page expansion bounds (Step 5): at most 2 pages, ~16k chars of
# appended context per general query.
_EXPAND_MAX_PAGES = 2
_EXPAND_MAX_CHARS = 16000
_EXPAND_PLACEHOLDER_DIST = 1.0

_doc_store = None
_parent_index = None


def _load_parent_index():
    """id -> doc + parent_id -> [docs], lazily from the persisted BM25 doc
    store (data/bm25_index.pkl). Built once per process."""
    global _doc_store, _parent_index
    if _doc_store is not None:
        return _doc_store, _parent_index
    _, all_docs = load_bm25_index()
    _doc_store = {d["id"]: d for d in all_docs}
    index = {}
    for d in all_docs:
        pid = d.get("metadata", {}).get("parent_id")
        if pid:
            index.setdefault(pid, []).append(d)
    _parent_index = index
    return _doc_store, _parent_index


def _expand_wiki_context(ids, documents, metadatas, distances, store):
    """Pull sibling chunks of wiki pages hit by retrieval.

    A general query retrieves only the top chunks of a page; the answer often
    sits in another section of the same page. Expands at most
    `_EXPAND_MAX_PAGES` pages, id-deduped, bounded to `_EXPAND_MAX_CHARS` of
    appended text. Appended docs get `_EXPAND_PLACEHOLDER_DIST`; all four
    lists stay index-aligned. Non-wiki docs are untouched.
    """
    parent_ids = []
    seen = set()
    for meta in metadatas:
        if not isinstance(meta, dict):
            continue
        parent = meta.get("parent_id")
        if parent and parent not in seen:
            seen.add(parent)
            parent_ids.append(parent)
        if len(parent_ids) >= _EXPAND_MAX_PAGES:
            break

    if not parent_ids:
        return ids, documents, metadatas, distances

    parents = {pid: [] for pid in parent_ids}
    for pid in parent_ids:
        for doc in store.values():
            dmeta = doc.get("metadata", {})
            if isinstance(dmeta, dict) and dmeta.get("parent_id") == pid:
                parents[pid].append(doc)

    existing = set(ids)
    per_parent = {}  # parent -> (added ids, texts, metas, dists)
    used_chars = 0

    for pid in parent_ids:
        added_ids, added_texts, added_metas, added_dists = [], [], [], []
        for doc in parents.get(pid, []):
            if doc["id"] in existing:
                continue
            if used_chars + len(doc["text"]) > _EXPAND_MAX_CHARS:
                continue
            existing.add(doc["id"])
            added_ids.append(doc["id"])
            added_texts.append(doc["text"])
            added_metas.append(doc["metadata"])
            added_dists.append(_EXPAND_PLACEHOLDER_DIST)
            used_chars += len(doc["text"])
        if added_ids:
            per_parent[pid] = (
                added_ids, added_texts, added_metas, added_dists,
            )

    if not per_parent:
        return ids, documents, metadatas, distances

    # Splice each page's siblings right after its first retrieved chunk so
    # the page stays contiguous and the specific answer is not buried at the
    # end of the context.
    result_ids, result_texts, result_metas, result_dists = [], [], [], []
    spliced = set()
    for i in range(len(ids)):
        result_ids.append(ids[i])
        result_texts.append(documents[i])
        result_metas.append(metadatas[i])
        result_dists.append(distances[i])
        pid = (
            metadatas[i].get("parent_id")
            if isinstance(metadatas[i], dict) else None
        )
        if pid in per_parent and pid not in spliced:
            spliced.add(pid)
            a_ids, a_texts, a_metas, a_dists = per_parent[pid]
            result_ids.extend(a_ids)
            result_texts.extend(a_texts)
            result_metas.extend(a_metas)
            result_dists.extend(a_dists)

    # A parent not present in the retrieved list is impossible (parents come
    # from retrieved metadata), but append defensively.
    for pid, (a_ids, a_texts, a_metas, a_dists) in per_parent.items():
        if pid not in spliced:
            result_ids.extend(a_ids)
            result_texts.extend(a_texts)
            result_metas.extend(a_metas)
            result_dists.extend(a_dists)

    return result_ids, result_texts, result_metas, result_dists

MISSING_REGEX = re.compile(
    r"\b(i do not know|not found|no information|unable to find)\b",
    re.IGNORECASE,
)
SUBJECT_REGEX = re.compile(
    r"i do not know (?:how to|about) ([^.,!?\n]+)",
    re.IGNORECASE,
)


def _persist_synthesized(doc):
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    slug = doc["id"].replace("synthesized_", "", 1)
    slug = re.sub(r'[<>:"/\\|?*]', "-", slug)
    path = CURATED_DIR / f"synthesized_{slug}_curated.txt"
    path.write_text(doc["text"], encoding="utf-8")


_SUMMARY_CANDIDATES = 10

# Query terms hinting at a gathering/harvesting question
_GATHERING_TERMS = {
    "mushroom", "mushrooms", "gather", "gathering", "harvest", "harvestable",
    "pick", "pickable", "forage", "foraging", "mine", "mining", "fish",
    "fishing", "mycology", "tanning", "collect", "collecting",
}

# Query terms hinting at a crafting/recipe question — these must NOT route to
# gathering summaries, even when they share a skill word (e.g. "Mycology
# recipe" is about recipes, not harvesting).
_CRAFTING_TERMS = {
    "recipe", "recipes", "craft", "crafting", "cook", "cooking", "make",
    "makeable", "learn", "train", "ability", "abilities", "craftable",
}


def _summary_score(question, doc, meta, dist):
    """Score a summary candidate: lexical overlap + domain preference - distance."""
    query_terms = {t for t in re.findall(r"[a-z]+", question.lower()) if len(t) > 2}
    text_lower = doc.lower()
    overlap = sum(1 for t in query_terms if t in text_lower)

    name_lower = str(meta.get("name", "")).lower()
    score = overlap - dist

    has_gathering = bool(query_terms & _GATHERING_TERMS)
    has_crafting = bool(query_terms & _CRAFTING_TERMS)
    gathering_named = "gathering" in name_lower

    if has_gathering and not has_crafting and gathering_named:
        score += 3.0
    elif has_crafting and gathering_named:
        # Crafting questions want recipe summaries, not harvest tables
        score -= 2.0
    # Wiki-derived summaries carry curated, complete tables
    if "wiki" in name_lower:
        score += 1.0

    return score


def _find_matching_summary(question):
    """Find the most relevant summary document for a comparison query."""
    client = chromadb.PersistentClient(path="data/chroma")
    collection = client.get_collection(name="project_gorgon")

    embedding = embed_text(question)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=_SUMMARY_CANDIDATES,
        where={"type": "summary"},
    )

    if not results["ids"][0]:
        return None

    best, best_score = None, None
    for doc_id, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = _summary_score(question, doc, meta, dist)
        if best_score is None or score > best_score:
            best, best_score = doc, score
    return best


def _generate_with(question, documents, query_type):
    context = "\n\n---\n\n".join(documents)
    prompt = build_prompt(question, context, query_type=query_type)
    return generate(prompt)


def _build_sources(ids, distances, metadatas):
    return [
        {
            "id": doc_id,
            "distance": distance,
            "metadata": metadata,
            "citation": f"{metadata.get('name', doc_id)} ({metadata.get('table', 'unknown')})"
        }
        for doc_id, distance, metadata in zip(ids, distances, metadatas)
    ]


def _gap_fill(question, answer, ids, docs, metas, dists, query_type):
    """One-shot targeted re-retrieval on the missing subject (V36).
    Bare "I do not know." carries no subject -> fall back to the question itself.
    Empty answer also counts as missing.
    """
    if (answer or "").strip() and not MISSING_REGEX.search(answer or ""):
        return answer, ids, docs, metas, dists, False
    if not (answer or "").strip():
        answer = _generate_with(question, docs, query_type)
        if (answer or "").strip() and not MISSING_REGEX.search(answer or ""):
            return answer, ids, docs, metas, dists, False
    m = SUBJECT_REGEX.search(answer)
    subject = m.group(1).strip() if m else question
    extra = retrieve(
        f"{subject} {question}",
        count=5,
        hybrid=True,
        rerank=True,
    )
    seen = set(ids)
    for i in range(len(extra["ids"][0])):
        did = extra["ids"][0][i]
        if did in seen:
            continue
        seen.add(did)
        ids.append(did)
        docs.append(extra["documents"][0][i])
        metas.append(extra["metadatas"][0][i])
        dists.append(extra["distances"][0][i])
    answer = _generate_with(question, docs, query_type)
    return answer, ids, docs, metas, dists, extra.get("rerank_used", False)


def _prepare_entity(question):
    """Retrieve an entity dossier without generating. Returns
    (ids, docs, metas, dists, rerank_used) or None if no hub found."""
    hub_id, _ = find_entity(question)
    ctx = build_entity_context(question, hub_id)
    if ctx is None:
        return None
    return (
        list(ctx["ids"][0]),
        list(ctx["documents"][0]),
        list(ctx["metadatas"][0]),
        list(ctx["distances"][0]),
        ctx.get("rerank_used", False),
    )


def _ask_entity(question):
    prepared = _prepare_entity(question)
    if prepared is None:
        return None
    ids, docs, metas, dists, rerank_used = prepared

    answer = _generate_with(question, docs, "entity")
    answer, ids, docs, metas, dists, gap_used = _gap_fill(
        question, answer, ids, docs, metas, dists, "entity"
    )

    return {
        "answer": answer,
        "documents": docs,
        "query_type": "entity",
        "rerank_used": rerank_used or gap_used,
        "sources": _build_sources(ids, dists, metas),
    }


def _prepare_general(question, query_type, metadata_filter=None):
    """Retrieve (and possibly synthesize) context for a general/comparison
    query. Returns (query_type, ids, documents, metadatas, distances,
    rerank_used). Does not call the LLM."""
    # "lookup" queries ("Where can I find X?") need the same wide hybrid
    # recall + wiki expansion as "general" — an answer is a fact, not a
    # comparison, so a 3-doc dense-only window starves it.
    is_wide = query_type in ("general", "lookup")
    results = retrieve(
        question,
        metadata_filter=metadata_filter,
        query_type=query_type,
        count=20 if is_wide else 3,
        hybrid=is_wide,
    )

    documents = results["documents"][0]
    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    _expanded = False
    if is_wide and any(
        isinstance(m, dict) and m.get("parent_id") for m in metadatas
    ):
        # Wiki page expansion (Step 5): pull sibling chunks via parent_id so
        # "how-to" answers aren't lost in a non-retrieved chunk.
        store, _ = _load_parent_index()
        before = len(ids)
        ids, documents, metadatas, distances = _expand_wiki_context(
            ids, documents, metadatas, distances, store,
        )
        _expanded = len(ids) > before

    if query_type == "comparison":
        summary = _find_matching_summary(question)
        if summary:
            documents = [summary] + documents

    # Check if synthesis should be triggered. When wiki expansion spliced
    # sibling chunks into the context, the assembled page IS the coherent
    # answer — synthesis over the top-3 retrieved docs (often stale curated
    # summaries) would throw the specific row away.
    result_dicts = [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]

    if should_synthesize(result_dicts, query_type) and not _expanded:
        # Synthesize scattered results
        try:
            synthesized = synthesize_answer(question, result_dicts[:3])  # Limit to 3 sources
            # Persist so future queries benefit (V24) — survive purge via curated dir (V20)
            curated_doc = create_curated_doc(question, result_dicts[:3], synthesized_text=synthesized)
            _persist_synthesized(curated_doc)
            # Use synthesized as context instead of raw docs
            documents = [synthesized]
        except Exception as exc:
            logger.warning(
                "synthesis failed for %r (query_type=%s): %s",
                question,
                query_type,
                exc,
            )

    return query_type, ids, documents, metadatas, distances, results.get("rerank_used", False)


def _stream_generation(question, documents, query_type):
    context = "\n\n---\n\n".join(documents)
    prompt = build_prompt(question, context, query_type=query_type)
    return stream_generate(prompt)


def _stream_answer(question, ids, docs, metas, dists, query_type):
    """Stream the answer (and any gap-fill re-answer) for a prepared context.

    Yields {"type": "token", "text"} deltas and {"type": "reset"} before a
    gap-fill regeneration. Returns (answer, ids, docs, metas, dists,
    gap_used) once the final generation completes.
    """
    answer = ""
    for delta in _stream_generation(question, docs, query_type):
        answer += delta
        yield {"type": "token", "text": delta}

    if (answer or "").strip() and not MISSING_REGEX.search(answer or ""):
        return answer, ids, docs, metas, dists, False

    if not (answer or "").strip():
        answer = ""
        yield {"type": "reset"}
        for delta in _stream_generation(question, docs, query_type):
            answer += delta
            yield {"type": "token", "text": delta}
        if (answer or "").strip() and not MISSING_REGEX.search(answer or ""):
            return answer, ids, docs, metas, dists, False

    m = SUBJECT_REGEX.search(answer)
    subject = m.group(1).strip() if m else question
    extra = retrieve(
        f"{subject} {question}",
        count=5,
        hybrid=True,
        rerank=True,
    )
    seen = set(ids)
    for i in range(len(extra["ids"][0])):
        did = extra["ids"][0][i]
        if did in seen:
            continue
        seen.add(did)
        ids.append(did)
        docs.append(extra["documents"][0][i])
        metas.append(extra["metadatas"][0][i])
        dists.append(extra["distances"][0][i])

    answer = ""
    yield {"type": "reset"}
    for delta in _stream_generation(question, docs, query_type):
        answer += delta
        yield {"type": "token", "text": delta}
    return answer, ids, docs, metas, dists, extra.get("rerank_used", False)


def ask_stream(question, metadata_filter=None):
    """Streaming variant of ask().

    Yields {"type": "token", "text"} deltas (with {"type": "reset"} between
    gap-fill re-answers) and finally {"type": "final", "result": {...}}
    where result mirrors ask()'s return value.
    """
    query_type = classify_query(question)

    if query_type == "entity":
        prepared = _prepare_entity(question)
        if prepared is not None:
            ids, docs, metas, dists, rerank_used = prepared
            answer, ids, docs, metas, dists, gap_used = yield from _stream_answer(
                question, ids, docs, metas, dists, "entity"
            )
            yield {"type": "final", "result": {
                "answer": answer,
                "documents": docs,
                "query_type": "entity",
                "rerank_used": rerank_used or gap_used,
                "sources": _build_sources(ids, dists, metas),
            }}
            return
        query_type = "general"

    qt, ids, documents, metadatas, distances, rerank_used = _prepare_general(
        question, query_type, metadata_filter
    )
    answer, ids, documents, metadatas, distances, gap_used = yield from _stream_answer(
        question, ids, documents, metadatas, distances, qt
    )
    yield {"type": "final", "result": {
        "answer": answer,
        "documents": documents,
        "query_type": qt,
        "rerank_used": rerank_used or gap_used,
        "sources": [
            {
                "id": doc_id,
                "distance": distance,
                "metadata": metadata,
                "citation": f"{metadata.get('name', doc_id)} ({metadata.get('table', 'unknown')})"
            }
            for doc_id, distance, metadata in zip(
                ids,
                distances,
                metadatas
            )
        ],
    }}


def ask(question, metadata_filter=None):

    query_type = classify_query(question)

    if query_type == "entity":
        prepared = _prepare_entity(question)
        if prepared is not None:
            return _ask_entity(question)
        query_type = "general"

    qt, ids, documents, metadatas, distances, rerank_used = _prepare_general(
        question, query_type, metadata_filter
    )

    context = "\n\n---\n\n".join(documents)

    prompt = build_prompt(
        question,
        context,
        query_type=qt
    )

    answer = generate(prompt)
    answer, ids, documents, metadatas, distances, gap_used = _gap_fill(
        question, answer, ids, documents, metadatas, distances, qt
    )

    return {
        "answer": answer,
        "documents": documents,
        "query_type": qt,
        "rerank_used": rerank_used or gap_used,
        "sources": [
            {
                "id": doc_id,
                "distance": distance,
                "metadata": metadata,
                "citation": f"{metadata.get('name', doc_id)} ({metadata.get('table', 'unknown')})"
            }
            for doc_id, distance, metadata in zip(
                ids,
                distances,
                metadatas
            )
        ]
    }
