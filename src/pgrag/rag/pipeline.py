import logging
import re

import chromadb
from pathlib import Path

from pgrag.embeddings.llama_embeddings import embed_text
from pgrag.rag.query_classifier import classify_query, find_entity
from pgrag.rag.entity_retrieval import build_entity_context
from pgrag.rag.retriever import retrieve
from pgrag.rag.prompts import build_prompt
from pgrag.rag.llm import generate
from pgrag.rag.synthesis_detector import should_synthesize
from pgrag.rag.synthesis_generator import synthesize_answer, create_curated_doc

logger = logging.getLogger(__name__)

CURATED_DIR = Path("data/wiki/curated")

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


def _find_matching_summary(question):
    """Find the most relevant summary document for a comparison query."""
    client = chromadb.PersistentClient(path="data/chroma")
    collection = client.get_collection(name="project_gorgon")

    embedding = embed_text(question)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=5,
        where={"type": "summary"},
    )

    if results["ids"][0]:
        return results["documents"][0][0]
    return None


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


def _ask_entity(question):
    hub_id, _ = find_entity(question)
    ctx = build_entity_context(question, hub_id)
    if ctx is None:
        return None

    ids = list(ctx["ids"][0])
    docs = list(ctx["documents"][0])
    metas = list(ctx["metadatas"][0])
    dists = list(ctx["distances"][0])

    answer = _generate_with(question, docs, "entity")
    answer, ids, docs, metas, dists, gap_used = _gap_fill(
        question, answer, ids, docs, metas, dists, "entity"
    )

    return {
        "answer": answer,
        "documents": docs,
        "query_type": "entity",
        "rerank_used": ctx.get("rerank_used", False) or gap_used,
        "sources": _build_sources(ids, dists, metas),
    }


def ask(question, metadata_filter=None):

    query_type = classify_query(question)

    if query_type == "entity":
        result = _ask_entity(question)
        if result is not None:
            return result
        query_type = "general"

    results = retrieve(
        question,
        metadata_filter=metadata_filter,
        query_type=query_type,
        count=20 if query_type == "general" else 3,
        hybrid=query_type == "general",
    )

    documents = results["documents"][0]
    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    if query_type == "comparison":
        summary = _find_matching_summary(question)
        if summary:
            documents = [summary] + documents

    # Check if synthesis should be triggered
    result_dicts = [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]
    
    if should_synthesize(result_dicts, query_type):
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

    context = "\n\n---\n\n".join(
        documents
    )

    prompt = build_prompt(
        question,
        context,
        query_type=query_type
    )

    answer = generate(prompt)
    answer, ids, documents, distances, metadatas, gap_used = _gap_fill(
        question, answer, ids, documents, distances, metadatas, query_type
    )

    return {
        "answer": answer,
        "documents": documents,
        "query_type": query_type,
        "rerank_used": results.get("rerank_used", False) or gap_used,
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
