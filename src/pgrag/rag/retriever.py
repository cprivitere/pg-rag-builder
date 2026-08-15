import logging

import chromadb

from pgrag.embeddings.llama_embeddings import embed_text
from pgrag.rag.spelling import correct_query

logger = logging.getLogger(__name__)

RERANK_MULTIPLIER = 3
RRF_K = 60
HYBRID_MULTIPLIER = 3


def _term_overlap(query, document):
    query_terms = set(query.lower().split())
    if not query_terms:
        return 0.0
    doc_text = document.lower()
    matches = sum(1 for t in query_terms if t in doc_text)
    return matches / len(query_terms)


def _rerank(query, ids, documents, metadatas, distances, count):
    scored = []
    for rank, (doc_id, doc, meta, dist) in enumerate(
        zip(ids, documents, metadatas, distances)
    ):
        orig_score = 1.0 / (rank + 1)
        term_score = _term_overlap(query, doc)
        scored.append((orig_score + term_score, doc_id, doc, meta, dist))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:count]
    return (
        [t[1] for t in top],
        [t[2] for t in top],
        [t[3] for t in top],
        [t[4] for t in top],
    )


def _hybrid_fuse(dense_ids, dense_texts, dense_metadatas, dense_distances,
                  bm25_ids, all_docs, count):
    dense_info = {}
    for rank, (doc_id, text, meta, dist) in enumerate(
        zip(dense_ids, dense_texts, dense_metadatas, dense_distances)
    ):
        dense_info[doc_id] = (rank, text, meta, dist)

    bm25_ranks = {doc_id: rank for rank, doc_id in enumerate(bm25_ids)}
    doc_lookup = {d["id"]: d for d in all_docs}

    all_ids = set(dense_ids) | set(bm25_ranks.keys())
    fallback_rank = max(len(dense_ids), len(bm25_ids)) * 2

    scored = []
    for doc_id in all_ids:
        dr = dense_info[doc_id][0] if doc_id in dense_info else fallback_rank
        br = bm25_ranks.get(doc_id, fallback_rank)
        rrf = 1.0 / (RRF_K + dr + 1) + 1.0 / (RRF_K + br + 1)
        scored.append((rrf, doc_id))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_ids = [s[1] for s in scored[:count]]

    ids, texts, metas, dists = [], [], [], []
    for doc_id in top_ids:
        if doc_id in dense_info:
            _, text, meta, dist = dense_info[doc_id]
        else:
            doc = doc_lookup.get(doc_id, {})
            text = doc.get("text", "")
            meta = doc.get("metadata", {})
            dist = 0.0
        ids.append(doc_id)
        texts.append(text)
        metas.append(meta)
        dists.append(dist)

    return ids, texts, metas, dists


def retrieve(question, count=3, metadata_filter=None, rerank=True, hybrid=False, query_type="general"):
    question = correct_query(question)
    client = chromadb.PersistentClient(
        path="data/chroma"
    )

    collection = client.get_collection(
        name="project_gorgon"
    )

    embedding = embed_text(question)

    effective_count = 20 if query_type == "comparison" else (count or 3)

    dense_count = effective_count
    if hybrid:
        dense_count *= HYBRID_MULTIPLIER
    if rerank and not hybrid:
        dense_count *= RERANK_MULTIPLIER

    query_kwargs = dict(
        query_embeddings=[embedding],
        n_results=dense_count,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    if metadata_filter is not None:
        query_kwargs["where"] = metadata_filter

    results = collection.query(**query_kwargs)
    results["rerank_used"] = False

    if hybrid:
        from pgrag.rag.bm25 import load_bm25_index
        fuse_target = effective_count * RERANK_MULTIPLIER if rerank else effective_count
        bm25_model, all_docs = load_bm25_index()
        bm25_indices, _ = bm25_model.search(question, k=fuse_target)
        bm25_ids = [all_docs[i]["id"] for i in bm25_indices]

        fused_ids, fused_texts, fused_metas, fused_dists = _hybrid_fuse(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            bm25_ids,
            all_docs,
            fuse_target,
        )
        results["ids"] = [fused_ids]
        results["documents"] = [fused_texts]
        results["metadatas"] = [fused_metas]
        results["distances"] = [fused_dists]

    # Post-fusion metadata filter (BM25 ignores where clause)
    if metadata_filter is not None and results["ids"][0]:
        filtered = [
            i for i in range(len(results["ids"][0]))
            if all(results["metadatas"][0][i].get(k) == v
                   for k, v in metadata_filter.items())
        ]
        results["ids"] = [[results["ids"][0][i] for i in filtered]]
        results["documents"] = [[results["documents"][0][i] for i in filtered]]
        results["metadatas"] = [[results["metadatas"][0][i] for i in filtered]]
        results["distances"] = [[results["distances"][0][i] for i in filtered]]

    if rerank and len(results["ids"][0]) > effective_count:
        ids, docs, metas, dists, used = _rerank_or_cross_encoder(
            question,
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            effective_count,
        )
        results["rerank_used"] = used
        results["ids"] = [ids]
        results["documents"] = [docs]
        results["metadatas"] = [metas]
        results["distances"] = [dists]

    return results


def _rerank_or_cross_encoder(query, ids, documents, metadatas, distances, count):
    """Cross-encoder rerank via :8082; lexical fallback on failure (V60, V61).

    Returns reordered quads + used flag.
    """
    try:
        from pgrag.rag.reranker_client import rerank_documents, record_failure, record_success

        indices = rerank_documents(query, documents, count)
        reranked = (
            [ids[i] for i in indices],
            [documents[i] for i in indices],
            [metadatas[i] for i in indices],
            [distances[i] for i in indices],
        )
        record_success()
        return reranked + (True,)
    except Exception as exc:
        logger.warning("[WARN] reranker (:8082) unavailable/failed: %s", exc)
        try:  # stats update must not break retrieval (V64)
            record_failure()
        except Exception:
            pass
        return _rerank(query, ids, documents, metadatas, distances, count) + (False,)
