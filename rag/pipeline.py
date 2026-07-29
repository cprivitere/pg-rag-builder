import chromadb

from embeddings.llama_embeddings import embed_text
from rag.query_classifier import classify_query
from rag.retriever import retrieve
from rag.prompts import build_prompt
from rag.llm import generate


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


def ask(question, metadata_filter=None):

    query_type = classify_query(question)

    results = retrieve(question, metadata_filter=metadata_filter, query_type=query_type)

    documents = results["documents"][0]
    ids = results["ids"][0]
    distances = results["distances"][0]

    if query_type == "comparison":
        summary = _find_matching_summary(question)
        if summary:
            documents = [summary] + documents

    context = "\n\n---\n\n".join(
        documents
    )

    prompt = build_prompt(
        question,
        context,
        query_type=query_type
    )

    answer = generate(prompt)

    return {
        "answer": answer,
        "documents": documents,
        "query_type": query_type,
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
                results["metadatas"][0]
            )
        ]
    }
