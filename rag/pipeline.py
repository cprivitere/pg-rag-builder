from rag.query_classifier import classify_query
from rag.retriever import retrieve
from rag.prompts import build_prompt
from rag.llm import generate


def ask(question, metadata_filter=None):

    query_type = classify_query(question)

    results = retrieve(question, metadata_filter=metadata_filter, query_type=query_type)

    documents = results["documents"][0]
    ids = results["ids"][0]
    distances = results["distances"][0]

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
