from rag.retriever import retrieve
from rag.prompts import build_prompt
from rag.llm import generate


def ask(question, metadata_filter=None):

    results = retrieve(question, metadata_filter=metadata_filter)

    documents = results["documents"][0]
    ids = results["ids"][0]
    distances = results["distances"][0]

    context = "\n\n---\n\n".join(
        documents
    )

    prompt = build_prompt(
        question,
        context
    )

    answer = generate(prompt)

    return {
        "answer": answer,
        "documents": documents,
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