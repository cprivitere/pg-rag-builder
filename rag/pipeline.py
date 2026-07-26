from rag.retriever import retrieve
from rag.prompts import build_prompt
from rag.llm import generate


def ask(question):

    results = retrieve(question)

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
                "metadata": metadata
            }
            for doc_id, distance, metadata in zip(
                ids,
                distances,
                results["metadatas"][0]
            )
        ]
    }