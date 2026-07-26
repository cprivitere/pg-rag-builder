import chromadb

from embeddings.llama_embeddings import embed_text


def retrieve(question, count=3):
    client = chromadb.PersistentClient(
        path="data/chroma"
    )

    collection = client.get_collection(
        name="project_gorgon"
    )

    embedding = embed_text(question)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=count,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return results