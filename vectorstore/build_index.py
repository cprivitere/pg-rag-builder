import json
import chromadb

from embeddings.llama_embeddings import embed_batch
from vectorstore.hashes import document_hash

def load_documents():
    with open("data/documents.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_index():
    documents = load_documents()

    client = chromadb.PersistentClient(
        path="data/chroma"
    )

    collection = client.get_or_create_collection(
        name="project_gorgon"
    )

    existing = collection.get(
        include=["metadatas"]
    )

    existing_hashes = {}

    for doc_id, metadata in zip(
        existing["ids"],
        existing["metadatas"]
    ):
        existing_hashes[doc_id] = metadata["hash"]


    current_ids = set(
        doc["id"] for doc in documents
    )

    existing_ids = set(
        existing["ids"]
    )

    deleted_ids = existing_ids - current_ids

    if deleted_ids:
        print(f"Deleting {len(deleted_ids)} removed documents")

        collection.delete(
            ids=list(deleted_ids)
        )
    else:
        print("No deleted documents found")

    documents_to_embed = []

    for doc in documents:
        doc_id = doc["id"]
        doc_hash = document_hash(doc)

        if existing_hashes.get(doc_id) == doc_hash:
            continue

        documents_to_embed.append(doc)

    print(
        f"Need to embed {len(documents_to_embed)} "
        f"of {len(documents)} documents"
    )

    ids = []
    embeddings = []
    texts = []
    metadatas = []

    batch_size = 1000

    if not documents_to_embed:
        print("No documents need embedding.")

    else:
        print(
            f"Embedding {len(documents_to_embed)} documents..."
        )

        for start in range(0, len(documents_to_embed), batch_size):

            batch = documents_to_embed[start:start + batch_size]

            print(
                f"Embedding {start}/{len(documents_to_embed)}"
            )

            batch_embeddings = embed_batch(
                [doc["text"] for doc in batch]
            )

            for doc, embedding in zip(batch, batch_embeddings):

                ids.append(doc["id"])
                embeddings.append(embedding)
                texts.append(doc["text"])

                metadata = dict(
                    doc["metadata"]
                )

                metadata["type"] = doc["type"]
                metadata["hash"] = document_hash(doc)

                metadatas.append(metadata)

    BATCH_SIZE = 5000

    for i in range(0, len(ids), BATCH_SIZE):
        print(f"Adding vectors {i} - {min(i+BATCH_SIZE, len(ids))}")

        collection.upsert(
            ids=ids[i:i+BATCH_SIZE],
            embeddings=embeddings[i:i+BATCH_SIZE],
            documents=texts[i:i+BATCH_SIZE],
            metadatas=metadatas[i:i+BATCH_SIZE]
        )

    print("Done.")


if __name__ == "__main__":
    build_index()