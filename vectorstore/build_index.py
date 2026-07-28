import json
import chromadb

from embeddings.llama_embeddings import embed_batch, validate_embeddings
from vectorstore.hashes import embedding_hash, metadata_hash

BATCH_SIZE = 5000
EMBED_BATCH_SIZE = 1000


def load_documents():
    with open("data/documents.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _get_existing_dim(collection):
    # lightweight — fetches full vector only to check its length
    existing = collection.get(limit=1, include=["embeddings"])
    if len(existing["ids"]) > 0 and len(existing["embeddings"]) > 0:
        emb = existing["embeddings"][0]
        if emb is not None and len(emb) > 0:
            return len(emb)
    return None


def build_index(documents=None, chroma_path="data/chroma"):
    if documents is None:
        documents = load_documents()

    client = chromadb.PersistentClient(
        path=chroma_path
    )

    collection = client.get_or_create_collection(
        name="project_gorgon"
    )

    expected_dim = _get_existing_dim(collection)
    if expected_dim is not None:
        print(f"Existing collection dimension: {expected_dim}")
    else:
        print("No existing collection — skipping dimension check")

    existing = collection.get(
        include=["metadatas"]
    )

    existing_hashes = {}

    for doc_id, metadata in zip(
        existing["ids"],
        existing["metadatas"]
    ):
        if "embedding_hash" in metadata:
            existing_hashes[doc_id] = {
                "embedding_hash": metadata["embedding_hash"],
                "metadata_hash": metadata["metadata_hash"],
            }

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
    metadata_only_updates = []

    for doc in documents:
        doc_id = doc["id"]
        doc_embed_hash = embedding_hash(doc)
        doc_meta_hash = metadata_hash(doc)

        existing = existing_hashes.get(doc_id)

        if existing is not None:
            existing_embed_hash = existing.get("embedding_hash")
            existing_meta_hash = existing.get("metadata_hash")

            if existing_embed_hash == doc_embed_hash and existing_meta_hash == doc_meta_hash:
                continue

            if existing_embed_hash == doc_embed_hash and existing_meta_hash != doc_meta_hash:
                metadata_only_updates.append(doc)
                continue

        documents_to_embed.append(doc)

    print(
        f"Need to embed {len(documents_to_embed)} of {len(documents)} documents"
    )

    if metadata_only_updates:
        print(f"Metadata-only updates: {len(metadata_only_updates)} documents")

        meta_ids = []
        meta_metadatas = []

        for doc in metadata_only_updates:
            metadata = dict(doc["metadata"])
            metadata["type"] = doc["type"]
            metadata["embedding_hash"] = embedding_hash(doc)
            metadata["metadata_hash"] = metadata_hash(doc)

            meta_ids.append(doc["id"])
            meta_metadatas.append(metadata)

        for i in range(0, len(meta_ids), BATCH_SIZE):
            print(
                f"Updating metadata {i} - {min(i + BATCH_SIZE, len(meta_ids))}"
            )

            collection.update(
                ids=meta_ids[i:i + BATCH_SIZE],
                metadatas=meta_metadatas[i:i + BATCH_SIZE]
            )

    ids = []
    embeddings = []
    texts = []
    metadatas = []

    if not documents_to_embed:
        print("No documents need embedding.")
    else:
        print(
            f"Embedding {len(documents_to_embed)} documents..."
        )

        for start in range(0, len(documents_to_embed), EMBED_BATCH_SIZE):

            batch = documents_to_embed[start:start + EMBED_BATCH_SIZE]

            print(
                f"Embedding {start}/{len(documents_to_embed)}"
            )

            batch_embeddings = embed_batch(
                [doc["text"] for doc in batch]
            )

            if expected_dim is not None:
                validate_embeddings(batch_embeddings, expected_dim=expected_dim)

            for doc, embedding in zip(batch, batch_embeddings):

                ids.append(doc["id"])
                embeddings.append(embedding)
                texts.append(doc["text"])

                metadata = dict(
                    doc["metadata"]
                )

                metadata["type"] = doc["type"]
                metadata["embedding_hash"] = embedding_hash(doc)
                metadata["metadata_hash"] = metadata_hash(doc)

                metadatas.append(metadata)

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
