from database import GameDatabase
from loaders.cdn_loader import load_database
from loaders.wiki_loader import load_wiki
from documents.builder import build_documents


def main():

    print("Hello from pg-rag-builder!")

    db = GameDatabase()

    load_database(db)
    load_wiki(db)

    documents = build_documents(db)

    import json

    with open("documents.json", "w", encoding="utf-8") as f:
        json.dump(
            documents,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("Saved documents.json")

    query = "Bunny Juice"

    matches = [
        doc for doc in documents
        if query.lower() in doc["text"].lower()
    ]

    print(f"Found {len(matches)} matches")

    for doc in matches[:5]:
        print("\n---")
        print(doc["text"])

    print(f"Created {len(documents)} documents")



if __name__ == "__main__":
    main()