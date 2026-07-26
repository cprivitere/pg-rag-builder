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

    with open("data/documents.json", "w", encoding="utf-8") as f:
        json.dump(
            documents,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("Saved documents.json")

    print(f"Created {len(documents)} documents")



if __name__ == "__main__":
    main()