import sys
from database import GameDatabase
from loaders.cdn_loader import load_database
from loaders.wiki_loader import load_wiki
from documents.builder import build_documents

sys.stdout.reconfigure(line_buffering=True)


def main():

    print("Hello from pg-rag-builder!")

    db = GameDatabase()

    load_database(db)
    load_wiki(db)

    documents = build_documents(db)

    import json
    import os

    tmp = "data/documents.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            documents,
            f,
            indent=2,
            ensure_ascii=False
        )
    os.replace(tmp, "data/documents.json")

    print("Saved documents.json")

    print(f"Created {len(documents)} documents")



if __name__ == "__main__":
    main()