import json
import os
import sys

from pgrag.loaders.database import GameDatabase
from pgrag.loaders.cdn_loader import load_database
from pgrag.loaders.wiki_loader import load_wiki
from pgrag.documents.builder import build_documents

sys.stdout.reconfigure(line_buffering=True)


def generate_documents() -> None:
    db = GameDatabase()

    load_database(db)
    load_wiki(db)

    documents = build_documents(db)

    tmp = "data/documents.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            documents,
            f,
            indent=2,
            ensure_ascii=False,
        )
    os.replace(tmp, "data/documents.json")

    print("Saved documents.json")
    print(f"Created {len(documents)} documents")