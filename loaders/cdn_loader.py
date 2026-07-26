import json
from pathlib import Path

from config import CDN_DIR


def load_database(db):
    """
    Load every JSON file found in the CDN directory.

    Each file becomes:

        db.tables["items"]
        db.tables["recipes"]
        db.tables["skills"]
        ...

    automatically.
    """

    for file in sorted(CDN_DIR.glob("*.json")):

        table_name = file.stem.lower()

        print(f"Loading {table_name}...")

        with open(file, "r", encoding="utf-8") as f:

            db.tables[table_name] = json.load(f)

    print(f"Loaded {len(db.tables)} tables.")