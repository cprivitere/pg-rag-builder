from config import WIKI_DIR


def load_wiki(db):
    for file in sorted(WIKI_DIR.glob("*.txt")):
        page_name = file.stem
        db.wiki[page_name] = file.read_text(encoding="utf-8")

    print(f"Loaded {len(db.wiki)} wiki pages.")
