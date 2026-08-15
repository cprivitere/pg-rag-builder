from pgrag.config import WIKI_DIR


def load_wiki(db):
    db.wiki_mtimes = {}
    for file in sorted(WIKI_DIR.glob("*.txt")):
        page_name = file.stem
        db.wiki[page_name] = file.read_text(encoding="utf-8")
        db.wiki_mtimes[page_name] = file.stat().st_mtime

    print(f"Loaded {len(db.wiki)} wiki pages.")
