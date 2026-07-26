from database import GameDatabase
from loaders.cdn_loader import load_database
from loaders.wiki_loader import load_wiki

def main():
    print("Hello from pg-rag-builder!")
    db = GameDatabase()

    load_database(db)
    load_wiki(db)

    print(db.tables.keys())
    print(db.tables.keys())
    print(len(db.wiki))

if __name__ == "__main__":
    main()
