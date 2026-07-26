import json


def load_documents():
    with open("documents.json", "r", encoding="utf-8") as f:
        return json.load(f)

def search_documents(documents, query, doc_type=None):
    query = query.lower().strip()

    results = []

    for doc in documents:

        if doc_type and doc.get("type") != doc_type:
            continue

        text = doc["text"].lower()

        if query in text:
            results.append(doc)

    return results

if __name__ == "__main__":
    documents = load_documents()

    query = input("Search: ")

    results = search_documents(
        documents,
        query
    )

    print(f"\nFound {len(results)} results\n")

    for doc in results[:5]:
        print("---")
        print("TYPE:", doc["type"])
        print("ID:", doc["id"])
        print()
        print(doc["text"])