documents = build_documents(db)

print(len(documents))

for doc in documents[:3]:
    print(doc)