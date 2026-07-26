import chromadb


def main():

    client = chromadb.PersistentClient(
        path="data/chroma"
    )

    collection = client.get_collection(
        name="project_gorgon"
    )

    result = collection.get(
        ids=["recipe_24401"],
        include=["metadatas"]
    )

    print(result["metadatas"][0])


if __name__ == "__main__":
    main()