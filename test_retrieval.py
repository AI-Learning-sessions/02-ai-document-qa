from src.vector_store import get_collection


collection = get_collection()


while True:

    query = input(
        "\nEnter a question "
        "(or type 'exit'): "
    ).strip()


    if query.lower() == "exit":
        break


    results = collection.query(
        query_texts=[query],
        n_results=3
    )


    print("\nRetrieved chunks:")


    for i in range(len(results["documents"][0])):

        print(f"\n--- Result {i + 1} ---")

        print(
            "Distance:",
            results["distances"][0][i]
        )

        print(
            "Source:",
            results["metadatas"][0][i]["source"]
        )

        print(
            "Page:",
            results["metadatas"][0][i]["page"]
        )

        print(
            "Document:"
        )

        print(
            results["documents"][0][i]
        )