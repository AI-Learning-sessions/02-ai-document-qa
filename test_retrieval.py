from src.config import DISTANCE_THRESHOLD
from src.vector_store import get_collection
from src.embeddings import create_embeddings


collection = get_collection()


while True:

    query = input(
        "\nEnter a question "
        "(or type 'exit'): "
    ).strip()


    if query.lower() == "exit":
        break


    # Convert question into an embedding
    query_embedding = create_embeddings(
        [query]
    )[0]


    # Retrieve top 3 results
    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=3
    )


    print("\nRaw retrieval results:")


    for i in range(
        len(results["documents"][0])
    ):

        distance = results["distances"][0][i]

        metadata = results["metadatas"][0][i]

        document = results["documents"][0][i]


        print(f"\n--- Result {i + 1} ---")

        print(
            f"Distance: {distance:.4f}"
        )

        print(
            f"Source: {metadata['source']}"
        )

        print(
            f"Page: {metadata['page']}"
        )

        print("Document:")

        print(document)


    print(
        "\n\nFiltered results "
        f"(threshold = {DISTANCE_THRESHOLD}):"
    )


    found_relevant = False


    for i in range(
        len(results["documents"][0])
    ):

        distance = results["distances"][0][i]


        if distance <= DISTANCE_THRESHOLD:

            found_relevant = True

            metadata = results["metadatas"][0][i]

            document = results["documents"][0][i]


            print(
                f"\n--- Relevant Result {i + 1} ---"
            )

            print(
                f"Distance: {distance:.4f}"
            )

            print(
                f"Page: {metadata['page']}"
            )

            print("Document:")

            print(document)


    if not found_relevant:

        print(
            "\nNo relevant chunks found."
        )