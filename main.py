from src.vector_store import get_collection
from src.retriever import retrieve_relevant_chunks
from src.llm import create_client, generate_answer


def main():

    try:

        collection = get_collection()

    except Exception as e:

        print(
            "Could not connect to the vector database."
        )
        print(f"Error: {e}")
        return

    print(
        "Documents available:",
        collection.count()
    )

    if collection.count() == 0:

        print(
            "The vector database is empty."
        )

        print(
            "Run 'python ingest.py' first."
        )

        return

    try:

        llm_client = create_client()

    except ValueError as e:

        print(e)
        return

    while True:

        query = input(
            "\nAsk a question about the document "
            "(or type 'exit'): "
        ).strip()

        if query.lower() == "exit":

            print("Goodbye!")
            break

        if not query:

            print(
                "Please enter a question."
            )
            continue

        documents, metadatas, distances = (
            retrieve_relevant_chunks(
                collection,
                query
            )
        )

        if not documents:

            print(
                "\nI could not find relevant "
                "information in the document."
            )

            continue

        context = "\n\n".join(documents)

        try:

            answer = generate_answer(
                llm_client,
                query,
                context
            )

            print("\nAnswer:")
            print(answer)

            print("\nSources:")

            seen_sources = set()

            for metadata in metadatas:

                source = (
                    f"{metadata['source']} "
                    f"— Page {metadata['page']}"
                )

                if source not in seen_sources:

                    print(
                        f"- {source}"
                    )

                    seen_sources.add(
                        source
                    )

        except Exception as e:

            print(
                "\nGemini API error:"
            )

            print(e)


if __name__ == "__main__":
    main()