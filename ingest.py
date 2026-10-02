from src.document_loader import load_and_split_document
from src.vector_store import store_chunks


def main():

    print("Loading and processing PDF...")

    try:

        chunks = load_and_split_document()

    except (FileNotFoundError, ValueError) as e:

        print(f"\nIngestion error: {e}")
        return

    print(
        "Chunks created:",
        len(chunks)
    )

    try:

        collection = store_chunks(chunks)

    except Exception as e:

        print(
            "\nVector store error:",
            e
        )
        return

    print(
        "Documents stored:",
        collection.count()
    )

    print(
        "Ingestion completed successfully."
    )


if __name__ == "__main__":
    main()