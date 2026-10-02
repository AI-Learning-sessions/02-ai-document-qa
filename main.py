from src.vector_store import get_collection
from src.retriever import retrieve_relevant_chunks
from src.llm import create_client, generate_answer


# Connect to ChromaDB
collection = get_collection()

print(
    "Documents available:",
    collection.count()
)


# Create LLM client
try:

    llm_client = create_client()

except ValueError as e:

    print(e)
    exit()


# Question-answering loop
while True:

    query = input(
        "\nAsk a question about the document "
        "(or type 'exit'): "
    ).strip()


    if not query:

        print("Please enter a question.")

        continue


    if query.lower() == "exit":

        print("Goodbye!")

        break


    # Retrieve relevant chunks
    documents, metadatas, distances = (
        retrieve_relevant_chunks(
            collection,
            query
        )
    )


    # No relevant information
    if not documents:

        print(
            "\nI could not find relevant information "
            "in the document."
        )

        continue


    # Combine retrieved chunks
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

                print(f"- {source}")

                seen_sources.add(source)


    except Exception as e:

        print("\nGemini API error:")

        print(e)