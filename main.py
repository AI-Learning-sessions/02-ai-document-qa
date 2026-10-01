import os

import chromadb

from dotenv import load_dotenv
from google import genai


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "network_protocols"

DISTANCE_THRESHOLD = 1.4


# Load environment variables
load_dotenv()


# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("GEMINI_API_KEY not found.")
    exit()


# Create Gemini client
gemini_client = genai.Client(
    api_key=api_key
)


# Connect to persistent Chroma database
chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


print("Documents available:", collection.count())


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
    results = collection.query(
        query_texts=[query],
        n_results=3
    )


    # Filter results based on distance
    filtered_documents = []
    filtered_metadatas = []


    for i, distance in enumerate(results["distances"][0]):

        if distance <= DISTANCE_THRESHOLD:

            filtered_documents.append(
                results["documents"][0][i]
            )

            filtered_metadatas.append(
                results["metadatas"][0][i]
            )


    # No relevant information found
    if not filtered_documents:

        print(
            "\nI could not find relevant information "
            "in the document."
        )

        continue


    # Combine retrieved chunks
    context = "\n\n".join(
        filtered_documents
    )


    # Create prompt
    prompt = f"""
You are a document question-answering assistant.

Answer the question using only the provided context.

Rules:
- Do not use outside knowledge.
- If the answer cannot be found in the context, say:
  "I could not find the answer in the provided document."
- Keep the answer clear and concise.

Context:
{context}

Question:
{query}

Answer:
"""


    try:

        # Send request to Gemini
        response = gemini_client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )


        print("\nAnswer:")
        print(response.output_text)


        # Display sources
        print("\nSources:")

        seen_sources = set()


        for metadata in filtered_metadatas:

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