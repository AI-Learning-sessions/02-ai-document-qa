import pymupdf
import chromadb
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("GEMINI_API_KEY not found.")
    exit()

gemini_client = genai.Client(api_key=api_key)


pdf_path = "data/network_protocols.pdf"

document = pymupdf.open(pdf_path)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = []

for page_number, page in enumerate(document):
    text = page.get_text().strip()

    if not text:
        continue

    page_chunks = text_splitter.split_text(text)

    for chunk in page_chunks:
        chunks.append({
            "text": chunk,
            "page": page_number + 1,
            "source": "network_protocols.pdf"
        })


chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="network_protocols"
)


documents = []
ids = []
metadatas = []

for i, chunk in enumerate(chunks):
    documents.append(chunk["text"])
    ids.append(f"chunk_{i + 1}")

    metadatas.append({
        "source": chunk["source"],
        "page": chunk["page"]
    })


collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print("Documents stored:", collection.count())

while True:
    query = input("\nAsk a question about the document (or type 'exit'): ").strip()

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

    # Filter results by distance
    DISTANCE_THRESHOLD = 1.4

    filtered_documents = []
    filtered_metadatas = []

    for i, distance in enumerate(results["distances"][0]):
        if distance <= DISTANCE_THRESHOLD:
            filtered_documents.append(results["documents"][0][i])
            filtered_metadatas.append(results["metadatas"][0][i])

    # No relevant information found
    if not filtered_documents:
        print("\nI could not find relevant information in the document.")
        continue

    # Combine retrieved chunks
    context = "\n\n".join(filtered_documents)

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

    # Generate answer using Gemini
    try:
        response = gemini_client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        print("\nAnswer:")
        print(response.output_text)

        print("\nSources:")

        seen_sources = set()

        for metadata in filtered_metadatas:
            source = f"{metadata['source']} — Page {metadata['page']}"

            if source not in seen_sources:
                print(f"- {source}")
                seen_sources.add(source)

    except Exception as e:
        print("\nGemini API error:")
        print(e)