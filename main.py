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


query = "What are the seven layers of the OSI model?"

results = collection.query(
    query_texts=[query],
    n_results=3
)

context = "\n\n".join(results["documents"][0])

print("\nSearch results:")

for i, document in enumerate(results["documents"][0]):
    print(f"\n--- Result {i + 1} ---")

    print("Document:")
    print(document)

    print("\nID:")
    print(results["ids"][0][i])

    print("\nMetadata:")
    print(results["metadatas"][0][i])

    print("\nDistance:")
    print(results["distances"][0][i])

prompt = f"""
You are a document question-answering assistant.

Answer the question using only the provided context.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{query}

Answer:
"""

response = gemini_client.interactions.create(
    model="gemini-3.6-flash",
    input=prompt
)

print("\nAnswer:")
print(response.output_text)