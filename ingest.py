import pymupdf
import chromadb

from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = "data/network_protocols.pdf"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "network_protocols"


# -----------------------------
# 1. Load PDF
# -----------------------------

print("Loading PDF...")

document = pymupdf.open(PDF_PATH)

print("Number of pages:", len(document))


# -----------------------------
# 2. Create text splitter
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


# -----------------------------
# 3. Create chunks
# -----------------------------

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


print("Chunks created:", len(chunks))


# -----------------------------
# 4. Create persistent Chroma
# -----------------------------

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


# -----------------------------
# 5. Create collection
# -----------------------------

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


# -----------------------------
# 6. Prepare data
# -----------------------------

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


# -----------------------------
# 7. Store chunks
# -----------------------------

collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)


print("Documents stored:", collection.count())
print("Ingestion completed successfully.")