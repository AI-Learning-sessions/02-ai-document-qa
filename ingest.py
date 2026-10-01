import pymupdf
import chromadb

from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = "data/network_protocols.pdf"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "network_protocols"


print("Loading PDF...")

document = pymupdf.open(PDF_PATH)

print("Number of pages:", len(document))


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


print("Chunks created:", len(chunks))


print("Creating persistent Chroma database...")

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
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


collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)


print("Documents stored:", collection.count())

print("Ingestion completed successfully.")