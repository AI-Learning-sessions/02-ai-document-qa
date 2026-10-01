from src.document_loader import load_and_split_document
from src.vector_store import store_chunks


print("Loading and processing PDF...")

chunks = load_and_split_document()

print("Chunks created:", len(chunks))


collection = store_chunks(chunks)

print("Documents stored:", collection.count())

print("Ingestion completed successfully.")