import chromadb

from src.config import (
    CHROMA_PATH,
    COLLECTION_NAME
)

from src.embeddings import create_embeddings


def get_collection():

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection


def store_chunks(chunks):

    collection = get_collection()

    documents = []
    ids = []
    metadatas = []


    for i, chunk in enumerate(chunks):

        documents.append(
            chunk["text"]
        )

        ids.append(
            f"chunk_{i + 1}"
        )

        metadatas.append({
            "source": chunk["source"],
            "page": chunk["page"]
        })


    print("Creating embeddings...")

    embeddings = create_embeddings(
        documents
    )


    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )


    return collection