from src.config import (
    DISTANCE_THRESHOLD,
    RETRIEVAL_K,
    MAX_CONTEXT_CHUNKS
)

from src.embeddings import create_embeddings


def retrieve_relevant_chunks(
    collection,
    query
):
    """
    Retrieve relevant document chunks for a query.

    Returns:
        documents: Relevant text chunks.
        metadatas: Source and page information.
        distances: Chroma similarity distances.
    """
    query_embedding = create_embeddings(
        [query]
    )[0]

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=RETRIEVAL_K
    )

    filtered_documents = []
    filtered_metadatas = []
    filtered_distances = []

    seen_documents = set()

    for i, distance in enumerate(
        results["distances"][0]
    ):

        if distance > DISTANCE_THRESHOLD:
            continue

        document = results["documents"][0][i]
        metadata = results["metadatas"][0][i]

        if document in seen_documents:
            continue

        seen_documents.add(document)

        filtered_documents.append(document)
        filtered_metadatas.append(metadata)
        filtered_distances.append(distance)

        if len(filtered_documents) >= MAX_CONTEXT_CHUNKS:
            break

    return (
        filtered_documents,
        filtered_metadatas,
        filtered_distances
    )