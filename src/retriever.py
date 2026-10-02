from src.config import DISTANCE_THRESHOLD
from src.embeddings import create_embeddings


def retrieve_relevant_chunks(
    collection,
    query,
    n_results=3
):

    query_embedding = create_embeddings(
        [query]
    )[0]


    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=n_results
    )


    filtered_documents = []
    filtered_metadatas = []
    filtered_distances = []


    for i, distance in enumerate(
        results["distances"][0]
    ):

        if distance <= DISTANCE_THRESHOLD:

            filtered_documents.append(
                results["documents"][0][i]
            )

            filtered_metadatas.append(
                results["metadatas"][0][i]
            )

            filtered_distances.append(
                distance
            )


    return (
        filtered_documents,
        filtered_metadatas,
        filtered_distances
    )