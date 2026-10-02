from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


model = SentenceTransformer(
    EMBEDDING_MODEL
)


def create_embeddings(texts):
    """
    Convert text into vector embeddings.

    Args:
        texts: List of text strings.

    Returns:
        NumPy array containing the embeddings.
    """
    return model.encode(texts)