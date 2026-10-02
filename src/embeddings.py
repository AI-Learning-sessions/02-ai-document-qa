from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


model = SentenceTransformer(
    EMBEDDING_MODEL
)


def create_embeddings(texts):
    return model.encode(texts)