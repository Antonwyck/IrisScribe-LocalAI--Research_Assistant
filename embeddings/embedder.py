from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.astype("float32").reshape(1, -1)