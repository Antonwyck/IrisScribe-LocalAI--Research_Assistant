import faiss
import numpy as np
import pickle


class FaissStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.text_chunks = []
        self.metadata = []

    def add(self, embeddings: np.ndarray, chunks: list[str], metadata: list[dict]):
        if len(embeddings) != len(chunks) or len(chunks) != len(metadata):
            raise ValueError("Embeddings, chunks, and metadata must have the same length.")

        self.index.add(embeddings)
        self.text_chunks.extend(chunks)
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 3, fetch_k: int = 8):
        distances, indices = self.index.search(query_embedding, fetch_k)

        results = []
        seen_chunks = set()
        source_counts = {}

        for rank, idx in enumerate(indices[0]):
            if not (0 <= idx < len(self.text_chunks)):
                continue

            chunk = self.text_chunks[idx]
            meta = self.metadata[idx]
            source = meta.get("source", "unknown")

            normalized_chunk = " ".join(chunk.split()).strip().lower()
            if normalized_chunk in seen_chunks:
                continue

            # Limit repeated chunks from same source
            source_counts.setdefault(source, 0)
            if source_counts[source] >= 1:
                continue

            results.append({
                "chunk": chunk,
                "metadata": meta,
                "distance": float(distances[0][rank])
            })

            seen_chunks.add(normalized_chunk)
            source_counts[source] += 1

            if len(results) >= top_k:
                break

        return results

    def remove_source(self, source_name: str):
        keep_chunks = []
        keep_metadata = []
        keep_indices = []

        for i, meta in enumerate(self.metadata):
            if meta.get("source") != source_name:
                keep_chunks.append(self.text_chunks[i])
                keep_metadata.append(meta)
                keep_indices.append(i)

        if not keep_chunks:
            self.index.reset()
            self.text_chunks = []
            self.metadata = []
            return

        # rebuild embeddings subset
        vectors = self.index.reconstruct_n(0, self.index.ntotal)
        vectors = vectors[keep_indices]

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)

        self.text_chunks = keep_chunks
        self.metadata = keep_metadata

    def save(self, index_path: str, chunks_path: str, metadata_path: str):
        faiss.write_index(self.index, index_path)

        with open(chunks_path, "wb") as f:
            pickle.dump(self.text_chunks, f)

        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, index_path: str, chunks_path: str, metadata_path: str):
        self.index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:
            self.text_chunks = pickle.load(f)

        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        if len(self.text_chunks) != len(self.metadata):
            raise ValueError("Loaded chunks and metadata have different lengths.")