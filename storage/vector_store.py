"""
Minimal local vector store, no external service required.

Stores vectors as a numpy array plus a parallel list of metadata dicts,
persisted to disk. Good enough for thousands of chunks/entities; swap for
FAISS/Chroma/pgvector once you outgrow brute-force cosine search.
"""

import json
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, storage_dir: str, name: str, dim: int):
        self.dim = dim
        self.vec_path = Path(storage_dir) / f"{name}_vectors.npy"
        self.meta_path = Path(storage_dir) / f"{name}_meta.json"
        self.vec_path.parent.mkdir(parents=True, exist_ok=True)

        if self.vec_path.exists() and self.meta_path.exists():
            self.vectors = np.load(self.vec_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata: list[dict] = json.load(f)
        else:
            self.vectors = np.zeros((0, dim), dtype=np.float32)
            self.metadata = []

    def save(self):
        np.save(self.vec_path, self.vectors)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def add(self, vectors: list[list[float]], metadata: list[dict]):
        if not vectors:
            return
        new_vecs = np.array(vectors, dtype=np.float32)
        self.vectors = (
            new_vecs if self.vectors.size == 0 else np.vstack([self.vectors, new_vecs])
        )
        self.metadata.extend(metadata)
        self.save()

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        if self.vectors.shape[0] == 0:
            return []

        q = np.array(query_vector, dtype=np.float32)
        # cosine similarity = dot product of L2-normalized vectors
        vec_norms = np.linalg.norm(self.vectors, axis=1)
        q_norm = np.linalg.norm(q)
        vec_norms[vec_norms == 0] = 1e-10
        q_norm = q_norm if q_norm != 0 else 1e-10

        sims = (self.vectors @ q) / (vec_norms * q_norm)
        top_k = min(top_k, len(sims))
        top_indices = np.argsort(-sims)[:top_k]

        return [
            {**self.metadata[i], "score": float(sims[i])} for i in top_indices
        ]