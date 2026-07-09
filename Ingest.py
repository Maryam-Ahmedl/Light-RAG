import hashlib
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from Chunking import chunk_text
from Extraction import extract_entities_and_relationships
from LLM import EMBEDDING_DIM, embed
from storage.graph_store import GraphStore
from storage.Kv_store import KVStore
from storage.vector_store import VectorStore
from storage.latency_log import log_ingest

load_dotenv()
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage_data")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ingest_file(filepath: str, chunk_size: int = 300, overlap: int = 50) -> dict:
    doc_start = time.perf_counter()

    text = Path(filepath).read_text(encoding="utf-8")
    doc_hash = _hash(text)

    doc_status = KVStore(STORAGE_DIR, "doc_status")
    chunk_kv = KVStore(STORAGE_DIR, "chunks")
    graph = GraphStore(STORAGE_DIR)
    chunk_vectors = VectorStore(STORAGE_DIR, "chunks", EMBEDDING_DIM)
    entity_vectors = VectorStore(STORAGE_DIR, "entities", EMBEDDING_DIM)

    if doc_status.get(doc_hash, {}).get("status") == "done":
        return {"status": "skipped", "reason": "already ingested", "doc_hash": doc_hash}

    doc_status.set(doc_hash, {"status": "processing", "file": str(filepath)})

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    new_entity_names = set()

    extraction_times = []
    embed_times = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_hash}-{i}"
        chunk_kv.set(chunk_id, {"text": chunk, "doc_hash": doc_hash})

        t0 = time.perf_counter()
        extracted = extract_entities_and_relationships(chunk)
        extraction_times.append(time.perf_counter() - t0)

        for entity in extracted["entities"]:
            name = entity.get("name", "").strip()
            if not name:
                continue
            graph.upsert_entity(name, entity.get("type", "unknown"), entity.get("description", ""))
            new_entity_names.add(name)

        for rel in extracted["relationships"]:
            source, target = rel.get("source", "").strip(), rel.get("target", "").strip()
            if not source or not target:
                continue
            graph.upsert_relationship(source, target, rel.get("description", ""))

        t0 = time.perf_counter()
        chunk_embedding = embed([chunk])
        embed_times.append(time.perf_counter() - t0)
        chunk_vectors.add(chunk_embedding, [{"chunk_id": chunk_id, "text": chunk}])

        print(f"  chunk {i+1}/{len(chunks)}  extract={extraction_times[-1]:.2f}s  embed={embed_times[-1]:.2f}s")

    graph.save()

    entity_texts, entity_meta = [], []
    for name in new_entity_names:
        entity = graph.get_entity(name)
        if entity:
            entity_texts.append(f"{name}: {entity.get('description', '')}")
            entity_meta.append({"name": name})

    entity_embed_seconds = 0.0
    if entity_texts:
        t0 = time.perf_counter()
        entity_embeddings = embed(entity_texts)
        entity_embed_seconds = time.perf_counter() - t0
        entity_vectors.add(entity_embeddings, entity_meta)

    doc_status.set(doc_hash, {"status": "done", "file": str(filepath), "chunks": len(chunks)})

    total_seconds = time.perf_counter() - doc_start
    avg_extraction = sum(extraction_times) / len(extraction_times) if extraction_times else 0.0
    avg_embed = sum(embed_times) / len(embed_times) if embed_times else 0.0

    log_ingest(
        file=str(filepath),
        doc_hash=doc_hash,
        chunks=len(chunks),
        timings={
            "avg_extraction_seconds": round(avg_extraction, 4),
            "avg_chunk_embed_seconds": round(avg_embed, 4),
            "entity_embed_seconds": round(entity_embed_seconds, 4),
            "total_seconds": round(total_seconds, 2),
        },
    )

    return {
        "status": "ingested",
        "doc_hash": doc_hash,
        "chunks": len(chunks),
        "entities": len(new_entity_names),
        "total_seconds": round(total_seconds, 2),
    }


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "data/sample.txt"
    result = ingest_file(target)
    print(result)