"""
Query engine with four retrieval modes, mirroring LightRAG's core idea:

- naive:  plain vector RAG over text chunks. Baseline for comparison.
- local:  vector search finds the most relevant ENTITIES, then pulls their
          direct neighborhood from the graph. Good for specific, factual
          questions ("who does X report to?").
- global: like local but expands 2 hops out, gathering the broader web of
          relationships. Good for thematic questions ("what are the main
          conflicts in this document?").
- hybrid: combines local graph context with naive chunk retrieval, so the
          answer has both structured relationships AND raw supporting text.
"""
import time
import os
from dotenv import load_dotenv
from LLM import EMBEDDING_DIM, chat_stream_timed, embed
from storage.graph_store import GraphStore
from storage.vector_store import VectorStore
from storage.latency_log import log_query

load_dotenv()
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage_data")

ANSWER_SYSTEM_PROMPT = """You answer questions using ONLY the provided context.
If the context doesn't contain the answer, say so plainly rather than guessing.
Be concise and cite which part of the context (entity, relationship, or chunk)
your answer relies on."""


def _load_stores():
    graph = GraphStore(STORAGE_DIR)
    chunk_vectors = VectorStore(STORAGE_DIR, "chunks", EMBEDDING_DIM)
    entity_vectors = VectorStore(STORAGE_DIR, "entities", EMBEDDING_DIM)
    return graph, chunk_vectors, entity_vectors


def _format_entity_context(entities: list[dict]) -> str:
    lines = []
    for e in entities:
        line = f"- {e['name']} ({e.get('type', 'unknown')}): {e.get('description', '')}"
        if "relationship_description" in e:
            line += f"  [related to {e.get('relationship_to')}: {e['relationship_description']}]"
        lines.append(line)
    return "\n".join(lines) if lines else "(none found)"


def _format_chunk_context(chunks: list[dict]) -> str:
    return "\n---\n".join(c["text"] for c in chunks) if chunks else "(none found)"



def query(question: str, mode: str = "hybrid", top_k: int = 5) -> dict:
    overall_start = time.perf_counter()
    graph, chunk_vectors, entity_vectors = _load_stores()

    t0 = time.perf_counter()
    query_vector = embed([question])[0]
    embedding_seconds = time.perf_counter() - t0

    context_parts = []
    raw_contexts = [] 
    t0 = time.perf_counter()

    if mode in ("naive", "hybrid", "mix"):
        chunks = chunk_vectors.search(query_vector, top_k=top_k)
        context_parts.append("## Retrieved text chunks\n" + _format_chunk_context(chunks))
        raw_contexts.extend(c["text"] for c in chunks)
    if mode in ("local", "hybrid", "global", "mix"):
        matched_entities = entity_vectors.search(query_vector, top_k=top_k)
        entity_context = []
        seen = set()

        hops = 2 if mode == "global" else 1
        for match in matched_entities:
            name = match["name"]
            entity = graph.get_entity(name)
            if entity and name not in seen:
                entity_context.append(entity)
                seen.add(name)
            for neighbor in graph.get_neighbors(name, hops=hops):
                if neighbor["name"] not in seen:
                    entity_context.append(neighbor)
                    seen.add(neighbor["name"])

        context_parts.append("## Knowledge graph context\n" + _format_entity_context(entity_context))
        raw_contexts.extend(f"{e['name']}: {e.get('description','')}" for e in entity_context)
        
    retrieval_seconds = time.perf_counter() - t0
    full_context = "\n\n".join(context_parts)
    prompt = f"Context:\n{full_context}\n\nQuestion: {question}"
    llm_result = chat_stream_timed(ANSWER_SYSTEM_PROMPT, prompt)
    answer = llm_result["text"]
    end_to_end_seconds = time.perf_counter() - overall_start

    timings = {
        "embedding_seconds": round(embedding_seconds, 4),
        "retrieval_seconds": round(retrieval_seconds, 4),
        "ttft_seconds": round(llm_result["ttft_seconds"], 4),
        "llm_seconds": round(llm_result["total_seconds"], 4),
        "end_to_end_seconds": round(end_to_end_seconds, 4),
    }
    log_query(mode, question, timings)
    return {"mode": mode, "answer": answer, "context": full_context, "timings": timings, "contexts": raw_contexts}


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"
    m = sys.argv[2] if len(sys.argv) > 2 else "hybrid"
    result = query(q, mode=m)
    print(f"\n[mode={result['mode']}]\n{result['answer']}\n")