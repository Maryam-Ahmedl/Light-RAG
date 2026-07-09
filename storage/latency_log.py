import csv
from datetime import datetime
from pathlib import Path

QUERY_LOG_PATH = Path("logs/query_latency.csv")
INGEST_LOG_PATH = Path("logs/ingest_latency.csv")

QUERY_FIELDS = [
    "timestamp", "mode", "question",
    "embedding_seconds", "retrieval_seconds",
    "ttft_seconds", "llm_seconds", "end_to_end_seconds",
]

INGEST_FIELDS = [
    "timestamp", "file", "doc_hash", "chunks",
    "avg_extraction_seconds", "avg_chunk_embed_seconds",
    "entity_embed_seconds", "total_seconds",
]


def _append_row(path: Path, fields: list[str], row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def log_query(mode: str, question: str, timings: dict):
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "question": question,
        **timings,
    }
    _append_row(QUERY_LOG_PATH, QUERY_FIELDS, row)


def log_ingest(file: str, doc_hash: str, chunks: int, timings: dict):
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "file": file,
        "doc_hash": doc_hash,
        "chunks": chunks,
        **timings,
    }
    _append_row(INGEST_LOG_PATH, INGEST_FIELDS, row)


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _stats(values: list[float]) -> dict:
    if not values:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    return {"avg": sum(values) / len(values), "min": min(values), "max": max(values)}


def _trend_arrow(older_avg: float, newer_avg: float) -> str:
    if older_avg == 0:
        return "–"
    change = ((newer_avg - older_avg) / older_avg) * 100
    if abs(change) < 5:
        return f"→ ({change:+.1f}%)"
    return f"{'↑' if change > 0 else '↓'} ({change:+.1f}%)"


def query_report(last_n: int = None) -> str:
    """Summarize query latency: overall stats per stage, plus a trend
    comparing the first half vs second half of the (optionally windowed)
    history, so you can see if things are getting faster or slower."""
    rows = _read_rows(QUERY_LOG_PATH)
    if not rows:
        return "No query latency data logged yet."

    if last_n:
        rows = rows[-last_n:]

    fields = ["embedding_seconds", "retrieval_seconds", "ttft_seconds", "llm_seconds", "end_to_end_seconds"]
    lines = [f"Query Latency Report  ({len(rows)} queries)", "=" * 50]

    mid = len(rows) // 2
    older, newer = rows[:mid], rows[mid:] if mid > 0 else rows

    for field in fields:
        values = [float(r[field]) for r in rows if r.get(field)]
        stats = _stats(values)

        older_vals = [float(r[field]) for r in older if r.get(field)]
        newer_vals = [float(r[field]) for r in newer if r.get(field)]
        trend = _trend_arrow(_stats(older_vals)["avg"], _stats(newer_vals)["avg"]) if older else "–"

        label = field.replace("_seconds", "").replace("_", " ").upper()
        lines.append(
            f"{label:12s} avg={stats['avg']:.3f}s  min={stats['min']:.3f}s  "
            f"max={stats['max']:.3f}s  trend={trend}"
        )

    return "\n".join(lines)


def ingest_report() -> str:
    rows = _read_rows(INGEST_LOG_PATH)
    if not rows:
        return "No ingestion latency data logged yet."

    lines = [f"Ingestion Latency Report  ({len(rows)} documents)", "=" * 50]
    for r in rows:
        lines.append(
            f"{r['file']:30s} chunks={r['chunks']:>4s}  "
            f"avg_extract={float(r['avg_extraction_seconds']):.3f}s  "
            f"avg_embed={float(r['avg_chunk_embed_seconds']):.3f}s  "
            f"entity_embed={float(r['entity_embed_seconds']):.3f}s  "
            f"total={float(r['total_seconds']):.2f}s"
        )
    return "\n".join(lines)