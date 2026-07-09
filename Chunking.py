def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        chunks.append(chunk)
        start += step
    return chunks