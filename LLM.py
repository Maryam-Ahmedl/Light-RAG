import os
from dotenv import load_dotenv
from openai import OpenAI
import time

load_dotenv()

_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))


def chat(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    response = _client.chat.completions.create(
        model=LLM_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""

def chat_with_latency(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    result = chat_stream_timed(system_prompt, user_prompt, temperature)
    print(f"[TTFT: {result['ttft_seconds']:.3f}s]")
    return result["text"]

def chat_stream_timed(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict:
    start = time.perf_counter()
    first_token_time = None
    chunks = []
    stream = _client.chat.completions.create(
        model=LLM_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            chunks.append(delta)
    end = time.perf_counter()
    return {
        "text": "".join(chunks),
        "ttft_seconds": (first_token_time - start) if first_token_time else 0.0,
        "total_seconds": end - start,
    }


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    BATCH_LIMIT = 100
    all_embeddings = []
    for i in range(0, len(texts), BATCH_LIMIT):
        batch = texts[i : i + BATCH_LIMIT]
        response = _client.embeddings.create(
            model=EMBEDDING_MODEL, input=batch, dimensions=EMBEDDING_DIM
        )
        all_embeddings.extend(item.embedding for item in response.data)
    return all_embeddings
