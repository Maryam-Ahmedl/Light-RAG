"""
Minimal JSON-file KV store.

Mirrors LightRAG's KV_STORAGE role: holds raw text chunks, per-document
processing status (for incremental updates / dedup), and an optional LLM
response cache. Swap this class for a Redis/SQLite backend later without
touching ingest.py or query.py --- that's the whole point of the interface.
"""

import json
import os
from pathlib import Path


class KVStore:
    def __init__(self, storage_dir: str, name: str):
        self.path = Path(storage_dir) / f"{name}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value):
        self._data[key] = value
        self.save()

    def has(self, key: str) -> bool:
        return key in self._data

    def all(self) -> dict:
        return self._data