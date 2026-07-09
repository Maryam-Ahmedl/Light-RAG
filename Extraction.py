import json
import re
from LLM import chat

EXTRACTION_SYSTEM_PROMPT = """You extract entities and relationships from text \
for a knowledge graph. Read the text and identify:

1. Entities: important people, organizations, concepts, places, or things.
2. Relationships: how two entities are connected, described in one sentence.

Respond with ONLY valid JSON, no markdown fences, no commentary, in this \
exact shape:

{
  "entities": [
    {"name": "...", "type": "...", "description": "..."}
  ],
  "relationships": [
    {"source": "...", "target": "...", "description": "..."}
  ]
}

Rules:
- Entity names must be consistent (same real-world thing = same exact name \
string) so they merge correctly across chunks.
- Only include relationships between entities you also listed.
- If nothing meaningful is found, return {"entities": [], "relationships": []}.
"""


def extract_entities_and_relationships(chunk: str) -> dict:
    raw = chat(EXTRACTION_SYSTEM_PROMPT, chunk)
    return _safe_parse_json(raw)

def _safe_parse_json(raw: str) -> dict:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"entities": [], "relationships": []}
    data.setdefault("entities", [])
    data.setdefault("relationships", [])
    return data