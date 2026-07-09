# Light-RAG

A custom, from-scratch Retrieval-Augmented Generation (RAG) pipeline, inspired by [LightRAG](https://github.com/HKUDS/LightRAG). It combines vector search with a knowledge graph for retrieval, supports four distinct query modes, and runs entirely on local storage (no external vector DB required).

Originally built against OpenAI, then migrated to run on **Google Gemini** via its OpenAI-compatible endpoint — no application logic had to change, only the LLM/embedding client configuration.

## Features

- **Four retrieval modes**: `naive` (plain vector RAG), `local` (entity-focused graph lookup), `global` (multi-hop graph traversal), and `hybrid` (combines chunk + graph context)
- **Knowledge graph construction**: entities and relationships are extracted from each chunk via LLM and merged incrementally across documents (`networkx`-backed)
- **Incremental ingestion**: documents are hashed and tracked, so re-running ingestion skips unchanged files
- **Latency instrumentation**: per-stage timing (embedding, retrieval, LLM, TTFT) logged to CSV for every query and every ingestion run, with a built-in trend report
- **RAGAS evaluation**: automated scoring of faithfulness, answer relevancy, context precision, and context recall — using real questions pulled from logged usage

## Architecture

```
Light-RAG/
├── cli.py                        Interactive CLI: ingest a file/folder, then ask questions
├── LLM.py                        Thin wrapper around the OpenAI-compatible client (chat, embed, latency helpers)
├── Chunking.py                   Word-based text chunker with overlap
├── Extraction.py                 LLM-based entity/relationship extraction per chunk
├── Ingest.py                     End-to-end ingestion: chunk → extract → embed → store
├── Query.py                      Query engine implementing the four retrieval modes
├── latency_tracker.py            Standalone in-memory latency tracker utility (context-manager based)
├── ragas_eval.py                 Runs RAGAS metrics against the eval dataset
│
├── storage/
│   ├── graph_store.py            NetworkX-backed knowledge graph (entities + relationships)
│   ├── Kv_store.py               Minimal JSON key-value store (chunks, doc status)
│   ├── vector_store.py           Minimal local vector store (numpy, brute-force cosine search)
│   └── latency_log.py            CSV-based latency logging and trend reports (used by cli.py/Query.py)
│
└──  generate_eval_Questions.py  Builds a RAGAS eval set from real logged user questions
```


## Setup

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd Light-RAG
python -m venv venv
```

Activate it:

```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/), then edit `.env`:

```env
OPENAI_API_KEY=your-actual-gemini-api-key
```

### 4. Add source documents

Place `.txt` files in a `data/` folder:


## Usage

### Ingest and query interactively

```bash
python cli.py data/sample.txt      # ingest one file
python cli.py data/                 # ingest an entire folder
```

In the interactive prompt:
what is SOAP?                    # asks using default 'hybrid' mode
mode:local what is SOAP?         # asks using a specific mode (naive/local/global/hybrid)
report                           # shows query latency trends over time
ingest_report                    # shows ingestion latency per document
quit                             # exits

### Run programmatically

```python
from Query import query

result = query("What is SOAP?", mode="hybrid")
print(result["answer"])
print(result["timings"])
print(result["contexts"])   # list of raw retrieved context strings, useful for eval pipelines
```

## Evaluation (RAGAS)

Evaluation uses **real questions you've actually asked** (pulled from `logs/query_latency.csv`), not synthetic ones.

### 1. Ask some questions first
Run the CLI and ask a handful of real questions (see Usage above).

### 2. Build the evaluation dataset
```bash
python evaluation/generate_eval_dataset.py
```
Saves to `eval_dataset.json`. **Review/edit the `ground_truth` fields** before scoring if accuracy matters — auto-generated references inherit any gaps in retrieval.

### 3. Score it
```bash
python ragas_eval.py
```
Prints aggregate scores and saves a per-question breakdown to `ragas_results.csv`:

| Metric | What it measures |
|---|---|
| `faithfulness` | Does the answer only use facts present in the retrieved context? |
| `answer_relevancy` | Does the answer actually address the question asked? |
| `context_precision` | Of the retrieved chunks, how many were relevant? |
| `context_recall` | Did retrieval find everything needed to answer correctly? |
