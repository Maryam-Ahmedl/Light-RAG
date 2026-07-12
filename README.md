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
## Latency Tracking

Every query and every document ingestion is automatically timed at each pipeline stage and logged to CSV for trend analysis over time — no setup needed, this happens automatically whenever you use `cli.py`.

### What gets measured

| Stage | What it captures |
|---|---|
| `embedding_seconds` | Time to embed the question into a vector |
| `retrieval_seconds` | Time spent on vector search + graph traversal (local, no API call) |
| `ttft_seconds` | Time To First Token — how long until the LLM starts streaming a response |
| `llm_seconds` | Total time for the LLM to finish generating the full answer |
| `end_to_end_seconds` | Total wall-clock time for the whole query, embedding through final answer |

For ingestion, per-chunk extraction and embedding times are tracked, plus the batched entity-embedding step at the end of each document.

### Viewing latency reports

Latency is logged to `logs/query_latency.csv` and `logs/ingest_latency.csv`. From the interactive CLI prompt:
report            # query latency trends: avg/min/max per stage, plus
# whether each stage is trending up or down over time
ingest_report     # per-document ingestion timing breakdown

Example `report` output:
#### Query Latency Report *Based on 14 queries*

| Stage         | Avg (s) | Min (s) | Max (s) | Trend |
|---------------|:-------:|:-------:|:-------:|:-----:|
| Embedding     | 0.183   | 0.120   | 0.310   | ↓ 12.4% |
| Retrieval     | 0.021   | 0.010   | 0.045   | → 2.1%  |
| TTFT          | 0.402   | 0.290   | 0.610   | ↑ 18.7% |
| LLM (total)   | 1.612   | 1.100   | 2.400   | ↑ 15.3% |
| **End-to-End**| **1.816** | **1.230** | **2.700** | **↑ 16.1%** |

> **Trend legend:** ↓ improving · → stable · ↑ increasing (relative to prior run)

The trend column compares the first half vs second half of logged history, so you can catch regressions early — e.g. TTFT creeping up as the knowledge graph grows and prompts get larger.

### Why this matters

- **Retrieval** should almost always be tiny (pure local compute — numpy/networkx, no network). A spike here signals your vector store or graph is outgrowing brute-force search.
- **Embedding** and **LLM** stages are network calls — these are the ones actually affected by API/provider performance.
- **Ingestion** timing (via `ingest_report`) reveals cost and scaling risk: since one `chat()` call happens per chunk during extraction, a slow average extraction time multiplied across a large document tells you how long a new document takes to become queryable, and whether you're at risk of hitting rate limits on bigger ingests.


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
