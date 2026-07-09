import sys
from pathlib import Path
from Ingest import ingest_file
from Query import query
from storage.latency_log import query_report, ingest_report


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "data/sample.txt"
    path = Path(target)

    if path.is_dir():
        files = sorted(path.glob("*.txt"))
        print(f"Found {len(files)} files in {path}\n")
        for f in files:
            print(f"Ingesting {f} ...")
            result = ingest_file(str(f))
            print(result, "\n")
    else:
        print(f"Ingesting {path} ...")
        result = ingest_file(str(path))
        print(result, "\n")

    print("Type a question (or 'quit'). Prefix with 'mode:local ' / 'mode:global ' / 'mode:naive ' to change mode.")
    print("Type 'report' to see query latency trends. Type 'ingest_report' for ingestion history.\n")

    while True:
        raw = input("> ").strip()
        if raw.lower() in ("quit", "exit"):
            break
        if not raw:
            continue
        if raw.lower() == "report":
            print("\n" + query_report() + "\n")
            continue
        if raw.lower() == "ingest_report":
            print("\n" + ingest_report() + "\n")
            continue

        mode = "hybrid"
        question = raw
        if raw.startswith("mode:"):
            mode_token, _, question = raw.partition(" ")
            mode = mode_token.split(":", 1)[1]

        result = query(question, mode=mode)
        t = result["timings"]
        print(f"\n[{result['mode']}] {result['answer']}")
        print(
            f"(embed: {t['embedding_seconds']:.3f}s | retrieval: {t['retrieval_seconds']:.3f}s | "
            f"ttft: {t['ttft_seconds']:.3f}s | llm: {t['llm_seconds']:.3f}s | "
            f"total: {t['end_to_end_seconds']:.3f}s)\n"
        )

if __name__ == "__main__":
    main()