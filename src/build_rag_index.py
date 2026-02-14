import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.local_rag import build_local_documents, save_documents


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(base_dir, ".."))
    csv_path = os.getenv(
        "PROCESSED_CSV_PATH",
        os.path.normpath(os.path.join(project_root, "data", "processed", "processed_school_data.csv"))
    )
    out_path = os.getenv(
        "RAG_INDEX_PATH",
        os.path.normpath(os.path.join(project_root, "data", "processed", "rag_chunks.json"))
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Processed dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    docs = build_local_documents(df)
    if not docs:
        raise RuntimeError("No RAG documents were generated from available data.")

    saved = save_documents(docs, out_path)
    print(f"RAG index saved: {saved}")
    print(f"Total chunks: {len(docs)}")


if __name__ == "__main__":
    main()
