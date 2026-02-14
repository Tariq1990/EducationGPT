import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    score: float
    text: str


class LocalRAGAdvisor:
    def __init__(self, documents: List[Dict[str, str]]):
        self.documents = documents
        self.doc_by_id = {d["id"]: d for d in self.documents}
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )
        corpus = [d["text"] for d in self.documents]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.08) -> List[RetrievedChunk]:
        q = (query or "").strip()
        if not q:
            return []
        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        ranked_idx = sims.argsort()[::-1]

        results: List[RetrievedChunk] = []
        for idx in ranked_idx[:top_k]:
            score = float(sims[idx])
            if score < min_score:
                continue
            doc = self.documents[int(idx)]
            results.append(
                RetrievedChunk(
                    chunk_id=doc["id"],
                    source=doc["source"],
                    score=score,
                    text=doc["text"]
                )
            )
        return results

    def get_chunk_by_id(self, chunk_id: str) -> RetrievedChunk | None:
        doc = self.doc_by_id.get(chunk_id)
        if not doc:
            return None
        return RetrievedChunk(
            chunk_id=doc["id"],
            source=doc["source"],
            score=1.0,
            text=doc["text"]
        )


def _safe_num(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def build_local_documents(df: pd.DataFrame) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    if df is None or len(df) == 0:
        return docs

    total = len(df)
    critical = int(df["is_critical"].sum()) if "is_critical" in df.columns else 0
    avg_teachers = _safe_num(df["total_teachers"].mean()) if "total_teachers" in df.columns else 0.0
    docs.append(
        {
            "id": "dataset-overview",
            "source": "data/processed/processed_school_data.csv",
            "text": (
                f"Dataset overview. Total schools: {total}. "
                f"Critical schools: {critical}. "
                f"Average teachers per school: {avg_teachers:.2f}."
            )
        }
    )

    if "District" in df.columns:
        grouped = df.groupby("District", dropna=False)
        for district, g in grouped:
            district_name = str(district)
            total_d = len(g)
            critical_d = int(g["is_critical"].sum()) if "is_critical" in g.columns else 0
            avg_teacher_d = _safe_num(g["total_teachers"].mean()) if "total_teachers" in g.columns else 0.0
            docs.append(
                {
                    "id": f"district-{district_name}",
                    "source": "data/processed/processed_school_data.csv",
                    "text": (
                        f"District summary for {district_name}. "
                        f"Schools: {total_d}. "
                        f"Critical: {critical_d}. "
                        f"Average teachers: {avg_teacher_d:.2f}."
                    )
                }
            )

    sample_cols = [
        "District", "Gender", "LEVEL", "total_teachers",
        "No of Pakka Class Rooms", "Presentbalance", "is_critical"
    ]
    available_cols = [c for c in sample_cols if c in df.columns]
    # Cap row-level profiles for latency and prompt size control.
    max_profiles = min(len(df), 300)
    for i, row in df.head(max_profiles).iterrows():
        parts = [f"{c}: {row[c]}" for c in available_cols]
        docs.append(
            {
                "id": f"profile-{int(i)}",
                "source": "data/processed/processed_school_data.csv",
                "text": "School profile. " + ". ".join(parts) + "."
            }
        )

    for path in ["README.md", "PROJECT_REPORT.md"]:
        if os.path.exists(path):
            try:
                text = open(path, "r", encoding="utf-8").read().strip()
                if text:
                    # Chunk long docs to improve retrieval precision.
                    chunk_size = 900
                    overlap = 120
                    start = 0
                    chunk_no = 0
                    while start < len(text):
                        end = min(start + chunk_size, len(text))
                        chunk = text[start:end].strip()
                        if chunk:
                            docs.append(
                                {
                                    "id": f"doc-{path.lower().replace('.', '-')}-{chunk_no}",
                                    "source": path,
                                    "text": chunk
                                }
                            )
                            chunk_no += 1
                        if end >= len(text):
                            break
                        start = max(end - overlap, start + 1)
            except Exception:
                # Ignore non-critical doc-loading issues.
                pass

    return docs


def save_documents(documents: List[Dict[str, str]], out_path: str) -> str:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    return out_path


def load_documents(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    cleaned: List[Dict[str, str]] = []
    for d in data:
        if not isinstance(d, dict):
            continue
        chunk_id = str(d.get("id", "")).strip()
        source = str(d.get("source", "")).strip()
        text = str(d.get("text", "")).strip()
        if chunk_id and source and text:
            cleaned.append({"id": chunk_id, "source": source, "text": text})
    return cleaned
