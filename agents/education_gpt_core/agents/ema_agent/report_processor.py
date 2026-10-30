"""Data processing pipeline for EMA dashboard reports.

Reads CSV / XLSX files downloaded by the scraper, cleans them,
and produces summary metrics suitable for LLM context windows.
"""

import math
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# Common column aliases we attempt to normalise
_COL_ALIASES = {
    "school_name": ["school", "school name", "schoolname", "name"],
    "district": ["district", "dist", "district_name"],
    "enrollment": ["total_enrollment", "enrollment", "enrol", "students"],
    "attendance": ["attendance", "attend", "present"],
    "teachers": ["teachers", "total_teachers", "num_teachers", "teacher_count"],
    "performance_score": ["score", "performance", "perf_score", "avg_score"],
    "facility_index": ["facility", "facility_index", "infra", "infrastructure"],
}


def sanitize_for_json(val: Any) -> Any:
    """Recursively converts all NaN, NaT, Inf, and non-serializable objects to None or valid JSON types."""
    if val is None:
        return None
    if val is pd.NA or val is pd.NaT:
        return None
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (datetime, date, pd.Timestamp)):
        return val.isoformat()
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, dict):
        return {str(k): sanitize_for_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [sanitize_for_json(v) for v in val]
    if isinstance(val, pd.Series):
        return sanitize_for_json(val.to_dict())
    if isinstance(val, pd.DataFrame):
        return sanitize_for_json(val.to_dict(orient="records"))
    if isinstance(val, np.ndarray):
        return sanitize_for_json(val.tolist())
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    return val


class EMAReportProcessor:
    """Load, clean, and summarise EMA report files."""

    def __init__(self, download_dir: Optional[str] = None):
        raw_dir = download_dir or os.getenv("EMA_DOWNLOAD_DIR", "./data/raw_reports")
        p = Path(raw_dir)
        if not p.is_absolute():
            # Check relative to current working dir first, then PROJECT_ROOT
            if p.exists():
                self.download_dir = p.resolve()
            elif (PROJECT_ROOT / raw_dir).exists():
                self.download_dir = (PROJECT_ROOT / raw_dir).resolve()
            else:
                self.download_dir = (PROJECT_ROOT / raw_dir).resolve()
        else:
            self.download_dir = p.resolve()

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------
    def discover_files(self) -> List[Path]:
        """Return sorted list of processable report files."""
        if not self.download_dir.exists():
            return []
        files = [
            p
            for p in sorted(self.download_dir.iterdir())
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return files

    def latest_file(self) -> Optional[Path]:
        """Return the most recently modified supported file."""
        files = self.discover_files()
        if not files:
            return None
        return max(files, key=lambda p: p.stat().st_mtime)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    @staticmethod
    def read_file(path: Path) -> pd.DataFrame:
        """Read a CSV or Excel file into a DataFrame."""
        ext = path.suffix.lower()
        if ext == ".csv":
            # Try common encodings
            for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    return pd.read_csv(path, encoding=enc)
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            return pd.read_csv(path, encoding="utf-8", errors="replace")
        elif ext in (".xlsx", ".xls"):
            try:
                return pd.read_excel(path, engine="openpyxl")
            except Exception:
                return pd.read_excel(path)
        raise ValueError(f"Unsupported file extension: {ext}")

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------
    @classmethod
    def clean(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names, drop empty rows, coerce numerics."""
        if df is None or df.empty:
            return pd.DataFrame()

        # Lowercase & strip column names
        df.columns = [
            re.sub(r"[^\w]+", "_", c.strip().lower()).strip("_")
            for c in df.columns
        ]

        # Drop rows that are entirely NaN
        df = df.dropna(how="all").reset_index(drop=True)

        # Attempt to coerce numeric-looking columns
        for col in df.columns:
            if df[col].dtype == object:
                converted = pd.to_numeric(df[col], errors="coerce")
                # If > 50 % converted successfully, keep as numeric
                if converted.notna().mean() > 0.50:
                    df[col] = converted

        return df

    # ------------------------------------------------------------------
    # Normalise known columns
    # ------------------------------------------------------------------
    @classmethod
    def normalise_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Rename known alias columns to canonical names."""
        rename_map: Dict[str, str] = {}
        existing = set(df.columns)
        for canonical, aliases in _COL_ALIASES.items():
            if canonical in existing:
                continue
            for alias in aliases:
                if alias in existing:
                    rename_map[alias] = canonical
                    break
        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------
    @classmethod
    def compute_metrics(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Derive key education metrics from the cleaned DataFrame."""
        metrics: Dict[str, Any] = {
            "total_records": len(df),
            "columns": list(df.columns),
        }

        # Per-column statistics for numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        col_stats: Dict[str, Dict[str, Optional[float]]] = {}
        for col in num_cols:
            s = df[col].dropna()
            count_val = int(s.count())
            mean_val = round(float(s.mean()), 2) if count_val > 0 and not math.isnan(s.mean()) else 0.0
            median_val = round(float(s.median()), 2) if count_val > 0 and not math.isnan(s.median()) else 0.0
            min_val = round(float(s.min()), 2) if count_val > 0 and not math.isnan(s.min()) else 0.0
            max_val = round(float(s.max()), 2) if count_val > 0 and not math.isnan(s.max()) else 0.0
            std_val = round(float(s.std()), 2) if count_val > 1 and not math.isnan(s.std()) else 0.0

            col_stats[col] = {
                "count": count_val,
                "mean": mean_val,
                "median": median_val,
                "min": min_val,
                "max": max_val,
                "std": std_val,
            }
        metrics["numeric_summary"] = col_stats

        # --- Domain-specific aggregates ---
        if "enrollment" in df.columns:
            s_enrol = df["enrollment"].dropna()
            if len(s_enrol) > 0:
                metrics["total_enrollment"] = int(s_enrol.sum())
                metrics["avg_enrollment"] = round(float(s_enrol.mean()), 1)

        if "attendance" in df.columns:
            s_att = df["attendance"].dropna()
            if len(s_att) > 0:
                metrics["avg_attendance"] = round(float(s_att.mean()), 2)

        if "teachers" in df.columns:
            s_teach = df["teachers"].dropna()
            if len(s_teach) > 0:
                metrics["total_teachers"] = int(s_teach.sum())
                metrics["avg_teachers_per_school"] = round(float(s_teach.mean()), 1)

        if "performance_score" in df.columns:
            s_perf = df["performance_score"].dropna()
            if len(s_perf) > 0:
                metrics["avg_performance_score"] = round(float(s_perf.mean()), 2)

        if "facility_index" in df.columns:
            s_fac = df["facility_index"].dropna()
            if len(s_fac) > 0:
                metrics["avg_facility_index"] = round(float(s_fac.mean()), 2)

        # District breakdown (top-10)
        if "district" in df.columns:
            dist_counts = df["district"].value_counts().head(10).to_dict()
            metrics["top_districts"] = {
                str(k): int(v) for k, v in dist_counts.items()
            }

        return sanitize_for_json(metrics)

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------
    def process_latest(self) -> Dict[str, Any]:
        """Process the most recent file and return a result dict."""
        path = self.latest_file()
        if path is None:
            return {"status": "error", "message": "No report files found."}
        return self.process_file(path)

    def process_file(self, path: Path) -> Dict[str, Any]:
        """Full pipeline for a single file."""
        try:
            raw_df = self.read_file(path)
            clean_df = self.clean(raw_df)
            clean_df = self.normalise_columns(clean_df)
            metrics = self.compute_metrics(clean_df)
            # Clean dataframe records with None instead of NaN
            head_records = sanitize_for_json(clean_df.head(10).to_dict(orient="records"))
            return {
                "status": "ok",
                "file": str(path),
                "shape": list(clean_df.shape),
                "metrics": metrics,
                "head": head_records,
                "dataframe": clean_df,
            }
        except Exception as exc:
            return {
                "status": "error",
                "file": str(path),
                "message": str(exc),
            }

    def process_all(self) -> List[Dict[str, Any]]:
        """Process every discovered file."""
        return [self.process_file(p) for p in self.discover_files()]

    # ------------------------------------------------------------------
    # LLM-ready context string
    # ------------------------------------------------------------------
    @staticmethod
    def metrics_to_context(metrics: Dict[str, Any], max_chars: int = 3000) -> str:
        """Format metrics dict into a compact text block for LLM prompts."""
        lines = ["=== EMA Report Metrics ==="]
        lines.append(f"Total records: {metrics.get('total_records', 'N/A')}")

        for key in (
            "total_enrollment", "avg_enrollment", "avg_attendance",
            "total_teachers", "avg_teachers_per_school",
            "avg_performance_score", "avg_facility_index",
        ):
            if key in metrics and metrics[key] is not None:
                label = key.replace("_", " ").title()
                lines.append(f"{label}: {metrics[key]}")

        if "top_districts" in metrics and metrics["top_districts"]:
            lines.append("Top Districts:")
            for dist, cnt in metrics["top_districts"].items():
                lines.append(f"  - {dist}: {cnt} schools")

        if "numeric_summary" in metrics and metrics["numeric_summary"]:
            lines.append("\nNumeric Column Summaries:")
            for col, stats in list(metrics["numeric_summary"].items())[:10]:
                lines.append(
                    f"  {col}: mean={stats['mean']}, "
                    f"median={stats['median']}, "
                    f"min={stats['min']}, max={stats['max']}"
                )

        text = "\n".join(lines)
        return text[:max_chars]


if __name__ == "__main__":
    import json
    processor = EMAReportProcessor()
    result = processor.process_latest()
    if result["status"] == "ok":
        print(f"Processed: {result['file']}  shape={result['shape']}")
        dumped = json.dumps(result["metrics"], allow_nan=False)
        print("Metrics successfully serialized without NaN! Length:", len(dumped))
    else:
        print(f"Error: {result.get('message', 'unknown')}")
