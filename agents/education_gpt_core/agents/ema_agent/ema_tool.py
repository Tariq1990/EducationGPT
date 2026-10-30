"""EducationGPT tool wrappers for the EMA Dashboard Agent.

Exposes three core tools for the EducationGPT ADK agent:
  - download_latest_ema_report: Triggers headless browser scraping of latest dashboard reports
  - get_ema_metrics_summary: Aggregates metrics, KPI summaries, and district breakdowns
  - query_dashboard_data: Filtered school-level queries by district and text search
"""

import asyncio
import concurrent.futures
import math
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from agents.ema_agent.dashboard_scraper import EMADashboardScraper
    from agents.ema_agent.report_processor import EMAReportProcessor
except ImportError:
    from .dashboard_scraper import EMADashboardScraper
    from .report_processor import EMAReportProcessor


# ---------------------------------------------------------------------------
# Strict JSON Sanitizer (Converts NaN, NaT, Inf to None)
# ---------------------------------------------------------------------------
def sanitize_for_json(val: Any) -> Any:
    """Recursively converts all NaN, NaT, Inf, and non-serializable objects to None or valid JSON types.

    Guarantees that the resulting data structure can be serialized by json.dumps(..., allow_nan=False)
    and accepted by Gemini / Google GenAI without payload validation errors (INVALID_ARGUMENT).
    """
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


# ---------------------------------------------------------------------------
# Asynchronous execution handler
# ---------------------------------------------------------------------------
def _run_coroutine_sync(coro):
    """Safely execute an async coroutine from synchronous tool wrappers.

    If an event loop is already active (such as within ADK async runtime,
    FastAPI, or Uvicorn), running asyncio.run() directly raises:
    'RuntimeError: asyncio.run() cannot be called from a running event loop'.
    This wrapper delegates the coroutine to a dedicated background thread with
    its own clean event loop, ensuring safe execution in all contexts.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Singleton-style instances (lazy)
# ---------------------------------------------------------------------------
_processor: Optional[EMAReportProcessor] = None


def _get_processor() -> EMAReportProcessor:
    global _processor
    if _processor is None:
        _processor = EMAReportProcessor()
    return _processor


# ---------------------------------------------------------------------------
# Tool 1: download_latest_ema_report
# ---------------------------------------------------------------------------
def download_latest_ema_report() -> Dict[str, Any]:
    """Log into the EMA dashboard and download the latest available reports.

    Use this tool when the user explicitly requests downloading fresh monitoring reports,
    scraping new monthly data from the EMA/IMU web dashboard, or when local datasets are
    missing or require updating.

    Returns:
        dict: Status ('ok' or 'error'), status message, count of saved files, and file paths.
    """
    try:
        scraper = EMADashboardScraper()
        saved_files: List[Path] = _run_coroutine_sync(scraper.run())

        # Reset processor so next read sees the newly downloaded file
        global _processor
        _processor = None

        if not saved_files:
            return sanitize_for_json({
                "status": "warning",
                "message": "Scraper executed but no new report files were exported from the dashboard.",
                "files": [],
                "count": 0,
            })

        return sanitize_for_json({
            "status": "ok",
            "message": f"Successfully downloaded {len(saved_files)} file(s) from EMA dashboard.",
            "files": [str(f) for f in saved_files],
            "count": len(saved_files),
        })
    except Exception as exc:
        return sanitize_for_json({
            "status": "error",
            "message": f"EMA dashboard scraping failed: {str(exc)}",
            "files": [],
            "count": 0,
        })


# ---------------------------------------------------------------------------
# Tool 2: get_ema_metrics_summary
# ---------------------------------------------------------------------------
def get_ema_metrics_summary(force_refresh: bool = False) -> Dict[str, Any]:
    """Process the most recently downloaded EMA report and return aggregated education metrics.

    Use this tool to retrieve district-wide metrics, enrollment totals, student-to-teacher ratios,
    attendance averages, basic facility index distributions, and top district breakdowns.
    If no local report exists, or if force_refresh is True, it triggers an automatic scrape first.

    Args:
        force_refresh (bool): If True, triggers a fresh download from the EMA dashboard before processing.

    Returns:
        dict: Processing status, report filename, data shape, clean numeric metrics, and context text.
    """
    if force_refresh:
        download_latest_ema_report()

    processor = _get_processor()
    result = processor.process_latest()

    # If no report files exist locally, attempt automatic download
    if result["status"] == "error" and "No report" in result.get("message", ""):
        dl = download_latest_ema_report()
        if dl.get("status") not in ("ok", "warning") or dl.get("count", 0) == 0:
            return sanitize_for_json({
                "status": "error",
                "message": f"No local reports found and automatic scrape failed: {dl.get('message', 'No files downloaded')}",
            })
        processor = _get_processor()
        result = processor.process_latest()

    if result["status"] != "ok":
        return sanitize_for_json(result)

    context_text = EMAReportProcessor.metrics_to_context(result["metrics"])
    response = {
        "status": "ok",
        "file": result["file"],
        "shape": result["shape"],
        "metrics": result["metrics"],
        "context": context_text,
    }
    return sanitize_for_json(response)


# ---------------------------------------------------------------------------
# Tool 3: query_dashboard_data
# ---------------------------------------------------------------------------
def query_dashboard_data(
    query: str = "",
    district: Optional[str] = None,
    top_n: int = 20,
    columns: Optional[List[str]] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Query processed EMA school records with district filtering and keyword search.

    Use this tool to inspect detailed school-level records, filter schools by district name
    (e.g., Mohmand, Peshawar, Swat), check specific infrastructure parameters (toilets, electricity,
    water, boundary walls, IT labs), or search for individual schools.
    All NaN, NaT, and non-compliant numeric values are sanitized to None for safe JSON transport.

    Args:
        query (str): Free-text search term matched across string columns (e.g. school name, facility type).
        district (str, optional): Name of the educational district to filter (case-insensitive partial match).
        top_n (int): Maximum number of records to return (default 20, max 100).
        columns (list of str, optional): Optional subset of column names to return for compact payloads.
        force_refresh (bool): If True, forces a fresh dashboard scrape before querying.

    Returns:
        dict: Status ('ok' or 'error'), total matched record count, returned count, and list of row dicts.
    """
    if force_refresh:
        download_latest_ema_report()

    processor = _get_processor()
    result = processor.process_latest()

    # If no report exists, attempt automatic download
    if result["status"] == "error" and "No report" in result.get("message", ""):
        dl = download_latest_ema_report()
        if dl.get("status") == "ok" and dl.get("count", 0) > 0:
            processor = _get_processor()
            result = processor.process_latest()

    if result["status"] != "ok":
        return sanitize_for_json(result)

    df = result["dataframe"].copy()

    # District filter
    if district and "district" in df.columns:
        clean_dist = district.strip()
        if clean_dist.lower().endswith(" district"):
            clean_dist = clean_dist[:-9].strip()
        mask = df["district"].astype(str).str.contains(clean_dist, case=False, na=False)
        df = df[mask]

    # Free-text search across string columns
    if query:
        str_cols = df.select_dtypes(include=["object"]).columns
        if len(str_cols) > 0:
            pattern = query.strip().lower()
            mask = df[str_cols].apply(
                lambda col: col.astype(str).str.lower().str.contains(pattern, na=False)
            ).any(axis=1)
            df = df[mask]

    total = len(df)
    limit = min(max(top_n, 1), 100)
    subset = df.head(limit).copy()

    # Optional column selection
    if columns:
        valid_cols = [c for c in columns if c in subset.columns]
        if valid_cols:
            subset = subset[valid_cols]

    # Convert records to dictionary and sanitize every value
    raw_records = subset.to_dict(orient="records")
    sanitized_records = sanitize_for_json(raw_records)

    response = {
        "status": "ok",
        "total_matched": total,
        "returned": len(sanitized_records),
        "records": sanitized_records,
    }
    return sanitize_for_json(response)


if __name__ == "__main__":
    import json
    print("=== EMA Tool Self-Test ===")
    summary = get_ema_metrics_summary()
    print("Status:", summary.get("status"))
    dumped_summary = json.dumps(summary, allow_nan=False)
    print("Summary JSON length (allow_nan=False):", len(dumped_summary))

    query_res = query_dashboard_data(district="Mohmand", top_n=3)
    print("Query status:", query_res.get("status"))
    dumped_query = json.dumps(query_res, allow_nan=False)
    print("Query JSON length (allow_nan=False):", len(dumped_query))
