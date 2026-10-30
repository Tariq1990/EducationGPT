"""Production tool interfaces for EducationGPT agents.

Provides clean, strictly-sanitized tool functions for:
  - Scraping & downloading fresh EMA reports
  - Aggregating district KPIs and attendance metrics
  - Querying school-level census records
  - Auditing district bottlenecks and facility deficits
  - Predicting critical resource shortage risk via trained ML models
"""

import asyncio
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.ema_agent.dashboard_scraper import (
    EMADashboardScraper,
    HTTPDashboardClient,
    get_latest_cached_dashboard_html,
)
from agents.ema_agent.metrics_engine import EMAMetricsEngine, sanitize_for_json

# Global engine singleton
_ENGINE = EMAMetricsEngine()


def _run_async(coro):
    """Executes an async coroutine safely across both sync and async runtimes."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def download_latest_ema_report() -> Dict[str, Any]:
    """Triggers retrieval and downloading of latest reports from the live EMA web dashboard.

    Use this tool whenever the user asks to "fetch fresh data", "scrape the dashboard",
    "download the latest report", or when local datasets are missing or need updating.
    """
    client = HTTPDashboardClient()
    html = client.fetch_dashboard_page()
    if html:
        return sanitize_for_json({
            "status": "ok",
            "message": "Successfully retrieved and cached latest EMA dashboard snapshot via HTTP client.",
            "mode": "http_session",
        })

    # Fallback to Playwright scraper if available
    try:
        scraper = EMADashboardScraper()
        files = _run_async(scraper.run())
        return sanitize_for_json({
            "status": "ok" if files else "warning",
            "downloaded_files": [str(f) for f in files],
            "message": f"Retrieved {len(files)} files via headless browser.",
            "mode": "playwright",
        })
    except Exception as e:
        return sanitize_for_json({
            "status": "error",
            "message": f"Scraper execution encountered error: {e}. Falling back to cached reports.",
        })


def get_ema_metrics_summary(
    district: str = "mohmand",
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Loads and computes high-level education metrics and KPI summaries.

    Includes total enrollment, school counts, teacher counts, student-to-teacher ratios (STR),
    attendance averages, and basic facility index distributions (water, electricity, toilets, boundary wall).
    Use this tool whenever the user asks for district summaries, overall KPIs, aggregate enrollment figures,
    or general monitoring trends.
    """
    return _ENGINE.get_dashboard_summary(district=district, force_refresh=force_refresh)


def query_dashboard_data(
    query: str = "",
    district: Optional[str] = "mohmand",
    level: Optional[str] = None,
    top_n: int = 20,
    columns: Optional[List[str]] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Queries individual school records with district filtering and free-text search.

    Use this tool whenever the user asks about specific districts, individual schools,
    facility deficits (water, electricity, toilets, boundary walls, IT labs), or teacher vacancies.
    """
    if force_refresh:
        download_latest_ema_report()
    return _ENGINE.query_schools(
        query=query,
        district=district,
        level=level,
        top_n=top_n,
        columns=columns,
    )


def audit_district_shortages(district: str = "mohmand") -> Dict[str, Any]:
    """Performs an in-depth audit of district infrastructure, bottlenecks, and policy compliance.

    Identifies single-teacher schools, closed schools, unauthorized teacher absence alerts,
    and missing basic facilities. Provides actionable recommendations.
    """
    return _ENGINE.audit_district_vulnerabilities(district=district)


def predict_school_shortage_risk(school_data: Dict[str, Any]) -> Dict[str, Any]:
    """Runs the trained Random Forest model to predict if a school faces critical resource shortages.

    Evaluates classrooms, teachers, enrollment, and facilities to assign a risk score.
    """
    return _ENGINE.predict_shortage_risk(school_data)
