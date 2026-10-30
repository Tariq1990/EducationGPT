"""EMA Dashboard Automation Agent for EducationGPT."""

from .dashboard_scraper import EMADashboardScraper
from .report_processor import EMAReportProcessor
from .ema_tool import (
    download_latest_ema_report,
    get_ema_metrics_summary,
    query_dashboard_data,
    sanitize_for_json,
)
from .agent import root_agent, SchoolAnalyticsReport

__all__ = [
    "EMADashboardScraper",
    "EMAReportProcessor",
    "download_latest_ema_report",
    "get_ema_metrics_summary",
    "query_dashboard_data",
    "sanitize_for_json",
    "root_agent",
    "SchoolAnalyticsReport",
]
