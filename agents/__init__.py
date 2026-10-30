"""EducationGPT Agents Package."""

from agents.ema_agent import (
    SchoolAnalyticsReport,
    audit_district_shortages,
    download_latest_ema_report,
    get_ema_metrics_summary,
    predict_school_shortage_risk,
    query_dashboard_data,
    query_education_gpt,
    root_agent,
)

__all__ = [
    "root_agent",
    "query_education_gpt",
    "SchoolAnalyticsReport",
    "download_latest_ema_report",
    "get_ema_metrics_summary",
    "query_dashboard_data",
    "audit_district_shortages",
    "predict_school_shortage_risk",
]
