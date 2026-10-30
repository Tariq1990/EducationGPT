"""Education Monitoring Authority (KPEMA / IMU) Agent Package for EducationGPT."""

from agents.ema_agent.agent import (
    SchoolAnalyticsReport,
    query_education_gpt,
    root_agent,
)
from agents.ema_agent.tools import (
    audit_district_shortages,
    download_latest_ema_report,
    get_ema_metrics_summary,
    predict_school_shortage_risk,
    query_dashboard_data,
    sanitize_for_json,
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
    "sanitize_for_json",
]
