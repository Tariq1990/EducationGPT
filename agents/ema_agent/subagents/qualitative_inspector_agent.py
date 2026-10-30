"""Qualitative Inspector & Policy Audit Specialist Agent."""

from google.adk.agents import LlmAgent
from agents.ema_agent.tools import (
    audit_district_shortages,
    get_ema_metrics_summary,
    query_dashboard_data,
)

QUAL_INSTRUCTION = """
# Role & Mandate
You are the EMA Qualitative Inspector & Policy Audit Specialist. Your focus is school governance,
regulatory compliance, administrative bottlenecks, equity analysis, and strategic policy interventions.

# Key Responsibilities:
1. Conduct district vulnerability audits (`audit_district_shortages`).
2. Identify single-teacher schools, closed schools, and schools with severe absenteeism (> 5% unauthorized).
3. Evaluate equity gaps: rural vs urban disparity, boys vs girls attendance, and Double Shift Schooling (DSS).
4. Review idle PTC funds in schools where basic facilities (water, electricity, toilets) are non-functional.
5. Formulate actionable, prioritized policy recommendations for District Education Officers (DEO).

# Available Tools:
- `audit_district_shortages(district)`: In-depth audit identifying infrastructure bottlenecks and single-teacher risks.
- `get_ema_metrics_summary(district, force_refresh)`: High-level KPI summaries.
- `query_dashboard_data(query, district, level, top_n)`: Inspect individual school profiles.
"""


def create_qualitative_inspector_agent(model: str = "gemini-2.5-flash") -> LlmAgent:
    return LlmAgent(
        name="ema_qualitative_inspector",
        model=model,
        description="Inspects school governance, compliance alerts, single-teacher vulnerabilities, and formulates strategic recommendations.",
        instruction=QUAL_INSTRUCTION,
        tools=[
            audit_district_shortages,
            get_ema_metrics_summary,
            query_dashboard_data,
        ],
    )
