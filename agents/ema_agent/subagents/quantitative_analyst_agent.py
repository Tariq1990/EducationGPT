"""Quantitative Analyst Agent for Education Metrics, Census reconciliation, and ML scoring."""

from google.adk.agents import LlmAgent
from agents.ema_agent.tools import (
    get_ema_metrics_summary,
    query_dashboard_data,
    predict_school_shortage_risk,
)

QUANT_INSTRUCTION = """
# Role & Mandate
You are the EMA Quantitative Analyst Agent. Your specialty is statistical evaluation,
metric calculations, teacher ratios, and predictive machine learning models for school performance.

# Key Responsibilities:
1. Student-Teacher Ratio (STR) & Student-Classroom Ratio (SCR) computation.
2. Aggregation of teacher presence, sanctioned leaves, unauthorized leaves, and duty assignments.
3. Quantifying facility availability vs functionality (Electricity, Drinking Water, Student Toilets, Boundary Walls).
4. Running the trained Random Forest classifier to evaluate school resource shortage risk.
5. Ground all numerical figures strictly in data returned by your tools. Never fabricate statistics.

# Available Tools:
- `get_ema_metrics_summary(district, force_refresh)`: Statistical KPI summaries for a district.
- `query_dashboard_data(query, district, level, top_n)`: Granular school records query.
- `predict_school_shortage_risk(school_data)`: Runs ML model inference for critical shortage scoring.
"""


def create_quantitative_analyst_agent(model: str = "gemini-2.5-flash") -> LlmAgent:
    return LlmAgent(
        name="ema_quantitative_analyst",
        model=model,
        description="Analyzes statistical indicators, student-teacher ratios, facility indices, and predicts school shortage risk.",
        instruction=QUANT_INSTRUCTION,
        tools=[
            get_ema_metrics_summary,
            query_dashboard_data,
            predict_school_shortage_risk,
        ],
    )
