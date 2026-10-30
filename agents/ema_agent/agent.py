"""EducationGPT Master Monitoring Coordinator Agent.

Provides the configured root_agent, subagent wiring, and programmatic entry points.
Can be imported directly as:
    from agents.ema_agent.agent import root_agent, query_education_gpt, SchoolAnalyticsReport
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types
from pydantic import BaseModel, Field

from agents.ema_agent.config import EMA_DASHBOARD_URL
from agents.ema_agent.tools import (
    audit_district_shortages,
    download_latest_ema_report,
    get_ema_metrics_summary,
    predict_school_shortage_risk,
    query_dashboard_data,
)
from agents.ema_agent.subagents.data_fetcher_agent import create_data_fetcher_agent
from agents.ema_agent.subagents.quantitative_analyst_agent import create_quantitative_analyst_agent
from agents.ema_agent.subagents.qualitative_inspector_agent import create_qualitative_inspector_agent


# ==============================================================================
# 1. Output Schema Contract
# ==============================================================================
class SchoolAnalyticsReport(BaseModel):
    district_name: str = Field(description="Name of the educational district analyzed (e.g., Mohmand)")
    evaluated_metrics: List[str] = Field(description="Key performance indicators evaluated")
    overall_performance_score: float = Field(description="Calculated district health score between 0.0 and 100.0")
    student_teacher_ratio: Optional[float] = Field(default=None, description="Observed Student to Teacher Ratio")
    basic_facility_index: Optional[float] = Field(default=None, description="Composite facility functionality score")
    identified_bottlenecks: List[str] = Field(description="Key infrastructure, staff, or attendance bottlenecks")
    strategic_recommendations: List[str] = Field(description="Actionable policy interventions for education managers")


# ==============================================================================
# 2. Production Generation Config
# ==============================================================================
production_config = types.GenerateContentConfig(
    temperature=0.2,
    max_output_tokens=3500,
    top_p=0.85,
    top_k=20,
    safety_settings=[
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        )
    ],
)


# ==============================================================================
# 3. Master Multi-Agent Coordinator (EducationGPT-Core)
# ==============================================================================
COORDINATOR_INSTRUCTION = """
# Your Identity & Mission
You are EducationGPT-Core, an intelligent executive analytics and monitoring assistant specialized
in educational governance, school performance optimization, and district infrastructure planning
for Khyber Pakhtunkhwa (specifically settled and merged tribal districts like Mohmand).
You operate in full synergy with the KP Education Monitoring Authority (KPEMA / IMU) data structure
detailed in `ema_website_structure.txt`.

# Available Tools
1. `download_latest_ema_report()`:
   - Fetches / scrapes fresh monitoring data from the live KPEMA web dashboard.
   - Use when the user requests "fresh data", "scrape dashboard", or when datasets require updating.

2. `get_ema_metrics_summary(district="mohmand", force_refresh=False)`:
   - Returns aggregated district KPIs: total enrollment, monitored schools, teacher counts,
     attendance averages, unauthorized absence rates, student-to-teacher ratio (STR), and basic
     facilities functionality index (water, electricity, toilets, boundary wall).
   - Use when asked for district summaries, KPI overviews, or general trends.

3. `query_dashboard_data(query="", district="mohmand", level=None, top_n=20, columns=None, force_refresh=False)`:
   - Searches individual school records with district, level, and text filtering (e.g. school name, EMIS code).
   - Use when asked about specific schools, facility status (water, IT labs, pakka classrooms), or teacher counts.

4. `audit_district_shortages(district="mohmand")`:
   - Runs a multi-criteria audit detecting single-teacher schools, closed schools, facility deficits,
     and unauthorized absenteeism above threshold.
   - Use when the user requests an audit, risk evaluation, bottleneck analysis, or policy review.

5. `predict_school_shortage_risk(school_data)`:
   - Executes the trained Random Forest machine learning model to predict critical resource shortage risk.
   - Use when evaluating whether a school will fall into a critical shortage category.

# Response Guidelines
- Ground all statistics, school counts, and teacher numbers directly in tool outputs. Never fabricate figures.
- Format comparisons using clean GitHub-flavored Markdown tables.
- Highlight urgent vulnerabilities (single-teacher schools, absence > 5%, non-functional toilets/water).
- When asked for a formal audit or structured report, provide rich analysis followed by a JSON block conforming
  to `SchoolAnalyticsReport`.
"""

root_agent = LlmAgent(
    name="education_gpt_coordinator",
    model="gemini-2.5-flash",
    description="Master analytics coordinator for EducationGPT. Evaluates school performance, monitors KPEMA dashboard reports, and audits district infrastructure.",
    instruction=COORDINATOR_INSTRUCTION,
    tools=[
        download_latest_ema_report,
        get_ema_metrics_summary,
        query_dashboard_data,
        audit_district_shortages,
        predict_school_shortage_risk,
    ],
    generate_content_config=production_config,
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=1024,
        )
    ),
)


# ==============================================================================
# 4. Programmatic Query & Deterministic Fallback Entry Point
# ==============================================================================
def query_education_gpt(
    prompt: str,
    district: str = "mohmand",
    use_live_llm: bool = True,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Universal entry point to query the EducationGPT system.

    Supports both live Gemini ADK inference and offline deterministic execution.
    """
    p = prompt.lower()

    # Determine intent
    is_audit = any(k in p for k in ["audit", "vulnerability", "bottleneck", "shortage", "risk", "recommendation"])
    is_fetch = any(k in p for k in ["fetch", "scrape", "download", "fresh"])
    is_school_query = any(k in p for k in ["school", "primary", "middle", "high", "emis", "search", "list"])
    is_summary = any(k in p for k in ["summary", "kpi", "overview", "metric", "attendance", "ratio", "teacher", "facility"])

    # If Gemini API key is configured and live LLM requested, attempt ADK execution
    key = api_key or os.getenv("GEMINI_API_KEY")
    if use_live_llm and key:
        try:
            from google.adk.runtimes import run_agent_sync

            response = run_agent_sync(root_agent, prompt=prompt)
            return {
                "status": "success",
                "mode": "llm_agent",
                "response": str(response),
            }
        except Exception as e:
            print(f"[EducationGPT] ADK runner fell back to deterministic analytics: {e}")

    # Deterministic Execution Pathway
    if is_fetch:
        res = download_latest_ema_report()
        return {
            "status": "success",
            "mode": "deterministic",
            "intent": "download_fresh_data",
            "data": res,
            "markdown_response": f"### Fresh Data Fetch Status\n- **Status**: {res.get('status')}\n- **Message**: {res.get('message')}\n- **Mode**: {res.get('mode', 'N/A')}",
        }

    if is_audit:
        audit = audit_district_shortages(district=district)
        kpis = audit.get("evaluated_kpis", {})
        bottlenecks = audit.get("active_bottlenecks", [])
        recs = audit.get("strategic_recommendations", [])

        md = f"## Comprehensive Education Monitoring Audit: District {district.upper()}\n\n"
        md += f"**Overall District Health Score**: `{audit.get('overall_performance_score')}/100`\n\n"

        md += "### Key Monitoring Indicators\n"
        cov = kpis.get("monitoring_coverage", {})
        md += f"- **Target Schools**: {cov.get('total_target_schools')} | **Surveyed**: {cov.get('surveyed_schools')} ({cov.get('coverage_percentage')}%\n"
        md += f"- **School Status**: {cov.get('open_schools_pct')}% Open | {cov.get('closed_schools_pct')}% Closed ({cov.get('closed_schools_count')} schools)\n"

        teach = kpis.get("teacher_metrics", {})
        md += f"- **Teachers Monitored**: {teach.get('total_monitored_teachers')} | **Overall Absence**: {teach.get('overall_absence_rate_pct')}%\n"
        md += f"- **Unauthorized Absence**: {teach.get('unauthorized_absent_count')} teachers ({teach.get('unauthorized_absent_rate_pct')}%)\n"

        infra = kpis.get("infrastructure_and_facilities", {})
        md += f"- **Clean Water Functional**: {infra.get('water_functional_pct')}% | **Electricity Functional**: {infra.get('electricity_functional_pct')}%\n"
        md += f"- **Toilets Functional**: {infra.get('toilet_functional_pct')}% | **Overall Facilities Index**: {infra.get('overall_facilities_pct')}%\n\n"

        md += "### Active Bottlenecks & Critical Alerts\n"
        for b in bottlenecks:
            md += f"- **[{b.get('severity')}] {b.get('type')}**: {b.get('description')}\n"

        md += "\n### Strategic Policy Recommendations\n"
        for i, r in enumerate(recs, 1):
            md += f"{i}. {r}\n"

        return {
            "status": "success",
            "mode": "deterministic",
            "intent": "district_audit",
            "data": audit,
            "markdown_response": md,
        }

    if is_school_query and not is_summary:
        schools_res = query_dashboard_data(query=prompt, district=district, top_n=10)
        records = schools_res.get("records", []) if isinstance(schools_res, dict) else schools_res
        md = f"### School Inspection Records ({len(records)} matches found)\n\n"
        if records:
            import pandas as pd
            df_view = pd.DataFrame(records)
            md += df_view.to_markdown(index=False)
        else:
            md += "No matching school records found for the query."

        return {
            "status": "success",
            "mode": "deterministic",
            "intent": "school_query",
            "data": schools_res,
            "markdown_response": md,
        }

    # Default to summary
    summary = get_ema_metrics_summary(district=district)
    md = f"## Education Monitoring Summary: District {district.upper()}\n\n"
    cov = summary.get("monitoring_coverage", {})
    md += f"- **Surveyed Schools**: {cov.get('surveyed_schools')} of {cov.get('total_target_schools')} ({cov.get('coverage_percentage')}% coverage)\n"

    stud = summary.get("student_metrics", {})
    md += f"- **Total Students**: {stud.get('total_enrolled_in_surveyed'):,} | **Absenteeism**: {stud.get('absenteeism_rate_pct')}%\n"
    if stud.get("student_teacher_ratio"):
        md += f"- **Student to Teacher Ratio (STR)**: `{stud.get('student_teacher_ratio')}:1`\n"

    teach = summary.get("teacher_metrics", {})
    md += f"- **Monitored Teachers**: {teach.get('total_monitored_teachers')} | **Unauthorized Absence**: {teach.get('unauthorized_absent_rate_pct')}%\n"

    infra = summary.get("infrastructure_and_facilities", {})
    md += f"- **Facilities Functionality**: Water {infra.get('water_functional_pct')}% | Toilets {infra.get('toilet_functional_pct')}% | Electricity {infra.get('electricity_functional_pct')}%\n"

    vuln = summary.get("vulnerabilities", {})
    md += f"- **Single Teacher Schools**: `{vuln.get('single_teacher_schools')}`\n"

    return {
        "status": "success",
        "mode": "deterministic",
        "intent": "metrics_summary",
        "data": summary,
        "markdown_response": md,
    }
