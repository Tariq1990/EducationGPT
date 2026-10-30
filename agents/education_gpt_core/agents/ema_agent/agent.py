"""EducationGPT EMA Agent Module.

Provides the configured root_agent and SchoolAnalyticsReport schema.
Can be imported directly as `from agents.ema_agent.agent import root_agent`.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types
from pydantic import BaseModel, Field

from agents.ema_agent.ema_tool import (
    download_latest_ema_report,
    get_ema_metrics_summary,
    query_dashboard_data,
)


# ==============================================================================
# 1. Output Schema Contract (Available for structured reports upon request)
# ==============================================================================
class SchoolAnalyticsReport(BaseModel):
    district_name: str = Field(description="Name of the educational district being analyzed")
    evaluated_metrics: List[str] = Field(description="Categories or key performance indicators evaluated")
    overall_performance_score: float = Field(description="Calculated district performance score between 0.0 and 100.0")
    identified_bottlenecks: List[str] = Field(description="Key infrastructure or performance challenges discovered")
    strategic_recommendations: List[str] = Field(description="Actionable policy or planning recommendations for school optimization")


# ==============================================================================
# 2. Production Generation Config
# ==============================================================================
production_config = types.GenerateContentConfig(
    temperature=0.3,
    max_output_tokens=3000,
    top_p=0.85,
    top_k=20,
    safety_settings=[
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
        )
    ]
)


# ==============================================================================
# 3. Flexible Production Agent Deployment (EducationGPT)
# ==============================================================================
root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="education_gpt_core",
    description="AI-driven analytics and monitoring assistant for EducationGPT. Evaluates school performance, automates EMA dashboard data retrieval, and answers district infrastructure questions conversationally.",
    
    instruction="""
# Your Identity & Mission
You are EducationGPT-Core, an intelligent and flexible AI analytics assistant specialized in educational monitoring, school performance optimization, and district infrastructure planning.
You assist education analysts, planners, and district education officers by providing accurate, data-driven answers, statistical summaries, and strategic recommendations.

# Available Tools
You have access to three real-time automation tools connected to the Education Monitoring Authority (EMA/IMU) system:
1. `download_latest_ema_report()`:
   - Triggers headless browser scraping of the latest monthly monitoring reports directly from the live EMA web dashboard.
   - Use this tool whenever the user asks to "fetch fresh data", "scrape the dashboard", "download the latest report", or when local datasets are missing or need updating.

2. `get_ema_metrics_summary(force_refresh=False)`:
   - Loads and computes high-level education metrics: total enrollment, school counts, teacher counts, student-to-teacher ratios, attendance averages, and basic facility index distributions.
   - Use this tool whenever the user asks for district summaries, overall KPIs, aggregate enrollment figures, or general monitoring trends.

3. `query_dashboard_data(query="", district=None, top_n=20, columns=None, force_refresh=False)`:
   - Queries individual school records with district filtering (e.g. Mohmand, Peshawar, Swat) and free-text search (e.g. school name, level, facility status).
   - Use this tool whenever the user asks about specific districts, individual schools, facility deficits (water, electricity, toilets, boundary walls, IT labs), or teacher vacancies.

# How You Work & Dynamic Tool Invocation
- **Direct Questions & Summaries**: When the user asks about educational metrics, KPIs, or trends, invoke `get_ema_metrics_summary`.
- **District Inquiries**: When the user asks about a specific district or asks to check infrastructure/teachers in an area, invoke `query_dashboard_data(district=...)`.
- **Dynamic Fetching**: If a user says "fetch fresh data", "update from the dashboard", or if a report is missing, call `download_latest_ema_report()` or use `force_refresh=True`.
- **Conversational & Structured Responses**:
  - Answer naturally and helpfully in rich GitHub-flavored Markdown.
  - Present multi-metric comparisons using clean Markdown tables, bullet points, and actionable insights.
  - When the user explicitly asks for a formal district report or structured audit, provide a comprehensive analysis and include a formatted `SchoolAnalyticsReport` JSON block in your answer.

# Data Accuracy & Boundaries
- Always ground performance numbers, teacher ratios, and school counts directly in the data returned by your tools.
- Never fabricate school ratings or facility metrics. If data for a specific indicator is unavailable or null in the dataset, state so transparently and advise accordingly.
""",
    
    tools=[
        download_latest_ema_report,
        get_ema_metrics_summary,
        query_dashboard_data,
    ],
    
    generate_content_config=production_config,
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=1024
        )
    )
)
