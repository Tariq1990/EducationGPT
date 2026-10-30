"""Specialized subagents for the EducationGPT Multi-Agent Monitoring Architecture."""

from agents.ema_agent.subagents.data_fetcher_agent import create_data_fetcher_agent
from agents.ema_agent.subagents.quantitative_analyst_agent import create_quantitative_analyst_agent
from agents.ema_agent.subagents.qualitative_inspector_agent import create_qualitative_inspector_agent

__all__ = [
    "create_data_fetcher_agent",
    "create_quantitative_analyst_agent",
    "create_qualitative_inspector_agent",
]
