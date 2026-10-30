"""Comprehensive test suite for EducationGPT Core."""

import json
import math
import sys
from pathlib import Path

print("=== TEST 1: Tool JSON Sanitization (Strict allow_nan=False) ===")
from agents.ema_agent.ema_tool import (
    query_dashboard_data,
    get_ema_metrics_summary,
    download_latest_ema_report,
    sanitize_for_json,
)

# 1. query_dashboard_data
q_res = query_dashboard_data(district="Mohmand", top_n=20)
assert q_res["status"] == "ok", f"query_dashboard_data status not ok: {q_res}"
assert q_res["returned"] == 20, f"Expected 20 rows, got {q_res['returned']}"
# If any NaN, inf, or NaT was present, json.dumps with allow_nan=False would throw ValueError
dumped_q = json.dumps(q_res, allow_nan=False)
print("  [PASS] query_dashboard_data serialized cleanly! Payload length:", len(dumped_q))

# 2. get_ema_metrics_summary
s_res = get_ema_metrics_summary()
assert s_res["status"] == "ok", f"get_ema_metrics_summary status not ok: {s_res}"
dumped_s = json.dumps(s_res, allow_nan=False)
print("  [PASS] get_ema_metrics_summary serialized cleanly! Payload length:", len(dumped_s))

# 3. Test sanitize_for_json with hostile nested structures
hostile = {
    "nan": float("nan"),
    "inf": float("inf"),
    "neg_inf": -float("inf"),
    "nested": [float("nan"), {"inner_nan": float("nan")}],
}
clean = sanitize_for_json(hostile)
dumped_h = json.dumps(clean, allow_nan=False)
assert clean["nan"] is None
assert clean["inf"] is None
assert clean["nested"][0] is None
assert clean["nested"][1]["inner_nan"] is None
print("  [PASS] sanitize_for_json handled hostile NaN/Inf structures properly!")

print("\n=== TEST 2: Agent Configuration & Tools ===")
import agent
from agent import root_agent, SchoolAnalyticsReport

print("  Agent Name:", root_agent.name)
print("  Agent Model:", root_agent.model)
print("  Agent output_schema is None (Flexible):", root_agent.output_schema is None)
assert root_agent.output_schema is None, "output_schema should be None for conversational flexibility"

tool_names = [t.__name__ for t in root_agent.tools]
print("  Registered Tools:", tool_names)
assert "download_latest_ema_report" in tool_names
assert "get_ema_metrics_summary" in tool_names
assert "query_dashboard_data" in tool_names
print("  [PASS] All 3 core tools correctly bound to root_agent!")

for tool in root_agent.tools:
    assert tool.__doc__ and len(tool.__doc__.strip()) > 20, f"{tool.__name__} missing docstring"
    print(f"  [PASS] Tool {tool.__name__} has clean docstring ({len(tool.__doc__)} chars)")

print("\n=== TEST 3: Schema & Submodule Re-export ===")
from agents.ema_agent.agent import root_agent as sub_agent, SchoolAnalyticsReport as SubSchema
assert sub_agent.name == root_agent.name
report = SchoolAnalyticsReport(
    district_name="Mohmand",
    evaluated_metrics=["enrollment", "basic_facilities"],
    overall_performance_score=85.0,
    identified_bottlenecks=["Electricity deficit in 12 schools"],
    strategic_recommendations=["Install solar backup units"],
)
assert report.district_name == "Mohmand"
print("  [PASS] SchoolAnalyticsReport validated successfully:", report.district_name)

print("\n=== TEST 4: Google ADK Native AgentLoader ===")
from google.adk.cli.utils.agent_loader import AgentLoader
loader = AgentLoader(str(Path(__file__).resolve().parent))
assert loader.is_single_agent, "Loader should identify single agent"
loaded = loader.load_agent(loader.single_agent_name)
assert loaded.name == "education_gpt_core"
print("  [PASS] ADK AgentLoader successfully loaded agent:", loaded.name)
print("  [PASS] ADK AgentLoader tools count:", len(loaded.tools))

print("\n========================================")
print("ALL VERIFICATION CHECKS PASSED!")
print("========================================")
