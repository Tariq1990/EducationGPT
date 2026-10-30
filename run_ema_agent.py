#!/usr/bin/env python
"""EducationGPT EMA Agent CLI & Execution Entry Point.

Easily query, monitor, and analyze Khyber Pakhtunkhwa Education Monitoring Authority (KPEMA) reports.

Usage examples:
    python run_ema_agent.py --summary
    python run_ema_agent.py --audit --district mohmand
    python run_ema_agent.py --query "Primary" --district mohmand
    python run_ema_agent.py --fetch
    python run_ema_agent.py --interactive
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.ema_agent.agent import query_education_gpt
from agents.ema_agent.tools import (
    audit_district_shortages,
    download_latest_ema_report,
    get_ema_metrics_summary,
    query_dashboard_data,
)


def print_banner():
    banner = """
================================================================================
           EducationGPT | KPEMA Education Monitoring Analytics
================================================================================
    """
    print(banner)


def handle_summary(district: str):
    print(f"\n[+] Computing monitoring metrics summary for District: {district.upper()}...")
    summary = get_ema_metrics_summary(district=district)
    print("\n" + "=" * 70)
    print(f"DISTRICT KPI SUMMARY: {district.upper()} ({summary.get('evaluation_date', 'N/A')})")
    print("=" * 70)

    cov = summary.get("monitoring_coverage", {})
    print(f"  * Monitoring Target Schools : {cov.get('total_target_schools')}")
    print(f"  * Schools Visited/Surveyed  : {cov.get('surveyed_schools')} ({cov.get('coverage_percentage')}%)")
    print(f"  * Functioning Status        : {cov.get('open_schools_pct')}% Open, {cov.get('closed_schools_pct')}% Closed ({cov.get('closed_schools_count')} schools)")

    stud = summary.get("student_metrics", {})
    print(f"\n  * Total Students Enrolled   : {stud.get('total_enrolled_in_surveyed'):,}")
    print(f"  * Student Absenteeism Rate  : {stud.get('absenteeism_rate_pct')}% ({stud.get('absent_students_count'):,} absent)")
    if stud.get("student_teacher_ratio"):
        print(f"  * Student-to-Teacher Ratio  : {stud.get('student_teacher_ratio')}:1")

    teach = summary.get("teacher_metrics", {})
    print(f"\n  * Monitored Teachers Count  : {teach.get('total_monitored_teachers')}")
    print(f"  * Overall Teacher Absence   : {teach.get('overall_absence_rate_pct')}%")
    print(f"  * Unauthorized Absence Rate : {teach.get('unauthorized_absent_rate_pct')}% ({teach.get('unauthorized_absent_count')} staff)")

    infra = summary.get("infrastructure_and_facilities", {})
    print("\n  * Basic Facilities Functionality:")
    print(f"      - Clean Drinking Water  : {infra.get('water_functional_pct')}%")
    print(f"      - Working Electricity   : {infra.get('electricity_functional_pct')}%")
    print(f"      - Student Toilets       : {infra.get('toilet_functional_pct')}%")
    print(f"      - Boundary Wall         : {infra.get('boundary_wall_functional_pct')}%")
    print(f"      - Overall Index         : {infra.get('overall_facilities_pct')}%")

    vuln = summary.get("vulnerabilities", {})
    print(f"\n  * Single Teacher Schools    : {vuln.get('single_teacher_schools')}")
    print("=" * 70)


def handle_audit(district: str):
    print(f"\n[+] Executing deep multi-criteria audit for District: {district.upper()}...")
    audit = audit_district_shortages(district=district)
    print("\n" + "=" * 70)
    print(f"DISTRICT AUDIT REPORT: {district.upper()}")
    print(f"Overall District Health Score: {audit.get('overall_performance_score')}/100")
    print("=" * 70)

    print("\n[!] Identified Bottlenecks & Vulnerabilities:")
    for b in audit.get("active_bottlenecks", []):
        print(f"  [{b.get('severity')}] {b.get('type')}: {b.get('description')}")

    print("\n[*] Strategic Recommendations for Education Planners:")
    for i, r in enumerate(audit.get("strategic_recommendations", []), 1):
        print(f"  {i}. {r}")
    print("=" * 70)


def handle_query(query_text: str, district: str, top_n: int):
    print(f"\n[+] Searching school census records for: '{query_text}' in {district.upper()}...")
    res = query_dashboard_data(query=query_text, district=district, top_n=top_n)
    records = res.get("records", [])
    print(f"Found {res.get('total_matched', len(records))} match(es) (displaying top {len(records)}):\n")

    if not records:
        print("  No records found matching criteria.")
        return

    for idx, r in enumerate(records, 1):
        name = r.get("SchoolName") or r.get("school_name") or r.get("EMIS Code") or "School"
        emis = r.get("EMIS Code", "N/A")
        lvl = r.get("LEVEL", "N/A")
        gender = r.get("Gender", "N/A")
        teachers = r.get("total_teachers", "N/A")
        enrol = r.get("total_enrollment", "N/A")
        print(f"  {idx}. [{emis}] {name} ({lvl}, {gender})")
        print(f"     Enrolment: {enrol} | Teachers: {teachers} | PTC Balance: PKR {r.get('Presentbalance', '0')}")


def handle_fetch():
    print("\n[+] Triggering live retrieval from EMA dashboard...")
    res = download_latest_ema_report()
    print(f"Status  : {res.get('status')}")
    print(f"Message : {res.get('message')}")
    print(f"Mode    : {res.get('mode')}")


def handle_interactive(district: str, offline: bool):
    print_banner()
    print(f"Interactive Assistant started for District: {district.upper()}")
    print("Type your questions (e.g. 'Audit Mohmand shortages', 'Show single teacher schools', 'quit').\n")

    while True:
        try:
            user_input = input("EducationGPT> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye.")
            break

        res = query_education_gpt(
            prompt=user_input,
            district=district,
            use_live_llm=not offline,
        )

        md = res.get("markdown_response") or res.get("response") or str(res)
        print("\n" + md + "\n")


def main():
    parser = argparse.ArgumentParser(description="EducationGPT EMA Agent CLI")
    parser.add_argument("--district", type=str, default="mohmand", help="Target educational district (default: mohmand)")
    parser.add_argument("--summary", action="store_true", help="Display district KPI summary")
    parser.add_argument("--audit", action="store_true", help="Run comprehensive district shortage and vulnerability audit")
    parser.add_argument("--query", type=str, default=None, help="Search school records with keyword")
    parser.add_argument("--top-n", type=int, default=10, help="Max records for query (default: 10)")
    parser.add_argument("--fetch", action="store_true", help="Trigger live dashboard data sync")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch conversational REPL session")
    parser.add_argument("--offline", action="store_true", help="Run in deterministic offline mode without calling LLM APIs")

    args = parser.parse_args()

    if args.summary:
        handle_summary(args.district)
    elif args.audit:
        handle_audit(args.district)
    elif args.query is not None:
        handle_query(args.query, args.district, args.top_n)
    elif args.fetch:
        handle_fetch()
    elif args.interactive or len(sys.argv) == 1:
        handle_interactive(args.district, args.offline)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
