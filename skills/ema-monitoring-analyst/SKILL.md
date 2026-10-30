---
name: ema-monitoring-analyst
description: Autonomous specialist skill for retrieving, tracking, and analyzing Khyber Pakhtunkhwa Education Monitoring Authority (KPEMA/IMU) reports, school census data, and district infrastructure metrics.
---

# EMA Monitoring Analyst Skill

## Overview
This skill provides a comprehensive operational handbook and toolset for the **Khyber Pakhtunkhwa Education Monitoring Authority (KPEMA / IMU)** system. It enables agents and analysts to retrieve, inspect, track, and deeply analyze school performance metrics, teacher/staff presence, student attendance, and basic physical facilities across Khyber Pakhtunkhwa districts (focusing on settled districts and newly merged tribal districts such as Mohmand).

The authoritative handbook of the EMA web dashboard is maintained in `ema_website_structure.txt`.

---

## 1. When to Use This Skill
Activate this skill whenever a task involves:
- Fetching or scraping live monitoring data from the KPEMA/IMU web dashboard (`http://175.107.63.44:81/imudashboard/index.php`).
- Analyzing monthly monitoring reports (Surveys, School Status, Teacher Presence, Student Attendance, Facilities).
- Ingesting Annual School Census (ASC) datasets (`Basic Facilities`, `Infrastructure`, `Sanctioned & Filled Staff`, `Classwise Enrollment`, `PTC Info`, `IT Lab`, `Disabled Students`).
- Calculating district education KPIs: Student-Teacher Ratio (STR), Student-Classroom Ratio (SCR), Teacher Unauthorized Absence Rate, Basic Facilities Functionality Index (BFFI), and PTC fund utilization.
- Identifying critical resource shortages, single-teacher school vulnerabilities, or missing basic utilities (drinking water, electricity, student toilets, boundary walls).
- Generating structured audit reports (`SchoolAnalyticsReport`) for District Education Officers (DEO) or education planners.

---

## 2. System Architecture & Portal Layout
The KPEMA dashboard operates on an authenticated PHP portal with AMCharts and HighCharts data visualizers backed by relational survey databases.

### Core Endpoints:
- **Base Dashboard**: `http://175.107.63.44:81/imudashboard/dashboard.php`
- **Authentication**: `POST /imudashboard/index.php` with credentials `username` & `password`.
- **Report Drilldown**: `reports.php?r=<report_slug>&districtCommaSep=<district>&reports_start_date=<date_range>`
- **Direct Search**:
  - `teacher-search.php` (Search teachers by CNIC, Personal No, or Name)
  - `non-teacher-search.php` (Search non-teaching staff)
  - `daily-visits.php` (FMO GPS track logs)
  - `schools-on-map.php` (GIS coordinates of schools)

---

## 3. Key Report Slugs & Categories

### A. Surveys
- `monitor-wise-survey-done`: Surveys completed per Field Monitoring Officer (FMO).
- `monitor-wise-coverage`: Coverage target percentage per monitor.
- `roaster-assignment`: Monthly scheduled inspection targets.
- `school-not-surveyed`: Schools missed during active monthly cycle.
- `survey-timing-report`: Arrival and departure time records at school sites.

### B. School Status & Infrastructure
- `closed-schools`: Unscheduled closed schools with recorded reasons.
- `basic-facilities`: Availability and functionality of electricity, water, toilets, and boundary walls.
- `building-availability`: Physical building existence and construction status (Pakka vs Kacha).
- `single-teacher`: Schools operating with only one teacher (critical vulnerability).
- `partially-occupied-schools` & `wholly-occupied-schools`: Encroachment or unauthorized occupation.

### C. Teaching & Non-Teaching Staff
- `absent-teaching-staff-unauth`: Unauthorized absent teachers (triggers formal salary deduction).
- `teaching-staff-attendance`: Overall presence rate (Present, Sanctioned Leave, Duty, Absent).
- `vacant-posts`: Unfilled sanctioned positions by school level and designation.
- `teacher-wrong-posting`: Subject-specialist teachers teaching unrelated subjects.
- `proxy-staff`: Discovered substitute/proxy teachers teaching illegally.

### D. Annual School Census (ASC)
- `furniture`: Student desks, chairs, and teacher furniture requirements vs available.
- `infrastructure`: Classroom counts, total land, covered area, uncovered area.
- `sanctioned-filled-teaching-staff`: Baseline sanctioned posts vs actual filled posts.
- `classwise-enrollment`: Detailed grade-by-grade enrollment (Nursery to Class 12).
- `census-ptc-info`: Parent Teacher Council bank accounts, balances, and annual funds.
- `PTCUnspentAmount`: Analysis of idle government funds sitting in PTC accounts.

---

## 4. Key Performance Indicators (KPIs) & Evaluation Rules

| Metric | Formula | Good Standard | Critical Alert |
| :--- | :--- | :--- | :--- |
| **Student-to-Teacher Ratio (STR)** | `Total Enrollment / Total Filled Teachers` | $\le 40:1$ | $> 50:1$ |
| **Student-to-Classroom Ratio (SCR)** | `Total Enrollment / Pakka Classrooms` | $\le 45:1$ | $> 60:1$ |
| **Teacher Unauthorized Absence** | `(Unauth Absent / Sanctioned Posts) * 100` | $< 2\%$ | $> 5\%$ |
| **Basic Facility Functionality Index** | `Mean(Water, Elec, Toilet, Wall, Bldg Func)` | $\ge 85\%$ | $< 65\%$ |
| **Single-Teacher School Count** | Total schools with $\le 1$ teacher | $0$ | $\ge 1$ |
| **PTC Fund Idle Balance** | Present Balance with $\ge 1$ non-func facility | PKR 0 - 50k | $> \text{PKR } 100\text{k}$ |

---

## 5. Multi-Agent Operational Workflow

When answering inquiries or conducting audits:
1. **Fetch / Verify Data**:
   - Check if fresh monthly report exists in `data/raw_reports/` or `data/reports/`.
   - If user requests fresh data, call the scraping tool `download_latest_ema_report()` or use the HTTP scraper.
   - For historical census data, load corresponding files from `data/raw/` (`Basic Facilities`, `Infrastructure`, `Sanctioned Filled Teaching Staff`, etc.).

2. **Quantitative Analysis**:
   - Compute district and school-level aggregations (enrollment totals, teacher vacancies, student-to-teacher ratios, attendance percentages).
   - Evaluate facility functionality rates (available vs working).

3. **Qualitative & Anomaly Inspection**:
   - Detect severe bottlenecks: single-teacher schools, closed schools, proxy teachers, unauthorized teacher absenteeism.
   - Cross-reference PTC fund balances with facility deficits (e.g. school has PKR 200,000 balance but no functional toilet).

4. **Predictive Risk Assessment**:
   - Feed school features into `models/school_model.pkl` or run heuristic shortage scoring to identify schools at imminent risk of critical failure.

5. **Synthesize Findings**:
   - Present clean Markdown tables with district comparisons.
   - Output structured JSON according to the `SchoolAnalyticsReport` schema when formal audits are requested.
