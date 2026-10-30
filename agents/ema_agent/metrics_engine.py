"""Quantitative and qualitative metrics engine for KPEMA reports and census data.

Bridges:
  1. Live or cached dashboard JSON states (Teacher Presence, Facilities, Attendance)
  2. Local Annual School Census (ASC) datasets and processed school data
  3. Machine Learning resource shortage prediction models
"""

import math
import os
import pickle
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from agents.ema_agent.config import (
    DATA_RAW_DIR,
    MODEL_PATH,
    MOHMAND_TEHSILS,
    PROCESSED_CSV_PATH,
)
from agents.ema_agent.dashboard_scraper import (
    HTTPDashboardClient,
    extract_js_variables_from_html,
    get_latest_cached_dashboard_html,
)


def sanitize_for_json(val: Any) -> Any:
    """Recursively converts all NaN, NaT, Inf, and non-serializable objects to None or valid JSON types."""
    if val is None:
        return None
    if val is pd.NA or val is pd.NaT:
        return None
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(float(val), 2)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (datetime, date, pd.Timestamp)):
        return val.isoformat()
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, dict):
        return {str(k): sanitize_for_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [sanitize_for_json(v) for v in val]
    if isinstance(val, pd.Series):
        return sanitize_for_json(val.to_dict())
    if isinstance(val, pd.DataFrame):
        return sanitize_for_json(val.to_dict(orient="records"))
    if isinstance(val, np.ndarray):
        return sanitize_for_json(val.tolist())
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    return val


class EMAMetricsEngine:
    """Core analytics engine for Education Monitoring Authority data."""

    def __init__(self):
        self._processed_df: Optional[pd.DataFrame] = None
        self._ml_model: Optional[Any] = None
        self._model_features: Optional[List[str]] = None
        self._load_processed_data()
        self._load_ml_model()

    def _load_processed_data(self):
        """Loads the processed school dataset if available."""
        if PROCESSED_CSV_PATH.exists():
            try:
                self._processed_df = pd.read_csv(PROCESSED_CSV_PATH, low_memory=False)
            except Exception as e:
                print(f"[EMAMetricsEngine] Could not read processed CSV: {e}")

    def _load_ml_model(self):
        """Loads the trained Random Forest shortage prediction model."""
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    model_bundle = pickle.load(f)
                if isinstance(model_bundle, dict):
                    self._ml_model = model_bundle.get("model")
                    self._model_features = model_bundle.get("features")
                else:
                    self._ml_model = model_bundle
            except Exception as e:
                print(f"[EMAMetricsEngine] Could not load ML model: {e}")

    # -------------------------------------------------------------------------
    # 1. High-Level Dashboard Metrics (Live / Cached)
    # -------------------------------------------------------------------------
    def get_dashboard_summary(
        self,
        district: str = "mohmand",
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Extracts and computes consolidated KPIs from the EMA dashboard."""
        html_text = None
        if force_refresh:
            client = HTTPDashboardClient()
            html_text = client.fetch_dashboard_page(district=district)

        if not html_text:
            html_text = get_latest_cached_dashboard_html()

        if not html_text:
            return {
                "status": "warning",
                "message": "No live or cached EMA dashboard HTML found. Please run download_latest_ema_report().",
            }

        js_data = extract_js_variables_from_html(html_text)
        survey = js_data.get("surveydata") or {}
        school_status = js_data.get("schoolStatusData") or {}
        teachers = js_data.get("TeacherPresenceData") or {}
        non_teachers = js_data.get("non_teacherPresence") or {}
        facilities = js_data.get("facilities") or {}
        students = js_data.get("students_attendance") or {}
        top_districts = js_data.get("TopDistricts") or {}
        single_teachers = js_data.get("singleTeacherSchools") or {}
        vacant_posts = js_data.get("vacantTeachers") or {}

        # Derived KPI calculations
        total_schools = int(survey.get("schoolscount") or school_status.get("totalschools") or 0)
        surveyed_schools = int(survey.get("totalsurveysdone") or 0)
        coverage_pct = float(survey.get("totalsurveypercentage") or 0.0)

        open_pct = float(school_status.get("openschoolspercentage") or 0.0)
        closed_pct = float(school_status.get("closedschoolspercentage") or 0.0)

        teacher_absent_pct = float(teachers.get("percentage") or 0.0)
        unauth_leave_pct = float(teachers.get("unauthorizedleavepercentage") or 0.0)
        teacher_sanctioned_leave = int(teachers.get("sanctionedleave") or 0)
        teacher_unauth_count = int(teachers.get("unauthorizedabent") or 0)

        student_absent_pct = float(students.get("totalabsentstudentspercent") or 0.0)
        student_absent_count = int(students.get("totalabsentstudents") or 0)
        total_students = int(students.get("totalstudents") or 0)

        # Basic Facilities Functionality Index
        facility_kpis = {
            "electricity_available_pct": float(facilities.get("totalelectricityavailablepercentage") or 0.0),
            "electricity_functional_pct": float(facilities.get("totalelectricityfunctionalpercentage") or 0.0),
            "water_available_pct": float(facilities.get("totalwateravailablepercentage") or 0.0),
            "water_functional_pct": float(facilities.get("totalwaterfunctionalpercentage") or 0.0),
            "toilet_available_pct": float(facilities.get("totaltoiletavailablepercentage") or 0.0),
            "toilet_functional_pct": float(facilities.get("totaltoiletfunctionalpercentage") or 0.0),
            "boundary_wall_available_pct": float(facilities.get("boundarywallavailablepercentage") or 0.0),
            "boundary_wall_functional_pct": float(facilities.get("boundarywallfunctionalpercentage") or 0.0),
            "building_available_pct": float(facilities.get("buildingavailablepercentage") or 0.0),
            "overall_facilities_pct": float(facilities.get("totalfacilitiespercentage") or 0.0),
        }

        # Student to Teacher Ratio
        total_teachers = int(teachers.get("totalteachers") or 0)
        str_ratio = round(total_students / max(total_teachers, 1), 1) if total_students > 0 else None

        summary = {
            "status": "ok",
            "district": district.upper(),
            "evaluation_date": js_data.get("selectedDate", "Latest Monitoring Cycle"),
            "monitoring_coverage": {
                "total_target_schools": total_schools,
                "surveyed_schools": surveyed_schools,
                "coverage_percentage": coverage_pct,
                "open_schools_pct": open_pct,
                "closed_schools_pct": closed_pct,
                "closed_schools_count": int(school_status.get("closedschools") or 0),
            },
            "student_metrics": {
                "total_enrolled_in_surveyed": total_students,
                "absent_students_count": student_absent_count,
                "absenteeism_rate_pct": student_absent_pct,
                "boys_absent_pct": float(students.get("totalboysabsentpercent") or 0.0),
                "girls_absent_pct": float(students.get("totalgirlsabsentpercent") or 0.0),
                "student_teacher_ratio": str_ratio,
            },
            "teacher_metrics": {
                "total_monitored_teachers": total_teachers,
                "overall_absence_rate_pct": teacher_absent_pct,
                "unauthorized_absent_count": teacher_unauth_count,
                "unauthorized_absent_rate_pct": unauth_leave_pct,
                "sanctioned_leave_count": teacher_sanctioned_leave,
                "on_duty_count": int(teachers.get("onduty") or 0),
                "proxy_teachers_detected": int(teachers.get("proxyteachers") or 0),
            },
            "infrastructure_and_facilities": facility_kpis,
            "vulnerabilities": {
                "single_teacher_schools": int(single_teachers.get("Single_Teacher_Schools") or 0),
                "vacant_teaching_posts": vacant_posts.get("total", 0),
            },
            "top_tehsils_by_attendance": top_districts.get("topStudentsArray", []),
            "source": "KPEMA Live/Cached Monitoring Feed",
        }

        return sanitize_for_json(summary)

    # -------------------------------------------------------------------------
    # 2. Granular School Records Query (Local Dataset + Census)
    # -------------------------------------------------------------------------
    def query_schools(
        self,
        query: str = "",
        district: Optional[str] = "mohmand",
        tehsil: Optional[str] = None,
        level: Optional[str] = None,
        top_n: int = 20,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Filters school-level records with free-text search and multi-attribute filters."""
        empty_res = {"status": "ok", "total_matched": 0, "returned": 0, "records": []}
        if self._processed_df is None or self._processed_df.empty:
            return empty_res

        df = self._processed_df.copy()

        # District filter
        if district:
            district_col = next((c for c in df.columns if c.lower() in ["district", "district_name", "dist"]), None)
            if district_col:
                df = df[df[district_col].astype(str).str.lower().str.contains(district.lower(), na=False)]

        # Level filter
        if level:
            level_col = next((c for c in df.columns if c.lower() in ["level", "school_level"]), None)
            if level_col:
                df = df[df[level_col].astype(str).str.lower().str.contains(level.lower(), na=False)]

        # Free text search across school name or EMIS
        if query and query.strip():
            q = query.strip().lower()
            text_cols = [c for c in df.columns if any(k in c.lower() for k in ["school", "name", "emis", "tehsil"])]
            if text_cols:
                mask = pd.Series(False, index=df.index)
                for col in text_cols:
                    mask = mask | df[col].astype(str).str.lower().str.contains(q, na=False)
                df = df[mask]

        if df.empty:
            return empty_res

        # Default informative columns if none requested
        if not columns:
            candidates = [
                "EMIS Code",
                "SchoolName",
                "school_name",
                "District",
                "LEVEL",
                "Gender",
                "total_teachers",
                "total_enrollment",
                "No of Pakka Class Rooms",
                "Presentbalance",
                "basic_water_functional",
                "basic_electric_functional",
                "basic_toilet_functional",
                "is_critical",
            ]
            columns = [c for c in candidates if c in df.columns]

        selected_cols = [c for c in columns if c in df.columns]
        results = df[selected_cols].head(top_n).to_dict(orient="records")
        return sanitize_for_json({
            "status": "ok",
            "total_matched": len(df),
            "returned": len(results),
            "records": results,
        })

    # -------------------------------------------------------------------------
    # 3. Deep Vulnerability & Shortage Audit
    # -------------------------------------------------------------------------
    def audit_district_vulnerabilities(self, district: str = "mohmand") -> Dict[str, Any]:
        """Conducts a deep cross-dataset audit to uncover critical school shortages."""
        summary = self.get_dashboard_summary(district=district)
        vulnerabilities = []

        # 1. Single teacher school alert
        single_ts = summary.get("vulnerabilities", {}).get("single_teacher_schools", 0)
        if single_ts > 0:
            vulnerabilities.append({
                "type": "Single-Teacher Vulnerability",
                "severity": "HIGH",
                "count": single_ts,
                "description": f"{single_ts} schools operate with only 1 teacher, creating severe multi-grade learning loss.",
            })

        # 2. Facility Deficits
        infra = summary.get("infrastructure_and_facilities", {})
        elec_func = infra.get("electricity_functional_pct", 100)
        water_func = infra.get("water_functional_pct", 100)
        toilet_func = infra.get("toilet_functional_pct", 100)

        if elec_func < 70:
            vulnerabilities.append({
                "type": "Electricity Functionality Deficit",
                "severity": "MEDIUM",
                "functional_rate": f"{elec_func}%",
                "description": f"Over {round(100 - elec_func)}% of surveyed schools lack working electricity.",
            })

        if water_func < 85:
            vulnerabilities.append({
                "type": "Clean Drinking Water Deficit",
                "severity": "HIGH",
                "functional_rate": f"{water_func}%",
                "description": f"Sanitary drinking water is non-functional in {round(100 - water_func)}% of visited schools.",
            })

        # 3. Unauthorized Absenteeism
        teachers = summary.get("teacher_metrics", {})
        unauth_rate = teachers.get("unauthorized_absent_rate_pct", 0)
        if unauth_rate > 5.0:
            vulnerabilities.append({
                "type": "Unauthorized Teacher Absenteeism",
                "severity": "CRITICAL",
                "rate": f"{unauth_rate}%",
                "unauth_count": teachers.get("unauthorized_absent_count", 0),
                "description": f"Unauthorized teacher absenteeism is {unauth_rate}%, well above the 5% disciplinary threshold.",
            })

        # 4. Student Absenteeism
        students = summary.get("student_metrics", {})
        stud_absent_rate = students.get("absenteeism_rate_pct", 0)
        if stud_absent_rate > 30.0:
            vulnerabilities.append({
                "type": "Severe Student Absenteeism",
                "severity": "HIGH",
                "rate": f"{stud_absent_rate}%",
                "description": f"{stud_absent_rate}% student absenteeism observed during latest monitoring cycles.",
            })

        # Calculate district health score out of 100
        health_score = 100.0
        health_score -= max(0, (50 - (summary.get("infrastructure_and_facilities", {}).get("overall_facilities_pct", 80) * 0.5)))
        health_score -= min(25, unauth_rate * 0.5)
        health_score -= min(20, (stud_absent_rate * 0.3))
        health_score = max(10.0, min(100.0, round(health_score, 1)))

        audit_report = {
            "district": district.upper(),
            "overall_performance_score": health_score,
            "active_bottlenecks": vulnerabilities,
            "evaluated_kpis": summary,
            "strategic_recommendations": [
                "Prioritize deployment of secondary teachers to single-teacher schools to eliminate multi-grade bottlenecks.",
                "Initiate departmental disciplinary action against persistent unauthorized absent staff.",
                "Direct District Education Officers (DEO) to utilize idle PTC funds for immediate restoration of non-functional toilets and water supply.",
                "Implement targeted parent engagement campaigns in tehsils with student absenteeism exceeding 40%.",
            ],
        }

        return sanitize_for_json(audit_report)

    # -------------------------------------------------------------------------
    # 4. ML Model Inference Integration
    # -------------------------------------------------------------------------
    def predict_shortage_risk(self, school_data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the trained Random Forest model to predict resource shortage risk."""
        if not self._ml_model:
            return {
                "status": "unavailable",
                "message": "Trained ML model (models/school_model.pkl) is not loaded.",
            }

        try:
            input_df = pd.DataFrame([school_data])
            # Align features
            if self._model_features:
                for col in self._model_features:
                    if col not in input_df.columns:
                        input_df[col] = 0.0
                input_df = input_df[self._model_features]

            pred = self._ml_model.predict(input_df)[0]
            probs = self._ml_model.predict_proba(input_df)[0].tolist() if hasattr(self._ml_model, "predict_proba") else [0.0, 1.0]

            return sanitize_for_json({
                "status": "success",
                "is_critical": int(pred),
                "category": "CRITICAL RESOURCE SHORTAGE" if pred == 1 else "NORMAL STATUS",
                "confidence": max(probs),
                "probabilities": probs,
            })
        except Exception as e:
            return {
                "status": "error",
                "message": f"Inference failure: {e}",
            }
