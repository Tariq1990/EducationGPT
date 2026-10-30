import os
import sys
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Inject project root if not in sys.path
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.db_manager import init_db, log_prediction, save_prediction_feedback, get_recent_predictions, get_db_path, get_model_metrics_history
from src.interpretability import compute_local_attribution
from src.monitoring import calculate_drift

# EMA Dashboard Agent tools
from agents.ema_agent.ema_tool import (
    download_latest_ema_report,
    get_ema_metrics_summary,
    query_dashboard_data,
    audit_district_shortages,
)

app = FastAPI(title="School Resource Shortage API")

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.normpath(os.path.join(PROJECT_ROOT, "models", "school_model.pkl")),
)

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
else:
    model_data = None
    print("Warning: school_model.pkl not found. Please run src/train_model.py first.")


class SchoolData(BaseModel):
    Gender: str
    LEVEL: str
    District: str
    BuildingOwnerShip: str
    TotalLand_Allocated: float
    Covered_Area: float
    UnCovered_Area: float
    No_of_Pakka_Class_Rooms: float
    No_Of_Pakka_Other_Rooms: float
    Desks_Two_Seater_Useable: float
    Tablet_Chairs_Useable: float
    Fans_Useable: float
    total_teachers: float
    avg_bps_teachers: float
    qualified_teachers: float
    total_non_teachers: float
    Presentbalance: float
    Fund_Recieved_from_Govt_in_This_Year: float

    # Optional advanced fields used if provided by external clients.
    total_enrollment: Optional[float] = None
    enrollment_primary: Optional[float] = None
    enrollment_middle_secondary: Optional[float] = None
    enrollment_higher_secondary: Optional[float] = None
    teach_sanctioned_posts: Optional[float] = None
    teach_filled_posts: Optional[float] = None
    teach_vacancy_posts: Optional[float] = None
    nonteach_sanctioned_posts: Optional[float] = None
    nonteach_filled_posts: Optional[float] = None
    nonteach_vacancy_posts: Optional[float] = None
    disabled_students_total: Optional[float] = None
    itlab_available: Optional[float] = None
    itlab_functional: Optional[float] = None
    itlab_total_computers: Optional[float] = None
    itlab_functional_computers: Optional[float] = None
    itlab_internet_available: Optional[float] = None


class FeedbackRequest(BaseModel):
    prediction_id: int
    feedback: str  # 'Correct' or 'Incorrect'


@app.on_event("startup")
def startup_event():
    db_path = get_db_path()
    init_db(db_path)
    print("FastAPI startup: initialized SQLite database.")


@app.get("/")
def read_root():
    db_status = "Initialized" if os.path.exists(get_db_path()) else "Not found"
    model_status = "Loaded" if model_data else "Not loaded"
    return {
        "message": "School Resource Shortage Prediction API is running",
        "database_status": db_status,
        "model_status": model_status
    }


@app.post("/predict")
def predict_shortage(data: SchoolData):
    if not model_data:
        raise HTTPException(status_code=500, detail="Model not loaded")

    input_dict = data.dict()
    features = model_data.get("features", [])

    # Start with feature defaults so newer models remain backward-compatible.
    formatted_input = {f: np.nan for f in features}

    # Map UI payload keys to training feature names.
    mapped = {
        "Gender": input_dict["Gender"],
        "LEVEL": input_dict["LEVEL"],
        "District": input_dict["District"],
        "BuildingOwnerShip": input_dict["BuildingOwnerShip"],
        "TotalLand_Allocated": input_dict["TotalLand_Allocated"],
        "Covered_Area": input_dict["Covered_Area"],
        "UnCovered_Area": input_dict["UnCovered_Area"],
        "No of Pakka Class Rooms": input_dict["No_of_Pakka_Class_Rooms"],
        "No. Of Pakka Other Rooms": input_dict["No_Of_Pakka_Other_Rooms"],
        "Desks Two Seater Useable": input_dict["Desks_Two_Seater_Useable"],
        "Tablet Chairs Useable": input_dict["Tablet_Chairs_Useable"],
        "Fans Useable": input_dict["Fans_Useable"],
        "total_teachers": input_dict["total_teachers"],
        "avg_bps_teachers": input_dict["avg_bps_teachers"],
        "qualified_teachers": input_dict["qualified_teachers"],
        "total_non_teachers": input_dict["total_non_teachers"],
        "Presentbalance": input_dict["Presentbalance"],
        "Fund Recieved from Govt in This Year": input_dict["Fund_Recieved_from_Govt_in_This_Year"],
    }

    # Apply canonical map.
    for k, v in mapped.items():
        if k in formatted_input:
            formatted_input[k] = v

    # Allow direct assignment for optional fields if names already match model features.
    for feat in features:
        if feat in input_dict and input_dict[feat] is not None:
            formatted_input[feat] = input_dict[feat]

    X_input = pd.DataFrame([formatted_input], columns=features)

    # Encode categorical features
    for col, le in model_data.get("encoders", {}).items():
        if col not in X_input.columns:
            continue
        try:
            X_input[col] = le.transform(X_input[col].astype(str))
        except ValueError:
            # Handle unseen categorical features gracefully
            X_input[col] = 0

    numerical_cols = [c for c in model_data.get("numerical_cols", []) if c in X_input.columns]
    if numerical_cols:
        X_input[numerical_cols] = model_data["imputer"].transform(X_input[numerical_cols])

    probability = model_data["model"].predict_proba(X_input)[0].tolist()
    threshold = model_data.get("decision_threshold", 0.5)
    prediction = int(probability[1] >= threshold)

    # 1. Log prediction to database
    db_path = get_db_path()
    try:
        pred_id = log_prediction(
            db_path=db_path,
            gender=input_dict["Gender"],
            level=input_dict["LEVEL"],
            district=input_dict["District"].upper(),
            teachers=input_dict["total_teachers"],
            rooms=input_dict["No_of_Pakka_Class_Rooms"],
            balance=input_dict["Presentbalance"],
            covered=input_dict["Covered_Area"],
            desks=input_dict["Desks_Two_Seater_Useable"],
            fans=input_dict["Fans_Useable"],
            is_critical=prediction,
            prob_critical=probability[1],
            threshold=threshold
        )
    except Exception as e:
        print(f"Error logging prediction: {e}")
        pred_id = None

    # 2. Compute local feature attribution
    explanation = {}
    try:
        attr = compute_local_attribution(model_data["model"], X_input, features)
        explanation = attr.get("contributions", {})
    except Exception as e:
        print(f"Error computing local attribution: {e}")

    # 3. Compute global feature importance
    global_importance = {}
    try:
        importances = model_data["model"].feature_importances_
        global_importance = dict(zip(features, [float(v) for v in importances]))
        global_importance = dict(sorted(global_importance.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        print(f"Error computing global feature importance: {e}")

    return {
        "prediction_id": pred_id,
        "is_critical": int(prediction),
        "status": "Critical" if prediction == 1 else "Normal",
        "probability": probability,
        "threshold": threshold,
        "explanation": explanation,
        "global_importance": global_importance
    }


@app.post("/predict/feedback")
def submit_feedback(data: FeedbackRequest):
    db_path = get_db_path()
    try:
        save_prediction_feedback(db_path, data.prediction_id, data.feedback)
        return {"status": "success", "message": f"Feedback registered for prediction {data.prediction_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/drift")
def get_drift():
    try:
        report = calculate_drift()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/stats")
def get_stats():
    db_path = get_db_path()
    try:
        preds = get_recent_predictions(db_path, limit=1000)
        if not preds:
            return {
                "total_predictions": 0,
                "critical_count": 0,
                "normal_count": 0,
                "feedback_accuracy": 0.0,
                "feedback_count": 0
            }

        total = len(preds)
        critical = sum(1 for p in preds if p["is_critical"] == 1)
        normal = total - critical

        feedback_preds = [p for p in preds if p["feedback"] is not None]
        correct_feedback = sum(1 for p in feedback_preds if p["feedback"] == "Correct")
        feedback_count = len(feedback_preds)
        accuracy = (correct_feedback / feedback_count) if feedback_count > 0 else 0.0

        return {
            "total_predictions": total,
            "critical_count": critical,
            "normal_count": normal,
            "feedback_accuracy": accuracy,
            "feedback_count": feedback_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/history")
def get_metrics_history():
    db_path = get_db_path()
    try:
        history = get_model_metrics_history(db_path, limit=20)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================
# EMA Dashboard Agent Routes
# ===========================================================================
@app.post("/ema/download")
def ema_download():
    """Trigger a browser-based login + download from the EMA dashboard."""
    result = download_latest_ema_report()
    if result["status"] != "ok":
        raise HTTPException(status_code=502, detail=result.get("message", "Download failed"))
    return result


@app.get("/ema/metrics")
def ema_metrics():
    """Return aggregated metrics from the most recent EMA report."""
    result = get_ema_metrics_summary()
    if result["status"] != "ok":
        raise HTTPException(status_code=404, detail=result.get("message", "No data"))
    return result


@app.get("/ema/query")
def ema_query(query: str = "", district: Optional[str] = None, top_n: int = 20):
    """Query processed EMA data with optional filters."""
    result = query_dashboard_data(query=query, district=district, top_n=top_n)
    if result["status"] != "ok":
        raise HTTPException(status_code=404, detail=result.get("message", "No data"))
    return result


@app.get("/ema/audit")
def ema_audit(district: str = "mohmand"):
    """Run comprehensive vulnerability and shortage audit for a district."""
    result = audit_district_shortages(district=district)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

