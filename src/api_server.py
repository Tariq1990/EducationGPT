from typing import Optional

import numpy as np
import os
import pandas as pd
import pickle
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="School Resource Shortage API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.normpath(os.path.join(BASE_DIR, "..", "models", "school_model.pkl")),
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


@app.get("/")
def read_root():
    return {"message": "School Resource Shortage Prediction API is running"}


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

    for col, le in model_data.get("encoders", {}).items():
        if col not in X_input.columns:
            continue
        try:
            X_input[col] = le.transform(X_input[col].astype(str))
        except ValueError:
            X_input[col] = 0

    numerical_cols = [c for c in model_data.get("numerical_cols", []) if c in X_input.columns]
    if numerical_cols:
        X_input[numerical_cols] = model_data["imputer"].transform(X_input[numerical_cols])

    probability = model_data["model"].predict_proba(X_input)[0].tolist()
    threshold = model_data.get("decision_threshold", 0.5)
    prediction = int(probability[1] >= threshold)

    return {
        "is_critical": int(prediction),
        "status": "Critical" if prediction == 1 else "Normal",
        "probability": probability,
        "threshold": threshold,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
