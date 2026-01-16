from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os
import uvicorn

app = FastAPI(title="School Resource Shortage API")

# Load model and preprocessors
MODEL_PATH = r"c:\Users\Engr.Tariq Jamal\Downloads\EMA_ML_model\school_model.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
else:
    model_data = None
    print("Warning: school_model.pkl not found. Please run train_model.py first.")

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

@app.get("/")
def read_root():
    return {"message": "School Resource Shortage Prediction API is running"}

@app.post("/predict")
def predict_shortage(data: SchoolData):
    if not model_data:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Prepare input data
    input_dict = data.dict()
    # Handle underscore vs space in feature names
    formatted_input = {
        'Gender': input_dict['Gender'],
        'LEVEL': input_dict['LEVEL'],
        'District': input_dict['District'],
        'BuildingOwnerShip': input_dict['BuildingOwnerShip'],
        'TotalLand_Allocated': input_dict['TotalLand_Allocated'],
        'Covered_Area': input_dict['Covered_Area'],
        'UnCovered_Area': input_dict['UnCovered_Area'],
        'No of Pakka Class Rooms': input_dict['No_of_Pakka_Class_Rooms'],
        'No. Of Pakka Other Rooms': input_dict['No_Of_Pakka_Other_Rooms'],
        'Desks Two Seater Useable': input_dict['Desks_Two_Seater_Useable'],
        'Tablet Chairs Useable': input_dict['Tablet_Chairs_Useable'],
        'Fans Useable': input_dict['Fans_Useable'],
        'total_teachers': input_dict['total_teachers'],
        'avg_bps_teachers': input_dict['avg_bps_teachers'],
        'qualified_teachers': input_dict['qualified_teachers'],
        'total_non_teachers': input_dict['total_non_teachers'],
        'Presentbalance': input_dict['Presentbalance'],
        'Fund Recieved from Govt in This Year': input_dict['Fund_Recieved_from_Govt_in_This_Year']
    }


    X_input = pd.DataFrame([formatted_input])

    # Preprocess
    for col, le in model_data['encoders'].items():
        # Handle unseen labels by mapping to a default if necessary (simplistic)
        try:
            X_input[col] = le.transform(X_input[col].astype(str))
        except ValueError:
            # If unseen, map to the first class as a fallback
            X_input[col] = 0 

    X_input[model_data['numerical_cols']] = model_data['imputer'].transform(X_input[model_data['numerical_cols']])

    # Predict
    prediction = model_data['model'].predict(X_input)[0]
    probability = model_data['model'].predict_proba(X_input)[0].tolist()

    return {
        "is_critical": int(prediction),
        "status": "Critical" if prediction == 1 else "Normal",
        "probability": probability
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
