import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


FEATURES = [
    "Gender",
    "LEVEL",
    "District",
    "BuildingOwnerShip",
    "TotalLand_Allocated",
    "Covered_Area",
    "UnCovered_Area",
    "No of Pakka Class Rooms",
    "No. Of Pakka Other Rooms",
    "Desks Two Seater Useable",
    "Tablet Chairs Useable",
    "Fans Useable",
    "total_teachers",
    "avg_bps_teachers",
    "qualified_teachers",
    "total_non_teachers",
    "Presentbalance",
    "Fund Recieved from Govt in This Year",
    "itlab_available",
    "itlab_functional",
    "itlab_total_computers",
    "itlab_functional_computers",
    "itlab_internet_available",
    "total_enrollment",
    "enrollment_primary",
    "enrollment_middle_secondary",
    "enrollment_higher_secondary",
    "teach_sanctioned_posts",
    "teach_filled_posts",
    "teach_vacancy_posts",
    "nonteach_sanctioned_posts",
    "nonteach_filled_posts",
    "nonteach_vacancy_posts",
    "disabled_students_total",
    "basic_electric_available",
    "basic_electric_functional",
    "basic_water_available",
    "basic_water_functional",
    "basic_toilet_available",
    "basic_toilet_functional",
    "basic_wall_available",
    "basic_wall_functional",
]

CATEGORICAL_COLS = ["Gender", "LEVEL", "District", "BuildingOwnerShip"]


def train_school_model(csv_path):
    print("Loading processed data...")
    df = pd.read_csv(csv_path)

    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    X = df[FEATURES].copy()
    y = df["is_critical"].copy()

    print("Preprocessing data...")
    numerical_cols = [c for c in FEATURES if c not in CATEGORICAL_COLS]

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    imputer = SimpleImputer(strategy="median")
    X[numerical_cols] = imputer.fit_transform(X[numerical_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Initializing RandomForestClassifier...")
    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )

    print("Training (Fitting) model...")
    classifier.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

    thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_macro_f1 = -1.0
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        macro_f1 = f1_score(y_test, y_pred_t, average="macro", zero_division=0)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = float(t)

    print(f"Selected decision threshold: {best_threshold:.2f} (macro F1: {best_macro_f1:.4f})")

    model_dir = os.path.normpath(os.path.join(os.path.dirname(csv_path), "..", "..", "models"))
    os.makedirs(model_dir, exist_ok=True)
    model_data = {
        "model": classifier,
        "encoders": encoders,
        "imputer": imputer,
        "features": FEATURES,
        "categorical_cols": CATEGORICAL_COLS,
        "numerical_cols": numerical_cols,
        "decision_threshold": best_threshold,
    }

    out_model = os.path.join(model_dir, "school_model.pkl")
    with open(out_model, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\nModel and preprocessors saved to {out_model}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.getenv(
        "PROCESSED_CSV_PATH",
        os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "processed_school_data.csv")),
    )
    train_school_model(CSV_PATH)
