import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Inject project root if not in sys.path
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.db_manager import init_db, log_model_metrics, get_db_path

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


def resample_data(X_train, y_train, strategy="balanced"):
    """
    Applies the chosen imbalance strategy to the training data.
    """
    if strategy == "balanced":
        # Random Forest handles class weighting internally
        print("Using class_weight='balanced' in RandomForest. No preprocessing resampling.")
        return X_train, y_train, "balanced"

    elif strategy == "smote":
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            print(f"Applied SMOTE: resampled training size from {len(X_train)} to {len(X_res)}")
            return X_res, y_res, "smote"
        except ImportError:
            print("Warning: imbalanced-learn (imblearn) not installed. Falling back to Random Over-Sampling.")
            strategy = "random_oversample"

    if strategy == "random_oversample":
        # Standard Random Over-Sampling implementation using pandas
        np.random.seed(42)
        classes, counts = np.unique(y_train, return_counts=True)
        if len(classes) < 2:
            return X_train, y_train, "none"

        max_count = np.max(counts)
        X_resampled = []
        y_resampled = []

        for c in classes:
            X_c = X_train[y_train == c]
            y_c = y_train[y_train == c]
            if len(X_c) < max_count:
                indices = np.random.choice(len(X_c), size=max_count, replace=True)
                X_resampled.append(X_c.iloc[indices])
                y_resampled.append(y_c.iloc[indices])
            else:
                X_resampled.append(X_c)
                y_resampled.append(y_c)

        X_res = pd.concat(X_resampled)
        y_res = pd.concat(y_resampled)
        print(f"Applied Random Over-Sampling: resampled training size from {len(X_train)} to {len(X_res)}")
        return X_res, y_res, "random_oversample"

    return X_train, y_train, "none"


def train_school_model(csv_path, strategy="balanced"):
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

    # Apply resampling strategy
    X_train_res, y_train_res, applied_strategy = resample_data(X_train, y_train, strategy)

    print(f"Initializing RandomForestClassifier (imbalance strategy: {applied_strategy})...")
    if applied_strategy == "balanced":
        classifier = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight=None,
        )

    print("Training (Fitting) model...")
    classifier.fit(X_train_res, y_train_res)

    print("Evaluating...")
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_macro_f1 = -1.0
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        macro_f1_t = f1_score(y_test, y_pred_t, average="macro", zero_division=0)
        if macro_f1_t > best_macro_f1:
            best_macro_f1 = macro_f1_t
            best_threshold = float(t)

    print(f"Selected decision threshold: {best_threshold:.2f} (macro F1: {best_macro_f1:.4f})")

    # Logging to SQLite database
    db_path = get_db_path()
    init_db(db_path)
    log_model_metrics(
        db_path=db_path,
        accuracy=acc,
        f1_score=f1,
        macro_f1_score=best_macro_f1,
        decision_threshold=best_threshold,
        imbalance_strategy=applied_strategy,
        features_list=FEATURES
    )
    print("Logged model metrics to SQLite.")

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
        "imbalance_strategy": applied_strategy,
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
    strategy = os.getenv("IMBALANCE_STRATEGY", "balanced")
    train_school_model(CSV_PATH, strategy)

