import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import os

def train_school_model(csv_path):
    print("Loading processed data...")
    df = pd.read_csv(csv_path)

    features = [
        'Gender', 'LEVEL', 'District', 'BuildingOwnerShip', 
        'TotalLand_Allocated', 'Covered_Area', 'UnCovered_Area',
        'No of Pakka Class Rooms', 'No. Of Pakka Other Rooms',
        'Desks Two Seater Useable', 'Tablet Chairs Useable', 'Fans Useable',
        'total_teachers', 'avg_bps_teachers', 'qualified_teachers',
        'total_non_teachers', 'Presentbalance', 'Fund Recieved from Govt in This Year'
    ]
    
    X = df[features].copy()
    y = df['is_critical'].copy()

    # Preprocessing
    print("Preprocessing data...")
    categorical_cols = ['Gender', 'LEVEL', 'District', 'BuildingOwnerShip']

    numerical_cols = [c for c in features if c not in categorical_cols]

    # Handle Categorical
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle unseen labels by adding a 'Unknown' class if needed
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Handle Numerical Missing Values
    imputer = SimpleImputer(strategy='median')
    X[numerical_cols] = imputer.fit_transform(X[numerical_cols])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # RandomForest (Fallback from TabPFN due to environment issues)
    print("Initializing RandomForestClassifier...")
    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
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

    # Tune threshold for better class balance (macro F1)
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


    # Save components
    model_dir = os.path.normpath(os.path.join(os.path.dirname(csv_path), "..", "..", "models"))
    os.makedirs(model_dir, exist_ok=True)
    model_data = {
        'model': classifier,
        'encoders': encoders,
        'imputer': imputer,
        'features': features,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'decision_threshold': best_threshold
    }
    
    with open(os.path.join(model_dir, 'school_model.pkl'), 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel and preprocessors saved to {os.path.join(model_dir, 'school_model.pkl')}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.getenv(
        "PROCESSED_CSV_PATH",
        os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "processed_school_data.csv"))
    )
    train_school_model(CSV_PATH)
