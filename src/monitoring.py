import os
import sys
import json
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

# Inject project root if not in sys.path
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.db_manager import get_db_path, get_recent_predictions, log_drift

FEATURES_TO_MONITOR = [
    ("total_teachers", "total_teachers"),
    ("No of Pakka Class Rooms", "pakka_class_rooms"),
    ("Presentbalance", "present_balance"),
    ("Covered_Area", "covered_area"),
    ("Desks Two Seater Useable", "usable_desks"),
    ("Fans Useable", "usable_fans"),
]

def calculate_drift(csv_path=None, db_path=None):
    """
    Computes Kolmogorov-Smirnov test for numerical features comparing baseline (CSV)
    and target (SQLite logged predictions).
    """
    if csv_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.normpath(os.path.join(base_dir, "..", "data", "processed", "processed_school_data.csv"))
        
    if db_path is None:
        db_path = get_db_path()
        
    if not os.path.exists(csv_path):
        return {"status": "error", "message": "Baseline processed CSV file not found."}
        
    # 1. Load baseline
    baseline_df = pd.read_csv(csv_path)
    
    # 2. Load predictions from DB
    preds = get_recent_predictions(db_path, limit=1000)
    if len(preds) < 5:
        return {
            "status": "awaiting_data",
            "message": f"Awaiting more predictions (Current: {len(preds)}/5 required for statistical drift analysis)."
        }
        
    target_df = pd.DataFrame(preds)
    
    drift_report = {}
    drift_detected = False
    
    for feat_csv, feat_db in FEATURES_TO_MONITOR:
        if feat_csv not in baseline_df.columns or feat_db not in target_df.columns:
            continue
            
        baseline_vals = baseline_df[feat_csv].dropna().values
        target_vals = target_df[feat_db].dropna().values
        
        if len(baseline_vals) == 0 or len(target_vals) == 0:
            continue
            
        # Run Kolmogorov-Smirnov test
        stat, p_val = ks_2samp(baseline_vals, target_vals)
        
        # We flag drift if p-value is less than 0.05
        is_drifted = bool(p_val < 0.05)
        if is_drifted:
            drift_detected = True
            
        drift_report[feat_csv] = {
            "feature": feat_csv,
            "ks_statistic": float(stat),
            "p_value": float(p_val),
            "is_drifted": is_drifted,
            "baseline_mean": float(np.mean(baseline_vals)),
            "target_mean": float(np.mean(target_vals)),
        }
        
    report = {
        "status": "success",
        "timestamp": pd.Timestamp.now().isoformat(),
        "drift_detected": drift_detected,
        "sample_size": len(preds),
        "metrics": drift_report
    }
    
    # Save log to database
    try:
        log_drift(db_path, json.dumps(report))
    except Exception as e:
        print(f"Error logging drift report: {e}")
        
    return report
