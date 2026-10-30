import os
import sqlite3
import json
from datetime import datetime

DEFAULT_DB_NAME = "education_gpt.db"

def get_db_path():
    # Resolves C:\Users\Engr.Tariq Jamal\Documents\GitHub\educationGPT\data\education_gpt.db
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.normpath(os.path.join(base_dir, "..", "data"))
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, DEFAULT_DB_NAME)

def get_connection(db_path=None):
    if db_path is None:
        db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path=None):
    if db_path is None:
        db_path = get_db_path()
        
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # 1. Predictions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        gender TEXT,
        level TEXT,
        district TEXT,
        total_teachers REAL,
        pakka_class_rooms REAL,
        present_balance REAL,
        covered_area REAL,
        usable_desks REAL,
        usable_fans REAL,
        is_critical INTEGER,
        probability_critical REAL,
        threshold REAL,
        feedback TEXT
    )
    """)
    
    # 2. Chat History Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        role TEXT,
        content TEXT,
        model_name TEXT,
        sources TEXT
    )
    """)
    
    # 3. Model Metrics Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        accuracy REAL,
        f1_score REAL,
        macro_f1_score REAL,
        decision_threshold REAL,
        imbalance_strategy TEXT,
        features_list TEXT
    )
    """)
    
    # 4. Drift Logs Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS drift_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        metrics_json TEXT
    )
    """)
    
    conn.commit()
    conn.close()
    print(f"Database initialized at: {db_path}")

def log_prediction(db_path, gender, level, district, teachers, rooms, balance, covered, desks, fans, is_critical, prob_critical, threshold):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO predictions (
        gender, level, district, total_teachers, pakka_class_rooms,
        present_balance, covered_area, usable_desks, usable_fans,
        is_critical, probability_critical, threshold, feedback
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        gender, level, district, teachers, rooms,
        balance, covered, desks, fans,
        int(is_critical), float(prob_critical), float(threshold), None
    ))
    pred_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return pred_id

def save_prediction_feedback(db_path, pred_id, feedback):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE predictions SET feedback = ? WHERE id = ?
    """, (feedback, pred_id))
    conn.commit()
    conn.close()

def get_recent_predictions(db_path, limit=100):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def log_chat_message(db_path, session_id, role, content, model_name="", sources=""):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO chat_history (session_id, role, content, model_name, sources)
    VALUES (?, ?, ?, ?, ?)
    """, (session_id, role, content, model_name, sources))
    conn.commit()
    conn.close()

def get_chat_history(db_path, session_id, limit=50):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM chat_history WHERE session_id = ? ORDER BY timestamp ASC LIMIT ?
    """, (session_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def log_model_metrics(db_path, accuracy, f1_score, macro_f1_score, decision_threshold, imbalance_strategy, features_list):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO model_metrics (accuracy, f1_score, macro_f1_score, decision_threshold, imbalance_strategy, features_list)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        float(accuracy), float(f1_score), float(macro_f1_score),
        float(decision_threshold), imbalance_strategy, json.dumps(features_list)
    ))
    conn.commit()
    conn.close()

def get_model_metrics_history(db_path, limit=20):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM model_metrics ORDER BY timestamp DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def log_drift(db_path, metrics_json_str):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO drift_logs (metrics_json) VALUES (?)
    """, (metrics_json_str,))
    conn.commit()
    conn.close()

def get_latest_drift(db_path):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM drift_logs ORDER BY timestamp DESC LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None
