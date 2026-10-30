"""Configuration and path settings for the EMA Agent ecosystem."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# EMA Dashboard credentials and endpoints
EMA_DASHBOARD_URL: str = os.getenv(
    "EMA_DASHBOARD_URL",
    os.getenv("DASHBOARD_URL", "http://175.107.63.44:81/imudashboard/index.php"),
)
EMA_DASHBOARD_USER: str = os.getenv(
    "EMA_DASHBOARD_USER",
    os.getenv("DASHBOARD_USER", "dmo.mohmand"),
)
EMA_DASHBOARD_PASS: str = os.getenv(
    "EMA_DASHBOARD_PASS",
    os.getenv("DASHBOARD_PASS", "imu@mohmand"),
)

# Storage directories
_env_download_dir = os.getenv("EMA_DOWNLOAD_DIR", "./data/raw_reports")
DOWNLOAD_DIR = (
    Path(_env_download_dir).resolve()
    if Path(_env_download_dir).is_absolute()
    else (PROJECT_ROOT / _env_download_dir).resolve()
)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "school_model.pkl"
PROCESSED_CSV_PATH = DATA_PROCESSED_DIR / "processed_school_data.csv"
DB_PATH = PROJECT_ROOT / "data" / "education_gpt.db"

# Reference structure file
EMA_STRUCTURE_PATH = PROJECT_ROOT / "agents" / "ema_website_structure.txt"

# Known tehsils for Mohmand district
MOHMAND_TEHSILS = [
    "SAFI",
    "PANDIALI",
    "AMBAR",
    "KHWEZAI",
    "PRANG GHAR",
    "HALIMZAI",
    "EKKA GHUND",
]

# Report catalog mapping slugs to categories
EMA_REPORT_CATALOG = {
    "surveys": [
        "monitor-wise-survey-done",
        "monitor-wise-coverage",
        "roaster-assignment",
        "school-not-surveyed",
        "monitor-daily-visits",
        "survey-done",
        "survey-timing-report",
    ],
    "schools": [
        "closed-schools",
        "partially-occupied-schools",
        "wholly-occupied-schools",
        "building-availability",
        "basic-facilities",
        "adminisrative-visits",
        "schoolleaders-visits",
        "ptc-council",
        "single-teacher",
    ],
    "enrollment": [
        "ftb-new-application",
        "class-wise-enrollment",
        "enrollment-gap",
        "stipend",
        "new-enrollment",
    ],
    "teaching_staff": [
        "absent-teaching-staff-unauth",
        "teaching-staff-attendance",
        "vacant-posts",
        "teacher-wrong-posting",
        "teachers-appointed-by-otherorg",
    ],
    "non_teaching_staff": [
        "absent-non-teaching-staff-unauth",
        "non-teaching-staff-attendance",
        "vacant-nonteaching-staff",
    ],
    "revisits": [
        "teaching-staff-attendance-revisit",
        "absent-teaching-staff-unauth-revisit",
        "non-teaching-staff-attendance-revisit",
        "absent-nonteaching-staff-unauth-revisit",
        "school-revisit",
        "student-enrollment-revisit",
    ],
    "asc_census": [
        "furniture",
        "infrastructure",
        "sanctioned-filled-teaching-staff",
        "sancioned-filled-non-teaching-staff",
        "classwise-enrollment",
        "census-teacherInfo",
        "census-nonteacherInfo",
        "census-agewise-enrollment",
        "census-disabled-students",
        "census-ptc-info",
    ],
}
