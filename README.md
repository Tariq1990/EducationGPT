# EducationGPT — Complete Project Documentation

EducationGPT is an enterprise-grade analytics dashboard, machine learning API, and AI-grounded planning advisor for school resource management in Khyber Pakhtunkhwa (KP), Pakistan. 

The system automates the scraping and downloading of monitoring reports, processes census datasets, builds local retrieval index structures (RAG), runs a Random Forest classifier to flag critical school shortages, and hosts both a REST API and a Streamlit user interface.

---

## 📂 Project Architecture & Directory Layout

```
EducationGPT/
├── agents/
│   └── ema_agent/
│       ├── __init__.py           # Package declarations
│       ├── dashboard_scraper.py   # Playwright-based browser automation (headless login & download)
│       ├── report_processor.py   # Pandas-based clean/parse and metric engine
│       └── ema_tool.py           # Core agent tool integration wrappers
├── data/
│   ├── raw/                      # Raw source census spreadsheets
│   ├── processed/                # Merged CSVs and RAG indexes
│   ├── raw_reports/              # Scraper downloads directory
│   └── reports/                  # Archive for downloaded dashboards
├── models/
│   └── school_model.pkl          # Trained Random Forest classifier bundle
├── src/
│   ├── api_server.py             # FastAPI REST service & routes
│   ├── data_pipeline.py          # Data ingestion and consolidation
│   ├── train_model.py            # ML modeling and tuning
│   ├── build_rag_index.py        # Local vector index builder
│   ├── db_manager.py             # SQLite predictions auditor
│   ├── local_rag.py              # Term frequency advisor retriever
│   ├── monitoring.py             # Feature drift auditor
│   └── interpretability.py       # Local feature contribution calculator
├── ui/
│   └── web_app.py                # Streamlit analytics & chat frontend
├── tests/
│   └── test_ema_agent.py         # Test suite for the EMA agent
├── .env                          # Local environment secrets and paths
├── requirements.txt              # Project dependencies
└── Dockerfile                    # Container configuration file
```

---

## 🔧 Prerequisites & Setup

### 1. Python Environment
Make sure you have Python 3.10+ installed.

### 2. Install Project Dependencies
In your terminal, navigate to the project root and run:
```bash
pip install -r requirements.txt
```

### 3. Install Playwright Web Drivers
Initialize Chromium for the Playwright scraper module:
```bash
playwright install chromium
```

### 4. Environment Configuration
Create or edit the `.env` file in the project root:
```ini
# Cloud LLM Key (Optional, for Cloud AI Advisor Chat)
GEMINI_API_KEY=AIzaSyALSl0ibut_mwBJFZmHwCdZeixZmUy9CIk

# Local LLM config (Ollama)
USE_LOCAL_LLM=false
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.2:1b
OLLAMA_MODELS=llama3.2:1b,llama3:8b,qwen2.5:7b

# Data and Model Paths
PROCESSED_CSV_PATH=data/processed/processed_school_data.csv
MODEL_PATH=models/school_model.pkl
DATA_DIR=data/raw
RAG_INDEX_PATH=data/processed/rag_chunks.json
PREDICT_API_URL=http://localhost:8000/predict

# EMA Dashboard Agent Parameters
EMA_DASHBOARD_URL=http://175.107.63.44:81/imudashboard/index.php
EMA_DASHBOARD_USER=dmo.mohmand
EMA_DASHBOARD_PASS=imu@mohmand
EMA_DOWNLOAD_DIR=./data/raw_reports
```

---

## 🏃 How to Run the Project

Follow these steps in sequence to prep the data, train the model, and launch the apps.

### Step 1: Ingest & Process Census Data
Processes the raw spreadsheets in `data/raw/` and creates the consolidated dataset:
```bash
python src/data_pipeline.py
```

### Step 2: Train the ML Model
Trains a Random Forest classifier with class-weight tuning to output `models/school_model.pkl`:
```bash
python src/train_model.py
```

### Step 3: Build the Local RAG Index
Builds TF-IDF vector representations of the dataset for the AI Chat Advisor:
```bash
python src/build_rag_index.py
```

### Step 4: Run the API Server
Starts the FastAPI REST server on port `8000`:
```bash
uvicorn src.api_server:app --port 8000
```

### Step 5: Start the Analytics Dashboard
Launches the Streamlit frontend in a separate terminal:
```bash
streamlit run ui/web_app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🤖 The EMA Dashboard Automation Agent

The EMA Agent is an autonomous scraping and analytical engine designed to log into the internal dashboard, retrieve records, and process the metrics.

### 1. Components
*   **`dashboard_scraper.py`**: Utilizes Playwright to navigate, login, and capture downloads. Includes automatic retry mechanisms (max 3) and page content fallbacks if downloads fail to trigger.
*   **`report_processor.py`**: Takes raw CSV or Excel downloads, handles encoding variations, normalizes features, maps alternative column labels (like `School Name` → `school_name`), and builds analytical data structures.
*   **`ema_tool.py`**: Integrates the scraper and processor into three primary tool APIs.

### 2. Programmatic Usage in Python
```python
from agents.ema_agent.ema_tool import (
    download_latest_ema_report,
    get_ema_metrics_summary,
    query_dashboard_data
)

# 1. Trigger Playwright login and download reports
download_result = download_latest_ema_report()
print(download_result)
# Output: {'status': 'ok', 'files': ['data/raw_reports/rep_2026.csv'], 'count': 1}

# 2. Get processed metrics & text context for LLM prompt injects
summary = get_ema_metrics_summary()
print(summary["context"])
# Outputs formatted summary: Total enrollment, attendance rates, teacher counts, etc.

# 3. Query processed dashboard reports
results = query_dashboard_data(query="Primary", district="Peshawar")
print(results["records"][:3])
```

---

## 🔌 API Endpoints Reference

Once `api_server.py` is running, the following endpoints are available:

### 1. Prediction & Model Audits
*   **`POST /predict`**: Submit school characteristics (enrollment, class rooms, teachers, funding) to get resource shortage predictions, confidence scores, and local feature attributions.
*   **`POST /predict/feedback`**: Log feedback (`Correct` / `Incorrect`) for audit history.
*   **`GET /monitoring/drift`**: Returns feature drift analysis compared to baseline.
*   **`GET /monitoring/stats`**: Returns model accuracy stats from audit history.

### 2. EMA Dashboard Agent
*   **`POST /ema/download`**: Programmatically triggers Playwright to log in and download the latest reports from the configured dashboard URL.
*   **`GET /ema/metrics`**: Returns computed summaries (averages, sums, and column statistics) from the latest downloaded report.
*   **`GET /ema/query`**: Queries parsed dashboard records.
    *   *Parameters*: `query` (text search filter), `district` (partial district name), `top_n` (number of rows).

---

## 🧪 Testing the Agent

Verify your agent's functionality using the pre-configured test suite.

```bash
python -m pytest tests/test_ema_agent.py -v --tb=short
```

This verifies:
1. Headless browser initialization.
2. Excel (`.xlsx`) and Comma-separated (`.csv`) file parsers.
3. Clean pipelines, column mapping, and metric calculations.
4. Tool wraps and lookup filters.

---

## 🐳 Docker Deployment

To build and run the entire application within a single isolated container (FastAPI + Streamlit):

```bash
# Build the Docker image
docker build -t educationgpt .

# Run the container (maps API to 8000 and Dashboard to 8501)
docker run -p 8000:8000 -p 8501:8501 --env-file .env educationgpt
```
