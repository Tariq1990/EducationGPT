# Deployment & Operations Guide

## Local Deployment

1.  **Environment Setup**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Data Preparation & Training**:
    ```bash
    python src/data_pipeline.py
    python src/train_model.py
    ```
3.  **Run Backend (FastAPI)**:
    ```bash
    uvicorn src.api_server:app --port 8000
    ```
4.  **Run Frontend (Streamlit)**:
    ```bash
    streamlit run ui/web_app.py
    ```

---

## Environment Configuration

The `.env` file controls runtime paths and keys:
1. `DEEPSEEK_API_KEY` (optional for LLM chat)
2. `PROCESSED_CSV_PATH` (defaults to `data/processed/processed_school_data.csv`)
3. `PREDICT_API_URL` (defaults to `http://localhost:8000/predict`)
4. `MODEL_PATH` (defaults to `models/school_model.pkl`)
5. `DATA_DIR` (defaults to `data/raw`)
6. `USE_LOCAL_LLM` (set `true` to use Ollama locally)
7. `OLLAMA_MODEL` (default local model)
8. `OLLAMA_MODELS` (comma-separated local model list)

---

## Docker Deployment

1. Build the image:
   ```bash
   docker build -t educationgpt .
   ```
2. Run:
   ```bash
   docker run -p 8000:8000 -p 8501:8501 --env-file .env educationgpt
   ```

---

## Mobile App Strategy

To expose this ML model to a mobile app (Flutter or React Native), follow this architecture:

### 1. Architecture
*   **Backend**: The existing **FastAPI** server acts as the central hub.
*   **Mobile Frontend**: 
    *   **Flutter**: Use the `http` package to send POST requests to the `/predict` endpoint.
    *   **React Native**: Use `axios` or `fetch` for API communication.
*   **Authentication**: Add **JWT (JSON Web Tokens)** to the FastAPI server to secure mobile access.

### 2. Implementation Steps
*   **API Wrapper**: The current `src/api_server.py` is already structured to serve mobile clients.
*   **Cross-Origin Resource Sharing (CORS)**: If using a web-based mobile framework, ensure FastAPI has CORS configured:
    ```python
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
    ```

### 3. Cloud Deployment
*   **Containerization**: Use the included `Dockerfile`.
*   **Hosting**: Deploy the container to **Google Cloud Run**, **AWS App Runner**, or **Azure Container Instances**.
*   **Database**: Transition from local CSVs to a cloud database like **PostgreSQL** for real-time data storage and historic tracking.
