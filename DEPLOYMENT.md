# Deployment & Mobile Strategy Guide

## Local Deployment

1.  **Environment Setup**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Data Preparation & Training**:
    ```bash
    python data_pipeline.py
    python train_model.py
    ```
3.  **Run Backend (FastAPI)**:
    ```bash
    uvicorn main:app --port 8000
    ```
4.  **Run Frontend (Streamlit)**:
    ```bash
    streamlit run dashboard.py
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
*   **API Wrapper**: The current `main.py` is already structured to serve mobile clients.
*   **Cross-Origin Resource Sharing (CORS)**: If using a web-based mobile framework, ensure FastAPI has CORS configured:
    ```python
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
    ```

### 3. Cloud Deployment
*   **Containerization**: Create a `Dockerfile` for the FastAPI backend.
*   **Hosting**: Deploy the container to **Google Cloud Run**, **AWS App Runner**, or **Azure Container Instances**.
*   **Database**: Transition from local CSVs to a cloud database like **PostgreSQL** for real-time data storage and historic tracking.
