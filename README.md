# EducationGPT

Professional school resource analytics and AI advisory dashboard with a FastAPI backend and Streamlit UI.

**Structure**
1. `src/` Application backend, training, and data pipeline scripts
2. `ui/` Streamlit dashboard and chat UI
3. `data/raw/` Raw source spreadsheets
4. `data/processed/` Processed CSVs for training and analytics
5. `models/` Trained model artifacts
6. `config/` Configuration files (reserved)

**Quick Start**
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Build the dataset:
   - `python src/data_pipeline.py`
3. Train the model:
   - `python src/train_model.py`
4. Build local RAG index:
   - `python src/build_rag_index.py`
5. Run the API:
   - `uvicorn src.api_server:app --port 8000`
6. Run the dashboard:
   - `streamlit run ui/web_app.py`

**Environment**
Edit `.env` as needed:
- `GEMINI_API_KEY`
- `PROCESSED_CSV_PATH`
- `PREDICT_API_URL`
- `MODEL_PATH`
- `DATA_DIR`
- `USE_LOCAL_LLM` (set `true` to use Ollama)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434/v1`)
- `OLLAMA_MODEL`
- `OLLAMA_MODELS`
- `LOCAL_RAG_TOP_K` (retrieved chunks for local grounded chat)
- `LOCAL_RAG_MIN_SCORE` (minimum retrieval similarity; higher = stricter refusal)
- `RAG_INDEX_PATH` (local JSON chunk index used for grounded Ollama responses)

