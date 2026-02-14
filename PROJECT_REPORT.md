<!-- cspell:ignore Streamlit uvicorn -->
# EducationGPT Project Report

**Date:** 2026-02-09  
**Owner:** Engr. T. Jamal  
**Project:** EducationGPT (School Resource Analytics + AI Advisor)

## 1. Executive Summary
EducationGPT is a professional dashboard and AI advisory system for school resource planning. It combines a FastAPI prediction API, a Streamlit analytics dashboard, and an optional LLM chat advisor. The system ingests multi-source school census and infrastructure data, produces a consolidated dataset, and trains a predictive model to flag potential critical resource shortages.

## 2. Objectives
1. Provide a data-driven dashboard for resource planning across districts, levels, and gender.
2. Predict critical shortages using structured school features.
3. Offer an AI advisor chat that can reason over the dataset and recent predictions.
4. Deliver a portable, production-ready project layout aligned with research standards.

## 3. Architecture Overview
**Runtime Layers**
1. **Data Pipeline:** `src/data_pipeline.py`  
2. **Model Training:** `src/train_model.py`  
3. **API Service:** `src/api_server.py`  
4. **Dashboard & Chat UI:** `ui/web_app.py`

**Artifacts**
1. Processed dataset: `data/processed/processed_school_data.csv`
2. Model package: `models/school_model.pkl`

## 4. Data Sources
Raw inputs are stored in `data/raw/` and include:
1. Infrastructure
2. Furniture
3. Teacher and Non-Teacher census
4. PTC funding data

The pipeline merges these sources, creates a derived target label (`is_critical`), and outputs a clean dataset for training and analytics.

## 5. Modeling
**Model Type:** RandomForestClassifier  
**Handling Imbalance:**  
1. Class-weighted training  
2. Threshold tuning for macro-F1 on validation  

The model bundle includes the encoders, imputer, and a calibrated decision threshold.

## 6. Dashboard & Analytics
**Key Capabilities**
1. Filtering by District, Level, and Gender
2. KPI summaries for critical risk, staffing, and funding
3. Visuals by district and gender
4. Prediction form with API integration

**AI Advisor (Chat)**
1. Multi-turn chat with history
2. Optional DeepSeek integration
3. Contextual grounding based on dataset and last prediction

## 7. Deployment
See `DEPLOYMENT.md` for detailed instructions.

**Quick Start**
1. `python src/data_pipeline.py`
2. `python src/train_model.py`
3. `uvicorn src.api_server:app --port 8000`
4. `streamlit run ui/web_app.py`

## 8. Risks & Limitations
1. Class imbalance can reduce minority-class precision.
2. Prediction quality depends on data completeness and reporting quality.
3. The LLM advisor relies on external API availability and usage limits.

## 9. Next Steps
1. Add SMOTE or alternative imbalance strategies.
2. Add interoperability (feature importance and attribution summaries).
3. Add model monitoring and drift detection.
4. Integrate a lightweight DB for audit and analytics history.
