# EducationGPT Core - Google Cloud Deployment Guide

This package provides the core ADK agent for EducationGPT, fully configured with Playwright headless browser scraping and data processing tools.

## Prerequisites

1. **Google Cloud SDK (`gcloud`)** installed and authenticated:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```
2. **Google ADK CLI** (available in your python virtual environment):
   ```bash
   pip install google-adk
   ```

---

## Deployment Option 1: Direct Cloud Run via ADK CLI (Recommended)

Google ADK provides a single-command deployment to Google Cloud Run:

```bash
adk deploy cloud_run \
    --project YOUR_PROJECT_ID \
    --region us-central1 \
    --service_name education-gpt-core \
    --env GOOGLE_API_KEY=YOUR_GEMINI_API_KEY \
    .
```

---

## Deployment Option 2: Vertex AI Agent Engine

Deploy to Vertex AI Agent Engine using ADK:

```bash
adk deploy agent_engine \
    --project YOUR_PROJECT_ID \
    --region us-central1 \
    --display_name "education-gpt-core" \
    .
```

---

## Deployment Option 3: Container Build & Cloud Run Deploy

Build and deploy using Google Cloud Build and Cloud Run:

```bash
# 1. Build container image using Cloud Build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/education-gpt-core:latest .

# 2. Deploy container to Cloud Run
gcloud run deploy education-gpt-core \
    --image gcr.io/YOUR_PROJECT_ID/education-gpt-core:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --set-env-vars GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

---

## Local Testing & Verification

1. **Verify agent initialization and tool schemas:**
   ```bash
   python -c "from agent import root_agent, SchoolAnalyticsReport; print(root_agent.name, root_agent.tools)"
   ```

2. **Run interactive ADK session locally:**
   ```bash
   adk run .
   ```

3. **Start local ADK Web UI:**
   ```bash
   adk web . --port 8000
   ```
