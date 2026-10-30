# ==============================================================================
# EducationGPT Core - Google Cloud Run Container Specification
# ==============================================================================
FROM python:3.11-slim

# Prevent Python buffering and writing bytecode
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# Install system dependencies required for headless Chromium (Playwright)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    ca-certificates \
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    playwright install chromium

# Copy application files
COPY . .

# Launch ADK API server exposing Cloud Run port
CMD ["python", "-m", "google.adk.cli", "api_server", ".", "--port", "8080", "--host", "0.0.0.0"]
