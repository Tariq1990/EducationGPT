# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports for Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Script to run both backend and frontend
RUN echo '#!/bin/bash\nuvicorn src.api_server:app --host 0.0.0.0 --port 8000 & \nstreamlit run ui/web_app.py --server.port 8501 --server.address 0.0.0.0' > run.sh
RUN chmod +x run.sh

# Run the startup script
CMD ["./run.sh"]
