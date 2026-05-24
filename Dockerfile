FROM python:3.11-slim

WORKDIR /app

# Install only minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY frontend/ ./frontend/

# Create data directories
RUN mkdir -p data/docs data/faiss_index

EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
