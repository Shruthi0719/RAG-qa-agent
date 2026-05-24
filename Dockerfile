FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for unstructured
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
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

# Railway injects $PORT; fall back to 8000 for local / docker-compose
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
