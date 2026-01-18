FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY experiments/ ./experiments/
COPY data/ ./data/

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "experiments/run_corrected_analysis.py"]
