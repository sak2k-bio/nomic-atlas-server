
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if needed (usually minimal for this stack)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .
COPY .env.example .env

# Expose port
EXPOSE 10000

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "10000"]
