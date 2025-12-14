
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
# Create a startup script to handle credentials manually
# Note: 'expires' field is required by nomic library (KeyError fix)
RUN echo '#!/bin/sh\n\
mkdir -p /root/.nomic\n\
echo "{\"token\": \"$NOMIC_API_KEY\", \"tenant\": \"production\", \"expires\": 1999999999}" > /root/.nomic/credentials\n\
echo "Credentials written to /root/.nomic/credentials"\n\
exec uvicorn server:app --host 0.0.0.0 --port 10000\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
