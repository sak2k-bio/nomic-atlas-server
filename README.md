
# Nomic RAG Server

A standalone FastAPI service that generates embeddings using **Nomic** and searches a **Qdrant Cloud** database.

## Prerequisites

- Docker
- Nomic API Key ([Get one here](https://atlas.nomic.ai))
- Qdrant Cloud URL and API Key

## Configuration

Create a `.env` file in this directory (or pass environment variables to Docker):

```env
NOMIC_API_KEY=nk-your-key-here
QDRANT_URL=https://your-cluster.qdrant.tech
QDRANT_API_KEY=your-qdrant-key
```

## Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run server:
   ```bash
   python server.py
   ```

## Deploying on VPS with Docker

1. **Build the image**:
   ```bash
   docker build -t nomic-rag-server .
   ```

2. **Run the container**:
   ```bash
   docker run -d \
     --name nomic-server \
     -p 10000:10000 \
     --restart unless-stopped \
     -e NOMIC_API_KEY="nk-..." \
     -e QDRANT_URL="https://..." \
     -e QDRANT_API_KEY="..." \
     nomic-rag-server
   ```

## API Usage

### Search Endpoint `POST /search`

**Payload:**
```json
{
  "query": "symptoms of diabetes",
  "collection_name": "medical_guidelines",
  "limit": 5
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "doc_123",
      "score": 0.89,
      "payload": {
        "text": "..."
      }
    }
  ],
  "query_embedding_sample": [0.012, -0.045, ...]
}
```
