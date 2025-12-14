
import os
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import nomic
# Move other imports here
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nomic-rag-server")

# Load environment variables
load_dotenv()

# Configuration
NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialize Nomic - MUST be done before importing embed
if NOMIC_API_KEY:
    try:
        nomic.login(NOMIC_API_KEY)
        logger.info("Nomic login successful")
    except Exception as e:
        logger.warning(f"Nomic login failed: {e}")
else:
    logger.warning("NOMIC_API_KEY not found. Nomic operations may fail.")

# Import embed after login
try:
    from nomic import embed
except ImportError:
    # Fallback or error handling if nomic isn't installed correctly
    logger.error("Failed to import nomic.embed")
    embed = None
except ValueError as e:
    # Handle the specific case where it still complains
    logger.error(f"Failed to import embed (likely auth issue): {e}")
    embed = None

# Initialize Qdrant
qdrant_client = None
if QDRANT_URL:
    try:
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=10
        )
        logger.info(f"Connected to Qdrant at {QDRANT_URL}")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")

app = FastAPI(title="Nomic RAG Server", version="1.0.0")

class SearchRequest(BaseModel):
    query: str
    collection_name: str
    limit: int = 5
    score_threshold: float = 0.0
    task_type: str = "search_query" # Nomic specific parameter

class SearchResult(BaseModel):
    id: Any
    score: float
    payload: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_embedding_sample: List[float] = Field(..., description="First 5 dimensions of embedding for verification")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "nomic_configured": bool(NOMIC_API_KEY),
        "qdrant_configured": bool(qdrant_client)
    }

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if not NOMIC_API_KEY:
        raise HTTPException(status_code=500, detail="NOMIC_API_KEY not configured")
    if not qdrant_client:
        raise HTTPException(status_code=500, detail="Qdrant client not initialized")

    try:
        # 1. Generate Embedding
        logger.info(f"Generating embedding for query: '{request.query}'")
        output = embed.text(
            texts=[request.query],
            model='nomic-embed-text-v1.5',
            task_type=request.task_type
        )
        
        if not output or 'embeddings' not in output:
            raise HTTPException(status_code=500, detail="Failed to generate embeddings")
            
        params = output.get('usage', {}) # Checking usage if needed
        query_vector = output['embeddings'][0]
        
        # 2. Search Qdrant
        logger.info(f"Searching Qdrant collection '{request.collection_name}'")
        search_result = qdrant_client.search(
            collection_name=request.collection_name,
            query_vector=query_vector,
            limit=request.limit,
            score_threshold=request.score_threshold
        )
        
        # 3. Format Response
        formatted_results = []
        for hit in search_result:
            formatted_results.append({
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload or {}
            })
            
        return {
            "results": formatted_results,
            "query_embedding_sample": query_vector[:5]
        }

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
