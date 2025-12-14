
import os
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from dotenv import load_dotenv
# NOTE: Removing top-level nomic imports to prevent startup crashes
# import nomic
# from nomic import embed
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

# Explicitly set env var for Nomic library to find
if NOMIC_API_KEY:
    os.environ["NOMIC_API_KEY"] = NOMIC_API_KEY

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

@app.get("/")
async def root():
    return {"message": "Nomic RAG Server is running", "endpoints": ["/health", "/search", "/embed", "/debug"]}

class SearchRequest(BaseModel):
    query: str
    collection_name: str
    limit: int = 5
    score_threshold: float = 0.0
    task_type: str = "search_query"

class SearchResult(BaseModel):
    id: Any
    score: float
    payload: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_embedding_sample: List[float] = Field(..., description="First 5 dimensions of embedding for verification")

class EmbedRequest(BaseModel):
    texts: List[str]
    task_type: str = "search_document"

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "nomic_key_present": bool(NOMIC_API_KEY),
        "qdrant_configured": bool(qdrant_client)
    }

@app.get("/debug")
async def debug_info():
    """Endpoint to diagnose environment and import issues"""
    import_status = "Not attempted"
    login_status = "Skipped"
    
    try:
        import nomic
        from nomic import embed
        import_status = "Success"
        
        try:
            # check if we can actually login
            if NOMIC_API_KEY:
                nomic.login(NOMIC_API_KEY)
                login_status = "Success"
            else:
                login_status = "No Key provided"
        except Exception as e:
            login_status = f"Failed: {str(e)}"
            
    except ImportError as e:
        import_status = f"ImportError: {str(e)}"
    except ValueError as e:
        import_status = f"ValueError: {str(e)}"
    except Exception as e:
        import_status = f"Error: {str(e)}"

    return {
        "env_vars": {
            "NOMIC_API_KEY": "Present" if NOMIC_API_KEY else "Missing",
            "QDRANT_URL": QDRANT_URL
        },
        "python_env_nomic_key": "Present" if os.environ.get("NOMIC_API_KEY") else "Missing",
        "import_status": import_status,
        "login_status": login_status
    }

@app.post("/embed", response_model=EmbedResponse)
async def generate_embeddings(request: EmbedRequest):
    if not NOMIC_API_KEY:
        raise HTTPException(status_code=500, detail="NOMIC_API_KEY not configured")
        
    try:
        # Lazy import - Login MUST happen before importing embed in some versions/contexts
        import nomic
        # Ensure login
        nomic.login(NOMIC_API_KEY)
        
        from nomic import embed
        
        logger.info(f"Generating embeddings for {len(request.texts)} texts")
        output = embed.text(
            texts=request.texts,
            model='nomic-embed-text-v1.5',
            task_type=request.task_type
        )
        
        if not output or 'embeddings' not in output:
             raise HTTPException(status_code=500, detail="Failed to generate embeddings")
             
        return {"embeddings": output['embeddings']}
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        # Return full traceback in detail for debugging
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)} | {traceback.format_exc()}")


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if not NOMIC_API_KEY:
        raise HTTPException(status_code=500, detail="NOMIC_API_KEY not configured")
    if not qdrant_client:
        raise HTTPException(status_code=500, detail="Qdrant client not initialized")

    try:
        # Lazy import - Login first
        import nomic
        nomic.login(NOMIC_API_KEY)
        
        from nomic import embed

        # 1. Generate Embedding
        logger.info(f"Generating embedding for query: '{request.query}'")
        output = embed.text(
            texts=[request.query],
            model='nomic-embed-text-v1.5',
            task_type=request.task_type
        )
        
        if not output or 'embeddings' not in output:
            raise HTTPException(status_code=500, detail="Failed to generate embeddings")
            
        params = output.get('usage', {}) 
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
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)} | {traceback.format_exc()}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
