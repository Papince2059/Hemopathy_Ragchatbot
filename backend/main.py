"""
Homeopathy RAG System - FastAPI Backend

This backend provides semantic search and RAG capabilities
for homeopathy remedies using FAISS vector database.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import time
import logging
import os
from pathlib import Path

# Load environment variables (always from repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = REPO_ROOT / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Homeopathy RAG API",
    description="Semantic search and Q&A for homeopathy remedies",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and clients
INDEX_NAME = "homeopathy_remedies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "data/faiss_index.bin"
FAISS_META_PATH = "data/faiss_meta.json"

faiss_index = None
metadata_store = None
embedding_model = None
gemini_model = None
GEMINI_MODEL_FALLBACKS = [
    "models/gemini-flash-latest",
    "models/gemini-pro-latest",
    "models/gemini-2.0-flash",
    "models/gemini-2.5-flash",
]


# Pydantic models for request/response validation
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = []


class RemedyResult(BaseModel):
    id: str
    remedy_name: str
    alternative_names: str
    similarity: float
    text_preview: str
    full_text: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[RemedyResult]
    query_time_ms: float
    total_results: int


class ChatResponse(BaseModel):
    answer: str
    sources: List[RemedyResult]


class StatsResponse(BaseModel):
    total_remedies: int
    index_name: str
    dimension: int
    metric: str
    status: str


# Startup event - initialize models
@app.on_event("startup")
async def startup_event():
    global faiss_index, metadata_store, gemini_model, embedding_model
    
    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        logger.info("Initializing Gemini...")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found in .env, chat endpoint may fail")
        else:
            genai.configure(api_key=api_key)
            # Default model (fallbacks are attempted at request time if needed)
            gemini_model = genai.GenerativeModel(GEMINI_MODEL_FALLBACKS[0])

        logger.info(f"Loading FAISS index from: {FAISS_INDEX_PATH}...")
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)

        logger.info(f"Loading FAISS metadata from: {FAISS_META_PATH}...")
        with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
            metadata_store = json.load(f)

        if not isinstance(metadata_store, list):
            raise ValueError("FAISS metadata file must contain a JSON list")

        logger.info(f"FAISS index loaded (vectors: {faiss_index.ntotal}, dim: {faiss_index.d})")
        logger.info("DONE: Backend initialized successfully")
    except Exception as e:
        logger.error(f"ERROR: Failed to initialize backend: {e}")
        # raise  <-- Commented out to prevent crash
        logger.error("Keeping container alive for debugging/ingestion...")
        while True:
            time.sleep(60)


# Health check endpoint
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Homeopathy RAG API",
        "version": "1.0.0",
        "faiss_loaded": faiss_index is not None,
        "metadata_loaded": metadata_store is not None,
        "embedding_loaded": embedding_model is not None
    }

def embed_text(text: str) -> np.ndarray:
    vec = embedding_model.encode(text, normalize_embeddings=True)
    return np.asarray(vec, dtype="float32", order="C")

def reload_faiss_index():
    global faiss_index, metadata_store
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
        metadata_store = json.load(f)

def search_faiss(query_vector: np.ndarray, top_k: int):
    try:
        return faiss_index.search(query_vector, top_k)
    except Exception as e:
        logger.error(f"FAISS search failed (will retry after reload): {e}")
        reload_faiss_index()
        return faiss_index.search(query_vector, top_k)


# Search endpoint
@app.post("/api/search", response_model=SearchResponse)
async def search_remedies(request: SearchRequest):
    """
    Semantic search for homeopathy remedies
    
    Args:
        request: SearchRequest with query and top_k
        
    Returns:
        SearchResponse with matching remedies and metadata
    """
    try:
        start_time = time.time()
        
        # Validate input
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if request.top_k < 1 or request.top_k > 50:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")
        
        logger.info(f"Search query: '{request.query}' (top_k={request.top_k})")
        
        if faiss_index is None or metadata_store is None:
            raise HTTPException(status_code=503, detail="FAISS index not loaded. Run ingestion first.")

        # Generate query embedding
        query_text = request.query.replace("\r\n", " ").replace("\n", " ").strip()
        query_embedding = embed_text(query_text)
        query_vector = np.atleast_2d(query_embedding).astype("float32", copy=False, order="C")
        
        # Search FAISS index (cosine similarity via inner product)
        scores, indices = search_faiss(query_vector, request.top_k)
        
        # Format results
        remedy_results = []
        for rank, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(metadata_store):
                continue
            meta = metadata_store[idx]
            remedy_results.append(RemedyResult(
                id=meta.get('id', f"chunk_{idx}"),
                remedy_name=meta.get('remedy_name', 'Unknown'),
                alternative_names=meta.get('alternative_names', ''),
                similarity=float(scores[0][rank]),
                text_preview=meta.get('text_preview', ''),
                full_text=meta.get('full_text', '')
            ))
        
        query_time = (time.time() - start_time) * 1000  # Convert to ms
        
        logger.info(f"Found {len(remedy_results)} results in {query_time:.2f}ms")
        
        return SearchResponse(
            results=remedy_results,
            query_time_ms=round(query_time, 2),
            total_results=len(remedy_results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# Statistics endpoint
@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get database statistics
    
    Returns:
        StatsResponse with index information
    """
    try:
        if faiss_index is None:
            raise HTTPException(status_code=404, detail="FAISS index not found. Run ingestion first.")
        
        return StatsResponse(
            total_remedies=int(faiss_index.ntotal),
            index_name=INDEX_NAME,
            dimension=int(faiss_index.d),
            metric="cosine (inner product on normalized vectors)",
            status="active"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# Chat endpoint (RAG)
@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_remedies(request: ChatRequest):
    """
    RAG Chat endpoint using Gemini
    """
    try:
        if not gemini_model:
            raise HTTPException(status_code=503, detail="Gemini model not initialized. Check server logs/API key.")

        # 1. Search for relevant remedies
        search_request = SearchRequest(query=request.query, top_k=5)
        search_res = await search_remedies(search_request) # Re-use search logic effectively
        
        # 2. Construct Prompt context
        context_text = ""
        for idx, res in enumerate(search_res.results):
            context_text += f"Remedy {idx+1}: {res.remedy_name}\n"
            context_text += f"{res.text_preview}\n"
            if res.full_text:
                 context_text += f"Full Text Snippet: {res.full_text[:500]}...\n"
            context_text += "---\n"

        prompt = f"""
        You are a helpful Homeopathy Assistant. Use ONLY the context below.

        Task:
        1) Summarize the top matching remedies in 3-5 short bullet points.
        2) Then give a concise final answer to the user's question.

        Requirements:
        - Cite remedy names you used.
        - If the answer is not in the context, say so.

        Context:
        {context_text}

        User Question: {request.query}
        """

        # 3. Generate Answer with model fallbacks
        last_error = None
        response = None
        for model_name in GEMINI_MODEL_FALLBACKS:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                break
            except Exception as e:
                last_error = e
                continue

        if response is None:
            raise HTTPException(status_code=500, detail=f"Chat generation failed: {last_error}")
        
        return ChatResponse(
            answer=response.text,
            sources=search_res.results
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")


# Run with: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
