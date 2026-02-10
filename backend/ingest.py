"""
Homeopathy Remedy Ingestion to FAISS Vector Database

This script:
1. Loads remedy chunks from remedy_chunks.json
2. Generates embeddings using Gemini embedding model
3. Builds a FAISS index (cosine similarity via inner product on normalized vectors)
4. Writes FAISS index + metadata to disk
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

# Configuration
CHUNKS_FILE = "data/remedy_chunks.json"
INDEX_NAME = "homeopathy_remedies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "data/faiss_index.bin"
FAISS_META_PATH = "data/faiss_meta.json"
FORCE_REBUILD = os.getenv("FAISS_FORCE_REBUILD", "0") == "1"

print("=" * 80)
print("HOMEOPATHY REMEDY INGESTION TO FAISS")
print("=" * 80)

# Load environment variables
load_dotenv()

# Skip regeneration if artifacts already exist (unless forced)
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_META_PATH) and not FORCE_REBUILD:
    print("\n[0] Found existing FAISS artifacts.")
    print(f"  Index: {FAISS_INDEX_PATH}")
    print(f"  Metadata: {FAISS_META_PATH}")
    print("  Set FAISS_FORCE_REBUILD=1 to regenerate embeddings.")
    raise SystemExit(0)


def embed_text(model: SentenceTransformer, text: str) -> np.ndarray:
    vec = model.encode(text, normalize_embeddings=True)
    return np.array(vec, dtype="float32")


# Step 1: Initialize embedding model
print("\n[1] Initializing local embeddings...")
try:
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print(f"DONE: Loaded model {EMBEDDING_MODEL}")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    raise SystemExit(1)

# Step 2: Load remedy chunks
print(f"\n[2] Loading remedy chunks from {CHUNKS_FILE}...")
try:
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"DONE: Loaded {len(chunks)} remedy chunks")
except Exception as e:
    print(f"ERROR: Failed to load chunks: {e}")
    raise SystemExit(1)

# Step 3: Generate embeddings
print(f"\n[3] Generating embeddings with: {EMBEDDING_MODEL}...")
vectors = []
metadata = []
try:
    for idx, chunk in enumerate(tqdm(chunks, desc="Embedding chunks")):
        vec = embed_text(embedder, chunk["text"])
        vectors.append(vec)
        metadata.append({
            "id": f"chunk_{idx}",
            "remedy_name": chunk.get("remedy_name", "Unknown"),
            "alternative_names": chunk.get("alternative_names", ""),
            "remedy_index": chunk.get("remedy_index", 0),
            "chunk_type": chunk.get("chunk_type", "flat_window"),
            "text_preview": (chunk.get("text", "")[:300] + "...") if chunk.get("text") else "",
            "full_text": chunk.get("text", "")
        })
    print(f"DONE: Generated {len(vectors)} embeddings")
except Exception as e:
    print(f"ERROR: Failed to generate embeddings: {e}")
    raise SystemExit(1)

# Step 4: Build FAISS index
print("\n[4] Building FAISS index...")
try:
    if not vectors:
        raise ValueError("No vectors were generated.")
    dim = len(vectors[0])
    index = faiss.IndexFlatIP(dim)
    index.add(np.vstack(vectors))
    print(f"DONE: FAISS index built (vectors: {index.ntotal}, dim: {index.d})")
except Exception as e:
    print(f"ERROR: Failed to build FAISS index: {e}")
    raise SystemExit(1)

# Step 5: Write index and metadata
print("\n[5] Writing FAISS index and metadata...")
try:
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"DONE: Wrote {FAISS_INDEX_PATH} and {FAISS_META_PATH}")
except Exception as e:
    print(f"ERROR: Failed to write files: {e}")
    raise SystemExit(1)

# Step 6: Verify ingestion
print(f"\n[6] Verifying ingestion...")
try:
    test_query = "headache with nausea and vomiting"
    print(f"  Test query: '{test_query}'")
    query_vec = embed_text(embedder, test_query)
    scores, indices = index.search(np.expand_dims(query_vec, axis=0), 3)

    print("\n  Top 3 Results:")
    for i, idx in enumerate(indices[0], 1):
        if idx < 0 or idx >= len(metadata):
            continue
        remedy_name = metadata[idx].get("remedy_name", "Unknown")
        similarity = float(scores[0][i - 1])
        print(f"    {i}. {remedy_name} (similarity: {similarity:.4f})")

    print("\nDONE: Verification successful!")
except Exception as e:
    print(f"ERROR: Verification failed: {e}")

# Summary
print("\n" + "=" * 80)
print("INGESTION COMPLETE")
print("=" * 80)
print(f"Index Name: {INDEX_NAME}")
print(f"Total Remedies: {len(chunks)}")
print(f"Metric: Cosine Similarity")
print(f"Index File: {FAISS_INDEX_PATH}")
print(f"Metadata File: {FAISS_META_PATH}")
print("\nYour homeopathy RAG system is ready!")
print("=" * 80)
