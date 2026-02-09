"""
Compatibility wrapper.

This project now uses FAISS + Gemini embeddings.
Run the FAISS ingestion script from backend/ingest.py.
"""

import runpy

if __name__ == "__main__":
    runpy.run_path("backend/ingest.py", run_name="__main__")
