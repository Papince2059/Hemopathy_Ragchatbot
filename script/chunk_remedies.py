"""
Homeopathy Remedy Chunker - Flat Strategy (Reverted)

This script chunks the Boericke Materia Medica using a simple sliding window approach.
Strategy:
1. Extract full text of each remedy.
2. Split into chunks of ~1500 chars with ~400 char overlap.
3. Prepend [REMEDY NAME] to every chunk for context.
4. No section splitting.
"""

import re
import json
import os
from typing import List, Dict, Tuple

class RemedyChunker:
    def __init__(self, file_path: str, chunk_size: int = 1500, overlap: int = 400):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.overlap = overlap

    def is_remedy_header(self, line: str) -> bool:
        """Detect if a line is a remedy header (all caps)."""
        line = line.strip()
        if not line: return False
        # Matches all caps, allowing for hyphens, commas, ampersands and parentheses
        if line.isupper() and re.match(r'^[A-Z\s\-\(\),\&]+$', line):
            if len(line) >= 4:
                return True
        return False

    def extract_remedies(self) -> List[Dict]:
        """Parses the text file and groups lines by remedy."""
        if not os.path.exists(self.file_path):
            print(f"Error: {self.file_path} not found.")
            return []

        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        remedies = []
        current_name = None
        current_lines = []
        
        for line in lines:
            if self.is_remedy_header(line):
                if current_name and current_lines:
                    remedies.append({
                        "name": current_name,
                        "text": "".join(current_lines)
                    })
                current_name = line.strip()
                current_lines = [line]
            else:
                if current_name:
                    current_lines.append(line)
                    
        # Add last one
        if current_name and current_lines:
            remedies.append({
                "name": current_name,
                "text": "".join(current_lines)
            })
            
        return remedies

    def create_chunks(self, remedies: List[Dict]) -> List[Dict]:
        """Splits remedy texts into overlapping chunks."""
        chunks = []
        
        for idx, remedy in enumerate(remedies):
            text = remedy['text']
            name = remedy['name']
            
            # Simple manual sliding window
            if len(text) <= self.chunk_size:
                chunk_text = text
                chunks.append(self._make_chunk(idx, name, chunk_text, 0, 1))
            else:
                start = 0
                sub_idx = 0
                stride = self.chunk_size - self.overlap
                
                # Estimate total sub chunks
                total_subs = (len(text) // stride) + 1
                
                while start < len(text):
                    end = min(start + self.chunk_size, len(text))
                    chunk_slice = text[start:end]
                    
                    # Prepend context if it's not the very first chunk (which usually has the header)
                    # Actually, better to ALWAYS prepend [REMEDY NAME] so embedding model knows context
                    # strict flat strategy:
                    display_text = f"[{name}] {chunk_slice}"
                    
                    chunks.append(self._make_chunk(idx, name, display_text, sub_idx, total_subs))
                    
                    if end == len(text):
                        break
                        
                    start += stride
                    sub_idx += 1
                    
        return chunks

    def _make_chunk(self, r_idx, name, text, s_idx, total) -> Dict:
        return {
            "remedy_index": r_idx,
            "remedy_name": name,
            "text": text,
            "metadata": {
                "source": "boericke",
                "remedy": name
            },
            "char_count": len(text),
            "chunk_type": "flat_window",
            "sub_chunk_index": s_idx,
            "total_sub_chunks": total
        }

    def save(self, chunks: List[Dict], path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(chunks)} chunks to {path}")

def main():
    project_root = os.getcwd()
    INPUT_FILE = os.path.join(project_root, "data", "boericke_full_text.txt")
    OUTPUT_FILE = os.path.join(project_root, "data", "remedy_chunks.json")
    
    # Flat strategy configuration
    chunker = RemedyChunker(INPUT_FILE, chunk_size=1500, overlap=400)
    
    print("Extracting remedies...")
    remedies = chunker.extract_remedies()
    print(f"Found {len(remedies)} remedies.")
    
    print("Generating flat chunks...")
    chunks = chunker.create_chunks(remedies)
    
    chunker.save(chunks, OUTPUT_FILE)

if __name__ == "__main__":
    main()
