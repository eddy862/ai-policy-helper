import os, re, hashlib
from typing import List, Dict, Tuple
from .settings import settings
import logging

logger = logging.getLogger(__name__)

def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _md_sections(text: str) -> List[Tuple[str, str]]:
    # Very simple section splitter by Markdown headings
    parts = re.split(r"\n(?=#+\s)", text)
    out = []
    for p in parts:
        p = p.strip()

        if not p:
            continue
        lines = p.splitlines()
        title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else "Body"
        
        p = p.lstrip("# ").strip()
        
        # If the title is repeated as a heading in the body, we skip it to avoid redundancy
        if p == title:
            continue
        
        # remove the title from p
        if p.startswith(title):
            p = p[len(title):].strip()
        
        # I dont want like **bulky items**,  remove the ** or any markdown formatting for better chunking and retrieval
        p = re.sub(r"[*_~`]+", "", p)

        out.append((title, p))
    return out or [("Body", text)]

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(tokens): break
        i += chunk_size - overlap
    return chunks

def load_documents(data_dir: str) -> List[Dict]:
    docs = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith((".md", ".txt")):
            continue
        path = os.path.join(data_dir, fname)
        text = _read_text_file(path)
        for section, body in _md_sections(text):
            docs.append({
                "title": fname,
                "section": section,
                "text": body
            })
    logger.info("Docs loaded: count=%d sample=%s", len(docs), docs[:2])
    return docs

def doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
