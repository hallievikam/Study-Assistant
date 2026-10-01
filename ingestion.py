"""
ingestion.py - Ingestion pipeline

Parses PDFs / .txt / .md, splits into overlapping chunks (~500 tokens,
approximated by words), and tags each chunk with source filename + page.
"""

import uuid
from pathlib import Path
from typing import List, Dict, Optional

import pdfplumber

import db
import vectorstore

CHUNK_SIZE_WORDS = 400   # ~500 tokens
CHUNK_OVERLAP_WORDS = 60


def _chunk_words(words: List[str], size: int, overlap: int) -> List[str]:
    """Split a list of words into overlapping chunks, joined back to text."""
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if not piece:
            continue
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def ingest_pdf(file_path: str) -> int:
    """Extract text per page, chunk each page, store chunks with page numbers."""
    filename = Path(file_path).name
    doc_id = db.add_document(filename, "pdf")

    chunk_index = 0
    records = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            words = text.split()
            for piece in _chunk_words(words, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS):
                chunk_id = str(uuid.uuid4())
                db.add_chunk(chunk_id, doc_id, page_num, chunk_index, piece)
                records.append({"chunk_id": chunk_id, "text": piece, "document_id": doc_id,
                                 "filename": filename, "page": page_num})
                chunk_index += 1

    vectorstore.embed_and_store_chunks_batch(records)
    return doc_id


def ingest_text_file(file_path: str) -> int:
    """Ingest a .txt / .md file (no page numbers)."""
    filename = Path(file_path).name
    filetype = Path(file_path).suffix.lstrip(".") or "txt"
    doc_id = db.add_document(filename, filetype)

    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    words = text.split()
    records = []
    for chunk_index, piece in enumerate(_chunk_words(words, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)):
        chunk_id = str(uuid.uuid4())
        db.add_chunk(chunk_id, doc_id, None, chunk_index, piece)
        records.append({"chunk_id": chunk_id, "text": piece, "document_id": doc_id,
                         "filename": filename, "page": None})

    vectorstore.embed_and_store_chunks_batch(records)
    return doc_id


def ingest_raw_text(text: str, source_name: str = "pasted-notes") -> int:
    """Ingest pasted lecture transcript / notes text directly (no file on disk)."""
    doc_id = db.add_document(source_name, "pasted")
    words = text.split()
    records = []
    for chunk_index, piece in enumerate(_chunk_words(words, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)):
        chunk_id = str(uuid.uuid4())
        db.add_chunk(chunk_id, doc_id, None, chunk_index, piece)
        records.append({"chunk_id": chunk_id, "text": piece, "document_id": doc_id,
                         "filename": source_name, "page": None})
    vectorstore.embed_and_store_chunks_batch(records)
    return doc_id


def ingest_file(file_path: str) -> int:
    """Dispatch based on file extension."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return ingest_pdf(file_path)
    elif suffix in (".txt", ".md"):
        return ingest_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
