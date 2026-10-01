"""
vectorstore.py - Embedding + vector store

Uses sentence-transformers (local, free) to embed chunks and Chroma
(local, persistent) to store + search them. Metadata (document_id, page,
filename) travels alongside each vector so retrieval results can be cited.
"""

from pathlib import Path
from typing import List, Dict

import chromadb

PERSIST_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "study_chunks"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for this use case

_model = None
_client = None
_collection = None


def _get_model():
    # Imported lazily so this module can be imported (e.g. for tests) without
    # requiring sentence-transformers/torch to be installed.
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def _get_collection():
    global _client, _collection
    if _collection is None:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(PERSIST_DIR))
        _collection = _client.get_or_create_collection(COLLECTION_NAME)
    return _collection


def embed_and_store_chunk(chunk_id: str, text: str, document_id: int, filename: str, page):
    """Embed a single chunk and upsert it into Chroma with metadata."""
    model = _get_model()
    embedding = model.encode(text).tolist()
    collection = _get_collection()
    collection.upsert(
        ids=[chunk_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            "document_id": document_id,
            "filename": filename,
            "page": page if page is not None else -1,
        }],
    )


def embed_and_store_chunks_batch(chunk_records: List[Dict]):
    """
    chunk_records: list of dicts with keys:
        chunk_id, text, document_id, filename, page
    Batching is much faster than one-by-one for larger documents.
    """
    if not chunk_records:
        return
    model = _get_model()
    texts = [r["text"] for r in chunk_records]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()
    collection = _get_collection()
    collection.upsert(
        ids=[r["chunk_id"] for r in chunk_records],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "document_id": r["document_id"],
                "filename": r["filename"],
                "page": r["page"] if r["page"] is not None else -1,
            }
            for r in chunk_records
        ],
    )


def query(text: str, top_k: int = 5, document_id: int = None) -> List[Dict]:
    """Retrieve top-k relevant chunks for a query, optionally scoped to one document."""
    model = _get_model()
    query_embedding = model.encode(text).tolist()
    collection = _get_collection()

    where = {"document_id": document_id} if document_id is not None else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )

    hits = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for i in range(len(ids)):
        hits.append({
            "chunk_id": ids[i],
            "text": docs[i],
            "filename": metas[i].get("filename"),
            "page": metas[i].get("page"),
            "distance": dists[i],
        })
    return hits
