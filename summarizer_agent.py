"""
summarizer_agent.py - Summarizer Agent

Generates chapter/topic-level summaries or "cheat sheets" built from
retrieved chunks (not the whole raw document), keeping it grounded and
reusing the same retrieval path as the Q&A agent.
"""

import db
import vectorstore
import llm

SYSTEM_PROMPT = """You are a study assistant that writes clear, well-organized
summaries ("cheat sheets") from the given source material. Rules:
- Use only the provided context. Do not invent facts not present in it.
- Structure the summary with short headings and bullet points.
- Keep it dense but scannable - this is for exam revision.
"""


def summarize_document(document_id: int, topic: str = None) -> str:
    """
    Summarize a whole document by pulling all its chunks (in order) from
    SQLite, or - if `topic` is given - retrieve only the chunks relevant
    to that topic from the vector store (per-topic cheat sheet).
    """
    if topic:
        hits = vectorstore.query(topic, top_k=10, document_id=document_id)
        text_blocks = [h["text"] for h in hits]
        heading = f"Topic: {topic}"
    else:
        conn = db.get_conn()
        rows = conn.execute(
            "SELECT text FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC",
            (document_id,),
        ).fetchall()
        conn.close()
        text_blocks = [r["text"] for r in rows]
        heading = "Full document"

    if not text_blocks:
        return "No content found to summarize. Make sure the document was ingested."

    context = "\n\n".join(text_blocks)
    user_prompt = f"{heading}\n\nSource material:\n\n{context}\n\nWrite a study summary."
    return llm.complete(SYSTEM_PROMPT, user_prompt, max_tokens=1200)
