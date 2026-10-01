"""
qa_agent.py - Q&A Agent (the RAG core)

Embeds the query, retrieves top-k relevant chunks, and asks the LLM to
answer using ONLY those chunks, with citations back to source/page.
"""

import vectorstore
import llm

SYSTEM_PROMPT = """You are a study assistant answering questions using ONLY the
provided context chunks from the user's own notes/PDFs. Rules:
- Answer only using information present in the context below. If the context
  doesn't contain the answer, say so clearly instead of guessing.
- After each claim, cite the source in square brackets like [source.pdf, p.3].
  If a chunk has no page number, cite just [source.pdf].
- Keep the answer concise and directly useful for studying.
"""


def _format_context(hits) -> str:
    parts = []
    for h in hits:
        loc = f"{h['filename']}" + (f", p.{h['page']}" if h.get("page") not in (None, -1) else "")
        parts.append(f"[{loc}]\n{h['text']}")
    return "\n\n---\n\n".join(parts)


def answer_question(question: str, document_id: int = None, top_k: int = 5) -> dict:
    hits = vectorstore.query(question, top_k=top_k, document_id=document_id)

    if not hits:
        return {
            "answer": "I couldn't find any relevant content in your uploaded materials. "
                      "Try uploading a document first or rephrasing the question.",
            "sources": [],
        }

    context = _format_context(hits)
    user_prompt = f"Context:\n\n{context}\n\nQuestion: {question}"
    answer = llm.complete(SYSTEM_PROMPT, user_prompt, max_tokens=800)

    sources = [
        {"filename": h["filename"], "page": h["page"] if h.get("page") != -1 else None}
        for h in hits
    ]
    return {"answer": answer, "sources": sources}
