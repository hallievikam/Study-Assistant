"""
quiz_agent.py - Quiz/Flashcard Generator Agent

Generates MCQs and cloze-deletion flashcards from retrieved content, stores
them in SQLite, and schedules reviews using the SM-2 algorithm so cards
resurface based on how the user answered them.
"""

import json
import re

import db
import vectorstore
import llm
import sm2

SYSTEM_PROMPT = """You are a quiz generator for a spaced-repetition study app.
Given source material, generate flashcards STRICTLY as a JSON array (no
markdown fences, no preamble, no commentary). Each element must be an object:

For multiple choice:
{"type": "mcq", "question": "...", "options": ["A", "B", "C", "D"], "answer": "the correct option text"}

For cloze deletion:
{"type": "cloze", "question": "The mitochondria is the ... of the cell.", "answer": "powerhouse"}

Rules:
- Base every card only on the given source material.
- Mix mcq and cloze types.
- Keep questions unambiguous and answers concise.
- Return ONLY the JSON array, nothing else.
"""


def _extract_json_array(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    return json.loads(raw)


def generate_cards(document_id: int, topic: str = None, num_cards: int = 8) -> list:
    """Generate `num_cards` flashcards from a document (optionally scoped to a topic)."""
    if topic:
        hits = vectorstore.query(topic, top_k=8, document_id=document_id)
    else:
        hits = vectorstore.query("key concepts, definitions, and important facts",
                                  top_k=8, document_id=document_id)

    if not hits:
        return []

    context = "\n\n".join(h["text"] for h in hits)
    source_chunk_id = hits[0]["chunk_id"]  # best-effort attribution

    user_prompt = f"Source material:\n\n{context}\n\nGenerate {num_cards} flashcards as a JSON array."
    raw = llm.complete(SYSTEM_PROMPT, user_prompt, max_tokens=1500)

    try:
        cards_raw = _extract_json_array(raw)
    except (json.JSONDecodeError, ValueError):
        return []

    created = []
    for c in cards_raw:
        card_type = c.get("type")
        question = c.get("question")
        answer = c.get("answer")
        options = c.get("options") if card_type == "mcq" else None
        if not question or not answer:
            continue
        card_id = db.add_quiz_card(
            document_id=document_id,
            topic=topic,
            card_type=card_type,
            question=question,
            options=options,
            answer=answer,
            source_chunk_id=source_chunk_id,
        )
        created.append(card_id)

    return created


def review_card(card_id: int, quality: int):
    """
    Record a review result (quality 0-5, see sm2.py) and reschedule the
    card using SM-2.
    """
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM review_history WHERE card_id = ?", (card_id,)).fetchone()
    conn.close()
    if row is None:
        raise ValueError(f"No review_history row for card {card_id}")

    repetitions, ease_factor, interval_days, due_date = sm2.sm2(
        quality, row["repetitions"], row["ease_factor"], row["interval_days"]
    )
    db.update_review(card_id, repetitions, ease_factor, interval_days, due_date)
    return {"repetitions": repetitions, "ease_factor": ease_factor,
            "interval_days": interval_days, "due_date": due_date}


def get_due_cards(limit: int = 20):
    return db.get_due_cards(limit=limit)
