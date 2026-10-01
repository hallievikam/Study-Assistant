"""
db.py - Shared knowledge store (SQLite)

Tables:
    documents        - one row per uploaded file
    chunks           - text chunks with source/page metadata (vectors live in Chroma;
                        this table is the "source of truth" you can inspect/demo)
    quiz_cards       - generated MCQ / cloze flashcards
    review_history   - SM-2 spaced repetition state, one row per card
"""

import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "data" / "knowledge_store.db"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filetype TEXT NOT NULL,
            uploaded_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,          -- also used as the Chroma vector id
            document_id INTEGER NOT NULL,
            page INTEGER,                 -- NULL for non-paginated sources (.txt/.md)
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        );

        CREATE TABLE IF NOT EXISTS quiz_cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            topic TEXT,
            card_type TEXT NOT NULL,      -- 'mcq' or 'cloze'
            question TEXT NOT NULL,
            options TEXT,                 -- JSON list, only for mcq
            answer TEXT NOT NULL,
            source_chunk_id TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id),
            FOREIGN KEY (source_chunk_id) REFERENCES chunks(id)
        );

        CREATE TABLE IF NOT EXISTS review_history (
            card_id INTEGER PRIMARY KEY,
            repetitions INTEGER NOT NULL DEFAULT 0,
            ease_factor REAL NOT NULL DEFAULT 2.5,
            interval_days INTEGER NOT NULL DEFAULT 0,
            due_date TEXT NOT NULL,
            last_reviewed TEXT,
            FOREIGN KEY (card_id) REFERENCES quiz_cards(id)
        );
        """
    )
    conn.commit()
    conn.close()


# ---------------- documents / chunks ----------------

def add_document(filename: str, filetype: str) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO documents (filename, filetype, uploaded_at) VALUES (?, ?, ?)",
        (filename, filetype, datetime.utcnow().isoformat()),
    )
    conn.commit()
    doc_id = cur.lastrowid
    conn.close()
    return doc_id


def add_chunk(chunk_id: str, document_id: int, page: Optional[int], chunk_index: int, text: str):
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO chunks (id, document_id, page, chunk_index, text) VALUES (?, ?, ?, ?, ?)",
        (chunk_id, document_id, page, chunk_index, text),
    )
    conn.commit()
    conn.close()


def get_chunk(chunk_id: str) -> Optional[sqlite3.Row]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    conn.close()
    return row


def list_documents():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
    conn.close()
    return rows


# ---------------- quiz cards ----------------

def add_quiz_card(document_id, topic, card_type, question, options, answer, source_chunk_id) -> int:
    import json

    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO quiz_cards
           (document_id, topic, card_type, question, options, answer, source_chunk_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            document_id,
            topic,
            card_type,
            question,
            json.dumps(options) if options else None,
            answer,
            source_chunk_id,
            datetime.utcnow().isoformat(),
        ),
    )
    card_id = cur.lastrowid
    # seed review_history so it's due immediately
    conn.execute(
        "INSERT INTO review_history (card_id, due_date) VALUES (?, ?)",
        (card_id, date.today().isoformat()),
    )
    conn.commit()
    conn.close()
    return card_id


def get_due_cards(limit: int = 20):
    conn = get_conn()
    rows = conn.execute(
        """SELECT qc.*, rh.repetitions, rh.ease_factor, rh.interval_days, rh.due_date
           FROM quiz_cards qc
           JOIN review_history rh ON rh.card_id = qc.id
           WHERE rh.due_date <= ?
           ORDER BY rh.due_date ASC
           LIMIT ?""",
        (date.today().isoformat(), limit),
    ).fetchall()
    conn.close()
    return rows


def get_mastery_breakdown() -> dict:
    """
    Buckets every card by review progress, for a dashboard chart:
        New       - repetitions == 0 (never successfully reviewed)
        Learning  - repetitions 1-2
        Reviewing - repetitions 3-5
        Mastered  - repetitions > 5
    """
    conn = get_conn()
    rows = conn.execute(
        """SELECT rh.repetitions FROM quiz_cards qc
           JOIN review_history rh ON rh.card_id = qc.id"""
    ).fetchall()
    conn.close()

    buckets = {"New": 0, "Learning": 0, "Reviewing": 0, "Mastered": 0}
    for r in rows:
        rep = r["repetitions"]
        if rep == 0:
            buckets["New"] += 1
        elif rep <= 2:
            buckets["Learning"] += 1
        elif rep <= 5:
            buckets["Reviewing"] += 1
        else:
            buckets["Mastered"] += 1
    return buckets


def update_review(card_id: int, repetitions: int, ease_factor: float, interval_days: int, due_date: str):
    conn = get_conn()
    conn.execute(
        """UPDATE review_history
           SET repetitions = ?, ease_factor = ?, interval_days = ?, due_date = ?, last_reviewed = ?
           WHERE card_id = ?""",
        (repetitions, ease_factor, interval_days, due_date, datetime.utcnow().isoformat(), card_id),
    )
    conn.commit()
    conn.close()
