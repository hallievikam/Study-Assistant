# Study Assistant — RAG + Quiz Generator

A multi-agent study tool: upload PDFs/notes, ask grounded questions with
citations, generate topic summaries, and get spaced-repetition flashcards.

## Architecture

```
User → Ingestion Pipeline → Embedding + Vector Store (Chroma)
                                     |
              +----------------------+----------------------+
              |                      |                       |
         Q&A Agent           Summarizer Agent        Quiz/Flashcard Agent
       (retrieval + cite)   (per-topic notes)        (MCQ/cloze + SM-2)
              |                      |                       |
              +----------- Shared Knowledge Store (SQLite) ---+
                                     |
                              Streamlit UI
```

| File                    | Component                                      |
|-------------------------|-------------------------------------------------|
| `ingestion.py`          | 1. Ingestion pipeline (PDF/txt/md → chunks)      |
| `vectorstore.py`        | 2. Embedding + vector store (sentence-transformers + Chroma) |
| `qa_agent.py`           | 3. Q&A Agent (RAG core, cited answers)           |
| `summarizer_agent.py`   | 4. Summarizer Agent                              |
| `quiz_agent.py` + `sm2.py` | 5. Quiz/Flashcard Agent + SM-2 scheduler      |
| `db.py`                 | 6. Shared knowledge store (SQLite)               |
| `llm.py`                | Anthropic API wrapper used by all sub-agents     |
| `agent.py`              | **AI Agent orchestrator** — exposes the three sub-agents as tools and lets Claude decide which to call, via native tool-use, based on plain-language requests |
| `app.py`                | Streamlit UI (AI Agent chat, dashboard, flashcard review, etc.) |

### The AI Agent (`agent.py`)

Rather than the user picking a tab for "ask a question" vs "summarize" vs
"quiz me", the AI Agent page is a single chat where Claude itself decides
which tool to call, using the same function-calling mechanism real
production agents use:

1. The user's message + a list of available tools (`list_documents`,
   `ask_question`, `summarize_document`, `generate_flashcards`) are sent to
   Claude.
2. If Claude responds with a `tool_use` block, `agent.py` executes that
   tool locally (calling straight into `qa_agent.py` / `summarizer_agent.py`
   / `quiz_agent.py`) and sends the result back.
3. This repeats — the agent can chain multiple tool calls in one turn
   (e.g. list documents, then summarize the one it finds) — until Claude
   returns a plain-text final answer.

This is worth highlighting in a viva if asked "is this actually agentic or
just a wrapper?" — the tool selection and argument-filling are genuinely
done by the model, not by keyword-matching in your own code.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in this folder with your API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
streamlit run app.py
```

This opens the UI at `http://localhost:8501` with four tabs:

1. **Upload** — ingest a PDF, `.txt`/`.md` file, or pasted notes.
2. **Q&A Chat** — ask a question, get an answer grounded in your material
   with `[filename, p.N]` citations, scoped to one document or all of them.
3. **Summaries** — generate a full-document or per-topic cheat sheet.
4. **Flashcards** — generate MCQ/cloze cards from a document or topic, then
   review due cards with a 4-button (Again/Hard/Good/Easy) SM-2 review flow.

## Data you can inspect for your viva

Everything lives in `data/knowledge_store.db` (SQLite) — you can open it with
any SQLite browser (or `sqlite3 data/knowledge_store.db`) and show the
`documents`, `chunks`, `quiz_cards`, and `review_history` tables live to
demonstrate the RAG + spaced-repetition mechanics. Vector embeddings
themselves live in `chroma_db/` (Chroma's on-disk store).

## Notes / things to flag in your report

- **Chunking**: ~400 words per chunk (~500 tokens) with 60-word overlap, done
  per-page for PDFs so citations can include page numbers.
- **Embeddings**: `all-MiniLM-L6-v2` via sentence-transformers — runs fully
  locally, no API cost, ~80MB model.
- **Groundedness**: both the Q&A and Summarizer agents are prompted to
  answer *only* from retrieved chunks, not the model's general knowledge —
  worth calling out explicitly as your RAG design choice.
- **SM-2**: implemented in `sm2.py`, using a simplified 4-button quality
  scale (Again=0, Hard=3, Good=4, Easy=5) mapped onto SuperMemo-2's 0–5 scale.
- **Extending it**: the retrieval call in `vectorstore.query()` is the one
  spot every agent shares — a nice thing to point at in a viva when asked
  "how do the three agents relate to each other?"
