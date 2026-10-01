"""
agent.py - AI Agent orchestrator

This turns the three separate sub-agents (Q&A, Summarizer, Quiz/Flashcard)
into tools that a single top-level agent can call on its own. Instead of
the user picking a tab, they type a plain-language request and the agent
decides — via Claude's native tool-use — which underlying capability to
invoke, with which arguments, and can chain multiple calls before giving
a final answer (e.g. "list my documents, then quiz me on the second one").

This is a real agent loop: the model chooses tool calls, we execute them
against the local sub-agents, feed the results back, and repeat until the
model returns a plain text answer.
"""

import json

import db
import qa_agent
import summarizer_agent
import quiz_agent
import llm

SYSTEM_PROMPT = """You are an AI study agent with access to tools that act on the
user's uploaded documents. Decide which tool(s) to call based on what the user
asks, in plain conversation. Rules:
- If the user asks a question about content, use ask_question.
- If they want a summary / cheat sheet / notes, use summarize_document.
- If they want flashcards / a quiz / to be tested, use generate_flashcards.
- If you need to figure out which document they mean by name, call list_documents first.
- You may call more than one tool in sequence to satisfy a request.
- After tool results come back, give a clear, conversational final answer.
  If ask_question returned sources, mention them briefly.
- Never fabricate document IDs — only use IDs seen from list_documents or the
  conversation's context.
"""

TOOLS = [
    {
        "name": "list_documents",
        "description": "List all documents currently in the knowledge base, with their IDs, filenames and types. Call this first if you need to resolve a document name to an ID.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "ask_question",
        "description": "Answer a question grounded only in the user's uploaded materials, with citations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The user's question."},
                "document_id": {"type": "integer", "description": "Optional: restrict retrieval to this document ID. Omit to search all documents."},
            },
            "required": ["question"],
        },
    },
    {
        "name": "summarize_document",
        "description": "Generate a study summary / cheat sheet for a document, optionally focused on one topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "integer", "description": "The document to summarize."},
                "topic": {"type": "string", "description": "Optional topic to focus the summary on."},
            },
            "required": ["document_id"],
        },
    },
    {
        "name": "generate_flashcards",
        "description": "Generate MCQ/cloze flashcards from a document, optionally focused on one topic, and schedule them for spaced-repetition review.",
        "input_schema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "integer", "description": "The document to generate flashcards from."},
                "topic": {"type": "string", "description": "Optional topic to focus the flashcards on."},
                "num_cards": {"type": "integer", "description": "How many cards to generate (default 8)."},
            },
            "required": ["document_id"],
        },
    },
]


def _run_tool(name: str, tool_input: dict) -> dict:
    """Execute a tool call locally and return a JSON-serializable result."""
    if name == "list_documents":
        docs = db.list_documents()
        return {"documents": [
            {"id": d["id"], "filename": d["filename"], "filetype": d["filetype"]}
            for d in docs
        ]}

    if name == "ask_question":
        result = qa_agent.answer_question(
            tool_input["question"],
            document_id=tool_input.get("document_id"),
        )
        return result

    if name == "summarize_document":
        summary = summarizer_agent.summarize_document(
            tool_input["document_id"],
            topic=tool_input.get("topic"),
        )
        return {"summary": summary}

    if name == "generate_flashcards":
        created = quiz_agent.generate_cards(
            tool_input["document_id"],
            topic=tool_input.get("topic"),
            num_cards=tool_input.get("num_cards", 8),
        )
        return {"cards_created": len(created), "card_ids": created}

    return {"error": f"Unknown tool: {name}"}


def run_agent(user_message: str, conversation_history: list = None, max_hops: int = 5) -> dict:
    """
    Runs the agent loop for a single user turn. `conversation_history` is a
    list of prior {"role": ..., "content": ...} messages (plain text turns,
    not raw tool-call blocks) used for short-term context.

    Returns {"answer": str, "tool_calls": [ {name, input, result_summary}, ... ]}
    so the UI can optionally show what the agent did.
    """
    client = llm.get_client()

    messages = []
    for turn in (conversation_history or []):
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    tool_call_log = []

    for _ in range(max_hops):
        response = client.messages.create(
            model=llm.MODEL,
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text")
            return {"answer": final_text, "tool_calls": tool_call_log}

        # Assistant turn included at least one tool_use block - execute all of them
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = _run_tool(block.name, block.input)
            tool_call_log.append({"name": block.name, "input": block.input, "result": result})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result)[:6000],  # keep payloads bounded
            })

        messages.append({"role": "user", "content": tool_results})

    return {
        "answer": "I wasn't able to finish that in a reasonable number of steps — "
                  "could you narrow down the request?",
        "tool_calls": tool_call_log,
    }
