"""
llm.py - thin wrapper around the Anthropic API.

Requires an ANTHROPIC_API_KEY environment variable (set it in a .env
file or export it in your shell before running the app).
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

_client = None
MODEL = "claude-sonnet-4-6"


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to a .env file "
                "or export it in your shell."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def complete(system: str, user: str, max_tokens: int = 1000) -> str:
    client = get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
