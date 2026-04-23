"""
agent/memory.py — Lightweight session-based conversational memory

Stores chat history per session_id in-memory. Messages use the format
expected by Gemini's chat API: {"role": "user"|"model", "parts": [str]}

This enables multi-turn conversations where the AI remembers context.
"""

import logging
from collections import defaultdict

log = logging.getLogger(__name__)

# In-memory session store: session_id → list of messages
_sessions: dict[str, list[dict]] = defaultdict(list)

# Maximum messages per session to prevent unbounded memory growth
MAX_HISTORY_LENGTH = 20


def get_history(session_id: str) -> list[dict]:
    """Return the full message history for a session."""
    return _sessions.get(session_id, [])


def add_message(session_id: str, role: str, content: str):
    """
    Append a message to the session history.
    role: "user" or "model"
    content: the text content
    """
    if not session_id or not content:
        return

    _sessions[session_id].append({
        "role": role,
        "parts": [content],
    })

    # Trim oldest messages if history gets too long
    if len(_sessions[session_id]) > MAX_HISTORY_LENGTH:
        _sessions[session_id] = _sessions[session_id][-MAX_HISTORY_LENGTH:]

    log.debug(f"Memory: session {session_id[:8]}... now has {len(_sessions[session_id])} messages")


def get_context_summary(session_id: str) -> str:
    """
    Return a compact summary of the last 2-3 exchanges for use in query expansion.
    This helps the retriever understand conversational context (e.g., "what about section 34?"
    after discussing the Consumer Protection Act).
    """
    history = get_history(session_id)
    if not history:
        return ""

    # Take last 4 messages (2 exchanges) for context
    recent = history[-4:]
    parts = []
    for msg in recent:
        role_label = "User" if msg["role"] == "user" else "AI"
        text = msg["parts"][0][:200]  # truncate long messages
        parts.append(f"{role_label}: {text}")

    return "\n".join(parts)


def clear_session(session_id: str):
    """Wipe the history for a session."""
    if session_id in _sessions:
        del _sessions[session_id]
        log.info(f"Memory: cleared session {session_id[:8]}...")


def get_active_sessions() -> int:
    """Return count of active sessions (for monitoring)."""
    return len(_sessions)
