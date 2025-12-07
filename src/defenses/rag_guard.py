# src/defenses/rag_guard.py
"""
Compatibility wrapper for the RAG guard.

We re-export `guard_context` from src.rag.filter so callers can import
`from src.defenses.rag_guard import guard_context` without duplicating logic.
"""
__all__ = ["guard_context"]
from src.rag.filter import guard_context  # noqa: F401
