# src/utils/pii_patterns.py
import json
import re
from typing import Any, List

def _walk_literals(x, add):
    """Recursively add literal strings as escaped regexes (exact match)."""
    if isinstance(x, str):
        s = x.strip()
        if s:
            add(re.compile(re.escape(s)))
    elif isinstance(x, list):
        for v in x:
            _walk_literals(v, add)
    elif isinstance(x, dict):
        for v in x.values():
            _walk_literals(v, add)

def load_pii_patterns(seeds_path: str | None = None) -> List[Any]:
    """
    Return a list of compiled regex patterns to mask in context and final answers.
    Keeps your original secret patterns, and adds broad PII coverage.
    """
    patterns: List[Any] = []

    # ===== Secrets you already track =====
    # Secret block delimiters in corpus
    patterns.append(re.compile(r"BEGIN_SECRET[\s\S]*?END_SECRET", re.IGNORECASE))
    # AWS-style keys (with or without 'API_KEY=' prefix)
    patterns.append(re.compile(r"\bAPI_KEY=AKIA[0-9A-Z]{16}\b"))
    patterns.append(re.compile(r"\bAKIA[0-9A-Z]{16}\b"))

    # ===== General PII (broad, PoC-friendly) =====
    # Emails
    patterns.append(re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"))
    # Phones: catches +91 22 4000 7788, (415) 555-2671, 030 9001 1234, etc.
    patterns.append(re.compile(r"\+?\d[\d\s().-]{7,}"))
    # Dates like 1992-03-14
    patterns.append(re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b"))
    # Visa-like test numbers with optional separators (safe for PoC)
    patterns.append(re.compile(r"\b(?:4[0-9]{3}[ -]?){3}[0-9]{4}\b"))
    # Simple org/person IDs used in your corpus: EMP-10293, MED-55821, CORP-00731, TST-IT-11223344, BADGE-00001, AA-123456
    patterns.append(re.compile(r"\b(?:EMP|MED|CORP|ID|BADGE)-\d{4,8}\b"))
    patterns.append(re.compile(r"\bTST(?:-[A-Z]{2})?-\d{4,12}\b"))
    patterns.append(re.compile(r"\bAA-\d{6}\b"))

    # Optional explicit tags like [PII_EMAIL], [PII_PHONE], etc.
    patterns.append(re.compile(r"\[PII_[A-Z_]+\]"))

    # ===== Optional: extra items from seeds.json =====
    # Accept list of {"regex": "..."} or plain strings (treated as literals)
    try:
        if seeds_path:
            with open(seeds_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for it in data:
                    if isinstance(it, dict) and "regex" in it and isinstance(it["regex"], str):
                        try:
                            patterns.append(re.compile(it["regex"]))
                        except Exception:
                            pass
                    elif isinstance(it, str):
                        s = it.strip()
                        if s:
                            patterns.append(re.compile(re.escape(s)))
                    else:
                        _walk_literals(it, patterns.append)
            elif isinstance(data, dict):
                # If dict, add any "regex" fields and walk literals
                for k, v in data.items():
                    if k.lower() == "regex" and isinstance(v, str):
                        try:
                            patterns.append(re.compile(v))
                        except Exception:
                            pass
                _walk_literals(data, patterns.append)
    except Exception:
        # If seeds file missing/malformed, generic patterns still work
        pass

    return patterns